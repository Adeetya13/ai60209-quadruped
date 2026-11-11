"""Fill the metrics table. Works on SB3 runs, the from-scratch REINFORCE,
the scripted gait and a random policy, so every row is measured identically.

    python eval/evaluate.py --run runs/ppo_s0 --episodes 20
    python eval/evaluate.py --all --episodes 20        # every run, one table
    python eval/evaluate.py --policy random --episodes 20
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ENV_ID = "Ant-v5"
METRICS = ["return", "x_velocity", "steps", "mean_sq_action"]


def rollout(env, act_fn, seed: int):
    obs, _ = env.reset(seed=seed)
    total, steps, vx, sq = 0.0, 0, [], []
    while True:
        a = act_fn(obs, steps, env)
        obs, reward, terminated, truncated, info = env.step(a)
        total += reward
        steps += 1
        vx.append(info.get("x_velocity", 0.0))
        sq.append(float(np.mean(np.asarray(a) ** 2)))
        if terminated or truncated:
            break
    return {"return": total, "x_velocity": float(np.mean(vx)),
            "steps": steps, "mean_sq_action": float(np.mean(sq))}


# ---------------------------------------------------------------- policies
def sb3_policy(run_dir: Path):
    from stable_baselines3 import A2C, PPO, SAC
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    cfg = json.loads((run_dir / "config.json").read_text())
    algo = {"ppo": PPO, "sac": SAC, "a2c": A2C}[cfg["algo"]]
    model = algo.load(run_dir / "final_model", device="cpu")

    norm = None
    vec_path = run_dir / "vecnormalize.pkl"
    if vec_path.exists():
        env_id = cfg.get("env", ENV_ID)
        norm = VecNormalize.load(str(vec_path), DummyVecEnv([lambda: gym.make(env_id)]))
        norm.training = False

    def act(obs, _step, _env):
        o = norm.normalize_obs(obs) if norm is not None else obs
        action, _ = model.predict(o, deterministic=True)
        return action

    return act


def reinforce_policy(run_dir: Path):
    import torch
    from reinforce import GaussianPolicy

    ckpt = torch.load(run_dir / "final_model.pt", map_location="cpu",
                      weights_only=False)
    policy = GaussianPolicy(ckpt["obs_dim"], ckpt["act_dim"])
    policy.load_state_dict(ckpt["policy"])
    policy.eval()

    def act(obs, _step, env):
        with torch.no_grad():
            mu = policy.mu(torch.as_tensor(obs, dtype=torch.float32)).numpy()
        return np.clip(mu, env.action_space.low, env.action_space.high)

    return act


def scripted_policy():
    from baselines.scripted_gait import action as gait_action, load_params
    params = load_params()
    return lambda _obs, step, env: gait_action(step * env.unwrapped.dt, params)


def random_policy():
    return lambda _obs, _step, env: env.action_space.sample()


def run_env_id(run_dir: Path) -> str:
    """Evaluate a policy in the environment it was actually trained on."""
    cfg = run_dir / "config.json"
    if cfg.exists():
        try:
            return json.loads(cfg.read_text()).get("env", ENV_ID)
        except Exception:
            pass
    return ENV_ID


def resolve(run: str | None, policy: str | None):
    """Returns (name, action function, env id)."""
    if policy == "random":
        return "random", random_policy(), ENV_ID
    if policy == "scripted":
        return "scripted_gait", scripted_policy(), ENV_ID
    run_dir = Path(run)
    env_id = run_env_id(run_dir)
    if (run_dir / "final_model.pt").exists():
        return run_dir.name, reinforce_policy(run_dir), env_id
    return run_dir.name, sb3_policy(run_dir), env_id


# ------------------------------------------------------------------ driver
def measure(name, act_fn, episodes: int, env_id: str = ENV_ID):
    env = gym.make(env_id)
    rows = [rollout(env, act_fn, 1000 + i) for i in range(episodes)]
    env.close()
    out = {"name": name, "episodes": episodes}
    for m in METRICS:
        vals = [r[m] for r in rows]
        out[f"{m}_mean"] = round(float(np.mean(vals)), 3)
        out[f"{m}_std"] = round(float(np.std(vals)), 3)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run")
    ap.add_argument("--policy", choices=["random", "scripted"])
    ap.add_argument("--all", action="store_true", help="every run in runs/")
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--out", default="docs/metrics.csv")
    args = ap.parse_args()

    targets = []
    if args.all:
        for d in sorted(Path("runs").glob("*")):
            if (d / "final_model.zip").exists() or (d / "final_model.pt").exists():
                targets.append((str(d), None))
        targets += [(None, "scripted"), (None, "random")]
    else:
        targets = [(args.run, args.policy)]

    results = []
    for run, policy in targets:
        try:
            name, act_fn, env_id = resolve(run, policy)
            results.append(measure(name, act_fn, args.episodes, env_id))
            print(f"[eval] {name:<28} return "
                  f"{results[-1]['return_mean']:9.1f} +/- {results[-1]['return_std']:7.1f}")
        except Exception as exc:
            print(f"[eval] SKIPPED {run or policy}: {exc}")

    if not results:
        print("[eval] nothing to evaluate yet")
        return

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0]))
        w.writeheader()
        w.writerows(results)
    print(f"\n[eval] metrics table -> {out_path}")


if __name__ == "__main__":
    main()
