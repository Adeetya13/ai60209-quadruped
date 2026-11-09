"""Train an SB3 agent on the Ant-v5 quadruped.

One run = one algorithm + one seed. Everything needed to reproduce a run is
written to runs/<tag>/config.json, so no result in the report is ever orphaned.

    python train.py --algo ppo --seed 0
    python train.py --algo sac --seed 1 --steps 500000
"""
import argparse
import json
import subprocess
import time
from pathlib import Path

import gymnasium as gym
from stable_baselines3 import A2C, PPO, SAC
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize

ENV_ID = "Ant-v5"
ALGOS = {"ppo": PPO, "sac": SAC, "a2c": A2C}

# On-policy methods want many parallel envs; SAC is off-policy and does not.
DEFAULT_N_ENVS = {"ppo": 8, "a2c": 8, "sac": 1}
DEFAULT_STEPS = {"ppo": 1_000_000, "a2c": 1_000_000, "sac": 500_000}


def git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def build_env(n_envs: int, seed: int, run_dir: Path):
    """Vectorised Ant-v5 with observation normalisation.

    Observations are normalised for every algorithm and rewards for none, so the
    three algorithms see identical preprocessing. That keeps the comparison fair,
    which matters more here than squeezing the last few points out of PPO.
    """
    venv = make_vec_env(
        ENV_ID, n_envs=n_envs, seed=seed, monitor_dir=str(run_dir / "monitor")
    )
    return VecNormalize(venv, norm_obs=True, norm_reward=False, clip_obs=10.0)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--algo", choices=sorted(ALGOS), default="ppo")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=None, help="default depends on algo")
    p.add_argument("--n-envs", type=int, default=None)
    p.add_argument("--out", default="runs")
    args = p.parse_args()

    steps = args.steps if args.steps is not None else DEFAULT_STEPS[args.algo]
    n_envs = args.n_envs if args.n_envs is not None else DEFAULT_N_ENVS[args.algo]

    tag = f"{args.algo}_s{args.seed}"
    run_dir = Path(args.out) / tag
    run_dir.mkdir(parents=True, exist_ok=True)

    env = build_env(n_envs, args.seed, run_dir)
    model = ALGOS[args.algo](
        "MlpPolicy", env, seed=args.seed, verbose=1,
        tensorboard_log=str(Path(args.out) / "tb"),
    )

    config = {
        "env": ENV_ID,
        "algo": args.algo,
        "seed": args.seed,
        "total_timesteps": steps,
        "n_envs": n_envs,
        "normalize_obs": True,
        "normalize_reward": False,
        "hyperparameters": "stable-baselines3 defaults",
        "sb3_policy": "MlpPolicy",
        "git_commit": git_commit(),
        "started": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2))
    print(f"[train] {tag}: {steps:,} steps, {n_envs} env(s)")

    ckpt = CheckpointCallback(
        save_freq=max(50_000 // n_envs, 1),
        save_path=str(run_dir / "checkpoints"),
        name_prefix=tag,
    )

    t0 = time.time()
    model.learn(total_timesteps=steps, tb_log_name=tag, callback=ckpt,
                progress_bar=False)
    elapsed = time.time() - t0

    model.save(run_dir / "final_model")
    env.save(str(run_dir / "vecnormalize.pkl"))

    config["wall_clock_seconds"] = round(elapsed, 1)
    config["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    (run_dir / "config.json").write_text(json.dumps(config, indent=2))
    print(f"[train] {tag} done in {elapsed/60:.1f} min -> {run_dir}")


if __name__ == "__main__":
    main()
