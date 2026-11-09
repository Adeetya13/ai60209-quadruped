"""Robustness tests. No retraining: perturb the world at evaluation time and
measure how much return survives.

Three perturbations, each a one-line change to the MuJoCo model or the action:

    mass      scale every body mass          (carrying a payload)
    gravity   scale gravity                  (a different planet, or a slope)
    noise     add Gaussian noise to actions  (imperfect actuators)

    python eval/robustness.py --run runs/ppo_s0
    python eval/robustness.py --all
"""
import argparse
import csv
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluate import ENV_ID, resolve, rollout  # noqa: E402

SWEEPS = {
    "mass": [1.0, 1.5, 2.0],
    "gravity": [1.0, 1.25, 1.5],
    "noise": [0.0, 0.1, 0.2],
}


def make_env(kind: str, level: float):
    env = gym.make(ENV_ID)
    env.reset(seed=0)
    model = env.unwrapped.model
    if kind == "mass":
        model.body_mass[:] = model.body_mass * level
    elif kind == "gravity":
        model.opt.gravity[2] = model.opt.gravity[2] * level
    return env


def wrap_noise(act_fn, sigma: float, rng):
    if sigma <= 0:
        return act_fn

    def noisy(obs, step, env):
        a = np.asarray(act_fn(obs, step, env), dtype=np.float64)
        a = a + rng.normal(0.0, sigma, size=a.shape)
        return np.clip(a, env.action_space.low, env.action_space.high)

    return noisy


def run_sweep(name, act_fn, episodes: int):
    rows = []
    for kind, levels in SWEEPS.items():
        for level in levels:
            rng = np.random.default_rng(0)
            env = make_env(kind, level)
            fn = wrap_noise(act_fn, level, rng) if kind == "noise" else act_fn
            vals = [rollout(env, fn, 2000 + i)["return"] for i in range(episodes)]
            env.close()
            rows.append({
                "name": name, "perturbation": kind, "level": level,
                "return_mean": round(float(np.mean(vals)), 2),
                "return_std": round(float(np.std(vals)), 2),
            })
            print(f"[robust] {name:<24} {kind:<8} x{level:<5} "
                  f"return {rows[-1]['return_mean']:9.1f}")
    # Express each sweep as a fraction of its own unperturbed level.
    for kind in SWEEPS:
        base = next(r["return_mean"] for r in rows
                    if r["perturbation"] == kind and r["level"] == SWEEPS[kind][0])
        for r in rows:
            if r["perturbation"] == kind:
                r["retained_pct"] = (round(100.0 * r["return_mean"] / base, 1)
                                     if abs(base) > 1e-9 else float("nan"))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run")
    ap.add_argument("--policy", choices=["random", "scripted"])
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--out", default="docs/robustness.csv")
    args = ap.parse_args()

    targets = []
    if args.all:
        for d in sorted(Path("runs").glob("*")):
            if (d / "final_model.zip").exists() or (d / "final_model.pt").exists():
                targets.append((str(d), None))
        targets.append((None, "scripted"))
    else:
        targets = [(args.run, args.policy)]

    rows = []
    for run, policy in targets:
        try:
            name, act_fn = resolve(run, policy)
            rows += run_sweep(name, act_fn, args.episodes)
        except Exception as exc:
            print(f"[robust] SKIPPED {run or policy}: {exc}")

    if not rows:
        print("[robust] nothing to evaluate yet")
        return

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\n[robust] -> {out}")


if __name__ == "__main__":
    main()
