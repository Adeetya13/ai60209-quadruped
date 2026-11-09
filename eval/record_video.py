"""Record an mp4 of a policy. Run this on every checkpoint, including the
early hopeless ones: the flailing-to-running montage is the best slide you
will have, and it cannot be reconstructed after the fact.

    python eval/record_video.py --run runs/ppo_s0
    python eval/record_video.py --policy scripted
    python eval/record_video.py --run runs/ppo_s0 --all-checkpoints
"""
import argparse
import sys
from pathlib import Path

import gymnasium as gym
import imageio.v2 as imageio
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluate import ENV_ID, resolve  # noqa: E402


def record(act_fn, out_path: Path, seconds: float = 10.0, seed: int = 0):
    env = gym.make(ENV_ID, render_mode="rgb_array")
    obs, _ = env.reset(seed=seed)
    fps = int(round(1.0 / env.unwrapped.dt))
    frames, step = [], 0
    while step < int(seconds * fps):
        a = act_fn(obs, step, env)
        obs, _, terminated, truncated, _ = env.step(a)
        frames.append(env.render())
        step += 1
        if terminated or truncated:
            obs, _ = env.reset(seed=seed + step)
    env.close()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(out_path, frames, fps=fps)
    print(f"[video] {len(frames)} frames at {fps} fps -> {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run")
    ap.add_argument("--policy", choices=["random", "scripted"])
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--outdir", default="figures/video")
    ap.add_argument("--all-checkpoints", action="store_true",
                    help="one clip per saved checkpoint, for the montage")
    args = ap.parse_args()

    if args.all_checkpoints:
        from stable_baselines3 import A2C, PPO, SAC
        import json
        run_dir = Path(args.run)
        cfg = json.loads((run_dir / "config.json").read_text())
        algo = {"ppo": PPO, "sac": SAC, "a2c": A2C}[cfg["algo"]]
        for ckpt in sorted((run_dir / "checkpoints").glob("*.zip")):
            model = algo.load(ckpt, device="cpu")
            steps = ckpt.stem.split("_")[-2] if "_" in ckpt.stem else ckpt.stem
            record(
                lambda o, s, e: model.predict(o, deterministic=True)[0],
                Path(args.outdir) / f"{run_dir.name}_{steps}.mp4",
                seconds=min(args.seconds, 5.0),
            )
        return

    name, act_fn = resolve(args.run, args.policy)
    record(act_fn, Path(args.outdir) / f"{name}.mp4", seconds=args.seconds)


if __name__ == "__main__":
    main()
