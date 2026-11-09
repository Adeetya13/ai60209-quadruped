"""Learning curves, mean with shaded spread across seeds.

The shaded band is the point of this figure. Lecture 07 warns that deep RL
results vary across random seeds; plotting one curve per algorithm hides that,
plotting the spread shows it.

    python eval/plot_curves.py
    python eval/plot_curves.py --out figures/curves.png --bins 60
"""
import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

COLORS = {"ppo": "#1D5A85", "sac": "#0F6A4F", "a2c": "#8C5A0B",
          "reinforce": "#9E3327"}


def load_sb3(run_dir: Path):
    """Episode returns against cumulative environment steps, across all workers."""
    frames = []
    for csv_path in sorted((run_dir / "monitor").glob("*.csv")):
        try:
            df = pd.read_csv(csv_path, skiprows=1)
        except Exception:
            continue
        if {"r", "l"} <= set(df.columns) and len(df):
            df = df.sort_values("t")
            df["steps"] = df["l"].cumsum()
            frames.append(df[["steps", "r"]])
    if not frames:
        return None
    n = len(frames)
    out = pd.concat(frames).sort_values("steps")
    out["steps"] = out["steps"] * n  # each worker only counts its own steps
    return out


def load_reinforce(run_dir: Path):
    path = run_dir / "progress.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    return pd.DataFrame({"steps": df["timesteps"], "r": df["mean_ep_return"]})


def bin_curve(df, bins: int, max_steps: int):
    edges = np.linspace(0, max_steps, bins + 1)
    idx = np.digitize(df["steps"], edges) - 1
    out = np.full(bins, np.nan)
    for b in range(bins):
        sel = df["r"][idx == b]
        if len(sel):
            out[b] = sel.mean()
    return pd.Series(out).ffill().to_numpy(), (edges[:-1] + edges[1:]) / 2


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--out", default="figures/learning_curves.png")
    ap.add_argument("--bins", type=int, default=50)
    args = ap.parse_args()

    by_algo = defaultdict(list)
    for run_dir in sorted(Path(args.runs).glob("*")):
        if not run_dir.is_dir() or not (run_dir / "config.json").exists():
            continue
        algo = json.loads((run_dir / "config.json").read_text()).get("algo")
        df = load_reinforce(run_dir) if algo == "reinforce" else load_sb3(run_dir)
        if df is not None and len(df) > 2:
            by_algo[algo].append((run_dir.name, df))

    if not by_algo:
        print("[plot] no runs with logged episodes yet")
        return

    fig, ax = plt.subplots(figsize=(8.0, 4.8), dpi=150)
    for algo, runs in sorted(by_algo.items()):
        max_steps = min(float(df["steps"].max()) for _, df in runs)
        curves = []
        for _, df in runs:
            y, x = bin_curve(df, args.bins, max_steps)
            curves.append(y)
        curves = np.vstack(curves)
        mean, std = np.nanmean(curves, axis=0), np.nanstd(curves, axis=0)
        color = COLORS.get(algo, None)
        ax.plot(x, mean, label=f"{algo.upper()}  (n={len(runs)} seeds)",
                color=color, linewidth=2)
        ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.18,
                        linewidth=0)

    ax.set_xlabel("environment steps")
    ax.set_ylabel("episode return")
    ax.set_title("Ant-v5: mean episode return, shaded band is $\\pm$1 s.d. across seeds")
    ax.grid(alpha=0.25, linewidth=0.6)
    ax.legend(frameon=False)
    fig.tight_layout()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    print(f"[plot] {sum(len(v) for v in by_algo.values())} runs -> {out}")


if __name__ == "__main__":
    main()
