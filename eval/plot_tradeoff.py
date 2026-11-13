"""Two figures the tables cannot carry.

1. tradeoff.png   speed against control effort, one point per policy.
   This is the discussion's whole thesis in one picture: each rung of the
   ladder buys speed and pays for it in energy, and the hand-written gait
   sits alone in the cheap corner.

2. robustness.png degradation under each perturbation, as a grouped bar
   chart. The robustness table is 5 rows by 7 columns of numbers, which
   nobody reads off a projector.

    python eval/plot_tradeoff.py
"""
import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

COLORS = {"sac": "#0F6A4F", "a2c": "#8C5A0B", "ppo": "#1D5A85",
          "reinforce": "#9E3327", "scripted_gait": "#5A5A5A",
          "random": "#AAAAAA"}
PRETTY = {"sac": "SAC", "a2c": "A2C", "ppo": "PPO",
          "reinforce": "REINFORCE (ours)", "scripted_gait": "Scripted trot",
          "random": "Random"}
ORDER = ["sac", "a2c", "ppo", "reinforce", "scripted_gait", "random"]

# Hand-placed, because the two cheap policies sit almost on top of each other
# near effort 0.01 and their labels collided.
OFFSETS = {
    "sac":           (-0.013,  0.00, "right", "center"),
    "a2c":           ( 0.013,  0.10, "left",  "bottom"),
    "ppo":           ( 0.013, -0.10, "left",  "top"),
    "scripted_gait": ( 0.015,  0.17, "left",  "bottom"),
    "reinforce":     ( 0.015, -0.15, "left",  "top"),
    "random":        (-0.013,  0.00, "right", "center"),
}


def group(name: str):
    """'ppo_s0' -> 'ppo'. Ablation variants and other environments return None:
    'reinforce_s0_full_none' and 'reinforce-halfcheetah_s1' must not be
    averaged into the main REINFORCE row, which is 2 seeds, not 4."""
    name = str(name)
    if name in ("scripted_gait", "random"):
        return name
    m = re.fullmatch(r"([a-z0-9]+)_s\d+", name)
    return m.group(1) if m else None


def load(metrics_csv: str) -> pd.DataFrame:
    m = pd.read_csv(metrics_csv)
    m["algo"] = [group(n) for n in m["name"]]
    m = m[m["algo"].isin(ORDER)]
    g = m.groupby("algo").agg(
        vel=("x_velocity_mean", "mean"),
        vel_sd=("x_velocity_mean", "std"),
        eff=("mean_sq_action_mean", "mean"),
        eff_sd=("mean_sq_action_mean", "std"),
        ret=("return_mean", "mean"),
        n=("name", "count"),
    )
    return g.fillna(0.0)


def tradeoff(g: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 5.0), dpi=150)
    for algo in ORDER:
        if algo not in g.index:
            continue
        r = g.loc[algo]
        c = COLORS[algo]
        ax.errorbar(r["eff"], r["vel"], xerr=r["eff_sd"], yerr=r["vel_sd"],
                    fmt="o", ms=11, color=c, ecolor=c, elinewidth=1.2,
                    capsize=3, alpha=0.9, zorder=3)
        dx, dy, ha, va = OFFSETS.get(algo, (0.012, 0.0, "left", "center"))
        ax.annotate(f"{PRETTY[algo]} (n={int(r['n'])})",
                    (r["eff"] + dx, r["vel"] + dy), ha=ha, va=va,
                    fontsize=10.5, color=c, zorder=4)

    ax.set_xlabel("control effort  (mean squared action, lower is cheaper)")
    ax.set_ylabel("forward velocity  (m/s)")
    ax.set_title("Speed is bought with energy: the scripted gait sits alone "
                 "in the cheap corner", fontsize=12.5)
    ax.grid(alpha=0.25, linewidth=0.6)
    ax.set_xlim(-0.03, 0.40)
    ax.set_ylim(-0.25, 2.9)
    ax.axhline(0, color="#999", linewidth=0.8, zorder=1)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    print(f"[tradeoff] -> {out}")


def robustness(csv: str, out: Path) -> None:
    rob = pd.read_csv(csv)
    rob["algo"] = [group(n) for n in rob["name"]]
    sweeps = [("mass", 2.0, "mass x2"), ("gravity", 1.5, "gravity x1.5"),
              ("noise", 0.2, "action noise 0.2")]

    algos = [a for a in ORDER if a in set(rob["algo"]) and a != "random"]
    fig, ax = plt.subplots(figsize=(8.4, 4.8), dpi=150)
    width = 0.8 / (len(sweeps) + 1)
    x = np.arange(len(algos))

    base = []
    for a in algos:
        s = rob[(rob["algo"] == a) & (rob["perturbation"] == "mass")
                & (rob["level"] == 1.0)]
        base.append(s["return_mean"].mean() if len(s) else np.nan)
    ax.bar(x, base, width, label="unperturbed", color="#C9D3CC",
           edgecolor="#5A5A5A", linewidth=0.7)

    for i, (kind, lvl, label) in enumerate(sweeps, start=1):
        vals = []
        for a in algos:
            s = rob[(rob["algo"] == a) & (rob["perturbation"] == kind)
                    & (rob["level"] == lvl)]
            vals.append(s["return_mean"].mean() if len(s) else np.nan)
        ax.bar(x + i * width, vals, width, label=label)

    ax.set_xticks(x + width * len(sweeps) / 2)
    ax.set_xticklabels([PRETTY[a] for a in algos], fontsize=10)
    ax.set_ylabel("mean return")
    ax.set_title("Only action noise hurts: extra mass and gravity help the "
                 "learned policies stay upright", fontsize=12.5)
    ax.grid(alpha=0.25, axis="y", linewidth=0.6)
    ax.legend(frameon=False, fontsize=10, ncol=2)
    fig.tight_layout()
    fig.savefig(out)
    print(f"[robustness] -> {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", default="docs/metrics.csv")
    ap.add_argument("--robustness", default="docs/robustness.csv")
    ap.add_argument("--outdir", default="figures")
    args = ap.parse_args()

    g = load(args.metrics)
    tradeoff(g, Path(args.outdir) / "tradeoff.png")
    if Path(args.robustness).exists():
        robustness(args.robustness, Path(args.outdir) / "robustness.png")


if __name__ == "__main__":
    main()
