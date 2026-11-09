"""Open-loop sinusoidal trot gait for Ant-v5. The non-learned baseline.

No learning, no state feedback: each joint just follows a sine wave. Diagonal
legs move together, which is what a trot is. The amplitudes, frequency and
ankle phase offset are found by a coarse grid search, so the comparison against
the RL policies is fair rather than a straw man.

    python baselines/scripted_gait.py --search      # find good parameters
    python baselines/scripted_gait.py --episodes 20 # evaluate the best found
"""
import argparse
import itertools
import json
from pathlib import Path

import gymnasium as gym
import numpy as np

ENV_ID = "Ant-v5"

# Ant actuators, in order: hip_1 ankle_1 hip_2 ankle_2 hip_3 ankle_3 hip_4 ankle_4
# Legs 1 and 3 form one diagonal pair, legs 2 and 4 the other, so a trot drives
# those two pairs half a cycle apart.
LEG_PHASE = np.array([0.0, np.pi, np.pi, 0.0])

DEFAULTS = {"hip_amp": 0.6, "ankle_amp": 0.6, "freq": 1.4, "ankle_lag": 1.2}
PARAM_FILE = Path(__file__).with_name("best_gait.json")


def action(t: float, p: dict) -> np.ndarray:
    """Eight joint torques at time t, from four scalar parameters."""
    w = 2.0 * np.pi * p["freq"]
    a = np.zeros(8, dtype=np.float32)
    for leg in range(4):
        phase = w * t + LEG_PHASE[leg]
        a[2 * leg] = p["hip_amp"] * np.sin(phase)
        a[2 * leg + 1] = p["ankle_amp"] * np.sin(phase + p["ankle_lag"])
    return np.clip(a, -1.0, 1.0)


def rollout(env, p: dict, seed: int):
    obs, _ = env.reset(seed=seed)
    dt = env.unwrapped.dt
    total, steps, vx, sq_action = 0.0, 0, [], []
    while True:
        a = action(steps * dt, p)
        obs, reward, terminated, truncated, info = env.step(a)
        total += reward
        steps += 1
        vx.append(info.get("x_velocity", 0.0))
        sq_action.append(float(np.mean(a ** 2)))
        if terminated or truncated:
            break
    return {
        "return": total,
        "steps": steps,
        "x_velocity": float(np.mean(vx)),
        "mean_sq_action": float(np.mean(sq_action)),
    }


def evaluate(p: dict, episodes: int, seed0: int = 0):
    env = gym.make(ENV_ID)
    rows = [rollout(env, p, seed0 + i) for i in range(episodes)]
    env.close()
    keys = rows[0].keys()
    return {k: float(np.mean([r[k] for r in rows])) for k in keys}


def search(episodes: int = 3) -> dict:
    grid = {
        "hip_amp": [0.3, 0.6, 0.9],
        "ankle_amp": [0.3, 0.6, 0.9],
        "freq": [0.8, 1.4, 2.0],
        "ankle_lag": [0.0, 1.2, 2.4],
    }
    names = list(grid)
    best, best_score = None, -1e18
    combos = list(itertools.product(*(grid[k] for k in names)))
    print(f"[gait] grid search over {len(combos)} parameter sets, "
          f"{episodes} episodes each")
    for i, values in enumerate(combos, 1):
        p = dict(zip(names, values))
        stats = evaluate(p, episodes)
        if stats["return"] > best_score:
            best, best_score = p, stats["return"]
            print(f"[gait] {i:>3}/{len(combos)}  new best return "
                  f"{best_score:8.1f}  {p}")
    PARAM_FILE.write_text(json.dumps(best, indent=2))
    print(f"[gait] best parameters written to {PARAM_FILE}")
    return best


def load_params() -> dict:
    if PARAM_FILE.exists():
        return json.loads(PARAM_FILE.read_text())
    return dict(DEFAULTS)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--search", action="store_true", help="run the grid search")
    ap.add_argument("--episodes", type=int, default=20)
    args = ap.parse_args()

    params = search() if args.search else load_params()
    stats = evaluate(params, args.episodes)
    print(f"\n[gait] parameters: {params}")
    print(f"[gait] over {args.episodes} episodes:")
    for k, v in stats.items():
        print(f"         {k:<16} {v:10.3f}")


if __name__ == "__main__":
    main()
