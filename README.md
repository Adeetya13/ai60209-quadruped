# Quadruped locomotion: comparing the policy-gradient family

AI60209 AI/ML for Robot Autonomy, Autumn 2026, IIT Kharagpur. Course project.

We implement the policy-gradient algorithms **in the order the course derives
them** and measure what each refinement is actually worth on a quadruped
locomotion task.

| Algorithm | Source | Lecture link |
|---|---|---|
| REINFORCE | written from scratch in `reinforce.py` | Lecture 07, Monte Carlo policy gradient |
| A2C | Stable-Baselines3 | Lecture 07, actor-critic |
| PPO | Stable-Baselines3 | Lecture 07, policy optimisation branch of the RL taxonomy |
| SAC | Stable-Baselines3 | Lecture 07, off-policy actor-critic |
| Scripted trot | `baselines/scripted_gait.py` | non-learned control baseline |
| Random policy | `eval/evaluate.py --policy random` | the floor |

**Environment:** Gymnasium `Ant-v5`, a four-legged MuJoCo robot.
105-dimensional observation (27 pose/velocity terms plus 78 external
contact-force terms), 8-dimensional continuous action (two joints per leg),
episodes capped at 1000 steps. Pass `include_cfrc_ext_in_observation=False`
to get the 27-dim version used by `Ant-v4`.

We use Gymnasium's built-in quadruped rather than a Unitree model. Building a
custom environment around a Unitree MJCF means writing and tuning an
observation space, termination rule and reward function, which the available
time did not allow. `Ant-v5` is a quadruped in a simulation environment, so it
satisfies the project brief, and the time went into the comparison instead.
This is stated plainly in the report.

---

## Setup

Windows, from a path with no spaces. Do **not** put this inside OneDrive.

```
cd C:\dev\quadproj
python -m venv rl_env
rl_env\Scripts\activate
pip install -r requirements.txt
```

Smoke test, should print the observation and action shapes:

```
python -c "import gymnasium as gym; e=gym.make('Ant-v5'); print(e.observation_space.shape, e.action_space.shape)"
```

### Measured throughput on this machine (CPU, 9 Nov)

| Algorithm | Observed | 1M steps would take |
|---|---|---|
| PPO, 2 envs | ~1370 fps | ~12 min |
| PPO, 8 envs | expect ~3000 fps | ~6 min |
| SAC, 1 env | much slower, one gradient step per env step | hours, use Colab |

So PPO and A2C can run locally. **SAC is the one that needs the GPU notebook.**

PyTorch here is the **CPU** build on purpose. Local machines are for
development; training happens on Colab GPUs (see `colab/train_colab.ipynb`).

---

## The run matrix

Fixed before any run was launched. Do not change it once runs have started.

| Runs | Budget | Seeds | Why it is in the report |
|---|---|---|---|
| PPO | 1M steps | 0, 1, 2 | strong on-policy actor-critic |
| SAC | 500k steps | 0, 1, 2 | off-policy, far more sample efficient |
| A2C | 1M steps | 0, 1, 2 | plain actor-critic, expected to trail PPO |
| REINFORCE (ours) | 1M steps | 0, 1 | the from-scratch implementation |
| Scripted trot | n/a | n/a | non-learned baseline |
| Random | n/a | n/a | the floor |

```
python train.py --algo ppo --seed 0
python train.py --algo sac --seed 0
python train.py --algo a2c --seed 0
python reinforce.py --seed 0
```

Each run writes `runs/<tag>/config.json` with the algorithm, seed, timestep
budget and git commit. Nothing goes in the report without one.

### Variance-reduction ablation

The two tricks Lecture 07 proves are flags on our own implementation, so we can
measure them rather than assert them:

```
python reinforce.py --seed 0 --returns to-go --baseline value   # full method
python reinforce.py --seed 0 --returns to-go --baseline none    # no baseline
python reinforce.py --seed 0 --returns full  --baseline none    # neither
```

---

## Evaluation

```
python baselines/scripted_gait.py --search     # tune the gait first, once
python eval/evaluate.py --all --episodes 20    # the metrics table
python eval/plot_curves.py                     # learning curves, seed spread
python eval/robustness.py --all                # payload, gravity, action noise
python eval/record_video.py --run runs/ppo_s0  # mp4 for the slides
```

Outputs land in `docs/metrics.csv`, `docs/robustness.csv` and `figures/`.

### Metrics

| Metric | What it tells you |
|---|---|
| Episode return | the headline number |
| Mean forward velocity | whether it learned to move, or just to stand and collect the healthy bonus |
| Mean survival steps (of 1000) | stability |
| Mean squared action | control effort; the scripted gait may win here |
| Steps to a return threshold | sample efficiency |
| Return retained under perturbation | robustness |

---

## Fairness notes

Things we deliberately held constant so the comparison means something:

- **Identical preprocessing.** Observation normalisation on, reward
  normalisation off, for all three SB3 algorithms.
- **Stable-Baselines3 default hyperparameters throughout.** No per-algorithm
  tuning, because tuning one algorithm and not another would invalidate the
  comparison.
- **Same evaluation protocol.** 20 episodes, deterministic actions, seeds
  1000 onward, which are disjoint from the training seeds.
- **Same seeds.** 0, 1, 2 for every algorithm.

Known asymmetry, stated in the report: SAC gets half the environment steps,
because it is off-policy and sample efficient. The comparison is therefore
"what each algorithm achieves within a realistic compute budget" rather than
"at equal steps". The learning curves are plotted against environment steps so
a reader can judge both.

---

## Layout

```
train.py                  SB3 runner: PPO / SAC / A2C
reinforce.py              REINFORCE from scratch, with ablation flags
baselines/scripted_gait.py   open-loop sinusoidal trot + grid search
eval/evaluate.py          metrics table over all runs
eval/plot_curves.py       learning curves with seed spread
eval/robustness.py        mass, gravity and action-noise sweeps
eval/record_video.py      mp4 clips, including per-checkpoint montage
colab/train_colab.ipynb   GPU training notebook
docs/FAILURE_LOG.md       one line per thing that broke. Keep it current.
docs/RESULTS.md           the metrics table, filled as results arrive
docs/REPORT_OUTLINE.md    report skeleton
runs/                     checkpoints and logs (gitignored)
```

## Team

| Role | Owns | Name |
|---|---|---|
| 1 REINFORCE from scratch | `reinforce.py` | |
| 2 Training runs | `train.py`, `runs/`, Colab | |
| 3 Scripted baseline | `baselines/` | |
| 4 Evaluation and figures | `eval/`, `figures/` | |
| 5 Report and talk | `docs/` | |

One person merges to `main` and keeps it runnable.
