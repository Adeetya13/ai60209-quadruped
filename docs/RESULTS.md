# Results

Fill cells as runs finish. Regenerate the CSVs with:

    python eval/evaluate.py --all --episodes 20
    python eval/robustness.py --all

## Main table (20 eval episodes, deterministic, seeds 1000+)

| Policy | Return | Fwd velocity (m/s) | Survival steps | Mean sq. action | Steps to return 1000 |
|---|---|---|---|---|---|
| PPO (3 seeds) | | | | | |
| SAC (3 seeds) | | | | | |
| A2C (3 seeds) | | | | | |
| REINFORCE, ours (2 seeds) | | | | | |
| Scripted trot | | | | | |
| Random policy | | | | | |

## Variance-reduction ablation, our REINFORCE

| Variant | Return | Notes |
|---|---|---|
| reward-to-go + value baseline | | full method |
| reward-to-go, no baseline | | |
| full return, no baseline | | the version Sec 5.7 derives before any fix |

## Robustness (percentage of unperturbed return retained)

| Policy | mass x1.5 | mass x2.0 | gravity x1.25 | gravity x1.5 | noise 0.1 | noise 0.2 |
|---|---|---|---|---|---|---|
| PPO | | | | | | |
| SAC | | | | | | |
| Scripted trot | | | | | | |

## Wall clock

| Run | Steps | Minutes | Hardware |
|---|---|---|---|
| | | | |
