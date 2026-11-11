# Failure log

One line per thing that broke, what we changed, and what happened.

This file is not housekeeping. End-sem Question 1(c) was worth **8 marks** in
2025 and asked how you would redo the project using course concepts. This log
is the raw material for that answer, and it cannot be reconstructed later.

| Date | Who | What broke / what we tried | What we changed | Outcome |
|---|---|---|---|---|
| 2026-11-09 | | Project scaffold created, CPU-only torch chosen because C: had 8 GB free | installed `torch` from the CPU wheel index | venv ~1 GB instead of ~4 GB |
| 2026-11-09 | | `Ant-v5` observation came back as 105-dim, not the 27 we had written | read the v5 docs: 78 external contact-force terms are included by default | README corrected; `include_cfrc_ext_in_observation=False` gives the 27-dim v4 version |
| 2026-11-09 | | Scripted gait fell over after ~16 of 1000 steps, return 7.3 | search grid started at torque amplitude 0.3, far too strong; widened the grid down to 0.05 | tuned gait now survives all 1000 steps, return 1291, and uses ~30x less control effort than a learned policy |
| 2026-11-09 | | A 3000-step untrained PPO already scored return ~1000, which looked like a bug | it is not: Ant pays +1 per step for staying healthy, so standing still for 1000 steps scores ~1000 | **return alone is not a measure of locomotion on this task.** Always report it next to forward velocity. Worth a sentence in the discussion |
| 2026-11-09 | | Relaunch script deleted a live run's `config.json` and checkpoints | `rm -rf runs/ppo_s0 && ...`: the old process still held the monitor CSVs open, so `rm` partially succeeded, returned non-zero, and the `&&` chain silently skipped the whole training loop | use `;` not `&&` between cleanup and work, and never delete a run directory while a process may still own it. The run survived because `train.py` rewrites `config.json` on completion |
| 2026-11-12 | | Robustness table claimed PPO "retained 412%" of its return under stronger gravity | the metric was a ratio against each policy's own unperturbed return; PPO's baseline is ~135 with a comparable s.d., so the ratio was pure noise | switched the table to absolute returns with the unperturbed value as its own column. **Report this as a methodology note**: a retention ratio over a weak, high-variance baseline is meaningless |
| | | | | |
