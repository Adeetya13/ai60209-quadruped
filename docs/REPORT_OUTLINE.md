# Report outline

Six to eight pages. Skeleton goes up Monday, filled as results arrive. Starting
from a blank page on Friday night is how teams submit something incoherent.

## 1. Problem (0.5 page)
Quadruped locomotion. Why it is hard: underactuated, contact-rich, 8 continuous
torques at ~50 Hz, reward is sparse in the sense that standing still is a local
optimum.

## 2. MDP formulation (1 page)
State, action, transition, reward, termination for `Ant-v5`, written out
properly. Note that position/velocity both appear in the state because position
alone is not Markov.

## 3. Algorithms (1.5 pages)
The ladder, with the update rule for each and a sentence on what it adds:
- REINFORCE: the policy gradient theorem, reward-to-go, baseline
- A2C: bootstrapped critic replaces the Monte Carlo return
- PPO: clipped surrogate objective limits the step size
- SAC: off-policy, entropy-regularised
Plus the scripted trot: open-loop sinusoids, no feedback.

## 4. Our REINFORCE implementation (1 page)
Design choices, the two ablation flags, and what we got wrong first. Pull
specifics from FAILURE_LOG.md.

## 5. Experimental setup (0.5 page)
Run matrix, seeds, budgets, evaluation protocol, the fairness notes from the
README, and the one known asymmetry (SAC's smaller step budget).

## 6. Results (2 pages)
- Learning curves, mean with shaded spread across seeds. **Say explicitly that
  the band is the seed variance Lecture 07 warns about.**
- Metrics table
- Variance-reduction ablation
- Robustness table

## 7. Discussion (1 page)
What each rung of the ladder bought. Where the scripted baseline wins and why.
What the seed spread implies about single-run results in the literature.

## 8. Limitations and what we would do differently (0.5 page)
Ant rather than a Unitree model, and why. Then name course concepts: a Go2 MJCF
with a shaped reward, MPC for the body trajectory, Lyapunov analysis of the
learned closed loop, imitation from a reference gait. This paragraph is the
answer to end-sem Question 1(c).
