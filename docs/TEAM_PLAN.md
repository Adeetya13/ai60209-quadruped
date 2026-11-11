AI60209 QUADRUPED PROJECT - TEAM PLAN

Submit: Sunday 15 Nov, 2-3 AM.  Code freeze: Saturday 14 Nov, 9 PM.
Full detail, results and figures: https://claude.ai/artifact/6x8z1dryCcMo7unq1VeMdL


WHAT WE ARE DOING

We implement the policy-gradient algorithms in the order the course taught
them (REINFORCE, then A2C, PPO, SAC) and measure what each one is actually
worth on a quadruped. Environment is Gymnasium Ant-v5, a four-legged MuJoCo
robot. We also wrote a hand-coded sine-wave gait as a non-learned baseline.

It is a measurement study, not an attempt at something novel. We beat last
year on rigour: three seeds per algorithm with the spread reported, one
algorithm written from scratch, and a tuned baseline to compare against.


WHAT IS ALREADY DONE

- All 10 Ant runs trained and evaluated: PPO x3 seeds, A2C x3, REINFORCE x2,
  plus 2 ablation variants
- Robustness sweep done: added mass, stronger gravity, action noise
- Variance-reduction ablation rerunning on HalfCheetah right now
- All code written and verified end to end
- Results in docs/RESULTS.md, figures in figures/


HEADLINE RESULTS (these go in the report)

- A2C is the only method that learned to locomote: 1.10 m/s
- Our scripted sine-wave gait beat both PPO and REINFORCE
- REINFORCE scored 962 but moves at 0.06 m/s. It learned to stand still and
  collect the +1 per step healthy bonus, which is the local optimum the
  course warns about
- PPO with default hyperparameters is unstable: falls after about 200 steps
- A2C gets BETTER under added mass and gravity, because the extra weight
  keeps the robot upright longer


WHO DOES WHAT


PERSON 1 - SAC RUNS            deadline: Friday night
Blocked until Adeetya pushes the repo. Ping him if it is not up.

1. Open colab/train_colab.ipynb in Google Colab
2. Runtime > Change runtime type > T4 GPU
3. In cell 5, set REPO to our GitHub URL
4. Run the mount, install and clone cells
5. In the training cell set ALGO = "sac", SEED = 0, run it
6. Repeat for SEED = 1 and SEED = 2
7. Download the sac_s0, sac_s1, sac_s2 folders from Drive and send them to
   Adeetya to drop into runs/
8. Tell Adeetya when they land so he regenerates the tables

SAC is the algorithm most likely to beat A2C, so this is the highest-value
run left.


PERSON 2 - VIDEOS AND FIGURES  deadline: Friday night
Do the setup below first.

1. python eval/record_video.py --run runs/a2c_s0
   (our best mover, this is the opening slide)
2. python eval/record_video.py --policy scripted
   (the baseline, for the side-by-side)
3. python eval/record_video.py --run runs/reinforce_s0
   (shows it standing still, this proves our headline finding visually)
4. python eval/record_video.py --run runs/a2c_s0 --all-checkpoints
   (the flailing-to-running montage)
5. Open figures/learning_curves.png and check it is readable at projector
   size, not just on a laptop
6. Put every mp4 in a shared Drive folder and post the link


PERSON 3 - REPORT, METHOD HALF    deadline: Friday night
Sections 1 to 5 of docs/REPORT_OUTLINE.md. Roughly 4 pages.

1. Problem: quadruped locomotion and why it is hard
2. MDP formulation of Ant-v5: state (105-dim), action (8 joint torques),
   reward, termination. Write it out properly, do not describe it in prose
3. The four algorithms with their update rules, each linked to the lecture
   it came from
4. Our REINFORCE: design choices and the two ablation flags
5. Experimental setup: run matrix, seeds, budgets, and the fairness
   constraints (identical preprocessing, SB3 defaults throughout, same
   seeds, same eval protocol)


PERSON 4 - REPORT, RESULTS HALF   deadline: Saturday noon
Sections 6 to 8. Roughly 4 pages.

1. Results: paste the tables straight from docs/RESULTS.md and reference
   the figures. Always show forward velocity next to return, because return
   alone is misleading on this task
2. Discussion: what each rung of the ladder bought, where the scripted
   baseline wins and why, what the seed spread implies
3. Read docs/FAILURE_LOG.md and write the "what went wrong" section from
   it. There are six real incidents logged, including a robustness metric
   we had to throw away
4. Limitations and what we would do differently. NAME COURSE CONCEPTS here:
   a Unitree Go2 model with a shaped reward, MPC for the body trajectory,
   Lyapunov analysis of the learned closed loop. This paragraph is the
   answer to an 8-mark end-sem question, so do not rush it


PERSON 5 - SLIDES AND REHEARSAL   deadline: Saturday 6 PM
Six slides, assume ten minutes.

1. Opening slide is the A2C VIDEO, not a title card
2. MDP formulation, one slide
3. All four learning curves on one axis with the seed band. Say out loud
   that the band is the seed variance Lecture 07 warns about
4. Why REINFORCE stood still, tied to the variance argument
5. Scripted baseline vs learned policy, side-by-side video
6. Honest limitations, including that we used Ant rather than a Unitree
   robot, and why

Then rehearse twice with a timer. Everyone must be able to explain any
slide, because questions do not go to the person who made it.


ADEETYA
- Push the repo to GitHub TODAY. Person 1 is blocked until this happens
- Finish the HalfCheetah ablation (running now)
- Regenerate all tables once SAC lands
- Integration: only one person merges to main, and that is me


TIMELINE

Thu 12 Nov          repo pushed, ablation done, everyone set up
Fri 13 Nov          SAC done, videos done, report sections 1-5 drafted
Sat 14 Nov, 12 PM   report complete, both halves merged
Sat 14 Nov, 6 PM    slides done, first rehearsal
Sat 14 Nov, 9 PM    CODE FREEZE - nothing new is written after this
Sat 14 Nov, 10 PM   second rehearsal
Sun 15 Nov, 2-3 AM  SUBMIT


SETUP - everyone, about 10 minutes

Use a path with NO SPACES. Do not put it inside OneDrive: MuJoCo mishandles
spaces and OneDrive will fight you over the thousands of files a training
run produces.

  git clone <repo URL> C:\dev\quadproj
  cd C:\dev\quadproj
  python -m venv rl_env
  rl_env\Scripts\activate
  pip install torch --index-url https://download.pytorch.org/whl/cpu
  pip install "gymnasium[mujoco]" stable-baselines3 tensorboard matplotlib pandas imageio imageio-ffmpeg

Check it works:

  python -c "import gymnasium as gym; e=gym.make('Ant-v5'); print(e.observation_space.shape)"

Should print (105,). Install CPU torch first, it keeps the environment about
1 GB instead of 4 GB.


TWO RULES THAT ARE NOT OPTIONAL

1. Every training run writes its own config.json with algorithm, seed,
   budget and git commit. train.py does this automatically. Do not bypass
   it, or we end up with curves nobody can label.

2. When something breaks, add ONE line to docs/FAILURE_LOG.md: what broke,
   what you changed, what happened. That log is the raw material for an
   8-mark end-sem question and it cannot be reconstructed afterwards.


IF WE FALL BEHIND, CUT IN THIS ORDER

1. SAC seeds 1 and 2, keeping seed 0
2. The checkpoint montage video
3. The robustness section of the report

NEVER CUT: the three-seed spread, the from-scratch REINFORCE, or the
failure log. Those three are what the submission is actually judged on.


ONE MORE THING

The end-sem is Thursday 26 Nov, 3 to 5 PM. In 2025 its first question was
20 of 80 marks and asked only about your own project: what you learned (5),
what you learned from OTHER TEAMS' presentations (5), how you would redo it
using course concepts (8), and one topic not covered (2).

So two things cost nothing now and pay later: keep the failure log current,
and actually attend the other presentations and take notes.
