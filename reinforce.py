"""REINFORCE from scratch, on Ant-v5.

This is the from-scratch implementation for the report. It follows the
derivation in Lecture 07 exactly, and the two variance-reduction steps the
lecture proves are command-line flags so they can be ablated:

    --returns full        total episode return G(tau) on every step  (Sec 5.7)
    --returns to-go       reward-to-go G_t                           (Sec 5.8)
    --baseline none       no baseline
    --baseline mean       subtract the batch mean of G_t
    --baseline value      subtract a learned V_w(s)                  (Sec 5.11)

Default is the full method: reward-to-go with a learned value baseline.

    python reinforce.py --seed 0
    python reinforce.py --seed 0 --returns full --baseline none   # the ablation
"""
import argparse
import csv
import json
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal

ENV_ID = "Ant-v5"


def mlp(sizes, act=nn.Tanh, out_act=nn.Identity):
    layers = []
    for i in range(len(sizes) - 1):
        layers += [nn.Linear(sizes[i], sizes[i + 1])]
        layers += [act() if i < len(sizes) - 2 else out_act()]
    return nn.Sequential(*layers)


class GaussianPolicy(nn.Module):
    """pi(a|s) = N(mu_theta(s), sigma^2), with a state-independent log sigma.

    This is the parameterisation from Lecture 07: the mean comes from the
    network, and log sigma is a free parameter so sigma stays positive.
    """

    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 64):
        super().__init__()
        self.mu = mlp([obs_dim, hidden, hidden, act_dim])
        self.log_std = nn.Parameter(-0.5 * torch.ones(act_dim))

    def distribution(self, obs: torch.Tensor) -> Normal:
        return Normal(self.mu(obs), self.log_std.exp())

    def log_prob(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        # Sum over action dimensions: independent Gaussians per joint.
        return self.distribution(obs).log_prob(act).sum(axis=-1)


def discounted_to_go(rewards, gamma: float):
    """G_t = sum_{k=t}^{T-1} gamma^(k-t) R_{k+1}, computed backwards in one pass."""
    out = np.zeros(len(rewards), dtype=np.float32)
    running = 0.0
    for i in reversed(range(len(rewards))):
        running = rewards[i] + gamma * running
        out[i] = running
    return out


def collect_batch(env, policy, batch_steps: int, gamma: float, returns_mode: str):
    """Run the policy until batch_steps transitions are gathered."""
    obs_buf, act_buf, ret_buf, ep_returns, ep_lengths = [], [], [], [], []
    obs, _ = env.reset()
    ep_rewards = []

    while len(obs_buf) < batch_steps:
        obs_t = torch.as_tensor(obs, dtype=torch.float32)
        with torch.no_grad():
            action = policy.distribution(obs_t).sample().numpy()
        clipped = np.clip(action, env.action_space.low, env.action_space.high)
        next_obs, reward, terminated, truncated, _ = env.step(clipped)

        obs_buf.append(obs)
        act_buf.append(action)
        ep_rewards.append(float(reward))
        obs = next_obs

        if terminated or truncated:
            if returns_mode == "to-go":
                ret_buf.extend(discounted_to_go(ep_rewards, gamma))
            else:  # "full": every step gets the whole episode return
                total = float(discounted_to_go(ep_rewards, gamma)[0])
                ret_buf.extend([total] * len(ep_rewards))
            ep_returns.append(sum(ep_rewards))
            ep_lengths.append(len(ep_rewards))
            ep_rewards = []
            obs, _ = env.reset()

    # Drop a trailing partial episode so returns and observations stay aligned.
    n = len(ret_buf)
    return (
        torch.as_tensor(np.array(obs_buf[:n]), dtype=torch.float32),
        torch.as_tensor(np.array(act_buf[:n]), dtype=torch.float32),
        torch.as_tensor(np.array(ret_buf), dtype=torch.float32),
        ep_returns,
        ep_lengths,
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=1_000_000)
    p.add_argument("--batch-steps", type=int, default=5000)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--vf-lr", type=float, default=1e-3)
    p.add_argument("--returns", choices=["to-go", "full"], default="to-go")
    p.add_argument("--baseline", choices=["none", "mean", "value"], default="value")
    p.add_argument("--out", default="runs")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    tag = f"reinforce_s{args.seed}"
    if args.returns != "to-go" or args.baseline != "value":
        tag += f"_{args.returns}_{args.baseline}"
    run_dir = Path(args.out) / tag
    run_dir.mkdir(parents=True, exist_ok=True)

    env = gym.make(ENV_ID)
    env.reset(seed=args.seed)
    env.action_space.seed(args.seed)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    policy = GaussianPolicy(obs_dim, act_dim)
    pi_opt = torch.optim.Adam(policy.parameters(), lr=args.lr)

    value_net, v_opt = None, None
    if args.baseline == "value":
        value_net = mlp([obs_dim, 64, 64, 1])
        v_opt = torch.optim.Adam(value_net.parameters(), lr=args.vf_lr)

    (run_dir / "config.json").write_text(json.dumps({
        "env": ENV_ID, "algo": "reinforce", "seed": args.seed,
        "total_timesteps": args.steps, "batch_steps": args.batch_steps,
        "gamma": args.gamma, "lr": args.lr, "returns": args.returns,
        "baseline": args.baseline, "implementation": "from scratch",
        "started": time.strftime("%Y-%m-%d %H:%M:%S"),
    }, indent=2))

    log_path = run_dir / "progress.csv"
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(
            ["timesteps", "mean_ep_return", "std_ep_return", "mean_ep_len", "n_eps"]
        )

    steps_done, t0 = 0, time.time()
    while steps_done < args.steps:
        obs_b, act_b, ret_b, ep_returns, ep_lengths = collect_batch(
            env, policy, args.batch_steps, args.gamma, args.returns
        )
        steps_done += len(obs_b)

        # ---- the policy gradient step -------------------------------------
        # grad J = E[ (G_t - b(s_t)) * grad log pi(a_t|s_t) ].
        # We maximise J, so the loss we minimise is its negative.
        if args.baseline == "none":
            advantage = ret_b
        elif args.baseline == "mean":
            advantage = ret_b - ret_b.mean()
        else:
            with torch.no_grad():
                advantage = ret_b - value_net(obs_b).squeeze(-1)
        advantage = advantage / (advantage.std() + 1e-8)

        pi_opt.zero_grad()
        loss = -(policy.log_prob(obs_b, act_b) * advantage).mean()
        loss.backward()
        pi_opt.step()

        if value_net is not None:
            for _ in range(5):
                v_opt.zero_grad()
                v_loss = ((value_net(obs_b).squeeze(-1) - ret_b) ** 2).mean()
                v_loss.backward()
                v_opt.step()

        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([
                steps_done,
                round(float(np.mean(ep_returns)), 2),
                round(float(np.std(ep_returns)), 2),
                round(float(np.mean(ep_lengths)), 1),
                len(ep_returns),
            ])
        print(f"[reinforce] {steps_done:>8,} steps  "
              f"return {np.mean(ep_returns):8.1f} +/- {np.std(ep_returns):6.1f}  "
              f"len {np.mean(ep_lengths):6.1f}")

    torch.save({"policy": policy.state_dict(), "obs_dim": obs_dim,
                "act_dim": act_dim}, run_dir / "final_model.pt")
    print(f"[reinforce] done in {(time.time()-t0)/60:.1f} min -> {run_dir}")


if __name__ == "__main__":
    main()
