"""
Simple Population-Based Training (PBT) launcher for GRU training.

This script runs a population of independent train_cuda.py jobs from scratch
each generation. No checkpoint resume or copying is performed.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import subprocess
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class PBTConfig:
    lr: float
    clip_coef: float
    std_init: float
    predicted_yaw_slope: float
    predicted_yaw_tolerance: float
    miss_penalty: float
    hit_reward: float


def _default_config() -> PBTConfig:
    return PBTConfig(
        lr=1e-4,
        clip_coef=0.05,
        std_init=0.2,
        predicted_yaw_slope=10.0,
        predicted_yaw_tolerance=2.0,
        miss_penalty=150.0,
        hit_reward=200.0,
    )


def _mutate(cfg: PBTConfig, rng: random.Random) -> PBTConfig:
    def jitter(x: float, lo: float, hi: float, scale: float) -> float:
        factor = math.exp(rng.uniform(-scale, scale))
        return float(min(max(x * factor, lo), hi))

    return PBTConfig(
        lr=jitter(cfg.lr, 1e-6, 1e-3, 0.4),
        clip_coef=jitter(cfg.clip_coef, 0.01, 0.3, 0.3),
        std_init=jitter(cfg.std_init, 0.05, 1.0, 0.4),
        predicted_yaw_slope=jitter(cfg.predicted_yaw_slope, 1.0, 200.0, 0.5),
        predicted_yaw_tolerance=jitter(cfg.predicted_yaw_tolerance, 0.2, 30.0, 0.5),
        miss_penalty=jitter(cfg.miss_penalty, 10.0, 500.0, 0.4),
        hit_reward=jitter(cfg.hit_reward, 10.0, 500.0, 0.4),
    )


def _parse_last_eval_reward(log_path: str) -> float:
    if not os.path.exists(log_path):
        return float("-inf")
    last = None
    with open(log_path, "r") as f:
        for line in f:
            if "eval/reward_mean" in line:
                last = line.strip()
    if last is None:
        return float("-inf")
    try:
        key = "eval/reward_mean:"
        idx = last.index(key) + len(key)
        return float(last[idx:].split("|")[0].strip())
    except Exception:
        return float("-inf")


def _run_member(
    member_id: int,
    cfg: PBTConfig,
    base_args: List[str],
    save_root: str,
    timesteps: int,
    generation: int,
) -> float:
    save_dir = os.path.join(save_root, f"gen_{generation:02d}", f"member_{member_id:02d}")
    os.makedirs(save_dir, exist_ok=True)

    cmd = [
        "python", "train_cuda.py",
        "--save-dir", save_dir,
        "--total-timesteps", str(timesteps),
        "--lr", str(cfg.lr),
        "--clip-coef", str(cfg.clip_coef),
        "--std-init", str(cfg.std_init),
        "--predicted-yaw-slope", str(cfg.predicted_yaw_slope),
        "--predicted-yaw-tolerance", str(cfg.predicted_yaw_tolerance),
        "--miss-penalty", str(cfg.miss_penalty),
        "--hit-reward", str(cfg.hit_reward),
    ] + base_args

    print(f"\n=== PBT member {member_id} (gen {generation}) ===")
    print(" ".join(cmd))
    subprocess.run(cmd, check=False)

    log_path = os.path.join(save_dir, "training.log")
    score = _parse_last_eval_reward(log_path)
    print(f"Member {member_id} last eval reward: {score:.2f}")
    return score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--population", type=int, default=6)
    parser.add_argument("--generations", type=int, default=3)
    parser.add_argument("--timesteps", type=int, default=500_000)
    parser.add_argument("--save-root", type=str, default="pbt_runs")
    parser.add_argument("--seed", type=int, default=0)

    # Common train_cuda args
    parser.add_argument("--n-envs", type=int, default=128)
    parser.add_argument("--n-steps", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--n-epochs", type=int, default=2)
    parser.add_argument("--target-kl", type=float, default=0.02)
    parser.add_argument("--hidden-dim", type=int, default=16)
    parser.add_argument("--num-layers", type=int, default=16)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    base_args = [
        "--n-envs", str(args.n_envs),
        "--n-steps", str(args.n_steps),
        "--batch-size", str(args.batch_size),
        "--n-epochs", str(args.n_epochs),
        "--target-kl", str(args.target_kl),
        "--hidden-dim", str(args.hidden_dim),
        "--num-layers", str(args.num_layers),
    ]

    os.makedirs(args.save_root, exist_ok=True)
    with open(os.path.join(args.save_root, "pbt_config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    # Initialize population
    base = _default_config()
    pop = [_mutate(base, rng) for _ in range(args.population)]

    # Generational loop (from scratch each gen)
    for gen in range(1, args.generations + 1):
        scores: List[Tuple[int, float]] = []
        for i, cfg in enumerate(pop):
            score = _run_member(i, cfg, base_args, args.save_root, args.timesteps, gen)
            scores.append((i, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        print(f"\nBest member in gen {gen}: {scores[0][0]} reward={scores[0][1]:.2f}")

        # Mutate bottom half toward top half for next generation
        half = max(1, len(pop) // 2)
        top_indices = [idx for idx, _ in scores[:half]]
        bottom_indices = [idx for idx, _ in scores[half:]]
        for b, t in zip(bottom_indices, top_indices * (len(bottom_indices) // max(1, len(top_indices)) + 1)):
            pop[b] = _mutate(pop[t], rng)


if __name__ == "__main__":
    main()
