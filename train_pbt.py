"""
Population-Based Training (PBT) launcher based on train_cuda.py.

This is a lightweight PBT manager that spawns multiple training runs with
different hyperparameters, evaluates them, and then mutates the weakest
configs toward the best.

NOTE:
  This script currently restarts each member from scratch per generation,
  because train_cuda.py does not yet implement a true resume mechanism.
  If you later add resume support, this script can promote the best model
  by copying its checkpoint and continuing training (see TODOs below).
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import shutil
import subprocess
import time
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple


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
        lr=1e-5,
        clip_coef=0.05,
        std_init=0.2,
        predicted_yaw_slope=10.0,
        predicted_yaw_tolerance=2.0,
        miss_penalty=150.0,
        hit_reward=200.0,
    )


def _mutate(cfg: PBTConfig, rng: random.Random) -> PBTConfig:
    """Mutate hyperparams with small multiplicative noise."""
    out = copy.deepcopy(cfg)
    def jitter(x: float, lo: float, hi: float, scale: float = 0.3) -> float:
        factor = math.exp(rng.uniform(-scale, scale))
        return float(min(max(x * factor, lo), hi))

    out.lr = jitter(out.lr, 1e-6, 1e-3)
    out.clip_coef = jitter(out.clip_coef, 0.01, 0.3, scale=0.25)
    out.std_init = jitter(out.std_init, 0.05, 1.0, scale=0.4)
    out.predicted_yaw_slope = jitter(out.predicted_yaw_slope, 1.0, 200.0, scale=0.4)
    out.predicted_yaw_tolerance = jitter(out.predicted_yaw_tolerance, 0.2, 30.0, scale=0.4)
    out.miss_penalty = jitter(out.miss_penalty, 10.0, 500.0, scale=0.4)
    out.hit_reward = jitter(out.hit_reward, 10.0, 500.0, scale=0.4)
    return out


def _parse_last_eval_reward(log_path: str) -> float:
    """Return last eval reward from training.log (or -inf if not found)."""
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
        # Example: [Iter 10] eval/reward_mean: 123.4 | ...
        prefix = "eval/reward_mean:"
        idx = last.index(prefix) + len(prefix)
        val = float(last[idx:].split("|")[0].strip())
        return val
    except Exception:
        return float("-inf")


def _run_member(
    member_id: int,
    cfg: PBTConfig,
    base_args: List[str],
    save_root: str,
    timesteps: int,
    resume_path: str | None = None,
) -> float:
    save_dir = os.path.join(save_root, f"member_{member_id:02d}")
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
    if resume_path:
        cmd += ["--resume", resume_path]

    print(f"\n=== PBT member {member_id} ===")
    print(" ".join(cmd))
    subprocess.run(cmd, check=False)

    log_path = os.path.join(save_dir, "training.log")
    score = _parse_last_eval_reward(log_path)
    print(f"Member {member_id} last eval reward: {score:.2f}")
    return score


def _find_latest_checkpoint(save_dir: str) -> str | None:
    """Return path to latest checkpoint_{iter}.pt or None."""
    if not os.path.isdir(save_dir):
        return None
    best_iter = -1
    best_path = None
    for name in os.listdir(save_dir):
        if name.startswith("checkpoint_") and name.endswith(".pt"):
            try:
                it = int(name[len("checkpoint_"):-3])
            except ValueError:
                continue
            if it > best_iter:
                best_iter = it
                best_path = os.path.join(save_dir, name)
    return best_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--population", type=int, default=6)
    parser.add_argument("--generations", type=int, default=3)
    parser.add_argument("--timesteps", type=int, default=500_000)
    parser.add_argument("--save-root", type=str, default="pbt_runs")
    parser.add_argument("--seed", type=int, default=0)
    # Forwarded train_cuda args (common)
    parser.add_argument("--n-envs", type=int, default=128)
    parser.add_argument("--n-steps", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--n-epochs", type=int, default=2)
    parser.add_argument("--target-kl", type=float, default=0.02)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=4)
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
    pop: List[PBTConfig] = []
    base = _default_config()
    for _ in range(args.population):
        pop.append(_mutate(base, rng))

    # Generational PBT loop
    resume_paths: Dict[int, str | None] = {i: None for i in range(args.population)}
    for gen in range(args.generations):
        print(f"\n=== Generation {gen + 1}/{args.generations} ===")
        scores: List[Tuple[int, float]] = []
        for i, cfg in enumerate(pop):
            score = _run_member(i, cfg, base_args, args.save_root, args.timesteps, resume_paths.get(i))
            scores.append((i, score))

        # Rank
        scores.sort(key=lambda x: x[1], reverse=True)
        best_idx = scores[0][0]
        print(f"Best member: {best_idx} reward={scores[0][1]:.2f}")

        # Exploit/explore: replace bottom half with mutated copies of top half
        half = max(1, len(pop) // 2)
        top_indices = [idx for idx, _ in scores[:half]]
        bottom_indices = [idx for idx, _ in scores[half:]]
        for b, t in zip(bottom_indices, top_indices * (len(bottom_indices) // max(1, len(top_indices)) + 1)):
            pop[b] = _mutate(pop[t], rng)
            # Copy best member checkpoint to resume from
            src_dir = os.path.join(args.save_root, f"member_{t:02d}")
            dst_dir = os.path.join(args.save_root, f"member_{b:02d}")
            src_ckpt = _find_latest_checkpoint(src_dir)
            if src_ckpt:
                os.makedirs(dst_dir, exist_ok=True)
                dst_ckpt = os.path.join(dst_dir, "resume.pt")
                shutil.copy2(src_ckpt, dst_ckpt)
                resume_paths[b] = dst_ckpt
            else:
                resume_paths[b] = None

        time.sleep(0.1)


if __name__ == "__main__":
    main()
