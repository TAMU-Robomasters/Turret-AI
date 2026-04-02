import re
import argparse
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import time
plt.style.use("dark_background")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot training.log metrics and save to PNG.")
    parser.add_argument("--log", type=str, default="checkpoints/training.log")
    parser.add_argument("--out", type=str, default="plots/training_plot.png")
    parser.add_argument("--max-points", type=int, default=2000)
    return parser.parse_args()


def live_plot() -> None:
    args = parse_args()

    rollout_re = re.compile(r"\[Iter (\d+)\].*?rollout/reward_mean: ([\-0-9\.eE]+)")
    eval_re = re.compile(r"\[Iter (\d+)\].*?eval/reward_mean: ([\-0-9\.eE]+)")

    rollout_x, rollout_y = [], []
    eval_x, eval_y = [], []

    with open(args.log, "r") as f:
        for line in f:
            m = rollout_re.search(line)
            if m:
                rollout_x.append(int(m.group(1)))
                rollout_y.append(float(m.group(2)))
            m = eval_re.search(line)
            if m:
                eval_x.append(int(m.group(1)))
                eval_y.append(float(m.group(2)))

    if args.max_points > 0:
        rollout_x = rollout_x[-args.max_points:]
        rollout_y = rollout_y[-args.max_points:]
        eval_x = eval_x[-args.max_points:]
        eval_y = eval_y[-args.max_points:]

    fig, ax1 = plt.subplots()
    fig.patch.set_facecolor("#0e0f12")
    ax1.set_facecolor("#0e0f12")

    # Create second y-axis
    ax2 = ax1.twinx()
    ax2.set_facecolor("#0e0f12")

    # Plot rollout on left axis
    if rollout_x:
        ax1.plot(rollout_x, rollout_y, label="rollout/reward_mean")
        ax1.set_ylabel("Rollout Reward")

    # Plot eval on right axis
    if eval_x:
        ax2.plot(eval_x, eval_y, linestyle="--", label="eval/reward_mean", color="red")
        ax2.set_ylabel("Eval Reward")

    ax1.set_xlabel("Iteration")
    ax1.set_title("Training Performance")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    fig.clear()
    plt.close(fig)


if __name__ == "__main__":
    while True:
        live_plot()
        time.sleep(5)
