import argparse
import csv
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot GRU training.log CSV metrics to PNG.")
    parser.add_argument("--log", type=str, default="checkpoints/training.log")
    parser.add_argument("--out", type=str, default="plots/gru_training_plot.png")
    parser.add_argument("--max-points", type=int, default=2000)
    return parser.parse_args()


def _read_rows(log_path: Path) -> list[dict]:
    if not log_path.exists():
        return []
    with log_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        return [row for row in reader if row]


def live_plot() -> None:
    args = parse_args()
    log_path = Path(args.log)

    rows = _read_rows(log_path)
    if not rows:
        return

    train_x, train_y = [], []
    train_avg_x, train_avg_y = [], []
    eval_x, eval_y = [], []

    for row in rows:
        row_type = row.get("type", "").strip()
        ep = row.get("episode", "").strip()
        if not ep:
            continue
        try:
            episode = int(float(ep))
        except ValueError:
            continue

        if row_type == "train":
            if "return" in row and row["return"]:
                train_x.append(episode)
                train_y.append(float(row["return"]))
            if "avg_return" in row and row["avg_return"]:
                train_avg_x.append(episode)
                train_avg_y.append(float(row["avg_return"]))
        elif row_type == "eval":
            if "eval_return" in row and row["eval_return"]:
                eval_x.append(episode)
                eval_y.append(float(row["eval_return"]))

    if args.max_points > 0:
        train_x = train_x[-args.max_points:]
        train_y = train_y[-args.max_points:]
        train_avg_x = train_avg_x[-args.max_points:]
        train_avg_y = train_avg_y[-args.max_points:]
        eval_x = eval_x[-args.max_points:]
        eval_y = eval_y[-args.max_points:]

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    if train_x:
        ax1.plot(train_x, train_y, label="train/return", alpha=0.35)
    if train_avg_x:
        ax1.plot(train_avg_x, train_avg_y, label="train/avg_return")
    if eval_x:
        ax2.plot(eval_x, eval_y, linestyle="--", color="red", label="eval/return")

    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Train Return")
    ax2.set_ylabel("Eval Return")
    ax1.set_title("GRU Training Performance")

    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="lower left")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    fig.clear()
    plt.close(fig)


if __name__ == "__main__":
    while True:
        live_plot()
        time.sleep(10)
