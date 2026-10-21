import argparse
import json
import math
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

STEP = re.compile(
    r"step\s+(\d+)/\d+ \| loss ([\d.]+) \| lr ([\de.+-]+) \| grad ([\d.]+) "
    r"\| ([\d.]+) s/step \| ([\d.]+)k tok/s"
)
BLUE = "#1a73e8"
RED = "#d93025"
GREEN = "#188038"
GRAY = "#9aa0a6"

plt.rcParams.update(
    {
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "legend.frameon": False,
    }
)


def load(run_dir):
    hist = json.loads((Path(run_dir) / "history.json").read_text())
    run = {"step": [], "loss": [], "lr": [], "grad": [], "sec": [], "speed": []}
    for line in (Path(run_dir) / "log.txt").read_text().splitlines():
        match = STEP.search(line)
        if not match:
            continue
        run["step"].append(int(match[1]))
        run["loss"].append(float(match[2]))
        run["lr"].append(float(match[3]))
        run["grad"].append(float(match[4]))
        run["sec"].append(float(match[5]))
        run["speed"].append(float(match[6]) * 1000)
    run["hist"] = hist
    return run


def smooth(xs, window):
    out = []
    total = 0.0
    for i, x in enumerate(xs):
        total += x
        if i >= window:
            total -= xs[i - window]
            out.append(total / window)
        else:
            out.append(total / (i + 1))
    return out


def save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(f"wrote {path}")


def loss_plot(run, path):
    hist = run["hist"]
    best = min(hist, key=lambda r: r["val"])
    fig, ax = plt.subplots(figsize=(9, 4.6))
    ax.plot(run["step"], run["loss"], color=GRAY, lw=1, alpha=0.55, label="training batch loss")
    ax.plot(
        [r["step"] for r in hist],
        [r["train"] for r in hist],
        color=BLUE,
        lw=2,
        label="training loss at eval",
    )
    ax.plot(
        [r["step"] for r in hist],
        [r["val"] for r in hist],
        color=RED,
        lw=2,
        label="validation loss",
    )
    ax.axvline(best["step"], color=GREEN, ls="--", lw=1.2)
    ax.annotate(
        f"best validation loss {best['val']:.4f} at step {best['step']}",
        xy=(best["step"], best["val"]),
        xytext=(best["step"] + 220, 6.0),
        arrowprops={"arrowstyle": "->", "color": GREEN},
        color=GREEN,
    )
    ax.set_title("Training and validation loss")
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    ax.set_xlim(0, max(run["step"]))
    ax.grid(alpha=0.25)
    ax.legend()
    save(fig, path)


def ppl_plot(run, path):
    hist = run["hist"]
    best = min(hist, key=lambda r: r["val"])
    fig, ax = plt.subplots(figsize=(9, 4.4))
    ax.semilogy(
        [r["step"] for r in hist],
        [math.exp(r["val"]) for r in hist],
        color=RED,
        marker="o",
        ms=3,
        lw=1.8,
    )
    ax.axvline(best["step"], color=GREEN, ls="--", lw=1.2)
    ax.annotate(
        f"best perplexity {math.exp(best['val']):.2f} at step {best['step']}",
        xy=(best["step"], math.exp(best["val"])),
        xytext=(best["step"] + 220, 250),
        arrowprops={"arrowstyle": "->", "color": GREEN},
        color=GREEN,
    )
    ax.set_title("Validation perplexity")
    ax.set_xlabel("step")
    ax.set_ylabel("perplexity")
    ax.set_xlim(0, max(run["step"]))
    ax.grid(alpha=0.25, which="both")
    save(fig, path)


def lr_plot(run, path):
    fig, ax = plt.subplots(figsize=(9, 4.0))
    ax.plot(run["step"], run["lr"], color=BLUE, lw=2)
    ax.axvline(100, color=GRAY, ls="--", lw=1)
    ax.annotate("warmup ends", xy=(100, max(run["lr"]) * 0.9), xytext=(160, max(run["lr"]) * 0.9), color=GRAY)
    ax.set_title("Learning rate schedule")
    ax.set_xlabel("step")
    ax.set_ylabel("learning rate")
    ax.set_xlim(0, max(run["step"]))
    ax.grid(alpha=0.25)
    save(fig, path)


def grad_plot(run, path):
    fig, ax = plt.subplots(figsize=(9, 4.0))
    ax.plot(run["step"], run["grad"], color=GRAY, lw=1, alpha=0.5, label="gradient norm")
    ax.plot(run["step"], smooth(run["grad"], 25), color=BLUE, lw=2, label="smoothed")
    ax.set_title("Gradient norm")
    ax.set_xlabel("step")
    ax.set_ylabel("norm")
    ax.set_xlim(0, max(run["step"]))
    ax.grid(alpha=0.25)
    ax.legend()
    save(fig, path)


def speed_plot(run, path):
    fig, ax = plt.subplots(figsize=(9, 4.0))
    ax.plot(run["step"], run["speed"], color=GRAY, lw=1, alpha=0.5, label="tokens per second")
    ax.plot(run["step"], smooth(run["speed"], 25), color=BLUE, lw=2, label="smoothed")
    ax.yaxis.set_major_formatter(lambda x, _: f"{x / 1000:.0f}k")
    ax.set_title("Training throughput")
    ax.set_xlabel("step")
    ax.set_ylabel("tokens per second")
    ax.set_xlim(0, max(run["step"]))
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.25)
    ax.legend()
    save(fig, path)


def main():
    ap = argparse.ArgumentParser(description="plot the training run")
    ap.add_argument("--run-dir", default="run")
    ap.add_argument("--out-dir", default="plots")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    run = load(args.run_dir)
    loss_plot(run, out / "loss.png")
    ppl_plot(run, out / "perplexity.png")
    lr_plot(run, out / "learning_rate.png")
    grad_plot(run, out / "gradient_norm.png")
    speed_plot(run, out / "throughput.png")


if __name__ == "__main__":
    main()
