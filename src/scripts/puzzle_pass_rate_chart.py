#!/usr/bin/env python3
"""Evaluate puzzle pass rates binned by puzzle rating and plot results."""

import csv
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

PUZZLE_CSV = Path(
    "/home/rgodha/Developer/knightfall/training/data/lichess_puzzle_transformed.csv"
)
STOCKFISH_DIR = Path("/home/rgodha/Developer/stockfish-trainer/stockfish")

ENGINES = {
    "LayerStacks": {
        "bin": STOCKFISH_DIR / "stockfish-layerstacks-1-local",
        "evalfile": "/mnt/external/models/layerstacks-1/epoch_0089-fixed.nnue",
    },
    "MOE": {
        "bin": STOCKFISH_DIR / "stockfish-moe-fixed-local",
        "evalfile": "/mnt/external/models/moe-fr/final-fixed.nnue",
    },
    "NonStacks": {
        "bin": STOCKFISH_DIR / "stockfish-nonstacks-local",
        "evalfile": "/mnt/external/models/nonstacks/final.nnue",
    },
}

SMALL_NET = STOCKFISH_DIR / "nn-47fc8b7fff06.nnue"
NUM_PUZZLES = 10000
BIN_SIZE = 200


def spawn_engine(name: str, bin_path: Path, evalfile: str):
    proc = subprocess.Popen(
        [str(bin_path)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        cwd=STOCKFISH_DIR,
    )
    assert proc.stdin is not None and proc.stdout is not None
    proc.stdin.write(f"uci\nsetoption name EvalFile value {evalfile}\n")
    proc.stdin.write(f"setoption name EvalFileSmall value {SMALL_NET}\n")
    proc.stdin.write("isready\n")
    proc.stdin.flush()
    while True:
        line = proc.stdout.readline()
        if line.startswith("readyok"):
            break
    return proc


def get_bestmove(proc: subprocess.Popen, fen: str, depth: int) -> str:
    proc.stdin.write(f"position fen {fen}\ngo depth {depth}\n")
    proc.stdin.flush()
    bestmove = None
    while True:
        line = proc.stdout.readline()
        if line.startswith("bestmove"):
            parts = line.split()
            if len(parts) >= 2:
                bestmove = parts[1]
            break
    return bestmove


def main():
    # load puzzles
    puzzles = []
    with open(PUZZLE_CSV, newline="") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i >= NUM_PUZZLES:
                break
            fen = row["FEN"]
            moves = row["Moves"].split()
            rating = int(row["Rating"])
            puzzles.append((row["PuzzleId"], fen, moves[0], rating))

    # spawn engines
    procs = {}
    for name, cfg in ENGINES.items():
        print(f"Spawning {name}...", file=sys.stderr)
        procs[name] = spawn_engine(name, cfg["bin"], cfg["evalfile"])

    # per-puzzle results: {(name, depth): [(rating, correct), ...]}
    raw_results = defaultdict(list)

    for idx, (pid, fen, expected, rating) in enumerate(puzzles, 1):
        if idx % 100 == 0 or idx == 1:
            print(f"Puzzle {idx}/{NUM_PUZZLES}", file=sys.stderr)
        for name, proc in procs.items():
            for d in (1, 3, 10):
                try:
                    best = get_bestmove(proc, fen, d)
                    correct = best == expected
                    raw_results[(name, d)].append((rating, correct))
                except Exception as e:
                    print(f"Error {name} {pid} d={d}: {e}", file=sys.stderr)

    # quit engines
    for proc in procs.values():
        proc.stdin.write("quit\n")
        proc.stdin.flush()
        try:
            proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            proc.kill()

    # bin by rating
    all_ratings = [r for _, _, _, r in puzzles]
    min_r = min(all_ratings)
    max_r = max(all_ratings)
    bins = list(range((min_r // BIN_SIZE) * BIN_SIZE, max_r + BIN_SIZE, BIN_SIZE))
    bin_centers = [(bins[i] + bins[i + 1]) / 2 for i in range(len(bins) - 1)]

    def bin_results(results):
        counts = defaultdict(lambda: [0, 0])  # [correct, total]
        for rating, correct in results:
            for i in range(len(bins) - 1):
                if bins[i] <= rating < bins[i + 1]:
                    counts[i][1] += 1
                    if correct:
                        counts[i][0] += 1
                    break
        rates = []
        totals = []
        for i in range(len(bins) - 1):
            c, t = counts[i]
            rates.append(c / t * 100 if t > 0 else 0)
            totals.append(t)
        return rates, totals

    # plot
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    palette = sns.color_palette("tab10", n_colors=3)
    colors = {"LayerStacks": palette[0], "MOE": palette[1], "NonStacks": palette[2]}

    for ax, depth in zip(axes, (1, 3, 10)):
        # Use NonStacks totals for sample size annotations
        _, totals = bin_results(raw_results[("NonStacks", depth)])
        for name in ENGINES:
            rates, _ = bin_results(raw_results[(name, depth)])
            # Filter out bins with < 10 puzzles to avoid noisy tail
            filtered_x = [x for x, t in zip(bin_centers, totals) if t >= 10]
            filtered_y = [y for y, t in zip(rates, totals) if t >= 10]
            ax.plot(filtered_x, filtered_y, marker="o", label=name, color=colors[name])
        # annotate sample sizes
        for x, t in zip(bin_centers, totals):
            if t >= 10:
                ax.annotate(str(t), xy=(x, 2), fontsize=7, ha="center", color="gray")
        ax.set_xlabel("Puzzle Rating (Elo)")
        ax.set_ylabel("Pass Rate (%)")
        ax.set_title(f"Depth = {depth}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 105)

    plt.tight_layout()
    out = "/tmp/puzzle_pass_rate_by_rating.png"
    plt.savefig(out, dpi=150)
    print(f"Saved plot to {out}")

    # also save raw data
    data_out = "/tmp/puzzle_pass_rate_by_rating.json"
    with open(data_out, "w") as f:
        json.dump(
            {
                "bins": bins,
                "bin_centers": bin_centers,
                "results": {
                    f"{name}_d{depth}": bin_results(raw_results[(name, depth)])[0]
                    for name in ENGINES
                    for depth in (1, 3, 10)
                },
            },
            f,
            indent=2,
        )
    print(f"Saved data to {data_out}")


if __name__ == "__main__":
    main()
