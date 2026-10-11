#!/usr/bin/env python3
"""
Plot summary and distribution charts from a single pytest-benchmark JSON file
(the file produced by `--benchmark-json=...`).

Usage:
    python plot_benchmark_results.py knn3dtest.json
    python plot_benchmark_results.py knn3dtest.json --outdir plots/
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "figure.dpi": 150,
})

PALETTE = plt.cm.tab10.colors


def load_benchmarks(json_path):
    with open(json_path) as f:
        data = json.load(f)
    return data["benchmarks"]


def short_name(full_name):
    # "test_compute_TracerAuto3D" -> "TracerAuto3D"
    return full_name.replace("test_compute_", "")


def build_color_map(benchmarks):
    names = sorted({short_name(b["name"]) for b in benchmarks})
    return {name: PALETTE[i % len(PALETTE)] for i, name in enumerate(names)}


def style_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.set_axisbelow(True)
    # ax.grid(axis="x", which="major", alpha=0.35, linewidth=0.8)
    # ax.grid(axis="x", which="minor", alpha=0.15, linewidth=0.5)
    ax.tick_params(axis="y", length=0)


def print_table(benchmarks):
    header = (f"{'function':32s} {'mean (s)':>12s} {'stddev':>12s} "
              f"{'min':>12s} {'max':>12s} {'rounds':>8s}")
    print(header)
    print("-" * len(header))
    for b in sorted(benchmarks, key=lambda b: b["stats"]["mean"]):
        s = b["stats"]
        print(f"{short_name(b['name']):32s} {s['mean']:12.4e} {s['stddev']:12.4e} "
              f"{s['min']:12.4e} {s['max']:12.4e} {s['rounds']:8d}")


def plot_summary_bar(benchmarks, color_map, outpath):
    ordered = sorted(benchmarks, key=lambda b: b["stats"]["mean"])
    names = [short_name(b["name"]) for b in ordered]
    means = [b["stats"]["mean"] for b in ordered]
    stddevs = [b["stats"]["stddev"] for b in ordered]
    colors = [color_map[n] for n in names]

    fig, ax = plt.subplots(figsize=(9, 0.7 * len(names) + 1.5))
    y = np.arange(len(names))
    ax.barh(y, means, xerr=stddevs, capsize=4, color=colors,
            edgecolor="black", linewidth=0.6,
            error_kw={"elinewidth": 1.2, "ecolor": "black"})
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.set_xlabel("Mean time per call (s)")
    ax.set_xscale("log")
    ax.set_title("Benchmark mean runtime (\u00b1 stddev)", pad=12)
    style_axes(ax)

    # annotate each bar with its value so exact numbers don't need a legend lookup
    x_max = max(m + s for m, s in zip(means, stddevs))
    for yi, m in zip(y, means):
        ax.text(m * 1.15, yi, f"{m:.3g} s", va="center", fontsize=9, color="dimgray")
    ax.set_xlim(right=x_max * 3)  # headroom for the annotations

    fig.tight_layout()
    fig.savefig(outpath)
    print(f"wrote {outpath}")
    plt.close(fig)


def plot_distributions(benchmarks, color_map, outpath):
    have_data = [b for b in benchmarks if b["stats"].get("data")]
    if not have_data:
        print("no per-round 'data' arrays found in JSON; skipping distribution plot")
        return

    have_data = sorted(have_data, key=lambda b: b["stats"]["mean"])
    names = [short_name(b["name"]) for b in have_data]
    datasets = [np.asarray(b["stats"]["data"]) for b in have_data]
    n_rounds = have_data[0]["stats"]["rounds"]

    fig, ax = plt.subplots(figsize=(9, 0.8 * len(names) + 1.5))
    rng = np.random.default_rng(0)

    for i, (data, name) in enumerate(zip(datasets, names)):
        color = color_map[name]
        y_jitter = i + rng.uniform(-0.12, 0.12, size=len(data))
        ax.scatter(data, y_jitter, color=color, edgecolor="black", linewidth=0.4,
                   s=45, alpha=0.9, zorder=3, label="_nolegend_")
        mean, std = data.mean(), data.std()
        ax.errorbar([mean], [i], xerr=[std], fmt="D", color="black",
                    markersize=6, capsize=4, elinewidth=1.3, zorder=4)

    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    ax.set_ylim(-0.6, len(names) - 0.4)
    ax.set_xlabel("Time per call (s)")
    ax.set_xscale("log")
    ax.set_title(
        f"Per-round timings ({n_rounds} rounds/function)\n"
        "dots = individual rounds, \u25c6 = mean \u00b1 stddev",
        fontsize=12, pad=12
    )
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(outpath)
    print(f"wrote {outpath}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("json_path", help="pytest-benchmark JSON file (from --benchmark-json)")
    parser.add_argument("--outdir", default=".", help="directory to write PNGs to")
    args = parser.parse_args()

    benchmarks = load_benchmarks(args.json_path)
    if not benchmarks:
        sys.exit("no benchmarks found in JSON file")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    color_map = build_color_map(benchmarks)

    print_table(benchmarks)
    plot_summary_bar(benchmarks, color_map, outdir / "benchmark_summary.png")
    plot_distributions(benchmarks, color_map, outdir / "benchmark_distributions.png")


if __name__ == "__main__":
    main()
