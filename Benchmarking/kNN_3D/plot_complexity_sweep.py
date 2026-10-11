#!/usr/bin/env python3
"""
Combine several pytest-benchmark JSON files -- each produced with a different
value of some size parameter (e.g. --n_tracer or --NGRID) -- into a single
runtime-vs-problem-size plot, with a fitted power-law exponent to characterise
the empirical time complexity of each function.

This relies on each benchmark's `extra_info` dict containing the swept
parameter (test_kNN_3D.py already records n_tracer, n_query, NGRID, n_reals,
etc. via `benchmark.extra_info[...] = ...`). If a JSON file has no matching
extra_info key, its numeric value is instead parsed out of the filename.

Usage:
    # after generating e.g. knn3d_n2000.json, knn3d_n5000.json, ... via a
    # shell loop that varies --n_tracer:
    python plot_complexity_sweep.py "knn3d_n*.json" --param n_tracer

    # or list files explicitly:
    python plot_complexity_sweep.py knn3d_n2000.json knn3d_n5000.json \\
        knn3d_n10000.json --param n_tracer
"""
import argparse
import glob
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def short_name(full_name):
    return full_name.replace("test_compute_", "")


def get_param_value(benchmark, json_path, param):
    extra = benchmark.get("extra_info", {}) or {}
    if param in extra:
        return extra[param]
    # Fall back: pull the LAST run of digits out of the filename, since the
    # swept value is conventionally the trailing number (e.g. "knn3d_n2000"
    # -> 2000). Taking the first run of digits is wrong here -- stems like
    # "kNN_3D"/"knn3d" contain a digit of their own ("3") that would
    # otherwise be matched before the real swept value.
    matches = re.findall(r"\d+", Path(json_path).stem)
    if matches:
        return int(matches[-1])
    return None


def load_all(paths, param):
    # name -> list of (size, mean, stddev)
    series = defaultdict(list)
    for p in paths:
        with open(p) as f:
            data = json.load(f)
        for b in data["benchmarks"]:
            size = get_param_value(b, p, param)
            if size is None:
                print(f"warning: could not determine '{param}' for "
                      f"{b['name']} in {p}; skipping", file=sys.stderr)
                continue
            series[short_name(b["name"])].append(
                (size, b["stats"]["mean"], b["stats"]["stddev"])
            )
    for name in series:
        series[name].sort(key=lambda t: t[0])
    return series


def fit_power_law(sizes, means):
    """Fit log(mean) = slope * log(size) + intercept. Returns (slope, intercept)."""
    log_x = np.log(sizes)
    log_y = np.log(means)
    slope, intercept = np.polyfit(log_x, log_y, 1)
    return slope, intercept


def plot_complexity(series, param, outpath):
    fig, ax = plt.subplots(figsize=(8, 6))

    for name, points in series.items():
        if len(points) < 2:
            print(f"note: only one data point for {name}; skipping fit/plot for it")
            continue
        sizes = np.array([p[0] for p in points], dtype=float)
        means = np.array([p[1] for p in points], dtype=float)
        stddevs = np.array([p[2] for p in points], dtype=float)

        slope, intercept = fit_power_law(sizes, means)
        label = f"{name}  (\u221d {param}^{slope:.2f})"

        line, = ax.plot(sizes, means, "o-", label=label)
        ax.errorbar(sizes, means, yerr=stddevs, fmt="none",
                    ecolor=line.get_color(), capsize=3, alpha=0.6)

        # overlay the fitted power-law line across the sampled range
        fit_x = np.geomspace(sizes.min(), sizes.max(), 50)
        fit_y = np.exp(intercept) * fit_x ** slope
        ax.plot(fit_x, fit_y, "--", color=line.get_color(), alpha=0.4)

        print(f"{name}: empirical scaling exponent \u2248 {slope:.3f} "
              f"(time \u221d {param}^{slope:.3f})")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(param)
    ax.set_ylabel("mean time per call (s)")
    ax.set_title(f"Runtime vs. {param} (log\u2013log, dashed = power-law fit)")
    #ax.grid(which="both", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    print(f"wrote {outpath}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("paths", nargs="+",
                         help="JSON files or glob pattern(s) from a parameter sweep")
    parser.add_argument("--param", default="n_tracer",
                         help="extra_info key (or filename-embedded value) to plot "
                              "against, e.g. n_tracer, NGRID, n_reals (default: n_tracer)")
    parser.add_argument("--outdir", default=".", help="directory to write the PNG to")
    args = parser.parse_args()

    # expand any glob patterns the shell didn't already expand
    paths = []
    for p in args.paths:
        matches = glob.glob(p)
        paths.extend(matches if matches else [p])
    paths = sorted(set(paths))

    if not paths:
        sys.exit("no JSON files matched")

    series = load_all(paths, args.param)
    if not series:
        sys.exit("no usable data found across the given files")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    plot_complexity(series, args.param, outdir / f"complexity_vs_{args.param}.png")


if __name__ == "__main__":
    main()
