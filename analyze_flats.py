"""
Flat frame noise analysis for DNG sequences.

Usage:
    python analyze_flats.py <seq1_dir> <seq2_dir> <seq3_dir> [--output output_dir]

Each directory should contain DNG flat-field images captured under a single
lighting condition. The script produces:
  - sample_frames.png        : one representative raw frame per sequence
  - mean_frames.png          : per-pixel mean frame per sequence
  - histograms_raw.png       : overlaid pixel-value histograms (raw)
  - histograms_residual.png  : overlaid histograms after mean subtraction
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import rawpy


# --------------------------------------------------------------------------- #
# I/O helpers                                                                   #
# --------------------------------------------------------------------------- #

def find_dngs(directory: str) -> list[Path]:
    p = Path(directory)
    dngs = sorted(p.glob("*.dng")) + sorted(p.glob("*.DNG"))
    if not dngs:
        sys.exit(f"No DNG files found in {directory}")
    return dngs


def load_raw(path: Path) -> np.ndarray:
    """Return the raw Bayer data as float32 (no demosaicing)."""
    with rawpy.imread(str(path)) as raw:
        data = raw.raw_image_visible.astype(np.float32)
    return data


def load_sequence(paths: list[Path], verbose: bool = True) -> np.ndarray:
    """Load all frames and return array of shape (N, H, W)."""
    frames = []
    for i, p in enumerate(paths):
        if verbose:
            print(f"  [{i+1}/{len(paths)}] {p.name}", end="\r", flush=True)
        frames.append(load_raw(p))
    if verbose:
        print()
    return np.stack(frames, axis=0)


# --------------------------------------------------------------------------- #
# Plotting                                                                      #
# --------------------------------------------------------------------------- #

COLORS = ["steelblue", "tomato", "mediumseagreen"]
ALPHA  = 0.55


def plot_sample_frames(samples: list[np.ndarray], labels: list[str], out: Path):
    n = len(samples)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]
    for ax, frame, label in zip(axes, samples, labels):
        vmin, vmax = np.percentile(frame, [0.5, 99.5])
        im = ax.imshow(frame, cmap="gray", vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_title(f"{label}\n(sample frame)", fontsize=11)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    fig.suptitle("Sample raw frames (Bayer, no demosaic)", fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def plot_mean_frames(means: list[np.ndarray], labels: list[str], out: Path):
    n = len(means)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]
    for ax, mean, label in zip(axes, means, labels):
        vmin, vmax = np.percentile(mean, [0.5, 99.5])
        im = ax.imshow(mean, cmap="gray", vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_title(f"{label}\n(mean frame)", fontsize=11)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    fig.suptitle("Per-pixel mean frames", fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def plot_histograms(
    stacks: list[np.ndarray],
    labels: list[str],
    out: Path,
    title: str,
    n_bins: int = 512,
):
    fig, ax = plt.subplots(figsize=(9, 5))
    for stack, label, color in zip(stacks, labels, COLORS):
        flat = stack.ravel()
        ax.hist(
            flat,
            bins=n_bins,
            color=color,
            alpha=ALPHA,
            label=label,
            density=True,
            histtype="stepfilled",
            linewidth=0.8,
            edgecolor=color,
        )
    ax.set_xlabel("Pixel value (ADU)", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, linestyle="--")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved {out}")


# --------------------------------------------------------------------------- #
# Main                                                                          #
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(description="Flat-frame noise analysis")
    parser.add_argument("sequences", nargs=3, metavar="DIR",
                        help="Three directories with DNG flat frames")
    parser.add_argument("--output", default="output", metavar="DIR",
                        help="Output directory for PNGs (default: ./output)")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="Cap number of frames per sequence (for quick tests)")
    parser.add_argument("--bins", type=int, default=512,
                        help="Number of histogram bins (default: 512)")
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    seq_dirs  = args.sequences
    labels    = [Path(d).name for d in seq_dirs]

    # ------------------------------------------------------------------ #
    # 1. Load frames                                                        #
    # ------------------------------------------------------------------ #
    all_stacks: list[np.ndarray] = []
    for i, (d, label) in enumerate(zip(seq_dirs, labels)):
        print(f"\nLoading sequence {i+1}/3: {label}")
        paths = find_dngs(d)
        if args.max_frames:
            paths = paths[: args.max_frames]
        print(f"  {len(paths)} DNG files found")
        stack = load_sequence(paths)
        all_stacks.append(stack)
        print(f"  Shape: {stack.shape}, dtype: {stack.dtype}")

    # ------------------------------------------------------------------ #
    # 2. Sample frames                                                      #
    # ------------------------------------------------------------------ #
    print("\nPlotting sample frames …")
    samples = [s[len(s) // 2] for s in all_stacks]   # middle frame
    plot_sample_frames(samples, labels, out_dir / "sample_frames.png")

    # ------------------------------------------------------------------ #
    # 3. Mean frames                                                        #
    # ------------------------------------------------------------------ #
    print("Computing mean frames …")
    means = [s.mean(axis=0) for s in all_stacks]
    plot_mean_frames(means, labels, out_dir / "mean_frames.png")

    # ------------------------------------------------------------------ #
    # 4. Histograms of raw stacks                                           #
    # ------------------------------------------------------------------ #
    print("Plotting raw histograms …")
    plot_histograms(
        all_stacks, labels,
        out_dir / "histograms_raw.png",
        title="Pixel-value histograms — raw frames (all frames stacked)",
        n_bins=args.bins,
    )

    # ------------------------------------------------------------------ #
    # 5. Histograms after mean subtraction                                  #
    # ------------------------------------------------------------------ #
    print("Plotting residual histograms (mean-subtracted) …")
    residuals = [s - m[np.newaxis, :, :] for s, m in zip(all_stacks, means)]
    plot_histograms(
        residuals, labels,
        out_dir / "histograms_residual.png",
        title="Pixel-value histograms — residuals (frames − mean frame)",
        n_bins=args.bins,
    )

    print(f"\nDone. All plots saved to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
