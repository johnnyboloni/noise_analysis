"""
Flat frame noise analysis for DNG sequences.

Edit the CONFIG block below, then run:
    python analyze_flats.py

Outputs (saved to OUTPUT_DIR):
  - sample_frames.png        : one representative RGB frame per sequence
  - mean_frames.png          : per-pixel mean frame per sequence (RGB, half-res)
  - histograms_raw.png       : overlaid pixel-value histograms (calibrated [0,1])
  - histograms_residual.png  : overlaid histograms after mean subtraction
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import rawpy


# ============================================================
# CONFIG — edit paths and options here
# ============================================================
SEQUENCE_DIRS = [
    "/path/to/seq1",
    "/path/to/seq2",
    "/path/to/seq3",
]
OUTPUT_DIR = "output"
MAX_FRAMES = None   # int to cap frames per sequence, None = load all
HIST_BINS  = 512
# ============================================================


# --------------------------------------------------------------------------- #
# I/O helpers                                                                   #
# --------------------------------------------------------------------------- #

def find_dngs(directory: str) -> list[Path]:
    p = Path(directory)
    dngs = sorted(p.glob("*.dng")) + sorted(p.glob("*.DNG"))
    if not dngs:
        sys.exit(f"No DNG files found in {directory}")
    return dngs


def get_raw_metadata(path: Path) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Return (bayer_pattern, black_level_per_channel, white_level) from a DNG.
      bayer_pattern          : 2×2 int array, values 0=R 1=G 2=B 3=G2
      black_level_per_channel: float32 array shape (4,), indexed by pattern value
      white_level            : scalar float
    """
    with rawpy.imread(str(path)) as raw:
        pattern = raw.raw_pattern.copy()
        black   = np.array(raw.black_level_per_channel, dtype=np.float32)
        white   = float(raw.white_level)
    return pattern, black, white


def load_raw(path: Path) -> np.ndarray:
    """Return the raw Bayer data as float32 (ADU, no demosaicing)."""
    with rawpy.imread(str(path)) as raw:
        return raw.raw_image_visible.astype(np.float32)


def load_raw_rgb(path: Path) -> np.ndarray:
    """Demosaic + white-balance a single DNG; returns H×W×3 uint8."""
    with rawpy.imread(str(path)) as raw:
        return raw.postprocess(use_camera_wb=True, no_auto_bright=False, output_bps=8)


def load_sequence(paths: list[Path], verbose: bool = True) -> np.ndarray:
    """Load all frames as raw Bayer float32; returns (N, H, W)."""
    frames = []
    for i, p in enumerate(paths):
        if verbose:
            print(f"  [{i+1}/{len(paths)}] {p.name}", end="\r", flush=True)
        frames.append(load_raw(p))
    if verbose:
        print()
    return np.stack(frames, axis=0)


def calibrate(stack: np.ndarray, pattern: np.ndarray,
              black: np.ndarray, white: float) -> np.ndarray:
    """
    Subtract per-channel black level and normalize by (white - black).
    Each Bayer channel gets its own black level value.
    Returns float32 array nominally in [0, 1]; clipped to that range.
    """
    out = np.empty_like(stack, dtype=np.float32)
    for r in range(2):
        for c in range(2):
            ch = int(pattern[r, c])
            bl = black[ch]
            out[:, r::2, c::2] = (stack[:, r::2, c::2] - bl) / (white - bl)
    return np.clip(out, 0.0, 1.0)


def bayer_to_rgb(bayer: np.ndarray, pattern: np.ndarray) -> np.ndarray:
    """
    Half-resolution RGB from a 2-D float32 Bayer array.
    Input is expected to already be calibrated (values ~[0, 1]).
    Each channel is percentile-stretched for display.
    """
    channels = {int(pattern[r, c]): bayer[r::2, c::2]
                for r in range(2) for c in range(2)}
    R = channels[0].astype(np.float32)
    G = ((channels[1] + channels[3]) / 2).astype(np.float32)
    B = channels[2].astype(np.float32)
    rgb = np.stack([R, G, B], axis=-1)
    for i in range(3):
        lo, hi = np.percentile(rgb[..., i], [0.5, 99.5])
        rgb[..., i] = np.clip((rgb[..., i] - lo) / max(hi - lo, 1e-6), 0, 1)
    return rgb


# --------------------------------------------------------------------------- #
# Plotting                                                                      #
# --------------------------------------------------------------------------- #

COLORS = ["steelblue", "tomato", "mediumseagreen"]
ALPHA  = 0.55


def plot_sample_frames(rgb_frames: list[np.ndarray], labels: list[str], out: Path):
    n = len(rgb_frames)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]
    for ax, rgb, label in zip(axes, rgb_frames, labels):
        ax.imshow(rgb, aspect="auto")
        ax.set_title(f"{label}\n(sample frame)", fontsize=11)
        ax.axis("off")
    fig.suptitle("Sample frames (demosaiced RGB)", fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def plot_mean_frames(rgb_means: list[np.ndarray], labels: list[str], out: Path):
    n = len(rgb_means)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]
    for ax, rgb, label in zip(axes, rgb_means, labels):
        ax.imshow(rgb, aspect="auto")
        ax.set_title(f"{label}\n(mean frame)", fontsize=11)
        ax.axis("off")
    fig.suptitle("Per-pixel mean frames (half-res RGB, per-channel stretched)", fontsize=13, y=1.01)
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
        ax.hist(
            stack.ravel(),
            bins=n_bins,
            color=color,
            alpha=ALPHA,
            label=label,
            density=True,
            histtype="stepfilled",
            linewidth=0.8,
            edgecolor=color,
        )
    ax.set_xlabel("Calibrated pixel value  (ADU − black) / (white − black)", fontsize=10)
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
    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    labels = [Path(d).name for d in SEQUENCE_DIRS]

    # ------------------------------------------------------------------ #
    # 1. Load + calibrate frames                                            #
    # ------------------------------------------------------------------ #
    all_stacks:     list[np.ndarray] = []
    all_patterns:   list[np.ndarray] = []
    all_sample_rgb: list[np.ndarray] = []

    for i, (d, label) in enumerate(zip(SEQUENCE_DIRS, labels)):
        print(f"\nLoading sequence {i+1}/{len(SEQUENCE_DIRS)}: {label}")
        paths = find_dngs(d)
        if MAX_FRAMES:
            paths = paths[:MAX_FRAMES]
        print(f"  {len(paths)} DNG files found")

        pattern, black, white = get_raw_metadata(paths[0])
        all_patterns.append(pattern)
        print(f"  Black levels (R,G,G,B): {black}  |  White level: {white}")

        mid = len(paths) // 2
        print(f"  Loading sample frame (index {mid}) as RGB …")
        all_sample_rgb.append(load_raw_rgb(paths[mid]))

        print(f"  Loading {len(paths)} raw frames …")
        raw_stack = load_sequence(paths)
        cal_stack = calibrate(raw_stack, pattern, black, white)
        all_stacks.append(cal_stack)
        print(f"  Stack shape: {cal_stack.shape}  value range: "
              f"[{cal_stack.min():.4f}, {cal_stack.max():.4f}]")

    # ------------------------------------------------------------------ #
    # 2. Sample frames (RGB via rawpy postprocess)                          #
    # ------------------------------------------------------------------ #
    print("\nPlotting sample frames …")
    plot_sample_frames(all_sample_rgb, labels, out_dir / "sample_frames.png")

    # ------------------------------------------------------------------ #
    # 3. Mean frames (Bayer mean → half-res RGB)                            #
    # ------------------------------------------------------------------ #
    print("Computing mean frames …")
    means    = [s.mean(axis=0) for s in all_stacks]
    mean_rgb = [bayer_to_rgb(m, p) for m, p in zip(means, all_patterns)]
    plot_mean_frames(mean_rgb, labels, out_dir / "mean_frames.png")

    # ------------------------------------------------------------------ #
    # 4. Histograms of calibrated stacks                                    #
    # ------------------------------------------------------------------ #
    print("Plotting calibrated histograms …")
    plot_histograms(
        all_stacks, labels,
        out_dir / "histograms_raw.png",
        title="Pixel-value histograms — calibrated frames (all frames)",
        n_bins=HIST_BINS,
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
        n_bins=HIST_BINS,
    )

    print(f"\nDone. All plots saved to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
