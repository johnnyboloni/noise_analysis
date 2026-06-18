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
seqs_root = '/stage/algo-datasets/DB/DeepISP/MotionCam_RawVideos/S23+_lowlight_uniform/'
SEQUENCE_DIRS = [
    seqs_root + '260518_085011_VIDEO_24mm',
    seqs_root + '260518_085327_VIDEO_24mm',
    seqs_root + '260518_090426_VIDEO_24mm',
]
OUTPUT_DIR = "output"
MAX_FRAMES = None   # int to cap frames per sequence, None = load all
HIST_BINS  = 512
RESIDUAL_HIST_RANGE = 0.05   # histogram x-axis spans ±this value after mean sub
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


def calibrate_frame(frame: np.ndarray, pattern: np.ndarray,
                    black: np.ndarray, white: float) -> np.ndarray:
    """
    Subtract per-channel black level and normalize by (white - black).
    Input : (H, W) float32 Bayer frame in ADU.
    Output: (H, W) float32, nominally [0, 1], clipped.
    """
    out = np.empty_like(frame, dtype=np.float32)
    for r in range(2):
        for c in range(2):
            ch = int(pattern[r, c])
            bl = black[ch]
            out[r::2, c::2] = (frame[r::2, c::2] - bl) / (white - bl)
    return np.clip(out, 0.0, 1.0)


def stream_mean(paths: list[Path], pattern: np.ndarray,
                black: np.ndarray, white: float) -> np.ndarray:
    """
    Pass 1 — load and calibrate frames one at a time, return per-pixel mean.
    Peak memory: two (H, W) float32 arrays (accumulator + current frame).
    """
    accum: np.ndarray | None = None
    for i, p in enumerate(paths):
        print(f"  pass 1  [{i+1}/{len(paths)}] {p.name}", end="\r", flush=True)
        frame = calibrate_frame(load_raw(p), pattern, black, white)
        if accum is None:
            accum = frame.astype(np.float64)
        else:
            accum += frame
    print()
    return (accum / len(paths)).astype(np.float32)


def stream_histograms(
    paths: list[Path],
    mean: np.ndarray,
    pattern: np.ndarray,
    black: np.ndarray,
    white: float,
    n_bins: int,
    residual_half_range: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Pass 2 — accumulate raw and residual histogram counts simultaneously.
    Returns (raw_counts, raw_edges, res_counts, res_edges).
    Peak memory: two (H, W) float32 arrays (mean + current frame).
    """
    raw_edges = np.linspace(0.0, 1.0, n_bins + 1)
    res_edges = np.linspace(-residual_half_range, residual_half_range, n_bins + 1)
    raw_counts = np.zeros(n_bins, dtype=np.int64)
    res_counts = np.zeros(n_bins, dtype=np.int64)

    for i, p in enumerate(paths):
        print(f"  pass 2  [{i+1}/{len(paths)}] {p.name}", end="\r", flush=True)
        frame = calibrate_frame(load_raw(p), pattern, black, white)
        h, _ = np.histogram(frame, bins=raw_edges)
        raw_counts += h
        h, _ = np.histogram(frame - mean, bins=res_edges)
        res_counts += h
    print()
    return raw_counts, raw_edges, res_counts, res_edges


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
    hist_data: list[tuple[np.ndarray, np.ndarray]],
    labels: list[str],
    out: Path,
    title: str,
    xlabel: str,
):
    fig, ax = plt.subplots(figsize=(9, 5))
    for (counts, edges), label, color in zip(hist_data, labels, COLORS):
        width = edges[1] - edges[0]
        density = counts / (counts.sum() * width)
        ax.stairs(density, edges, fill=True, color=color, alpha=ALPHA,
                  label=label, linewidth=0.8)
    ax.set_xlabel(xlabel, fontsize=10)
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

    all_means:      list[np.ndarray] = []
    all_patterns:   list[np.ndarray] = []
    all_sample_rgb: list[np.ndarray] = []
    raw_hist_data:  list[tuple[np.ndarray, np.ndarray]] = []
    res_hist_data:  list[tuple[np.ndarray, np.ndarray]] = []

    for i, (d, label) in enumerate(zip(SEQUENCE_DIRS, labels)):
        print(f"\nSequence {i+1}/{len(SEQUENCE_DIRS)}: {label}")
        paths = find_dngs(d)
        if MAX_FRAMES:
            paths = paths[:MAX_FRAMES]
        print(f"  {len(paths)} DNG files")

        pattern, black, white = get_raw_metadata(paths[0])
        all_patterns.append(pattern)
        print(f"  Black levels (R,G,G,B): {black}  |  White level: {white}")

        mid = len(paths) // 2
        print(f"  Loading sample frame (index {mid}) as RGB …")
        all_sample_rgb.append(load_raw_rgb(paths[mid]))

        # ---- Pass 1: compute mean (one frame in memory at a time) ----------
        print(f"  Pass 1/{len(paths)} frames — accumulating mean …")
        mean = stream_mean(paths, pattern, black, white)
        all_means.append(mean)
        print(f"  Mean range: [{mean.min():.4f}, {mean.max():.4f}]")

        # ---- Pass 2: accumulate raw + residual histograms -------------------
        print(f"  Pass 2/{len(paths)} frames — accumulating histograms …")
        rc, re, sc, se = stream_histograms(
            paths, mean, pattern, black, white,
            HIST_BINS, RESIDUAL_HIST_RANGE,
        )
        raw_hist_data.append((rc, re))
        res_hist_data.append((sc, se))

    # ------------------------------------------------------------------ #
    # Plots                                                                 #
    # ------------------------------------------------------------------ #
    print("\nPlotting sample frames …")
    plot_sample_frames(all_sample_rgb, labels, out_dir / "sample_frames.png")

    print("Plotting mean frames …")
    mean_rgb = [bayer_to_rgb(m, p) for m, p in zip(all_means, all_patterns)]
    plot_mean_frames(mean_rgb, labels, out_dir / "mean_frames.png")

    print("Plotting calibrated histograms …")
    plot_histograms(
        raw_hist_data, labels,
        out_dir / "histograms_raw.png",
        title="Pixel-value histograms — calibrated frames (all frames)",
        xlabel="Calibrated value  (ADU − black) / (white − black)",
    )

    print("Plotting residual histograms …")
    plot_histograms(
        res_hist_data, labels,
        out_dir / "histograms_residual.png",
        title="Pixel-value histograms — residuals (frames − mean frame)",
        xlabel=f"Residual  [range ±{RESIDUAL_HIST_RANGE}]",
    )

    print(f"\nDone. All plots saved to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
