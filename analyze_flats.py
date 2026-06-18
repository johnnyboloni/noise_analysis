"""
Flat frame noise analysis for DNG sequences.

Edit the CONFIG block below, then run:
    python analyze_flats.py

Outputs (saved to OUTPUT_DIR):
  - sample_frames.png        : one representative RGB frame per sequence
  - mean_frames.png          : per-pixel mean frame (full-res, VNG demosaiced)
  - histograms_adu.png       : raw ADU histograms (before any calibration)
  - histograms_cal.png       : calibrated [0,1] histograms
  - histograms_residual.png  : histograms after mean subtraction
  - spatial_correlation.png  : 2D noise autocorrelation (Bayer domain)
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import rawpy

try:
    import cv2
    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False
    print("Warning: opencv-python not found — mean frames will use half-res channel extraction")


# ============================================================
# CONFIG — edit paths and options here
# ============================================================
seqs_root = '/stage/algo-datasets/DB/DeepISP/MotionCam_RawVideos/S23+_lowlight_uniform/'
SEQUENCE_DIRS = [
    seqs_root + '260518_085011_VIDEO_24mm',
    seqs_root + '260518_085327_VIDEO_24mm',
    seqs_root + '260518_090426_VIDEO_24mm',
]
OUTPUT_DIR          = "output"
MAX_FRAMES          = None  # int to cap frames per sequence, None = load all
HIST_BINS           = 512
RESIDUAL_HIST_RANGE = 0.05  # ± half-range for residual histogram x-axis
AUTOCORR_LAGS       = 32   # display ± this many pixels in autocorrelation plots
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
    Return (bayer_pattern, black_level_per_channel, white_level).
      bayer_pattern          : 2×2 int array, values 0=R 1=G 2=B 3=G2
      black_level_per_channel: float32 array shape (4,)
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
    """Demosaic + white-balance a single DNG via rawpy; returns H×W×3 uint8."""
    with rawpy.imread(str(path)) as raw:
        return raw.postprocess(use_camera_wb=True, no_auto_bright=False, output_bps=8)


def calibrate_frame(frame: np.ndarray, pattern: np.ndarray,
                    black: np.ndarray, white: float) -> np.ndarray:
    """
    Subtract per-channel black level, normalize by (white − black).
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


# --------------------------------------------------------------------------- #
# Demosaicing                                                                   #
# --------------------------------------------------------------------------- #

def _cv2_bayer_code(pattern: np.ndarray) -> int:
    """Map rawpy 2×2 Bayer pattern to cv2 VNG demosaic code."""
    tl = int(pattern[0, 0])   # top-left pixel color: 0=R 1=G 2=B 3=G2
    tr = int(pattern[0, 1])
    if   tl == 0:                    return cv2.COLOR_BayerRG2RGB_VNG  # RGGB
    elif tl == 2:                    return cv2.COLOR_BayerBG2RGB_VNG  # BGGR
    elif tl in (1, 3) and tr == 0:   return cv2.COLOR_BayerGR2RGB_VNG  # GRBG
    else:                            return cv2.COLOR_BayerGB2RGB_VNG  # GBRG


def demosaic_to_rgb(bayer: np.ndarray, pattern: np.ndarray) -> np.ndarray:
    """
    Full-resolution RGB from a 2D float32 calibrated Bayer array.
    Uses cv2 VNG demosaicing if available, otherwise half-res channel extraction.
    Returns float32 H×W×3 in [0, 1], percentile-stretched per channel for display.
    """
    if _HAS_CV2:
        u16 = (np.clip(bayer, 0, 1) * 65535).astype(np.uint16)
        rgb16 = cv2.cvtColor(u16, _cv2_bayer_code(pattern))
        rgb = rgb16.astype(np.float32) / 65535.0
    else:
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
# Streaming passes                                                               #
# --------------------------------------------------------------------------- #

def stream_pass1(
    paths: list[Path],
    pattern: np.ndarray,
    black: np.ndarray,
    white: float,
    n_bins: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Single streaming pass that simultaneously:
      - accumulates the per-pixel calibrated mean
      - builds the raw ADU histogram (before calibration)

    Returns (mean float32, adu_counts int64, adu_edges float64).
    Peak extra memory: one (H, W) float64 accumulator + one float32 frame.
    """
    adu_edges  = np.linspace(0.0, white + 1, n_bins + 1)
    adu_counts = np.zeros(n_bins, dtype=np.int64)
    accum: np.ndarray | None = None

    for i, p in enumerate(paths):
        print(f"  pass 1  [{i+1}/{len(paths)}] {p.name}", end="\r", flush=True)
        raw = load_raw(p)
        h, _ = np.histogram(raw, bins=adu_edges)
        adu_counts += h
        cal = calibrate_frame(raw, pattern, black, white)
        del raw
        if accum is None:
            accum = cal.astype(np.float64)
        else:
            accum += cal
    print()
    mean = (accum / len(paths)).astype(np.float32)
    return mean, adu_counts, adu_edges


def stream_pass2(
    paths: list[Path],
    mean: np.ndarray,
    pattern: np.ndarray,
    black: np.ndarray,
    white: float,
    n_bins: int,
    res_half_range: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, tuple]:
    """
    Single streaming pass that simultaneously:
      - builds the calibrated-value histogram
      - builds the residual histogram (frame − mean)
      - accumulates the FFT power spectrum for autocorrelation

    Returns (cal_counts, cal_edges, res_counts, res_edges, avg_power, frame_shape).
    Peak extra memory: mean frame + one float32 frame + one float64 power spectrum.
    """
    cal_edges   = np.linspace(0.0, 1.0, n_bins + 1)
    res_edges   = np.linspace(-res_half_range, res_half_range, n_bins + 1)
    cal_counts  = np.zeros(n_bins, dtype=np.int64)
    res_counts  = np.zeros(n_bins, dtype=np.int64)
    power_accum: np.ndarray | None = None
    frame_shape: tuple | None = None

    for i, p in enumerate(paths):
        print(f"  pass 2  [{i+1}/{len(paths)}] {p.name}", end="\r", flush=True)
        cal = calibrate_frame(load_raw(p), pattern, black, white)

        h, _ = np.histogram(cal, bins=cal_edges)
        cal_counts += h

        residual = cal - mean
        h, _ = np.histogram(residual, bins=res_edges)
        res_counts += h

        # Subtract frame mean to ensure zero DC before FFT
        residual -= residual.mean()
        fft_power = np.abs(np.fft.rfft2(residual)) ** 2
        if power_accum is None:
            power_accum = fft_power.astype(np.float64)
            frame_shape = residual.shape
        else:
            power_accum += fft_power
    print()
    avg_power = power_accum / len(paths)
    return cal_counts, cal_edges, res_counts, res_edges, avg_power, frame_shape


def compute_autocorr(avg_power: np.ndarray, frame_shape: tuple,
                     lags: int) -> np.ndarray:
    """
    Compute normalized 2D spatial autocorrelation from averaged power spectrum.
    Returns a (2*lags+1) × (2*lags+1) array, value = 1 at zero lag.
    The zero-lag pixel is set to NaN so the colorbar is scaled by off-center values.
    """
    full = np.fft.irfft2(avg_power, s=frame_shape)   # circular autocorrelation
    centered = np.fft.fftshift(full)                  # zero lag at centre
    H, W = centered.shape
    cy, cx = H // 2, W // 2
    norm = centered[cy, cx]
    normalized = centered / max(norm, 1e-12)
    crop = normalized[cy - lags : cy + lags + 1,
                      cx - lags : cx + lags + 1].copy()
    mid = lags   # center of crop
    crop[mid, mid] = np.nan   # hide trivial self-correlation peak
    return crop


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
    fig.suptitle("Sample frames (rawpy demosaiced RGB)", fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def plot_mean_frames(rgb_means: list[np.ndarray], labels: list[str], out: Path):
    method = "VNG demosaic" if _HAS_CV2 else "half-res channel extraction"
    n = len(rgb_means)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]
    for ax, rgb, label in zip(axes, rgb_means, labels):
        ax.imshow(rgb, aspect="auto")
        ax.set_title(f"{label}\n(mean frame)", fontsize=11)
        ax.axis("off")
    fig.suptitle(f"Per-pixel mean frames ({method}, per-channel stretched)",
                 fontsize=13, y=1.01)
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


def plot_autocorrelations(
    autocorrs: list[np.ndarray],
    labels: list[str],
    out: Path,
    lags: int,
):
    n = len(autocorrs)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    if n == 1:
        axes = [axes]

    # Scale colorbar to the largest off-center absolute value across all sequences
    vmax = max(np.nanmax(np.abs(a)) for a in autocorrs)
    vmax = max(vmax, 1e-6)

    extent = [-lags - 0.5, lags + 0.5, -lags - 0.5, lags + 0.5]
    for ax, autocorr, label in zip(axes, autocorrs, labels):
        im = ax.imshow(autocorr, extent=extent, cmap="RdBu_r",
                       vmin=-vmax, vmax=vmax, aspect="equal",
                       origin="lower", interpolation="nearest")
        ax.set_title(label, fontsize=10)
        ax.set_xlabel("lag x (px)", fontsize=9)
        ax.set_ylabel("lag y (px)", fontsize=9)
        ax.axhline(0, color="k", linewidth=0.5, alpha=0.4)
        ax.axvline(0, color="k", linewidth=0.5, alpha=0.4)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(
        "Spatial noise autocorrelation — Bayer domain (lag-0 hidden, normalized to variance)",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
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
    adu_hist_data:  list[tuple[np.ndarray, np.ndarray]] = []
    cal_hist_data:  list[tuple[np.ndarray, np.ndarray]] = []
    res_hist_data:  list[tuple[np.ndarray, np.ndarray]] = []
    autocorrs:      list[np.ndarray] = []

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

        # Pass 1: mean + ADU histogram
        print(f"  Pass 1 — mean + ADU histogram …")
        mean, adu_counts, adu_edges = stream_pass1(
            paths, pattern, black, white, HIST_BINS,
        )
        all_means.append(mean)
        adu_hist_data.append((adu_counts, adu_edges))
        print(f"  Mean calibrated range: [{mean.min():.4f}, {mean.max():.4f}]")

        # Pass 2: calibrated histogram + residual histogram + autocorrelation
        print(f"  Pass 2 — calibrated hist + residuals + autocorrelation …")
        cc, ce, rc, re, avg_power, fshape = stream_pass2(
            paths, mean, pattern, black, white,
            HIST_BINS, RESIDUAL_HIST_RANGE,
        )
        cal_hist_data.append((cc, ce))
        res_hist_data.append((rc, re))
        autocorrs.append(compute_autocorr(avg_power, fshape, AUTOCORR_LAGS))

    # ------------------------------------------------------------------ #
    # Plots                                                                 #
    # ------------------------------------------------------------------ #
    print("\nPlotting sample frames …")
    plot_sample_frames(all_sample_rgb, labels, out_dir / "sample_frames.png")

    print("Plotting mean frames …")
    mean_rgb = [demosaic_to_rgb(m, p) for m, p in zip(all_means, all_patterns)]
    plot_mean_frames(mean_rgb, labels, out_dir / "mean_frames.png")

    print("Plotting ADU histograms …")
    plot_histograms(
        adu_hist_data, labels,
        out_dir / "histograms_adu.png",
        title="Pixel-value histograms — raw ADU (before calibration)",
        xlabel="Raw pixel value (ADU)",
    )

    print("Plotting calibrated histograms …")
    plot_histograms(
        cal_hist_data, labels,
        out_dir / "histograms_cal.png",
        title="Pixel-value histograms — calibrated frames",
        xlabel="Calibrated value  (ADU − black) / (white − black)",
    )

    print("Plotting residual histograms …")
    plot_histograms(
        res_hist_data, labels,
        out_dir / "histograms_residual.png",
        title="Pixel-value histograms — residuals (frames − mean frame)",
        xlabel=f"Residual  [±{RESIDUAL_HIST_RANGE}]",
    )

    print("Plotting spatial autocorrelations …")
    plot_autocorrelations(
        autocorrs, labels,
        out_dir / "spatial_correlation.png",
        lags=AUTOCORR_LAGS,
    )

    print(f"\nDone. All plots saved to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
