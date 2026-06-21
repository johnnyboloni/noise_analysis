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
  - variance_vs_mean.png     : per-pixel temporal variance vs mean (Poisson test)
  - temporal_correlation.png : temporal noise autocorrelation function
"""

import sys
import hashlib
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

# GN3v4 note: sequence names ending in '1' are 10-bit; ending in '0' are 12-bit.
# white_level is read per-file from DNG metadata so calibration adapts automatically.
gn3_root = '/stage/algo-datasets/DB/LME/raw/20250126_GN3v4_train'

SEQUENCE_DIRS = [
    seqs_root + '260518_085011_VIDEO_24mm',
    seqs_root + '260518_085327_VIDEO_24mm',
    seqs_root + '260518_090426_VIDEO_24mm',
    gn3_root,
]
OUTPUT_DIR          = "output"
MAX_FRAMES          = None   # int to cap frames per sequence, None = load all
HIST_BINS           = 512
RESIDUAL_HIST_RANGE = 0.05   # ± half-range for residual histogram x-axis
AUTOCORR_LAGS       = 32     # display ± this many pixels in autocorrelation plots
MAX_TEMPORAL_LAGS   = 10     # number of frame lags for temporal autocorrelation
CACHE_DIR           = "cache"  # directory for intermediate results; None to disable
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
# Caching helpers                                                                #
# --------------------------------------------------------------------------- #

def _seq_cache_key(paths: list[Path], max_frames) -> str:
    """MD5 of file names, sizes, modification times, and max_frames."""
    h = hashlib.md5()
    for p in paths:
        st = p.stat()
        h.update(f"{p.name},{st.st_size},{st.st_mtime_ns}".encode())
    h.update(str(max_frames).encode())
    return h.hexdigest()[:16]


def _cache_paths(cache_dir: Path, seq_name: str, key: str):
    return (
        cache_dir / f"{seq_name}_{key}.npz",
        cache_dir / f"{seq_name}_{key}_residuals.npy",
    )


def _try_load_cache(npz_path: Path, memmap_path: Path):
    if not npz_path.exists():
        return None
    data = np.load(npz_path, allow_pickle=False)
    result = {k: data[k] for k in data.files}
    result['frame_shape'] = tuple(int(x) for x in result['frame_shape'])
    if memmap_path.exists():
        result['residuals_mm'] = np.load(str(memmap_path), mmap_mode='r')
    else:
        result['residuals_mm'] = None
    return result


def _save_cache(npz_path: Path, **arrays):
    np.savez_compressed(npz_path, **arrays)
    print(f"  [cache] saved {npz_path.name}")


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
    memmap_path: Path | None = None,
) -> tuple:
    """
    Single streaming pass that simultaneously:
      - builds the calibrated-value histogram
      - builds the residual histogram (frame − mean)
      - accumulates the FFT power spectrum for autocorrelation
      - accumulates per-pixel temporal variance
      - optionally saves per-frame residuals to a memory-mapped file

    Returns (cal_counts, cal_edges, res_counts, res_edges, avg_power, frame_shape,
             var_frame, residuals_mm).
    """
    cal_edges    = np.linspace(0.0, 1.0, n_bins + 1)
    res_edges    = np.linspace(-res_half_range, res_half_range, n_bins + 1)
    cal_counts   = np.zeros(n_bins, dtype=np.int64)
    res_counts   = np.zeros(n_bins, dtype=np.int64)
    power_accum: np.ndarray | None = None
    var_accum:   np.ndarray | None = None
    frame_shape: tuple | None = None
    residuals_mm: np.ndarray | None = None

    for i, p in enumerate(paths):
        print(f"  pass 2  [{i+1}/{len(paths)}] {p.name}", end="\r", flush=True)
        cal = calibrate_frame(load_raw(p), pattern, black, white)

        h, _ = np.histogram(cal, bins=cal_edges)
        cal_counts += h

        residual = cal - mean
        h, _ = np.histogram(residual, bins=res_edges)
        res_counts += h

        # Per-pixel temporal variance accumulation
        res_f64 = residual.astype(np.float64)
        if var_accum is None:
            var_accum = res_f64 ** 2
        else:
            var_accum += res_f64 ** 2

        # Save residuals to memmap for temporal correlation
        if memmap_path is not None:
            if residuals_mm is None:
                H, W = residual.shape
                residuals_mm = np.lib.format.open_memmap(
                    str(memmap_path), mode='w+', dtype=np.float32,
                    shape=(len(paths), H, W),
                )
            residuals_mm[i] = residual

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
    var_frame = (var_accum / len(paths)).astype(np.float32)
    return (cal_counts, cal_edges, res_counts, res_edges,
            avg_power, frame_shape, var_frame, residuals_mm)


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
    mid = lags
    crop[mid, mid] = np.nan   # hide trivial self-correlation peak
    return crop


def compute_temporal_autocorr(residuals_mm: np.ndarray, max_lag: int) -> np.ndarray:
    """
    Normalized temporal ACF: C(k) = <r(t)·r(t+k)> / <r(t)²>
    averaged over all pixels and valid frame pairs.
    C(0) = 1 by definition; C(k) ≈ 0 for k > 0 means temporally uncorrelated frames.
    """
    N, H, W = residuals_mm.shape
    pixel_count = H * W

    var0 = 0.0
    for t in range(N):
        var0 += float(np.sum(residuals_mm[t].astype(np.float64) ** 2))
    var0 /= N * pixel_count

    acf = np.ones(max_lag + 1)
    for k in range(1, min(max_lag + 1, N)):
        print(f"    temporal lag {k}/{min(max_lag, N-1)} …", end="\r", flush=True)
        cross = 0.0
        n_pairs = N - k
        for t in range(n_pairs):
            r_t  = residuals_mm[t].astype(np.float64).ravel()
            r_tk = residuals_mm[t + k].astype(np.float64).ravel()
            cross += float(np.dot(r_t, r_tk))
        acf[k] = cross / (n_pairs * pixel_count * var0)

    if max_lag >= N:
        acf[N:] = np.nan
    print()
    return acf


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
    display_lags: int = 10,
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
        ax.set_xlim(-display_lags, display_lags)
        ax.set_ylim(-display_lags, display_lags)
        ax.grid(True, alpha=0.3, linestyle="--")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(
        "Spatial noise autocorrelation — Bayer domain (lag-0 hidden, normalized to variance)",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def plot_variance_vs_mean(
    var_frames: list[np.ndarray],
    mean_frames: list[np.ndarray],
    labels: list[str],
    out: Path,
):
    """
    Hexbin scatter of per-pixel temporal variance vs per-pixel mean.
    For Poisson shot noise: variance = mean / K  (K = white − black in ADU).
    """
    n = len(var_frames)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    if n == 1:
        axes = [axes]

    for ax, var_frame, mean_frame, label in zip(axes, var_frames, mean_frames, labels):
        mean_vals = mean_frame.ravel().astype(np.float64)
        var_vals  = var_frame.ravel().astype(np.float64)

        hb = ax.hexbin(mean_vals, var_vals, gridsize=80, mincnt=1,
                       cmap="viridis", bins="log", linewidths=0.2)
        fig.colorbar(hb, ax=ax, label="pixel count (log)")

        # Best-fit slope through origin: var = alpha * mean  (Poisson: alpha = 1/K)
        alpha = float(np.dot(mean_vals, var_vals) / np.dot(mean_vals, mean_vals))
        x = np.array([mean_vals.min(), mean_vals.max()])
        ax.plot(x, alpha * x, "r--", linewidth=1.5,
                label=f"fit: var = {alpha:.4f}·mean")

        ax.set_xlabel("Per-pixel mean (calibrated)", fontsize=9)
        ax.set_ylabel("Per-pixel temporal variance", fontsize=9)
        ax.set_title(label, fontsize=10)
        ax.legend(fontsize=8)

    fig.suptitle(
        "Per-pixel temporal variance vs mean\n"
        "(Poisson shot noise → var ∝ mean; slope = 1/(white−black) in ADU)",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def plot_temporal_autocorr(
    temporal_acfs: list[np.ndarray],
    labels: list[str],
    out: Path,
    max_lag: int,
):
    lags = np.arange(max_lag + 1)
    fig, ax = plt.subplots(figsize=(8, 4))
    for acf, label, color in zip(temporal_acfs, labels, COLORS):
        ax.plot(lags, acf, "o-", color=color, label=label, markersize=5, linewidth=1.5)
    ax.axhline(0, color="k", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_xlabel("Temporal lag (frames)", fontsize=10)
    ax.set_ylabel("Normalized correlation", fontsize=10)
    ax.set_title(
        "Temporal noise autocorrelation\n"
        "(0 = independent frames, expected for Poisson shot noise)",
        fontsize=12,
    )
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

    cache_dir = Path(CACHE_DIR) if CACHE_DIR else None
    if cache_dir:
        cache_dir.mkdir(parents=True, exist_ok=True)

    labels = [Path(d).name for d in SEQUENCE_DIRS]

    all_means:      list[np.ndarray] = []
    all_var_frames: list[np.ndarray] = []
    all_patterns:   list[np.ndarray] = []
    all_sample_rgb: list[np.ndarray] = []
    adu_hist_data:  list[tuple[np.ndarray, np.ndarray]] = []
    cal_hist_data:  list[tuple[np.ndarray, np.ndarray]] = []
    res_hist_data:  list[tuple[np.ndarray, np.ndarray]] = []
    autocorrs:      list[np.ndarray] = []
    temporal_acfs:  list[np.ndarray] = []

    for i, (d, label) in enumerate(zip(SEQUENCE_DIRS, labels)):
        print(f"\nSequence {i+1}/{len(SEQUENCE_DIRS)}: {label}")
        paths = find_dngs(d)
        if MAX_FRAMES:
            paths = paths[:MAX_FRAMES]
        print(f"  {len(paths)} DNG files")

        seq_name = Path(d).name
        npz_path = memmap_path = None
        cached = None
        if cache_dir:
            key = _seq_cache_key(paths, MAX_FRAMES)
            npz_path, memmap_path = _cache_paths(cache_dir, seq_name, key)
            cached = _try_load_cache(npz_path, memmap_path)

        # Always load a sample RGB frame (single DNG, fast)
        mid = len(paths) // 2
        print(f"  Loading sample frame (index {mid}) as RGB …")
        all_sample_rgb.append(load_raw_rgb(paths[mid]))

        if cached:
            print(f"  [cache hit] {npz_path.name}")
            mean         = cached['mean_frame']
            var_frame    = cached['var_frame']
            avg_power    = cached['avg_power']
            frame_shape  = cached['frame_shape']
            adu_counts   = cached['adu_counts']
            adu_edges    = cached['adu_edges']
            cal_counts   = cached['cal_counts']
            cal_edges    = cached['cal_edges']
            res_counts   = cached['res_counts']
            res_edges    = cached['res_edges']
            pattern      = cached['pattern']
            black        = cached['black']
            white        = float(cached['white'])
            residuals_mm = cached.get('residuals_mm')
        else:
            pattern, black, white = get_raw_metadata(paths[0])
            print(f"  Black levels (R,G,G,B): {black}  |  White level: {white}")

            # Pass 1: mean + ADU histogram
            print(f"  Pass 1 — mean + ADU histogram …")
            mean, adu_counts, adu_edges = stream_pass1(
                paths, pattern, black, white, HIST_BINS,
            )
            print(f"  Mean calibrated range: [{mean.min():.4f}, {mean.max():.4f}]")

            # Pass 2: calibrated hist + residuals + autocorrelation + variance
            print(f"  Pass 2 — calibrated hist + residuals + autocorrelation + variance …")
            (cal_counts, cal_edges, res_counts, res_edges,
             avg_power, frame_shape, var_frame, residuals_mm) = stream_pass2(
                paths, mean, pattern, black, white,
                HIST_BINS, RESIDUAL_HIST_RANGE,
                memmap_path=memmap_path,
            )

            if cache_dir and npz_path:
                _save_cache(
                    npz_path,
                    mean_frame=mean,
                    var_frame=var_frame,
                    avg_power=avg_power,
                    frame_shape=np.array(frame_shape),
                    adu_counts=adu_counts,
                    adu_edges=adu_edges,
                    cal_counts=cal_counts,
                    cal_edges=cal_edges,
                    res_counts=res_counts,
                    res_edges=res_edges,
                    pattern=pattern,
                    black=black,
                    white=np.array(white),
                )

        all_means.append(mean)
        all_var_frames.append(var_frame)
        all_patterns.append(pattern)
        adu_hist_data.append((adu_counts, adu_edges))
        cal_hist_data.append((cal_counts, cal_edges))
        res_hist_data.append((res_counts, res_edges))
        autocorrs.append(compute_autocorr(avg_power, frame_shape, AUTOCORR_LAGS))

        if residuals_mm is not None and MAX_TEMPORAL_LAGS > 0:
            print(f"  Computing temporal autocorrelation (max lag {MAX_TEMPORAL_LAGS}) …")
            temporal_acfs.append(
                compute_temporal_autocorr(residuals_mm, MAX_TEMPORAL_LAGS)
            )

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

    print("Plotting per-pixel variance vs mean …")
    plot_variance_vs_mean(
        all_var_frames, all_means, labels,
        out_dir / "variance_vs_mean.png",
    )

    if temporal_acfs:
        print("Plotting temporal autocorrelations …")
        plot_temporal_autocorr(
            temporal_acfs, labels[:len(temporal_acfs)],
            out_dir / "temporal_correlation.png",
            max_lag=MAX_TEMPORAL_LAGS,
        )

    print(f"\nDone. All plots saved to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
