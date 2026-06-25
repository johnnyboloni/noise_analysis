"""
Flat frame noise analysis for DNG and GN3 raw sequences.

Edit the CONFIG block below, then run:
    python analyze_flats.py

Outputs (saved to OUTPUT_DIR):
  - sample_frames.png        : one representative RGB frame per sequence
  - mean_and_variance_frames.png : per-pixel mean (RGB) and temporal variance (heatmap)
  - histograms_adu.png       : raw ADU histograms (before any calibration)
  - histograms_cal.png       : calibrated [0,1] histograms
  - histograms_residual.png  : histograms after mean subtraction
  - spatial_correlation.png  : 2D noise autocorrelation (Bayer domain)
  - variance_vs_mean.png     : per-pixel temporal variance vs mean (Poisson test)
  - temporal_correlation.png : temporal noise autocorrelation function
"""

import argparse
import json
import math
import sys
import hashlib
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
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

# GN3v4: sequence names ending '0' = 12-bit, ending '1' = 10-bit.
# white_level is derived from imageType in the .imgprops sidecar per frame.
# Black level is not in the metadata; 256 matches observed sensor floor.
gn3_root = '/stage/algo-datasets/DB/LME/raw/20250126_GN3v4_train'
GN3_BLACK_LEVEL = 256

seqs= {'/stage/algo-datasets/DB/DeepISP/MotionCam_RawVideos/S23+_lowlight_uniform/':
       ['260518_085011_VIDEO_24mm',
        '260518_085327_VIDEO_24mm',
        '260518_090426_VIDEO_24mm'],
      
       '/stage/algo-datasets/DB/LME/raw/20250126_GN3v4_train': 
       ['/0241_GN3',
        '/0250_GN3',
        '/0251_GN3',
        '/0260_GN3',
        '/0261_GN3',
        '/0270_GN3',
        '/0271_GN3','/0240_GN3',]
      }

           


OUTPUT_DIR          = "output"
MAX_FRAMES          = None   # int to cap frames per sequence, None = load all
HIST_BINS           = 512
RESIDUAL_HIST_RANGE = 0.05   # ± half-range for residual histogram x-axis
AUTOCORR_LAGS       = 32     # display ± this many pixels in autocorrelation plots
MAX_TEMPORAL_LAGS   = 10     # number of frame lags for temporal autocorrelation
CACHE_DIR           = "cache"  # directory for intermediate results; None to disable

# Set True to process only QUICK_TEST_FRAMES frames per sequence — fast sanity check
QUICK_TEST        = False
QUICK_TEST_FRAMES = 5
# ============================================================


# --------------------------------------------------------------------------- #
# DNG I/O helpers                                                               #
# --------------------------------------------------------------------------- #

def find_dngs(directory: str) -> list[Path]:
    p = Path(directory)
    dngs = sorted(p.glob("*.dng")) + sorted(p.glob("*.DNG"))
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


# --------------------------------------------------------------------------- #
# GN3 .raw I/O helpers                                                          #
# --------------------------------------------------------------------------- #

# Maps bayerOrder string → 2×2 pattern array (0=R 1=G 2=B 3=G2)
_BAYER_ORDER_TO_PATTERN: dict[str, np.ndarray] = {
    'RGGB': np.array([[0, 1], [3, 2]], dtype=np.int32),
    'GRBG': np.array([[1, 0], [2, 3]], dtype=np.int32),
    'BGGR': np.array([[2, 3], [1, 0]], dtype=np.int32),
    'GBRG': np.array([[1, 2], [0, 3]], dtype=np.int32),
}


def find_raws(directory: str) -> list[Path]:
    p = Path(directory)
    return sorted(p.glob("*.raw"))


def get_raw_metadata_gn3(path: Path) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Read GN3 .imgprops sidecar JSON (same stem as .raw file).
    Returns (bayer_pattern, black_level_per_channel, white_level).
    white_level is derived from imageType (BAYER10 → 1023, BAYER12 → 4095, etc.).
    black_level is uniform across channels (GN3_BLACK_LEVEL config value).
    """
    sidecar = path.with_suffix('.imgprops')
    meta = json.loads(sidecar.read_text())
    pattern = _BAYER_ORDER_TO_PATTERN[meta['bayerOrder']]
    bit_depth = int(''.join(c for c in meta['imageType'] if c.isdigit()))
    white = float((1 << bit_depth) - 1)
    black = np.full(4, GN3_BLACK_LEVEL, dtype=np.float32)
    return pattern, black, white


def load_raw_gn3(path: Path, shape: tuple[int, int]) -> np.ndarray:
    """Load a GN3 .raw file (uint16 LE, no header) and return float32 (H, W)."""
    return np.frombuffer(path.read_bytes(), dtype='<u2').reshape(shape).astype(np.float32)


def load_raw_rgb_gn3(path: Path, shape: tuple[int, int],
                     pattern: np.ndarray, black: np.ndarray, white: float) -> np.ndarray:
    """Load, calibrate, and demosaic a GN3 frame for display; returns H×W×3 uint8."""
    raw = load_raw_gn3(path, shape)
    cal = calibrate_frame(raw, pattern, black, white)
    rgb_f = demosaic_to_rgb(cal, pattern)
    return (rgb_f * 255).astype(np.uint8)


# --------------------------------------------------------------------------- #
# Format detection                                                               #
# --------------------------------------------------------------------------- #

def _detect_format(directory: str) -> str:
    """Return 'dng' or 'raw' based on which files are present."""
    p = Path(directory)
    if find_dngs(directory):
        return 'dng'
    if find_raws(directory):
        return 'raw'
    sys.exit(f"No recognized files (.dng / .raw) in {directory}")


# --------------------------------------------------------------------------- #
# Calibration + demosaicing                                                     #
# --------------------------------------------------------------------------- #

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
        mm = np.load(str(memmap_path), mmap_mode='r')
        # Spot-check: if the first few frames are all zeros the file was never properly
        # written (e.g. a previous crash before flush).  Treat it as missing so the
        # next run regenerates it.
        n_check = min(3, mm.shape[0]) if mm.ndim == 3 else 0
        if n_check > 0 and not np.any(mm[:n_check]):
            print(f"  [cache] residuals file looks empty — will regenerate: {memmap_path.name}")
            result['residuals_mm'] = None
        else:
            result['residuals_mm'] = mm
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
    loader,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Single streaming pass:
      - accumulates the per-pixel calibrated mean
      - builds the raw ADU histogram (before calibration)

    Returns (mean float32, adu_counts int64, adu_edges float64).
    """
    adu_edges  = np.linspace(0.0, white + 1, n_bins + 1)
    adu_counts = np.zeros(n_bins, dtype=np.int64)
    accum: np.ndarray | None = None

    for i, p in enumerate(paths):
        print(f"  pass 1  [{i+1}/{len(paths)}] {p.name}", end="\r", flush=True)
        raw = loader(p)
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
    loader,
    memmap_path: Path | None = None,
) -> tuple:
    """
    Single streaming pass:
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
        cal = calibrate_frame(loader(p), pattern, black, white)

        h, _ = np.histogram(cal, bins=cal_edges)
        cal_counts += h

        residual = cal - mean
        h, _ = np.histogram(residual, bins=res_edges)
        res_counts += h

        res_f64 = residual.astype(np.float64)
        if var_accum is None:
            var_accum = res_f64 ** 2
        else:
            var_accum += res_f64 ** 2

        if memmap_path is not None:
            if residuals_mm is None:
                H, W = residual.shape
                residuals_mm = np.lib.format.open_memmap(
                    str(memmap_path), mode='w+', dtype=np.float32,
                    shape=(len(paths), H, W),
                )
            residuals_mm[i] = residual

        residual -= residual.mean()
        fft_power = np.abs(np.fft.rfft2(residual)) ** 2
        if power_accum is None:
            power_accum = fft_power.astype(np.float64)
            frame_shape = residual.shape
        else:
            power_accum += fft_power

    print()
    if residuals_mm is not None:
        residuals_mm.flush()   # ensure writes reach disk before the file is reused as cache
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
    full = np.fft.irfft2(avg_power, s=frame_shape)
    centered = np.fft.fftshift(full)
    H, W = centered.shape
    cy, cx = H // 2, W // 2
    norm = centered[cy, cx]
    normalized = centered / max(norm, 1e-12)
    crop = normalized[cy - lags : cy + lags + 1,
                      cx - lags : cx + lags + 1].copy()
    crop[lags, lags] = np.nan
    return crop


def compute_temporal_autocorr(residuals_mm: np.ndarray, max_lag: int) -> np.ndarray:
    """
    Normalized temporal ACF: C(k) = <r(t)·r(t+k)> / <r(t)²>
    averaged over all pixels and valid frame pairs.
    C(0) = 1 by definition; C(k) ≈ 0 for k > 0 means temporally uncorrelated frames.

    Bias correction: subtracting the sample mean introduces a known negative bias of
    -1/(N-1) at every lag k > 0 (the mean frame was estimated from the same N frames).
    We correct for this analytically so the estimator is unbiased under shot noise.
    """
    N, H, W = residuals_mm.shape
    pixel_count = H * W

    var0 = 0.0
    for t in range(N):
        var0 += float(np.sum(residuals_mm[t].astype(np.float64) ** 2))
    var0 /= N * pixel_count

    if var0 == 0.0:
        print(f"\n  WARNING: residuals are all zero "
              f"(shape={residuals_mm.shape}, dtype={residuals_mm.dtype}). "
              f"Delete {CACHE_DIR!r} and re-run to regenerate.")
        return np.full(max_lag + 1, np.nan)

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

    # Correct for mean-subtraction bias: E[C(k)] = -1/(N-1) for k>0 under i.i.d. frames.
    if N > 1:
        acf[1:] += 1.0 / (N - 1)

    print()
    return acf


# --------------------------------------------------------------------------- #
# Plotting                                                                      #
# --------------------------------------------------------------------------- #

# tab10 palette — 10 distinct colors, enough for all sequences
COLORS = list(plt.cm.tab10.colors)
ALPHA  = 0.55
DPI    = 200


def _grid(n: int, max_cols: int = 4) -> tuple[int, int]:
    """Return (nrows, ncols) for a roughly-square grid of n panels."""
    ncols = min(n, max_cols)
    nrows = math.ceil(n / ncols)
    return nrows, ncols


def _make_grid_axes(n: int, panel_w: float, panel_h: float, max_cols: int = 4):
    """Create a figure + flat list of n axes in a grid; hide leftover cells."""
    nrows, ncols = _grid(n, max_cols)
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(panel_w * ncols, panel_h * nrows),
                             squeeze=False)
    flat = axes.ravel().tolist()
    for ax in flat[n:]:   # hide unused cells
        ax.set_visible(False)
    return fig, flat[:n]


def plot_sample_frames(rgb_frames: list[np.ndarray], labels: list[str], out: Path):
    n = len(rgb_frames)
    fig, axes = _make_grid_axes(n, panel_w=6, panel_h=5)
    for ax, rgb, label in zip(axes, rgb_frames, labels):
        ax.imshow(rgb, aspect="auto")
        ax.set_title(f"{label}\n(sample frame)", fontsize=9)
        ax.axis("off")
    fig.suptitle("Sample frames (demosaiced RGB)", fontsize=13)
    fig.tight_layout()
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def plot_mean_and_variance_frames(
    rgb_means: list[np.ndarray],
    var_frames: list[np.ndarray],
    labels: list[str],
    out: Path,
    max_cols: int = 4,
):
    """Two rows per column group: top = demosaiced mean, bottom = variance heatmap."""
    method = "VNG demosaic" if _HAS_CV2 else "half-res channel extraction"
    n = len(labels)
    ncols = min(n, max_cols)
    n_groups = math.ceil(n / ncols)
    nrows = 2 * n_groups   # mean row + variance row per group

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(6 * ncols, 5 * nrows),
                             squeeze=False)
    for ax in axes.ravel():
        ax.set_visible(False)

    for i, (rgb, var, label) in enumerate(zip(rgb_means, var_frames, labels)):
        grp, col = divmod(i, ncols)
        mean_row = grp * 2
        var_row  = grp * 2 + 1

        ax_m = axes[mean_row, col]
        ax_m.set_visible(True)
        ax_m.imshow(rgb, aspect="auto")
        ax_m.set_title(f"{label}\n(mean)", fontsize=9)
        ax_m.axis("off")

        ax_v = axes[var_row, col]
        ax_v.set_visible(True)
        im = ax_v.imshow(var, cmap="hot", aspect="auto")
        ax_v.set_title(f"{label}\n(temporal variance)", fontsize=9)
        ax_v.axis("off")
        fig.colorbar(im, ax=ax_v, fraction=0.046, pad=0.04, label="variance")

    fig.suptitle(
        f"Per-pixel mean ({method}, stretched) and temporal variance",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
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
    for i, ((counts, edges), label) in enumerate(zip(hist_data, labels)):
        width = edges[1] - edges[0]
        density = counts / (counts.sum() * width)
        ax.stairs(density, edges, fill=True, color=COLORS[i % len(COLORS)],
                  alpha=ALPHA, label=label, linewidth=0.8)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, linestyle="--")
    fig.tight_layout()
    fig.savefig(out, dpi=DPI)
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
    fig, axes = _make_grid_axes(n, panel_w=5, panel_h=5)

    vmax = max(np.nanmax(np.abs(a)) for a in autocorrs)
    vmax = max(vmax, 1e-6)

    extent = [-lags - 0.5, lags + 0.5, -lags - 0.5, lags + 0.5]
    for ax, autocorr, label in zip(axes, autocorrs, labels):
        im = ax.imshow(autocorr, extent=extent, cmap="RdBu_r",
                       vmin=-vmax, vmax=vmax, aspect="equal",
                       origin="lower", interpolation="nearest")
        ax.set_title(label, fontsize=9)
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
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def plot_variance_vs_mean(
    var_frames: list[np.ndarray],
    mean_frames: list[np.ndarray],
    labels: list[str],
    out: Path,
):
    n = len(var_frames)
    fig, axes = _make_grid_axes(n, panel_w=5, panel_h=5)

    for ax, var_frame, mean_frame, label in zip(axes, var_frames, mean_frames, labels):
        mean_vals = mean_frame.ravel().astype(np.float64)
        var_vals  = var_frame.ravel().astype(np.float64)

        hb = ax.hexbin(mean_vals, var_vals, gridsize=80, mincnt=1,
                       cmap="viridis", bins="log", linewidths=0.2)
        fig.colorbar(hb, ax=ax, label="pixel count (log)")

        alpha = float(np.dot(mean_vals, var_vals) / np.dot(mean_vals, mean_vals))
        x = np.array([mean_vals.min(), mean_vals.max()])
        ax.plot(x, alpha * x, "r--", linewidth=1.5,
                label=f"fit: var = {alpha:.4f}·mean")

        ax.set_xlabel("Per-pixel mean (calibrated)", fontsize=9)
        ax.set_ylabel("Per-pixel temporal variance", fontsize=9)
        ax.set_title(label, fontsize=9)
        ax.legend(fontsize=8)

    fig.suptitle(
        "Per-pixel temporal variance vs mean\n"
        "(Poisson shot noise → var ∝ mean; slope = 1/(white−black) in ADU)",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
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
    for i, (acf, label) in enumerate(zip(temporal_acfs, labels)):
        ax.plot(lags, acf, "o-", color=COLORS[i % len(COLORS)],
                label=label, markersize=5, linewidth=1.5)
    ax.axhline(0, color="k", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_xlabel("Temporal lag (frames)", fontsize=10)
    ax.set_ylabel("Normalized correlation", fontsize=10)
    ax.set_title(
        "Temporal noise autocorrelation (bias-corrected for sample-mean subtraction)\n"
        "0 = independent frames, expected for Poisson shot noise",
        fontsize=12,
    )
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, linestyle="--")
    fig.tight_layout()
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    print(f"Saved {out}")


# --------------------------------------------------------------------------- #
# Main                                                                          #
# --------------------------------------------------------------------------- #

def _none_or_auto(s: str):
    """argparse type for config values that can be None or a scalar."""
    if s.lower() == 'none':
        return None
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        pass
    return s


def _apply_cli_overrides() -> None:
    """Override any ALL_CAPS scalar config constant via a matching --lower-kebab-case flag."""
    g = globals()
    scalar = (bool, int, float, str, type(None))
    keys = sorted(k for k in g if k.isupper() and not k.startswith('_') and isinstance(g[k], scalar))

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    for key in keys:
        val = g[key]
        flag = '--' + key.lower().replace('_', '-')
        if isinstance(val, bool):
            parser.add_argument(flag, dest=key, default=None,
                                action=argparse.BooleanOptionalAction,
                                help=f"(default: {val})")
        elif isinstance(val, int):
            parser.add_argument(flag, dest=key, type=int, default=None, metavar='N',
                                help=f"(default: {val})")
        elif isinstance(val, float):
            parser.add_argument(flag, dest=key, type=float, default=None, metavar='F',
                                help=f"(default: {val})")
        else:  # str or None
            parser.add_argument(flag, dest=key, type=_none_or_auto, default=None, metavar='S',
                                help=f"(default: {val!r}; pass 'none' to clear)")

    args = parser.parse_args()
    for key, new_val in vars(args).items():
        if new_val is not None:
            g[key] = new_val


def _process_sequence(d: str, label: str, effective_max_frames, cache_dir: Path | None):
    """Run both streaming passes for one sequence; return a result dict."""
    fmt = _detect_format(d)
    paths = find_dngs(d) if fmt == 'dng' else find_raws(d)
    if not paths:
        print(f"  WARNING: no files found, skipping")
        return None

    if effective_max_frames:
        paths = paths[:effective_max_frames]
    n_frames = len(paths)
    print(f"  {n_frames} {fmt.upper()} files")
    label = f"{label}  (N={n_frames})"

    seq_name = Path(d).name
    npz_path = memmap_path = None
    cached = None
    if cache_dir:
        key = _seq_cache_key(paths, effective_max_frames)
        npz_path, memmap_path = _cache_paths(cache_dir, seq_name, key)
        if not QUICK_TEST:
            cached = _try_load_cache(npz_path, memmap_path)

    if fmt == 'dng':
        pattern, black, white = get_raw_metadata(paths[0])
        loader = load_raw
        sample_rgb = load_raw_rgb(paths[len(paths) // 2])
    else:
        pattern, black, white = get_raw_metadata_gn3(paths[0])
        meta = json.loads(paths[0].with_suffix('.imgprops').read_text())
        shape = (meta['height'], meta['width'])
        loader = lambda p, _s=shape: load_raw_gn3(p, _s)
        sample_rgb = load_raw_rgb_gn3(paths[len(paths) // 2], shape, pattern, black, white)

    print(f"  Black levels (R,G,G2,B): {black}  |  White level: {white}")

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
        # temporal_acf stored as array in cache; all-NaN means it wasn't computed
        _tacf = cached.get('temporal_acf')
        if _tacf is not None and not np.all(np.isnan(_tacf)):
            temporal_acf = _tacf
        else:
            # Old cache without temporal_acf — try memmap fallback
            temporal_acf = None
            residuals_mm = cached.get('residuals_mm')
            if residuals_mm is not None and MAX_TEMPORAL_LAGS > 0:
                print(f"  Computing temporal autocorrelation (max lag {MAX_TEMPORAL_LAGS}) …")
                temporal_acf = compute_temporal_autocorr(residuals_mm, MAX_TEMPORAL_LAGS)
            else:
                print(f"  [warn] temporal_acf missing from cache and no residuals memmap found.")
                print(f"         Delete cache/*.npz and re-run to regenerate.")
    else:
        print(f"  Pass 1 — mean + ADU histogram …")
        mean, adu_counts, adu_edges = stream_pass1(
            paths, pattern, black, white, HIST_BINS, loader,
        )
        print(f"  Mean calibrated range: [{mean.min():.4f}, {mean.max():.4f}]")

        print(f"  Pass 2 — calibrated hist + residuals + autocorrelation + variance …")
        (cal_counts, cal_edges, res_counts, res_edges,
         avg_power, frame_shape, var_frame, residuals_mm) = stream_pass2(
            paths, mean, pattern, black, white,
            HIST_BINS, RESIDUAL_HIST_RANGE, loader,
            memmap_path=memmap_path,
        )

        temporal_acf = None
        if residuals_mm is not None and MAX_TEMPORAL_LAGS > 0:
            print(f"  Computing temporal autocorrelation (max lag {MAX_TEMPORAL_LAGS}) …")
            temporal_acf = compute_temporal_autocorr(residuals_mm, MAX_TEMPORAL_LAGS)

        if cache_dir and npz_path and not QUICK_TEST:
            _tacf_save = temporal_acf if temporal_acf is not None else np.full(MAX_TEMPORAL_LAGS + 1, np.nan)
            _save_cache(
                npz_path,
                mean_frame=mean, var_frame=var_frame,
                avg_power=avg_power, frame_shape=np.array(frame_shape),
                adu_counts=adu_counts, adu_edges=adu_edges,
                cal_counts=cal_counts, cal_edges=cal_edges,
                res_counts=res_counts, res_edges=res_edges,
                pattern=pattern, black=black, white=np.array(white),
                temporal_acf=_tacf_save,
            )

    return dict(
        label=label, sample_rgb=sample_rgb,
        mean=mean, var_frame=var_frame, pattern=pattern,
        adu_hist=(adu_counts, adu_edges),
        cal_hist=(cal_counts, cal_edges),
        res_hist=(res_counts, res_edges),
        autocorr=compute_autocorr(avg_power, frame_shape, AUTOCORR_LAGS),
        temporal_acf=temporal_acf,
    )


def _plot_group(results: list[dict], group_label: str, out_dir: Path):
    """Generate all plots for one group of sequences, saved into a subdirectory."""
    labels      = [r['label']      for r in results]
    sample_rgbs = [r['sample_rgb'] for r in results]
    means       = [r['mean']       for r in results]
    var_frames  = [r['var_frame']  for r in results]
    patterns    = [r['pattern']    for r in results]
    adu_hists   = [r['adu_hist']   for r in results]
    cal_hists   = [r['cal_hist']   for r in results]
    res_hists   = [r['res_hist']   for r in results]
    autocorrs   = [r['autocorr']   for r in results]
    tacfs       = [r['temporal_acf'] for r in results if r['temporal_acf'] is not None]
    tacf_labels = [r['label']        for r in results if r['temporal_acf'] is not None]

    group_dir = out_dir / group_label
    group_dir.mkdir(parents=True, exist_ok=True)

    plot_sample_frames(sample_rgbs, labels, group_dir / "sample_frames.png")

    mean_rgb = [demosaic_to_rgb(m, p) for m, p in zip(means, patterns)]
    plot_mean_and_variance_frames(mean_rgb, var_frames, labels,
                                  group_dir / "mean_and_variance_frames.png")

    plot_histograms(adu_hists, labels, group_dir / "histograms_adu.png",
                    title=f"Raw ADU histograms — {group_label}",
                    xlabel="Raw pixel value (ADU)")

    plot_histograms(cal_hists, labels, group_dir / "histograms_cal.png",
                    title=f"Calibrated histograms — {group_label}",
                    xlabel="Calibrated value  (ADU − black) / (white − black)")

    plot_histograms(res_hists, labels, group_dir / "histograms_residual.png",
                    title=f"Residual histograms — {group_label}",
                    xlabel=f"Residual  [±{RESIDUAL_HIST_RANGE}]")

    plot_autocorrelations(autocorrs, labels, group_dir / "spatial_correlation.png",
                          lags=AUTOCORR_LAGS)

    plot_variance_vs_mean(var_frames, means, labels, group_dir / "variance_vs_mean.png")

    if tacfs:
        plot_temporal_autocorr(tacfs, tacf_labels, group_dir / "temporal_correlation.png",
                               max_lag=MAX_TEMPORAL_LAGS)
    else:
        print(f"  [warn] No temporal ACF data for {group_label} — skipping temporal_correlation.png")


def main():
    _apply_cli_overrides()

    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    cache_dir = Path(CACHE_DIR) if CACHE_DIR else None
    if cache_dir:
        cache_dir.mkdir(parents=True, exist_ok=True)

    effective_max_frames = QUICK_TEST_FRAMES if QUICK_TEST else MAX_FRAMES
    if QUICK_TEST:
        print(f"*** QUICK_TEST mode: capping each sequence to {QUICK_TEST_FRAMES} frames ***\n")

    any_processed = False
    for root, seq_names in seqs.items():
        group_label = Path(root.rstrip('/')).name
        print(f"\n{'='*60}\nGroup: {group_label}  ({len(seq_names)} sequences)\n{'='*60}")

        results = []
        for i, seq_name in enumerate(seq_names):
            d = root + seq_name
            label = Path(seq_name.lstrip('/')).name
            print(f"\n  [{i+1}/{len(seq_names)}] {label}")
            r = _process_sequence(d, label, effective_max_frames, cache_dir)
            if r is not None:
                results.append(r)

        if not results:
            print(f"  No sequences processed for {group_label}, skipping plots.")
            continue

        print(f"\nPlotting {len(results)} sequences for {group_label} …")
        _plot_group(results, group_label, out_dir)
        any_processed = True

    if not any_processed:
        sys.exit("No sequences processed.")

    print(f"\nDone. Plots saved under: {out_dir.resolve()}/<group>/")



if __name__ == "__main__":
    main()
