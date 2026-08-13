"""
GT frame aggregation analysis for a static lowlight sequence.

Edit the CONFIG block below, then run:
    python analyze_gt_sequence.py

Outputs (saved to OUTPUT_DIR):
  - gt_sample_frames.png   : evenly-spaced sample frames (RGB)
  - gt_aggregated.png      : mean / median / trimmed-mean side-by-side (RGB)
  - gt_differences.png     : (mean−median), (mean−trimmed), (median−trimmed) heatmaps
  - gt_mean_darksub_rgb_*  : mean frame minus a sigma-clipped master dark,
                             with a report of how many dark frames the master
                             actually wants (only when DARK_DIR is set)
  - comparison/            : GT candidates side by side -- mean, dark-subtracted
                             mean, and gain=1 stills -- as full frames, 100%
                             crops, and pairwise difference maps, with their
                             residual pixel noise printed (only when DARK_DIR
                             and/or STILLS_DIR is set)
  - gt_temporal_noise.png  : per-pixel temporal std heatmap (noise map)
  - gt_noise_cv.png        : σ/μ — coefficient of variation (relative noise)
  - gt_noise_shot_norm.png : σ/√μ — deviation from shot-noise-limited (1 = pure Poisson)
  - gt_convergence.png     : std across disjoint N-frame blocks vs N (log-log,
                             two panels: plain std, and shot-noise-normalized
                             σ/√μ), each with a 1/√N reference — an earlier
                             version of this plot compared the running mean
                             against the full-sequence mean, which is biased
                             (the running mean's frames are a subset of the
                             frames used for the "ground truth" it's compared
                             against, forcing the error toward zero as N
                             approaches the total frame count regardless of
                             actual noise behavior). This version instead
                             splits the sequence into independent,
                             non-overlapping N-frame blocks and measures the
                             spread across their means directly, which has no
                             such bias. The shot-noise-normalized panel
                             additionally removes the signal-level dependence
                             that a plain std mixes together across pixels.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from raw_utils import (
    detect_format, find_dngs, find_raws,
    get_raw_metadata, get_raw_metadata_gn3, get_color_metadata,
    load_raw, load_raw_gn3, load_raw_rgb, load_raw_rgb_gn3,
    calibrate_frame, demosaic_to_rgb, plot_sample_frames, save_rgb_png,
    highpass_std, bayer_plane_median3,
)


# ============================================================
# CONFIG — edit paths and options here
# ============================================================
SEQUENCE_DIR  = "/path/to/static/sequence"
OUTPUT_DIR    = "output/gt_analysis"
GN3_BLACK_LEVEL = 256    # uniform black level for GN3 .raw files

MAX_FRAMES    = None  # int to cap total frames loaded, None = all
MAX_STACK     = 60    # max frames loaded into RAM for median / trimmed-mean
TRIM_FRAC     = 0.05  # total fraction trimmed (symmetric: TRIM_FRAC/2 from each tail)
N_SAMPLES     = 5     # number of evenly-spaced sample frames to show

CONV_ROI_FRAC   = 0.25  # central crop (fraction of H and W) used for the block-std
                        # convergence check -- kept small so per-N accumulators stay cheap
CONV_N_LEVELS   = 8     # number of log-spaced block sizes N to test (plus N=1)

N_CHECKPOINTS   = 5     # running-mean snapshots saved while streaming (log-spaced in N)
CHECKPOINT_CROP = 400   # centre-crop size (px) for the 100%-zoom checkpoint comparison

DARK_DIR        = None  # dir of dark frames (lens capped, same gain/exposure).
                        # Sigma-clipped master dark is subtracted from the mean
                        # frame. None = skip dark subtraction.
DARK_MAX_FRAMES = None  # how many dark frames to average. None = all of them,
                        # which is normally what you want: the master needs far
                        # more frames than feels necessary, and too few makes
                        # the correction worse (the run prints the number your
                        # own data calls for).
DARK_SIGMA_CLIP = 4.0   # reject dark samples beyond this many sigma from the
                        # per-pixel mean (cosmic rays, dropped frames)
HOT_PIXEL_SIGMA = 5.0   # flag master-dark pixels this many residual-sigma from
                        # their same-colour neighbours as defects, and repair
                        # them by interpolation. 0 or None = skip.
STILLS_DIR      = None  # dir of gain=1 long-exposure stills to compare against.
                        # None = skip the comparison.
MATCH_STILL_INTENSITY = True   # rescale the stills by a robust ratio so the
                               # comparison is not dominated by an exposure
                               # mismatch between the two capture settings
COMPARISON_CROP = 400   # centre-crop size (px) for the 100%-zoom comparison
# ============================================================


# --------------------------------------------------------------------------- #
# Format-aware loader builder                                                   #
# --------------------------------------------------------------------------- #

def _make_loaders(directory: str):
    """
    Detect format and return (fmt, paths, pattern, black, white, loader, sample_fn).
    loader(path)     → float32 (H, W) raw Bayer ADU
    sample_fn(path)  → uint8  (H, W, 3) demosaiced RGB
    """
    fmt = detect_format(directory)
    if fmt == 'dng':
        paths     = find_dngs(directory)
        pattern, black, white = get_raw_metadata(paths[0])
        loader    = load_raw
        sample_fn = load_raw_rgb
    else:
        paths     = find_raws(directory)
        pattern, black, white = get_raw_metadata_gn3(paths[0], GN3_BLACK_LEVEL)
        meta      = json.loads(paths[0].with_suffix('.imgprops').read_text())
        shape     = (meta['height'], meta['width'])
        loader    = lambda p, _s=shape: load_raw_gn3(p, _s)
        sample_fn = lambda p, _s=shape, _pat=pattern, _bl=black, _wh=white: \
            load_raw_rgb_gn3(p, _s, _pat, _bl, _wh)
    return fmt, paths, pattern, black, white, loader, sample_fn


# --------------------------------------------------------------------------- #
# Streaming passes                                                               #
# --------------------------------------------------------------------------- #

def _stream_mean(paths, pattern, black, white, loader,
                 checkpoints=None, on_checkpoint=None):
    """Pass 1 — compute per-pixel calibrated mean, streaming one frame at a time.

    Accumulates raw ADU in float64 so sub-black noise cancels across frames,
    then calibrates (and clips) the final mean once.

    If `checkpoints` (a set of 1-based frame counts) and `on_checkpoint` are
    given, the running mean is calibrated and handed to on_checkpoint(idx, frame)
    as each count is reached -- letting the caller save intermediate averages
    without a second pass. The callback is expected to consume the frame
    immediately (save it / keep a crop) rather than retain it, so peak memory
    stays at one frame regardless of how many checkpoints are requested.

    Frames are accumulated in two disjoint halves (even- and odd-indexed) rather
    than one running sum, which costs one extra float64 accumulator but enables
    the split-half noise estimate below at no extra I/O. The full running sum is
    just their sum.

    Returns (full_mean, metrics), where metrics is a list of dicts with keys
    n, temporal (split-half temporal noise) and highpass (residual pixel noise),
    both in calibrated units -- see _checkpoint_metrics.
    """
    checkpoints = set(checkpoints or ())
    acc_e = acc_o = None
    n_e = n_o = 0
    metrics = []
    n = len(paths)

    for i, p in enumerate(paths):
        idx = i + 1
        print(f"  mean  [{idx}/{n}] {p.name}", end="\r", flush=True)
        raw = loader(p).astype(np.float64)
        if i % 2 == 0:
            acc_e = raw if acc_e is None else acc_e + raw
            n_e += 1
        else:
            acc_o = raw if acc_o is None else acc_o + raw
            n_o += 1

        if idx in checkpoints:
            total   = acc_e if acc_o is None else acc_e + acc_o
            running = calibrate_frame((total / idx).astype(np.float32),
                                      pattern, black, white)
            del total          # free before the split-half diff allocates

            # Split-half difference, in calibrated units. The scene and any
            # fixed-pattern noise are identical in both halves and cancel
            # exactly, so this is a picture of the temporal noise alone --
            # the only component averaging can remove. Computed once and
            # reused for both the metric and the saved image.
            if n_o > 0:
                half_diff = ((acc_e / n_e) - (acc_o / n_o)).astype(np.float32)
                half_diff /= float(white - black[0])
            else:
                half_diff = None          # N=1: no second half to compare

            metrics.append(_checkpoint_metrics(idx, half_diff, n_e, n_o,
                                               running, pattern))
            if on_checkpoint is not None:
                on_checkpoint(idx, running, half_diff)
            del running, half_diff
    print()

    total    = acc_e if acc_o is None else acc_e + acc_o
    mean_adu = (total / n).astype(np.float32)
    # The raw-ADU mean is returned alongside the calibrated one because dark
    # subtraction has to happen before black-level removal (see _dark_correct).
    return calibrate_frame(mean_adu, pattern, black, white), mean_adu, metrics


def _checkpoint_metrics(idx, half_diff, n_e, n_o, running, pattern):
    """
    Two complementary noise measures for the running mean at N=idx, both in
    calibrated units (fraction of full scale) so they are unaffected by the
    auto-brighten and gamma that the displayed PNGs go through.

    temporal -- split-half estimate. Average the even- and odd-indexed frames
      separately and subtract: the scene AND any fixed-pattern noise are
      identical in both halves and cancel exactly, leaving only temporal noise.
      With var(A-B) = sigma^2 (1/n_e + 1/n_o), the temporal noise of the full
      N-frame average is std(A-B) * sqrt(n_e*n_o/(n_e+n_o)) / sqrt(N). This is
      unbiased (the halves share no frames) and should track 1/sqrt(N).

    highpass -- std of the running mean's own high-pass residual. Includes
      temporal noise AND fixed-pattern noise, which averaging cannot remove.
      Once this flattens while `temporal` keeps falling, the frame is
      FPN-limited and more frames will not visibly help.
    """
    if half_diff is not None:
        temporal = float(half_diff.std()) * np.sqrt(n_e * n_o / (n_e + n_o)) \
                   / np.sqrt(n_e + n_o)
    else:
        temporal = float('nan')              # N=1: no second half to compare
    return {'n': idx,
            'temporal': temporal,
            'highpass': highpass_std(running, pattern)}


def _dark_master(paths, n_use, loader, sigma_clip=4.0):
    """
    Sigma-clipped mean master dark, in raw ADU (never calibrated).

    Calibrating dark frames first would be wrong twice over: calibrate_frame
    subtracts the black level (which is most of what a dark frame IS) and then
    clips at zero, destroying the below-black half of the read-noise and DSNU
    distribution and biasing the master dark upward.

    Sigma-clipped rather than trimmed: both reject outliers (a cosmic-ray hit
    or a dropped frame would otherwise print itself into every corrected frame),
    but a trimmed mean must sort the whole stack, which needs all N frames
    resident. At 12 MP that is ~50 MB per frame, so a few hundred darks would
    need tens of GB of scratch disk. Clipping needs only two streaming passes --
    Welford for the mean and std, then a mean over pixels within sigma_clip of
    it -- so memory is flat and the frame count is unbounded, which matters
    because the master dark wants many more frames than is intuitive
    (see _dark_advice).

    Returns (master_adu, n_used).
    """
    n = min(n_use, len(paths)) if n_use else len(paths)

    mean = M2 = None
    for i in range(n):
        print(f"  dark pass1 [{i+1}/{n}] {paths[i].name}", end="\r", flush=True)
        x = loader(paths[i]).astype(np.float64)
        if mean is None:
            mean, M2 = x.copy(), np.zeros_like(x)
        else:
            d     = x - mean
            mean += d / (i + 1)
            M2   += d * (x - mean)
    print()
    std = np.sqrt(M2 / max(n - 1, 1))
    del M2

    # float32 bounds: these are only thresholds, and mean is recoverable as
    # their midpoint, so the full-precision mean array can be released here.
    lo = (mean - sigma_clip * std).astype(np.float32)
    hi = (mean + sigma_clip * std).astype(np.float32)
    del std, mean

    s = np.zeros(lo.shape, dtype=np.float64)
    c = np.zeros(lo.shape, dtype=np.int32)
    for i in range(n):
        print(f"  dark pass2 [{i+1}/{n}] {paths[i].name}", end="\r", flush=True)
        x = loader(paths[i]).astype(np.float32)
        m = (x >= lo) & (x <= hi)
        s += np.where(m, x, 0.0)
        c += m
    print()

    out = np.where(c > 0, s / np.maximum(c, 1), (lo + hi) / 2.0).astype(np.float32)
    rejected = float((n - c.mean()) / n * 100.0)
    print(f"  sigma-clip (±{sigma_clip}σ) rejected {rejected:.3f}% of samples")
    return out, n


def _dark_advice(sigma1_cal, master_hp, residual_cal, n_used):
    """
    Report how many dark frames the master actually wants.

    The master removes a static pattern of amplitude D but adds its own
    residual sigma1/sqrt(N_dark). Subtraction only breaks even once that
    residual drops below D, so:

        N_break_even = (sigma1 / D)^2          residual == D
        N_recommended = 9 * N_break_even       residual == D/3, a clear win

    D is recovered from the master's own high-pass std, which contains the
    static pattern and the residual in quadrature: D = sqrt(hp^2 - residual^2).
    """
    d_sq = master_hp ** 2 - residual_cal ** 2
    print(f"    single-frame dark sigma      : {sigma1_cal:.6f}")
    print(f"    master dark residual         : {residual_cal:.6f} "
          f"(from {n_used} frames)")
    if d_sq <= 0:
        print("    static dark pattern          : not resolvable — the master "
              "is still dominated by its own noise.")
        print("    -> far too few dark frames; subtraction will add noise, "
              "not remove it.")
        return
    D = np.sqrt(d_sq)
    n_break = (sigma1_cal / D) ** 2
    print(f"    static dark pattern (DSNU)   : {D:.6f}")
    print(f"    break-even N_dark            : {n_break:.0f}  "
          f"(residual == pattern; no net gain)")
    print(f"    recommended N_dark           : {9 * n_break:.0f}  "
          f"(residual == pattern/3)")
    print(f"    for residual == pattern/5    : {25 * n_break:.0f}")
    if n_used < n_break:
        print("    -> you have too few; the subtraction is making things worse.")
    elif n_used < 9 * n_break:
        print("    -> usable, but more darks would still help noticeably.")
    else:
        print("    -> comfortably enough.")


def _dark_master_residual(paths, n_used, loader, black, white):
    """
    Estimate the master dark's OWN leftover noise, in calibrated units.

    Subtracting a master dark removes the static dark pattern but adds whatever
    random noise the master still carries -- so a master built from too few
    frames injects more noise than the pattern it removes and makes the result
    worse. This is the number to weigh against that pattern's amplitude.

    Single-frame dark sigma comes from the difference of two frames (the static
    pattern cancels, leaving sqrt(2) times the per-frame noise); the master's
    residual is then that divided by sqrt(n_used).
    """
    if len(paths) < 2:
        return None
    a = loader(paths[0]).astype(np.float64)
    b = loader(paths[1]).astype(np.float64)
    sigma1 = float((a - b).std()) / np.sqrt(2.0)
    return sigma1 / np.sqrt(n_used) / float(white - black[0])


def _defect_map(dark_adu, pattern, resid_adu, n_sigma):
    """
    Locate defective pixels in the master dark.

    A dark frame has no scene, so a pixel far from its same-colour neighbours
    there is a sensor defect and nothing else. That is what makes this test
    reliable where the same test on a light frame is not: on a light frame a
    hot pixel and a genuine single-pixel highlight are spatially identical, and
    no threshold separates them.

    Sensitivity is set by the master's own residual noise, sigma1/sqrt(N_dark),
    not by the per-frame noise -- which is the whole reason to detect here
    rather than per frame. With a few hundred darks that is an order of
    magnitude finer, and it reaches the mildly-hot pixels that dominate by
    count and are invisible in any single frame.

    Returns (mask, n_hot, n_cold). Cold (stuck-low) pixels are flagged too;
    they are equally wrong in the output and cost nothing extra to find.
    """
    med  = bayer_plane_median3(dark_adu, pattern)
    dev  = dark_adu - med
    thr  = n_sigma * resid_adu
    hot  = dev > thr
    cold = dev < -thr
    return (hot | cold), int(hot.sum()), int(cold.sum())


def _interpolate_defects(frame, pattern, mask):
    """Replace flagged pixels with the median of their same-colour neighbours."""
    if not mask.any():
        return frame
    med = bayer_plane_median3(frame, pattern)
    return np.where(mask, med, frame).astype(np.float32)


def _plot_defect_map(mask, n_hot, n_cold, out):
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.imshow(mask, cmap="gray", interpolation="nearest")
    frac = mask.mean() * 100
    ax.set_title(f"Defect map from master dark — {n_hot} hot, {n_cold} cold "
                 f"({frac:.4f}% of pixels)", fontsize=11)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def _dark_correct(light_adu, dark_adu, pattern, black, white):
    """
    (light - dark) / (white - black), per Bayer channel.

    The black level is present in BOTH terms and cancels in the subtraction, so
    it must not be subtracted a second time -- which is exactly what feeding the
    difference through calibrate_frame would do, pushing the whole frame one
    black level too low. Hence the explicit per-channel scaling here rather than
    reusing calibrate_frame.
    """
    out = np.empty_like(light_adu, dtype=np.float32)
    for r in range(2):
        for c in range(2):
            ch = int(pattern[r, c])
            out[r::2, c::2] = ((light_adu[r::2, c::2] - dark_adu[r::2, c::2])
                               / (white - black[ch]))
    return np.clip(out, 0.0, 1.0)


def _stream_welford(paths, pattern, black, white, loader):
    """
    Pass 2 — Welford online algorithm for per-pixel temporal std.
    Returns temporal_std : (H, W) float32  per-pixel noise standard deviation.
    """
    n = len(paths)
    wf_mean = None
    wf_M2   = None

    for i, p in enumerate(paths):
        idx = i + 1
        print(f"  Welford  [{idx}/{n}] {p.name}", end="\r", flush=True)
        raw = loader(p).astype(np.float64)
        cal = calibrate_frame(raw.astype(np.float32), pattern, black, white).astype(np.float64)

        if wf_mean is None:
            wf_mean = cal.copy()
            wf_M2   = np.zeros_like(cal)
        else:
            delta   = cal - wf_mean
            wf_mean += delta / idx
            wf_M2   += delta * (cal - wf_mean)

    print()
    var = np.where(n > 1, wf_M2 / (n - 1), 0.0).astype(np.float32)
    return np.sqrt(var)


def _welford_update(wf_mean, wf_M2, wf_n, key, x):
    wf_n[key] += 1
    if wf_mean[key] is None:
        wf_mean[key] = x.copy()
        wf_M2[key]   = np.zeros_like(x)
    else:
        delta = x - wf_mean[key]
        wf_mean[key] += delta / wf_n[key]
        wf_M2[key]   += delta * (x - wf_mean[key])


def _stream_block_std(paths, pattern, black, white, loader,
                      roi_frac=0.25, n_levels=8):
    """
    Unbiased noise-vs-N-averaged check.

    For several block sizes N (log-spaced, plus N=1), split the sequence into
    disjoint (non-overlapping) blocks of N consecutive frames, average within
    each block, then measure the std ACROSS those independent block means.
    Since blocks never share frames, this has no self-referencing bias --
    unlike comparing a running mean against a "ground truth" that contains
    the same frames (which forces the error toward zero as N approaches the
    total frame count, regardless of actual noise behavior).

    Computed over a central ROI crop (not the full frame) to keep the
    per-block-size accumulators cheap; std should scale ~1/√N if per-frame
    noise behaves independently across frames.

    Also reports a shot-noise-normalized curve, mean(std / sqrt(mu)) per N,
    using the per-pixel mean mu from the N=1 pass as a fixed reference. Plain
    std mixes together pixels at very different signal levels -- under shot
    noise, sigma ~ sqrt(mu), so a bright pixel and a dark pixel have genuinely
    different noise scales, and averaging their raw stds together produces a
    number that's a scene/ROI-dependent blend, not a value comparable across
    scenes. Dividing each pixel by sqrt(its own mu) first removes that
    signal-level dependence (mirrors gt_noise_shot_norm.png's convention:
    ~1 = pure Poisson) so the reported number means the same thing regardless
    of what's in the ROI. Both curves still scale as 1/√N in N, since sqrt(N)
    factors out of the spatial average either way -- normalizing changes the
    y-axis's meaning, not the shape being tested.

    Returns a list of (N, mean_std, mean_shot_norm, n_blocks) tuples, N=1 first.
    """
    n = len(paths)
    max_n = n // 4
    if max_n < 2:
        print(f"  Skipping block-std convergence: need >= 8 frames, have {n}")
        return []

    candidate_ns = sorted(set(np.geomspace(2, max_n, n_levels).astype(int).tolist()))
    all_ns = [1] + candidate_ns

    # Central ROI, cropped at even offsets so the Bayer phase (which absolute
    # position is R/G/B) matches what calibrate_frame expects.
    first = loader(paths[0])
    H, W  = first.shape
    rh, rw = max(2, int(H * roi_frac)) & ~1, max(2, int(W * roi_frac)) & ~1
    r0, c0 = ((H - rh) // 2) & ~1, ((W - rw) // 2) & ~1
    def roi(f):
        return f[r0:r0 + rh, c0:c0 + rw]

    wf_mean     = {N: None for N in all_ns}
    wf_M2       = {N: None for N in all_ns}
    wf_n        = {N: 0 for N in all_ns}
    block_accum = {N: None for N in candidate_ns}
    block_count = {N: 0 for N in candidate_ns}

    for i, p in enumerate(paths):
        print(f"  block-std  [{i+1}/{n}] {p.name}", end="\r", flush=True)
        raw = roi(loader(p)).astype(np.float64)
        cal = calibrate_frame(raw.astype(np.float32), pattern, black, white).astype(np.float64)

        _welford_update(wf_mean, wf_M2, wf_n, 1, cal)

        for N in candidate_ns:
            block_accum[N] = raw if block_accum[N] is None else block_accum[N] + raw
            block_count[N] += 1
            if block_count[N] == N:
                block_mean_adu = (block_accum[N] / N).astype(np.float32)
                block_mean_cal = calibrate_frame(block_mean_adu, pattern, black, white).astype(np.float64)
                _welford_update(wf_mean, wf_M2, wf_n, N, block_mean_cal)
                block_accum[N] = None
                block_count[N] = 0
    print()

    mu_map = wf_mean[1]           # per-pixel signal level, from the N=1 pass
    mask   = mu_map > 1e-4

    results = []
    for N in all_ns:
        if wf_n[N] > 1:
            std_map = np.sqrt(wf_M2[N] / (wf_n[N] - 1))
            shot_norm_map = np.where(mask, std_map / np.sqrt(np.where(mask, mu_map, 1.0)), np.nan)
            results.append((N, float(std_map.mean()), float(np.nanmean(shot_norm_map)), wf_n[N]))
    return results


def _compute_median_and_trimmed(paths, n_stack, pattern, black, white, loader,
                                trim_frac, tmp_path):
    """
    Compute per-pixel median and trimmed mean without loading the full stack
    into RAM.  Strategy:
      1. Write calibrated frames one-by-one to a disk-backed (N, H, W) memmap.
      2. Process the memmap in horizontal strips; each strip fits comfortably
         in RAM.  Peak extra RAM ≈ one strip = ~200 MB.
      3. Delete the temporary file on exit.
    """
    n = min(n_stack, len(paths))

    # Prime with first frame to get shape
    first = calibrate_frame(loader(paths[0]), pattern, black, white)
    H, W  = first.shape

    # Write all frames to the memmap
    mm = np.lib.format.open_memmap(str(tmp_path), mode='w+',
                                   dtype=np.float32, shape=(n, H, W))
    mm[0] = first
    del first
    for i in range(1, n):
        print(f"  stack [{i+1}/{n}] {paths[i].name}", end="\r", flush=True)
        mm[i] = calibrate_frame(loader(paths[i]), pattern, black, white)
    mm.flush()
    print()

    trim_k      = max(1, int(trim_frac / 2 * n))
    median_out  = np.empty((H, W), dtype=np.float32)
    trimmed_out = np.empty((H, W), dtype=np.float32)

    # Strip height that keeps each in-RAM strip ≈ 200 MB
    strip_h = max(1, int(200 * 1024 ** 2 // (n * W * 4)))

    for r0 in range(0, H, strip_h):
        r1    = min(r0 + strip_h, H)
        strip = mm[:, r0:r1, :].copy()     # (n, strip_h, W) — the only big alloc
        srt   = np.sort(strip, axis=0)
        del strip

        if n % 2 == 1:
            median_out[r0:r1] = srt[n // 2]
        else:
            median_out[r0:r1] = (srt[n // 2 - 1] + srt[n // 2]) / 2

        trimmed_out[r0:r1] = srt[trim_k : n - trim_k].mean(axis=0)
        del srt

    del mm
    tmp_path.unlink(missing_ok=True)

    return median_out, trimmed_out


# --------------------------------------------------------------------------- #
# Plots                                                                         #
# --------------------------------------------------------------------------- #

def _save_rgb_frame(bayer: np.ndarray, pattern: np.ndarray, out: Path,
                    wb: np.ndarray | None = None, ccm: np.ndarray | None = None) -> None:
    """Demosaic a calibrated Bayer frame and save it as a full-resolution RGB PNG."""
    rgb = demosaic_to_rgb(bayer, pattern, wb, ccm)
    save_rgb_png(rgb, out)


def _plot_halfdiff_crops(crops, out, crop_size, vmax):
    """
    Split-half difference at each checkpoint: a picture of temporal noise only.

    The scene and any fixed-pattern noise are identical in the two halves and
    cancel in the subtraction, so unlike the running-mean crops -- which sit on
    a static floor that averaging cannot touch -- these shrink by the full
    1/sqrt(N) and the improvement is plainly visible.

    All panels share one colour scale, fixed from the first (noisiest)
    checkpoint. Per-panel autoscaling would renormalise each image to its own
    range and hide exactly the shrinkage this plot exists to show.
    """
    n_panels = len(crops)
    fig, axes = plt.subplots(1, n_panels, figsize=(4.2 * n_panels, 5.0))
    if n_panels == 1:
        axes = [axes]
    im = None
    for ax, (idx, crop, sd) in zip(axes, crops):
        im = ax.imshow(crop, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                       interpolation="nearest")
        ax.set_title(f"N = {idx}\nstd = {sd:.6f}", fontsize=11)
        ax.axis("off")
    if im is not None:
        fig.colorbar(im, ax=axes, fraction=0.02, pad=0.02,
                     label="split-half difference [calibrated]")
    fig.suptitle(
        f"Temporal noise only — split-half difference, {crop_size}×{crop_size} "
        f"centre crop, shared colour scale (scene and FPN cancel)",
        fontsize=12, y=1.02)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def _stills_reference(directory, shape, match_to=None):
    """
    Mean of a directory of stills, calibrated with the stills' OWN metadata.

    These are a separate capture at a different gain and exposure, so their
    black and white levels need not match the sequence's -- calibrating them
    with the sequence's values would apply the wrong pedestal and scale.

    If match_to is given, the result is rescaled by the ratio of medians so an
    exposure mismatch between the two capture settings does not dominate every
    difference map. The ratio is taken over pixels above a low threshold, since
    near-black pixels give an unstable ratio.
    """
    fmt, paths, pattern, black, white, loader, _ = _make_loaders(directory)
    accum = None
    n = len(paths)
    for i, p in enumerate(paths):
        print(f"  stills [{i+1}/{n}] {p.name}", end="\r", flush=True)
        raw   = loader(p).astype(np.float64)
        accum = raw if accum is None else accum + raw
    print()

    cal = calibrate_frame((accum / n).astype(np.float32), pattern, black, white)
    if cal.shape != shape:
        sys.exit(f"Stills are {cal.shape}, sequence is {shape} -- "
                 f"cannot compare different resolutions.")

    scale = 1.0
    if match_to is not None:
        m = (match_to > 0.01) & (cal > 0.01)
        if m.sum() > 1000:
            scale = float(np.median(match_to[m]) / np.median(cal[m]))
            cal = np.clip(cal * scale, 0.0, 1.0)
    return cal, n, scale


def _plot_comparison(frames, labels, pattern, out, wb=None, ccm=None):
    """Side-by-side RGB of the GT candidates."""
    fig, axes = plt.subplots(1, len(frames), figsize=(6.2 * len(frames), 5.4))
    if len(frames) == 1:
        axes = [axes]
    for ax, f, lab in zip(axes, frames, labels):
        ax.imshow(demosaic_to_rgb(f, pattern, wb, ccm))
        ax.set_title(lab, fontsize=10)
        ax.axis("off")
    fig.suptitle("GT candidates", fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def _plot_comparison_crops(frames, labels, pattern, out, crop, wb=None, ccm=None):
    """100%-zoom centre crops -- downscaling would hide the pixel-level
    differences these candidates are being compared on."""
    fig, axes = plt.subplots(1, len(frames), figsize=(4.6 * len(frames), 5.2))
    if len(frames) == 1:
        axes = [axes]
    for ax, f, lab in zip(axes, frames, labels):
        rgb = demosaic_to_rgb(f, pattern, wb, ccm)
        cy, cx = rgb.shape[0] // 2, rgb.shape[1] // 2
        h = min(crop, rgb.shape[0]) // 2
        w = min(crop, rgb.shape[1]) // 2
        ax.imshow(rgb[cy - h: cy + h, cx - w: cx + w], interpolation="nearest")
        ax.set_title(lab, fontsize=10)
        ax.axis("off")
    fig.suptitle(f"GT candidates — {crop}×{crop} centre crop at 100% zoom",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def _plot_comparison_diffs(pairs, out):
    """Signed difference maps between GT candidates, symmetric colour scales."""
    fig, axes = plt.subplots(1, len(pairs), figsize=(6.2 * len(pairs), 5.2))
    if len(pairs) == 1:
        axes = [axes]
    for ax, (diff, lab) in zip(axes, pairs):
        v = max(float(np.percentile(np.abs(diff), 99.5)), 1e-9)
        im = ax.imshow(diff, cmap="RdBu_r", vmin=-v, vmax=v)
        ax.set_title(f"{lab}\nstd = {diff.std():.6f}", fontsize=10)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    fig.suptitle("Differences between GT candidates", fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def _report_floor(metrics, margins=(3.0, 5.0)):
    """
    Derive the static floor and the knee from the checkpoint measurements.

    Model: highpass(N)^2 = sigma1^2 / N + floor^2, where sigma1 is single-frame
    temporal noise and floor^2 collects everything averaging cannot remove
    (fixed-pattern noise, plus any fine scene detail the high-pass keeps --
    the metric cannot separate those two, see plot_highpass_steps.py).

    At N_knee the two terms are equal and highpass sits sqrt(2) above the
    floor; it is within 10% of the floor at ~4.8*N_knee and 5% at ~9.8*N_knee.

    Note the knee is NOT the point at which averaging stops improving a GT
    frame. The floor is present in the GT and in the frames it will be compared
    against, so it cancels out of the GT's error, which stays purely temporal
    and keeps falling as 1/sqrt(N). The knee marks where the frame stops
    LOOKING cleaner, not where it stops BEING more accurate.
    """
    usable = [m for m in metrics if np.isfinite(m['temporal'])]
    if not usable:
        return
    last     = usable[-1]
    sigma1   = last['temporal'] * np.sqrt(last['n'])
    floor_sq = last['highpass'] ** 2 - last['temporal'] ** 2

    print("  Static floor")
    print(f"    single-frame temporal sigma1 : {sigma1:.6f}")
    if floor_sq <= 0:
        print("    static floor                 : not resolvable "
              "(high-pass is at or below the temporal estimate — still "
              "temporal-noise limited at this N)")
        print()
        return
    floor = np.sqrt(floor_sq)
    knee  = (sigma1 / floor) ** 2
    print(f"    static floor (FPN + detail)  : {floor:.6f}")
    print(f"    knee N = (sigma1/floor)^2    : {knee:.0f}"
          f"   (highpass is 1.41x the floor here)")
    print(f"    N for highpass within 10%    : {4.76 * knee:.0f}")
    print(f"    frames used                  : {last['n']}"
          f"   ({'past' if last['n'] >= knee else 'below'} the knee)")
    print("    temporal noise below floor   : "
          + ",  ".join(f"{m:.0f}x -> N={m ** 2 * knee:.0f}" for m in margins))
    print("    (for a GT frame the floor cancels -- size N against your "
          "denoiser's residual, not against the knee)")
    print()


def _plot_checkpoint_noise(metrics, out):
    """
    Measured noise vs frames averaged, with a 1/sqrt(N) reference.

    Two curves, because they answer different questions: the split-half
    `temporal` curve says whether averaging is still removing noise, while
    `highpass` says whether that is still visible in the frame. They diverge
    once fixed-pattern noise dominates -- which is exactly when more frames
    stop making a visible difference even though the averaging still works.
    """
    ns   = np.array([m['n'] for m in metrics], dtype=float)
    temp = np.array([m['temporal'] for m in metrics], dtype=float)
    hp   = np.array([m['highpass'] for m in metrics], dtype=float)

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ok = np.isfinite(temp)
    ax.loglog(ns[ok], temp[ok], "o-", color="steelblue", linewidth=1.8,
              markersize=5, label="temporal noise (split-half, unbiased)")
    ax.loglog(ns, hp, "s-", color="darkorange", linewidth=1.8,
              markersize=5, label="residual pixel noise (high-pass, incl. FPN)")

    if ok.sum() >= 1:
        n0, t0 = ns[ok][0], temp[ok][0]
        ax.loglog(ns, t0 * np.sqrt(n0) / np.sqrt(ns), "--", color="gray",
                  linewidth=1.3, label=r"ideal $\propto 1/\sqrt{N}$")

    ax.set_xlabel("Frames averaged  (N)", fontsize=11)
    ax.set_ylabel("Noise  [calibrated units]", fontsize=11)
    ax.set_title("Does averaging still help?  Measured noise vs N", fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, which="both", alpha=0.3, linestyle="--")
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def _print_checkpoint_table(metrics):
    """Print the per-checkpoint noise numbers, with observed-vs-ideal ratios."""
    print("\n  Noise vs frames averaged")
    print(f"  {'N':>6}  {'temporal':>11}  {'vs N=1':>8}  {'ideal':>8}  "
          f"{'highpass':>11}  {'vs N=1':>8}")
    print("  " + "-" * 62)
    base_t = next((m['temporal'] for m in metrics
                   if np.isfinite(m['temporal'])), None)
    base_n = next((m['n'] for m in metrics
                   if np.isfinite(m['temporal'])), None)
    base_h = metrics[0]['highpass'] if metrics else None
    for m in metrics:
        t, h, nn = m['temporal'], m['highpass'], m['n']
        t_rat = f"{t / base_t:8.3f}" if (base_t and np.isfinite(t)) else f"{'--':>8}"
        ideal = f"{np.sqrt(base_n / nn):8.3f}" if base_n else f"{'--':>8}"
        t_str = f"{t:11.6f}" if np.isfinite(t) else f"{'--':>11}"
        print(f"  {nn:6d}  {t_str}  {t_rat}  {ideal}  "
              f"{h:11.6f}  {h / base_h:8.3f}")
    print()
    _report_floor(metrics)


def _plot_checkpoint_crops(crops, out, crop_size):
    """
    Side-by-side 100%-zoom crops of the running mean at increasing N.

    Shown at native pixel scale on purpose: any downscaling averages
    neighbouring pixels and hides exactly the per-pixel noise this plot exists
    to show, which would make every panel look equally clean regardless of N.
    """
    n_panels = len(crops)
    fig, axes = plt.subplots(1, n_panels, figsize=(4.2 * n_panels, 4.8))
    if n_panels == 1:
        axes = [axes]
    for ax, (idx, crop) in zip(axes, crops):
        ax.imshow(crop, interpolation="nearest")
        ax.set_title(f"N = {idx}", fontsize=11)
        ax.axis("off")
    fig.suptitle(
        f"Running mean vs frames averaged — {crop_size}×{crop_size} centre crop "
        f"at 100% zoom (noise should fall as 1/√N)",
        fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def _plot_aggregated(mean, median, trimmed, pattern, n_stack, out,
                     wb: np.ndarray | None = None, ccm: np.ndarray | None = None):
    frames = [mean, median, trimmed]
    titles = [
        "Mean (all frames)",
        f"Median  (N={n_stack})",
        f"Trimmed mean  (N={n_stack}, trim={TRIM_FRAC:.0%})",
    ]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, f, t in zip(axes, frames, titles):
        ax.imshow(demosaic_to_rgb(f, pattern, wb, ccm), aspect="auto")
        ax.set_title(t, fontsize=10)
        ax.axis("off")
    fig.suptitle("GT aggregation methods", fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def _plot_differences(mean, median, trimmed, out):
    pairs = [
        (mean - median,    "mean − median"),
        (mean - trimmed,   "mean − trimmed"),
        (median - trimmed, "median − trimmed"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, (diff, title) in zip(axes, pairs):
        vmax = max(float(np.percentile(np.abs(diff), 99.5)), 1e-6)
        im = ax.imshow(diff, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_title(title, fontsize=10)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    fig.suptitle("Method differences — where aggregators disagree (outlier / hot-pixel locations)",
                 fontsize=12, y=1.01)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def _plot_temporal_noise(temporal_std, out):
    fig, ax = plt.subplots(figsize=(9, 6))
    vmax = float(np.percentile(temporal_std, 99))
    im = ax.imshow(temporal_std, cmap="inferno", vmin=0, vmax=vmax, aspect="auto")
    ax.set_title("Per-pixel temporal std  (noise map, calibrated units)", fontsize=11)
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02, label="σ [calibrated]")
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def _plot_noise_cv(temporal_std, full_mean, out):
    """σ/μ — coefficient of variation; removes signal-level dependence."""
    mask = full_mean > 1e-4
    cv = np.where(mask, temporal_std / np.where(mask, full_mean, 1.0), np.nan)
    fig, ax = plt.subplots(figsize=(9, 6))
    vmax = float(np.nanpercentile(cv, 99))
    im = ax.imshow(cv, cmap="inferno", vmin=0, vmax=vmax, aspect="auto")
    ax.set_title("Coefficient of variation  σ/μ  (relative noise per pixel)", fontsize=11)
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02, label="σ/μ")
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def _plot_noise_shot_norm(temporal_std, full_mean, out):
    """σ/√μ — shot-noise-normalized std; equals ~1 for pure Poisson noise."""
    mask = full_mean > 1e-4
    shot_norm = np.where(mask, temporal_std / np.where(mask, np.sqrt(full_mean), 1.0), np.nan)
    fig, ax = plt.subplots(figsize=(9, 6))
    vmax = float(np.nanpercentile(shot_norm, 99))
    im = ax.imshow(shot_norm, cmap="inferno", vmin=0, vmax=vmax, aspect="auto")
    ax.set_title("Shot-noise-normalized std  σ/√μ  (1 = pure shot noise)", fontsize=11)
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02, label="σ/√μ")
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def _plot_block_std_convergence(results, out):
    """
    results: list of (N, mean_std, mean_shot_norm, n_blocks) from
    _stream_block_std. Each reference line is anchored to its own N=1 point
    (an independent measurement, not a fit through a possibly-lucky sample).

    Two panels, same N-dependence, different y-axis meaning:
      left  - plain std, mean(sigma) over the ROI: mixes pixels at different
              signal levels, so the absolute number is a scene/ROI-dependent
              blend, not comparable across scenes.
      right - shot-noise-normalized, mean(sigma/sqrt(mu)) over the ROI: each
              pixel divided by its own sqrt(signal) first, so brightness
              differences across pixels don't distort the blend; ~1 = pure
              Poisson, comparable across scenes (same convention as
              gt_noise_shot_norm.png).
    Both should still trace ~1/√N -- normalizing changes what the y-axis
    means, not the N-scaling being tested (sqrt(N) factors out of the
    spatial average the same way in either case).

    Each point gets an error bar from the standard error of a sample std
    estimated from M samples (M = n_blocks at that N): relative SE ≈
    1/√(2(M−1)). Fewer blocks (large N) means a much noisier estimate of
    that point -- e.g. M=256 -> ~4%, M=4 -> ~41% -- so the rightmost points
    are the least trustworthy on the plot, which the error bars make visible
    directly instead of something to keep in mind separately.
    """
    if not results:
        print(f"  Skipping {out.name}: no block-std results")
        return
    ns        = np.array([r[0] for r in results], dtype=float)
    std       = np.array([r[1] for r in results], dtype=float)
    shot_norm = np.array([r[2] for r in results], dtype=float)
    n_blocks  = np.array([r[3] for r in results], dtype=float)
    rel_se    = 1.0 / np.sqrt(2.0 * np.maximum(n_blocks - 1, 1e-9))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    panels = [
        (axes[0], std,       "Std across independent block means  [calibrated]",
         "Noise vs. N (disjoint blocks)"),
        (axes[1], shot_norm, r"Shot-noise-normalized  mean(σ/√μ)  (1 = pure Poisson)",
         "Shot-noise-normalized noise vs. N"),
    ]
    for ax, y, ylabel, title in panels:
        ax.errorbar(ns, y, yerr=y * rel_se, fmt="o-", color="steelblue",
                    linewidth=1.8, markersize=5, capsize=3,
                    ecolor="steelblue", alpha=0.9,
                    label="measured ± SE of the std estimate (across disjoint N-frame blocks)")
        ref = y[0] / np.sqrt(ns)
        ax.plot(ns, ref, "--", color="tomato", linewidth=1.4,
               label=r"$\propto 1/\sqrt{N}$ (anchored to N=1)")
        ax.set_xscale("log")
        ax.set_yscale("log")
        for x, yy, m in zip(ns, y, n_blocks.astype(int)):
            ax.annotate(f"{m} blocks", (x, yy), textcoords="offset points",
                        xytext=(4, 4), fontsize=7, color="gray")
        ax.set_xlabel("Frames averaged per block  (N)", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(title, fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(True, which="both", alpha=0.3, linestyle="--")
    fig.suptitle(f"Block-std convergence check  (N = 1 … {int(ns.max())})", fontsize=13)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved {out}")


# --------------------------------------------------------------------------- #
# Main analysis function                                                         #
# --------------------------------------------------------------------------- #

def analyze_gt_sequence(
    directory: str,
    out_dir: Path,
    max_frames: int | None = None,
    max_stack: int = 60,
) -> None:
    """
    Analyze a static lowlight sequence for GT frame generation.
    Saves outputs to out_dir/<sequence_name>/ (see module docstring).
    """
    seq_name = Path(directory).name
    seq_out  = out_dir / seq_name
    seq_out.mkdir(parents=True, exist_ok=True)

    fmt, paths, pattern, black, white, loader, sample_fn = _make_loaders(directory)
    if max_frames is not None:
        paths = paths[:max_frames]
    n = len(paths)
    print(f"  {n} {fmt.upper()} frames  |  white={white}  black={black[0]}")

    # DNG carries a real camera white-balance + color-correction profile;
    # GN3 has none, so demosaic_to_rgb falls back to gray-world WB / no CCM.
    wb, ccm = (get_color_metadata(paths[0]) if fmt == 'dng' else (None, None))

    # Sample frames
    sample_idx = np.linspace(0, n - 1, min(N_SAMPLES, n), dtype=int)
    sample_frames = [sample_fn(paths[i]) for i in sample_idx]
    plot_sample_frames(
        sample_frames,
        [f"frame {i}" for i in sample_idx],
        seq_out / f"gt_sample_frames_N{n}.png",
    )
    # Individual full-resolution PNGs (no matplotlib tiling/downsampling) so
    # sample frames can be pixel-inspected directly, e.g. to check whether an
    # apparent grid/moire artifact is real or an artifact of the composite
    # plot above being resampled to fit multiple panels in one figure.
    for i, frame in zip(sample_idx, sample_frames):
        save_rgb_png(frame, seq_out / f"gt_sample_frame_{i:04d}_N{n}_full.png")

    # Pass 1: full streaming mean, snapshotting the running mean along the way.
    # Checkpoints are log-spaced because noise falls as 1/sqrt(N) -- linear
    # spacing would put every panel in the flat tail and show no visible change.
    ckpt_ns = sorted(set(np.geomspace(1, n, min(N_CHECKPOINTS, n))
                         .astype(int).tolist()) | {n})
    ckpt_crops, half_crops = [], []
    half_vmax = []            # one-element cell: shared scale, set on first diff

    def _crop_centre(a, size):
        cy, cx = a.shape[0] // 2, a.shape[1] // 2
        h = min(size, a.shape[0]) // 2
        w = min(size, a.shape[1]) // 2
        return a[cy - h: cy + h, cx - w: cx + w].copy()

    def _on_checkpoint(idx, running, half_diff):
        # Demosaic once and reuse for both the full-res save and the crop --
        # demosaicing a full-size frame is the expensive part of this callback.
        rgb = demosaic_to_rgb(running, pattern, wb, ccm)
        save_rgb_png(rgb, seq_out / f"gt_running_mean_N{idx:05d}.png")
        ckpt_crops.append((idx, _crop_centre(rgb, CHECKPOINT_CROP)))
        if half_diff is None:
            return
        # Fix the display scale from the first (noisiest) checkpoint and keep
        # it for all later ones, so the panels stay directly comparable.
        if not half_vmax:
            half_vmax.append(max(float(np.percentile(np.abs(half_diff), 99.5)),
                                 1e-9))
        half_crops.append((idx, _crop_centre(half_diff, CHECKPOINT_CROP),
                           float(half_diff.std())))

    print(f"  Pass 1 — streaming mean ({n} frames), "
          f"checkpoints at N={ckpt_ns} …")
    full_mean, full_mean_adu, ckpt_metrics = _stream_mean(
        paths, pattern, black, white, loader,
        checkpoints=set(ckpt_ns), on_checkpoint=_on_checkpoint)
    print(f"  Mean range: [{full_mean.min():.4f}, {full_mean.max():.4f}]")
    _plot_checkpoint_crops(ckpt_crops,
                           seq_out / f"gt_running_mean_comparison_N{n}.png",
                           CHECKPOINT_CROP)
    if half_crops:
        _plot_halfdiff_crops(half_crops,
                             seq_out / f"gt_halfdiff_comparison_N{n}.png",
                             CHECKPOINT_CROP, half_vmax[0])
    _print_checkpoint_table(ckpt_metrics)
    _plot_checkpoint_noise(ckpt_metrics,
                           seq_out / f"gt_checkpoint_noise_N{n}.png")

    # Pass 2: Welford temporal std
    print(f"  Pass 2 — temporal std …")
    temporal_std = _stream_welford(paths, pattern, black, white, loader)
    print(f"  Temporal std  mean={temporal_std.mean():.5f}  max={temporal_std.max():.5f}")

    # Pass 3: block-std convergence (disjoint N-frame blocks, unbiased)
    print(f"  Pass 3 — block-std convergence …")
    block_std_results = _stream_block_std(
        paths, pattern, black, white, loader,
        roi_frac=CONV_ROI_FRAC, n_levels=CONV_N_LEVELS,
    )

    # Median + trimmed mean (disk-backed, no full-stack RAM alloc)
    n_stack  = min(max_stack, n)
    tmp_path = seq_out / "_stack_tmp.npy"
    trim_k   = max(1, int(TRIM_FRAC / 2 * n_stack))
    print(f"  Writing {n_stack}/{n} frames to disk, then computing "
          f"median and trimmed mean (trim_k={trim_k}) …")
    median_frame, trimmed_frame = _compute_median_and_trimmed(
        paths, n_stack, pattern, black, white, loader, TRIM_FRAC, tmp_path,
    )

    # Plots
    print("  Plotting …")
    _plot_aggregated(full_mean, median_frame, trimmed_frame, pattern, n_stack,
                     seq_out / f"gt_aggregated_N{n}_stack{n_stack}.png", wb, ccm)
    _plot_differences(full_mean, median_frame, trimmed_frame,
                      seq_out / f"gt_differences_N{n}_stack{n_stack}.png")
    _plot_temporal_noise(temporal_std, seq_out / f"gt_temporal_noise_N{n}.png")
    _plot_noise_cv(temporal_std, full_mean,       seq_out / f"gt_noise_cv_N{n}.png")
    _plot_noise_shot_norm(temporal_std, full_mean, seq_out / f"gt_noise_shot_norm_N{n}.png")
    _plot_block_std_convergence(block_std_results, seq_out / f"gt_convergence_N{n}.png")

    # Full-resolution RGB saves (one file per aggregation method)
    print("  Saving full-resolution RGB frames …")
    _save_rgb_frame(full_mean,     pattern, seq_out / f"gt_mean_rgb_N{n}.png",              wb, ccm)
    _save_rgb_frame(median_frame,  pattern, seq_out / f"gt_median_rgb_N{n_stack}.png",      wb, ccm)
    _save_rgb_frame(trimmed_frame, pattern, seq_out / f"gt_trimmed_mean_rgb_N{n_stack}.png", wb, ccm)

    # ---------------------------------------------------------------- #
    # Dark subtraction                                                   #
    # ---------------------------------------------------------------- #
    dark_corrected = None
    if DARK_DIR:
        print(f"\n  Dark frames: {DARK_DIR}")
        _, d_paths, d_pattern, _, _, d_loader, _ = _make_loaders(DARK_DIR)
        if not np.array_equal(d_pattern, pattern):
            sys.exit(f"Dark frames have Bayer pattern {d_pattern.tolist()}, "
                     f"sequence has {pattern.tolist()} -- not the same sensor "
                     f"layout, refusing to subtract.")
        n_dark_want = DARK_MAX_FRAMES or len(d_paths)
        print(f"  Sigma-clipped master dark from "
              f"{min(n_dark_want, len(d_paths))}/{len(d_paths)} frames …")
        dark_adu, d_stack = _dark_master(d_paths, n_dark_want, d_loader,
                                         DARK_SIGMA_CLIP)
        if dark_adu.shape != full_mean_adu.shape:
            sys.exit(f"Dark frames are {dark_adu.shape}, sequence is "
                     f"{full_mean_adu.shape} -- cannot subtract.")
        print(f"  Master dark ADU: mean={dark_adu.mean():.2f}  "
              f"min={dark_adu.min():.2f}  max={dark_adu.max():.2f}  "
              f"(black level {black[0]:.1f})")
        dark_corrected = _dark_correct(full_mean_adu, dark_adu, pattern, black, white)
        _save_rgb_frame(dark_corrected, pattern,
                        seq_out / f"gt_mean_darksub_rgb_N{n}.png", wb, ccm)

        hp_before = highpass_std(full_mean, pattern)
        hp_after  = highpass_std(dark_corrected, pattern)
        print(f"  High-pass std   mean={hp_before:.6f}  "
              f"dark-subtracted={hp_after:.6f}  "
              f"({(1 - hp_after / hp_before) * 100:+.1f}%)")

        resid = _dark_master_residual(d_paths, d_stack, d_loader, black, white)
        if resid is not None:
            # High-pass of the master needs no black subtraction -- a constant
            # pedestal has no high-frequency content, so scaling alone puts it
            # in calibrated units.
            master_hp = highpass_std(dark_adu / float(white - black[0]), pattern)
            print("\n  How many dark frames does this master want?")
            _dark_advice(resid * np.sqrt(d_stack), master_hp, resid, d_stack)
        if hp_after >= hp_before:
            print("  WARNING: dark subtraction made the frame NOISIER — prefer "
                  "the plain mean, or capture more darks.")

        # Defect map: detected in the master dark, repaired in the light frame.
        if HOT_PIXEL_SIGMA and resid is not None:
            resid_adu = resid * float(white - black[0])
            mask, n_hot, n_cold = _defect_map(dark_adu, pattern, resid_adu,
                                              HOT_PIXEL_SIGMA)
            print(f"\n  Defect map (>{HOT_PIXEL_SIGMA}σ from same-colour "
                  f"neighbours in the master dark): "
                  f"{n_hot} hot, {n_cold} cold, {mask.mean() * 100:.4f}% of pixels")
            if mask.any():
                defect_fixed = _interpolate_defects(dark_corrected, pattern, mask)
                hp_fixed = highpass_std(defect_fixed, pattern)
                print(f"  High-pass std   dark-subtracted={hp_after:.6f}  "
                      f"defects-interpolated={hp_fixed:.6f}  "
                      f"({(1 - hp_fixed / hp_after) * 100:+.1f}%)")
                _plot_defect_map(mask, n_hot, n_cold,
                                 seq_out / "gt_defect_map.png")
                np.save(seq_out / "gt_defect_map.npy", mask)
                _save_rgb_frame(defect_fixed, pattern,
                                seq_out / f"gt_mean_darksub_defectfix_rgb_N{n}.png",
                                wb, ccm)
                dark_corrected = defect_fixed

    # ---------------------------------------------------------------- #
    # Comparison of GT candidates                                        #
    # ---------------------------------------------------------------- #
    cands  = [(full_mean, f"Mean  (N={n})")]
    if dark_corrected is not None:
        cands.append((dark_corrected, f"Mean − master dark  (N={n})"))
    if STILLS_DIR:
        print(f"\n  Stills: {STILLS_DIR}")
        still, n_still, scale = _stills_reference(
            STILLS_DIR, full_mean.shape,
            match_to=full_mean if MATCH_STILL_INTENSITY else None)
        print(f"  {n_still} stills averaged"
              + (f", intensity-matched by ×{scale:.4f}" if MATCH_STILL_INTENSITY
                 else ", no intensity matching"))
        cands.append((still, f"Gain=1 stills  (N={n_still})"))

    if len(cands) > 1:
        cmp_dir = seq_out / "comparison"
        cmp_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n  Comparing {len(cands)} GT candidates …")

        frames, labels = [c[0] for c in cands], [c[1] for c in cands]
        _plot_comparison(frames, labels, pattern,
                         cmp_dir / "cmp_frames.png", wb, ccm)
        _plot_comparison_crops(frames, labels, pattern,
                               cmp_dir / "cmp_crops.png", COMPARISON_CROP, wb, ccm)
        _plot_comparison_diffs(
            [(frames[i] - frames[j], f"{labels[i]}  −  {labels[j]}")
             for i in range(len(frames)) for j in range(i + 1, len(frames))],
            cmp_dir / "cmp_differences.png")
        for f, lab in zip(frames, labels):
            stem = (lab.split('(')[0].strip()
                       .replace('−', 'minus').replace(' ', '_'))
            _save_rgb_frame(f, pattern, cmp_dir / f"cmp_{stem}.png", wb, ccm)

        print("\n  Residual pixel noise (high-pass std, calibrated units)")
        for f, lab in zip(frames, labels):
            print(f"    {lab:34s} {highpass_std(f, pattern):.6f}")
        print()

    print(f"\nDone. Outputs in {seq_out.resolve()}")


# --------------------------------------------------------------------------- #
# CLI overrides + entry point                                                   #
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


_UNSET = object()  # sentinel: distinguishes "user passed --foo none" from "flag not given"


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
            parser.add_argument(flag, dest=key, default=_UNSET,
                                action=argparse.BooleanOptionalAction,
                                help=f"(default: {val})")
        elif isinstance(val, int):
            parser.add_argument(flag, dest=key, type=int, default=_UNSET, metavar='N',
                                help=f"(default: {val})")
        elif isinstance(val, float):
            parser.add_argument(flag, dest=key, type=float, default=_UNSET, metavar='F',
                                help=f"(default: {val})")
        else:  # str or None
            parser.add_argument(flag, dest=key, type=_none_or_auto, default=_UNSET, metavar='S',
                                help=f"(default: {val!r}; pass 'none' to clear)")

    args = parser.parse_args()
    for key, new_val in vars(args).items():
        if new_val is not _UNSET:
            g[key] = new_val


def main():
    _apply_cli_overrides()
    print(f"GT sequence analysis: {SEQUENCE_DIR}")
    analyze_gt_sequence(
        SEQUENCE_DIR,
        Path(OUTPUT_DIR),
        max_frames=MAX_FRAMES,
        max_stack=MAX_STACK,
    )


if __name__ == "__main__":
    main()
