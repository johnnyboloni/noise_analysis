"""
GT frame aggregation analysis for a static lowlight sequence.

Edit the CONFIG block below, then run:
    python analyze_gt_sequence.py

Outputs (saved to OUTPUT_DIR/<sequence_name><RUN_SUFFIX>/):
  - run_info.json          : what produced this directory -- UTC timestamp, git
                             commit/branch/subject and whether the working tree
                             was dirty, plus the entire CONFIG block. A results
                             directory that cannot be traced back to a code
                             state and a set of settings is guesswork later.
  - gt_sample_frames.png   : evenly-spaced sample frames (RGB)
  - gt_aggregated.png      : mean / median / trimmed-mean side-by-side (RGB)
  - gt_differences.png     : (mean−median), (mean−trimmed), (median−trimmed) heatmaps
  - gt_mean_darksub_*      : mean frame minus a sigma-clipped master dark,
                             with a report of how many dark frames the master
                             actually wants (only when DARK_DIR is set)
  - gt_*_defectfix_*       : "defectfix" = pixels on the defect map rebuilt
                             from their same-colour neighbours (DEFECT_FILL
                             chooses median or directional). Produced both
                             with and without the dark subtraction, so the two
                             corrections can be judged separately. Every
                             repaired pixel is a guess, so a large defect map
                             costs real detail -- the run warns about that.
  - comparison/            : every GT candidate together -- mean, median,
                             trimmed mean, defect-repaired, dark-subtracted and
                             gain=1 stills -- as full frames, 100% crops, and
                             difference maps against the plain mean, with their
                             residual pixel noise printed
  - gt_checkpoint_noise.png: measured noise vs frames averaged -- split-half
                             temporal noise (unbiased; the two halves share no
                             frames) alongside the high-pass residual, against
                             a 1/sqrt(N) reference. Where the two diverge the
                             frame is fixed-pattern-noise limited.
  - gt_running_mean_*      : the running mean at log-spaced checkpoints, plus a
                             100%-zoom comparison and the split-half difference
                             at each checkpoint (temporal noise alone -- scene
                             and FPN cancel in the subtraction)
  - gt_defect_map.png/.npy : hot/cold pixels found in the master dark
                             (only when DARK_DIR is set)
  - gt_stationarity_*.png  : frame level over the run, plus a first-half vs
                             second-half check. Averaging assumes every frame
                             shows the same thing; if the sensor warmed or the
                             lighting drifted, that error does not average away
                             and the split-half noise estimate cannot see it.

Correction order matters and is fixed: average, then subtract the dark, then
interpolate defects. Interpolating before the subtraction removes a defect's
large dark value from an already-repaired pixel and punches a hole (measured:
1.98 ADU of residual error the right way round, 83.20 the wrong way). Repairing
only the averaged frame is also sufficient -- cubic fill is linear and the mask
is static, so repairing every frame first is bit-identical for N times the work.

The sequence is read exactly once; every measurement above rides that single
streaming pass.
"""

import argparse
import json
import time
import sys
from datetime import datetime, timezone
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
    highpass_std, bayer_plane_median3, directional_fill_bayer,
    progress, prefetch, format_duration,
    uncalibrate_frame, save_dng, get_dng_color_matrix, git_revision,
    read_dng_color_tags,
)


# ============================================================
# CONFIG — edit paths and options here
# ============================================================
SEQUENCE_DIR  = "/path/to/static/sequence"
OUTPUT_DIR    = "output/gt_analysis"
RUN_SUFFIX    = "_median_fill"   # appended to the per-sequence output dir, so
                                   # runs sit side by side instead of
                                   # overwriting each other and the directory
                                   # name says what was being tested.
                                   #
                                   # Maintained at commit time, not by hand:
                                   # any commit that changes what the outputs
                                   # look like updates this slug too. See
                                   # CLAUDE.md. run_info.json in each output
                                   # directory holds the precise record (commit,
                                   # branch, dirty flag, full config).
GN3_BLACK_LEVEL = 256    # uniform black level for GN3 .raw files

MAX_FRAMES    = None  # int to cap total frames loaded, None = all
MAX_STACK     = 60    # max frames loaded into RAM for median / trimmed-mean
TRIM_FRAC     = 0.05  # total fraction trimmed (symmetric: TRIM_FRAC/2 from each tail)
N_SAMPLES     = 5     # number of evenly-spaced sample frames to show

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
LOAD_WORKERS    = 4     # threads used to decode frames ahead of the accumulator.
                        # rawpy releases the GIL around LibRaw's decode, so
                        # threads give real parallelism here (measured ~3x at 4
                        # workers); more than the core count regresses. Frames
                        # are still consumed strictly in order -- the split-half,
                        # stationarity and checkpoint logic all depend on frame
                        # index. Costs about (workers + 2) frames of memory.
DEFECT_FILL     = "median"  # how repaired pixels are rebuilt: "median" (3x3
                            # per Bayer sub-plane, quieter, better on the smooth
                            # content a lowlight GT is mostly made of) or
                            # "directional" (follows edges, better on thin
                            # high-contrast detail, noisier elsewhere).
DEFECT_FRAC_WARN = 0.001  # warn once the defect map exceeds this fraction of the
                          # frame. Every flagged pixel is reconstructed from its
                          # neighbours, so a large map trades noise for lost
                          # detail; real sensors are well under 0.1%.
STILLS_DIR      = None  # dir of gain=1 long-exposure stills to compare against.
                        # None = skip the comparison.
MATCH_STILL_INTENSITY = True   # rescale the stills by a robust ratio so the
                               # comparison is not dominated by an exposure
                               # mismatch between the two capture settings
COMPARISON_CROP = 400   # centre-crop size (px) for the 100%-zoom comparison
SAVE_DNG        = True  # write a .dng next to every GT candidate PNG, in raw
                        # ADU with the black pedestal restored, so downstream
                        # tools read it exactly like an original capture
SAVE_NPY        = True  # write a .npy too: the calibrated float32 frame, before
                        # the integer rounding a DNG imposes. At 10-bit that
                        # rounding costs ~0.00065 in calibrated units, which is
                        # the same order as the temporal noise left after a few
                        # hundred frames -- so use the .npy, not the .dng, when
                        # computing metrics against the GT.
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

    # Stationarity accumulators, on a decimated grid (step 4 keeps Bayer phase,
    # 1/16 the memory). Alongside the even/odd split we keep a first-half /
    # second-half split: even/odd is deliberately blind to slow drift, since
    # interleaving means both halves warm up equally, so it cannot tell whether
    # the sequence stayed stationary. First/second is maximally sensitive to it,
    # and the two are directly comparable because the group sizes match.
    DEC = 4
    dec_acc = {'e': None, 'o': None, 'f1': None, 'f2': None}
    dec_n   = {'e': 0, 'o': 0, 'f1': 0, 'f2': 0}
    half    = n // 2
    frame_levels = []

    for i, (p, frame) in enumerate(progress(
            prefetch(paths, loader, LOAD_WORKERS), desc="  mean", total=n)):
        idx = i + 1
        raw = frame.astype(np.float64)
        if i % 2 == 0:
            acc_e = raw if acc_e is None else acc_e + raw
            n_e += 1
        else:
            acc_o = raw if acc_o is None else acc_o + raw
            n_o += 1

        frame_levels.append(float(raw.mean()))
        dec = raw[::DEC, ::DEC]
        for key in ('e' if i % 2 == 0 else 'o', 'f1' if i < half else 'f2'):
            dec_acc[key] = dec.copy() if dec_acc[key] is None else dec_acc[key] + dec
            dec_n[key] += 1

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
    stationarity = {'dec_acc': dec_acc, 'dec_n': dec_n,
                    'levels': np.array(frame_levels, dtype=np.float64)}
    return (calibrate_frame(mean_adu, pattern, black, white), mean_adu,
            metrics, stationarity)


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
    for i, (_p, frame) in enumerate(progress(
            prefetch(paths[:n], loader, LOAD_WORKERS),
            desc="  dark mean/std", total=n)):
        x = frame.astype(np.float64)
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
    for _p, frame in progress(prefetch(paths[:n], loader, LOAD_WORKERS),
                              desc="  dark clip", total=n):
        x = frame.astype(np.float32)
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
    """
    Repair flagged pixels from their same-colour neighbours.

    DEFECT_FILL picks the method:

      'median'      3x3 median of each Bayer sub-plane. Averages nine samples,
                    so it is the quieter estimate and wins on the smooth,
                    low-contrast content a lowlight GT frame is mostly made of.
                    Measured on realistic content (flat scene, sigma 0.006), mean
                    error at defects: 0.00411 / 0.00306 / 0.00422 at 0.1% / 0.5%
                    / 2% defect density, versus 0.00454 / 0.00357 / 0.00444 for
                    directional. Its weakness is thin high-contrast detail: on
                    1-px strokes it reaches across the edge and returns the
                    background.

      'directional' interpolate along the locally smoothest of four directions,
                    from two neighbours. Follows a stroke instead of crossing
                    it, so it is far better on fine high-contrast structure, but
                    two samples are noisier than nine and it loses on smooth
                    content -- which is most of the frame.

    Median is the default because it measured better on real sequences. Switch
    per-sequence if yours is unusually detailed; the difference is small either
    way next to how many pixels get flagged in the first place.
    """
    if not mask.any():
        return frame
    if str(DEFECT_FILL).lower().startswith('dir'):
        return directional_fill_bayer(frame, pattern, mask)
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
    for i, (_p, frame) in enumerate(progress(
            prefetch(paths[1:n], loader, LOAD_WORKERS),
            desc="  stack", total=n - 1), start=1):
        mm[i] = calibrate_frame(frame, pattern, black, white)
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
    for _p, frame in progress(prefetch(paths, loader, LOAD_WORKERS),
                              desc="  stills", total=n):
        raw   = frame.astype(np.float64)
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


def _grid(n, per_row=3):
    """Rows/cols for n panels -- a single row gets unreadably wide past three."""
    cols = min(per_row, n)
    return int(np.ceil(n / cols)), cols


def _plot_comparison(frames, labels, pattern, out, wb=None, ccm=None):
    """RGB of every GT candidate, in a grid."""
    rows, cols = _grid(len(frames))
    fig, axes = plt.subplots(rows, cols, figsize=(6.2 * cols, 5.4 * rows),
                             squeeze=False)
    flat = axes.ravel()
    for ax, f, lab in zip(flat, frames, labels):
        ax.imshow(demosaic_to_rgb(f, pattern, wb, ccm))
        ax.set_title(lab, fontsize=10)
        ax.axis("off")
    for ax in flat[len(frames):]:
        ax.axis("off")
    fig.suptitle("GT candidates", fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def _plot_comparison_crops(frames, labels, pattern, out, crop, wb=None, ccm=None):
    """100%-zoom centre crops -- downscaling would hide the pixel-level
    differences these candidates are being compared on."""
    rows, cols = _grid(len(frames))
    fig, axes = plt.subplots(rows, cols, figsize=(4.6 * cols, 5.2 * rows),
                             squeeze=False)
    flat = axes.ravel()
    for ax in flat[len(frames):]:
        ax.axis("off")
    for ax, f, lab in zip(flat, frames, labels):
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
    rows, cols = _grid(len(pairs))
    fig, axes = plt.subplots(rows, cols, figsize=(6.2 * cols, 5.2 * rows),
                             squeeze=False)
    flat = axes.ravel()
    for ax in flat[len(pairs):]:
        ax.axis("off")
    for ax, (diff, lab) in zip(flat, pairs):
        v = max(float(np.percentile(np.abs(diff), 99.5)), 1e-9)
        im = ax.imshow(diff, cmap="RdBu_r", vmin=-v, vmax=v)
        ax.set_title(f"{lab}\nstd = {diff.std():.6f}", fontsize=10)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    fig.suptitle("What each step changed, relative to the plain mean",
                 fontsize=13, y=1.01)
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


def _report_stationarity(st, white, black, out):
    """
    Did the capture stay stationary for its whole length?

    Averaging assumes every frame shows the same thing. Over a long run the
    sensor warms, dark current grows, lighting can drift -- and none of that
    averages away. The even/odd split-half estimator cannot see it: interleaved
    halves warm equally, which is exactly what makes it a good noise estimator
    and a useless drift detector.

    So compare the first half against the second. Both splits have the same
    group sizes, so under a stationary capture their differences must have the
    same magnitude. Two statistics, because drift has two components:

      level   -- mean of (first - second). Catches a uniform shift in level.
                 By far the more sensitive of the two: its noise floor is the
                 per-pixel floor divided by sqrt(number of pixels).
      pattern -- std of (first - second) over std of (even - odd). Catches
                 differential growth, e.g. hot pixels warming faster than the
                 rest. Expected ~1.0 when stationary.
    """
    a, k = st['dec_acc'], st['dec_n']
    if a['o'] is None or a['f2'] is None or min(k.values()) == 0:
        return
    scale = float(white - black[0])
    d_eo = (a['e'] / k['e'] - a['o'] / k['o']) / scale
    d_fs = (a['f1'] / k['f1'] - a['f2'] / k['f2']) / scale

    level    = float(abs(d_fs.mean()))
    floor    = float(d_eo.std()) / np.sqrt(d_eo.size)
    ratio    = float(d_fs.std() / d_eo.std()) if d_eo.std() > 0 else float('nan')
    drifting = level > 5 * floor or ratio > 1.3

    print("\n  Stationarity (first half vs second half)")
    print(f"    level shift            : {level:.6f}   "
          f"({level / floor:.1f}x the {floor:.6f} noise floor)")
    print(f"    pattern std ratio      : {ratio:.2f}   (1.0 = stationary)")
    if drifting:
        print("    -> the capture DRIFTED. Frames late in the run do not show "
              "the same thing as frames early on,")
        print("       and that error does not average away. Prefer a shorter "
              "run, or split it and check each part.")
    else:
        print("    -> stationary; the whole run can be averaged safely.")

    levels = st['levels'] / scale
    fig, ax = plt.subplots(figsize=(9, 4.4))
    ax.plot(levels, linewidth=0.9, color="steelblue")
    if len(levels) > 20:                       # running mean to expose slow trends
        w = max(5, len(levels) // 40)
        ker = np.ones(w) / w
        ax.plot(np.arange(w - 1, len(levels)), np.convolve(levels, ker, 'valid'),
                linewidth=2.0, color="crimson", label=f"{w}-frame running mean")
        ax.legend(fontsize=9)
    ax.set_xlabel("Frame index", fontsize=11)
    ax.set_ylabel("Frame mean level [calibrated]", fontsize=11)
    ax.set_title(f"Capture stationarity — level shift {level / floor:.1f}x floor, "
                 f"pattern ratio {ratio:.2f}", fontsize=12)
    ax.grid(True, alpha=0.3, linestyle="--")
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


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
    # A capped run is a different experiment from a full one, so it gets its own
    # directory rather than silently overwriting the full result.
    cap      = f"_max{max_frames}" if max_frames else ""
    seq_out  = out_dir / (seq_name + (RUN_SUFFIX or "") + cap)
    seq_out.mkdir(parents=True, exist_ok=True)

    rev = git_revision()
    if rev:
        print(f"  Code: {rev['short']} on {rev['branch']}"
              + ("  (WORKING TREE DIRTY)" if rev['dirty'] else "")
              + f"  — {rev['subject']}")

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
    full_mean, full_mean_adu, ckpt_metrics, stationarity = _stream_mean(
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
    _report_stationarity(stationarity, white, black,
                         seq_out / f"gt_stationarity_N{n}.png")

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

    # Full-resolution saves: an RGB PNG to look at, and a DNG to feed onward.
    # The DNG carries the sequence's own black/white levels and camera profile,
    # so a GT frame drops into the same tooling as an original capture.
    # Copy the capture's own colour/identity tags rather than synthesising a
    # profile: converters resolve colour by looking the camera up in their own
    # database, so a DNG naming an unknown camera renders wrong however good
    # its ColorMatrix1 is. GN3 has no source DNG, so those files stay raw data
    # containers with an identity matrix.
    dng_tags = read_dng_color_tags(paths[0]) if fmt == 'dng' else None
    dng_ccm  = get_dng_color_matrix(paths[0]) if fmt == 'dng' else None
    if fmt == 'dng':
        print(f"  DNG colour tags copied from source: "
              f"{len(dng_tags or [])} tag(s)")
    # AsShotNeutral is the camera-space value of a neutral patch, i.e. the
    # reciprocal of the white-balance gains that get applied to reach neutral.
    neutral = (1.0 / np.asarray(wb, dtype=float)) if wb is not None else None

    def save_candidate(frame, png_path):
        _save_rgb_frame(frame, pattern, png_path, wb, ccm)
        if SAVE_NPY:
            npy_path = png_path.with_name(
                png_path.stem.replace('_rgb', '') + '.npy')
            np.save(npy_path, frame.astype(np.float32))
            print(f"Saved {npy_path}")
        if SAVE_DNG:
            # Drop the "_rgb" marker from the DNG's name -- it describes the
            # PNG's demosaiced content, not the Bayer data in the DNG.
            dng_path = png_path.with_name(
                png_path.stem.replace('_rgb', '') + '.dng')
            save_dng(uncalibrate_frame(frame, pattern, black, white),
                     dng_path, pattern, black, white,
                     color_matrix=dng_ccm, as_shot_neutral=neutral,
                     model=f"noise_analysis GT ({seq_name})",
                     copy_tags=dng_tags)

    print("  Saving full-resolution frames …")
    save_candidate(full_mean,     seq_out / f"gt_mean_rgb_N{n}.png")
    save_candidate(median_frame,  seq_out / f"gt_median_rgb_N{n_stack}.png")
    save_candidate(trimmed_frame, seq_out / f"gt_trimmed_mean_rgb_N{n_stack}.png")

    # ---------------------------------------------------------------- #
    # Dark subtraction                                                   #
    # ---------------------------------------------------------------- #
    dark_corrected = None
    mean_defect_fixed = None
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
        save_candidate(dark_corrected,
                       seq_out / f"gt_mean_darksub_rgb_N{n}.png")

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
            frac = mask.mean() * 100
            print(f"\n  Defect map (>{HOT_PIXEL_SIGMA}σ from same-colour "
                  f"neighbours in the master dark): "
                  f"{n_hot} hot, {n_cold} cold, {frac:.4f}% of pixels")
            if frac > DEFECT_FRAC_WARN * 100:
                # Every flagged pixel is guessed from its neighbours, and a guess
                # on fine detail is damage. Real sensors sit well under 0.1%; a
                # larger map usually means the master dark is still noisy enough
                # that its own noise is clearing the threshold.
                print(f"  WARNING: that is a lot of pixels to interpolate. Each one "
                      f"is reconstructed from its neighbours,")
                print(f"           which costs real detail wherever the scene is "
                      f"fine (text, edges). Raise --hot-pixel-sigma or")
                print(f"           add dark frames until the map is nearer "
                      f"{DEFECT_FRAC_WARN * 100:.2f}%.")
            if mask.any():
                _plot_defect_map(mask, n_hot, n_cold,
                                 seq_out / "gt_defect_map.png")
                np.save(seq_out / "gt_defect_map.npy", mask)

                # Two repaired variants, so the dark subtraction can be judged
                # separately from the defect repair rather than bundled with it.
                #
                # Repair happens AFTER the subtraction, never before. A defect's
                # dark value is large; interpolating first and then subtracting
                # it removes that large value from an already-repaired pixel and
                # punches a hole. Measured on synthetic data: subtract-then-
                # interpolate leaves 1.98 ADU of error at defects, the reverse
                # order leaves 83.20.
                #
                # Repairing the averaged frame is also all that is needed --
                # cubic fill is linear and the mask is static, so repairing
                # every frame first gives a bit-identical result for N times
                # the work (verified: max difference 2e-5, float noise).
                mean_fixed = _interpolate_defects(full_mean, pattern, mask)
                dark_fixed = _interpolate_defects(dark_corrected, pattern, mask)

                print(f"  High-pass std   mean={hp_before:.6f}  "
                      f"mean+defectfix={highpass_std(mean_fixed, pattern):.6f}")
                print(f"                  darksub={hp_after:.6f}  "
                      f"darksub+defectfix={highpass_std(dark_fixed, pattern):.6f}")
                save_candidate(mean_fixed,
                               seq_out / f"gt_mean_defectfix_rgb_N{n}.png")
                save_candidate(dark_fixed,
                               seq_out / f"gt_mean_darksub_defectfix_rgb_N{n}.png")
                mean_defect_fixed = mean_fixed
                dark_corrected    = dark_fixed

    # ---------------------------------------------------------------- #
    # Comparison of GT candidates                                        #
    # ---------------------------------------------------------------- #
    # Ordered so the pure aggregators come first and each correction is added
    # on top of the plain mean, which makes the difference maps below read as
    # "what did this step change?" rather than an arbitrary pairing.
    # (frame, human label for figures, filename slug). The slug is explicit
    # rather than derived from the label -- deriving it produced names like
    # "cmp_Mean_+_defects_interpolated.png".
    cands = [(full_mean,     f"Mean  (N={n})",                 "mean"),
             (median_frame,  f"Median  (N={n_stack})",         "median"),
             (trimmed_frame, f"Trimmed mean  (N={n_stack})",   "trimmed_mean")]
    if mean_defect_fixed is not None:
        cands.append((mean_defect_fixed,
                      f"Mean + defects interpolated  (N={n})", "mean_defectfix"))
    if dark_corrected is not None:
        label = "Mean − dark" + (" + defects interpolated"
                                 if mean_defect_fixed is not None else "")
        slug  = "mean_darksub" + ("_defectfix"
                                  if mean_defect_fixed is not None else "")
        cands.append((dark_corrected, f"{label}  (N={n})", slug))
    if STILLS_DIR:
        print(f"\n  Stills: {STILLS_DIR}")
        still, n_still, scale = _stills_reference(
            STILLS_DIR, full_mean.shape,
            match_to=full_mean if MATCH_STILL_INTENSITY else None)
        print(f"  {n_still} stills averaged"
              + (f", intensity-matched by ×{scale:.4f}" if MATCH_STILL_INTENSITY
                 else ", no intensity matching"))
        cands.append((still, f"Gain=1 stills  (N={n_still})", "stills_gain1"))

    if len(cands) > 1:
        cmp_dir = seq_out / "comparison"
        cmp_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n  Comparing {len(cands)} GT candidates …")

        frames = [c[0] for c in cands]
        labels = [c[1] for c in cands]
        slugs  = [c[2] for c in cands]
        _plot_comparison(frames, labels, pattern,
                         cmp_dir / "cmp_frames.png", wb, ccm)
        _plot_comparison_crops(frames, labels, pattern,
                               cmp_dir / "cmp_crops.png", COMPARISON_CROP, wb, ccm)
        # Every candidate differenced against the plain mean, rather than all
        # pairs: with six candidates all-pairs is fifteen panels, and "what did
        # this step change relative to doing nothing" is the question that
        # actually gets asked.
        _plot_comparison_diffs(
            [(frames[i] - frames[0], f"{labels[i].split('(')[0].strip()}  −  Mean")
             for i in range(1, len(frames))],
            cmp_dir / "cmp_differences.png")
        for f, slug in zip(frames, slugs):
            save_candidate(f, cmp_dir / f"cmp_{slug}.png")

        print("\n  Residual pixel noise (high-pass std, calibrated units)")
        width = max(len(l) for l in labels) + 2
        base  = highpass_std(frames[0], pattern)
        for f, lab in zip(frames, labels):
            hp = highpass_std(f, pattern)
            print(f"    {lab:<{width}s} {hp:.6f}   {base / hp:5.2f}x vs mean")
        print()

    _write_run_info(seq_out, directory, fmt, n, n_stack, rev)
    print(f"\nDone. Outputs in {seq_out.resolve()}")


# --------------------------------------------------------------------------- #
# CLI overrides + entry point                                                   #
# --------------------------------------------------------------------------- #

def _write_run_info(seq_out, directory, fmt, n, n_stack, rev):
    """
    Record what produced these outputs, next to the outputs themselves.

    A results directory that cannot be traced back to a code state and a set of
    settings is guesswork a week later, and this pipeline has enough knobs that
    "which run was that?" is a real question. The whole CONFIG block is captured
    rather than a chosen subset, so nothing silently goes unrecorded when a new
    option is added.
    """
    g = globals()
    scalar = (bool, int, float, str, type(None))
    config = {k: g[k] for k in sorted(g)
              if k.isupper() and not k.startswith('_') and isinstance(g[k], scalar)}
    info = {
        "run": {
            "utc":       datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "script":    Path(__file__).name,
            "sequence":  str(Path(directory).resolve()),
            "format":    fmt,
            "frames":    n,
            "stack":     n_stack,
        },
        "code": rev or {"note": "git unavailable — provenance not recorded"},
        "config": config,
    }
    out = seq_out / "run_info.json"
    out.write_text(json.dumps(info, indent=2, sort_keys=False))
    print(f"Saved {out}")


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
    t0 = time.monotonic()
    analyze_gt_sequence(
        SEQUENCE_DIR,
        Path(OUTPUT_DIR),
        max_frames=MAX_FRAMES,
        max_stack=MAX_STACK,
    )
    print(f"Total time: {format_duration(time.monotonic() - t0)}")


if __name__ == "__main__":
    main()
