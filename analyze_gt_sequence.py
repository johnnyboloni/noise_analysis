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
  - comparison/            : every GT candidate -- mean, defect-repaired,
                             dark-subtracted (both alone and with defects also
                             interpolated), the gain=1 still, (with
                             ROBUST_AGGREGATORS) median and trimmed mean, and
                             (with SIGMA_CLIP_MEAN) a sigma-clipped mean --
                             each written full-resolution as .npy and .dng (to
                             feed onward) plus three PNGs to look at: _nogain
                             (gain 1.0, the scene as calibrated), _uniform (one
                             gain shared across every candidate, set by
                             GAIN_PERCENTILE so a lone outlier pixel can't hold
                             the whole set's brightness down -- use this to
                             compare candidates), and _max (this candidate's
                             own version of the same gain, ignoring the others
                             -- as bright as this one image alone can get, but
                             no longer comparable to its neighbours). All three
                             are gamma-encoded; none is auto-brightened.
                             Residual pixel noise is printed. This is the
                             only place frame data is written; there are no
                             duplicate copies at the top level.
  - comparison/crops/      : the same three PNGs per candidate, cropped to a
                             CROP_SIZE square at CROP_XY -- at the SAME gains
                             as the full-frame versions (not re-gained from
                             just the crop), so a 100%-zoom look at one region
                             without opening the full-resolution files. None
                             or CROP_SIZE = 0 skips this.
  - gt_checkpoint_noise.png: measured noise vs frames averaged -- split-half
                             temporal noise (unbiased; the two halves share no
                             frames) alongside the high-pass residual, against
                             a 1/sqrt(N) reference. Where the two diverge the
                             frame is fixed-pattern-noise limited.
  - gt_running_mean_comparison_*, gt_halfdiff_comparison_*,
    gt_highpass_comparison_*
                           : a 100%-zoom crop of the running mean at each
                             checkpoint; the split-half difference at each
                             checkpoint (temporal noise alone -- scene and FPN
                             cancel in the subtraction, so this keeps shrinking
                             with N); and the running mean's own high-pass
                             residual at each checkpoint (temporal noise AND
                             FPN together -- shrinks only until temporal noise
                             stops dominating, then visibly flattens, the same
                             floor gt_checkpoint_noise.png's two curves diverge
                             at, made visible panel to panel)
  - gt_dark_uniformity_*.png : whether the master dark is spatially flat --
                             full resolution, block-averaged, and row/column
                             median profiles, per-channel offset removed. The
                             evidence for finding defects locally rather than
                             against one global distribution (only when
                             DARK_DIR is set)
  - gt_defect_map.png/.npy : hot/cold pixels found in the master dark, via
                             DEFECT_METHOD ("local" (default) or "global" --
                             global is only valid once gt_dark_uniformity's
                             ratio is confirmed well under 3x; local vs global
                             land in separate output dirs so they can be
                             compared directly)
  - gt_defect_sigma_scan.png / _excess.png
                           : histogram of the local residual against a
                             Gaussian reference, and flagged-count vs
                             threshold on log-log axes -- for choosing
                             HOT_PIXEL_SIGMA from data. The _excess plot is
                             the actual test for a real defect population: a
                             floor before the final drop means one exists; a
                             smooth slope all the way down means the
                             threshold is just cutting into a continuum, no
                             matter how large "excess over chance" looks at
                             any single k (only when DARK_DIR is set)
  - gt_stuck_pixel_map.png/.npy : pixels whose TEMPORAL STD across the dark
                             sequence is anomalously low relative to their
                             same-colour neighbours (stuck/dead), via
                             STUCK_PIXEL_SIGMA. A different test from
                             gt_defect_map -- level vs noise -- so it catches
                             pixels with a normal mean that never fluctuate,
                             invisible to the level-based map (only when
                             DARK_DIR is set)
  - gt_stuck_pixel_sigma_scan.png / _excess.png
                           : same pair as gt_defect_sigma_scan, run on the
                             temporal std map's low-noise side, for choosing
                             STUCK_PIXEL_SIGMA from data (only when DARK_DIR
                             is set)
  - gt_drift_*.png         : per-frame position over the run, the path it
                             traced, and the distribution of drift magnitudes,
                             from phase correlation on a Bayer sub-plane crop.
                             Averaging misaligned frames convolves the result
                             with the spread of positions, so drift is blur
                             that more frames cannot remove -- the first thing
                             to check when the GT looks soft. MAX_DRIFT_PX
                             excludes any frame past it from every accumulator
                             (mean, split-half, stationarity); excluded frames
                             are marked on all three panels.
  - gt_stationarity_*.png  : frame level over the run, plus a first-half vs
                             second-half check. Averaging assumes every frame
                             shows the same thing; if the sensor warmed or the
                             lighting drifted, that error does not average away
                             and the split-half noise estimate cannot see it.

"defectfix" in a candidate name means pixels on the defect map were rebuilt
from their same-colour neighbours (DEFECT_FILL chooses median or directional).
Candidates are produced both with and without the dark subtraction, so the two
corrections can be judged separately. Every repaired pixel is a guess, so a
large defect map costs real detail -- the run warns about that.

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

import raw_utils
from raw_utils import (
    detect_format, find_dngs, find_raws,
    get_raw_metadata, get_raw_metadata_gn3, get_color_metadata,
    load_raw, load_raw_gn3,
    calibrate_frame, demosaic_linear, encode_rgb, uniform_gain, save_rgb_png,
    bayer_subplane_crop, phase_shift, drift_report,
    highpass_std, highpass_residual, bayer_plane_median3, directional_fill_bayer,
    temporal_std_map,
    progress, prefetch, format_duration,
    uncalibrate_frame, save_dng, get_dng_color_matrix, git_revision,
    read_dng_color_tags,
)


# ============================================================
# CONFIG — edit paths and options here
# ============================================================
SEQUENCE_DIR  = "/path/to/static/sequence"
OUTPUT_DIR    = "output/gt_analysis"
RUN_SUFFIX    = "_stuck_pixel_fix"   # appended to the per-sequence output dir, so
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
ROBUST_AGGREGATORS = False  # also produce a median and a trimmed mean, as extra
                      # GT candidates alongside the plain mean. Off by default:
                      # they cost a second read pass over the sequence plus a
                      # MAX_STACK-sized scratch file on disk, and they can only
                      # win if the sequence has outliers (cosmic rays, dropped
                      # or corrupt frames, a light flicker, something crossing
                      # the scene). For clean Gaussian noise the mean is the
                      # minimum-variance estimator and the median costs 25% more
                      # noise -- and because these run on MAX_STACK frames while
                      # the mean runs on all of them, the median here actually
                      # starts out 1.253*sqrt(N/MAX_STACK) noisier (~4.5x at
                      # N=760, MAX_STACK=60), so outliers would have to be
                      # blatant to overcome that.
                      #
                      # Turn back on to check: if RMS(mean - median) comes out
                      # near its outlier-free prediction 0.756*sigma1/sqrt(N),
                      # there are no outliers and the mean wins outright. To
                      # compare them fairly, raise MAX_STACK to N first.
MAX_STACK     = 60    # max frames loaded into RAM for median / trimmed-mean
TRIM_FRAC     = 0.05  # total fraction trimmed (symmetric: TRIM_FRAC/2 from each tail)

SIGMA_CLIP_MEAN = None  # also produce a per-pixel sigma-clipped mean, as an
                      # extra GT candidate alongside the plain mean -- reuses
                      # the same two-pass clip already used to build the
                      # master dark, applied here to the light sequence.
                      # Value is the sigma threshold (e.g. 4.0); 0 or None =
                      # skip. Off by default: it costs two extra full passes
                      # over the sequence. Unlike ROBUST_AGGREGATORS' median
                      # and trimmed mean, this is streaming and needs no
                      # full-stack RAM/disk, so it runs on all N frames rather
                      # than being capped at MAX_STACK -- directly comparable
                      # to the plain mean at the same N, with no built-in
                      # noise penalty from a smaller sample.

N_CHECKPOINTS   = 5     # running-mean snapshots taken while streaming (log-spaced in N)
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
                        # per-pixel mean (cosmic rays, dropped frames). Clipping
                        # is per-pixel and iterated twice, which makes the
                        # result nearly independent of this number: measured on
                        # 300 synthetic darks with 0.3% cosmic rays, 3 / 4 / 5
                        # all land within 0.001 ADU of each other. Pick it by
                        # how many clean Gaussian samples you are willing to
                        # throw away -- with N darks the expected count per
                        # pixel is N*P(|z|>k), so at N=300 that is 0.8 samples
                        # at k=3 but 0.02 at k=4. Below 3 the loss of real
                        # samples starts to cost noise (k=2 measured 10% worse);
                        # above 5 there is nothing left to gain.
HOT_PIXEL_SIGMA = 5.0   # flag master-dark pixels this many residual-sigma from
                        # their same-colour neighbours as defects, and repair
                        # them by interpolation. 0 or None = skip.
STUCK_PIXEL_SIGMA = 5.0  # flag pixels whose TEMPORAL STD across the dark
                        # sequence is this many robust-sigma BELOW their
                        # same-colour neighbours, and repair by interpolation.
                        # Not the same test as HOT_PIXEL_SIGMA: that one looks
                        # at the master dark's LEVEL, this one at its per-pixel
                        # NOISE, so it catches pixels that read a normal mean
                        # but never fluctuate (stuck/dead) -- invisible to a
                        # level-based threshold. Deliberately one-sided: the
                        # opposite tail (anomalously HIGH temporal noise, e.g.
                        # RTS) was tested (check_noisy_pixels.py) and found to
                        # be a continuous population with no natural cutoff,
                        # not a discrete defect -- thresholding it costs real
                        # detail for no measured benefit, so only the stuck/low
                        # side is corrected here. 0 or None = skip.
DEFECT_METHOD   = "local"  # "local" (default) subtracts a same-colour local
                        # median before thresholding, so real sensor structure
                        # (gradients, banding, channel offsets) isn't mistaken
                        # for defects. "global" skips that and thresholds the
                        # raw master dark per sub-plane -- only valid once
                        # gt_dark_uniformity's block-averaged-span/per-pixel-
                        # noise ratio is confirmed well under the 3x it warns
                        # at; measured at 0x/0.4x/6x that ratio, global matches
                        # local (and has fewer false positives) below ~1x but
                        # its recall collapses to 78.6% by 6x while local holds
                        # at 100%. See _defect_map's docstring for the numbers.
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
MEASURE_DRIFT   = True  # track per-frame translation against the first frame,
                        # by phase correlation on a DRIFT_CROP-sized crop of one
                        # Bayer sub-plane. Averaging misaligned frames convolves
                        # the result with the spread of positions, so drift is
                        # blur that more frames cannot remove -- measured on a
                        # synthetic target, 0.5 px of spread costs 4% of
                        # sharpness, 1 px costs 14% and 2 px costs 39%. Rides
                        # the existing pass at ~35 ms a frame.
DRIFT_CROP      = 512   # sub-plane crop size (px) used for registration
DRIFT_WARN_PX   = 0.5   # warn once the RMS spread of frame positions exceeds
                        # this. Do not set it near the estimator's own noise
                        # floor (~0.3 px, inherent to registering an aliased
                        # Bayer sub-plane -- verified to match scikit-image's
                        # phase_cross_correlation to within 0.01 px).
MAX_DRIFT_PX    = None  # exclude frames whose shift vs frame 0 exceeds this
                        # many pixels from every accumulator (mean, split-half,
                        # stationarity) -- forces drift measurement on even if
                        # MEASURE_DRIFT is False. None = keep every frame.
                        # Don't set this near or below the ~0.3 px estimator
                        # floor above -- it can't tell real sub-floor drift
                        # from its own noise, so a threshold that low would
                        # drop frames at random rather than for cause.
DEMOSAIC        = "menon"   # demosaic used for the PNG previews and comparison
                            # crops (the DNG/NPY ground truth stays in the Bayer
                            # domain and is never demosaiced). "menon" (DDFAPD)
                            # measures 6 dB better on edges and ~40% less false
                            # colour than the "ea" this used before, at ~27 s and
                            # ~2.5 GB per 12 MP frame; "malvar" is the cheap
                            # middle ground (~9 s), "ea" the old fast default.
                            # See raw_utils.demosaic_linear for the numbers.
DEFECT_FRAC_WARN = 0.005  # warn once the defect map exceeds this fraction of the
                          # frame. Every flagged pixel is reconstructed from its
                          # neighbours, so a large map trades noise for lost
                          # detail. Set above the ~0.2% a healthy sensor plus a
                          # correct threshold produces, so the warning means
                          # "something is wrong" rather than firing on a good
                          # map.
STILLS_DIR      = None  # dir of gain=1 long-exposure stills to compare against.
                        # None = skip the comparison.
STILLS_FRAMES   = 1     # how many stills to use. 1 (default) keeps this an
                        # independent single-shot reference against the averaged
                        # GT; averaging more would make it another aggregate and
                        # blur anything that moved between shots.
MATCH_STILL_INTENSITY = True   # rescale the stills by a robust ratio so the
                               # comparison is not dominated by an exposure
                               # mismatch between the two capture settings
GAIN_PERCENTILE = 99.99  # the display gain for _uniform/_max PNGs and the
                        # checkpoint-crop panel is set to put this percentile
                        # at full scale, not the literal max -- one hot pixel
                        # or interpolation-overshoot outlier otherwise holds
                        # the gain (and so every PNG's brightness) down to
                        # whatever keeps just that one pixel under the
                        # ceiling. Measured: 3 outlier pixels out of 245,760
                        # held a real comparison set to 1.17x when 99.99 would
                        # allow 1.60x (36% brighter), clipping only those same
                        # few pixels. Set to 100 for the old zero-clipping
                        # guarantee if that tradeoff is ever wrong here.
CROP_XY   = (1500, 1800)  # (x, y) pixel coords of the top-left corner for
                        # comparison/crops/ -- a 100%-crop region saved for
                        # every candidate, at the SAME gains as the
                        # full-frame PNGs (not re-gained from just the crop),
                        # so a crop is literally a sub-region of its full
                        # image, directly comparable crop to crop. None or
                        # CROP_SIZE = 0 skips this.
CROP_SIZE = 600           # crop side length in pixels (square)
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
    Detect format and return (fmt, paths, pattern, black, white, loader).
    loader(path) → float32 (H, W) raw Bayer ADU
    """
    fmt = detect_format(directory)
    if fmt == 'dng':
        paths     = find_dngs(directory)
        pattern, black, white = get_raw_metadata(paths[0])
        loader    = load_raw
    else:
        paths     = find_raws(directory)
        pattern, black, white = get_raw_metadata_gn3(paths[0], GN3_BLACK_LEVEL)
        meta      = json.loads(paths[0].with_suffix('.imgprops').read_text())
        shape     = (meta['height'], meta['width'])
        loader    = lambda p, _s=shape: load_raw_gn3(p, _s)
    return fmt, paths, pattern, black, white, loader


# --------------------------------------------------------------------------- #
# Streaming passes                                                               #
# --------------------------------------------------------------------------- #

def _stream_mean(paths, pattern, black, white, loader,
                 checkpoints=None, on_checkpoint=None):
    """Pass 1 — compute per-pixel calibrated mean, streaming one frame at a time.

    Accumulates raw ADU in float64 so sub-black noise cancels across frames,
    then calibrates (and clips) the final mean once.

    If `checkpoints` (a set of 1-based positions in the input sequence) and
    `on_checkpoint` are given, the running mean is calibrated and handed to
    on_checkpoint(n_used, frame) once processing reaches each position --
    letting the caller save intermediate averages without a second pass. The
    callback is expected to consume the frame immediately (save it / keep a
    crop) rather than retain it, so peak memory stays at one frame regardless
    of how many checkpoints are requested. `n_used` is frames actually
    averaged so far, not the checkpoint's nominal position -- they differ once
    MAX_DRIFT_PX has excluded any frames.

    MAX_DRIFT_PX (module config) excludes any frame whose measured drift vs
    frame 0 exceeds it from every accumulator here -- mean, split-half,
    stationarity alike -- so a few frames a camera bump moved do not blur the
    average that no later frame count can undo. See gt_drift_*.png.

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
    # Registration rides this pass: every frame is already decoded here, and the
    # correlation runs on a 512x512 crop of one Bayer sub-plane, so measuring
    # drift costs ~35 ms a frame rather than a whole extra read of the sequence.
    # MAX_DRIFT_PX needs a shift for every frame before deciding whether to
    # keep it, so it forces measurement on even when MEASURE_DRIFT is False.
    measure_drift = bool(MEASURE_DRIFT) or (MAX_DRIFT_PX is not None)
    ref_crop = None
    shifts   = []
    n_dropped = 0
    fired_final = False

    def _fire_checkpoint(n_used):
        total   = acc_e if acc_o is None else acc_e + acc_o
        running = calibrate_frame((total / n_used).astype(np.float32),
                                  pattern, black, white)
        del total          # free before the split-half diff allocates
        # Split-half difference, in calibrated units. The scene and any
        # fixed-pattern noise are identical in both halves and cancel exactly,
        # so this is a picture of the temporal noise alone -- the only
        # component averaging can remove. Computed once and reused for both
        # the metric and the saved image.
        if n_o > 0:
            half_diff = ((acc_e / n_e) - (acc_o / n_o)).astype(np.float32)
            half_diff /= float(white - black[0])
        else:
            half_diff = None          # N=1: no second half to compare
        metrics.append(_checkpoint_metrics(n_used, half_diff, n_e, n_o,
                                           running, pattern))
        if on_checkpoint is not None:
            on_checkpoint(n_used, running, half_diff)

    for i, (p, frame) in enumerate(progress(
            prefetch(paths, loader, LOAD_WORKERS), desc="  mean", total=n)):
        idx = i + 1
        raw = frame.astype(np.float64)

        # Drift is measured for every frame -- including ones about to be
        # dropped -- because gt_drift_*.png needs the full record to show what
        # got excluded and why, not just what survived.
        mag = None
        if measure_drift:
            crop = bayer_subplane_crop(frame, DRIFT_CROP)
            if ref_crop is None:
                ref_crop = crop
            shift = phase_shift(ref_crop, crop)
            shifts.append(shift)
            mag = float(np.hypot(*shift))

        frame_levels.append(float(raw.mean()))

        if MAX_DRIFT_PX is not None and mag is not None and mag > MAX_DRIFT_PX:
            n_dropped += 1
            continue    # excluded from every accumulator below, including the
                        # stationarity ones -- they should reflect the same
                        # frames that actually go into the mean. An
                        # intermediate checkpoint landing on a dropped frame is
                        # simply skipped (a slightly sparser convergence curve,
                        # harmless); the FINAL one is not -- see below, it's
                        # forced after the loop if it never fired here, because
                        # a monotonic drift makes the last frames the likeliest
                        # to be dropped, and skipping the final checkpoint would
                        # silently truncate the one number (the actual result)
                        # that must never go missing.

        if i % 2 == 0:
            acc_e = raw if acc_e is None else acc_e + raw
            n_e += 1
        else:
            acc_o = raw if acc_o is None else acc_o + raw
            n_o += 1

        dec = raw[::DEC, ::DEC]
        for key in ('e' if i % 2 == 0 else 'o', 'f1' if i < half else 'f2'):
            dec_acc[key] = dec.copy() if dec_acc[key] is None else dec_acc[key] + dec
            dec_n[key] += 1

        n_used = n_e + n_o
        if idx in checkpoints and n_used > 0:
            _fire_checkpoint(n_used)
            if idx == n:
                fired_final = True
    print()

    n_used = n_e + n_o
    if n_used == 0:
        sys.exit(f"MAX_DRIFT_PX={MAX_DRIFT_PX} excluded all {n} frames -- "
                 f"nothing left to average. Raise it.")
    if n_dropped:
        print(f"  Dropped {n_dropped}/{n} frames ({100 * n_dropped / n:.1f}%) "
              f"beyond MAX_DRIFT_PX={MAX_DRIFT_PX} px drift")
    if n in checkpoints and not fired_final:
        # The nominal final position (idx == n) landed on a dropped frame, so
        # the checkpoint that should represent the actual result never fired
        # in the loop above -- fire it now with the true final n_used. Guarded
        # on `n in checkpoints` so this never fires a checkpoint the caller
        # never asked for.
        _fire_checkpoint(n_used)

    total    = acc_e if acc_o is None else acc_e + acc_o
    mean_adu = (total / n_used).astype(np.float32)
    # The raw-ADU mean is returned alongside the calibrated one because dark
    # subtraction has to happen before black-level removal (see _dark_correct).
    stationarity = {'dec_acc': dec_acc, 'dec_n': dec_n,
                    'levels': np.array(frame_levels, dtype=np.float64),
                    'shifts': (np.array(shifts, dtype=np.float64)
                               if shifts else None),
                    'n_dropped': n_dropped}
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

    # Two clipping rounds, not one. The first round's bounds come from the raw
    # per-pixel mean and std -- both of which the outliers being rejected have
    # already contaminated, the std worst of all, so the window is too wide and
    # the very samples it exists to remove survive inside it. Measured on 300
    # synthetic darks with 0.3% cosmic rays: one round leaves a residual bias of
    # +0.014 / +0.036 / +0.070 ADU at sigma_clip 3 / 4 / 5 -- the higher the
    # threshold, the worse, because a wider window keeps more of what inflated
    # the std in the first place. Re-deriving the bounds from the clipped
    # statistics and clipping again collapses all three to +0.008 ADU, matching
    # a median/MAD-centred clip (which would need the whole stack resident, the
    # thing this streaming design exists to avoid). It also makes the result
    # nearly independent of sigma_clip, so the setting stops being a judgement
    # call. Cost is one more pass over the darks.
    out = None
    for rnd in range(2):
        s  = np.zeros(lo.shape, dtype=np.float64)
        sq = np.zeros(lo.shape, dtype=np.float64)
        c  = np.zeros(lo.shape, dtype=np.int32)
        for _p, frame in progress(prefetch(paths[:n], loader, LOAD_WORKERS),
                                  desc=f"  dark clip {rnd + 1}/2", total=n):
            x = frame.astype(np.float32)
            m = (x >= lo) & (x <= hi)
            v = np.where(m, x, 0.0)
            s  += v
            sq += v.astype(np.float64) ** 2
            c  += m
        print()

        cnt = np.maximum(c, 1)
        out = np.where(c > 0, s / cnt, (lo + hi) / 2.0).astype(np.float32)
        rejected = float((n - c.mean()) / n * 100.0)
        print(f"  sigma-clip round {rnd + 1} (±{sigma_clip}σ) rejected "
              f"{rejected:.3f}% of samples")
        if rnd == 0:
            var = np.maximum(sq / cnt - (s / cnt) ** 2, 0.0)
            sd  = np.sqrt(var).astype(np.float32)
            lo  = (out - sigma_clip * sd).astype(np.float32)
            hi  = (out + sigma_clip * sd).astype(np.float32)
            del var, sd
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


def _block_average(img: np.ndarray, blocks: int = 48) -> np.ndarray:
    """
    Downsample img to roughly `blocks` x `blocks` by averaging non-overlapping
    tiles, dropping leftover rows/columns that do not fill a full tile.

    A tile a few hundred pixels wide averages away per-pixel DSNU and drowns
    out a handful of hot/cold pixels (they cannot move a several-hundred-pixel
    mean by much), leaving only whatever varies smoothly across the sensor --
    exactly the structure a local defect detector is built to see through.
    """
    h, w = img.shape
    by, bx = max(1, h // blocks), max(1, w // blocks)
    h2, w2 = (h // by) * by, (w // bx) * bx
    return img[:h2, :w2].reshape(h2 // by, by, w2 // bx, bx).mean(axis=(1, 3))


def _report_dark_uniformity(dark_adu, pattern, out):
    """
    Show whether the master dark is spatially flat, or has the smooth,
    real structure (thermal/amp-glow gradients, column banding) that motivates
    finding defects locally rather than against one global distribution.

    Per-channel offset is removed before display: the four Bayer colours sit at
    different levels, so an unmodified plot is a checkerboard of colour
    offsets, not sensor structure. What's left after that is either flat
    (uniform sensor) or shows a spatial trend (it is not).

    Four panels, in increasing order of how much they average away per-pixel
    noise and isolated defects, so smooth structure gets easier to see left to
    right, top to bottom:
      - full resolution, colour-clipped to the 1st/99th percentile so a few hot
        pixels cannot wash out the scale
      - block-averaged to ~48x48 tiles, which drowns out anything pixel-sized
      - median per row / per column (median rather than mean so the rare
        defect pixel cannot move it) -- a monotonic trend here IS the
        gradient, made as simple as a single line plot.
    """
    centred = dark_adu.copy()
    for r in range(2):
        for c in range(2):
            v = centred[r::2, c::2]
            v -= np.median(v)

    coarse = _block_average(centred)
    row_profile = np.median(centred, axis=1)
    col_profile = np.median(centred, axis=0)
    lo, hi = np.percentile(centred, [1, 99])

    fig, axes = plt.subplots(2, 2, figsize=(13, 11))

    im0 = axes[0, 0].imshow(centred, cmap="viridis", vmin=lo, vmax=hi)
    axes[0, 0].set_title("Full resolution\n(per-channel offset removed; "
                         "colour range clipped to 1st/99th percentile)",
                        fontsize=10)
    axes[0, 0].axis("off")
    fig.colorbar(im0, ax=axes[0, 0], fraction=0.046,
                label="ADU above this channel's median")

    im1 = axes[0, 1].imshow(coarse, cmap="viridis")
    axes[0, 1].set_title(f"Block-averaged to {coarse.shape[1]}x{coarse.shape[0]} tiles\n"
                         "(per-pixel noise and defects cancel out here)",
                        fontsize=10)
    axes[0, 1].axis("off")
    fig.colorbar(im1, ax=axes[0, 1], fraction=0.046,
                label="ADU above this channel's median")

    axes[1, 0].plot(row_profile, linewidth=1.0, color="steelblue")
    axes[1, 0].axhline(0, color="k", linewidth=0.7)
    axes[1, 0].set_xlabel("row"); axes[1, 0].set_ylabel("median ADU")
    axes[1, 0].set_title(f"Row profile  "
                         f"(range {row_profile.max() - row_profile.min():.1f} ADU)")
    axes[1, 0].grid(alpha=0.3)

    axes[1, 1].plot(col_profile, linewidth=1.0, color="darkorange")
    axes[1, 1].axhline(0, color="k", linewidth=0.7)
    axes[1, 1].set_xlabel("column"); axes[1, 1].set_ylabel("median ADU")
    axes[1, 1].set_title(f"Column profile  "
                         f"(range {col_profile.max() - col_profile.min():.1f} ADU)")
    axes[1, 1].grid(alpha=0.3)

    fig.suptitle("Is the master dark spatially uniform?", fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")

    struct_amp = float(coarse.max() - coarse.min())
    dsnu = 1.4826 * float(np.median(np.abs(
        centred - bayer_plane_median3(centred, pattern))))
    print(f"\n  Is the master dark spatially uniform?")
    print(f"    row range                    : "
          f"{row_profile.max() - row_profile.min():6.1f} ADU")
    print(f"    column range                 : "
          f"{col_profile.max() - col_profile.min():6.1f} ADU")
    print(f"    block-averaged span          : {struct_amp:6.1f} ADU  "
          f"(smooth structure only -- defects and per-pixel noise cancel here)")
    print(f"    per-pixel noise (local resid): {dsnu:6.2f} ADU")
    if dsnu > 0 and struct_amp > 3 * dsnu:
        print(f"    -> smooth structure is {struct_amp / dsnu:.0f}x the "
              f"per-pixel noise: clearly not uniform.")
        print(f"       This is exactly what the local defect detector "
              f"subtracts out before thresholding; a single")
        print(f"       global mean/std over the whole frame would not, "
              f"and would inflate its own threshold by it.")
    else:
        print(f"    -> smooth structure is comparable to per-pixel noise "
              f"here -- close to uniform on this dark set.")


def _defect_map(dark_adu, pattern, resid_adu, n_sigma, method='local'):
    """
    Locate defective pixels in the master dark.

    method='local' (default): subtract a local same-colour median first, then
    threshold the residual. The subtraction removes everything a sensor has
    which is real but not a defect -- thermal/amp-glow gradients, column FPN,
    per-channel level offsets -- so only pixels standing out from their
    immediate surroundings survive. A purely global threshold cannot do this:
    on a dark with a 40 ADU thermal gradient it collapses to 9% recall,
    because the gradient inflates the spread it measures.

    method='global': threshold the raw master dark directly, per sub-plane,
    with no local subtraction. Only correct once _report_dark_uniformity has
    confirmed the master has no structure worth protecting against -- measured
    at the ratio (block-averaged span / per-pixel noise) that call reports:

        ratio    global recall   local recall   global false-pos   local false-pos
        ~0x            100.0%         100.0%                  0               167
        ~0.4x          100.0%         100.0%                  0               140
        ~6x             78.6%         100.0%                  0               159

    global matches local (and has fewer false positives) below roughly 1x, but
    degrades as the ratio grows and local does not move. Below the check,
    global is a strictly worse bet than checking once and using local always;
    above it, global is silently wrong. Use it only to compare against local on
    data already confirmed uniform, not as a default.

    Either way, the threshold comes from the MAD of the (local or raw) residual,
    per sub-plane, rather than from the master dark's own read-noise residual --
    that residual is dominated by ordinary DSNU spread (pixel-to-pixel variation
    that is normal, not defective), several times larger, and thresholding at
    k x residual-noise cuts deep into the healthy population (measured: 0.58-
    0.70% flagged against a true 0.2% rate). Taking the scale from the data
    instead (either method) gives 0.19-0.21%.

    resid_adu is still accepted and reported by the caller as a quality figure
    for the master dark, but no longer sets the threshold.

    Returns (mask, n_hot, n_cold). Cold (stuck-low) pixels are flagged too;
    they are equally wrong in the output and cost nothing extra to find.
    """
    if method == 'local':
        dev = dark_adu - bayer_plane_median3(dark_adu, pattern)
    elif method == 'global':
        dev = dark_adu
    else:
        raise ValueError(f"unknown defect method {method!r}; expected 'local' or 'global'")
    hot  = np.zeros(dark_adu.shape, dtype=bool)
    cold = np.zeros(dark_adu.shape, dtype=bool)
    for r in range(2):
        for c in range(2):
            v   = dev[r::2, c::2]
            med = np.median(v)
            # MAD -> Gaussian sigma. Robust by construction: the defects are the
            # outliers, and they must not be allowed to set the threshold meant
            # to catch them.
            sigma = 1.4826 * np.median(np.abs(v - med))
            if sigma <= 0:
                sigma = max(resid_adu, 1e-9)      # degenerate (e.g. synthetic) data
            hot[r::2, c::2]  = v > med + n_sigma * sigma
            cold[r::2, c::2] = v < med - n_sigma * sigma
    return (hot | cold), int(hot.sum()), int(cold.sum())


def _stuck_pixel_map(std_map, pattern, n_sigma, method='local'):
    """
    Locate stuck/dead pixels: temporal std anomalously LOW relative to their
    same-colour neighbours -- not literally std == 0. A stuck pixel need not
    be perfectly flat to be broken (a pinned ADC bit or a saturated node can
    still show a sliver of coupling/quantization noise well below the
    sensor's real read-noise floor), so a robust MAD-derived threshold on
    std_map catches it, the same way _defect_map's cold side catches a level
    that is merely far too low rather than exactly zero.

    Only the cold (low) side is flagged -- the opposite tail (excess temporal
    noise, e.g. RTS) was tested directly in check_noisy_pixels.py and found to
    be a continuous population with no natural cutoff, not a discrete defect,
    so it is deliberately not thresholded here. See STUCK_PIXEL_SIGMA's
    comment.

    Same local/global choice and MAD-derived robust threshold as _defect_map
    -- see its docstring. Returns (mask, n_stuck).
    """
    if method == 'local':
        dev = std_map - bayer_plane_median3(std_map, pattern)
    elif method == 'global':
        dev = std_map
    else:
        raise ValueError(f"unknown defect method {method!r}; expected 'local' or 'global'")
    mask = np.zeros(std_map.shape, dtype=bool)
    for r in range(2):
        for c in range(2):
            v     = dev[r::2, c::2]
            med   = np.median(v)
            sigma = 1.4826 * np.median(np.abs(v - med))
            if sigma <= 0:
                sigma = 1e-9
            mask[r::2, c::2] = v < med - n_sigma * sigma
    return mask, int(mask.sum())


def _defect_sigma_scan(dark_adu, pattern, out, current_sigma, method='local',
                       sided='both'):
    """
    Show where real defects start, so the threshold can be chosen from data
    rather than guessed.

    Because the threshold is MAD-derived, k maps directly onto a
    false-positive rate: on a Gaussian bulk, the chance of clearing k sigma
    happens to erfc(k/sqrt(2)) of healthy pixels for a two-sided test (half
    that for one-sided). On 12.5 MP that is tens of thousands of pixels at
    k=3 but a handful by k=5, so the choice matters far more than it looks.

    Real defects are the EXCESS over that chance expectation -- but a table
    over a token handful of sigma (this used to stop at k=7) can only show
    that excess exists, not whether it is a genuine separate population or
    just more of the same continuous tail. The real test is whether the
    flagged count, followed out to the full range the data actually reaches,
    FLATTENS into a floor before finally dropping to zero at the true
    maximum (a discrete defect population -- a fixed number of genuinely
    broken pixels, nothing past them) or just keeps sloping down with no such
    floor (a continuum -- e.g. dark current shot noise / DSNU's own
    physically continuous spread, which has no natural cutoff to threshold
    at). That is what the second saved plot (<out>_excess.<ext>) shows, on
    log-log axes so a floor is visible even though the counts span many
    orders of magnitude.

    sided: 'both' (flags |residual| > k -- HOT_PIXEL_SIGMA's hot-and-cold
    test), 'hot' (residual > k only) or 'cold' (residual < -k only --
    STUCK_PIXEL_SIGMA's low-noise-only test). Must match what the caller's
    detector actually thresholds, or this answers a different question than
    the one being asked.

    `method` must also match whatever the detector was tuned for ('local'
    subtracts a local same-colour median first, 'global' does not) -- same
    reasoning.
    """
    from math import erfc, sqrt as _sqrt

    if method == 'local':
        dev = dark_adu - bayer_plane_median3(dark_adu, pattern)
    elif method == 'global':
        dev = dark_adu
    else:
        raise ValueError(f"unknown defect method {method!r}; expected 'local' or 'global'")
    z = np.zeros_like(dev, dtype=np.float32)
    for r in range(2):
        for c in range(2):
            v     = dev[r::2, c::2]
            med   = np.median(v)
            sigma = 1.4826 * np.median(np.abs(v - med))
            z[r::2, c::2] = (v - med) / max(sigma, 1e-9)

    if sided == 'both':
        stat, chance_scale = np.abs(z), 1.0
    elif sided == 'hot':
        stat, chance_scale = z, 0.5
    elif sided == 'cold':
        stat, chance_scale = -z, 0.5
    else:
        raise ValueError(f"unknown sided {sided!r}; expected 'both', 'hot' or 'cold'")

    npx      = stat.size
    stat_max = float(np.nanmax(stat))
    k_max    = max(stat_max, float(current_sigma) * 1.5, 8.0)

    print(f"\n  Choosing the threshold ({sided}, excess over chance = likely real)")
    print(f"    {'k':>7} {'flagged':>10} {'by chance':>11} {'excess':>10}")
    print("    " + "-" * 42)
    table_ks = sorted(set(
        [3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 7.0, 8.0, 10.0, 15.0, 20.0,
         round(float(current_sigma), 2), round(k_max, 1)]))
    for k in table_ks:
        if k > k_max + 1e-9:
            continue
        flagged  = int((stat > k).sum())
        expected = chance_scale * erfc(k / _sqrt(2)) * npx
        mark = "  <- current" if abs(k - float(current_sigma)) < 1e-9 else ""
        print(f"    {k:>7.2f} {flagged:>10,} {expected:>11,.0f} "
              f"{max(flagged - expected, 0):>10,.0f}{mark}")

    # Histogram spans the full observed range, not a token window -- a
    # genuinely separate defect cluster then shows as actual gaps in the
    # bars (see mean_histogram.png), distinct from a smooth continuous decay.
    fig, ax = plt.subplots(figsize=(9, 5))
    hi   = max(k_max * 1.05, 12.0)
    lo   = -hi if sided == 'cold' else -8.0
    bins = np.linspace(lo, hi, 400)
    ax.hist(z.ravel(), bins=bins, color="steelblue", alpha=0.85,
            label="measured residual")
    centres = 0.5 * (bins[1:] + bins[:-1])
    gauss = (npx * (bins[1] - bins[0]) *
             np.exp(-centres ** 2 / 2) / np.sqrt(2 * np.pi))
    ax.plot(centres, gauss, "--", color="crimson", linewidth=1.6,
            label="Gaussian (healthy pixels)")
    line_k = -float(current_sigma) if sided == 'cold' else float(current_sigma)
    ax.axvline(line_k, color="k", linewidth=1.2, label=f"current k = {current_sigma}")
    ax.set_yscale("log")
    ax.set_ylim(0.5, None)
    ax.set_xlabel("local residual, in robust sigma", fontsize=11)
    ax.set_ylabel("pixels", fontsize=11)
    ax.set_title("Where do real defects start?  Tail above the Gaussian is the "
                 "defect population", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, linestyle="--")
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")

    # Excess-over-chance curve: the actual test for a knee, out to the full
    # observed range. Log-log so a floor is visible however far it sits.
    out = Path(out)
    ks = np.geomspace(2.0, k_max, 80)
    flagged_curve  = np.array([int((stat > k).sum()) for k in ks], dtype=np.float64)
    expected_curve = chance_scale * np.array([erfc(k / _sqrt(2)) for k in ks]) * npx

    out_excess = out.with_name(out.stem + "_excess" + out.suffix)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(ks, np.maximum(flagged_curve, 0.5), color="steelblue", linewidth=1.8,
            label="flagged (measured)")
    ax.plot(ks, np.maximum(expected_curve, 0.5), "--", color="crimson", linewidth=1.4,
            label="expected by chance (Gaussian)")
    ax.axvline(float(current_sigma), color="k", linewidth=1.0,
               label=f"current k = {current_sigma}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("threshold k (robust sigma)", fontsize=11)
    ax.set_ylabel("pixel count (log)", fontsize=11)
    ax.set_title("Flagged population vs threshold -- a floor before the final "
                 "drop is a real defect population; a smooth slope all the "
                 "way down is not", fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, which="both", alpha=0.3, linestyle="--")
    fig.tight_layout()
    fig.savefig(out_excess, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_excess}")


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


def _plot_defect_map(mask, n_hot, n_cold, out, source="master dark",
                     labels=("hot", "cold")):
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.imshow(mask, cmap="gray", interpolation="nearest")
    frac = mask.mean() * 100
    ax.set_title(f"Defect map from {source} — {n_hot} {labels[0]}, "
                 f"{n_cold} {labels[1]} ({frac:.4f}% of pixels)", fontsize=11)
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


def _plot_highpass_crops(crops, out, crop_size, vmax):
    """
    High-pass residual at each checkpoint: temporal noise AND fixed-pattern
    noise together, unlike _plot_halfdiff_crops which cancels FPN out.

    This is the direct visual counterpart of gt_checkpoint_noise.png's two
    curves: the split-half crops (halfdiff) keep shrinking with N, all the way
    down; these crops shrink only as long as temporal noise still dominates,
    then visibly stop changing once fixed-pattern noise is what's left -- the
    same floor the split-half-vs-highpass divergence identifies numerically,
    made visible panel to panel.

    All panels share one colour scale, fixed from the first (noisiest)
    checkpoint, for the same reason as _plot_halfdiff_crops: per-panel
    autoscaling would hide exactly the shrinkage (or lack of it) this exists
    to show.
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
                     label="high-pass residual [calibrated]")
    fig.suptitle(
        f"Temporal noise + FPN — high-pass residual, {crop_size}×{crop_size} "
        f"centre crop, shared colour scale (does not cancel FPN, unlike "
        f"split-half)",
        fontsize=12, y=1.02)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def _stills_reference(directory, shape, match_to=None):
    """
    Reference frame from the gain=1 stills, calibrated with the stills' OWN
    metadata.

    Uses STILLS_FRAMES stills (default 1, i.e. a single frame). The point of
    this reference is to show what one clean low-gain capture looks like, as an
    independent check on the averaged GT -- averaging several would make it a
    second aggregate rather than the independent single-shot comparison it is
    meant to be, and would blur anything that shifted between them.

    Calibrated with the stills' own black and white levels: this is a separate
    capture at a different gain and exposure, so using the sequence's values
    would apply the wrong pedestal and scale.

    If match_to is given, the result is rescaled by the ratio of medians so an
    exposure mismatch between the two capture settings does not dominate every
    difference map. The ratio is taken over pixels above a low threshold, since
    near-black pixels give an unstable ratio.
    """
    fmt, paths, pattern, black, white, loader = _make_loaders(directory)
    n = max(1, min(int(STILLS_FRAMES or 1), len(paths)))
    if n == 1:
        accum = loader(paths[0]).astype(np.float64)
    else:
        accum = None
        for _p, frame in progress(prefetch(paths[:n], loader, LOAD_WORKERS),
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


def _report_drift(shifts, out, max_drift_px=None):
    """
    Per-frame position over the run, and what its spread costs the average.

    Averaging frames that do not sit on top of each other convolves the result
    with the distribution of their positions -- the average is blurred by
    exactly the amount the camera wandered, and no number of extra frames fixes
    it. This is the first thing to check when a GT frame looks soft, because it
    is the only cause on the list that averaging makes worse rather than better.

    The estimator carries about 0.3 px of per-frame noise (see DRIFT_WARN_PX),
    which inflates the spread; a spread at or under that is consistent with a
    perfectly static sequence, so the printed verdict is deliberately cautious
    below DRIFT_WARN_PX.

    `shifts` is the full per-frame record, including any frame MAX_DRIFT_PX
    (`max_drift_px` here) went on to exclude from the actual average -- the
    verdict and the "spread" quoted in the path panel are computed on the KEPT
    frames only (what's actually blurring the result), while every panel plots
    the full record with excluded frames marked, so it is visible what got cut
    and why.
    """
    mag     = np.hypot(shifts[:, 0], shifts[:, 1])
    dropped = (mag > max_drift_px) if max_drift_px is not None else np.zeros(len(mag), bool)
    kept    = shifts[~dropped]

    d = drift_report(kept if len(kept) else shifts)
    print("\n  Frame-to-frame drift (registration on a Bayer sub-plane crop)")
    if max_drift_px is not None:
        print(f"    excluded (> {max_drift_px} px)        : "
              f"{int(dropped.sum())}/{len(mag)}  "
              f"({100 * dropped.mean():.1f}%)")
    print(f"    RMS spread about the centroid : {d['spread_px']:.2f} px"
          + ("  (kept frames only)" if dropped.any() else ""))
    print(f"    first frame to last           : {d['total_px']:.2f} px")
    print(f"    largest excursion             : {d['excursion_px']:.2f} px")
    # Blur from averaging over a spread s is roughly a Gaussian of that width;
    # these are the measured sharpness costs on a synthetic text target.
    for px, cost in ((2.0, 39), (1.0, 14), (0.5, 4)):
        if d['spread_px'] >= px:
            print(f"    => the average is being blurred; at this spread a "
                  f"synthetic text target lost about {cost}% of its sharpness.")
            print(f"       Steady the camera, or align frames before averaging.")
            break
    else:
        print(f"    => at or below the estimator's own noise floor -- "
              f"consistent with a static sequence.")
        print(f"       If the frame still looks soft, the cause is upstream "
              f"(focus, motion blur within a frame, optics), not the averaging.")

    fig, axes = plt.subplots(1, 3, figsize=(19, 4.6))
    idx = np.arange(len(shifts))
    axes[0].plot(idx, shifts[:, 1], linewidth=0.9, label="x (col)")
    axes[0].plot(idx, shifts[:, 0], linewidth=0.9, label="y (row)")
    if dropped.any():
        axes[0].scatter(idx[dropped], shifts[dropped, 1], s=14, c="crimson",
                        zorder=3, label="excluded")
        axes[0].scatter(idx[dropped], shifts[dropped, 0], s=14, c="crimson",
                        zorder=3)
    axes[0].axhline(0, color="k", linewidth=0.8)
    axes[0].set_xlabel("frame"); axes[0].set_ylabel("shift vs frame 0 (px)")
    axes[0].set_title("Drift over the run"); axes[0].legend(fontsize=9)
    axes[0].grid(alpha=0.3)

    axes[1].plot(shifts[:, 1], shifts[:, 0], linewidth=0.7, alpha=0.8)
    if dropped.any():
        axes[1].scatter(shifts[dropped, 1], shifts[dropped, 0], s=14,
                        c="crimson", zorder=3, label="excluded")
    axes[1].scatter(*shifts[0, ::-1], s=40, c="seagreen", zorder=4, label="first")
    axes[1].scatter(*shifts[-1, ::-1], s=40, c="darkorange", zorder=4, label="last")
    axes[1].set_xlabel("x shift (px)"); axes[1].set_ylabel("y shift (px)")
    axes[1].set_title(f"Path  (RMS spread {d['spread_px']:.2f} px"
                      f"{', kept only' if dropped.any() else ''})")
    axes[1].set_aspect("equal", adjustable="datalim")
    axes[1].legend(fontsize=9); axes[1].grid(alpha=0.3)

    axes[2].hist(mag, bins=min(40, max(10, len(mag) // 3)),
                color="steelblue", alpha=0.85, label="all frames")
    axes[2].axvline(DRIFT_WARN_PX, color="gray", linestyle="--", linewidth=1.2,
                    label=f"estimator floor ({DRIFT_WARN_PX} px)")
    if max_drift_px is not None:
        axes[2].axvline(max_drift_px, color="crimson", linewidth=1.4,
                        label=f"MAX_DRIFT_PX ({max_drift_px} px)")
    axes[2].set_xlabel("drift magnitude vs frame 0 (px)")
    axes[2].set_ylabel("frames")
    axes[2].set_title("Drift magnitude distribution")
    axes[2].legend(fontsize=9); axes[2].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")
    return d



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

    `crops` are scene-linear; all panels are encoded on one shared gain (the
    largest that clips none of them) so a difference between panels is a
    difference in the data, not in how each was exposed.
    """
    n_panels = len(crops)
    gain = uniform_gain([c for _, c in crops], percentile=GAIN_PERCENTILE)
    fig, axes = plt.subplots(1, n_panels, figsize=(4.2 * n_panels, 4.8))
    if n_panels == 1:
        axes = [axes]
    for ax, (idx, crop) in zip(axes, crops):
        ax.imshow(encode_rgb(crop, gain=gain), interpolation="nearest")
        ax.set_title(f"N = {idx}", fontsize=11)
        ax.axis("off")
    fig.suptitle(
        f"Running mean vs frames averaged — {crop_size}×{crop_size} centre crop "
        f"at 100% zoom, shared display gain ×{gain:.2f} "
        f"(noise should fall as 1/√N)",
        fontsize=12, y=1.02)
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
    # A capped run is a different experiment from a full one, and a global-
    # defect-method run a different one from local, so each gets its own
    # directory rather than silently overwriting the other's result -- exactly
    # what comparing the two methods needs.
    cap      = f"_max{max_frames}" if max_frames else ""
    dmethod  = f"_defect{DEFECT_METHOD}" if DEFECT_METHOD != "local" else ""
    dmax     = f"_maxdrift{MAX_DRIFT_PX}" if MAX_DRIFT_PX is not None else ""
    seq_out  = out_dir / (seq_name + (RUN_SUFFIX or "") + cap + dmethod + dmax)
    seq_out.mkdir(parents=True, exist_ok=True)

    rev = git_revision()
    if rev:
        print(f"  Code: {rev['short']} on {rev['branch']}"
              + ("  (WORKING TREE DIRTY)" if rev['dirty'] else "")
              + f"  — {rev['subject']}")

    fmt, paths, pattern, black, white, loader = _make_loaders(directory)
    if max_frames is not None:
        paths = paths[:max_frames]
    n = len(paths)
    print(f"  {n} {fmt.upper()} frames  |  white={white}  black={black[0]}")

    # DNG carries a real camera white-balance + color-correction profile;
    # GN3 has none, so demosaic_linear falls back to gray-world WB / no CCM.
    wb, ccm = (get_color_metadata(paths[0]) if fmt == 'dng' else (None, None))

    # Pass 1: full streaming mean, snapshotting the running mean along the way.
    # Checkpoints are log-spaced because noise falls as 1/sqrt(N) -- linear
    # spacing would put every panel in the flat tail and show no visible change.
    ckpt_ns = sorted(set(np.geomspace(1, n, min(N_CHECKPOINTS, n))
                         .astype(int).tolist()) | {n})
    ckpt_crops, half_crops, hp_crops = [], [], []
    half_vmax = []            # one-element cell: shared scale, set on first diff
    hp_vmax   = []            # same, for the high-pass crops

    def _crop_centre(a, size):
        cy, cx = a.shape[0] // 2, a.shape[1] // 2
        h = min(size, a.shape[0]) // 2
        w = min(size, a.shape[1]) // 2
        return a[cy - h: cy + h, cx - w: cx + w].copy()

    def _on_checkpoint(idx, running, half_diff):
        # Only the crop is kept; the full-size running-mean PNGs were dropped
        # once the convergence question was settled. The demosaic still runs on
        # the whole frame -- demosaicing a crop alone changes what the
        # interpolation sees at the crop border.
        #
        # The crop is stored scene-linear and encoded later, once all the
        # checkpoints are in and they can share one display gain. Encoding each
        # as it arrives would give the N=1 panel its own exposure and stop the
        # panels being comparable, which is the entire point of the figure.
        lin = demosaic_linear(running, pattern, wb, ccm)
        ckpt_crops.append((idx, _crop_centre(lin, CHECKPOINT_CROP)))

        # High-pass residual of the running mean itself: temporal noise AND
        # FPN together, computed on the full frame (never a crop alone -- the
        # box filter needs real neighbours at the crop border) then cropped
        # for display, same reasoning as the running-mean crop above.
        hp = highpass_residual(running, pattern)
        if not hp_vmax:
            hp_vmax.append(max(float(np.percentile(np.abs(hp), 99.5)), 1e-9))
        # highpass_std(running, ...), not hp.std(): the label should match the
        # exact number gt_checkpoint_noise.png plots and _print_checkpoint_table
        # prints (per-sub-plane std, averaged over the four), not a single std
        # over the reassembled residual, which is a slightly different quantity.
        hp_crops.append((idx, _crop_centre(hp, CHECKPOINT_CROP),
                         highpass_std(running, pattern)))

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
    if hp_crops:
        _plot_highpass_crops(hp_crops,
                             seq_out / f"gt_highpass_comparison_N{n}.png",
                             CHECKPOINT_CROP, hp_vmax[0])
    _print_checkpoint_table(ckpt_metrics)
    _plot_checkpoint_noise(ckpt_metrics,
                           seq_out / f"gt_checkpoint_noise_N{n}.png")
    _report_stationarity(stationarity, white, black,
                         seq_out / f"gt_stationarity_N{n}.png")
    if stationarity.get('shifts') is not None:
        _report_drift(stationarity['shifts'], seq_out / f"gt_drift_N{n}.png",
                     max_drift_px=MAX_DRIFT_PX)

    # Median + trimmed mean (disk-backed, no full-stack RAM alloc)
    median_frame = trimmed_frame = None
    n_stack = min(max_stack, n) if ROBUST_AGGREGATORS else 0
    if ROBUST_AGGREGATORS:
        tmp_path = seq_out / "_stack_tmp.npy"
        trim_k   = max(1, int(TRIM_FRAC / 2 * n_stack))
        print(f"  Writing {n_stack}/{n} frames to disk, then computing "
              f"median and trimmed mean (trim_k={trim_k}) …")
        median_frame, trimmed_frame = _compute_median_and_trimmed(
            paths, n_stack, pattern, black, white, loader, TRIM_FRAC, tmp_path,
        )

    # Sigma-clipped mean (streaming, all N frames -- reuses the same two-pass
    # clip the master dark is built with, see SIGMA_CLIP_MEAN's comment).
    sigma_clip_frame = None
    if SIGMA_CLIP_MEAN:
        print(f"  Sigma-clipped mean (±{SIGMA_CLIP_MEAN}σ, {n} frames) …")
        sc_adu, sc_n = _dark_master(paths, n, loader, SIGMA_CLIP_MEAN)
        sigma_clip_frame = calibrate_frame(sc_adu, pattern, black, white)

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
    crop_dir = None  # set below, inside the comparison block, if CROP_XY/CROP_SIZE

    def _save_three_gains(lin_rgb, gain, own_gain, base):
        save_rgb_png(encode_rgb(lin_rgb),
                     base.with_name(base.name + "_nogain.png"))
        save_rgb_png(encode_rgb(lin_rgb, gain=gain),
                     base.with_name(base.name + "_uniform.png"))
        save_rgb_png(encode_rgb(lin_rgb, gain=own_gain),
                     base.with_name(base.name + "_max.png"))

    def save_candidate(frame, lin_rgb, gain, base):
        # Three PNGs per candidate, all gamma-encoded, none auto-brightened:
        #   _nogain  -- gain 1.0, the scene exactly as calibrated. Dark for a
        #               lowlight capture, but it is the honest picture and the
        #               only one whose pixel values mean something absolute.
        #   _uniform -- one gain shared by every candidate in this run, set by
        #               GAIN_PERCENTILE rather than the literal max so a lone
        #               outlier pixel in one candidate can't hold every
        #               candidate's brightness down (measured: 3 outlier
        #               pixels out of 245,760 cost 36% of the achievable
        #               brightness before this). Still comparable frame to
        #               frame: brighter here really is brighter.
        #   _max     -- this candidate's OWN version of that gain, ignoring
        #               every other candidate. As bright as this one image can
        #               get -- but that breaks comparability:
        #               if one candidate has a dimmer peak (say defect repair
        #               removed its brightest hot pixel), its _max gain is
        #               higher than its neighbours', so two _max PNGs sitting
        #               side by side can look equally bright even when one
        #               scene is genuinely dimmer. Use _uniform to compare
        #               candidates, _max to look at just one on its own.
        own_gain = uniform_gain([lin_rgb], percentile=GAIN_PERCENTILE)
        _save_three_gains(lin_rgb, gain, own_gain, base)
        if crop_dir is not None:
            cx, cy = CROP_XY
            crop = lin_rgb[cy:cy + CROP_SIZE, cx:cx + CROP_SIZE]
            _save_three_gains(crop, gain, own_gain, crop_dir / base.name)
        if SAVE_NPY:
            np.save(base.with_suffix('.npy'), frame.astype(np.float32))
            print(f"Saved {base.with_suffix('.npy')}")
        if SAVE_DNG:
            save_dng(uncalibrate_frame(frame, pattern, black, white),
                     base.with_suffix('.dng'), pattern, black, white,
                     color_matrix=dng_ccm, as_shot_neutral=neutral,
                     model=f"noise_analysis GT ({seq_name})",
                     copy_tags=dng_tags)

    # ---------------------------------------------------------------- #
    # Dark subtraction                                                   #
    # ---------------------------------------------------------------- #
    dark_corrected = None        # dark-subtracted only, never reassigned below
    dark_corrected_fixed = None  # dark-subtracted + defects interpolated, if any
    mean_defect_fixed = None
    if DARK_DIR:
        print(f"\n  Dark frames: {DARK_DIR}")
        _, d_paths, d_pattern, _, _, d_loader = _make_loaders(DARK_DIR)
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
        _report_dark_uniformity(dark_adu, pattern,
                                seq_out / f"gt_dark_uniformity_N{d_stack}.png")
        dark_corrected = _dark_correct(full_mean_adu, dark_adu, pattern, black, white)

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
        mask = np.zeros(dark_adu.shape, dtype=bool)
        n_hot = n_cold = 0
        if HOT_PIXEL_SIGMA and resid is not None:
            resid_adu = resid * float(white - black[0])
            mask, n_hot, n_cold = _defect_map(dark_adu, pattern, resid_adu,
                                              HOT_PIXEL_SIGMA, method=DEFECT_METHOD)
            frac = mask.mean() * 100
            print(f"\n  Defect map ({DEFECT_METHOD}, >{HOT_PIXEL_SIGMA}σ"
                  f"{' from same-colour neighbours' if DEFECT_METHOD == 'local' else ''} "
                  f"in the master dark): "
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
            _defect_sigma_scan(dark_adu, pattern,
                               seq_out / "gt_defect_sigma_scan.png",
                               HOT_PIXEL_SIGMA, method=DEFECT_METHOD, sided='both')
            if mask.any():
                _plot_defect_map(mask, n_hot, n_cold,
                                 seq_out / "gt_defect_map.png")
                np.save(seq_out / "gt_defect_map.npy", mask)

        # Stuck/dead pixels: flagged from the dark sequence's TEMPORAL STD
        # rather than its level, so this runs independently of HOT_PIXEL_SIGMA
        # -- a stuck pixel can have a perfectly ordinary mean. See
        # STUCK_PIXEL_SIGMA's comment for why only the low-noise side is
        # thresholded.
        if STUCK_PIXEL_SIGMA:
            print(f"\n  Temporal std map from "
                  f"{min(n_dark_want, len(d_paths))}/{len(d_paths)} frames …")
            std_map = temporal_std_map(d_paths[:n_dark_want], d_loader, LOAD_WORKERS)
            stuck_mask, n_stuck = _stuck_pixel_map(
                std_map, pattern, STUCK_PIXEL_SIGMA, method=DEFECT_METHOD)
            print(f"  Stuck-pixel map ({DEFECT_METHOD}, temporal std >"
                  f"{STUCK_PIXEL_SIGMA}σ below same-colour neighbours): "
                  f"{n_stuck} pixels, {stuck_mask.mean() * 100:.4f}% of pixels")
            _defect_sigma_scan(std_map, pattern,
                               seq_out / "gt_stuck_pixel_sigma_scan.png",
                               STUCK_PIXEL_SIGMA, method=DEFECT_METHOD, sided='cold')
            if stuck_mask.any():
                _plot_defect_map(stuck_mask, 0, n_stuck,
                                 seq_out / "gt_stuck_pixel_map.png",
                                 source="dark-sequence temporal std map",
                                 labels=("(n/a)", "stuck"))
                np.save(seq_out / "gt_stuck_pixel_map.npy", stuck_mask)
            mask = mask | stuck_mask

        if mask.any():
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

            print(f"\n  High-pass std   mean={hp_before:.6f}  "
                  f"mean+defectfix={highpass_std(mean_fixed, pattern):.6f}")
            print(f"                  darksub={hp_after:.6f}  "
                  f"darksub+defectfix={highpass_std(dark_fixed, pattern):.6f}")
            mean_defect_fixed    = mean_fixed
            dark_corrected_fixed = dark_fixed

    # ---------------------------------------------------------------- #
    # Comparison of GT candidates                                        #
    # ---------------------------------------------------------------- #
    # Ordered so the pure aggregators come first and each correction is added
    # on top of the plain mean, which makes the difference maps below read as
    # "what did this step change?" rather than an arbitrary pairing.
    # (frame, human label for figures, filename slug). The slug is explicit
    # rather than derived from the label -- deriving it produced names like
    # "cmp_Mean_+_defects_interpolated.png".
    cands = [(full_mean, f"Mean  (N={n})", "mean")]
    if median_frame is not None:
        cands += [(median_frame,  f"Median  (N={n_stack})",       "median"),
                  (trimmed_frame, f"Trimmed mean  (N={n_stack})", "trimmed_mean")]
    if sigma_clip_frame is not None:
        cands.append((sigma_clip_frame,
                      f"Sigma-clipped mean, ±{SIGMA_CLIP_MEAN}σ  (N={n})",
                      "sigma_clip_mean"))
    if mean_defect_fixed is not None:
        cands.append((mean_defect_fixed,
                      f"Mean + defects interpolated  (N={n})", "mean_defectfix"))
    if dark_corrected is not None:
        cands.append((dark_corrected, f"Mean − dark  (N={n})", "mean_darksub"))
    if dark_corrected_fixed is not None:
        cands.append((dark_corrected_fixed,
                      f"Mean − dark + defects interpolated  (N={n})",
                      "mean_darksub_defectfix"))
    if STILLS_DIR:
        print(f"\n  Stills: {STILLS_DIR}")
        still, n_still, scale = _stills_reference(
            STILLS_DIR, full_mean.shape,
            match_to=full_mean if MATCH_STILL_INTENSITY else None)
        print(f"  {n_still} still{'s averaged' if n_still > 1 else ''}"
              + (f", intensity-matched by ×{scale:.4f}" if MATCH_STILL_INTENSITY
                 else ", no intensity matching"))
        cands.append((still, f"Gain=1 still  (N={n_still})", "still_gain1"))

    if len(cands) > 1:
        cmp_dir = seq_out / "comparison"
        cmp_dir.mkdir(parents=True, exist_ok=True)
        if CROP_XY and CROP_SIZE:
            crop_dir = cmp_dir / "crops"
            crop_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n  Comparing {len(cands)} GT candidates …")

        frames = [c[0] for c in cands]
        labels = [c[1] for c in cands]
        slugs  = [c[2] for c in cands]
        # No summary grids here any more -- the per-candidate files below are
        # full resolution, and a tiled figure of six downsampled panels never
        # showed anything the files themselves do not.
        #
        # Two passes, because the shared display gain cannot be known until
        # every candidate has been demosaiced: it is set by the brightest pixel
        # in the whole set. The linear frames are cached in a memmap rather than
        # held in RAM (a 12 MP RGB float32 frame is 144 MB, and the demosaic
        # itself already peaks around 2.5 GB) and rather than demosaiced twice
        # (that is the expensive step, ~27 s a frame).
        lin_path = cmp_dir / "_linear_tmp.npy"
        lin = np.lib.format.open_memmap(
            lin_path, mode='w+', dtype=np.float32,
            shape=(len(frames), *frames[0].shape, 3))
        try:
            for i, f in progress(list(enumerate(frames)), desc="  demosaic",
                                 total=len(frames)):
                lin[i] = demosaic_linear(f, pattern, wb, ccm)
            gain = uniform_gain(lin, percentile=GAIN_PERCENTILE)
            clip_note = (f"puts the {GAIN_PERCENTILE}th percentile at full "
                        f"scale, clipping only above it"
                        if GAIN_PERCENTILE < 100 else "clips nothing")
            print(f"  Uniform display gain: ×{gain:.3f} ({clip_note}); "
                  f"plus an ungained and a per-candidate-max version of each")
            for i, (f, slug) in enumerate(zip(frames, slugs)):
                save_candidate(f, lin[i], gain, cmp_dir / f"cmp_{slug}")
        finally:
            del lin
            lin_path.unlink(missing_ok=True)

        print("\n  Residual pixel noise (high-pass std, calibrated units)")
        width = max(len(l) for l in labels) + 2
        base  = highpass_std(frames[0], pattern)
        for f, lab in zip(frames, labels):
            hp = highpass_std(f, pattern)
            print(f"    {lab:<{width}s} {hp:.6f}   {base / hp:5.2f}x vs mean")
        print()

    _write_run_info(seq_out, directory, fmt, n, n_stack, rev,
                    n_dropped=stationarity.get('n_dropped', 0))
    print(f"\nDone. Outputs in {seq_out.resolve()}")


# --------------------------------------------------------------------------- #
# CLI overrides + entry point                                                   #
# --------------------------------------------------------------------------- #

def _write_run_info(seq_out, directory, fmt, n, n_stack, rev, n_dropped=0):
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
            "dropped_frames": n_dropped,   # excluded by MAX_DRIFT_PX, see gt_drift_*.png
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
    # demosaic_linear() is called from several places here with no method
    # argument; setting the library default once is simpler than threading it
    # through all of them, and keeps DEMOSAIC in the CONFIG block that
    # run_info.json captures.
    raw_utils.DEMOSAIC_METHOD = DEMOSAIC
    # Fail here, not at the first preview many minutes into the run, and print
    # what actually ran -- an unavailable method raises rather than quietly
    # demosaicing with something else, so this line and run_info.json agree.
    raw_utils.check_demosaic_method()
    print(f"GT sequence analysis: {SEQUENCE_DIR}")
    print(f"  Demosaic: {DEMOSAIC}")
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
