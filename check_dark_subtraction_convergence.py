"""
Does dark subtraction change WHERE the split-half and high-pass noise curves
diverge -- or just how far apart they end up?

gt_checkpoint_noise.png (from analyze_gt_sequence.py) plots two curves against
frames averaged: split-half (temporal noise only) and high-pass (temporal
noise + fixed-pattern noise together). They diverge once FPN starts to
dominate, because split-half keeps falling while high-pass flattens at a
floor. That floor is what dark subtraction is for -- but the main script only
dark-subtracts once, at the very end, using the finished average, so there is
no existing way to see whether subtraction actually raises the point where the
curves diverge without rerunning the whole pipeline.

The floor itself, in a dark frame, is DSNU (Dark Signal Non-Uniformity): each
pixel's dark-current generation rate varies slightly due to manufacturing
variation, and that variation is a fixed, pixel-specific pattern rather than
random from frame to frame -- it is imprinted identically on every exposure
taken under matching conditions (exposure time, gain, temperature), which is
exactly why it survives averaging the light sequence and has to be measured
and subtracted separately instead [1][2].

This script answers that on its own, streaming the light sequence exactly
once: at each log-spaced checkpoint it computes the running mean's high-pass
residual with no correction, with the master dark subtracted, and (if
HOT_PIXEL_SIGMA is set) with hot pixels also interpolated afterward -- the
same two-part correction the main pipeline applies, subtraction then
interpolation, tested together rather than assumed to both be pulling their
weight.

That third curve matters because subtraction and interpolation fix different
things. Subtraction removes the bulk, smooth part of DSNU; a genuine hot
pixel -- anomalous relative to its own local neighbourhood, not merely
offset from it -- is not a pattern subtraction can cleanly remove, and needs
individual detection and reconstruction instead. If dark-subtracted high-pass
plateaus well above split-half but adding interpolation closes most of that
gap, the remaining floor was hot pixels, not smooth DSNU; if interpolation
barely moves it, something subtraction and interpolation both cannot reach
is responsible instead -- most likely a capture-condition mismatch between
the dark and light sequences (Section 5.3.2's second, non-reducible residual
source), or a defect population too dense or too heavy-tailed for the current
HOT_PIXEL_SIGMA to isolate cleanly.

One shortcut worth knowing, because it halves the work here: split-half
is mathematically UNCHANGED by dark subtraction (and by interpolation, for
the same reason). It is a difference of two independent sub-averages,
(even - odd); anything common to both -- the fixed dark-current pattern, or a
per-pixel correction applied identically to both halves -- cancels out of the
difference exactly. So there is only one temporal curve to compute regardless
of how many correction curves are added -- which is itself worth confirming
against the plot this script produces.

Edit the CONFIG block below, then run:
    python check_dark_subtraction_convergence.py

Output (saved to OUTPUT_DIR/<sequence_name>/):
  - dark_subtraction_convergence.png : split-half, and high-pass with no
    correction / dark-subtracted / dark-subtracted + interpolated (the last
    only if HOT_PIXEL_SIGMA is set), plotted together against frames
    averaged, log-log, with a 1/sqrt(N) reference. Where each correction
    curve pulls away from the ones before it and tracks split-half further
    out is the point that correction earned its keep.
  - the same data printed as a table.

[1] J. R. Janesick, "Photon Transfer: DN -> lambda", SPIE Press Monograph
    PM170, 2007. DOI: 10.1117/3.725073.
[2] EMVA Standard 1288 -- "Standard for Characterization of Image Sensors and
    Cameras", European Machine Vision Association, Release 3.1.
    https://www.emva.org/standards-technology/emva-1288/
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from raw_utils import (
    detect_format, find_dngs, find_raws,
    get_raw_metadata, get_raw_metadata_gn3,
    load_raw, load_raw_gn3,
    calibrate_frame, highpass_std,
    progress, prefetch, format_duration,
)
import analyze_gt_sequence as gt_seq

# --------------------------------------------------------------------------- #
# CONFIG -- edit paths and options here
# --------------------------------------------------------------------------- #

SEQUENCE_DIR = "/path/to/light/frames"
DARK_DIR     = "/path/to/dark/frames"
OUTPUT_DIR   = "./dark_subtraction_convergence_output"

GN3_BLACK_LEVEL  = 256
MAX_FRAMES       = None   # cap the light sequence, None = every frame found
N_CHECKPOINTS    = 8      # log-spaced checkpoints -- more than the main
                          # script's default, since this is cheap (no
                          # demosaic, no defect map, no DNG/PNG writes)
DARK_MAX_FRAMES  = None   # cap the darks used to build the master, None = all
DARK_SIGMA_CLIP  = 4.0    # see analyze_gt_sequence.py's DARK_SIGMA_CLIP comment
HOT_PIXEL_SIGMA  = 5.0    # see analyze_gt_sequence.py's HOT_PIXEL_SIGMA comment
                          # -- 0 or None skips the third (interpolated) curve
DEFECT_METHOD    = "local"  # see analyze_gt_sequence.py's DEFECT_METHOD comment
DEFECT_FILL      = "median" # see analyze_gt_sequence.py's DEFECT_FILL comment
LOAD_WORKERS     = 4      # threads used to decode frames ahead of the pass


def _make_loader(directory: str):
    """Detect format and return (fmt, paths, pattern, black, white, loader)."""
    fmt = detect_format(directory)
    if fmt == 'dng':
        paths = find_dngs(directory)
        pattern, black, white = get_raw_metadata(paths[0])
        loader = load_raw
    else:
        paths = find_raws(directory)
        pattern, black, white = get_raw_metadata_gn3(paths[0], GN3_BLACK_LEVEL)
        meta  = json.loads(paths[0].with_suffix('.imgprops').read_text())
        shape = (meta['height'], meta['width'])
        loader = lambda p, _s=shape: load_raw_gn3(p, _s)
    return fmt, paths, pattern, black, white, loader


def _none_or_auto(s: str):
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


_UNSET = object()


def _apply_cli_overrides() -> None:
    g = globals()
    scalar = (bool, int, float, str, type(None))
    keys = sorted(k for k in g if k.isupper() and not k.startswith('_') and isinstance(g[k], scalar))
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    for key in keys:
        val = g[key]
        flag = '--' + key.lower().replace('_', '-')
        if isinstance(val, bool):
            parser.add_argument(flag, dest=key, default=_UNSET,
                                action=argparse.BooleanOptionalAction, help=f"(default: {val})")
        elif isinstance(val, int):
            parser.add_argument(flag, dest=key, type=int, default=_UNSET, metavar='N', help=f"(default: {val})")
        elif isinstance(val, float):
            parser.add_argument(flag, dest=key, type=float, default=_UNSET, metavar='F', help=f"(default: {val})")
        else:
            parser.add_argument(flag, dest=key, type=_none_or_auto, default=_UNSET, metavar='S',
                                help=f"(default: {val!r}; pass 'none' to clear)")
    args = parser.parse_args()
    for key, new_val in vars(args).items():
        if new_val is not _UNSET:
            g[key] = new_val


def _plot_convergence(rows, out, has_interp):
    ns   = np.array([r['n'] for r in rows], dtype=float)
    temp = np.array([r['temporal'] for r in rows], dtype=float)
    hp_u = np.array([r['highpass_uncorrected'] for r in rows], dtype=float)
    hp_c = np.array([r['highpass_corrected'] for r in rows], dtype=float)

    fig, ax = plt.subplots(figsize=(9, 6))
    ok = np.isfinite(temp)
    if ok.any():
        ax.loglog(ns[ok], temp[ok], "o-", color="steelblue", linewidth=1.8,
                  label="split-half (temporal only -- unaffected by dark correction)")
    ax.loglog(ns, hp_u, "s-", color="darkorange", linewidth=1.8,
              label="high-pass, no correction")
    ax.loglog(ns, hp_c, "^-", color="seagreen", linewidth=1.8,
              label="high-pass, dark-subtracted")
    if has_interp:
        hp_i = np.array([r['highpass_corrected_interp'] for r in rows], dtype=float)
        ax.loglog(ns, hp_i, "d-", color="crimson", linewidth=1.8,
                  label="high-pass, dark-subtracted + hot pixels interpolated")
    if ok.any() and temp[ok][0] > 0:
        ref_n = ns[ok]
        ax.loglog(ref_n, temp[ok][0] * np.sqrt(ns[ok][0]) / np.sqrt(ref_n),
                  "--", color="gray", linewidth=1.3, label=r"ideal $\propto 1/\sqrt{N}$")
    ax.set_xlabel("frames averaged (N)")
    ax.set_ylabel("noise (calibrated units)")
    ax.set_title("Does dark subtraction move where high-pass diverges from split-half?")
    ax.legend(fontsize=9)
    ax.grid(True, which="both", alpha=0.3, linestyle="--")
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def main():
    _apply_cli_overrides()
    gt_seq.LOAD_WORKERS = LOAD_WORKERS   # see check_dark_frames.py for why this is needed
    t0 = time.monotonic()

    print(f"Sequence: {SEQUENCE_DIR}")
    fmt, paths, pattern, black, white, loader = _make_loader(SEQUENCE_DIR)
    if MAX_FRAMES is not None:
        paths = paths[:MAX_FRAMES]
    n = len(paths)
    print(f"  {n} {fmt.upper()} frames  |  white={white}  black={black[0]}")

    print(f"\nDark frames: {DARK_DIR}")
    _, d_paths, d_pattern, _, _, d_loader = _make_loader(DARK_DIR)
    if not np.array_equal(d_pattern, pattern):
        raise SystemExit(f"Dark frames have Bayer pattern {d_pattern.tolist()}, "
                         f"sequence has {pattern.tolist()} -- refusing to subtract.")
    n_dark_want = DARK_MAX_FRAMES or len(d_paths)
    dark_adu, d_stack = gt_seq._dark_master(d_paths, n_dark_want, d_loader, DARK_SIGMA_CLIP)
    print(f"  Master dark ADU: mean={dark_adu.mean():.2f}  min={dark_adu.min():.2f}  "
          f"max={dark_adu.max():.2f}  (from {d_stack} frames)")

    # Defect mask, built once from the master dark -- same detection the main
    # pipeline uses, so the third curve below tests the SAME two-part
    # correction (subtract, then interpolate) rather than a different one.
    mask = None
    if HOT_PIXEL_SIGMA:
        resid = gt_seq._dark_master_residual(d_paths, d_stack, d_loader, black, white)
        if resid is not None:
            resid_adu = resid * float(white - black[0])
            mask, n_hot, n_cold = gt_seq._defect_map(dark_adu, pattern, resid_adu,
                                                     HOT_PIXEL_SIGMA, method=DEFECT_METHOD)
            print(f"  Defect map ({DEFECT_METHOD}, >{HOT_PIXEL_SIGMA}σ): "
                  f"{n_hot} hot, {n_cold} cold, {mask.mean()*100:.4f}% of pixels")
        else:
            print("  Need at least 2 dark frames for defect detection -- skipping "
                  "the interpolated curve.")
    gt_seq.DEFECT_FILL = DEFECT_FILL   # _interpolate_defects reads this as a
                                       # bare global on ITS module, not this
                                       # script's -- see check_dark_frames.py

    ckpt_ns = sorted(set(np.geomspace(1, n, min(N_CHECKPOINTS, n)).astype(int).tolist()) | {n})
    checkpoints = set(ckpt_ns)
    print(f"\nStreaming mean ({n} frames), checkpoints at N={ckpt_ns} …")

    acc_e = acc_o = None
    n_e = n_o = 0
    rows = []
    for i, (_p, frame) in enumerate(progress(
            prefetch(paths, loader, LOAD_WORKERS), desc="  mean", total=n)):
        idx = i + 1
        raw = frame.astype(np.float64)
        if i % 2 == 0:
            acc_e = raw if acc_e is None else acc_e + raw
            n_e += 1
        else:
            acc_o = raw if acc_o is None else acc_o + raw
            n_o += 1

        if idx not in checkpoints:
            continue
        n_used = n_e + n_o
        total  = acc_e if acc_o is None else acc_e + acc_o
        raw_mean_adu = (total / n_used).astype(np.float32)

        running = calibrate_frame(raw_mean_adu, pattern, black, white)
        hp_uncorrected = highpass_std(running, pattern)

        corrected = gt_seq._dark_correct(raw_mean_adu, dark_adu, pattern, black, white)
        hp_corrected = highpass_std(corrected, pattern)

        hp_corrected_interp = None
        if mask is not None:
            corrected_interp = gt_seq._interpolate_defects(corrected, pattern, mask)
            hp_corrected_interp = highpass_std(corrected_interp, pattern)

        if n_o > 0:
            half_diff = ((acc_e / n_e) - (acc_o / n_o)).astype(np.float32)
            half_diff /= float(white - black[0])
            temporal = float(half_diff.std()) * np.sqrt(n_e * n_o / (n_e + n_o)) / np.sqrt(n_used)
        else:
            temporal = float('nan')   # N=1: no second half to compare

        row = {'n': n_used, 'temporal': temporal,
              'highpass_uncorrected': hp_uncorrected,
              'highpass_corrected': hp_corrected}
        if hp_corrected_interp is not None:
            row['highpass_corrected_interp'] = hp_corrected_interp
        rows.append(row)
    print()

    has_interp = mask is not None
    width = 6
    header = f"{'N':>{width}} {'temporal':>12} {'highpass (raw)':>16} {'highpass (dark-sub)':>20}"
    if has_interp:
        header += f" {'+interpolated':>16}"
    header += f" {'ratio':>8}"
    print(header)
    print("-" * len(header))
    for r in rows:
        ratio = r['highpass_corrected'] / r['highpass_uncorrected'] if r['highpass_uncorrected'] else float('nan')
        t = f"{r['temporal']:.6f}" if np.isfinite(r['temporal']) else "--"
        line = (f"{r['n']:>{width}} {t:>12} {r['highpass_uncorrected']:>16.6f} "
               f"{r['highpass_corrected']:>20.6f}")
        if has_interp:
            line += f" {r['highpass_corrected_interp']:>16.6f}"
        line += f" {ratio:>8.3f}"
        print(line)

    out_dir = Path(OUTPUT_DIR) / Path(SEQUENCE_DIR).name
    out_dir.mkdir(parents=True, exist_ok=True)
    _plot_convergence(rows, out_dir / "dark_subtraction_convergence.png", has_interp)

    print(f"\nTotal time: {format_duration(time.monotonic() - t0)}")
    print(f"Outputs in {out_dir.resolve()}")


if __name__ == "__main__":
    main()
