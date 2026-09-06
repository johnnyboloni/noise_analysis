"""
Dark-frame-only diagnostics: is the master dark uniform, how many dark frames
does it want, and where are the defects?

Runs just the dark half of analyze_gt_sequence.py's pipeline, in isolation --
no light sequence is read or processed, so this is the fast way to iterate on
DARK_SIGMA_CLIP, HOT_PIXEL_SIGMA, or "do I have enough darks" without waiting
on a run over the (much larger) light sequence too. Reuses that script's own
dark-frame functions rather than a second copy of the same logic, so a fix
made there (e.g. the iterated sigma-clip) applies here automatically.

Edit the CONFIG block below, then run:
    python check_dark_frames.py

Outputs (saved to OUTPUT_DIR/<dark_dir_name>/):
  - dark_master.npy       : the master dark itself, raw ADU
  - dark_uniformity.png   : is the master dark spatially flat? full
                            resolution, block-averaged, and row/column median
                            profiles, block-averaged span measured against
                            per-pixel noise. See analyze_gt_sequence.py's
                            _report_dark_uniformity for what each panel shows.
  - "how many dark frames does this master want?" printed to the console,
                            from a single-pair noise estimate -- no need to
                            guess or rerun with different counts.
  - defect_sigma_scan.png : excess-over-chance table for choosing
                            HOT_PIXEL_SIGMA, and the residual histogram
                            against a Gaussian reference
  - defect_map.png/.npy   : hot/cold pixels found at the current
                            HOT_PIXEL_SIGMA (only if any are flagged)
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")

from raw_utils import (
    detect_format, find_dngs, find_raws,
    get_raw_metadata, get_raw_metadata_gn3,
    load_raw, load_raw_gn3,
    highpass_std, format_duration,
)
import analyze_gt_sequence as gt_seq

# --------------------------------------------------------------------------- #
# CONFIG -- edit paths and options here
# --------------------------------------------------------------------------- #

DARK_DIR        = "/path/to/dark/frames"
OUTPUT_DIR      = "./dark_check_output"

GN3_BLACK_LEVEL  = 256   # GN3 has no black-level metadata; DNG ignores this
DARK_MAX_FRAMES  = None  # cap the darks used, None = every frame found
DARK_SIGMA_CLIP  = 4.0   # see analyze_gt_sequence.py's DARK_SIGMA_CLIP comment
HOT_PIXEL_SIGMA  = 5.0   # see analyze_gt_sequence.py's HOT_PIXEL_SIGMA comment
DEFECT_FRAC_WARN = 0.005 # warn once the defect map exceeds this fraction
LOAD_WORKERS     = 4     # threads used to decode dark frames ahead of the pass


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
    # gt_seq._dark_master reads LOAD_WORKERS from ITS OWN module globals, not
    # this script's -- module-level names are looked up dynamically at call
    # time, so overwriting it here on the imported module is what makes this
    # script's --load-workers flag actually take effect (same trick
    # analyze_gt_sequence.py itself uses for raw_utils.DEMOSAIC_METHOD).
    gt_seq.LOAD_WORKERS = LOAD_WORKERS

    t0 = time.monotonic()
    print(f"Dark-only check: {DARK_DIR}")
    fmt, paths, pattern, black, white, loader = _make_loader(DARK_DIR)
    n = len(paths)
    print(f"  {n} {fmt.upper()} frames  |  white={white}  black={black[0]}")

    out_dir = Path(OUTPUT_DIR) / Path(DARK_DIR).name
    out_dir.mkdir(parents=True, exist_ok=True)

    n_want = DARK_MAX_FRAMES or n
    print(f"\nSigma-clipped master dark from {min(n_want, n)}/{n} frames …")
    dark_adu, d_stack = gt_seq._dark_master(paths, n_want, loader, DARK_SIGMA_CLIP)
    print(f"  Master dark ADU: mean={dark_adu.mean():.2f}  "
          f"min={dark_adu.min():.2f}  max={dark_adu.max():.2f}  "
          f"(black level {black[0]:.1f})")
    np.save(out_dir / "dark_master.npy", dark_adu.astype(np.float32))
    print(f"Saved {out_dir / 'dark_master.npy'}")

    gt_seq._report_dark_uniformity(dark_adu, pattern, out_dir / "dark_uniformity.png")

    resid = gt_seq._dark_master_residual(paths, d_stack, loader, black, white)
    if resid is None:
        print("\nNeed at least 2 dark frames for the 'how many darks' estimate "
              "and the defect map -- skipping both.")
        print(f"\nTotal time: {format_duration(time.monotonic() - t0)}")
        print(f"Outputs in {out_dir.resolve()}")
        return

    master_hp = highpass_std(dark_adu / float(white - black[0]), pattern)
    print("\nHow many dark frames does this master want?")
    gt_seq._dark_advice(resid * np.sqrt(d_stack), master_hp, resid, d_stack)

    if HOT_PIXEL_SIGMA:
        resid_adu = resid * float(white - black[0])
        mask, n_hot, n_cold = gt_seq._defect_map(dark_adu, pattern, resid_adu,
                                                 HOT_PIXEL_SIGMA)
        frac = mask.mean() * 100
        print(f"\nDefect map (>{HOT_PIXEL_SIGMA}σ from same-colour "
              f"neighbours in the master dark): "
              f"{n_hot} hot, {n_cold} cold, {frac:.4f}% of pixels")
        if frac > DEFECT_FRAC_WARN * 100:
            print(f"  WARNING: that is a lot of pixels to interpolate. Each one "
                  f"is reconstructed from its neighbours,")
            print(f"           which costs real detail wherever the scene is "
                  f"fine (text, edges). Raise --hot-pixel-sigma or")
            print(f"           add dark frames until the map is nearer "
                  f"{DEFECT_FRAC_WARN * 100:.2f}%.")
        gt_seq._defect_sigma_scan(dark_adu, pattern,
                                  out_dir / "defect_sigma_scan.png",
                                  HOT_PIXEL_SIGMA)
        if mask.any():
            gt_seq._plot_defect_map(mask, n_hot, n_cold, out_dir / "defect_map.png")
            np.save(out_dir / "defect_map.npy", mask)

    print(f"\nTotal time: {format_duration(time.monotonic() - t0)}")
    print(f"Outputs in {out_dir.resolve()}")


if __name__ == "__main__":
    main()
