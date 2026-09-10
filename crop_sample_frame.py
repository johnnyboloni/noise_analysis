"""
Render one raw frame from a sequence -- demosaiced, gamma-encoded, cropped --
so it can stand in as the "single input frame" example next to the mean and
dark-subtracted mean in a resulting-frame report figure.

Reuses the exact same rendering path analyze_gt_sequence.py's comparison/
candidates go through (calibrate_frame -> demosaic_linear -> encode_rgb), and
the same CROP_XY/CROP_SIZE region by default, so this crop drops in next to
comparison/crops/ at the same scale and the same part of the frame.

GAIN must be the "Uniform display gain" printed by the real comparison run
this crop needs to sit alongside -- it is not recomputed here from this one
frame alone. A single frame's own 99.99th percentile is a different number
from the shared gain computed across the whole candidate set (mean,
dark-subtracted, ...), so using a value computed from just this frame would
render it brighter or dimmer than its neighbours for no real reason,
defeating the entire point of a uniform-gain comparison. Read the value from
that run's console output ("Uniform display gain: x2.345 (...)") or its
saved log, and pass it here.

Edit the CONFIG block below, then run:
    python crop_sample_frame.py

Output (saved to OUTPUT_DIR/):
  - sample_frame_<index>_nogain.png  : gain 1.0, the frame as calibrated.
  - sample_frame_<index>_uniform.png : at GAIN -- the SAME shared gain used
                                        for the real comparison run's crops,
                                        so this is directly comparable to
                                        comparison/crops/*_uniform.png.
  - sample_frame_<index>_max.png     : this single frame's own gain, ignoring
                                        the comparison set (not comparable to
                                        the other crops' brightness -- see
                                        analyze_gt_sequence.py's save_candidate
                                        for why).
"""

import argparse
import json
from pathlib import Path

import numpy as np

import raw_utils
from raw_utils import (
    detect_format, find_dngs, find_raws,
    get_raw_metadata, get_raw_metadata_gn3, get_color_metadata,
    load_raw, load_raw_gn3,
    calibrate_frame, demosaic_linear, encode_rgb, uniform_gain, save_rgb_png,
)
import analyze_gt_sequence as gt_seq

# --------------------------------------------------------------------------- #
# CONFIG -- edit paths and options here
# --------------------------------------------------------------------------- #

SEQUENCE_DIR = "/path/to/light/sequence"
OUTPUT_DIR   = "./sample_frame_crop"
FRAME_INDEX  = 0        # which frame in the sequence to render (0 = first)

GAIN = None              # REQUIRED -- the "Uniform display gain" printed by
                         # the real analyze_gt_sequence.py comparison run
                         # this crop needs to match. Cannot be safely
                         # defaulted or recomputed here; see module docstring.

CROP_XY   = gt_seq.CROP_XY    # match the main pipeline's crop region by
CROP_SIZE = gt_seq.CROP_SIZE  # default -- override if that run used
                              # different values (check its CONFIG/run_info).

GN3_BLACK_LEVEL = 256    # uniform black level for GN3 .raw files
DEMOSAIC        = gt_seq.DEMOSAIC
GAIN_PERCENTILE = gt_seq.GAIN_PERCENTILE   # for this frame's own _max variant


def _make_loader(directory: str):
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


def main():
    _apply_cli_overrides()
    if not GAIN:
        raise SystemExit(
            'GAIN is not set. Read "Uniform display gain: x..." from the '
            "console output of the real analyze_gt_sequence.py comparison "
            "run and pass it here (--gain X.XXX), so this crop is on the "
            "same brightness scale as comparison/crops/.")
    if not CROP_XY or not CROP_SIZE:
        raise SystemExit("CROP_XY and CROP_SIZE must both be set.")

    raw_utils.DEMOSAIC_METHOD = DEMOSAIC
    raw_utils.check_demosaic_method()

    print(f"Sequence: {SEQUENCE_DIR}")
    fmt, paths, pattern, black, white, loader = _make_loader(SEQUENCE_DIR)
    print(f"  {len(paths)} {fmt.upper()} frames  |  white={white}  black={black[0]}")
    if FRAME_INDEX >= len(paths):
        raise SystemExit(f"FRAME_INDEX={FRAME_INDEX} but only {len(paths)} frames found.")

    wb, ccm = get_color_metadata(paths[0]) if fmt == 'dng' else (None, None)

    frame      = loader(paths[FRAME_INDEX])
    calibrated = calibrate_frame(frame, pattern, black, white)
    lin        = demosaic_linear(calibrated, pattern, wb, ccm)

    cx, cy = CROP_XY
    crop = lin[cy:cy + CROP_SIZE, cx:cx + CROP_SIZE]
    if crop.shape[0] < CROP_SIZE or crop.shape[1] < CROP_SIZE:
        print(f"  WARNING: crop region extends past the frame ({lin.shape[1]}x{lin.shape[0]}); "
              f"got a {crop.shape[1]}x{crop.shape[0]} crop instead of "
              f"{CROP_SIZE}x{CROP_SIZE}.")
    own_gain = uniform_gain([lin], percentile=GAIN_PERCENTILE)

    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    base = out_dir / f"sample_frame_{FRAME_INDEX}"
    save_rgb_png(encode_rgb(crop), base.with_name(base.name + "_nogain.png"))
    save_rgb_png(encode_rgb(crop, gain=GAIN), base.with_name(base.name + "_uniform.png"))
    save_rgb_png(encode_rgb(crop, gain=own_gain), base.with_name(base.name + "_max.png"))
    print(f"  This frame's own max gain: x{own_gain:.3f}  "
          f"(shared/uniform gain used: x{GAIN:.3f})")
    print(f"Saved crops to {out_dir.resolve()}")


if __name__ == "__main__":
    main()
