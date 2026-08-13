"""
Std of a fixed image crop vs number of frames aggregated, measured both in the
Bayer domain and after demosaicing, on one plot.

Edit the CONFIG block below, then run:
    python plot_crop_std.py --sequence-dir /path/to/sequence

Outputs (saved to OUTPUT_DIR/<sequence_name>/):
  - crop_std_N{n}.png      : std vs N -- absolute (left) and normalised to the
                             first checkpoint (right, with a 1/sqrt(N) guide)
  - crop_location_N{n}.png : where the crop sits in the frame, plus a zoom

Notes on what is measured
-------------------------
The crop is a plain std, so it measures scene structure as well as noise. Pick
a visually uniform patch or the curve will flatten early on the patch's own
contrast rather than on a noise floor.

Bayer std is computed per sub-plane and averaged, never across the raw mosaic:
neighbouring mosaic pixels are different colour channels, so a std taken
straight across them is dominated by the R/G/B level differences (a large,
N-independent constant) and would look flat no matter how many frames were
averaged.

Demosaiced std is computed per RGB channel and averaged, on a crop taken from
the FULL demosaiced frame -- not by demosaicing the crop alone. demosaic_to_rgb
auto-brightens by the frame's own 99th percentile, so demosaicing each crop
separately would renormalise every checkpoint to its own range and mask the
very improvement this plot is measuring.

The two curves are in different units: Bayer is calibrated [0, 1] sensor units,
demosaiced is display units after white balance, CCM, auto-brighten and gamma
(a roughly hundredfold gain in lowlight). They are plotted together on a log
axis as requested; the right-hand panel normalises both so their shapes can be
compared directly.
"""

import argparse
import json
import time
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from raw_utils import (
    detect_format, find_dngs, find_raws,
    get_raw_metadata, get_raw_metadata_gn3, get_color_metadata,
    load_raw, load_raw_gn3, calibrate_frame, demosaic_to_rgb,
    progress, format_duration,
)


# ============================================================
# CONFIG — edit paths and options here
# ============================================================
SEQUENCE_DIR    = "/path/to/static/sequence"
OUTPUT_DIR      = "output/crop_std"
GN3_BLACK_LEVEL = 256    # uniform black level for GN3 .raw files

# Crop in (x, y) pixel coordinates: top-left and bottom-right, x = column.
CROP_X0, CROP_Y0 = 2127, 989
CROP_X1, CROP_Y1 = 2230, 1096

MAX_FRAMES      = None   # int to cap frames streamed, None = all
N_CHECKPOINTS   = 12     # log-spaced values of N at which to measure
# ============================================================


def _make_loaders(directory: str):
    """Detect format and return (fmt, paths, pattern, black, white, loader)."""
    fmt = detect_format(directory)
    if fmt == 'dng':
        paths = find_dngs(directory)
        pattern, black, white = get_raw_metadata(paths[0])
        loader = load_raw
    else:
        paths = find_raws(directory)
        pattern, black, white = get_raw_metadata_gn3(paths[0], GN3_BLACK_LEVEL)
        meta   = json.loads(paths[0].with_suffix('.imgprops').read_text())
        shape  = (meta['height'], meta['width'])
        loader = lambda p, _s=shape: load_raw_gn3(p, _s)
    return fmt, paths, pattern, black, white, loader


def _resolve_crop(shape):
    """
    Clamp the configured crop to the frame and snap its origin to even pixels.

    Even alignment keeps Bayer sub-plane (r, c) corresponding to pattern[r, c];
    an odd origin would silently relabel the channels. The shift is at most one
    pixel in each axis.
    """
    H, W = shape
    x0, y0 = max(0, int(CROP_X0)), max(0, int(CROP_Y0))
    x1, y1 = min(W, int(CROP_X1)), min(H, int(CROP_Y1))
    x0 -= x0 % 2
    y0 -= y0 % 2
    if x1 <= x0 or y1 <= y0:
        sys.exit(f"Empty crop after clamping to frame {W}x{H}: "
                 f"x[{x0}:{x1}] y[{y0}:{y1}]")
    return x0, y0, x1, y1


def _bayer_crop_std(cal, pattern, box):
    """Mean of the per-Bayer-sub-plane stds over the crop (calibrated units)."""
    x0, y0, x1, y1 = box
    sub = cal[y0:y1, x0:x1]
    return float(np.mean([sub[r::2, c::2].std()
                          for r in range(2) for c in range(2)]))


def _rgb_crop_std(rgb, box):
    """Mean of the per-channel stds over the crop (display units)."""
    x0, y0, x1, y1 = box
    sub = rgb[y0:y1, x0:x1]
    return float(np.mean([sub[..., i].std() for i in range(3)]))


def collect(paths, pattern, black, white, loader, ns, wb, ccm):
    """Stream once; at each N measure the crop std in both domains."""
    accum, rows, box = None, [], None
    n_total, want = len(paths), set(ns)
    last_rgb = None

    for i, p in enumerate(progress(paths, desc="  streaming")):
        idx = i + 1
        raw   = loader(p).astype(np.float64)
        accum = raw if accum is None else accum + raw
        if idx not in want:
            continue

        cal = calibrate_frame((accum / idx).astype(np.float32),
                              pattern, black, white)
        if box is None:
            box = _resolve_crop(cal.shape)
        # Demosaic the whole frame, then crop: the auto-brighten inside
        # demosaic_to_rgb is driven by the full frame's 99th percentile.
        rgb = demosaic_to_rgb(cal, pattern, wb, ccm)
        rows.append({'n': idx,
                     'bayer': _bayer_crop_std(cal, pattern, box),
                     'rgb':   _rgb_crop_std(rgb, box)})
        last_rgb = rgb if idx == max(want) else last_rgb
        del cal
        if last_rgb is not rgb:
            del rgb
    print()
    return rows, box, last_rgb


def plot_std(rows, out, box):
    ns = np.array([r['n'] for r in rows], dtype=float)
    b  = np.array([r['bayer'] for r in rows], dtype=float)
    g  = np.array([r['rgb'] for r in rows], dtype=float)
    x0, y0, x1, y1 = box

    fig, (ax, axn) = plt.subplots(1, 2, figsize=(15, 5.6))

    ax.loglog(ns, b, "o-", color="steelblue", linewidth=1.8, markersize=5,
              label="Bayer (per sub-plane, calibrated units)")
    ax.loglog(ns, g, "s-", color="darkorange", linewidth=1.8, markersize=5,
              label="Demosaiced (per RGB channel, display units)")
    ax.set_xlabel("Frames aggregated  (N)", fontsize=11)
    ax.set_ylabel("Crop std", fontsize=11)
    ax.set_title("Absolute std  (note: different units, see docstring)", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, which="both", alpha=0.3, linestyle="--")

    axn.loglog(ns, b / b[0], "o-", color="steelblue", linewidth=1.8,
               markersize=5, label="Bayer")
    axn.loglog(ns, g / g[0], "s-", color="darkorange", linewidth=1.8,
               markersize=5, label="Demosaiced")
    axn.loglog(ns, np.sqrt(ns[0]) / np.sqrt(ns), "--", color="gray",
               linewidth=1.3, label=r"ideal $\propto 1/\sqrt{N}$")
    axn.set_xlabel("Frames aggregated  (N)", fontsize=11)
    axn.set_ylabel(f"std relative to N={int(ns[0])}", fontsize=11)
    axn.set_title("Normalised — shapes directly comparable", fontsize=11)
    axn.legend(fontsize=9)
    axn.grid(True, which="both", alpha=0.3, linestyle="--")

    fig.suptitle(f"Crop std vs frames aggregated — "
                 f"x[{x0}:{x1}] y[{y0}:{y1}]  ({x1-x0}×{y1-y0} px)",
                 fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def plot_location(rgb, box, out):
    """Full frame with the crop outlined, next to a 100% zoom of the crop."""
    x0, y0, x1, y1 = box
    fig, (ax, axz) = plt.subplots(1, 2, figsize=(15, 6))

    ax.imshow(rgb)
    ax.add_patch(mpatches.Rectangle((x0, y0), x1 - x0, y1 - y0,
                                    fill=False, edgecolor="lime", linewidth=1.6))
    ax.set_title(f"Crop location  x[{x0}:{x1}] y[{y0}:{y1}]", fontsize=11)
    ax.axis("off")

    axz.imshow(rgb[y0:y1, x0:x1], interpolation="nearest")
    axz.set_title(f"Crop at 100% zoom  ({x1-x0}×{y1-y0} px)", fontsize=11)
    axz.axis("off")

    fig.suptitle("A plain std also measures scene structure — "
                 "this patch should look uniform", fontsize=12, y=1.01)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def analyze(directory: str, out_dir: Path) -> None:
    seq_out = out_dir / Path(directory).name
    seq_out.mkdir(parents=True, exist_ok=True)

    fmt, paths, pattern, black, white, loader = _make_loaders(directory)
    if MAX_FRAMES is not None:
        paths = paths[:MAX_FRAMES]
    n = len(paths)
    wb, ccm = (get_color_metadata(paths[0]) if fmt == 'dng' else (None, None))

    ns = sorted(set(np.geomspace(1, n, min(N_CHECKPOINTS, n))
                    .astype(int).tolist()) | {n})
    print(f"  {n} {fmt.upper()} frames  |  crop x[{CROP_X0}:{CROP_X1}] "
          f"y[{CROP_Y0}:{CROP_Y1}]  |  {len(ns)} checkpoints")

    rows, box, last_rgb = collect(paths, pattern, black, white, loader,
                                  ns, wb, ccm)

    print(f"\n  Crop std vs N   (crop x[{box[0]}:{box[2]}] y[{box[1]}:{box[3]}])")
    print(f"  {'N':>6}  {'bayer':>12} {'rel':>7}  {'demosaic':>12} {'rel':>7}"
          f"  {'ideal':>7}")
    print("  " + "-" * 60)
    b0, g0, n0 = rows[0]['bayer'], rows[0]['rgb'], rows[0]['n']
    for r in rows:
        print(f"  {r['n']:6d}  {r['bayer']:12.6f} {r['bayer']/b0:7.3f}  "
              f"{r['rgb']:12.6f} {r['rgb']/g0:7.3f}  "
              f"{np.sqrt(n0/r['n']):7.3f}")
    print()

    plot_std(rows, seq_out / f"crop_std_N{n}.png", box)
    if last_rgb is not None:
        plot_location(last_rgb, box, seq_out / f"crop_location_N{n}.png")
    print(f"\nDone. Outputs in {seq_out.resolve()}")


def _none_or_auto(s: str):
    if s.lower() == 'none':
        return None
    for cast in (int, float):
        try:
            return cast(s)
        except ValueError:
            pass
    return s


_UNSET = object()


def _apply_cli_overrides() -> None:
    g = globals()
    scalar = (bool, int, float, str, type(None))
    keys = sorted(k for k in g
                  if k.isupper() and not k.startswith('_') and isinstance(g[k], scalar))
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for key in keys:
        val, flag = g[key], '--' + key.lower().replace('_', '-')
        if isinstance(val, bool):
            parser.add_argument(flag, dest=key, default=_UNSET,
                                action=argparse.BooleanOptionalAction,
                                help=f"(default: {val})")
        elif isinstance(val, int):
            parser.add_argument(flag, dest=key, type=int, default=_UNSET,
                                metavar='N', help=f"(default: {val})")
        elif isinstance(val, float):
            parser.add_argument(flag, dest=key, type=float, default=_UNSET,
                                metavar='F', help=f"(default: {val})")
        else:
            parser.add_argument(flag, dest=key, type=_none_or_auto, default=_UNSET,
                                metavar='S', help=f"(default: {val!r}; 'none' to clear)")
    args = parser.parse_args()
    for key, new_val in vars(args).items():
        if new_val is not _UNSET:
            g[key] = new_val


def main():
    _apply_cli_overrides()
    print(f"Crop std analysis: {SEQUENCE_DIR}")
    t0 = time.monotonic()
    analyze(SEQUENCE_DIR, Path(OUTPUT_DIR))
    print(f"Total time: {format_duration(time.monotonic() - t0)}")


if __name__ == "__main__":
    main()
