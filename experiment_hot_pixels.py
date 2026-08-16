"""
Experiment: can hot pixels be flagged from single frames by their brightness?

The idea under test is that in a dark lowlight capture a hot pixel is simply
one of the brightest pixels in the frame, so a plain "exceptionally high value"
threshold should find it -- no dark frames needed.

The catch is that a single high-gain frame also contains noise excursions, and
with megapixels per frame the largest of those reach 5 sigma or beyond. A
brightness threshold alone cannot tell those apart from a genuinely hot pixel.

What separates them is PERSISTENCE: a hot pixel is hot in every frame, a noise
excursion is high in one frame and ordinary in the next. So this script flags
each of several frames independently and then counts, per pixel, how many
frames flagged it. That histogram is the actual result -- if it splits into a
spike near "flagged once" (noise) and a spike near "flagged always" (real
defects), the idea works, and the persistent set is the map worth using.

Result on synthetic data with known ground truth (dark scene ~10 ADU above
black, per-frame sigma 22 ADU, hot pixels 40-160 ADU, 20 frames tested,
persistence cut at 80%):

    sigma  flags/frame  persistent   recall  precision
      3.0         1569         106    83.5%     100.0%
      4.0          339          88    69.3%     100.0%
      6.0           78          63    49.6%     100.0%
      8.0           46          28    22.0%     100.0%

The idea works, with one counter-intuitive consequence: precision is perfect
at every threshold because persistence -- not the threshold -- is what rejects
noise, so the threshold should be set LOOSE. At 3 sigma the per-frame test
flags 2.4% of the frame, almost all of it noise, and persistence still returns
a clean map with no false positives while recovering far more real defects.
Tightening the threshold only throws away genuine hot pixels.

What it does not reach are the mildly-hot pixels that never clear the threshold
in any single frame; those need a master dark, where the noise floor is
sigma/sqrt(N_dark) instead of sigma (see analyze_gt_sequence's defect map).

Edit the CONFIG block below, then run:
    python experiment_hot_pixels.py --sequence-dir /path/to/sequence

Outputs (saved to OUTPUT_DIR/<sequence_name>/):
  - hpx_marked.png       : one frame with flagged pixels circled, plus a zoom
  - hpx_persistence.png  : how often each flagged pixel is flagged, and the
                           frame-count map -- the experiment's real answer
  - hpx_gallery.png      : the most persistent candidates as zoomed patches,
                           before and after cubic interpolation
  - hpx_hot_mask.npy     : boolean mask of the persistent set, reusable
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from raw_utils import (
    detect_format, find_dngs, find_raws,
    get_raw_metadata, get_raw_metadata_gn3,
    load_raw, load_raw_gn3, calibrate_frame,
    bayer_plane_median3, cubic_fill_bayer,
    progress, format_duration,
)


# ============================================================
# CONFIG — edit paths and options here
# ============================================================
SEQUENCE_DIR    = "/path/to/static/sequence"
OUTPUT_DIR      = "output/hot_pixel_experiment"
GN3_BLACK_LEVEL = 256    # uniform black level for GN3 .raw files

N_FRAMES        = 20     # sample frames to test (evenly spaced through the run)
THRESH_SIGMA    = 3.0    # flag samples this many robust sigma above the plane
                         # median. Robust sigma comes from the MAD, so a few
                         # hot pixels cannot inflate the threshold that is
                         # supposed to catch them.
                         #
                         # Deliberately loose: the persistence cut below, not
                         # this threshold, is what provides precision. Measured
                         # against known ground truth (see module docstring), a
                         # tighter threshold only loses real defects while
                         # precision stays pinned at 100% either way.
PERSIST_FRAC    = 0.8    # a pixel counts as a real defect once flagged in this
                         # fraction of the tested frames
TOP_N           = 24     # candidates to show in the zoomed gallery
GALLERY_PATCH   = 15     # patch size (px, full-frame coords) per gallery tile
MARK_RADIUS     = 14     # circle radius (px) for the marked overview
MARK_MAX        = 400    # stop drawing circles past this many, to keep the
                         # overview legible
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


def flag_bright(cal, pattern, n_sigma):
    """
    Flag samples far above the median of their own Bayer sub-plane.

    Thresholding per sub-plane, never across the raw mosaic: the four channels
    sit at different levels on a coloured scene, so a single threshold over the
    mosaic would flag whole channels rather than defects.

    Spread comes from the MAD rather than the standard deviation -- hot pixels
    are exactly the outliers that would inflate an ordinary std and raise the
    threshold meant to catch them.
    """
    mask = np.zeros(cal.shape, dtype=bool)
    for r in range(2):
        for c in range(2):
            plane = cal[r::2, c::2]
            med   = np.median(plane)
            mad   = np.median(np.abs(plane - med))
            sigma = 1.4826 * mad                    # MAD -> Gaussian sigma
            if sigma <= 0:
                continue
            mask[r::2, c::2] = plane > med + n_sigma * sigma
    return mask


def flag_local(cal, pattern, n_sigma):
    """
    Same threshold, but relative to each sample's same-colour neighbours.

    Included only as a reference for the global test: it follows scene
    brightness, so it does not flag bright regions wholesale, but it also
    cannot distinguish a hot pixel from a real single-pixel highlight.
    """
    dev = cal - bayer_plane_median3(cal, pattern)
    mask = np.zeros(cal.shape, dtype=bool)
    for r in range(2):
        for c in range(2):
            d = dev[r::2, c::2]
            sigma = 1.4826 * np.median(np.abs(d - np.median(d)))
            if sigma <= 0:
                continue
            mask[r::2, c::2] = d > n_sigma * sigma
    return mask


def run(paths, pattern, black, white, loader, n_frames, n_sigma):
    """Flag each sampled frame independently; return per-pixel flag counts."""
    idx = np.linspace(0, len(paths) - 1, min(n_frames, len(paths)), dtype=int)
    counts = counts_local = None
    first_cal = None
    per_frame = []

    for i in progress(idx, desc="  frames", total=len(idx)):
        cal = calibrate_frame(loader(paths[i]), pattern, black, white)
        m   = flag_bright(cal, pattern, n_sigma)
        ml  = flag_local(cal, pattern, n_sigma)
        if counts is None:
            counts, counts_local = m.astype(np.int32), ml.astype(np.int32)
            first_cal = cal.copy()
        else:
            counts += m
            counts_local += ml
        per_frame.append((int(m.sum()), int(ml.sum())))
        del cal, m, ml
    return counts, counts_local, first_cal, len(idx), per_frame


# --------------------------------------------------------------------------- #
# Plots                                                                         #
# --------------------------------------------------------------------------- #

def plot_marked(cal, mask, out, radius, max_marks):
    """
    Overview with flagged pixels circled, next to a zoom on the densest region.

    Open circles rather than a tinted overlay: a single flagged pixel is
    invisible once a 12 MP frame is fitted to a figure, and a filled marker
    would hide the very pixel being judged. The circle sits around it instead.
    """
    ys, xs = np.nonzero(mask)
    shown = min(len(ys), max_marks)

    fig, (ax, axz) = plt.subplots(1, 2, figsize=(16, 7))
    lo, hi = np.percentile(cal, [1, 99.5])
    ax.imshow(cal, cmap="gray", vmin=lo, vmax=hi)
    ax.scatter(xs[:shown], ys[:shown], s=radius ** 2, facecolors="none",
               edgecolors="lime", linewidths=0.8)
    ax.set_title(f"Flagged in this frame: {len(ys)}"
                 + (f" (first {shown} circled)" if shown < len(ys) else ""),
                 fontsize=11)
    ax.axis("off")

    # Zoom where the flags are densest, so the marks are actually resolvable.
    if len(ys):
        cy, cx = int(np.median(ys)), int(np.median(xs))
    else:
        cy, cx = cal.shape[0] // 2, cal.shape[1] // 2
    half = 120
    y0, y1 = max(0, cy - half), min(cal.shape[0], cy + half)
    x0, x1 = max(0, cx - half), min(cal.shape[1], cx + half)
    axz.imshow(cal[y0:y1, x0:x1], cmap="gray", vmin=lo, vmax=hi,
               interpolation="nearest")
    sel = (ys >= y0) & (ys < y1) & (xs >= x0) & (xs < x1)
    axz.scatter(xs[sel] - x0, ys[sel] - y0, s=260, facecolors="none",
                edgecolors="lime", linewidths=1.2)
    axz.set_title(f"Zoom {x1-x0}×{y1-y0} px at 100% — {int(sel.sum())} flagged",
                  fontsize=11)
    axz.axis("off")

    fig.suptitle("Brightness-flagged pixels in a single frame", fontsize=13, y=1.0)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def plot_persistence(counts, counts_local, n_used, persist_frac, out):
    """
    The experiment's actual result: how often each flagged pixel gets flagged.

    A real defect is hot in every frame and lands at the far right. A noise
    excursion is high once by chance and lands at 1. A clean split between the
    two means brightness thresholding works -- provided it is read across
    frames rather than within one.
    """
    fig, (ax, axm) = plt.subplots(1, 2, figsize=(15, 5.6))

    bins = np.arange(0.5, n_used + 1.5)
    for c, lab, col in [(counts, "global threshold", "steelblue"),
                        (counts_local, "local (neighbour) threshold", "darkorange")]:
        vals = c[c > 0]
        ax.hist(vals, bins=bins, alpha=0.6, label=lab, color=col)
    ax.axvline(persist_frac * n_used, color="crimson", linestyle="--",
               linewidth=1.4,
               label=f"persistence cut ({persist_frac:.0%} of {n_used})")
    ax.set_yscale("log")
    ax.set_xlabel(f"Frames in which the pixel was flagged (out of {n_used})",
                  fontsize=11)
    ax.set_ylabel("Number of pixels", fontsize=11)
    ax.set_title("Persistence — the separator between defects and noise",
                 fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, linestyle="--")

    im = axm.imshow(counts, cmap="inferno", vmin=0, vmax=n_used,
                    interpolation="nearest")
    axm.set_title("Flag count per pixel (global threshold)", fontsize=12)
    axm.axis("off")
    fig.colorbar(im, ax=axm, fraction=0.035, pad=0.02, label="frames flagged")

    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def plot_gallery(cal, fixed, counts, n_used, out, top_n, patch):
    """Most persistent candidates as zoomed patches, before and after repair."""
    ys, xs = np.nonzero(counts >= 1)
    if not len(ys):
        return
    order = np.argsort(counts[ys, xs])[::-1][:top_n]
    ys, xs = ys[order], xs[order]

    n = len(ys)
    cols = min(8, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows * 2, cols, figsize=(2.0 * cols, 4.2 * rows))
    axes = np.atleast_2d(axes)
    h = patch // 2

    for k in range(rows * cols):
        r_before = (k // cols) * 2
        r_after  = r_before + 1
        ax_b = axes[r_before, k % cols]
        ax_a = axes[r_after,  k % cols]
        if k >= n:
            ax_b.axis("off"); ax_a.axis("off")
            continue
        y, x = int(ys[k]), int(xs[k])
        y0, y1 = max(0, y - h), min(cal.shape[0], y + h + 1)
        x0, x1 = max(0, x - h), min(cal.shape[1], x + h + 1)
        sub = cal[y0:y1, x0:x1]
        lo, hi = float(sub.min()), float(sub.max())
        ax_b.imshow(sub, cmap="gray", vmin=lo, vmax=hi, interpolation="nearest")
        ax_b.plot(x - x0, y - y0, "o", markerfacecolor="none",
                  markeredgecolor="lime", markersize=13, markeredgewidth=1.2)
        ax_b.set_title(f"{counts[y, x]}/{n_used}", fontsize=8)
        ax_b.axis("off")
        ax_a.imshow(fixed[y0:y1, x0:x1], cmap="gray", vmin=lo, vmax=hi,
                    interpolation="nearest")
        ax_a.axis("off")

    fig.suptitle(f"Top {n} candidates by persistence — "
                 f"top row raw (circled), bottom row after cubic interpolation, "
                 f"shared scale per pair", fontsize=12, y=1.0)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


# --------------------------------------------------------------------------- #
# Main                                                                          #
# --------------------------------------------------------------------------- #

def analyze(directory: str, out_dir: Path) -> None:
    seq_out = out_dir / Path(directory).name
    seq_out.mkdir(parents=True, exist_ok=True)

    fmt, paths, pattern, black, white, loader = _make_loaders(directory)
    print(f"  {len(paths)} {fmt.upper()} frames  |  testing {N_FRAMES} of them "
          f"at {THRESH_SIGMA}σ")

    counts, counts_local, cal, n_used, per_frame = run(
        paths, pattern, black, white, loader, N_FRAMES, THRESH_SIGMA)

    cut = max(1, int(np.ceil(PERSIST_FRAC * n_used)))
    persistent = counts >= cut
    transient  = (counts > 0) & (counts < cut)
    npx = counts.size

    print(f"\n  Per-frame flag counts (global): "
          f"min={min(f[0] for f in per_frame)}  "
          f"max={max(f[0] for f in per_frame)}  "
          f"mean={np.mean([f[0] for f in per_frame]):.0f}")
    print(f"  Pixels flagged at least once : {int((counts > 0).sum())} "
          f"({(counts > 0).mean() * 100:.4f}%)")
    print(f"  Persistent (>={cut}/{n_used})      : {int(persistent.sum())} "
          f"({persistent.mean() * 100:.4f}%)   <- candidate defects")
    print(f"  Transient (1..{cut - 1} frames)      : {int(transient.sum())} "
          f"   <- noise excursions")
    if (counts > 0).sum():
        print(f"  Persistent share of all flags: "
              f"{persistent.sum() / (counts > 0).sum() * 100:.1f}%")
    print()

    fixed = cubic_fill_bayer(cal, pattern, persistent)
    plot_marked(cal, flag_bright(cal, pattern, THRESH_SIGMA),
                seq_out / "hpx_marked.png", MARK_RADIUS, MARK_MAX)
    plot_persistence(counts, counts_local, n_used, PERSIST_FRAC,
                     seq_out / "hpx_persistence.png")
    plot_gallery(cal, fixed, counts, n_used, seq_out / "hpx_gallery.png",
                 TOP_N, GALLERY_PATCH)
    np.save(seq_out / "hpx_hot_mask.npy", persistent)
    print(f"Saved {seq_out / 'hpx_hot_mask.npy'}")
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
    print(f"Hot-pixel flagging experiment: {SEQUENCE_DIR}")
    t0 = time.monotonic()
    analyze(SEQUENCE_DIR, Path(OUTPUT_DIR))
    print(f"Total time: {format_duration(time.monotonic() - t0)}")


if __name__ == "__main__":
    main()
