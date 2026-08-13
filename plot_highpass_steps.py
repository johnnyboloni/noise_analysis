"""
Visualise every step of the high-pass noise metric used by analyze_gt_sequence.

Shows what highpass_std() actually does to a frame, stage by stage, so the
number it returns can be sanity-checked against the picture it came from --
in particular whether the plateau in the high-pass curve is genuine
fixed-pattern noise or just fine scene detail leaking through the filter.

Edit the CONFIG block below, then run:
    python plot_highpass_steps.py

Outputs (saved to OUTPUT_DIR/<sequence_name>/):
  - hp_pipeline_k{K}.png  : one row per N -- Bayer sub-plane, its box mean,
                            the residual, and the residual histogram
  - hp_ksize.png          : residual map + histogram at each kernel size, for
                            the largest N. Scene detail shrinks as k shrinks
                            (a tighter local mean subtracts more of it);
                            pixel-level FPN is spatially white and barely
                            moves. A floor that collapses between k=9 and k=3
                            was mostly scene detail, not noise.
  - hp_summary.png        : residual std vs N for every k, on one axis
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

from raw_utils import (
    detect_format, find_dngs, find_raws,
    get_raw_metadata, get_raw_metadata_gn3,
    load_raw, load_raw_gn3, calibrate_frame, demosaic_to_rgb,
    progress, format_duration, _HAS_CV2,
)


# ============================================================
# CONFIG — edit paths and options here
# ============================================================
SEQUENCE_DIR    = "/path/to/static/sequence"
OUTPUT_DIR      = "output/highpass_steps"
GN3_BLACK_LEVEL = 256    # uniform black level for GN3 .raw files

MAX_FRAMES      = None   # int to cap frames streamed, None = all
N_CHECKPOINTS   = 4      # running-mean snapshots to visualise (log-spaced in N)
KSIZES          = "3,5,9"  # box-filter sizes to compare (comma-separated)
MAIN_KSIZE      = 5      # kernel used for the per-N pipeline figure
CROP            = 256    # crop size (px, in sub-plane coords) for the maps
BAYER_PLANE     = 1      # which Bayer position to display: 0=(0,0) 1=(0,1)
                         # 2=(1,0) 3=(1,1). Usually a green site.
# ============================================================


# --------------------------------------------------------------------------- #
# Loading                                                                       #
# --------------------------------------------------------------------------- #

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


def _box_mean(plane: np.ndarray, k: int) -> np.ndarray:
    """Normalized k×k box mean -- the exact smoothing highpass_std() uses."""
    if _HAS_CV2:
        import cv2
        return cv2.blur(plane, (k, k))
    from numpy.lib.stride_tricks import sliding_window_view
    pad = k // 2
    padded = np.pad(plane, pad, mode='reflect')
    return sliding_window_view(padded, (k, k)).mean(axis=(-1, -2)).astype(np.float32)


def _centre_crop(a: np.ndarray, size: int) -> np.ndarray:
    cy, cx = a.shape[0] // 2, a.shape[1] // 2
    h = min(size, a.shape[0]) // 2
    w = min(size, a.shape[1]) // 2
    return a[cy - h: cy + h, cx - w: cx + w]


def collect(paths, pattern, black, white, loader, ns, ksizes, plane_idx, crop):
    """
    Stream frames once, and at each N in `ns` capture what the high-pass does
    to the running mean: full-frame residual std per kernel size (the actual
    metric) plus centre crops of the intermediate arrays (for display).

    Only crops are retained, so memory stays flat regardless of len(ns).
    """
    r, c = plane_idx // 2, plane_idx % 2
    accum, out = None, []
    n_total = len(paths)
    want = set(ns)

    for i, p in enumerate(progress(paths, desc="  streaming")):
        idx = i + 1
        raw   = loader(p).astype(np.float64)
        accum = raw if accum is None else accum + raw
        if idx not in want:
            continue

        running = calibrate_frame((accum / idx).astype(np.float32),
                                  pattern, black, white)
        plane   = np.ascontiguousarray(running[r::2, c::2], dtype=np.float32)
        rec     = {'n': idx,
                   'plane': _centre_crop(plane, crop),
                   'smooth': {}, 'resid': {}, 'std': {}}
        for k in ksizes:
            smooth = _box_mean(plane, k)
            resid  = plane - smooth
            rec['std'][k]    = float(resid.std())      # full frame, the real metric
            rec['smooth'][k] = _centre_crop(smooth, crop)
            rec['resid'][k]  = _centre_crop(resid, crop)
            del smooth, resid
        out.append(rec)
        del running, plane
    print()
    return out


# --------------------------------------------------------------------------- #
# Plots                                                                         #
# --------------------------------------------------------------------------- #

def plot_pipeline(recs, k, out):
    """One row per N: sub-plane -> box mean -> residual -> residual histogram."""
    rows = len(recs)
    fig, axes = plt.subplots(rows, 4, figsize=(17, 3.9 * rows))
    if rows == 1:
        axes = axes[None, :]

    for ax_row, rec in zip(axes, recs):
        plane, smooth = rec['plane'], rec['smooth'][k]
        resid, sd     = rec['resid'][k], rec['std'][k]

        # Shared scale for plane and its box mean so the smoothing is visible
        # as an actual change rather than a re-normalisation.
        lo, hi = np.percentile(plane, [0.5, 99.5])
        for ax, img, title in [
            (ax_row[0], plane,  f"1. Bayer sub-plane   (N={rec['n']})"),
            (ax_row[1], smooth, f"2. box mean, k={k}"),
        ]:
            ax.imshow(img, cmap="gray", vmin=lo, vmax=hi, interpolation="nearest")
            ax.set_title(title, fontsize=10)
            ax.axis("off")

        # Residual: symmetric diverging scale centred on zero.
        v = max(float(np.percentile(np.abs(resid), 99.5)), 1e-9)
        im = ax_row[2].imshow(resid, cmap="RdBu_r", vmin=-v, vmax=v,
                              interpolation="nearest")
        ax_row[2].set_title(f"3. residual = (1) − (2)\nstd = {sd:.6f}", fontsize=10)
        ax_row[2].axis("off")
        fig.colorbar(im, ax=ax_row[2], fraction=0.046, pad=0.02)

        ax_row[3].hist(resid.ravel(), bins=120, color="steelblue", alpha=0.85)
        ax_row[3].axvline(0, color="k", linewidth=0.7, alpha=0.5)
        ax_row[3].set_title(f"4. residual histogram\nfull-frame std = {sd:.6f}",
                            fontsize=10)
        ax_row[3].set_yscale("log")
        ax_row[3].grid(True, alpha=0.25, linestyle="--")

    fig.suptitle(
        f"High-pass metric, step by step (k={k}) — "
        f"residual shrinks with N only while temporal noise dominates",
        fontsize=13, y=1.005)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def plot_ksize(rec, ksizes, out):
    """Residual map + histogram at each kernel size, for a single N."""
    fig, axes = plt.subplots(2, len(ksizes), figsize=(5.2 * len(ksizes), 8.4))
    if len(ksizes) == 1:
        axes = axes[:, None]

    v = max(float(np.percentile(np.abs(rec['resid'][max(ksizes)]), 99.5)), 1e-9)
    for j, k in enumerate(ksizes):
        resid, sd = rec['resid'][k], rec['std'][k]
        im = axes[0, j].imshow(resid, cmap="RdBu_r", vmin=-v, vmax=v,
                               interpolation="nearest")
        axes[0, j].set_title(f"k = {k}\nstd = {sd:.6f}", fontsize=11)
        axes[0, j].axis("off")
        fig.colorbar(im, ax=axes[0, j], fraction=0.046, pad=0.02)

        axes[1, j].hist(resid.ravel(), bins=120, color="darkorange", alpha=0.85)
        axes[1, j].axvline(0, color="k", linewidth=0.7, alpha=0.5)
        axes[1, j].set_yscale("log")
        axes[1, j].set_title(f"residual histogram, k={k}", fontsize=10)
        axes[1, j].grid(True, alpha=0.25, linestyle="--")

    fig.suptitle(
        f"Kernel size sweep at N={rec['n']} (shared colour scale) — "
        f"structure that fades as k shrinks is scene detail, not noise",
        fontsize=13, y=1.005)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def plot_summary(recs, ksizes, out):
    """Residual std vs N, one curve per kernel size, with a 1/sqrt(N) guide."""
    ns = np.array([r['n'] for r in recs], dtype=float)
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    for k in ksizes:
        sd = np.array([r['std'][k] for r in recs], dtype=float)
        ax.loglog(ns, sd, "o-", linewidth=1.8, markersize=5, label=f"k = {k}")
    sd0 = recs[0]['std'][ksizes[0]]
    ax.loglog(ns, sd0 * np.sqrt(ns[0]) / np.sqrt(ns), "--", color="gray",
              linewidth=1.3, label=r"ideal $\propto 1/\sqrt{N}$")
    ax.set_xlabel("Frames averaged  (N)", fontsize=11)
    ax.set_ylabel("High-pass residual std  [calibrated units]", fontsize=11)
    ax.set_title("High-pass residual vs N, by kernel size", fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, which="both", alpha=0.3, linestyle="--")
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

    ksizes = [int(s) for s in str(KSIZES).split(',') if s.strip()]
    fmt, paths, pattern, black, white, loader = _make_loaders(directory)
    if MAX_FRAMES is not None:
        paths = paths[:MAX_FRAMES]
    n = len(paths)

    ns = sorted(set(np.geomspace(1, n, min(N_CHECKPOINTS, n))
                    .astype(int).tolist()) | {n})
    print(f"  {n} {fmt.upper()} frames  |  checkpoints N={ns}  |  k={ksizes}")

    recs = collect(paths, pattern, black, white, loader,
                   ns, ksizes, BAYER_PLANE, CROP)

    print("\n  High-pass residual std (full frame, calibrated units)")
    print("  " + "N".rjust(6) + "".join(f"{'k='+str(k):>14}" for k in ksizes))
    print("  " + "-" * (6 + 14 * len(ksizes)))
    for r in recs:
        print("  " + f"{r['n']:6d}" + "".join(f"{r['std'][k]:14.6f}" for k in ksizes))
    print()

    plot_pipeline(recs, MAIN_KSIZE if MAIN_KSIZE in ksizes else ksizes[0],
                  seq_out / f"hp_pipeline_k{MAIN_KSIZE}.png")
    plot_ksize(recs[-1], ksizes, seq_out / "hp_ksize.png")
    plot_summary(recs, ksizes, seq_out / "hp_summary.png")
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
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
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
    print(f"High-pass step visualisation: {SEQUENCE_DIR}")
    t0 = time.monotonic()
    analyze(SEQUENCE_DIR, Path(OUTPUT_DIR))
    print(f"Total time: {format_duration(time.monotonic() - t0)}")


if __name__ == "__main__":
    main()
