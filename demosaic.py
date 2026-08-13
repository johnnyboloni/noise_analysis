#!/usr/bin/env python3
"""
demosaic.py — demosaic frames and compare them against the mean/median frame.

Uses the shared pipeline in raw_utils (white balance → cv2 VNG demosaic →
camera CCM → auto-brighten → gamma), the same one analyze_gt_sequence.py uses,
so single frames are directly comparable to mean/median frames.

Subcommands
-----------
demosaic        Demosaic all frames in a directory and save as PNG or .npy.
compare         Compare a single demosaiced frame to a saved mean/median frame.
detect_shifts   Report per-frame translation shifts (no alignment applied).

Example usage
-------------
python demosaic.py demosaic /data/dngs --out output/
python demosaic.py compare  /data/dngs/frame_001.dng output_dark/seq/mean_frame.npy
python demosaic.py detect_shifts /data/dngs --ref 0 --crop 512 --plot shifts.png
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

from raw_utils import (
    find_dngs, get_raw_metadata, get_color_metadata,
    load_raw, calibrate_frame, demosaic_to_rgb, save_rgb_png,
)


# --------------------------------------------------------------------------- #
# Core helpers (importable)                                                     #
# --------------------------------------------------------------------------- #

def demosaic_path(path: Path, pattern: np.ndarray, black: np.ndarray,
                  white: float, wb: np.ndarray | None = None,
                  ccm: np.ndarray | None = None) -> np.ndarray:
    """
    Load a DNG, calibrate, and demosaic it with the shared raw_utils pipeline.
    Returns float32 (H, W, 3) in [0, 1], gamma-encoded.
    """
    bayer = load_raw(path)
    cal   = calibrate_frame(bayer, pattern, black, white)
    return demosaic_to_rgb(cal, pattern, wb=wb, ccm=ccm)


def demosaic_adu_frame(frame_adu: np.ndarray, pattern: np.ndarray,
                       black: np.ndarray, white: float,
                       wb: np.ndarray | None = None,
                       ccm: np.ndarray | None = None) -> np.ndarray:
    """
    Demosaic an in-memory raw-ADU Bayer frame (e.g. a saved mean_frame.npy)
    through the same pipeline as demosaic_path, so the two are comparable.
    """
    cal = calibrate_frame(frame_adu.astype(np.float32), pattern, black, white)
    return demosaic_to_rgb(cal, pattern, wb=wb, ccm=ccm)


# --------------------------------------------------------------------------- #
# Shift detection                                                               #
# --------------------------------------------------------------------------- #

def _phase_xcorr_map(ref_crop: np.ndarray, tgt_crop: np.ndarray) -> np.ndarray:
    """Normalized cross-power spectrum → correlation map, shifted so zero-lag is centred."""
    R = np.fft.rfft2(ref_crop)
    T = np.fft.rfft2(tgt_crop)
    G = R * np.conj(T)
    denom = np.abs(G)
    G = np.where(denom > 0, G / denom, 0)
    return np.fft.fftshift(np.fft.irfft2(G))


def detect_shifts(paths: list[Path], ref_idx: int = 0,
                  crop_size: int = 512) -> np.ndarray:
    """
    Estimate per-frame translation shifts vs a reference frame using phase
    cross-correlation on a central crop of the raw Bayer data.

    FFT-based with sub-pixel precision. Works on dark frames too: the
    fixed-pattern noise gives the correlation enough structure to lock on.

    Returns float64 (N, 2) — [row_shift, col_shift] per frame, relative to ref.
    """
    try:
        from skimage.registration import phase_cross_correlation
    except ImportError:
        sys.exit("scikit-image is required: pip install scikit-image")

    def _central_crop(arr: np.ndarray, size: int) -> np.ndarray:
        cy, cx = arr.shape[0] // 2, arr.shape[1] // 2
        h = min(size, arr.shape[0]) // 2
        w = min(size, arr.shape[1]) // 2
        return arr[cy - h : cy + h, cx - w : cx + w].astype(np.float64)

    ref_crop = _central_crop(load_raw(paths[ref_idx]), crop_size)

    shifts = np.zeros((len(paths), 2), dtype=np.float64)
    for i, p in enumerate(tqdm(paths, desc="detecting shifts", unit="frame")):
        if i == ref_idx:
            continue
        crop = _central_crop(load_raw(p), crop_size)
        shift, _, _ = phase_cross_correlation(ref_crop, crop, upsample_factor=10)
        shifts[i] = shift

    return shifts


# --------------------------------------------------------------------------- #
# Subcommands                                                                   #
# --------------------------------------------------------------------------- #

def cmd_demosaic(args):
    paths = find_dngs(args.directory)
    if not paths:
        sys.exit(f"No DNGs found in {args.directory}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    pattern, black, white = get_raw_metadata(paths[0])
    wb, ccm = get_color_metadata(paths[0])

    for p in tqdm(paths, desc="demosaicing", unit="frame"):
        rgb = demosaic_path(p, pattern, black, white, wb=wb, ccm=ccm)
        if args.format == "npy":
            np.save(out_dir / f"{p.stem}.npy", rgb)
        else:
            save_rgb_png(rgb, out_dir / f"{p.stem}.png")

    print(f"Saved {len(paths)} frames to {out_dir}/")


def cmd_compare(args):
    frame_path = Path(args.frame)
    mean_path  = Path(args.mean_npy)

    if not frame_path.exists():
        sys.exit(f"Frame not found: {frame_path}")
    if not mean_path.exists():
        sys.exit(f"Reference frame not found: {mean_path}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pattern, black, white = get_raw_metadata(frame_path)
    wb, ccm = get_color_metadata(frame_path)

    frame_rgb = demosaic_path(frame_path, pattern, black, white, wb=wb, ccm=ccm)
    mean_adu  = np.load(str(mean_path))
    mean_rgb  = demosaic_adu_frame(mean_adu, pattern, black, white, wb=wb, ccm=ccm)

    diff = frame_rgb - mean_rgb

    out_base = Path(args.out) if args.out else frame_path.parent / frame_path.stem
    save_rgb_png(frame_rgb, Path(f"{out_base}_frame.png"))
    save_rgb_png(mean_rgb,  Path(f"{out_base}_reference.png"))

    fig, axes = plt.subplots(1, 3, figsize=(19, 6))
    for ax, img, title in [
        (axes[0], frame_rgb, f"Frame: {frame_path.name}"),
        (axes[1], mean_rgb,  f"Reference: {mean_path.name}"),
    ]:
        ax.imshow(img, aspect="auto")
        ax.set_title(title, fontsize=10)
        ax.axis("off")

    diff_lum = diff.mean(axis=2)
    vmax = max(float(np.percentile(np.abs(diff_lum), 99)), 1e-6)
    im = axes[2].imshow(diff_lum, cmap="RdBu_r", aspect="auto",
                        vmin=-vmax, vmax=vmax)
    fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
    axes[2].set_title("Difference (frame − reference), luminance", fontsize=10)
    axes[2].axis("off")

    fig.tight_layout()
    cmp_out = Path(f"{out_base}_comparison.png")
    fig.savefig(cmp_out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {cmp_out}")

    print(f"\nDifference stats (gamma-encoded RGB):")
    print(f"  mean abs : {np.abs(diff).mean():.5f}")
    print(f"  std      : {diff.std():.5f}")
    print(f"  p99 abs  : {np.percentile(np.abs(diff), 99):.5f}")


def cmd_detect_shifts(args):
    paths = find_dngs(args.directory)
    if not paths:
        sys.exit(f"No DNGs found in {args.directory}")

    shifts = detect_shifts(paths, ref_idx=args.ref, crop_size=args.crop)
    mags   = np.linalg.norm(shifts, axis=1)

    print(f"\nShift report  (reference: frame {args.ref} = {paths[args.ref].name})")
    print(f"{'idx':>4}  {'row_shift':>10}  {'col_shift':>10}  {'magnitude':>10}  file")
    print("─" * 72)
    for i, (p, sh, mg) in enumerate(zip(paths, shifts, mags)):
        flag = " ← MOVED" if mg > args.threshold else ""
        print(f"{i:4d}  {sh[0]:10.3f}  {sh[1]:10.3f}  {mg:10.3f}  {p.name}{flag}")

    n_moved = int((mags > args.threshold).sum())
    print(f"\n{n_moved}/{len(paths)} frames exceed {args.threshold}px threshold.")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if args.plot:
        fig, (ax_r, ax_c, ax_m) = plt.subplots(3, 1, figsize=(12, 7), sharex=True)
        idx = np.arange(len(paths))
        for ax, vals, ylabel, color in [
            (ax_r, shifts[:, 0], "Row shift (px)", "C0"),
            (ax_c, shifts[:, 1], "Col shift (px)", "C1"),
            (ax_m, mags,         "Magnitude (px)", "C2"),
        ]:
            ax.plot(idx, vals, "o-", markersize=3, linewidth=0.8, color=color)
            ax.axhline(0, color="k", linewidth=0.5, alpha=0.4)
            ax.set_ylabel(ylabel, fontsize=9)
            ax.grid(True, alpha=0.3, linestyle="--")

        ax_m.axhline(args.threshold, color="r", linewidth=1.0, linestyle="--",
                     label=f"threshold = {args.threshold}px")
        ax_m.legend(fontsize=8)
        ax_m.set_xlabel("Frame index", fontsize=9)

        fig.suptitle(f"Frame shift detection  (ref = frame {args.ref})", fontsize=12)
        fig.tight_layout()
        fig.savefig(args.plot, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved shift plot to {args.plot}")

    if args.sample_crop:
        # Pick the most-shifted frame (excluding ref) for the comparison panel.
        mags_masked = mags.copy()
        mags_masked[args.ref] = -1
        tgt_idx = int(np.argmax(mags_masked))

        def _crop(path):
            raw = load_raw(path)
            cy, cx = raw.shape[0] // 2, raw.shape[1] // 2
            h = min(args.crop, raw.shape[0]) // 2
            w = min(args.crop, raw.shape[1]) // 2
            return raw[cy - h : cy + h, cx - w : cx + w].astype(np.float64)

        ref_crop = _crop(paths[args.ref])
        tgt_crop = _crop(paths[tgt_idx])
        xcorr    = _phase_xcorr_map(ref_crop, tgt_crop)
        sh       = shifts[tgt_idx]

        def _display(arr):
            lo, hi = np.percentile(arr, [0.5, 99.5])
            return np.clip((arr - lo) / max(hi - lo, 1e-6), 0, 1)

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        axes[0].imshow(_display(ref_crop), cmap="gray", aspect="auto")
        axes[0].set_title(f"Reference crop  (frame {args.ref})", fontsize=10)
        axes[0].axis("off")

        axes[1].imshow(_display(tgt_crop), cmap="gray", aspect="auto")
        axes[1].set_title(
            f"Most-shifted frame  (frame {tgt_idx})\n"
            f"shift = ({sh[0]:+.2f} row, {sh[1]:+.2f} col) px", fontsize=10)
        axes[1].axis("off")

        # Cross-correlation map — zoom into central ±20 px where the peak lives.
        cy, cx = xcorr.shape[0] // 2, xcorr.shape[1] // 2
        win = 20
        zoom = xcorr[cy - win : cy + win + 1, cx - win : cx + win + 1]
        im = axes[2].imshow(zoom, cmap="inferno", aspect="auto",
                            extent=[-win, win, win, -win])
        # Mark the detected peak
        axes[2].plot(sh[1], sh[0], "w+", markersize=12, markeredgewidth=1.5)
        axes[2].set_title(
            f"Normalized cross-correlation  (±{win} px window)\n"
            f"peak at ({sh[0]:+.2f}, {sh[1]:+.2f})", fontsize=10)
        axes[2].set_xlabel("col offset (px)")
        axes[2].set_ylabel("row offset (px)")
        fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

        fig.suptitle(
            f"Phase cross-correlation crop check  "
            f"(crop {args.crop}×{args.crop} px, upsample×10)",
            fontsize=12)
        fig.tight_layout()
        fig.savefig(args.sample_crop, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved sample-crop plot to {args.sample_crop}")


# --------------------------------------------------------------------------- #
# Entry point                                                                   #
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(
        description="Demosaic frames and compare against mean/median frames.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_dm = sub.add_parser("demosaic", help="Demosaic all DNGs in a directory.")
    p_dm.add_argument("directory", help="Directory containing DNG files.")
    p_dm.add_argument("--out", default="demosaiced", help="Output directory.")
    p_dm.add_argument("--format", choices=["png", "npy"], default="png",
                      help="Output format (default: png).")

    p_cmp = sub.add_parser("compare",
                            help="Compare one frame to a saved mean/median frame.")
    p_cmp.add_argument("frame",    help="Path to a DNG frame.")
    p_cmp.add_argument("mean_npy", help="mean_frame.npy or median_frame.npy (raw ADU).")
    p_cmp.add_argument("--out",    default=None,
                       help="Output path prefix (default: alongside the frame).")

    p_sh = sub.add_parser("detect_shifts",
                           help="Detect translation shifts between frames.")
    p_sh.add_argument("directory", help="Directory containing DNG files.")
    p_sh.add_argument("--ref",       type=int,   default=0,
                      help="Reference frame index (default: 0).")
    p_sh.add_argument("--crop",      type=int,   default=512,
                      help="Central crop size in pixels (default: 512).")
    p_sh.add_argument("--threshold", type=float, default=0.5,
                      help="Flag frames shifted more than this (default: 0.5 px).")
    p_sh.add_argument("--plot",        default=None, metavar="PATH",
                      help="Save a shift-vs-frame-index plot to PATH.")
    p_sh.add_argument("--sample-crop", default=None, metavar="PATH", dest="sample_crop",
                      help="Save a 3-panel figure: ref crop | most-shifted crop | "
                           "cross-correlation map with peak marked.")

    args = parser.parse_args()
    t0 = time.monotonic()
    {"demosaic":      cmd_demosaic,
     "compare":       cmd_compare,
     "detect_shifts": cmd_detect_shifts}[args.cmd](args)
    elapsed = time.monotonic() - t0
    print(f"Total time: {elapsed:.1f}s" if elapsed < 60
          else f"Total time: {int(elapsed // 60)}m {int(elapsed % 60):02d}s")


if __name__ == "__main__":
    main()
