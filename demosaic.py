#!/usr/bin/env python3
"""
demosaic.py — custom bilinear Bayer demosaic and frame utilities.

Subcommands
-----------
demosaic        Demosaic all DNGs in a directory and save as .npy or PNG.
compare         Subtract the saved mean/median frame from a single demosaiced frame.
detect_shifts   Report per-frame translation shifts (no alignment applied).

Example usage
-------------
python demosaic.py demosaic /data/dngs --out output/
python demosaic.py compare  /data/dngs/frame_001.dng output/mean_frame.npy
python demosaic.py detect_shifts /data/dngs --ref 0 --crop 512
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import rawpy
from scipy.ndimage import convolve
from tqdm import tqdm


# --------------------------------------------------------------------------- #
# Core demosaic utilities (importable)                                          #
# --------------------------------------------------------------------------- #

def load_bayer(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Load the raw Bayer array and metadata from a DNG.

    Returns
    -------
    bayer   : float32 (H, W)  — raw ADU values, visible area only
    pattern : int32   (2, 2)  — Bayer pattern; values 0=R,1=G1,2=B,3=G2
    black   : float32 (4,)    — per-channel black levels
    white   : float           — white level (saturation point)
    """
    with rawpy.imread(str(path)) as raw:
        bayer   = raw.raw_image_visible.copy().astype(np.float32)
        pattern = raw.raw_pattern.copy()
        black   = np.array(raw.black_level_per_channel, dtype=np.float32)
        white   = float(raw.white_level)
    return bayer, pattern, black, white


def demosaic_bilinear(bayer: np.ndarray, pattern: np.ndarray) -> np.ndarray:
    """
    Bilinear Bayer demosaic — no rawpy postprocessing.

    Separates the four CFA positions into R, G (both greens merged), B planes,
    then fills missing pixels for each channel via a distance-weighted 3×3
    bilinear kernel applied to the known neighbours.

    Parameters
    ----------
    bayer   : float32 (H, W)  — raw ADU values
    pattern : int32   (2, 2)  — rawpy raw_pattern; 0=R,1=G1,2=B,3=G2

    Returns
    -------
    rgb : float32 (H, W, 3)  — [R, G, B], values in raw ADU
    """
    CH_MAP = {0: 0, 1: 1, 2: 2, 3: 1}   # rawpy ch → RGB index (G1,G2 → 1)

    H, W = bayer.shape
    planes = np.zeros((3, H, W), dtype=np.float32)
    masks  = np.zeros((3, H, W), dtype=np.float32)

    for r in range(2):
        for c in range(2):
            ch     = int(pattern[r, c])
            out_ch = CH_MAP[ch]
            planes[out_ch, r::2, c::2] = bayer[r::2, c::2]
            masks[out_ch,  r::2, c::2] = 1.0

    # Bilinear weighting kernel: distance-weighted 3×3
    K = np.array([[1, 2, 1],
                  [2, 4, 2],
                  [1, 2, 1]], dtype=np.float32)

    rgb = np.empty((H, W, 3), dtype=np.float32)
    for ch in range(3):
        num = convolve(planes[ch] * masks[ch], K, mode="mirror")
        den = convolve(masks[ch],              K, mode="mirror")
        # Keep original values where the sensor sampled that channel;
        # use interpolated value only where it's missing.
        rgb[:, :, ch] = np.where(masks[ch] > 0,
                                  planes[ch],
                                  num / np.maximum(den, 1e-6))
    return rgb


def calibrate(rgb: np.ndarray, black: np.ndarray, white: float,
              pattern: np.ndarray) -> np.ndarray:
    """
    Black-subtract and normalise demosaiced [R,G,B] to [0, 1].
    black is the per-rawpy-channel array (4 values); we map it to RGB order.
    """
    CH_MAP = {0: 0, 1: 1, 2: 2, 3: 1}
    bl_rgb = [0.0, 0.0, 0.0]
    for r in range(2):
        for c in range(2):
            ch = int(pattern[r, c])
            bl_rgb[CH_MAP[ch]] = max(bl_rgb[CH_MAP[ch]], float(black[ch]))

    out = rgb.copy()
    for ch in range(3):
        out[:, :, ch] = np.clip(
            (rgb[:, :, ch] - bl_rgb[ch]) / max(white - bl_rgb[ch], 1.0),
            0.0, 1.0,
        )
    return out.astype(np.float32)


def compare_frame_to_mean(frame_rgb: np.ndarray,
                          mean_adu: np.ndarray,
                          pattern: np.ndarray) -> np.ndarray:
    """
    Subtract the (raw ADU) mean frame from a demosaiced frame's luminance.

    mean_adu is the Bayer-space mean (H, W). We compute per-channel mean
    images from it, then subtract channel-wise from frame_rgb.

    Returns diff (H, W, 3) in ADU — positive = frame brighter than mean.
    """
    CH_MAP = {0: 0, 1: 1, 2: 2, 3: 1}
    H, W = mean_adu.shape
    mean_rgb = np.zeros((H, W, 3), dtype=np.float32)
    masks    = np.zeros((H, W, 3), dtype=np.float32)

    for r in range(2):
        for c in range(2):
            ch     = int(pattern[r, c])
            out_ch = CH_MAP[ch]
            mean_rgb[..., out_ch][r::2, c::2] = mean_adu[r::2, c::2]
            masks[..., out_ch][r::2, c::2]    = 1.0

    # Demosaic the mean frame using the same bilinear kernel
    K = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=np.float32)
    mean_demosaiced = np.empty_like(mean_rgb)
    for ch in range(3):
        num = convolve(mean_rgb[..., ch] * masks[..., ch], K, mode="mirror")
        den = convolve(masks[..., ch],                     K, mode="mirror")
        mean_demosaiced[..., ch] = np.where(
            masks[..., ch] > 0, mean_rgb[..., ch], num / np.maximum(den, 1e-6)
        )

    return frame_rgb - mean_demosaiced


# --------------------------------------------------------------------------- #
# Shift detection                                                               #
# --------------------------------------------------------------------------- #

def detect_shifts(paths: list[Path], ref_idx: int = 0,
                  crop_size: int = 512) -> np.ndarray:
    """
    Estimate per-frame translation shifts vs a reference frame using
    phase cross-correlation on a central crop of the raw Bayer data.

    Phase cross-correlation is FFT-based and gives sub-pixel precision.
    It works on dark frames too: the fixed-pattern noise provides enough
    spatial structure for the correlation to lock on.

    Parameters
    ----------
    paths     : list of DNG paths
    ref_idx   : which frame to use as reference (default: 0)
    crop_size : side length of the central square crop (pixels)

    Returns
    -------
    shifts : float64 (N, 2)  — [row_shift, col_shift] per frame,
             relative to ref. A shift of (0,0) means no movement.
    """
    try:
        from skimage.registration import phase_cross_correlation
    except ImportError:
        sys.exit("scikit-image is required for shift detection: pip install scikit-image")

    def _central_crop(arr: np.ndarray, size: int) -> np.ndarray:
        cy, cx = arr.shape[0] // 2, arr.shape[1] // 2
        h = min(size, arr.shape[0]) // 2
        w = min(size, arr.shape[1]) // 2
        return arr[cy - h : cy + h, cx - w : cx + w].astype(np.float64)

    # Load reference crop
    ref_bayer, _, _, _ = load_bayer(paths[ref_idx])
    ref_crop = _central_crop(ref_bayer, crop_size)

    shifts = np.zeros((len(paths), 2), dtype=np.float64)
    for i, p in enumerate(tqdm(paths, desc="detecting shifts", unit="frame")):
        if i == ref_idx:
            continue
        bayer, _, _, _ = load_bayer(p)
        crop = _central_crop(bayer, crop_size)
        shift, _, _ = phase_cross_correlation(ref_crop, crop, upsample_factor=10)
        shifts[i] = shift   # [row_shift, col_shift]

    return shifts


# --------------------------------------------------------------------------- #
# CLI subcommands                                                               #
# --------------------------------------------------------------------------- #

def _find_dngs(directory: str) -> list[Path]:
    p = Path(directory)
    return sorted(p.glob("*.dng")) + sorted(p.glob("*.DNG"))


def _save_png(rgb: np.ndarray, path: Path) -> None:
    """Save an (H, W, 3) float32 array as a uint8 PNG using PIL."""
    from PIL import Image
    img = (np.clip(rgb, 0.0, 1.0) * 255).astype(np.uint8)
    Image.fromarray(img).save(path)


def cmd_demosaic(args):
    paths = _find_dngs(args.directory)
    if not paths:
        sys.exit(f"No DNGs found in {args.directory}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    for p in tqdm(paths, desc="demosaicing", unit="frame"):
        bayer, pattern, black, white = load_bayer(p)
        rgb = demosaic_bilinear(bayer, pattern)

        if args.calibrate:
            rgb    = calibrate(rgb, black, white, pattern)
            suffix = "_cal"
        else:
            suffix = ""

        if args.format == "npy":
            np.save(out_dir / f"{p.stem}{suffix}.npy", rgb)
        else:
            if not args.calibrate:
                lo, hi = rgb.min(), rgb.max()
                rgb = (rgb - lo) / max(hi - lo, 1.0)
            _save_png(rgb, out_dir / f"{p.stem}{suffix}.png")

    print(f"Saved {len(paths)} frames to {out_dir}/")


def cmd_compare(args):
    frame_path = Path(args.frame)
    mean_path  = Path(args.mean_npy)

    if not frame_path.exists():
        sys.exit(f"Frame not found: {frame_path}")
    if not mean_path.exists():
        sys.exit(f"Mean frame not found: {mean_path}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bayer, pattern, black, white = load_bayer(frame_path)
    rgb      = demosaic_bilinear(bayer, pattern)
    mean_adu = np.load(str(mean_path))

    diff = compare_frame_to_mean(rgb, mean_adu, pattern)

    # Plot: frame luminance, mean luminance, difference
    lum      = rgb.mean(axis=2)
    mean_lum = mean_adu   # scalar Bayer-space mean used for rough display
    diff_lum = diff.mean(axis=2)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for ax, img, title, cmap in [
        (axes[0], lum,      f"Frame: {frame_path.name}",  "gray"),
        (axes[1], np.load(str(mean_path)), f"Mean frame: {mean_path.name}", "gray"),
        (axes[2], diff_lum, "Difference (frame − mean)", "RdBu_r"),
    ]:
        p_lo, p_hi = np.percentile(img, [1, 99])
        kwargs = dict(vmin=-max(abs(p_lo), abs(p_hi)), vmax=max(abs(p_lo), abs(p_hi))) \
                 if "RdBu" in cmap else dict(vmin=p_lo, vmax=p_hi)
        im = ax.imshow(img, cmap=cmap, aspect="auto", **kwargs)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(title, fontsize=10)
        ax.axis("off")

    fig.tight_layout()
    out = Path(args.out) if args.out else frame_path.with_suffix("_vs_mean.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved comparison to {out}")


def cmd_detect_shifts(args):
    paths = _find_dngs(args.directory)
    if not paths:
        sys.exit(f"No DNGs found in {args.directory}")

    shifts = detect_shifts(paths, ref_idx=args.ref, crop_size=args.crop)
    mags   = np.linalg.norm(shifts, axis=1)

    print(f"\nShift report  (reference: frame {args.ref} = {paths[args.ref].name})")
    print(f"{'idx':>4}  {'row_shift':>10}  {'col_shift':>10}  {'magnitude':>10}  {'file'}")
    print("─" * 72)
    for i, (p, sh, mg) in enumerate(zip(paths, shifts, mags)):
        flag = " ← MOVED" if mg > args.threshold else ""
        print(f"{i:4d}  {sh[0]:10.3f}  {sh[1]:10.3f}  {mg:10.3f}  {p.name}{flag}")

    n_moved = int((mags > args.threshold).sum())
    print(f"\n{n_moved}/{len(paths)} frames exceed {args.threshold}px threshold.")

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, (ax_r, ax_c, ax_m) = plt.subplots(3, 1, figsize=(12, 7), sharex=True)
        idx = np.arange(len(paths))
        ax_r.plot(idx, shifts[:, 0], "o-", markersize=3, linewidth=0.8)
        ax_r.axhline(0, color="k", linewidth=0.5, alpha=0.4)
        ax_r.set_ylabel("Row shift (px)")
        ax_r.grid(True, alpha=0.3, linestyle="--")

        ax_c.plot(idx, shifts[:, 1], "o-", markersize=3, linewidth=0.8, color="C1")
        ax_c.axhline(0, color="k", linewidth=0.5, alpha=0.4)
        ax_c.set_ylabel("Col shift (px)")
        ax_c.grid(True, alpha=0.3, linestyle="--")

        ax_m.plot(idx, mags, "o-", markersize=3, linewidth=0.8, color="C2")
        ax_m.axhline(args.threshold, color="r", linewidth=1.0, linestyle="--",
                     label=f"threshold = {args.threshold}px")
        ax_m.set_ylabel("Magnitude (px)")
        ax_m.set_xlabel("Frame index")
        ax_m.legend(fontsize=8)
        ax_m.grid(True, alpha=0.3, linestyle="--")

        fig.suptitle(f"Frame shift detection  (ref = frame {args.ref})", fontsize=12)
        fig.tight_layout()
        plot_out = Path(args.plot)
        fig.savefig(plot_out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved shift plot to {plot_out}")


# --------------------------------------------------------------------------- #
# Entry point                                                                   #
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(
        description="Custom Bayer demosaic and frame analysis utilities.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    # demosaic subcommand
    p_dm = sub.add_parser("demosaic", help="Demosaic all DNGs in a directory.")
    p_dm.add_argument("directory", help="Directory containing DNG files.")
    p_dm.add_argument("--out", default="demosaiced", help="Output directory.")
    p_dm.add_argument("--format", choices=["npy", "png"], default="png",
                      help="Output format (default: png).")
    p_dm.add_argument("--calibrate", action="store_true",
                      help="Black-subtract and normalise to [0,1].")

    # compare subcommand
    p_cmp = sub.add_parser("compare", help="Compare one frame to the mean frame.")
    p_cmp.add_argument("frame",    help="Path to a DNG frame.")
    p_cmp.add_argument("mean_npy", help="Path to mean_frame.npy (from analyze_black_frames.py).")
    p_cmp.add_argument("--out",    default=None, help="Output PNG path.")

    # detect_shifts subcommand
    p_sh = sub.add_parser("detect_shifts",
                           help="Detect translation shifts between frames.")
    p_sh.add_argument("directory", help="Directory containing DNG files.")
    p_sh.add_argument("--ref",       type=int,   default=0,
                      help="Reference frame index (default: 0).")
    p_sh.add_argument("--crop",      type=int,   default=512,
                      help="Central crop size in pixels (default: 512).")
    p_sh.add_argument("--threshold", type=float, default=0.5,
                      help="Flag frames with shift > this many px (default: 0.5).")
    p_sh.add_argument("--plot",      default=None, metavar="PATH",
                      help="Save a shift-vs-frame-index plot to PATH.")

    args = parser.parse_args()
    {"demosaic": cmd_demosaic,
     "compare":  cmd_compare,
     "detect_shifts": cmd_detect_shifts}[args.cmd](args)


if __name__ == "__main__":
    main()
