"""
Is there a defect population invisible to level-based hot pixel detection?

analyze_gt_sequence.py's defect detector flags pixels whose LEVEL -- their
value in the averaged master dark -- is anomalous relative to their local
same-colour neighbours. That is blind by construction to a pixel whose level
is perfectly normal but whose NOISE is not: dark shot noise scales with dark
current, and random telegraph signal (RTS, a documented CMOS-specific
mechanism, more prominent at high gain) can elevate a pixel's temporal
variance without moving its average at all. Neither shows up in a mean.

This script builds the other map: per-pixel temporal standard deviation
across the same dark sequence (raw_utils.temporal_std_map), then runs the
exact same detection machinery analyze_gt_sequence.py already uses for level
defects -- local same-colour median subtraction, MAD-thresholded -- on THAT
map instead. The comparison that answers the question in the title: how many
of the pixels this flags were already caught by level-based detection, and
how many were not.

Motivated directly by a real result: on a gain64 capture, interpolating the
level-based hot-pixel map on top of dark subtraction made no measurable
difference to the residual noise floor (check_dark_subtraction_convergence.py
showed the dark-subtracted and dark-subtracted+interpolated curves
overlapping). That is consistent with the residual being a noise-level
problem, not a dark-current-level problem -- this script tests that directly
rather than leaving it as an inference.

Edit the CONFIG block below, then run:
    python check_noisy_pixels.py

Output (saved to OUTPUT_DIR/<dark_dir_name>/):
  - mean_histogram.png : histogram of the master dark's per-pixel mean values
    (raw ADU) -- the level distribution the noise-based map is checked
    against, so a genuinely bimodal/RTS-like level distribution is visible
    directly rather than only inferred from the noise map.
  - noise_uniformity.png : is the sensor's TEMPORAL NOISE spatially uniform?
    Same four-panel report as analyze_gt_sequence.py's gt_dark_uniformity,
    applied to the std map instead of the mean -- full resolution,
    block-averaged, row/column profiles.
  - noise_sigma_scan.png : excess-over-chance table and histogram for
    choosing NOISE_SIGMA, exactly like the level-based defect_sigma_scan.
  - noisy_pixel_map.png / noisy_pixels.npy : pixels flagged as anomalously
    noisy (elevated variance) or anomalously quiet (suppressed variance --
    a plausible signature of a stuck pixel that does not fluctuate at all).
  - a printed comparison against the level-based (hot/cold) defect map built
    from the same dark sequence: how many pixels both methods agree on, and
    -- the number that answers the question in the title -- how many are
    flagged as defective ONLY by the noise-based method.
"""

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
    temporal_std_map, format_duration,
)
import analyze_gt_sequence as gt_seq

# --------------------------------------------------------------------------- #
# CONFIG -- edit paths and options here
# --------------------------------------------------------------------------- #

DARK_DIR    = "/path/to/dark/frames"
OUTPUT_DIR  = "./noisy_pixel_output"

GN3_BLACK_LEVEL = 256
DARK_MAX_FRAMES = None   # cap the darks used, None = every frame found
DARK_SIGMA_CLIP = 4.0    # for the level-based (mean) master dark, comparison only
HOT_PIXEL_SIGMA = 5.0    # level-based threshold, for the comparison
NOISE_SIGMA     = 5.0    # noise-based threshold -- see analyze_gt_sequence.py's
                         # HOT_PIXEL_SIGMA comment; same reasoning, applied to
                         # the std map. Tune with noise_sigma_scan.png.
DEFECT_METHOD   = "local"   # see analyze_gt_sequence.py's DEFECT_METHOD comment
LOAD_WORKERS    = 4


def _plot_mean_histogram(dark_adu: np.ndarray, black: float, out: Path) -> None:
    """Histogram of the master dark's per-pixel mean values (raw ADU)."""
    vals = dark_adu.ravel()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(vals, bins=200, color="steelblue", log=True)
    ax.axvline(black, color="crimson", linestyle="--", linewidth=1,
               label=f"black level = {black:.1f}")
    ax.axvline(vals.mean(), color="black", linestyle=":", linewidth=1,
               label=f"mean = {vals.mean():.2f}")
    ax.set_xlabel("Master dark value (raw ADU)")
    ax.set_ylabel("Pixel count (log)")
    ax.set_title("Histogram of mean frame values (master dark)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


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


import argparse

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
    gt_seq.LOAD_WORKERS = LOAD_WORKERS
    t0 = time.monotonic()

    print(f"Dark frames: {DARK_DIR}")
    fmt, paths, pattern, black, white, loader = _make_loader(DARK_DIR)
    n = len(paths)
    print(f"  {n} {fmt.upper()} frames  |  white={white}  black={black[0]}")
    out_dir = Path(OUTPUT_DIR) / Path(DARK_DIR).name
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Level-based map, for comparison: the same master dark and defect ---
    # detection analyze_gt_sequence.py's dark-frame processing already builds.
    n_want = DARK_MAX_FRAMES or n
    print(f"\nSigma-clipped master dark (level) from {min(n_want, n)}/{n} frames …")
    dark_adu, d_stack = gt_seq._dark_master(paths, n_want, loader, DARK_SIGMA_CLIP)
    _plot_mean_histogram(dark_adu, black[0], out_dir / "mean_histogram.png")
    resid = gt_seq._dark_master_residual(paths, d_stack, loader, black, white)
    level_mask = np.zeros(dark_adu.shape, dtype=bool)
    if resid is not None:
        resid_adu = resid * float(white - black[0])
        level_mask, n_hot, n_cold = gt_seq._defect_map(
            dark_adu, pattern, resid_adu, HOT_PIXEL_SIGMA, method=DEFECT_METHOD)
        print(f"  Level-based map ({DEFECT_METHOD}, >{HOT_PIXEL_SIGMA}σ): "
              f"{n_hot} hot, {n_cold} cold, {level_mask.mean()*100:.4f}% of pixels")
    else:
        resid_adu = 1e-3   # degenerate-data fallback for _defect_map below
        print("  Need at least 2 dark frames for the level-based map -- skipping.")

    # --- Noise-based map: the same detection, run on temporal std instead. ---
    print(f"\nTemporal std map from {n_want}/{n} frames …")
    std_map = temporal_std_map(paths[:n_want], loader, LOAD_WORKERS)
    print(f"  Per-pixel temporal std: mean={std_map.mean():.3f}  "
          f"min={std_map.min():.3f}  max={std_map.max():.3f} ADU")

    gt_seq._report_dark_uniformity(std_map, pattern, out_dir / "noise_uniformity.png")

    noisy_mask, n_noisy, n_quiet = gt_seq._defect_map(
        std_map, pattern, resid_adu, NOISE_SIGMA, method=DEFECT_METHOD)
    frac = noisy_mask.mean() * 100
    print(f"\nNoise-based map ({DEFECT_METHOD}, >{NOISE_SIGMA}σ): "
          f"{n_noisy} excess-noise, {n_quiet} anomalously quiet "
          f"(possibly stuck), {frac:.4f}% of pixels")
    gt_seq._defect_sigma_scan(std_map, pattern,
                              out_dir / "noise_sigma_scan.png", NOISE_SIGMA,
                              method=DEFECT_METHOD)
    if noisy_mask.any():
        gt_seq._plot_defect_map(noisy_mask, n_noisy, n_quiet,
                                out_dir / "noisy_pixel_map.png",
                                source="dark-sequence temporal std map",
                                labels=("excess-noise", "anomalously quiet"))
        np.save(out_dir / "noisy_pixels.npy", noisy_mask)

    # --- The comparison this script exists for. ---
    both  = int((level_mask & noisy_mask).sum())
    only_level = int((level_mask & ~noisy_mask).sum())
    only_noise = int((noisy_mask & ~level_mask).sum())
    union = int((level_mask | noisy_mask).sum())
    print(f"\nLevel-based vs noise-based defect maps:")
    print(f"  flagged by both               : {both}")
    print(f"  flagged by level-based only    : {only_level}")
    print(f"  flagged by noise-based only    : {only_noise}  "
          f"<- invisible to the existing level-based detector")
    if union:
        print(f"  overlap (Jaccard)              : {both / union:.3f}  "
              f"(1.0 = identical populations, 0.0 = fully disjoint)")
        if only_noise > both:
            print(f"  -> the noise-based map found a defect population level-based "
                  f"detection mostly misses.")
        elif both == 0:
            print(f"  -> the two populations are entirely disjoint -- noisy pixels "
                  f"and hot pixels are not the same pixels here.")

    print(f"\nTotal time: {format_duration(time.monotonic() - t0)}")
    print(f"Outputs in {out_dir.resolve()}")


if __name__ == "__main__":
    main()
