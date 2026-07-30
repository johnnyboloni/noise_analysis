"""
GT frame aggregation analysis for a static lowlight sequence.

Edit the CONFIG block below, then run:
    python analyze_gt_sequence.py

Outputs (saved to OUTPUT_DIR):
  - gt_sample_frames.png   : evenly-spaced sample frames (RGB)
  - gt_aggregated.png      : mean / median / trimmed-mean side-by-side (RGB)
  - gt_differences.png     : (mean−median), (mean−trimmed), (median−trimmed) heatmaps
  - gt_temporal_noise.png  : per-pixel temporal std heatmap (noise map)
  - gt_convergence.png     : RMS vs N frames (log-log) with 1/√N reference
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from raw_utils import (
    detect_format, find_dngs, find_raws,
    get_raw_metadata, get_raw_metadata_gn3,
    load_raw, load_raw_gn3, load_raw_rgb, load_raw_rgb_gn3,
    calibrate_frame, demosaic_to_rgb, plot_sample_frames,
)


# ============================================================
# CONFIG — edit paths and options here
# ============================================================
SEQUENCE_DIR  = "/path/to/static/sequence"
OUTPUT_DIR    = "output/gt_analysis"
GN3_BLACK_LEVEL = 256    # uniform black level for GN3 .raw files

MAX_FRAMES    = None  # int to cap total frames loaded, None = all
MAX_STACK     = 60    # max frames loaded into RAM for median / trimmed-mean
TRIM_FRAC     = 0.05  # total fraction trimmed (symmetric: TRIM_FRAC/2 from each tail)
N_SAMPLES     = 5     # number of evenly-spaced sample frames to show
# ============================================================


# --------------------------------------------------------------------------- #
# Format-aware loader builder                                                   #
# --------------------------------------------------------------------------- #

def _make_loaders(directory: str):
    """
    Detect format and return (fmt, paths, pattern, black, white, loader, sample_fn).
    loader(path)     → float32 (H, W) raw Bayer ADU
    sample_fn(path)  → uint8  (H, W, 3) demosaiced RGB
    """
    fmt = detect_format(directory)
    if fmt == 'dng':
        paths     = find_dngs(directory)
        pattern, black, white = get_raw_metadata(paths[0])
        loader    = load_raw
        sample_fn = load_raw_rgb
    else:
        paths     = find_raws(directory)
        pattern, black, white = get_raw_metadata_gn3(paths[0], GN3_BLACK_LEVEL)
        meta      = json.loads(paths[0].with_suffix('.imgprops').read_text())
        shape     = (meta['height'], meta['width'])
        loader    = lambda p, _s=shape: load_raw_gn3(p, _s)
        sample_fn = lambda p, _s=shape, _pat=pattern, _bl=black, _wh=white: \
            load_raw_rgb_gn3(p, _s, _pat, _bl, _wh)
    return fmt, paths, pattern, black, white, loader, sample_fn


# --------------------------------------------------------------------------- #
# Streaming passes                                                               #
# --------------------------------------------------------------------------- #

def _stream_mean(paths, pattern, black, white, loader):
    """Pass 1 — compute per-pixel calibrated mean, streaming one frame at a time."""
    accum = None
    n = len(paths)
    for i, p in enumerate(paths):
        print(f"  mean  [{i+1}/{n}] {p.name}", end="\r", flush=True)
        cal   = calibrate_frame(loader(p), pattern, black, white)
        accum = cal.astype(np.float64) if accum is None else accum + cal
    print()
    return (accum / n).astype(np.float32)


def _stream_welford_and_convergence(paths, pattern, black, white, loader,
                                    full_mean, n_checkpoints=40):
    """
    Pass 2 — Welford online algorithm for temporal std, plus convergence tracking.

    Convergence: RMS of the running mean at log-spaced checkpoints vs the final
    full-sequence mean.  Follows ~1/√N for independent frames.

    Returns:
      temporal_std : (H, W) float32  per-pixel noise standard deviation
      convergence  : list of (n_frames, rms) tuples
    """
    n = len(paths)
    checkpoints = set(np.geomspace(1, n, min(n_checkpoints, n)).astype(int).tolist())
    checkpoints.add(n)

    wf_mean  = None
    wf_M2    = None
    run_sum  = None
    convergence = []

    for i, p in enumerate(paths):
        idx = i + 1
        print(f"  Welford  [{idx}/{n}] {p.name}", end="\r", flush=True)
        cal = calibrate_frame(loader(p), pattern, black, white).astype(np.float64)

        if wf_mean is None:
            wf_mean = cal.copy()
            wf_M2   = np.zeros_like(cal)
        else:
            delta   = cal - wf_mean
            wf_mean += delta / idx
            wf_M2   += delta * (cal - wf_mean)

        run_sum = cal.copy() if run_sum is None else run_sum + cal
        if idx in checkpoints:
            running = (run_sum / idx).astype(np.float32)
            rms = float(np.sqrt(np.mean((running - full_mean) ** 2)))
            convergence.append((idx, rms))

    print()
    var = np.where(n > 1, wf_M2 / (n - 1), 0.0).astype(np.float32)
    return np.sqrt(var), convergence


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
    for i in range(1, n):
        print(f"  stack [{i+1}/{n}] {paths[i].name}", end="\r", flush=True)
        mm[i] = calibrate_frame(loader(paths[i]), pattern, black, white)
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

def _plot_aggregated(mean, median, trimmed, pattern, n_stack, out):
    frames = [mean, median, trimmed]
    titles = [
        "Mean (all frames)",
        f"Median  (N={n_stack})",
        f"Trimmed mean  (N={n_stack}, trim={TRIM_FRAC:.0%})",
    ]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, f, t in zip(axes, frames, titles):
        ax.imshow(demosaic_to_rgb(f, pattern), aspect="auto")
        ax.set_title(t, fontsize=10)
        ax.axis("off")
    fig.suptitle("GT aggregation methods", fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def _plot_differences(mean, median, trimmed, out):
    pairs = [
        (mean - median,    "mean − median"),
        (mean - trimmed,   "mean − trimmed"),
        (median - trimmed, "median − trimmed"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, (diff, title) in zip(axes, pairs):
        vmax = max(float(np.percentile(np.abs(diff), 99.5)), 1e-6)
        im = ax.imshow(diff, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_title(title, fontsize=10)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    fig.suptitle("Method differences — where aggregators disagree (outlier / hot-pixel locations)",
                 fontsize=12, y=1.01)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def _plot_temporal_noise(temporal_std, out):
    fig, ax = plt.subplots(figsize=(9, 6))
    vmax = float(np.percentile(temporal_std, 99))
    im = ax.imshow(temporal_std, cmap="inferno", vmin=0, vmax=vmax, aspect="auto")
    ax.set_title("Per-pixel temporal std  (noise map, calibrated units)", fontsize=11)
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02, label="σ [calibrated]")
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def _plot_convergence(convergence, out):
    ns  = np.array([c[0] for c in convergence], dtype=float)
    rms = np.array([c[1] for c in convergence], dtype=float)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.loglog(ns, rms, "o-", color="steelblue", linewidth=1.8,
              markersize=4, label="measured RMS")
    ref = rms[0] * np.sqrt(ns[0]) / np.sqrt(ns)
    ax.loglog(ns, ref, "--", color="tomato", linewidth=1.4,
              label=r"$\propto 1/\sqrt{N}$")
    ax.set_xlabel("Number of frames averaged  (N)", fontsize=11)
    ax.set_ylabel("RMS vs full-sequence mean  [calibrated]", fontsize=11)
    ax.set_title("Convergence of temporal averaging", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, which="both", alpha=0.3, linestyle="--")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved {out}")


# --------------------------------------------------------------------------- #
# Main analysis function                                                         #
# --------------------------------------------------------------------------- #

def analyze_gt_sequence(
    directory: str,
    out_dir: Path,
    max_frames: int | None = None,
    max_stack: int = 60,
) -> None:
    """
    Analyze a static lowlight sequence for GT frame generation.
    Saves five plots to out_dir (see module docstring).
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    fmt, paths, pattern, black, white, loader, sample_fn = _make_loaders(directory)
    if max_frames:
        paths = paths[:max_frames]
    n = len(paths)
    print(f"  {n} {fmt.upper()} frames  |  white={white}  black={black[0]}")

    # Sample frames
    sample_idx = np.linspace(0, n - 1, min(N_SAMPLES, n), dtype=int)
    plot_sample_frames(
        [sample_fn(paths[i]) for i in sample_idx],
        [f"frame {i}" for i in sample_idx],
        out_dir / "gt_sample_frames.png",
    )

    # Pass 1: full streaming mean
    print(f"  Pass 1 — streaming mean ({n} frames) …")
    full_mean = _stream_mean(paths, pattern, black, white, loader)
    print(f"  Mean range: [{full_mean.min():.4f}, {full_mean.max():.4f}]")

    # Pass 2: Welford temporal std + convergence
    print(f"  Pass 2 — temporal std + convergence …")
    temporal_std, convergence = _stream_welford_and_convergence(
        paths, pattern, black, white, loader, full_mean,
    )
    print(f"  Temporal std  mean={temporal_std.mean():.5f}  max={temporal_std.max():.5f}")

    # Median + trimmed mean (disk-backed, no full-stack RAM alloc)
    n_stack  = min(max_stack, n)
    tmp_path = out_dir / "_stack_tmp.npy"
    trim_k   = max(1, int(TRIM_FRAC / 2 * n_stack))
    print(f"  Writing {n_stack}/{n} frames to disk, then computing "
          f"median and trimmed mean (trim_k={trim_k}) …")
    median_frame, trimmed_frame = _compute_median_and_trimmed(
        paths, n_stack, pattern, black, white, loader, TRIM_FRAC, tmp_path,
    )

    # Plots
    print("  Plotting …")
    _plot_aggregated(full_mean, median_frame, trimmed_frame, pattern, n_stack,
                     out_dir / "gt_aggregated.png")
    _plot_differences(full_mean, median_frame, trimmed_frame,
                      out_dir / "gt_differences.png")
    _plot_temporal_noise(temporal_std, out_dir / "gt_temporal_noise.png")
    _plot_convergence(convergence,     out_dir / "gt_convergence.png")

    print(f"\nDone. Outputs in {out_dir.resolve()}")


# --------------------------------------------------------------------------- #
# CLI overrides + entry point                                                   #
# --------------------------------------------------------------------------- #

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
            parser.add_argument(flag, dest=key, default=None,
                                action=argparse.BooleanOptionalAction,
                                help=f"(default: {val})")
        elif isinstance(val, int):
            parser.add_argument(flag, dest=key, type=int, default=None, metavar='N',
                                help=f"(default: {val})")
        elif isinstance(val, float):
            parser.add_argument(flag, dest=key, type=float, default=None, metavar='F',
                                help=f"(default: {val})")
        else:  # str or None
            parser.add_argument(flag, dest=key, type=_none_or_auto, default=None, metavar='S',
                                help=f"(default: {val!r}; pass 'none' to clear)")

    args = parser.parse_args()
    for key, new_val in vars(args).items():
        if new_val is not None:
            g[key] = new_val


def main():
    _apply_cli_overrides()
    print(f"GT sequence analysis: {SEQUENCE_DIR}")
    analyze_gt_sequence(
        SEQUENCE_DIR,
        Path(OUTPUT_DIR),
        max_frames=MAX_FRAMES,
        max_stack=MAX_STACK,
    )


if __name__ == "__main__":
    main()
