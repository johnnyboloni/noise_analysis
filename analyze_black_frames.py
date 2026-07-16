"""
Dark frame noise analysis for DNG sequences.

Edit the CONFIG block below, then run:
    python analyze_black_frames.py

For each subdir of ROOT_DIR, outputs (saved to OUTPUT_DIR/<subdir>/):
  - histogram_adu.png    : per-channel ADU histogram of mean frame, black levels marked
  - read_noise_map.png   : per-pixel temporal std heatmap
  - hot_pixels.png       : binary map of pixels > HOT_PIXEL_NSIGMA × median noise
  - fnp_profiles.png     : column and row fixed-pattern noise profiles
  - temporal_drift.png   : per-frame mean dark level over time
  - summary_all.png      : cross-condition comparison (written to OUTPUT_DIR/)
"""

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import rawpy


# ============================================================
# CONFIG — edit paths and options here
# ============================================================
ROOT_DIR         = "/home/yonif/motioncam/decoded/"  # root containing one subdir per condition
OUTPUT_DIR       = "output_dark"
HIST_BINS        = 512
HOT_PIXEL_NSIGMA = 5.0    # pixels > this many × median noise are flagged as hot
MAX_FRAMES       = None    # int to cap frames per subdir; None = all
# ============================================================

COLORS = list(plt.cm.tab10.colors)
ALPHA  = 0.7
DPI    = 200

_CH_NAMES  = {0: "R", 1: "G1", 2: "B", 3: "G2"}
_CH_COLORS = {0: "#d62728", 1: "#2ca02c", 2: "#1f77b4", 3: "#98df8a"}


# --------------------------------------------------------------------------- #
# CLI override (same pattern as analyze_flats.py)                              #
# --------------------------------------------------------------------------- #

def _none_or_auto(s: str):
    if s.lower() == "none":
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
    g = globals()
    scalar = (bool, int, float, str, type(None))
    keys = sorted(k for k in g if k.isupper() and not k.startswith("_") and isinstance(g[k], scalar))
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    for key in keys:
        val = g[key]
        flag = "--" + key.lower().replace("_", "-")
        if isinstance(val, bool):
            parser.add_argument(flag, dest=key, default=None,
                                action=argparse.BooleanOptionalAction,
                                help=f"(default: {val})")
        elif isinstance(val, int):
            parser.add_argument(flag, dest=key, type=int, default=None, metavar="N",
                                help=f"(default: {val})")
        elif isinstance(val, float):
            parser.add_argument(flag, dest=key, type=float, default=None, metavar="F",
                                help=f"(default: {val})")
        else:
            parser.add_argument(flag, dest=key, type=_none_or_auto, default=None, metavar="S",
                                help=f"(default: {val!r}; 'none' to clear)")
    args = parser.parse_args()
    for key, new_val in vars(args).items():
        if new_val is not None:
            g[key] = new_val


# --------------------------------------------------------------------------- #
# I/O helpers                                                                   #
# --------------------------------------------------------------------------- #

def find_dngs(directory: str) -> list[Path]:
    p = Path(directory)
    return sorted(p.glob("*.dng")) + sorted(p.glob("*.DNG"))


def get_dark_metadata(path: Path):
    """Return (pattern, black, white, iso).  iso is None if not available."""
    with rawpy.imread(str(path)) as raw:
        pattern = raw.raw_pattern.copy()
        black   = np.array(raw.black_level_per_channel, dtype=np.float32)
        white   = float(raw.white_level)
        try:
            iso = int(raw.camera_isoValue)
        except Exception:
            iso = None

    sidecar = path.with_suffix(".json")
    if sidecar.exists():
        try:
            import json
            iso = int(json.loads(sidecar.read_text())["iso"])
        except Exception:
            pass

    return pattern, black, white, iso


def _load_frame(path: Path) -> np.ndarray:
    with rawpy.imread(str(path)) as raw:
        return raw.raw_image_visible.astype(np.float32)


def calibrate_frame(frame: np.ndarray, pattern: np.ndarray,
                    black: np.ndarray, white: float) -> np.ndarray:
    out = np.empty_like(frame, dtype=np.float32)
    for r in range(2):
        for c in range(2):
            ch = int(pattern[r, c])
            bl = float(black[ch])
            out[r::2, c::2] = np.clip((frame[r::2, c::2] - bl) / (white - bl), 0.0, 1.0)
    return out


def _ch_slices(pattern: np.ndarray) -> dict[int, tuple[int, int]]:
    """Map channel index → (row_offset, col_offset) within the 2×2 Bayer kernel."""
    return {int(pattern[r, c]): (r, c) for r in range(2) for c in range(2)}


# --------------------------------------------------------------------------- #
# Grid helpers                                                                  #
# --------------------------------------------------------------------------- #

def _grid(n: int, max_cols: int = 4) -> tuple[int, int]:
    ncols = min(n, max_cols)
    return math.ceil(n / ncols), ncols


def _make_grid_axes(n: int, panel_w: float, panel_h: float, max_cols: int = 4):
    nrows, ncols = _grid(n, max_cols)
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(panel_w * ncols, panel_h * nrows),
                             squeeze=False)
    flat = axes.ravel().tolist()
    for ax in flat[n:]:
        ax.set_visible(False)
    return fig, flat[:n]


# --------------------------------------------------------------------------- #
# Streaming passes                                                               #
# --------------------------------------------------------------------------- #

def stream_dark(paths: list[Path], pattern: np.ndarray,
                black: np.ndarray, white: float):
    """
    Two-pass streaming over dark DNG frames.

    Returns
    -------
    mean_cal    : float32 (H, W)  — per-pixel temporal mean, calibrated [0,1]
    var_cal     : float32 (H, W)  — per-pixel temporal variance, calibrated
    frame_means : float64 (N,)    — per-frame spatial mean (for drift plot)
    adu_counts  : int64 (HIST_BINS,)
    adu_edges   : float64 (HIST_BINS+1,)
    """
    n_bins     = int(white) + 1
    adu_edges  = np.arange(0, white + 2, dtype=np.float64)  # one bin per ADU value
    adu_counts = np.zeros(n_bins, dtype=np.int64)
    accum: np.ndarray | None = None

    for i, p in enumerate(paths):
        print(f"  pass 1  [{i+1}/{len(paths)}] {p.name}", end="\r", flush=True)
        frame = _load_frame(p)
        h, _ = np.histogram(frame, bins=adu_edges)
        adu_counts += h
        cal = calibrate_frame(frame, pattern, black, white)
        accum = cal.astype(np.float64) if accum is None else accum + cal
    print()

    mean_cal = (accum / len(paths)).astype(np.float32)

    var_accum: np.ndarray | None = None
    frame_means = np.zeros(len(paths), dtype=np.float64)

    for i, p in enumerate(paths):
        print(f"  pass 2  [{i+1}/{len(paths)}] {p.name}", end="\r", flush=True)
        cal = calibrate_frame(_load_frame(p), pattern, black, white)
        frame_means[i] = float(cal.mean())
        res = cal.astype(np.float64) - mean_cal
        var_accum = res ** 2 if var_accum is None else var_accum + res ** 2
    print()

    var_cal = (var_accum / len(paths)).astype(np.float32)
    return mean_cal, var_cal, frame_means, adu_counts, adu_edges


# --------------------------------------------------------------------------- #
# Per-subdir plots                                                               #
# --------------------------------------------------------------------------- #

def plot_histogram_adu(adu_counts, adu_edges, black, title, out):
    """Raw ADU histogram (all frames summed) with a black-level marker."""
    bl_mean = float(np.mean(black))
    density = adu_counts / (adu_counts.sum() * 1.0)

    # clip x-axis to rightmost non-zero bin with a small margin
    nonzero = np.nonzero(adu_counts)[0]
    xmax = (adu_edges[nonzero[-1] + 1] * 1.05) if len(nonzero) else adu_edges[-1]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.stairs(density, adu_edges, fill=True, color=COLORS[0], alpha=ALPHA, linewidth=0.8)
    ax.axvline(bl_mean, color="red", linewidth=1.5, linestyle="--",
               label=f"black level ({bl_mean:.0f})")
    ax.set_xlim(left=0, right=xmax)
    ax.set_xlabel("Raw pixel value (ADU)", fontsize=10)
    ax.set_ylabel("Density", fontsize=10)
    ax.set_title(title, fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, linestyle="--")
    fig.tight_layout()
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    print(f"  Saved {out.name}")


def plot_read_noise_map(var_cal, title, out):
    std_cal = np.sqrt(var_cal)
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(std_cal, cmap="hot", aspect="auto")
    fig.colorbar(im, ax=ax, label="temporal σ (calibrated)")
    ax.set_title(title, fontsize=12)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out.name}")


def plot_hot_pixels(var_cal, title, out) -> int:
    std_cal  = np.sqrt(var_cal)
    thresh   = HOT_PIXEL_NSIGMA * float(np.median(std_cal))
    hot_mask = std_cal > thresh
    n_hot    = int(hot_mask.sum())
    pct      = 100.0 * n_hot / hot_mask.size

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(hot_mask.astype(np.uint8), cmap="Reds", aspect="auto", vmin=0, vmax=1)
    ax.text(0.01, 0.99,
            f"{n_hot:,} hot pixels ({pct:.3f}%)\nthreshold: >{HOT_PIXEL_NSIGMA}× median σ",
            transform=ax.transAxes, va="top", ha="left", fontsize=9,
            bbox=dict(facecolor="white", alpha=0.75, edgecolor="none"))
    ax.set_title(title, fontsize=12)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out.name}")
    return n_hot


def plot_fnp_profiles(mean_cal, title, out):
    col_profile = mean_cal.mean(axis=0)
    row_profile = mean_cal.mean(axis=1)

    fig, (ax_c, ax_r) = plt.subplots(2, 1, figsize=(10, 6))
    for ax, profile, xlabel, subtitle in [
        (ax_c, col_profile, "Column index", "Column FPN"),
        (ax_r, row_profile, "Row index",    "Row FPN"),
    ]:
        ax.plot(profile, linewidth=0.6)
        ax.axhline(float(np.median(profile)), color="k", linewidth=1.0,
                   linestyle="--", alpha=0.6, label="median")
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_ylabel("Mean dark (calibrated)", fontsize=9)
        ax.set_title(subtitle, fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, linestyle="--")

    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    print(f"  Saved {out.name}")


def plot_temporal_drift(frame_means, title, out):
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(frame_means, "o-", markersize=3, linewidth=1.0, color=COLORS[0])
    ax.axhline(float(np.median(frame_means)), color="k", linewidth=1.0,
               linestyle="--", alpha=0.6, label="median")
    ax.set_xlabel("Frame index", fontsize=10)
    ax.set_ylabel("Spatial mean dark (calibrated)", fontsize=10)
    ax.set_title(title, fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, linestyle="--")
    fig.tight_layout()
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    print(f"  Saved {out.name}")


# --------------------------------------------------------------------------- #
# Summary plot — across all conditions                                           #
# --------------------------------------------------------------------------- #

def plot_summary(results: list[dict], out: Path):
    n = len(results)
    fig, (ax_h, ax_n, ax_hp) = plt.subplots(1, 3, figsize=(15, 5))

    for i, r in enumerate(results):
        color = COLORS[i % len(COLORS)]
        vals    = r["mean_cal"].ravel().astype(np.float64)
        counts, edges = np.histogram(vals, bins=HIST_BINS, range=(0, 1))
        width   = edges[1] - edges[0]
        density = counts / (counts.sum() * width)
        ax_h.stairs(density, edges, fill=True, color=color, alpha=0.45,
                    label=r["label"], linewidth=0.8)

    ax_h.set_xlabel("Mean dark (calibrated)", fontsize=9)
    ax_h.set_ylabel("Density (log)", fontsize=9)
    ax_h.set_title("Mean dark distributions", fontsize=11)
    ax_h.set_yscale("log")
    ax_h.legend(fontsize=7)
    ax_h.grid(True, alpha=0.3, linestyle="--")

    xlabels = [r["label"] for r in results]
    xpos    = range(n)
    colors  = [COLORS[i % len(COLORS)] for i in range(n)]

    ax_n.bar(xpos, [r["median_noise"] for r in results], color=colors)
    ax_n.set_xticks(xpos)
    ax_n.set_xticklabels(xlabels, rotation=35, ha="right", fontsize=7)
    ax_n.set_ylabel("Median read noise σ (calibrated)", fontsize=9)
    ax_n.set_title("Read noise by condition", fontsize=11)
    ax_n.grid(True, alpha=0.3, linestyle="--", axis="y")

    ax_hp.bar(xpos, [r["n_hot"] for r in results], color=colors)
    ax_hp.set_xticks(xpos)
    ax_hp.set_xticklabels(xlabels, rotation=35, ha="right", fontsize=7)
    ax_hp.set_ylabel("Hot pixel count", fontsize=9)
    ax_hp.set_title(f"Hot pixels (>{HOT_PIXEL_NSIGMA}σ) by condition", fontsize=11)
    ax_hp.grid(True, alpha=0.3, linestyle="--", axis="y")

    fig.suptitle("Dark frame summary — all conditions", fontsize=13)
    fig.tight_layout()
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


# --------------------------------------------------------------------------- #
# Main                                                                          #
# --------------------------------------------------------------------------- #

def main():
    _apply_cli_overrides()

    root    = Path(ROOT_DIR)
    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not root.exists():
        sys.exit(f"ERROR: ROOT_DIR does not exist: {root}")

    subdirs = sorted(d for d in root.iterdir() if d.is_dir())
    if not subdirs:
        sys.exit(f"No subdirectories found under {root}")

    print(f"Root    : {root}  ({len(subdirs)} subdir(s))")
    print(f"Output  : {out_dir.resolve()}")
    if MAX_FRAMES:
        print(f"Frame cap: {MAX_FRAMES}")

    results = []
    for d in subdirs:
        paths = find_dngs(str(d))
        if not paths:
            print(f"\n[{d.name}] no DNGs found — skipping")
            continue
        if MAX_FRAMES:
            paths = paths[:MAX_FRAMES]
        n = len(paths)

        pattern, black, white, iso = get_dark_metadata(paths[0])
        iso_str  = str(iso) if iso is not None else "?"
        label    = f"{d.name}  (N={n}, ISO {iso_str})"
        t_suffix = f"{d.name}  |  ISO {iso_str}  (N={n})"

        print(f"\n{'─'*60}")
        print(f"  {label}")
        print(f"  Black levels: {black}  White: {white:.0f}")

        mean_cal, var_cal, frame_means, adu_counts, adu_edges = stream_dark(
            paths, pattern, black, white,
        )

        std_cal      = np.sqrt(var_cal)
        median_noise = float(np.median(std_cal))

        sub_out = out_dir / d.name
        sub_out.mkdir(exist_ok=True)

        plot_histogram_adu(adu_counts, adu_edges, black,
                           title=f"Dark ADU histogram — {t_suffix}",
                           out=sub_out / "histogram_adu.png")

        plot_read_noise_map(var_cal,
                            title=f"Per-pixel read noise — {t_suffix}",
                            out=sub_out / "read_noise_map.png")

        n_hot = plot_hot_pixels(var_cal,
                                title=f"Hot pixels (>{HOT_PIXEL_NSIGMA}σ) — {t_suffix}",
                                out=sub_out / "hot_pixels.png")

        plot_fnp_profiles(mean_cal,
                          title=f"Fixed-pattern noise profiles — {t_suffix}",
                          out=sub_out / "fnp_profiles.png")

        plot_temporal_drift(frame_means,
                            title=f"Temporal dark-level drift — {t_suffix}",
                            out=sub_out / "temporal_drift.png")

        print(f"  Median read noise σ : {median_noise:.6f} (calibrated)")
        print(f"  Hot pixels          : {n_hot:,} ({100*n_hot/std_cal.size:.3f}%)")

        results.append(dict(
            label=label, iso=iso,
            mean_cal=mean_cal, median_noise=median_noise, n_hot=n_hot,
        ))

    if not results:
        sys.exit("No sequences processed.")

    print(f"\n{'─'*60}")
    plot_summary(results, out_dir / "summary_all.png")
    print(f"\nDone. All plots saved under: {out_dir.resolve()}/")


if __name__ == "__main__":
    main()
