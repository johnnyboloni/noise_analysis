"""
Shared I/O, calibration, demosaicing, and minimal plot helpers
for DNG (rawpy) and GN3 .raw sequences.

Imported by analyze_flats.py and analyze_gt_sequence.py.
Does NOT call matplotlib.use() — each script sets its own backend first.
"""

import json
import sys
from pathlib import Path

import numpy as np
import rawpy
import matplotlib.pyplot as plt

try:
    import cv2
    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False


# --------------------------------------------------------------------------- #
# Bayer pattern lookup (GN3 .imgprops bayerOrder → 2×2 pattern array)         #
# --------------------------------------------------------------------------- #

_BAYER_ORDER_TO_PATTERN: dict[str, np.ndarray] = {
    'RGGB': np.array([[0, 1], [3, 2]], dtype=np.int32),
    'GRBG': np.array([[1, 0], [2, 3]], dtype=np.int32),
    'BGGR': np.array([[2, 3], [1, 0]], dtype=np.int32),
    'GBRG': np.array([[1, 2], [0, 3]], dtype=np.int32),
}


# --------------------------------------------------------------------------- #
# File discovery + format detection                                             #
# --------------------------------------------------------------------------- #

def find_dngs(directory: str) -> list[Path]:
    p = Path(directory)
    return sorted(p.glob("*.dng")) + sorted(p.glob("*.DNG"))


def find_raws(directory: str) -> list[Path]:
    return sorted(Path(directory).glob("*.raw"))


def detect_format(directory: str) -> str:
    """Return 'dng' or 'raw' based on which files are present; exit on neither."""
    if find_dngs(directory):
        return 'dng'
    if find_raws(directory):
        return 'raw'
    sys.exit(f"No recognized files (.dng / .raw) in {directory}")


# --------------------------------------------------------------------------- #
# Metadata                                                                      #
# --------------------------------------------------------------------------- #

def get_raw_metadata(path: Path) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Return (bayer_pattern, black_level_per_channel, white_level) from a DNG.
      bayer_pattern          : 2×2 int array, values 0=R 1=G 2=B 3=G2
      black_level_per_channel: float32 array shape (4,)
      white_level            : scalar float
    """
    with rawpy.imread(str(path)) as raw:
        pattern = raw.raw_pattern.copy()
        black   = np.array(raw.black_level_per_channel, dtype=np.float32)
        white   = float(raw.white_level)
    return pattern, black, white


def get_raw_metadata_gn3(path: Path,
                          gn3_black_level: float = 256,
                          ) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Read GN3 .imgprops sidecar JSON (same stem as .raw file).
    Returns (bayer_pattern, black_level_per_channel, white_level).
    white_level is derived from imageType  (BAYER10 → 1023, BAYER12 → 4095, …).
    black_level is uniform across all four channels (gn3_black_level).
    """
    sidecar = path.with_suffix('.imgprops')
    meta    = json.loads(sidecar.read_text())
    pattern = _BAYER_ORDER_TO_PATTERN[meta['bayerOrder']]
    bit_depth = int(''.join(c for c in meta['imageType'] if c.isdigit()))
    white   = float((1 << bit_depth) - 1)
    black   = np.full(4, gn3_black_level, dtype=np.float32)
    return pattern, black, white


# --------------------------------------------------------------------------- #
# Frame loading                                                                 #
# --------------------------------------------------------------------------- #

def load_raw(path: Path) -> np.ndarray:
    """Return the raw Bayer data as float32 (ADU, no demosaicing)."""
    with rawpy.imread(str(path)) as raw:
        return raw.raw_image_visible.astype(np.float32)


def load_raw_gn3(path: Path, shape: tuple[int, int]) -> np.ndarray:
    """Load a GN3 .raw file (uint16 LE, no header) and return float32 (H, W)."""
    return np.frombuffer(path.read_bytes(), dtype='<u2').reshape(shape).astype(np.float32)


def load_raw_rgb(path: Path) -> np.ndarray:
    """Demosaic + white-balance a single DNG via rawpy; returns H×W×3 uint8."""
    with rawpy.imread(str(path)) as raw:
        return raw.postprocess(use_camera_wb=True, no_auto_bright=False, output_bps=8)


def load_raw_rgb_gn3(path: Path, shape: tuple[int, int],
                     pattern: np.ndarray, black: np.ndarray,
                     white: float) -> np.ndarray:
    """Load, calibrate, and demosaic a GN3 frame for display; returns H×W×3 uint8."""
    raw   = load_raw_gn3(path, shape)
    cal   = calibrate_frame(raw, pattern, black, white)
    rgb_f = demosaic_to_rgb(cal, pattern)
    return (rgb_f * 255).astype(np.uint8)


# --------------------------------------------------------------------------- #
# Calibration                                                                   #
# --------------------------------------------------------------------------- #

def calibrate_frame(frame: np.ndarray, pattern: np.ndarray,
                    black: np.ndarray, white: float) -> np.ndarray:
    """
    Subtract per-channel black level, normalize by (white − black).
    Input : (H, W) float32 Bayer frame in ADU.
    Output: (H, W) float32, nominally [0, 1], clipped.
    """
    out = np.empty_like(frame, dtype=np.float32)
    for r in range(2):
        for c in range(2):
            ch = int(pattern[r, c])
            bl = black[ch]
            out[r::2, c::2] = (frame[r::2, c::2] - bl) / (white - bl)
    return np.clip(out, 0.0, 1.0)


# --------------------------------------------------------------------------- #
# Demosaicing                                                                   #
# --------------------------------------------------------------------------- #

def _cv2_bayer_code(pattern: np.ndarray) -> int:
    """
    Map rawpy 2×2 Bayer pattern to cv2 VNG demosaic code.
    cv2's BayerXX2RGB codes are named after the pixel one row/col down from
    rawpy's (0, 0) origin (i.e. pattern[1, 1] / pattern[1, 0], the latter by
    2×2 periodicity) -- not pattern[0, 0] / pattern[0, 1] directly. Using the
    top-left pixel instead effectively swaps R and B in the output.
    """
    c1 = int(pattern[1, 1])
    c2 = int(pattern[1, 0])
    if   c1 == 0:                  return cv2.COLOR_BayerRG2RGB_VNG  # BGGR
    elif c1 == 2:                  return cv2.COLOR_BayerBG2RGB_VNG  # RGGB
    elif c1 in (1, 3) and c2 == 0: return cv2.COLOR_BayerGR2RGB_VNG  # GBRG
    else:                          return cv2.COLOR_BayerGB2RGB_VNG  # GRBG


def demosaic_to_rgb(bayer: np.ndarray, pattern: np.ndarray) -> np.ndarray:
    """
    Full-resolution RGB from a 2D float32 calibrated Bayer array.
    Uses cv2 VNG demosaicing if available, otherwise half-res channel extraction.
    Returns float32 H×W×3 in [0, 1], percentile-stretched per channel for display.
    """
    if _HAS_CV2:
        u16   = (np.clip(bayer, 0, 1) * 65535).astype(np.uint16)
        rgb16 = cv2.cvtColor(u16, _cv2_bayer_code(pattern))
        rgb   = rgb16.astype(np.float32) / 65535.0
    else:
        channels = {int(pattern[r, c]): bayer[r::2, c::2]
                    for r in range(2) for c in range(2)}
        R = channels[0].astype(np.float32)
        G = ((channels[1] + channels[3]) / 2).astype(np.float32)
        B = channels[2].astype(np.float32)
        rgb = np.stack([R, G, B], axis=-1)

    for i in range(3):
        lo, hi = np.percentile(rgb[..., i], [0.5, 99.5])
        rgb[..., i] = np.clip((rgb[..., i] - lo) / max(hi - lo, 1e-6), 0, 1)

    # Sensor data is linear in scene light; sRGB display expects gamma-encoded
    # values, or the image looks dark and flat (rawpy's postprocess applies
    # this internally for the DNG path, but this cv2/manual path never did).
    rgb = rgb ** (1.0 / 2.2)
    return rgb


# --------------------------------------------------------------------------- #
# Shared plot helper                                                             #
# --------------------------------------------------------------------------- #

def plot_sample_frames(rgb_frames: list[np.ndarray], labels: list[str],
                       out: Path) -> None:
    n = len(rgb_frames)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]
    for ax, rgb, label in zip(axes, rgb_frames, labels):
        ax.imshow(rgb, aspect="auto")
        ax.set_title(label, fontsize=10)
        ax.axis("off")
    fig.suptitle("Sample frames (RGB)", fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")
