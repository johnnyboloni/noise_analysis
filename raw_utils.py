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
    print("WARNING: cv2 not importable -- demosaic_to_rgb() will fall back to "
          "half-resolution nearest-neighbor channel extraction instead of "
          "interpolated (VNG) demosaicing. Install opencv-python(-headless) "
          "for full-resolution, properly interpolated output.", file=sys.stderr)


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

def _cv2_bayer_code(pattern: np.ndarray, suffix: str = '_EA') -> int:
    """
    Map rawpy 2×2 Bayer pattern to a cv2 demosaic code.

    cv2's BayerXX2RGB codes are named after the pixel one row/col down from
    rawpy's (0, 0) origin (i.e. pattern[1, 1] / pattern[1, 0], the latter by
    2×2 periodicity) -- not pattern[0, 0] / pattern[0, 1] directly. Using the
    top-left pixel instead effectively swaps R and B in the output. Verified
    empirically against known ground-truth colors for all four pattern types,
    for '_VNG', '_EA' and '' (bilinear) alike -- the naming convention is the
    same across variants, so only the suffix changes.

    Default is '_EA' (edge-aware): unlike '_VNG' it accepts 16-bit input, which
    matters far more here than the choice of interpolation kernel (see
    demosaic_to_rgb).
    """
    c1 = int(pattern[1, 1])
    c2 = int(pattern[1, 0])
    if   c1 == 0:                  name = 'COLOR_BayerRG2RGB'  # BGGR
    elif c1 == 2:                  name = 'COLOR_BayerBG2RGB'  # RGGB
    elif c1 in (1, 3) and c2 == 0: name = 'COLOR_BayerGR2RGB'  # GBRG
    else:                          name = 'COLOR_BayerGB2RGB'  # GRBG
    return getattr(cv2, name + suffix)


def get_color_metadata(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Return (white_balance[3], color_matrix[3,3]) from a DNG's embedded camera
    profile, for color-correcting demosaic_to_rgb output. white_balance is
    normalized so the green gain is 1. GN3 sequences have no such profile;
    demosaic_to_rgb falls back to a gray-world WB estimate and no CCM.
    """
    with rawpy.imread(str(path)) as raw:
        wb = np.array(raw.camera_whitebalance[:3], dtype=np.float32)
        cm = np.array(raw.color_matrix[:3, :3], dtype=np.float32)
    return wb / wb[1], cm


def demosaic_to_rgb(bayer: np.ndarray, pattern: np.ndarray,
                    wb: np.ndarray | None = None,
                    ccm: np.ndarray | None = None) -> np.ndarray:
    """
    Full-resolution RGB from a 2D float32 calibrated Bayer array.

    White-balances at the Bayer stage (using `wb`, or a gray-world estimate
    if not given -- a per-channel *global percentile stretch* was tried
    previously and is NOT equivalent to white balance: it content-adaptively
    remaps each channel's own min/max to [0, 1], which crushes channels with
    a narrower range in any given scene to 0/1 and badly distorts color;
    verified against rawpy.postprocess() on a synthetic ground-truth DNG,
    e.g. a neutral gray patch came out as [0, 0, 81] instead of ~[113,109,106]).

    Demosaics with cv2 VNG if available (else half-res channel extraction),
    optionally applies a camera color-correction matrix (`ccm`, camera RGB ->
    output RGB, e.g. from get_color_metadata), then gamma-encodes.
    Returns float32 H×W×3 in [0, 1].
    """
    if wb is None:
        means = {}
        for r in range(2):
            for c in range(2):
                ch = int(pattern[r, c])
                means.setdefault(ch, []).append(float(bayer[r::2, c::2].mean()))
        means = {ch: np.mean(v) for ch, v in means.items()}
        g_mean = means[1]
        wb = np.array([g_mean / max(means[0], 1e-6),
                        1.0,
                        g_mean / max(means[2], 1e-6)], dtype=np.float32)

    gain_by_idx = {0: wb[0], 1: wb[1], 2: wb[2], 3: wb[1]}
    balanced = np.empty_like(bayer)
    for r in range(2):
        for c in range(2):
            ch = int(pattern[r, c])
            balanced[r::2, c::2] = np.clip(bayer[r::2, c::2] * gain_by_idx[ch], 0, 1)

    if _HAS_CV2:
        # Demosaic at 16 bits. The '_VNG' variant is 8-bit only (it asserts
        # depth == CV_8U), so this uses '_EA' (edge-aware), which accepts
        # CV_16U and is verified to follow the same pattern-code naming
        # convention -- see _cv2_bayer_code.
        #
        # Bit depth matters much more than the interpolation kernel here. This
        # is linear sensor data that gets gamma-encoded on the way out, and
        # gamma expands the shadows where a lowlight scene lives: one 8-bit
        # linear LSB (1/255) lands at (1/255)**(1/2.2) ~= 20 display levels
        # after encoding, so an 8-bit linear buffer bands visibly in exactly
        # the tones this pipeline cares about. At 16 bits the first LSB is
        # ~1.6 display levels instead.
        #
        # Stretch into the full range BEFORE quantizing, then undo it right
        # after. Calibrated lowlight data occupies only the bottom percent or
        # so of [0, 1]; quantizing at that native scale throws away the sub-LSB
        # precision that averaging hundreds of frames just bought, which the
        # auto-brighten below then amplifies back up as visible stepping.
        # Undoing the stretch keeps the CCM and auto-brighten operating on
        # linear scene-referred values (the CCM's clip does not commute with a
        # scale factor, so this has to be restored before it, not after).
        pre = float(np.percentile(balanced, 99.9))
        pre = pre if pre > 1e-6 else 1.0
        u16   = (np.clip(balanced / pre, 0, 1) * 65535).astype(np.uint16)
        rgb16 = cv2.cvtColor(u16, _cv2_bayer_code(pattern))
        rgb   = rgb16.astype(np.float32) / 65535.0 * pre
    else:
        channels = {int(pattern[r, c]): balanced[r::2, c::2]
                    for r in range(2) for c in range(2)}
        R = channels[0].astype(np.float32)
        G = ((channels[1] + channels[3]) / 2).astype(np.float32)
        B = channels[2].astype(np.float32)
        half_res = np.stack([R, G, B], axis=-1)
        # Upsample back to full Bayer resolution (nearest-neighbor) so output
        # dimensions match the cv2 path regardless of which one ran -- this
        # branch only samples one position per 2x2 tile, so it's half
        # resolution in both dimensions before this repeat.
        rgb = half_res.repeat(2, axis=0).repeat(2, axis=1)[:bayer.shape[0], :bayer.shape[1]]

    if ccm is not None:
        rgb = np.clip(rgb @ ccm.T, 0, 1)

    # Auto-brighten to match rawpy.postprocess()'s default (no_auto_bright=
    # False, used for the per-frame sample previews): without an adaptive
    # exposure boost, linear sensor data -- especially from a lowlight
    # sequence -- renders much darker here than in those previews even
    # though both are otherwise the same WB+CCM+gamma pipeline. Scale so the
    # 99th-percentile pixel lands near displayable white, leaving headroom.
    hi = float(np.percentile(rgb, 99))
    if hi > 1e-6:
        rgb = np.clip(rgb / hi * 0.92, 0, 1)

    # Sensor data is linear in scene light; sRGB display expects gamma-encoded
    # values, or the image looks dark and flat (rawpy's postprocess applies
    # this internally for the DNG path, but this cv2/manual path never did).
    return rgb ** (1.0 / 2.2)


def highpass_std(frame: np.ndarray, pattern: np.ndarray, ksize: int = 5) -> float:
    """
    Std of the high-pass residual (frame minus a local box mean), averaged over
    the four Bayer sub-planes.

    Isolates pixel-to-pixel noise from scene content: real scene structure is
    mostly low-frequency and is removed by the subtraction, while noise is
    broadband and survives. A plain std over the whole frame would instead be
    dominated by the scene's own contrast and stay ~flat no matter how many
    frames were averaged.

    Filtering happens per Bayer sub-plane, never across the raw mosaic --
    neighbouring mosaic pixels are different colour channels, so a box filter
    applied directly to the mosaic would read the R/G/B offsets as huge
    "high-frequency" content that has nothing to do with noise.
    """
    resid_stds = []
    for r in range(2):
        for c in range(2):
            plane = np.ascontiguousarray(frame[r::2, c::2], dtype=np.float32)
            if _HAS_CV2:
                smooth = cv2.blur(plane, (ksize, ksize))
                resid_stds.append(float((plane - smooth).std()))
            else:
                from numpy.lib.stride_tricks import sliding_window_view
                pad = ksize // 2
                smooth = sliding_window_view(plane, (ksize, ksize)).mean(axis=(-1, -2))
                resid_stds.append(float((plane[pad:-pad, pad:-pad] - smooth).std()))
    return float(np.mean(resid_stds))


def bayer_plane_median3(frame: np.ndarray, pattern: np.ndarray) -> np.ndarray:
    """
    3×3 median of each Bayer sub-plane, reassembled into a full-size frame.

    Per sub-plane, never across the raw mosaic: a median taken over mosaic
    neighbours mixes colour channels, so on a coloured scene every pixel looks
    like an outlier against its differently-coloured surroundings.

    Used both to find defects (a pixel far from its same-colour neighbours) and
    to repair them (replace it with their median).
    """
    out = np.empty_like(frame, dtype=np.float32)
    for r in range(2):
        for c in range(2):
            plane = np.ascontiguousarray(frame[r::2, c::2], dtype=np.float32)
            if _HAS_CV2:
                out[r::2, c::2] = cv2.medianBlur(plane, 3)
            else:
                pad = np.pad(plane, 1, mode="edge")
                stack = np.stack([pad[i:i + plane.shape[0], j:j + plane.shape[1]]
                                  for i in range(3) for j in range(3)])
                out[r::2, c::2] = np.median(stack, axis=0)
                del stack
    return out


def save_rgb_png(rgb: np.ndarray, out: Path) -> None:
    """
    Save an H×W×3 RGB array (float32 in [0, 1], or uint8) as a PNG at exact
    native resolution, with no resampling -- unlike plot_sample_frames, which
    tiles multiple frames into one matplotlib figure at a fixed dpi and
    shrinks each panel well below native resolution, which can alias a
    demosaiced image into visible grid/moire artifacts that aren't in the
    real data.
    """
    from PIL import Image
    if rgb.dtype != np.uint8:
        rgb = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
    Image.fromarray(rgb).save(str(out))
    print(f"Saved {out}")


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
