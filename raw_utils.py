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
    from tqdm import tqdm as _tqdm
    _HAS_TQDM = True
except ImportError:
    _HAS_TQDM = False


def progress(iterable, desc: str = "", total: int | None = None):
    """
    Wrap an iterable in a tqdm bar when tqdm is installed, else pass it through.

    Centralised so every script gets the same treatment and none of them hard-
    depend on tqdm: a missing progress bar should never stop an analysis.
    """
    if _HAS_TQDM:
        return _tqdm(iterable, desc=desc, total=total, unit="frame", leave=False)
    return iterable


def git_revision(repo_dir=None) -> dict:
    """
    Describe the working tree that is producing this run.

    The `dirty` flag matters as much as the hash: a commit id alone is
    misleading when there are uncommitted edits, which is the normal state
    while iterating. Returns {} rather than raising when git is unavailable or
    this is not a checkout -- provenance is nice to have, not a reason to fail
    an analysis.
    """
    import subprocess
    repo = Path(repo_dir) if repo_dir else Path(__file__).resolve().parent
    def run(*args):
        return subprocess.run(["git", "-C", str(repo), *args],
                              capture_output=True, text=True, timeout=10)
    try:
        head = run("rev-parse", "HEAD")
        if head.returncode != 0:
            return {}
        commit = head.stdout.strip()
        return {
            "commit":  commit,
            "short":   commit[:12],
            "branch":  run("rev-parse", "--abbrev-ref", "HEAD").stdout.strip(),
            "subject": run("log", "-1", "--pretty=%s").stdout.strip(),
            "dirty":   bool(run("status", "--porcelain").stdout.strip()),
        }
    except Exception:
        return {}


def format_duration(seconds: float) -> str:
    """Human-readable elapsed time, e.g. '1h 04m 12s', '3m 07s', '12.4s'."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    m, s = divmod(int(round(seconds)), 60)
    h, m = divmod(m, 60)
    return f"{h}h {m:02d}m {s:02d}s" if h else f"{m}m {s:02d}s"


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


#: Cubic (Lagrange) weights for estimating a sample from its neighbours at
#: -2, -1, +1, +2 -- the centre is excluded because that is the defect being
#: replaced. Derived by evaluating the Lagrange basis for those four nodes at
#: 0; they sum to 1, so a flat region is reproduced exactly.
_CUBIC_W = (-1.0 / 6.0, 2.0 / 3.0, 2.0 / 3.0, -1.0 / 6.0)


def cubic_fill_plane(plane: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Replace masked samples by cubic interpolation from their neighbours.

    Operates on a single Bayer sub-plane, so neighbours at +-1 and +-2 here are
    +-2 and +-4 in full-frame coordinates -- all the same colour channel.

    The horizontal and vertical cubic estimates are averaged, which is stabler
    than either alone near an edge. Neighbours that are themselves masked will
    contaminate the estimate; that is fine for isolated defects and degrades
    gracefully for small clusters.
    """
    if not mask.any():
        return plane
    H, W = plane.shape
    p = np.pad(plane.astype(np.float32), 2, mode="edge")
    horiz = sum(w * p[2:2 + H, o:o + W]
                for w, o in zip(_CUBIC_W, (0, 1, 3, 4)))
    vert = sum(w * p[o:o + H, 2:2 + W]
               for w, o in zip(_CUBIC_W, (0, 1, 3, 4)))
    return np.where(mask, 0.5 * (horiz + vert), plane).astype(np.float32)


def directional_fill_plane(plane: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Replace masked samples by interpolating ALONG the locally smoothest
    direction, rather than across it.

    A symmetric stencil averages over whatever it covers, so on a text stroke it
    reaches across the edge and pulls in the background. Measured on synthetic
    text (error at defect pixels): a 3x3 median leaves 0.761 on 1-px strokes and
    only 1% of them survive at all; direction-following leaves 0.574 and is also
    the best overall (0.047 vs 0.075).

    Four directions -- horizontal, vertical, both diagonals. Smoothness combines
    the near taps (+-1) with the far ones (+-2), so a direction only wins if it
    is smooth over the whole run, not just symmetric about the defect.
    Neighbours that are themselves masked send their direction to the back, so a
    pair of adjacent defects cannot feed each other; if every direction is
    blocked the sample falls back to the local median.

    The estimate is always the mean of two real neighbours, so unlike a cubic it
    cannot overshoot and ring at a high-contrast edge.

    Note the limit: a defect ON a one-pixel-wide feature is not recoverable from
    its own colour plane at all -- along and across the feature are both locally
    smooth, so the choice is a coin flip. Flagging fewer pixels beats any
    interpolator here.
    """
    if not mask.any():
        return plane
    H, W = plane.shape
    R = 2
    p = np.pad(plane.astype(np.float32), R, mode="edge")
    m = np.pad(mask, R, mode="constant", constant_values=True)
    g = lambda a, dy, dx: a[R + dy:R + dy + H, R + dx:R + dx + W]

    ests, costs = [], []
    for dy, dx in ((0, 1), (1, 0), (1, 1), (1, -1)):
        n1a, n1b = g(p, -dy, -dx), g(p, dy, dx)
        n2a, n2b = g(p, -2 * dy, -2 * dx), g(p, 2 * dy, 2 * dx)
        blocked = g(m, -dy, -dx) | g(m, dy, dx)
        ests.append(0.5 * (n1a + n1b))
        cost = np.abs(n1a - n1b) + 0.5 * (np.abs(n2a - n1a) + np.abs(n2b - n1b))
        costs.append(np.where(blocked, np.inf, cost))

    ests, costs = np.stack(ests), np.stack(costs)
    usable = np.isfinite(costs).any(axis=0)
    best = np.argmin(np.where(np.isfinite(costs), costs, 1e30), axis=0)
    est = np.take_along_axis(ests, best[None], axis=0)[0]

    if not usable.all():                       # every direction blocked
        fallback = _median3(plane.astype(np.float32))
        est = np.where(usable, est, fallback)
    return np.where(mask, est, plane).astype(np.float32)


def _median3(plane: np.ndarray) -> np.ndarray:
    if _HAS_CV2:
        return cv2.medianBlur(plane.astype(np.float32), 3)
    pad = np.pad(plane, 1, mode="edge")
    st = np.stack([pad[i:i + plane.shape[0], j:j + plane.shape[1]]
                   for i in range(3) for j in range(3)])
    return np.median(st, axis=0).astype(np.float32)


def directional_fill_bayer(frame: np.ndarray, pattern: np.ndarray,
                           mask: np.ndarray) -> np.ndarray:
    """Apply directional_fill_plane to each Bayer sub-plane of a full frame."""
    out = frame.astype(np.float32).copy()
    for r in range(2):
        for c in range(2):
            sub = mask[r::2, c::2]
            if sub.any():
                out[r::2, c::2] = directional_fill_plane(
                    np.ascontiguousarray(frame[r::2, c::2], dtype=np.float32), sub)
    return out


def cubic_fill_bayer(frame: np.ndarray, pattern: np.ndarray,
                     mask: np.ndarray) -> np.ndarray:
    """Apply cubic_fill_plane to each Bayer sub-plane of a full frame."""
    out = frame.astype(np.float32).copy()
    for r in range(2):
        for c in range(2):
            sub_mask = mask[r::2, c::2]
            if sub_mask.any():
                out[r::2, c::2] = cubic_fill_plane(
                    np.ascontiguousarray(frame[r::2, c::2], dtype=np.float32),
                    sub_mask)
    return out


def uncalibrate_frame(cal: np.ndarray, pattern: np.ndarray,
                      black: np.ndarray, white: float) -> np.ndarray:
    """
    Inverse of calibrate_frame: calibrated [0, 1] back to raw ADU as uint16.

    Restores the per-channel black pedestal, so the result sits in the same
    numeric space as the frames it came from and can be written to a DNG that
    downstream tools read exactly like an original capture.
    """
    out = np.empty(cal.shape, dtype=np.float64)
    for r in range(2):
        for c in range(2):
            bl = float(black[int(pattern[r, c])])
            out[r::2, c::2] = cal[r::2, c::2] * (white - bl) + bl
    return np.clip(np.rint(out), 0, 65535).astype(np.uint16)


def get_dng_color_matrix(path: Path) -> np.ndarray | None:
    """
    XYZ -> camera-RGB matrix for a DNG's ColorMatrix1 tag.

    This is rawpy's rgb_xyz_matrix (LibRaw's cam_xyz), NOT color_matrix --
    color_matrix goes camera -> sRGB, the opposite direction, and writing it
    into ColorMatrix1 would give downstream converters wrong colour.
    """
    try:
        with rawpy.imread(str(path)) as raw:
            m = np.array(raw.rgb_xyz_matrix[:3, :3], dtype=np.float64)
        return m if np.isfinite(m).all() and np.abs(m).sum() > 0 else None
    except Exception:
        return None


#: Tags that must describe OUR pixel data, so they are never copied from the
#: source. Everything else is passed through verbatim -- a blocklist rather than
#: an allowlist, because colour rendering depends on more tags than is obvious
#: (profiles, tone curves, hue maps, baseline exposure, CFA layout ...) and any
#: one of them missing sends a converter down a different path. Pointer-valued
#: tags are excluded too: their offsets refer to the source file's layout and
#: would dangle in ours.
_DNG_STRUCTURAL_TAGS = frozenset({
    254, 256, 257, 258, 259, 262,          # subfile type, size, depth, photometric
    273, 277, 278, 279, 284,               # strips, samples, planar config
    322, 323, 324, 325,                    # tiles
    330,                                   # SubIFDs (pointer)
    513, 514,                              # JPEG interchange (pointer)
    33422,                                 # CFAPattern -- written from our pattern
    50713, 50714, 50715, 50716, 50717,     # black/white levels and deltas
    50718, 50719, 50720,                   # default scale / crop
    50829, 50830,                          # ActiveArea, MaskedAreas
    34665, 34853,                          # Exif / GPS IFD (pointers)
    37500,                                 # MakerNote (may hold offsets)
})


def read_dng_passthrough_tags(path: Path) -> list:
    """
    Read every non-structural tag from a source DNG as tifffile `extratags`.

    Colour is not carried by ColorMatrix1 alone. Converters look the camera up
    by identity (Make / Model / UniqueCameraModel) against their own profile
    database, and then apply whatever profile tags the file provides -- tone
    curve, hue/sat maps, look table, baseline exposure, CFA layout. Copying a
    hand-picked subset leaves the rest at defaults and renders differently from
    the original, so everything that is not about our pixel geometry is passed
    through untouched.

    Values are re-emitted with their original TIFF type. Anything unreadable is
    skipped rather than guessed at; the whole function degrades to [].
    """
    try:
        import tifffile
    except ImportError:
        return []
    out = []
    try:
        with tifffile.TiffFile(str(path)) as tf:
            for tag in tf.pages[0].tags:
                if tag.code in _DNG_STRUCTURAL_TAGS or tag.value is None:
                    continue
                value = tag.value
                if isinstance(value, str):
                    out.append((tag.code, 's', 0, value, True))
                elif isinstance(value, bytes):
                    out.append((tag.code, int(tag.dtype), tag.count, value, True))
                else:
                    if isinstance(value, (int, float)):
                        value = (value,)
                    try:
                        out.append((tag.code, int(tag.dtype), tag.count,
                                    tuple(np.asarray(value).ravel().tolist()), True))
                    except Exception:
                        continue
    except Exception:
        return []
    return out


# Kept as an alias: the earlier name is still what analyze_gt_sequence imports.
read_dng_color_tags = read_dng_passthrough_tags


def _dng_rationals(values, denom: int = 10000):
    """Flatten floats into the (numerator, denominator) int pairs TIFF wants."""
    out = []
    for v in np.asarray(values, dtype=float).ravel():
        out += [int(round(v * denom)), denom]
    return tuple(out)


def save_dng(adu: np.ndarray, out: Path, pattern: np.ndarray,
             black: np.ndarray, white: float,
             color_matrix: np.ndarray | None = None,
             as_shot_neutral: np.ndarray | None = None,
             model: str = "noise_analysis GT",
             copy_tags: list | None = None) -> None:
    """
    Write a Bayer frame as an uncompressed DNG.

    DNG is a TIFF/EP variant, so this is tifffile plus the required DNG tags --
    chosen over pidng because pidng ships a C extension that has to compile,
    while the GN3 path additionally has no source DNG whose metadata could
    simply be copied.

    CFAPattern uses DNG's colour codes (0=R, 1=G, 2=B), so rawpy's second-green
    marker (3) maps to 1; LibRaw reconstructs the G2 designation itself on read.

    When copy_tags is supplied (from read_dng_color_tags on a source frame),
    the capture's own colour and identity tags are re-emitted verbatim and the
    synthesised colour_matrix / as_shot_neutral are not written. That is the
    only way to get correct colour: converters look the camera up by identity
    against their own profile database, so a file naming an unknown camera
    renders wrong no matter what ColorMatrix1 it carries.

    Verified round-trip through rawpy: pixels bit-exact, and raw_pattern,
    black_level_per_channel and white_level all read back as written.
    """
    try:
        import tifffile
    except ImportError:
        print(f"  SKIP {out.name}: tifffile is required to write DNG "
              f"(pip install tifffile)")
        return

    if adu.dtype != np.uint16:
        adu = np.clip(np.rint(adu), 0, 65535).astype(np.uint16)

    cfa = bytes([{0: 0, 1: 1, 2: 2, 3: 1}[int(pattern[r, c])]
                 for r in range(2) for c in range(2)])
    # BlackLevel with BlackLevelRepeatDim is in SPATIAL row-major order, while
    # rawpy indexes black_level_per_channel by COLOUR (0=R 1=G 2=B 3=G2). Those
    # orders differ whenever the pattern is not literally [[0,1],[2,3]], so the
    # values have to be re-ordered by position or two of them land swapped.
    bl_by_colour = np.asarray(black, dtype=float).ravel()
    bl = np.array([bl_by_colour[int(pattern[r, c])]
                   for r in range(2) for c in range(2)]
                  if bl_by_colour.size >= 4 else bl_by_colour, dtype=float)

    tags = [
        (33421, 'H', 2, (2, 2), True),                  # CFARepeatPatternDim
        (33422, 'B', 4, cfa, True),                     # CFAPattern
        (50706, 'B', 4, (1, 4, 0, 0), True),            # DNGVersion 1.4.0.0
        (50707, 'B', 4, (1, 1, 0, 0), True),            # DNGBackwardVersion
        (50717, 'I', 1, (int(round(white)),), True),    # WhiteLevel
    ]
    # Per-channel black levels need the repeat-dim tag; a single shared value
    # can be written on its own.
    if bl.size >= 4 and not np.allclose(bl[:4], bl[0]):
        tags += [(50713, 'H', 2, (2, 2), True),         # BlackLevelRepeatDim
                 (50714, 'H', 4, tuple(int(round(v)) for v in bl[:4]), True)]
    else:
        tags.append((50714, 'H', 1, (int(round(bl[0])),), True))

    # Drop any copied tag whose code we already emit ourselves. A TIFF IFD must
    # not carry the same tag twice; duplicates are malformed and readers may take
    # either one.
    base_codes = {t[0] for t in tags}
    copy_tags  = [t for t in (copy_tags or ()) if t[0] not in base_codes]
    copied     = {t[0] for t in copy_tags}
    if copy_tags:
        # The capture's own colour, identity and profile tags, verbatim.
        tags += copy_tags
    else:
        tags.append((50708, 's', 0, model, True))   # UniqueCameraModel
    if 50721 not in copied:
        # DNG requires ColorMatrix1. Identity is the honest placeholder when
        # there is no source profile to copy (e.g. GN3 .raw), and the file is
        # then a raw data container rather than a colour-managed image.
        tags.append((50721, '2i', 9,
                     _dng_rationals(color_matrix if color_matrix is not None
                                    else np.eye(3)), True))
    if 50728 not in copied:
        neutral = (np.ones(3) if as_shot_neutral is None
                   else np.asarray(as_shot_neutral))
        # AsShotNeutral is RATIONAL (unsigned), not SRATIONAL.
        tags.append((50728, '2I', 3, _dng_rationals(neutral), True))
    if 50778 not in copied:
        tags.append((50778, 'H', 1, (21,), True))   # CalibrationIlluminant1 D65

    tifffile.imwrite(str(out), adu, photometric='CFA', planarconfig='contig',
                     compression=None, extratags=tags)
    print(f"Saved {out}")


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
