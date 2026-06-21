"""
Inspect one GN3 sequence to confirm .raw packing format and metadata layout.
Run:  python inspect_gn3.py
"""

import json
import struct
import numpy as np
from pathlib import Path

SEQ_DIR = Path('/stage/algo-datasets/DB/LME/raw/20250126_GN3v4_train/0240_GN3')

raws = sorted(SEQ_DIR.glob('*.raw'))
print(f"Found {len(raws)} .raw files\n")

if not raws:
    raise SystemExit("No .raw files found")

r = raws[0]
stem = r.stem   # e.g. "0240_GN3_000"

# --------------------------------------------------------------------------- #
# imgprops sidecar
# --------------------------------------------------------------------------- #
print("=== .imgprops sidecar ===")
imgprops_path = r.with_suffix('.imgprops')
if imgprops_path.exists():
    raw_text = imgprops_path.read_text()
    print(f"  Path : {imgprops_path}")
    print(f"  Size : {imgprops_path.stat().st_size} bytes")
    try:
        meta = json.loads(raw_text)
        print(f"  JSON :\n{json.dumps(meta, indent=4)}")
    except json.JSONDecodeError:
        print(f"  (not JSON) raw content: {raw_text!r}")
else:
    print(f"  NOT FOUND at {imgprops_path}")
    meta = {}

# --------------------------------------------------------------------------- #
# Pixel layout
# --------------------------------------------------------------------------- #
print("\n=== Pixel layout ===")
W = meta.get('width', 0)
H = meta.get('height', 0)
itype = meta.get('imageType', '?')
file_size = r.stat().st_size
print(f"  File  : {r.name}  ({file_size:,} bytes  /  {file_size/1024/1024:.2f} MB)")
if W and H:
    print(f"  Meta  : {W}×{H}  imageType={itype}")
    for label, nbytes in [
        ('uint16  (2 B/px)',        W * H * 2),
        ('packed 10-bit (4px/5B)',  W * H * 10 // 8),
        ('packed 12-bit (2px/3B)',  W * H * 12 // 8),
    ]:
        match = '<-- MATCH' if nbytes == file_size else ''
        print(f"  {label:35s}: {nbytes:>12,} bytes  {match}")

# --------------------------------------------------------------------------- #
# Pixel statistics (uint16 LE assumption)
# --------------------------------------------------------------------------- #
print("\n=== Pixel statistics (uint16 LE, no header) ===")
pixels = np.frombuffer(r.read_bytes(), dtype='<u2')
print(f"  Count : {len(pixels):,}")
print(f"  Min   : {pixels.min()}")
print(f"  Max   : {pixels.max()}")
print(f"  Mean  : {pixels.mean():.2f}")
print(f"  Std   : {pixels.std():.2f}")
print(f"  p1    : {np.percentile(pixels, 1):.0f}")
print(f"  p99   : {np.percentile(pixels, 99):.0f}")
print(f"  First 16 values : {pixels[:16].tolist()}")

# Reshape if dimensions known
if W and H and len(pixels) == W * H:
    frame = pixels.reshape(H, W)
    print(f"\n  Reshaped to ({H}, {W}) OK")
    print(f"  Corner top-left 4×4:\n{frame[:4, :4]}")
