"""
Inspect one GN3 sequence to determine .raw packing format and metadata layout.
Run:  python inspect_gn3.py
"""

import json
import struct
from pathlib import Path

SEQ_DIR = Path('/stage/algo-datasets/DB/LME/raw/20250126_GN3v4_train/0240_GN3')

# --------------------------------------------------------------------------- #
# 1. imgprops
# --------------------------------------------------------------------------- #
print("=== .imgprops ===")
imgprops_dir = SEQ_DIR / '.imgprops'
if imgprops_dir.exists():
    for f in sorted(imgprops_dir.iterdir()):
        print(f"  {f.name}  ({f.stat().st_size} bytes)")
        if f.suffix.lower() == '.json':
            print(json.dumps(json.loads(f.read_text()), indent=4))
        elif f.stat().st_size < 512:
            print(f.read_bytes())
else:
    print("  .imgprops not found — checking for other metadata files:")
    for f in sorted(SEQ_DIR.iterdir())[:10]:
        print(f"  {f.name}  ({f.stat().st_size} bytes)")

# --------------------------------------------------------------------------- #
# 2. .raw files
# --------------------------------------------------------------------------- #
print("\n=== .raw files ===")
raws = sorted(SEQ_DIR.glob('*.raw'))
print(f"  {len(raws)} .raw files found")

if raws:
    r = raws[0]
    size = r.stat().st_size
    print(f"\n  First file : {r.name}")
    print(f"  File size  : {size} bytes  ({size/1024/1024:.2f} MB)")

    # Try to infer packing from metadata if available
    meta = {}
    for jf in imgprops_dir.glob('*.json') if imgprops_dir.exists() else []:
        meta = json.loads(jf.read_text())
        break

    W = meta.get('width', 0)
    H = meta.get('height', 0)
    itype = meta.get('imageType', '')

    if W and H:
        print(f"\n  Metadata   : {W}×{H}  imageType={itype}")
        print(f"  Expected sizes for {W}×{H}:")
        print(f"    uint16  (2 bytes/px)        : {W*H*2:>12,} bytes")
        print(f"    packed 10-bit (4px/5B)      : {W*H*10//8:>12,} bytes")
        print(f"    packed 12-bit (2px/3B)      : {W*H*12//8:>12,} bytes")
        print(f"    packed 14-bit (4px/7B)      : {W*H*14//8:>12,} bytes")
        print(f"  Actual size                   : {size:>12,} bytes")

        # Identify match
        candidates = {
            'uint16':       W * H * 2,
            'packed 10-bit': W * H * 10 // 8,
            'packed 12-bit': W * H * 12 // 8,
            'packed 14-bit': W * H * 14 // 8,
        }
        matches = [name for name, expected in candidates.items() if expected == size]
        print(f"  --> Matches: {matches if matches else 'none (might have a file header)'}")

    # Print first 32 bytes as hex
    raw_bytes = r.read_bytes()[:32]
    hex_str = ' '.join(f'{b:02x}' for b in raw_bytes)
    print(f"\n  First 32 bytes (hex): {hex_str}")

    # If uint16 LE: show first 16 pixel values
    import struct
    u16 = struct.unpack_from(f'<16H', r.read_bytes()[:32])
    print(f"  As uint16 LE (first 16 values): {list(u16)}")
