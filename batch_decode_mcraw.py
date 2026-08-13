"""
Batch-decode .mcraw files using the motioncam-decoder example binary.

INPUT_PATH may be either a single .mcraw file or a directory, which is searched
recursively for *.mcraw. For each file found the script:
  1. Creates  OUTPUT_ROOT/<stem>/
  2. Runs     DECODER_BIN  <file>  -o <out_subdir>  [--num-frames N]
  3. Counts resulting .dng files as a sanity check
  4. Exits non-zero if any file fails

Edit the CONFIG block below, then run:
    python batch_decode_mcraw.py

All CONFIG constants can also be overridden via CLI flags, e.g.:
    python batch_decode_mcraw.py --input-path /data/clip.mcraw --num-frames 50
    python batch_decode_mcraw.py --input-path /data/captures --output-root /tmp/out

--input-dir is accepted as an alias for --input-path.
"""

import argparse
import json
import os
import struct
import subprocess
import sys
import threading
from pathlib import Path

try:
    from tqdm import tqdm
    _HAS_TQDM = True
except ImportError:                      # decoding often runs on a machine
    _HAS_TQDM = False                    # without the analysis dependencies

    def tqdm(iterable, **_kwargs):       # passthrough so the loop is unchanged
        return iterable


def _write(msg: str) -> None:
    """Print without corrupting the progress bar."""
    if _HAS_TQDM:
        tqdm.write(msg)
    else:
        print(msg)

# ============================================================
# CONFIG — edit paths here
# ============================================================
INPUT_PATH  = "/path/to/mcraw/files"   # a single .mcraw file, or a directory
                                       # searched recursively for *.mcraw
OUTPUT_ROOT = "/path/to/output"        # one subdir per capture created here
DECODER_BIN = "decoder"               # path to motioncam-decoder binary
NUM_FRAMES  = None                     # int to decode only first N frames; None = all
# ============================================================


# --------------------------------------------------------------------------- #
# .mcraw container probing                                                      #
# --------------------------------------------------------------------------- #
#
# Layout, from motioncam-decoder's Container.hpp / Decoder.cpp:
#
#   offset 0 : Header  { uint8 ident[7] = "MOTION ", uint8 version }
#              Item    { uint32 type, uint32 size }   type 3 = METADATA
#              <size bytes of camera-metadata JSON>
#   ...frames and audio...
#   EOF - 24 : Item        { uint32 type, uint32 size }   type 0 = BUFFER_INDEX
#              BufferIndex { uint32 magic, int32 numOffsets, int64 indexDataOffset }
#
# numOffsets is the frame count: Decoder::reindexOffsets builds its frame list
# one-to-one from those offsets, and audio is carried by a separate AUDIO_INDEX
# read afterwards. Verified against MotionCam's published sample file, where the
# footer reports 116 frames and the index region is exactly 116 * 16 bytes.

_MCRAW_IDENT   = b"MOTION "
_MCRAW_MAGIC   = 0x8A905612
_TYPE_BUFFER_INDEX = 0
_TYPE_METADATA     = 3
_ITEM   = struct.Struct("<II")     # type, size
_BUFIDX = struct.Struct("<IiQ")    # magic, numOffsets, indexDataOffset
_FOOTER = _ITEM.size + _BUFIDX.size          # 24


def probe_mcraw(path: Path) -> dict | None:
    """
    Read frame count and camera metadata from a .mcraw without decoding it.

    Returns {'frames': int, 'version': int, 'metadata': dict|None}, or None if
    the file cannot be parsed. Probing is strictly an optimisation -- it only
    drives the progress bar and a post-decode sanity check -- so every failure
    path degrades to None rather than raising and stopping a decode that would
    otherwise have worked.
    """
    try:
        with open(path, "rb") as f:
            head = f.read(8)
            if len(head) < 8 or head[:7] != _MCRAW_IDENT:
                return None
            version = head[7]

            metadata = None
            item = f.read(_ITEM.size)
            if len(item) == _ITEM.size:
                itype, isize = _ITEM.unpack(item)
                if itype == _TYPE_METADATA and 0 < isize <= (8 << 20):
                    try:
                        metadata = json.loads(f.read(isize).decode("utf-8", "replace"))
                    except ValueError:
                        metadata = None

            if os.fstat(f.fileno()).st_size < _FOOTER:
                return None
            f.seek(-_FOOTER, os.SEEK_END)
            tail = f.read(_FOOTER)
            if len(tail) < _FOOTER:
                return None
            itype, _ = _ITEM.unpack(tail[:_ITEM.size])
            magic, n_offsets, _ = _BUFIDX.unpack(tail[_ITEM.size:])
            if itype != _TYPE_BUFFER_INDEX or magic != _MCRAW_MAGIC or n_offsets < 0:
                return None
            return {"frames": n_offsets, "version": version, "metadata": metadata}
    except (OSError, struct.error):
        return None


def _watch_frames(out_dir: Path, total: int, stop_evt: threading.Event,
                  poll: float = 0.25) -> None:
    """
    Drive an inner progress bar by counting .dng files as the decoder writes
    them. The decoder reports no progress of its own, so the output directory
    is the only observable.
    """
    with tqdm(total=total, desc="   frames", unit="frame",
              leave=False, position=1) as bar:
        seen = 0
        while True:
            try:
                n = sum(1 for e in os.scandir(out_dir)
                        if e.name.lower().endswith(".dng"))
            except OSError:
                n = seen
            n = min(n, total)
            if n > seen:
                bar.update(n - seen)
                seen = n
            if stop_evt.is_set():
                return
            stop_evt.wait(poll)


def resolve_inputs(path: Path) -> list[Path]:
    """
    Return the .mcraw files to decode.

    A file resolves to itself, a directory to every *.mcraw beneath it. The
    suffix is checked case-insensitively so an explicitly named .MCRAW is not
    rejected, while directory globbing covers both cases too.
    """
    if path.is_file():
        if path.suffix.lower() != ".mcraw":
            sys.exit(f"ERROR: not a .mcraw file: {path}")
        return [path]
    if path.is_dir():
        return sorted(set(path.rglob("*.mcraw")) | set(path.rglob("*.MCRAW")))
    sys.exit(f"ERROR: input path does not exist: {path}")


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
    return s


def _apply_cli_overrides() -> None:
    g = globals()
    scalar = (bool, int, float, str, type(None))
    keys = sorted(k for k in g if k.isupper() and not k.startswith("_") and isinstance(g[k], scalar))

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
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

    # Back-compat: INPUT_DIR was renamed to INPUT_PATH when single files became
    # valid input. Same dest, so either spelling sets the same value.
    parser.add_argument("--input-dir", dest="INPUT_PATH", type=_none_or_auto,
                        default=None, metavar="S",
                        help="alias for --input-path")

    args = parser.parse_args()
    for key, new_val in vars(args).items():
        if new_val is not None:
            g[key] = new_val


# --------------------------------------------------------------------------- #
# Decode one file                                                               #
# --------------------------------------------------------------------------- #

def decode_one(mcraw: Path, out_dir: Path, expected_frames: int | None = None) -> int:
    """
    Run DECODER_BIN on a single .mcraw file.
    Returns the number of .dng files written to out_dir.
    Raises RuntimeError on non-zero exit.

    When expected_frames is known (from probe_mcraw), a watcher thread counts
    output .dng files to drive a per-frame progress bar alongside the decode.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [DECODER_BIN, str(mcraw), "-o", str(out_dir)]
    if NUM_FRAMES is not None:
        cmd += ["--num-frames", str(NUM_FRAMES)]

    stop_evt = watcher = None
    if expected_frames and _HAS_TQDM:
        stop_evt = threading.Event()
        watcher  = threading.Thread(target=_watch_frames,
                                    args=(out_dir, expected_frames, stop_evt),
                                    daemon=True)
        watcher.start()
    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    finally:
        if stop_evt is not None:
            stop_evt.set()
            watcher.join(timeout=2.0)

    if result.stderr.strip():
        for line in result.stderr.strip().splitlines():
            _write(f"    [stderr] {line}")

    if result.returncode != 0:
        raise RuntimeError(
            f"Decoder exited with code {result.returncode}\n"
            f"  cmd : {' '.join(cmd)}\n"
            f"  stderr: {result.stderr.strip()}"
        )

    dng_count = len(list(out_dir.glob("*.dng"))) + len(list(out_dir.glob("*.DNG")))
    return dng_count


# --------------------------------------------------------------------------- #
# Main                                                                          #
# --------------------------------------------------------------------------- #

def main() -> None:
    _apply_cli_overrides()

    input_path  = Path(INPUT_PATH)
    output_root = Path(OUTPUT_ROOT)

    mcraw_files = resolve_inputs(input_path)
    if not mcraw_files:
        sys.exit(f"No .mcraw files found under {input_path}")

    if input_path.is_file():
        print(f"Decoding single file: {input_path}")
    else:
        print(f"Found {len(mcraw_files)} .mcraw file(s) under {input_path}")
    print(f"Output root : {output_root}")
    print(f"Decoder     : {DECODER_BIN}")
    if NUM_FRAMES is not None:
        print(f"Frame cap   : {NUM_FRAMES}")
    print()

    output_root.mkdir(parents=True, exist_ok=True)

    failures = []
    total_dngs = 0

    # The bar carries the running count, so per-file lines only report outcomes.
    bar = tqdm(mcraw_files, desc="decoding", unit="file",
               disable=not _HAS_TQDM, position=0)
    for mcraw in bar:
        if _HAS_TQDM:
            bar.set_postfix_str(mcraw.name[:40])
        out_dir = output_root / mcraw.stem
        size_mb = mcraw.stat().st_size / 1024 / 1024

        info     = probe_mcraw(mcraw)
        n_frames = info["frames"] if info else None
        expected = n_frames
        if expected is not None and NUM_FRAMES is not None:
            expected = min(expected, NUM_FRAMES)

        try:
            n_dng = decode_one(mcraw, out_dir, expected)
            total_dngs += n_dng
            _write(f"  ✓ {mcraw.name}  ({size_mb:.1f} MB)  →  {out_dir}"
                   f"  [{n_dng} DNG]")
            if n_dng == 0:
                _write(f"  WARNING: decoder succeeded but no .dng files "
                       f"found in {out_dir}")
            elif expected is not None and n_dng != expected:
                # The container's own index says how many frames it holds, so a
                # short decode is detectable here rather than downstream.
                _write(f"  WARNING: {mcraw.name} contains {expected} frame(s) "
                       f"but {n_dng} DNG(s) were written")
        except RuntimeError as exc:
            _write(f"  FAILED: {mcraw.name}: {exc}")
            failures.append(mcraw)

    print()
    print(f"Done.  {len(mcraw_files) - len(failures)}/{len(mcraw_files)} succeeded"
          f"  |  {total_dngs} total DNG(s) written")

    if failures:
        print(f"\nFailed files ({len(failures)}):")
        for f in failures:
            print(f"  {f}")
        sys.exit(1)


if __name__ == "__main__":
    main()
