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
import subprocess
import sys
from pathlib import Path

# ============================================================
# CONFIG — edit paths here
# ============================================================
INPUT_PATH  = "/path/to/mcraw/files"   # a single .mcraw file, or a directory
                                       # searched recursively for *.mcraw
OUTPUT_ROOT = "/path/to/output"        # one subdir per capture created here
DECODER_BIN = "decoder"               # path to motioncam-decoder binary
NUM_FRAMES  = None                     # int to decode only first N frames; None = all
# ============================================================


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

def decode_one(mcraw: Path, out_dir: Path) -> int:
    """
    Run EXAMPLE_BIN on a single .mcraw file.
    Returns the number of .dng files written to out_dir.
    Raises RuntimeError on non-zero exit.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [DECODER_BIN, str(mcraw), "-o", str(out_dir)]
    if NUM_FRAMES is not None:
        cmd += ["--num-frames", str(NUM_FRAMES)]

    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    if result.stderr.strip():
        for line in result.stderr.strip().splitlines():
            print(f"    [stderr] {line}")

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

    for i, mcraw in enumerate(mcraw_files, 1):
        out_dir = output_root / mcraw.stem
        size_mb = mcraw.stat().st_size / 1024 / 1024
        print(f"[{i}/{len(mcraw_files)}] {mcraw.name}  ({size_mb:.1f} MB)")
        print(f"          → {out_dir}")

        try:
            n_dng = decode_one(mcraw, out_dir)
            total_dngs += n_dng
            print(f"          ✓ {n_dng} DNG(s) written")
            if n_dng == 0:
                print(f"  WARNING: decoder succeeded but no .dng files found in {out_dir}")
        except RuntimeError as exc:
            print(f"  FAILED: {exc}")
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
