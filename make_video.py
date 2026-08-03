#!/usr/bin/env python3
"""
make_video.py — build a video from a directory of PNG frames.

Uses the ffmpeg binary bundled with imageio-ffmpeg, so no system ffmpeg
install is required. Frames are ordered by natural sort, so frame_2.png
comes before frame_10.png.

Example usage
-------------
python make_video.py demosaiced/ --out sequence.mp4
python make_video.py demosaiced/ --out sequence.mp4 --fps 10 --crf 18
python make_video.py demosaiced/ --pattern "*_cal.png" --scale 1280:-2
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path


def natural_key(path: Path):
    """Sort key that orders embedded numbers numerically (frame_2 < frame_10)."""
    return [int(t) if t.isdigit() else t.lower()
            for t in re.split(r"(\d+)", path.name)]


def find_ffmpeg() -> str:
    """Return the path to the ffmpeg binary bundled with imageio-ffmpeg."""
    try:
        import imageio_ffmpeg
    except ImportError:
        sys.exit("imageio-ffmpeg is required: pip install imageio-ffmpeg")
    return imageio_ffmpeg.get_ffmpeg_exe()


def build_video(frames: list[Path], out: Path, fps: int, crf: int,
                preset: str, scale: str | None, ffmpeg: str) -> None:
    """
    Pipe frames into ffmpeg via a concat list, encoding to H.264.

    A concat demuxer file is used rather than a numbered-sequence glob so
    that arbitrary filenames (and our natural sort order) are preserved
    exactly, without needing the files to be renamed to a strict pattern.
    """
    list_file = out.parent / f".{out.stem}_frames.txt"
    with list_file.open("w") as fh:
        for f in frames:
            # concat demuxer requires escaped single quotes in paths
            escaped = str(f.resolve()).replace("'", r"'\''")
            fh.write(f"file '{escaped}'\n")
            fh.write(f"duration {1 / fps:.6f}\n")
        # repeat the final frame so it isn't dropped by the demuxer
        escaped = str(frames[-1].resolve()).replace("'", r"'\''")
        fh.write(f"file '{escaped}'\n")

    vf = ["format=yuv420p"]
    if scale:
        vf.insert(0, f"scale={scale}")
    # H.264 requires even dimensions; pad up if a frame has an odd size
    vf.insert(0, "pad=ceil(iw/2)*2:ceil(ih/2)*2")

    cmd = [
        ffmpeg, "-y",
        "-f", "concat", "-safe", "0",
        "-i", str(list_file),
        "-vsync", "cfr", "-r", str(fps),
        "-c:v", "libx264",
        "-preset", preset,
        "-crf", str(crf),
        "-vf", ",".join(vf),
        "-movflags", "+faststart",
        str(out),
    ]

    print(f"Encoding {len(frames)} frames → {out}  ({fps} fps, crf {crf})")
    proc = subprocess.run(cmd, capture_output=True, text=True)
    list_file.unlink(missing_ok=True)

    if proc.returncode != 0:
        sys.stderr.write(proc.stderr[-4000:] + "\n")
        sys.exit(f"ffmpeg failed with exit code {proc.returncode}")


def main():
    ap = argparse.ArgumentParser(
        description="Create a video from a directory of PNG frames.")
    ap.add_argument("directory", help="Directory containing PNG frames.")
    ap.add_argument("--out", default=None,
                    help="Output video path (default: <directory>.mp4).")
    ap.add_argument("--pattern", default="*.png",
                    help="Glob pattern for frames (default: *.png).")
    ap.add_argument("--fps", type=int, default=24,
                    help="Frames per second (default: 24).")
    ap.add_argument("--crf", type=int, default=18,
                    help="H.264 quality, 0=lossless, 51=worst (default: 18).")
    ap.add_argument("--preset", default="medium",
                    choices=["ultrafast", "superfast", "veryfast", "faster",
                             "fast", "medium", "slow", "slower", "veryslow"],
                    help="x264 encoding preset (default: medium).")
    ap.add_argument("--scale", default=None, metavar="WxH",
                    help="Rescale, e.g. '1280:-2' to fix width and keep aspect.")
    ap.add_argument("--reverse", action="store_true",
                    help="Reverse frame order.")
    args = ap.parse_args()

    src = Path(args.directory)
    if not src.is_dir():
        sys.exit(f"Not a directory: {src}")

    frames = sorted(src.glob(args.pattern), key=natural_key)
    if not frames:
        sys.exit(f"No files matching '{args.pattern}' in {src}")
    if args.reverse:
        frames.reverse()

    out = Path(args.out) if args.out else src.parent / f"{src.name}.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)

    print(f"Found {len(frames)} frames: {frames[0].name} … {frames[-1].name}")
    build_video(frames, out, args.fps, args.crf, args.preset,
                args.scale, find_ffmpeg())

    size_mb = out.stat().st_size / 1024 ** 2
    print(f"Saved {out}  ({size_mb:.1f} MB, {len(frames) / args.fps:.1f}s)")


if __name__ == "__main__":
    main()
