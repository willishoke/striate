"""Strips and MP4s of frame folders (f000.png ...).

viz.py strip DIR OUT.jpg [--every 10] [--size 128]
viz.py mp4 DIR OUT.mp4 [--fps 120]      (ffmpeg from $FFMPEG or imageio_ffmpeg)
"""

import argparse, os, shutil, subprocess
import numpy as np
from PIL import Image


def frames(d):
    return [f"{d}/{f}" for f in sorted(os.listdir(d)) if f.startswith("f") and f.endswith(".png")]


def ffmpeg():
    if os.environ.get("FFMPEG"):
        return os.environ["FFMPEG"]
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        return shutil.which("ffmpeg")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["strip", "mp4"])
    ap.add_argument("dir")
    ap.add_argument("out")
    ap.add_argument("--every", type=int, default=10)
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--fps", type=int, default=120)
    ap.add_argument("--scale", type=int, default=0, help="mp4: output side length (0 = native)")
    a = ap.parse_args()
    fs = frames(a.dir)
    if a.cmd == "strip":
        sel = fs[:: a.every]
        out = Image.new("RGB", (len(sel) * a.size, a.size))
        for i, f in enumerate(sel):
            out.paste(
                Image.open(f).convert("RGB").resize((a.size, a.size), Image.LANCZOS),
                (i * a.size, 0),
            )
        out.save(a.out, quality=90)
    else:
        vf = f"scale={a.scale}:{a.scale}:flags=lanczos," if a.scale else ""
        subprocess.run(
            [
                ffmpeg(),
                "-y",
                "-loglevel",
                "error",
                "-framerate",
                str(a.fps),
                "-i",
                f"{a.dir}/f%03d.png",
                "-vf",
                vf + "format=yuv420p",
                "-c:v",
                "libx264",
                "-preset",
                "slow",
                "-crf",
                "14",
                "-movflags",
                "+faststart",
                a.out,
            ],
            check=True,
        )
    print(a.out, os.path.getsize(a.out) // 1024, "KB")


if __name__ == "__main__":
    main()
