#!/bin/zsh
# encode.sh FRAMEDIR OUTPREFIX   (frames f000.png ... f119.png)
# Needs an ffmpeg binary: set FF, or `uv pip install imageio-ffmpeg` into a scratch venv and point FF at it.
FF=${FF:-ffmpeg}
D=$1; O=$2
# 1024 MP4, true 120 fps, browser-safe 4:2:0
$FF -y -loglevel error -framerate 120 -i $D/f%03d.png -c:v libx264 -preset slow -crf 12 -pix_fmt yuv420p -movflags +faststart ${O}_1024_120fps.mp4
# 512 GIF at 50 fps (2/100 s per frame is the fastest delay browsers honor), one palette for the loop
$FF -y -loglevel error -framerate 50 -i $D/f%03d.png -vf "scale=512:512:flags=lanczos,palettegen=max_colors=256:stats_mode=full" ${O}_pal.png
$FF -y -loglevel error -framerate 50 -i $D/f%03d.png -i ${O}_pal.png -lavfi "scale=512:512:flags=lanczos[x];[x][1:v]paletteuse=dither=sierra2_4a" -loop 0 ${O}_512_50fps.gif
