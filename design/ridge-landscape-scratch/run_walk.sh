#!/bin/zsh
# run_walk.sh NET LAYER:CH TAG [OUTROOT]
# -> OUTROOT/ring$TAG, OUTROOT/smooth${TAG}_v1, OUTROOT/smooth${TAG}_v05, run one at a time.
# OUTROOT defaults to the current directory. Never point it at the repo's outputs/.
setopt pipefail
D=${0:A:h}                       # this scratch folder
R=${D:h:h}                       # repo root
O=${4:-$PWD}
export PYTHONPATH=$R:$D
py=$R/.venv/bin/python
q() { grep --line-buffered -vE 'Warning|fftn|run_backward|Yf = |permute|imgs = ' ; }
$py $D/ring.py --net $1 --target $2 --size 1024 --out $O/ring$3 2>&1 | q > $O/ring$3.log || exit 1
$py $D/smooth.py --frames-from $O/ring$3 --net $1 --target $2 --v 1.0 --wmax 0.125 --out $O/smooth$3_v1 2>&1 | q > $O/smooth$3_v1.log || exit 1
$py $D/smooth.py --frames-from $O/ring$3 --net $1 --target $2 --v 0.5 --wmax 0.0625 --out $O/smooth$3_v05 2>&1 | q > $O/smooth$3_v05.log || exit 1
echo "== done"
