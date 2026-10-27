#!/bin/zsh
# run_landscape.sh NET LAYER:CH OUTDIR [STEPS]
# Steps 1-4 of ../ridge-landscape-handoff.md at 256px, one job at a time, under caffeinate.
# STEPS is a comma list from corpus,ring,coverage,width,graph (default: all).
# Never point OUTDIR at the repo's outputs/.
setopt pipefail
D=${0:A:h}; R=${D:h:h}
NET=$1; TGT=$2; O=$3; STEPS=${4:-corpus,ring,coverage,width,graph}
export PYTHONPATH=$R:$D
py=(caffeinate -i ${STRIATE_PY:-$R/.venv/bin/python})   # STRIATE_PY: interpreter override (worktrees have no .venv)
q() { grep --line-buffered -vE 'Warning|warn|fftn|run_backward|Yf = |permute|imgs = |rfft2' ; }
mkdir -p $O
run() { local name=$1; shift; echo "== $name $(date +%H:%M)"; $py "$@" 2>&1 | q > $O/$name.log || { echo "$name failed"; exit 1; }; tail -1 $O/$name.log }
[[ $STEPS == *corpus* ]]   && run corpus   $D/landscape.py corpus --net $NET --target $TGT --n 64 --out $O
[[ $STEPS == *ring* ]]     && run ring     $D/ring.py --net $NET --target $TGT --size 256 --out $O/ring \
                           && run smooth   $D/smooth.py --frames-from $O/ring --net $NET --target $TGT --v 1.0 --wmax 0.125 --out $O/smooth_v1
[[ $STEPS == *coverage* ]] && run coverage $D/landscape.py coverage --corpus $O/corpus.pt --frames $O/ring --tag ring --out $O \
                           && run coverage2 $D/landscape.py coverage --corpus $O/corpus.pt --frames $O/smooth_v1 --tag smooth_v1 --out $O
[[ $STEPS == *width* ]]    && run width    $D/landscape.py width --corpus $O/corpus.pt --out $O
[[ $STEPS == *graph* ]]    && run graph    $D/landscape.py graph --corpus $O/corpus.pt --resume --out $O
echo "== done $(date +%H:%M)"
