#!/bin/bash

PYTHON=~/.conda/envs/py2env/bin/python2.7
SCRIPT=run_nnet.py
MODE=TRAIN

embeddings=("aquaint+wiki" "GoogleNews" )
dropouts=(0 0.1)

for emb in "${embeddings[@]}"; do
  for abcnn in "none" "abcnn1"; do
    for dr in "${dropouts[@]}"; do
        cmd="$PYTHON $SCRIPT $MODE --dropout-rate $dr -e $emb"

        if [[ $abcnn != "none" ]]; then
          cmd+=" -a $abcnn --similarity euclidean"
        fi

        echo "Running: $cmd"
        eval $cmd
    done
  done
done
