#!/bin/bash

PYTHON=~/.conda/envs/py2env/bin/python2.7
SCRIPT=run_nnet.py
MODE=TRAIN-ALL

embeddings=("GoogleNews" "aquaint+wiki")
dropouts=(0 0.5)

for emb in "${embeddings[@]}"; do
  for abcnn in "abcnn1" "none"; do
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
