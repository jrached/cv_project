#!/bin/bash

for i in {1..7}
do
  echo "Running test $i..."
  python3 scene_flow_clean.py \
    --model=models/raft-things.pth \
    --data_path=data/filter_net/new_bags/test${i}/bag_1 \
    --out_path=data/filter_net/new_processed_flow/test${i} \
    --viz_path=data/filter_net/viz/test${i}
done
