#!/bin/bash
docker run --rm -it \
           --gpus all \
           -e DISPLAY=$DISPLAY \
           -v /tmp/.X11-unix:/tmp/.X11-unix \
           -e TF_FORCE_GPU_ALLOW_GROWTH=true \
           -v $(realpath ~/datasets):/datasets \
           -v $(realpath .):/root/OverlapNet \
           $@ \
           overlapnet
