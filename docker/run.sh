#!/bin/bash

CUR_DIR=$(pwd)
PROJ_DIR=$(dirname $CUR_DIR)
DATA_VOLUMES=""

# Usage info
show_help() {
echo "
Usage: ./run.sh [-h]
[--sym]

--sym Indicates symlinks for data directories
"
}

while :; do
    case $1 in
    -h|-\?|--help)
        show_help    # Display a usage synopsis.
        exit
        ;;
    -s|--sym)
        DATA_VOLUMES+="--volume $(readlink -f ../data/kitti/training):/CaDDN/data/kitti/training "
        DATA_VOLUMES+="--volume $(readlink -f ../data/kitti/testing):/CaDDN/data/kitti/testing "
        ;;
    -?*)
        printf 'WARN: Unknown option (ignored): %s\n' "$1" >&2
        ;;
    *)               # Default case: No more options, so break out of the loop.
        break
    esac

    shift
done

PCDET_VOLUMES=""
for entry in $PROJ_DIR/pcdet/*
do
    name=$(basename $entry)

    if [ "$name" != "version.py" ] && [ "$name" != "ops" ]
    then
        PCDET_VOLUMES+="--volume $entry:/CaDDN/pcdet/$name "
    fi
done


CMD="docker run -it \
    --runtime=nvidia \
    --net=host \
    --privileged=true \
    --ipc=host \
    --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
    --volume="$XAUTHORITY:/root/.Xauthority:rw" \
    --env="DISPLAY" \
    --env="QT_X11_NO_MITSHM=1" \
    --hostname="inside-DOCKER" \
    --name="CaDDN" \
    --volume $PROJ_DIR/checkpoints:/CaDDN/checkpoints \
    --volume $PROJ_DIR/data:/CaDDN/data \
    --volume $PROJ_DIR/output:/CaDDN/output \
    --volume $PROJ_DIR/tools:/CaDDN/tools \
    --volume $PROJ_DIR/.git:/CaDDN/.git \
    $DATA_VOLUMES \
    $PCDET_VOLUMES \
    --rm \
    caddn bash
"
echo $CMD
eval $CMD