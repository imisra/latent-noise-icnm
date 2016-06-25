#!/usr/bin/bash

EXPDIR="./experiments/baselines"
EXP_PREF="milLatentNoiseCOCO_finetune"
GPUID=0;

CAFFEPATH=./caffe-icnm
TOOLS="$CAFFEPATH"/build/tools
export PYTHONPATH=$PYTHONPATH:"$CAFFEPATH/src/caffe/:$CAFFEPATH/python/"
#initialize after training baseline model for 2 epochs on COCO
FTMODEL=./experiments/baselines/cache/milCOCO_snapshot_iter_160000.caffemodel"
GLOG_logtostderr=1
CACHEDIR="$EXPDIR/cache"
LOGDIR="$EXPDIR/logs"
SOLVER="$EXPDIR"/"$EXP_PREF"_solver.prototxt

mkdir $CACHEDIR
mkdir $LOGDIR


set -x
set -e

export PYTHONUNBUFFERED="True"

LOG="$LOGDIR"/"$EXP_PREF"_log.txt.`date +'%Y-%m-%d_%H-%M-%S'`

echo $SOLVER
echo $TOOLS
echo $LOG
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

nice -19 $TOOLS/caffe.bin train --solver="$SOLVER" --weights="$FTMODEL" --gpu $GPUID 2>&1 
