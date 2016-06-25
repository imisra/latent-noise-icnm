#!/usr/bin/bash

EXPDIR="./experiments/baselines"
EXP_PREF="milCOCO_finetune"
GPUID=0;

CAFFEPATH=./caffe-icnm
TOOLS="$CAFFEPATH"/build/tools
export PYTHONPATH=$PYTHONPATH:"$CAFFEPATH/src/caffe/:$CAFFEPATH/python/"
FTMODEL=./experiments/baselines/cache/vgg16_fully_conv.caffemodel"
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
