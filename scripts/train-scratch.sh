#!/bin/bash
set -x
set -e


for i in "$@"
do
case $i in
    -s=*|--split=*)
    SPLIT="${i#*=}"
    shift # past argument=value
    ;;
    -c=*|--cuda=*)
    CUDA="${i#*=}"
    shift # past argument=value
    ;;
    *)
        # unknown option
    ;;
esac
done

CUDA_VISIBLE_DEVICES=$CUDA python train.py configs/ds2-sn-v1-brsdv2.yml -s results-local/alcaim_splits/ds2-sn-brsdv2-scratch-split-$SPLIT-complete --opt-level O1 -o "{\"train_dataset.manifest_filepath\": \"\$CODE_DIR/data/alcaim_splits/train-fold-$SPLIT.complete.si-ui.csv\"}" --loss-scale 1.0