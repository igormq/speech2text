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
    *)
        # unknown option
    ;;
esac
done

PT_DATA_DIR=/local/home/common/datasets CUDA_VISIBLE_DEVICES=$SPLIT python train.py configs/ds2-sn-v1-brsdv2.yml -s results-local/alcaim_splits/ds2-sn-brsdv2-finetune-split-$SPLIT-complete --opt-level O1 --fine-tune results-local/ds2-sn-librispeech/models/best-model.pth -r -o "{\"train_dataset.manifest_filepath\": \"\$CODE_DIR/data/alcaim_splits/train-fold-$SPLIT.complete.si-ui.csv\"}" --loss-scale 1.0