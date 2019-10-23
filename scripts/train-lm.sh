#!/bin/bash
set -x
set -e

LM_DIR = "/home/igor.quintanilha/repos/ctcdecode-pytorch"
AM_DIR = "/home/igor.quintanilha/repos/asr-study-pytorch"

RESULTS = ${AM_DIR}/results-local/alcaim_splits/

MODELS = ('finetune-all', 'finetune-extended', 'finetune-split-0', 'finetune-split-0-complete', 'finetune-split-1', 'finetune-split-1-complete', 'finetune-split-2', 'finetune-split-2-complete')

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

python ${LM_DIR}/tune.py ../asr-study-pytorch/results-local/ds2-sn-brsdv2-finetune/logits/lapsbm.val.pth 
--vocab-file pt-BR.vocab.txt --lm-path ../aes-lac-2018/LaPSLM/3gram.binary --lm-trie-path ../asr-stud
y-pytorch/data/text-corpora/lapslm.3-gram.trie  --beam-size 100 --output-file brsdv2.finetune.lapslm.3-gram.b100.csv --lm-unit word

python tune.py ../asr-study-pytorch/results-local/ds2-sn-brsdv2-finetune/logits/lapsbm.val.pth 
--vocab-file pt-BR.vocab.txt --lm-path ../asr-study-pytorch/data/text-corpora/pt-BR.word.3-gram.binar
y --lm-trie-path ../asr-study-pytorch/data/text-corpora/pt-BR.word.3-gram.trie  --beam-size 100 --output-file brsdv2.finetune.pt-BR.word.3-gram.b100.csv --lm-unit word

python tune.py ../asr-study-pytorch/results-local/ds2-sn-brsdv2-finetune/logits/lapsbm.val.pth 
--vocab-file pt-BR.vocab.txt --lm-path ../asr-study-pytorch/data/text-corpora/pt-BR.word.5-gram.binar
y --lm-trie-path ../asr-study-pytorch/data/text-corpora/pt-BR.word.5-gram.trie  --beam-size 100 --output-file brsdv2.finetune.pt-BR.word.5-gram.b100.csv --lm-unit word

python tune.py ../asr-study-pytorch/results-local/ds2-sn-brsdv2-finetune/logits/lapsbm.val.pth 
--vocab-file pt-BR.vocab.txt --lm-path ../asr-study-pytorch/data/text-corpora/pt-BR.char.5-gram.binar
y --beam-size 100 --output-file brsdv2.finetune.pt-BR.char.5-gram.b100.csv --lm-unit char

python tune.py ../asr-study-pytorch/results-local/ds2-sn-brsdv1-finetune/logits/lapsbm.val.pth 
--vocab-file pt-BR.vocab.txt --lm-path ../asr-study-pytorch/data/text-corpora/pt-BR.char.10-gram.bina
ry --beam-size 100 --output-file brsdv1.finetune.pt-BR.char.10-gram.b100.csv --lm-unit char --lm-workers 1 --num-workers 10

python tune.py ../asr-study-pytorch/results-local/ds2-sn-brsdv2-finetune/logits/lapsbm.val.pth 
--vocab-file pt-BR.vocab.txt --lm-path ../asr-study-pytorch/data/text-corpora/pt-BR.char.15-gram.bina
ry --beam-size 100 --output-file brsdv2.finetune.pt-BR.char.15-gram.b100.csv --lm-unit char

python tune.py ../asr-study-pytorch/results-local/ds2-sn-brsdv2-finetune/logits/lapsbm.val.pth 
--vocab-file pt-BR.vocab.txt --lm-path ../asr-study-pytorch/data/text-corpora/pt-BR.char.20-gram.bina
ry --beam-size 100 --output-file brsdv2.finetune.pt-BR.char.20-gram.b100.csv --lm-unit char --lm-work
ers 10