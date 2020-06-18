Speech2Text
=================

Implementation of "An open-source end-to-end ASR system for Brazilian Portuguese using DNNs built from newly assembled corpora" by Igor Quintanilha, Luiz Wagner Pereira Biscainho, and Sergio Lima Netto. (submitted).

### Requirements

- pytorch >= 1.0.1
- cudatoolkit >= 9.0
- torchvision
- torchaudio
- ignite
- pyyaml
- wget 
- num2words
- unidecode
- editdistance
- [ctcdecode](https://github.com/igormq/ctcdecode-pytorch/tree/cpp-backend)

### Datasets

All datasets can be found [here](http://igormq.github.io/datasets).

### Acoustic models


|      AM      | Trained on |   Method   |      Test      |                                   Download                                   |
|:------------:|------------|:----------:|:--------------:|:----------------------------------------------------------------------------:|
| DeepSpeech 2 |   BRSD v2  |   Scratch  | 52.55% (2.42%) | [Link](http://www02.smt.ufrj.br/~igor.quintanilha/ds2-brsdv2-scratch.tar.gz) |
| DeepSpeech 2 |   BRSD v2  | Fine-tuned | 47.41% (1.73%) | [Link](http://www02.smt.ufrj.br/~igor.quintanilha/ds2-brsdv2-finetune.tar.gz)|


### Language models

| Language model*                                                                         | RP | Size |          LapsBM |            BRTD |
|-----------------------------------------------------------------------------------------|----|-----:|----------------:|----------------:|
| [word 3-gram](http://www02.smt.ufrj.br/~igor.quintanilha/pt-BR.word.3-gram.binary)      | 25 | 1.9G |          173.79 |          161.29 |
| [word 5-gram](http://www02.smt.ufrj.br/~igor.quintanilha/pt-BR.word.5-gram.binary)      | 42 | 7.8G |          136.50 |          135.12 |
| [char 5-gram](http://www02.smt.ufrj.br/~igor.quintanilha/pt-BR.char.5-gram.binary)      | 5  |  41M |      <=2,334.48 |      <=2,694.51 |
| [char 10-gram](http://www02.smt.ufrj.br/~igor.quintanilha/pt-BR.char.10-gram.binary)    | 10 | 4.7G |       <=271.86$ |       <=323.71$ |
| [char 15-gram*](http://www02.smt.ufrj.br/~igor.quintanilha/pt-BR.char.15-gram.binary)   | 15 | 5.4G |       <=239.59$ |       <=198.49$ |
| [char 20-gram*](http://www02.smt.ufrj.br/~igor.quintanilha/pt-BR.char.20-gram.binary)   | 20 | 8.8G |       <=227.84$ |       <=189.53$ |

*All models were trained using KenLM. More detailed information in the paper.


## Training

Coming soon.

### AM

Coming soon.

### LM

Coming soon.

### CTC beam search decoding with an external LM

Coming soon.

## Evaluating

Coming soon.