# flake8: noqa
from .transforms import *
from .datasets import *

dataset_names = {'audio-dataset': AudioDataset}
loader_names = {'audio-loader': AudioDataLoader}
transform_names = {
    'to-spectrogram': ToSpectrogram,
    'to-tensor': ToTensor,
    'to-label': ToLabel
}
