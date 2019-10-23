import glob
import logging
import os
import zipfile

import torch

logger = logging.getLogger(__name__)

__all__ = ['AudioDataset', 'AudioDataLoader']


class AudioDataset(torch.utils.data.Dataset):
    def __init__(self,
                 data_dir,
                 manifest_filepath,
                 transforms=None,
                 target_transforms=None):
        super().__init__()

        self.data_dir = data_dir
        self.manifest_filepath = manifest_filepath

        with open(self.manifest_filepath) as f:
            data = f.readlines()

        self.audio_paths, self.transcript_paths, self.durations = zip(
            *[d.strip().split(',') for d in data])
        self.durations = [float(d) for d in self.durations]
        self.data = list(zip(self.audio_paths, self.transcript_paths))

        # Check if file exists in the system, otherwise look for a zipped file
        if zipfile.is_zipfile(self.data_dir):
            self.is_zip = True
            self.zfile = zipfile.ZipFile(self.data_dir)
        else:
            self.is_zip = False
            self.data = [[os.path.join(data_dir, path) for path in x]
                         for x in self.data]

        self.transforms = transforms
        self.target_transforms = target_transforms

    def _check_files(self):
        if self.is_zip:
            files_list = self.zfile.namelist()
        else:
            files_list = glob.glob(os.path.join(self.data_dir, '**', '*'),
                                   recursive=True)

        files_not_found = [
            x for paths in self.data for x in paths if x not in files_list
        ]

        if len(files_not_found):
            raise RuntimeError('Files not found: {}'.format(files_not_found))

    def __getitem__(self, index):
        audio_path, transcript_path = self.data[index]

        if self.is_zip:
            with self.zfile.open(audio_path) as afile:
                input = afile.read()

            with self.zfile.open(transcript_path) as tfile:
                target = tfile.read()
        else:
            input = audio_path
            target = transcript_path

        if self.transforms is not None:
            input = self.transforms(input)

        if self.target_transforms is not None:
            target = self.target_transforms(target)

        return input, target

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        return f'''AudioDataset:
            data_dir={self.data_dir}
            manifest_filepath={self.manifest_filepath}
            transforms={self.transforms}
            target_transforms={self.target_transforms}'''


class AudioDataLoader(torch.utils.data.DataLoader):
    def __init__(self, *args, **kwargs):
        """
        Creates a data loader for AudioDatasets.
        """
        kwargs.setdefault('collate_fn', self._collate_fn)

        super(AudioDataLoader, self).__init__(*args, **kwargs)

    def _collate_fn(self, batch):

        # Sorting batch in decreasing order due to the PackSequence
        batch = sorted(batch,
                       key=lambda sample: sample[0].shape[0],
                       reverse=True)

        sample, targets = zip(*batch)

        sample_lengths = torch.as_tensor([s.shape[0] for s in sample],
                                         dtype=torch.long)
        sample = torch.nn.utils.rnn.pad_sequence(sample,
                                                 batch_first=True).float()

        target_lengths = torch.as_tensor([t.shape[0] for t in targets],
                                         dtype=torch.long)
        targets = torch.cat(targets).type(torch.long)

        return sample, targets, sample_lengths, target_lengths
