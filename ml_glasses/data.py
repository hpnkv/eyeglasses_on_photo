import glob
import os

from PIL import Image
from torch.utils.data import Dataset


class MeGlassDataset(Dataset):
    def __init__(self, lines, images_path, transform=None):
        super(MeGlassDataset, self).__init__()

        self.images_path = images_path
        self.filenames = []
        self.labels = []
        self.transform = transform

        for line in lines:
            filename, label = line.split()
            label = int(label)

            self.filenames.append(filename)
            self.labels.append(label)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, item):
        filename = self.filenames[item]
        fp = os.path.join(self.images_path, filename)

        image = Image.open(fp)
        sample = {
            'image': image,
            'label': self.labels[item],
            'filename': filename
        }

        if self.transform is not None:
            sample['image'] = self.transform(sample['image'])

        return sample


class InferenceDataset(Dataset):
    def __init__(self, images_path, transform=None):
        super(InferenceDataset, self).__init__()
        self.images_path = images_path
        self.filenames = glob.glob(os.path.join(images_path, '*'))
        self.transform = transform

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, item):
        filename = self.filenames[item]

        image = Image.open(filename)
        sample = {
            'image': image,
            'filename': filename
        }
        if self.transform is not None:
            sample['image'] = self.transform(sample['image'])

        return sample
