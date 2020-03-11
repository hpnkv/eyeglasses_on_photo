import random

import torch
import torchvision
from PIL import Image
from tqdm import tqdm

from ml_glasses.commons import FaceAlignTransform

CELEBA_EYEGLASSES_SEED = 5


class ConvertPilJpegFileToImage:
    def __call__(self, image):
        image.load()
        return image._new(image.im)


def main():
    torch.manual_seed(CELEBA_EYEGLASSES_SEED)
    random.seed(CELEBA_EYEGLASSES_SEED)

    for split in ('train', 'valid'):
        celeba = torchvision.datasets.CelebA('.', split=split, transform=ConvertPilJpegFileToImage())
        aligner = FaceAlignTransform()

        print()
        total_eyeglasses = 0
        for _, attrs in tqdm(celeba):
            if attrs[15] == 1:
                total_eyeglasses += 1

        no_eyeglass_left = total_eyeglasses  # mathes the no. of images with eyeglasses

        idxs = list(range(len(celeba)))
        random.shuffle(idxs)  # to pick random no-eyeglasses samples

        meta_file = open(f'meta_celeba_{split}.txt', 'w')

        errors = 0

        for idx in tqdm(idxs):
            image, attrs = celeba[idx]

            has_eyeglasses = attrs[15] == 1

            try:
                if has_eyeglasses or no_eyeglass_left > 0:
                    label = 1 if has_eyeglasses else 0
                    image = aligner(image)

                    filename = f'celeba_eyeglasses_{split}/{idx}_{label}.png'
                    Image.fromarray(image).save(filename)
                    meta_file.write(f'{idx}_{label}.png {label}\n')

                    if label == 0 and no_eyeglass_left > 0:
                        Image.fromarray(image).save(filename)
                        no_eyeglass_left -= 1

            except IndexError:
                errors += 1

        meta_file.close()


if __name__ == '__main__':
    main()
