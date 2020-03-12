import argparse
import os
from datetime import datetime

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Compose

from ml_glasses.transforms import FaceAlignTransform, ToTensor
from ml_glasses.data import InferenceDataset
from ml_glasses.model import GlassesClassifier


def main(args):
    dataset = InferenceDataset(args.input, transform=Compose([
        FaceAlignTransform(args.face_detector, args.shape_predictor),
        ToTensor()
    ]))

    classifier = GlassesClassifier().to(args.device)
    classifier.load_state_dict(torch.load(args.model, map_location=torch.device(args.device)))
    classifier.train(False)

    if args.time:
        print('Ready for inference')
    inference_start = datetime.now()

    dataloader = DataLoader(dataset, batch_size=1)
    for sample in dataloader:
        image = sample['image'].to(args.device)
        preds = classifier.forward(image)
        _, labels = torch.max(preds.data, 1)
        if labels.item() == 1:
            filename = os.path.split(sample['filename'][0])[-1]
            print(filename)

    if args.time:
        avg_elapsed = (datetime.now() - inference_start).microseconds / 1000 / len(dataset)
        print(
            f'Inference complete in {avg_elapsed:.2f} ms per sample')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-m', '--model', type=str, required=False, default='checkpoint.pt',
                        help='Path to model checkpoint')
    parser.add_argument('-i', '--input', type=str, default='data', required=False,
                        help='Path to input images directory')
    parser.add_argument('-d', '--device', type=str, default='cuda', required=False,
                        help='Which device to run on (default is \'cuda\')')
    parser.add_argument('--shape-predictor', type=str, required=False,
                        help='Path to dlib\'s shape predictor model')
    parser.add_argument('--face-detector', type=str, required=False,
                        help='Path to dlib\'s face detector model')
    parser.add_argument('--time', type=bool, required=False, default=False,
                        help='Compute and print inference time per sample')

    args = parser.parse_args()
    main(args)
