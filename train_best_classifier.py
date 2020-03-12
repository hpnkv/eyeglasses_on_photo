import argparse
import random

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor

from ml_glasses.data import EyeGlassDataset
from ml_glasses.model import GlassesClassifier
from ml_glasses.transforms import RandomGaussianBlur, RandomHorizontalFlip


def validation_accuracy(model, dataloader, device='cuda'):
    model.train(False)
    correct = 0
    total = 0
    for batch in dataloader:
        X_val = batch['image'].to(device)
        y_val = batch['label'].to(device)

        pred = model(X_val)
        _, pred = torch.max(pred.data, 1)
        total += y_val.size(0)
        correct += (pred == y_val).sum().item()

    return correct / total


def train(model, dataloader_train, dataloader_val, n_epochs, optimizer_kwargs, checkpoint_path='./checkpoint.pt',
          device='cuda'):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), **optimizer_kwargs)
    best_epoch_idx = 0
    best_val_accuracy = 0.0

    for epoch in range(n_epochs):
        model.train(True)
        print(f'ep. {epoch} ', end='', flush=True)

        for i, batch in enumerate(dataloader_train):
            X = batch['image'].to(device)
            y = batch['label'].to(device)

            output = model(X)
            loss = criterion(output, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        val_accuracy = validation_accuracy(model, dataloader_val, device=device)
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_epoch_idx = epoch
            torch.save(model.state_dict(), checkpoint_path)

        print(f'val. accuracy: {100 * val_accuracy:.3f}%', flush=True)

    print(f'Best accuracy {100 * best_val_accuracy:.3f}% was achieved at epoch {best_epoch_idx}')


def get_dataloader(meta_file, image_folder, transform=None, **dataloader_kwargs):
    with open(meta_file, 'r') as file:
        lines = file.readlines()

    dataset = EyeGlassDataset(lines, image_folder, transform=transform)

    return DataLoader(dataset, **dataloader_kwargs)


def main(args):
    random.seed(5)
    torch.manual_seed(5)
    np.random.seed(5)

    model = GlassesClassifier()
    train_loader = get_dataloader(args.train_meta, args.train_files,
                                  transform=Compose([RandomHorizontalFlip(), RandomGaussianBlur(), ToTensor()]),
                                  batch_size=64, shuffle=True)
    val_loader = get_dataloader(args.val_meta, args.val_files, transform=ToTensor(),
                                batch_size=64, shuffle=True)

    train(model, train_loader, val_loader, 200, {'lr': 0.002, 'momentum': 0.9},
          checkpoint_path=args.output, device=args.device)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--train-meta', type=str, default='data/meta_joint_train.txt', required=False,
                        help='Path to dataset meta file')
    parser.add_argument('--train-files', type=str, default='data/joint_data/', required=False,
                        help='Path to dataset meta file')

    parser.add_argument('--val-meta', type=str, default='data/meta_celeba_valid.txt', required=False,
                        help='Path to dataset meta file')
    parser.add_argument('--val-files', type=str, default='data/celeba_eyeglasses_valid/', required=False,
                        help='Path to dataset meta file')

    parser.add_argument('-d', '--device', type=str, default='cuda', required=False,
                        help='Which device to run on (default is \'cuda\')')
    parser.add_argument('-o', '--output', type=str, required=False, default='checkpoint.pt',
                        help='Where to save checkpoints')
    parser.add_argument('--shape-predictor', type=str, required=False,
                        help='Path to dlib\'s shape predictor model')
    parser.add_argument('--face-detector', type=str, required=False,
                        help='Path to dlib\'s face detector model')

    args = parser.parse_args()

    main(args)
