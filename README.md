# Eyeglasses presence classifier test

This repository contains a pipeline to find images of faces of people wearing eyeglasses. It consists of a small CNN working on top of a face detector and face landmark predictor from `dlib`. It uses two datasets, [MeGlass](https://github.com/cleardusk/MeGlass) and a custom slice from [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html), for training and achieves a validation accuracy of 98.62% and 99.7% respectively.

If all dependencies are built with CUDA support, the inference time is under 10 ms per image, and is usually around 3–4 ms.

## Instructions

### Install CMake

This project uses `dlib`, which uses CMake to build and needs it installed in the system. On Ubuntu, it is as simple as:
```
sudo apt install cmake
```
(_Optional_) If dlib's build finds CUDA Toolkit during compilation, it will use CUDA. The easiest way to install the CUDA Toolkit is probably with `conda`:
```
conda install -c conda-forge cudatoolkit-dev
```

### Install Python packages

```
pip install -r requirements.txt
```

### Acquire datasets and premade face detectors/shape predictors



### Use the classifier

```
python infer.py -i <Directory with input images> -d <Device as in torch.to(), default is 'cpu'>
```

## Pipeline

### Inference

- Input image is scaled to 256 pixels by the smaller side, conserving the aspect ratio
- Face is detected on the photo using `dlib`'s CNN face detector
  - if the detector sees no faces, it takes a square center crop of the image
- Pose is estimated using `dlib`'s implementation of the paper
«One Millisecond Face Alignment with an Ensemble of Regression Trees» by
Vahid Kazemi and Josephine Sullivan, CVPR 2014,
 trained on the iBUG 300-W face landmark dataset
- Face is aligned into a square shape using a face aligner utility from `imutils`
- A convolutional neural network from `ml_glasses/model.py` is applied to get the prediction

Average inference time using `dlib` built with CUDA support is ~3-4 ms per image for all of the stages above.

### Preparation and training

All models were trained for 50 epochs.

## Metrics

| # | trained on | # epochs | acc., CelebA | acc., MeGlass | acc., joint | Comment |
|---|---|---|---|---|---|---|
| 1 | CelebA |  | **97.6%** | 96.02%  |   |   |
| 2 | MeGlass |  |  | 98.89% |   |   |
| 3 | Joint data  | 76 | 98.43% | 99.7% |   | 50% chance HFlip augmentation |
| 4 | Joint data | 47 | 98.62% |  |   | 50% chance HFlip, 10% chance blur |
| 5 | Joint data  | 17 | 98.55% | |   | Augmentations + resblocks |
| 6 |   |   |   |  | |   |
| 7 |   |   |   | |  |   |

## Error analysis

## Improvements

- Even though dlib's face detector uses GPU, it still ships data back to the CPU. This could be avoided by using a fully PyTorch-based (or -compatible) pipeline, reducing inference time and raising GPU utilization during training.
- If we needed an even higher quality, we could try approaches such as knowledge distillation from a bigger network (which can itself be a fine-tuned after a general image classifier), more augmentations such as color jitters.
- Face detector and shape predictor may be suboptimal, maybe there exist better ones.

## Datasets

- [MeGlass, Face Synthesis for Eyeglass-Robust Face Recognition](https://github.com/cleardusk/MeGlass)

Was made primarily to improve the robustness of face identification with glasses on. The dataset only contains black glasses, some of which were inserted artificially, relying on an inferred 3D model of the face.

- [Large-scale CelebFaces Attributes (CelebA)](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)

CelebA dataset contains ~10000 faces with attribute 'Eyeglasses'. All photos are real and contain faces in different rotations with various kinds of glasses. 

This repository contains a script `make_celeba_eyeglasses.py` to produce a dataset of aligned faces from CelebA, including all photos with eyeglasses and (fixed) random photos without eyeglasses to establish a 50/50 ratio.

| Dataset | # of samples | Eyeglasses | No eyeglasses |
|---|---|---|---|
| MeGlass, train | 47,917 | 14,832 | 33,085 |
| MeGlass, test | 6,840 | 3,420 | 3,420 |
| CelebA eyeglasses, train | 20,929 |   |   |
| CelebA eyeglasses, test | 2,752 |   |   |
| Joint, train |   |   |   |
| Joint, test |   |   |   |
|   |   |   |   |

## References