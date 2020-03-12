# Eyeglasses presence classifier test

This repository contains a pipeline to find images of faces of people wearing eyeglasses. It consists of a small CNN working on top of a face detector and face landmark predictor from `dlib`. It uses two datasets, [MeGlass](https://github.com/cleardusk/MeGlass) and a custom slice from [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html), for training and achieves a validation accuracy of 99.7% and 98.62% respectively.

If all dependencies are built with CUDA support, the inference time inside classification stage is usually around 20 ms. The weights of the model are stored in under 100 KB.

## Getting started

### Installation

This classifier requires a set of dependencies. To install them in the cleanest way possible, I suggest using the installation scripts I provided:
```
# If conda is not installed
chmod +x ./install_miniconda3.sh
./install_miniconda3.sh


# This creates a conda environment 'eyeglasses' and installs all dependencies
# inside (assuming CUDA 10.0 is supported). If conda doesn't run, you should
# correct the path $HOME/miniconda3/etc/profile.d/conda.sh in the script
# to match your actual conda location.

chmod +x ./bootstrap_conda_env.sh
./bootstrap_conda_env.sh
conda activate eyeglasses

# Download dlib models
chmod +x ./download_dlib_models.sh
./download_dlib_models.sh
```

### Inference

```
python infer.py -i <Directory with input images> -d <Device as in torch.to(), default is 'cuda'> --face-detector mmod_human_face_detector.dat --shape-predictor shape_predictor_68_face_landmarks.dat
```

### Training

Assuming you have all data in place, just run
```
python train_best_classifier.py
```

## Inference pipeline description

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

## What I did

- Downloaded the MeGlass dataset and wrote an appropriate PyTorch's DataLoader. 
- Made a CNN using four layers of a simple Conv2d + BatchNorm2d + ReLU pack, interleaved with poolings, followed by a fully connected layer with crossentropy loss
- Achieved an accuracy of 98.89% on validation. MeGlass is a quite easy dataset, since it only contains plain black glasses, so I:
- Made an eyeglass dataset off of CelebA by taking all images with eyeglasses and a 1:1 proportion of non-eyeglass images. I cropped and aligned the faces to be of the same size with MeGlass images using `dlib`'s and `imutils`' instruments.
- Tried learning on CelebA, MeGlass separately and jointly. Joint learning helped increase the accuracy.
- Tried augmentations (horizontal flip and Gaussian blur both gave + ~0.1pp to accuracy)
- Tried another architecture of two ResBlocks + necessary convolutions / poolings: didn’t help much.
- Wrote a data loader for inference data (which crops/aligns from images of any size on the fly)
- Wrote these docs, train and inference script, installation instructions

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

## Improvement ideas

- Overall pipeline
  - Face detector and shape predictor may be suboptimal, there may exist faster/better ones.
  - `libjpeg-turbo` and `Pillow-SIMD` for faster JPEG decoding and image processing on the CPU to improve overall time.
  - Even though dlib's face detector uses GPU, it still ships data back to the CPU. This could be avoided by using a fully PyTorch-based (or -compatible) pipeline, reducing inference time and raising GPU utilization during training (if we use a non-prepared dataset).

- Classification stage
  - If we needed an even higher quality, we could try approaches such as **knowledge distillation** from a bigger network (which can itself be a fine-tuned after a general image classifier), more augmentations such as color jitters.
  - Quantization / mixed precision training (especially useful when distilling big networks)
  
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