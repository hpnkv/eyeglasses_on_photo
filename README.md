# Eyeglasses presence classifier

This repository contains a pipeline to find images of faces of people wearing eyeglasses. It consists of a small CNN in PyTorch working on top of a face detector and face landmark predictor from `dlib`. It uses two datasets, [MeGlass](https://github.com/cleardusk/MeGlass) and a custom slice from [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html), for training and achieves a validation accuracy of 99.7% and 98.62% respectively.

If all dependencies are built with CUDA support, the inference time inside classification stage is usually below 10 ms. The weights of the model are stored in under 100 KB.

## Getting started

### Installation

This classifier requires a set of dependencies. To install them in the cleanest way possible, I suggest using the scripts I provided:
```
# If conda is not installed
chmod +x ./install_miniconda3.sh
./install_miniconda3.sh


# This creates a conda environment 'eyeglasses' and installs all dependencies
# inside (assuming CUDA 10.0 is supported). If conda doesn't run, you should
# correct the path $HOME/miniconda3/etc/profile.d/conda.sh in the script
# to match your actual conda location. These lines are going to be slow
# on cudatoolkit-dev stage

chmod +x ./bootstrap_conda_env.sh
./bootstrap_conda_env.sh
conda activate eyeglasses


# Download dlib models
chmod +x ./download_dlib_models.sh
./download_dlib_models.sh
```

Prepared data can be downloaded from [Google Drive](https://drive.google.com/file/d/1wng_UnUZznxiSDF5xzq-dbiE3G8-ap8c/view?usp=sharing). Unpack the zip file and overwrite the `data` folder. 

### Inference

```
python infer.py -i <Directory with input images> -d <Device as in torch.to(), default is 'cuda'> --face-detector mmod_human_face_detector.dat --shape-predictor shape_predictor_68_face_landmarks.dat
```
or, to see average elapsed time per photo (overall and in classifier):
```
python infer.py --time True -i <Directory with input images> -d <Device as in torch.to(), default is 'cuda'> --face-detector mmod_human_face_detector.dat --shape-predictor shape_predictor_68_face_landmarks.dat
```

### Training

Assuming you have all data in place, just run
```
python train_best_classifier.py
```

### Demo

Don't miss out on `camera_demo.py`! Just open it with Python and it should pop up a window telling if you're wearing eyeglasses. If you have CUDA enabled on your desktop, it will even happen at decent speed.

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

Average inference time using `dlib` built with CUDA support is under 50 ms per image for all of the stages above, excluding decoding.

## What I did

- Downloaded the MeGlass dataset and wrote an appropriate PyTorch's DataLoader. 
- Made a CNN using four layers of a simple Conv2d + BatchNorm2d + ReLU pack, interleaved with poolings, followed by a fully connected layer with crossentropy loss
- Achieved an accuracy of 98.89% on validation. MeGlass is quite an easy dataset, since it only contains plain black glasses, so I:
- Made an eyeglass dataset off of CelebA by taking all images with eyeglasses and a 1:1 proportion of non-eyeglass images. I cropped and aligned the faces to be of the same size with MeGlass images using `dlib`'s and `imutils`' instruments.
- Tried learning on CelebA, MeGlass separately and jointly. Joint learning helped increase the accuracy.
- Tried augmentations (horizontal flip and Gaussian blur both gave + ~0.1pp to accuracy)
- Tried another architecture of two ResBlocks + necessary convolutions / poolings: didn’t help much.
- Wrote a data loader for inference data (which crops/aligns from images of any size on the fly)
- Wrote these docs, train and inference script, installation instructions
- Made a simple demo for a web camera

## Metrics

All models were allowed to train for 200 epochs. Epochs column cites when the best accuracy was attained.

| # | trained on | # epochs | acc., CelebA | acc., MeGlass  | Comment |
|---|---|---|---|---|---|
| 1 | CelebA | 93 | 97.6% | 96.02%  |   |
| 2 | MeGlass | 100 | 76.3% | 98.89% | Evidently, CelebA based training generalizes to MeGlass samples, but not vice versa   |
| 3 | Joint data  | 76 | 98.43% | 99.7% | 50% chance HFlip augmentation |
| 4 | Joint data | 47 | 98.62% | 99.7% | 50% chance HFlip, 10% chance blur |
| 5 | Joint data  | 17 | 98.55% | 99.7% | Augmentations + resblocks |
| 6 | Joint data  | 140 | 98.73% | 99.7% | 50% chance HFlip, 10% chance blur, white balance correction  |
| 7 | Joint data | | | | Deeper architecture |

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
| CelebA eyeglasses, train | 20,929 | ~10,450 | ~10,450 |
| CelebA eyeglasses, test | 2,752 | ~1,375 | ~1,375  |
