# Eyeglasses presence classifier

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

## Metrics

## Error analysis

## Improvements

## Datasets

- [MeGlass, Face Synthesis for Eyeglass-Robust Face Recognition](https://github.com/cleardusk/MeGlass)

Was made primarily to improve the robustness of face identification with glasses on. The dataset only contains black glasses, some of which were inserted artificially, relying on an inferred 3D model of the face.

- [Large-scale CelebFaces Attributes (CelebA)](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)

CelebA dataset contains ~10000 faces with attribute 'Eyeglasses'. All photos are real and contain faces in different rotations with various kinds of glasses. 

This repository contains a script `make_celeba_eyeglasses.py` to produce a dataset of aligned faces from CelebA, including all photos with eyeglasses and (fixed) random photos without eyeglasses to establish a 50/50 ratio.

| Dataset | # of samples | Eyeglasses | No eyeglasses |
|---|---|---|---|
| MeGlass train | 47,917 | 14,832 | 33,085 |
| MeGlass test | 6,840 | 3,420 | 3,420 |
|   |   |   |   |
|   |   |   |   |
|   |   |   |   |
|   |   |   |   |
|   |   |   |   |

## References