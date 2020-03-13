import random

import dlib
import numpy as np
import torch
from PIL import ImageFilter
from imutils import opencv2matplotlib
from imutils.face_utils import FaceAligner
from torchvision.transforms import functional as F


class FaceAlignTransform:
    def __init__(self, detector_model='../mmod_human_face_detector.dat',
                 shape_predictor='../shape_predictor_68_face_landmarks.dat',
                 desired_face_width=120, desired_left_eye=(0.35, 0.5)):
        self.detector = dlib.cnn_face_detection_model_v1(detector_model)
        # it does something during the first call, so I call a dummy detection in __init__
        self.detector(np.zeros((256, 256), dtype=np.uint8), 1)

        self.predictor = dlib.shape_predictor(shape_predictor)
        self.aligner = FaceAligner(self.predictor, desiredFaceWidth=desired_face_width, desiredLeftEye=desired_left_eye)

        n_values = desired_face_width * desired_face_width * 3
        self.no_glasses_image = np.zeros((desired_face_width, desired_face_width, 3)) / n_values

    def __call__(self, image):
        image = F.resize(image, 256)
        gray = np.array(F.to_grayscale(image))

        detections = self.bounding_boxes(gray)
        if not detections:
            return self.no_glasses_image

        face_rect = detections[0].rect
        aligned_face = self.aligner.align(opencv2matplotlib(np.array(image)), gray, face_rect)
        return opencv2matplotlib(aligned_face)

    def bounding_boxes(self, gray_image):
        return self.detector(gray_image, 1)


class RandomHorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image):
        if random.random() < self.p:
            image = F.hflip(image)

        return image


class RandomGaussianBlur:
    def __call__(self, image):
        random_roll = random.random()
        if random_roll < 0.02:
            image = image.filter(ImageFilter.GaussianBlur(radius=2))
        elif random_roll < 0.1:
            image = image.filter(ImageFilter.GaussianBlur(radius=1))

        return image


class ToTensor:
    def __call__(self, image):
        image = np.array(image)

        mean = np.mean(image)
        channel_means = np.mean(image, axis=(0, 1))

        if mean == 0 and np.all(channel_means == 0):
            image = np.full_like(image, np.nan)
        else:
            image = mean * image / channel_means

        image = image.transpose((2, 0, 1))
        image = torch.from_numpy(image).float() / 255.

        return image
