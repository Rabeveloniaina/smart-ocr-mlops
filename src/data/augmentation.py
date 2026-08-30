import cv2
import numpy as np
import albumentations as A
from typing import Optional, Dict, Any


def get_training_augmentation(params_dict: Optional[Dict[str, Any]] = None) -> A.Compose:
    aug_params = params_dict.get("augmentation", {}) if params_dict else {}
    rot_range = aug_params.get("rotation_range", 7)
    elastic_alpha = aug_params.get("elastic_alpha", 80)
    elastic_sigma = aug_params.get("elastic_sigma", 6)

    return A.Compose([
        A.ElasticTransform(
            alpha=elastic_alpha,
            sigma=elastic_sigma,
            border_mode=cv2.BORDER_CONSTANT,
            p=0.40
        ),
        A.Affine(
            scale=(0.90, 1.10),
            rotate=(-rot_range, rot_range),
            translate_percent={"x": (-0.04, 0.04), "y": (-0.04, 0.04)},
            shear=(-8, 8),
            fill=255,
            p=0.55
        ),
        A.Perspective(scale=(0.02, 0.06), p=0.30),
        A.OneOf([
            A.Morphological(scale=(2, 3), operation="dilation", p=0.5),
            A.Morphological(scale=(2, 3), operation="erosion", p=0.5),
        ], p=0.35),
        A.OneOf([
            A.MotionBlur(blur_limit=3, p=0.5),
            A.GaussianBlur(blur_limit=3, p=0.5),
            A.MedianBlur(blur_limit=3, p=0.5),
        ], p=0.30),
        A.RandomBrightnessContrast(
            brightness_limit=0.20,
            contrast_limit=0.25,
            p=0.45
        ),
        A.RandomGamma(gamma_limit=(80, 120), p=0.25),
        A.GaussNoise(p=0.30),
        A.ImageCompression(quality_range=(60, 95), p=0.20),
    ])


def get_validation_augmentation() -> A.Compose:
    return A.Compose([])
