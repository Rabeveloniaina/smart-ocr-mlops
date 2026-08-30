import cv2
import numpy as np
import albumentations as A
from typing import Optional, Dict, Any


def get_training_augmentation(params_dict: Optional[Dict[str, Any]] = None) -> A.Compose:
    aug_params = params_dict.get("augmentation", {}) if params_dict else {}
    
    rot_range = aug_params.get("rotation_range", 5)
    elastic_alpha = aug_params.get("elastic_alpha", 50)
    elastic_sigma = aug_params.get("elastic_sigma", 5)
    
    return A.Compose([
        A.ElasticTransform(
            alpha=elastic_alpha,
            sigma=elastic_sigma,
            border_mode=cv2.BORDER_CONSTANT,
            p=0.3
        ),
        A.Affine(
            scale=(0.95, 1.05),
            rotate=(-rot_range, rot_range),
            translate_percent={"x": (-0.03, 0.03), "y": (-0.03, 0.03)},
            shear=(-5, 5),
            fill=255,
            p=0.5
        ),
        A.OneOf([
            A.MotionBlur(blur_limit=3, p=0.4),
            A.GaussianBlur(blur_limit=3, p=0.4),
        ], p=0.3),
        A.RandomBrightnessContrast(
            brightness_limit=0.15,
            contrast_limit=0.2,
            p=0.4
        ),
        A.GaussNoise(p=0.3),
    ])


def get_validation_augmentation() -> A.Compose:
    return A.Compose([])
