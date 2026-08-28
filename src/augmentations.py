"""
Custom Albumentations Augmentations for NoctVision.
Applies low-light simulation, synthetic fog, synthetic rain, and weather effects
specifically tailored for visible camera images in night-time/low-light pedestrian detection.
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np


def get_noctvision_transforms(img_size: int = 640, is_train: bool = True) -> A.Compose:
    """
    Returns Albumentations transform pipeline for training or evaluation.
    
    Includes low-light simulation, synthetic fog, rain effects, 
    RandomBrightnessContrast, Gamma, Gaussian Noise, and Blur.
    """
    if is_train:
        return A.Compose(
            [
                A.Resize(height=img_size, width=img_size),
                # Low-light & extreme lighting conditions simulation
                A.OneOf(
                    [
                        A.RandomGamma(gamma_limit=(30, 80), p=0.7),  # Simulates darker scenes
                        A.RandomBrightnessContrast(
                            brightness_limit=(-0.5, -0.1),
                            contrast_limit=(-0.3, 0.2),
                            p=0.7
                        ),
                    ],
                    p=0.8
                ),
                # Weather & Environmental conditions
                A.OneOf(
                    [
                        A.RandomRain(drop_length=15, drop_width=1, drop_color=(200, 200, 200), blur_value=3, p=0.5),
                        A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.3, alpha_coef=0.08, p=0.5),
                        A.MotionBlur(blur_limit=(3, 7), p=0.3),
                    ],
                    p=0.5
                ),
                # Sensor Noise common in low-light camera sensors
                A.OneOf(
                    [
                        A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
                        A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=0.5),
                    ],
                    p=0.4
                ),
                A.HorizontalFlip(p=0.5),
            ],
            bbox_params=A.BboxParams(
                format="yolo",
                label_fields=["class_labels"],
                min_visibility=0.2
            )
        )
    else:
        return A.Compose(
            [
                A.Resize(height=img_size, width=img_size),
            ],
            bbox_params=A.BboxParams(
                format="yolo",
                label_fields=["class_labels"]
            )
        )


if __name__ == "__main__":
    print("Testing Albumentations pipeline for NoctVision...")
    transforms = get_noctvision_transforms(img_size=640, is_train=True)
    print("Augmentation pipeline initialized successfully.")
