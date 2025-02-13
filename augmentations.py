import random
import numpy as np
import cv2
import albumentations as A
from albumentations.core.transforms_interface import ImageOnlyTransform
from albumentations.pytorch import ToTensorV2


class PixelDisplacementTransform(ImageOnlyTransform):
    """
    Кастомная аугментация: случайное смещение пикселей с использованием cv2.remap.
    """
    def __init__(self, stddev: float = 0.7, always_apply: bool = False, p: float = 0.5):
        super().__init__(always_apply, p)
        self.stddev = stddev

    def apply(self, image, **params):
        rows, cols, _ = image.shape
        displacement_x = np.random.normal(0, self.stddev, (rows, cols))
        displacement_y = np.random.normal(0, self.stddev, (rows, cols))
        x, y = np.meshgrid(np.arange(cols), np.arange(rows))
        map_x = np.clip(x + displacement_x, 0, cols - 1).astype(np.float32)
        map_y = np.clip(y + displacement_y, 0, rows - 1).astype(np.float32)
        distorted = cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        return distorted


def make_albumentations(
    use_pixeldisp: bool = False, pixel_disp_std: float = 0.7,
    use_flip: bool = False, flip_prob: float = 0.5,
    use_vflip: bool = False, vflip_prob: float = 0.5,
    use_brightness: bool = False, brightness_limit: float = 0.2,
    use_rotation: bool = False, max_rotation: float = 30.0,
    use_shift_scale_rot: bool = False, shift_limit: float = 0.0625, scale_limit: float = 0.1, rotate_limit: float = 45,
    use_gaussnoise: bool = False, gauss_varlimit: float = 30.0,
    use_gaussianblur: bool = False, blur_limit: int = 3,
    use_motionblur: bool = False, mblur_limit: int = 3,
    use_sharpen: bool = False, sharpen_alpha: float = 0.3,
) -> A.Compose:
    """
    Собирает пайплайн аугментаций Albumentations с заданными параметрами.
    """
    transforms = [A.Resize(64, 64)]
    if use_pixeldisp:
        transforms.append(PixelDisplacementTransform(stddev=pixel_disp_std, p=0.5))
    if use_flip:
        transforms.append(A.HorizontalFlip(p=flip_prob))
    if use_vflip:
        transforms.append(A.VerticalFlip(p=vflip_prob))
    if use_brightness:
        transforms.append(A.RandomBrightnessContrast(brightness_limit=brightness_limit, p=0.5))
    if use_rotation and max_rotation > 0:
        transforms.append(A.Rotate(limit=max_rotation, p=0.5))
    if use_shift_scale_rot:
        transforms.append(A.ShiftScaleRotate(
            shift_limit=shift_limit,
            scale_limit=scale_limit,
            rotate_limit=rotate_limit,
            border_mode=cv2.BORDER_REFLECT,
            p=0.5
        ))
    if use_gaussnoise:
        transforms.append(A.GaussNoise(var_limit=(10.0, gauss_varlimit), p=0.5))
    if use_gaussianblur:
        transforms.append(A.GaussianBlur(blur_limit=(3, blur_limit), p=0.5))
    if use_motionblur:
        transforms.append(A.MotionBlur(blur_limit=(3, mblur_limit), p=0.5))
    if use_sharpen:
        transforms.append(A.Sharpen(alpha=(0.1, sharpen_alpha), p=0.5))
    transforms.append(A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
    transforms.append(ToTensorV2())
    return A.Compose(transforms)
