import cv2
import numpy as np
import math


def rotated_image(image: np.ndarray, angle: int = 45) -> np.ndarray:
    """Rotate image by angle degrees."""
    width, height, _ = image.shape
    transform = cv2.getRotationMatrix2D((height / 2, width / 2),
                                        angle=angle, scale=1)
    result = cv2.warpAffine(image, transform, (height, width))
    return result


def rotated_image_fit_frames(image: np.ndarray, angle: int = 45) -> np.ndarray:
    """Rotate image by angle degrees."""
    width, height, _ = image.shape
    alpha = abs(angle)

    min_size = min([width, height])
    max_size = max([width, height])
    hypot = math.hypot(width, height)
    gamma = math.atan(min_size/max_size)

    if alpha > 90:
        alpha = math.radians(180 - alpha)
    else:
        alpha = math.radians(alpha)

    hypot_rotated = min_size / math.sin(gamma + alpha)
    scale = hypot_rotated / hypot

    transform = cv2.getRotationMatrix2D((height / 2, width / 2),
                                        angle=angle, scale=scale)
    result = cv2.warpAffine(image, transform, (height, width))
    return result