"""Image preprocessing, validation, normalization, and augmentation utilities."""
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import cv2
import numpy as np
from PIL import Image
import io

CANONICAL_CLASSES: List[str] = [
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Sad",
    "Surprise",
    "Neutral",
]

CLASS_TO_IDX: Dict[str, int] = {name: i for i, name in enumerate(CANONICAL_CLASSES)}
IDX_TO_CLASS: Dict[int, str] = {i: name for i, name in enumerate(CANONICAL_CLASSES)}

# Normalization mapping for case-insensitivity
_LOWER_TO_CANONICAL: Dict[str, str] = {name.lower(): name for name in CANONICAL_CLASSES}

def normalize_class_name(name: str) -> str:
    """
    Normalizes a folder name or label string into the canonical Emotion class name.
    Raises ValueError if class is unrecognized.
    """
    cleaned = name.strip().lower()
    if cleaned in _LOWER_TO_CANONICAL:
        return _LOWER_TO_CANONICAL[cleaned]
    raise ValueError(f"Unrecognized emotion class: '{name}'. Expected one of: {CANONICAL_CLASSES}")

def validate_image_file(file_path: Union[str, Path]) -> Tuple[bool, Optional[str]]:
    """
    Checks if an image file is readable and non-corrupted.
    Returns (is_valid, error_message).
    """
    path = Path(file_path)
    if not path.exists():
        return False, "File does not exist"
    if path.stat().st_size == 0:
        return False, "File is empty (0 bytes)"

    try:
        with Image.open(path) as img:
            img.verify()
        # Ensure OpenCV can also decode it
        arr = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if arr is None or arr.size == 0:
            return False, "Failed to decode image pixels via OpenCV"
        return True, None
    except Exception as exc:
        return False, f"Corrupted image format: {str(exc)}"

def preprocess_image(
    image_input: Union[str, Path, np.ndarray, bytes, Image.Image],
    target_size: Tuple[int, int] = (48, 48),
    normalize: bool = True,
) -> np.ndarray:
    """
    Preprocesses raw image into standardized 48x48 single-channel grayscale float32 array in range [0, 1].
    Output shape: (48, 48, 1).
    """
    if isinstance(image_input, (str, Path)):
        img_arr = cv2.imread(str(image_input), cv2.IMREAD_GRAYSCALE)
        if img_arr is None:
            raise ValueError(f"Could not read image from path: {image_input}")
    elif isinstance(image_input, bytes):
        pil_img = Image.open(io.BytesIO(image_input)).convert("L")
        img_arr = np.array(pil_img)
    elif isinstance(image_input, Image.Image):
        pil_img = image_input.convert("L")
        img_arr = np.array(pil_img)
    elif isinstance(image_input, np.ndarray):
        if image_input.ndim == 3 and image_input.shape[2] == 3:
            img_arr = cv2.cvtColor(image_input, cv2.COLOR_BGR2GRAY)
        elif image_input.ndim == 3 and image_input.shape[2] == 1:
            img_arr = image_input.squeeze(axis=-1)
        else:
            img_arr = image_input
    else:
        raise TypeError(f"Unsupported image input type: {type(image_input)}")

    # Resize if not already target size
    if img_arr.shape[:2] != target_size:
        img_arr = cv2.resize(img_arr, target_size, interpolation=cv2.INTER_AREA)

    # Standardize dtype & normalize to [0, 1]
    if normalize:
        img_arr = img_arr.astype(np.float32) / 255.0
    else:
        img_arr = img_arr.astype(np.uint8)

    # Ensure channel dimension: (H, W, 1)
    if img_arr.ndim == 2:
        img_arr = np.expand_dims(img_arr, axis=-1)

    return img_arr

def augment_image(
    image: np.ndarray,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Applies reproducible training-only data augmentation:
    - Random horizontal flip (50% prob)
    - Random rotation (-10 to +10 degrees)
    - Random translation (-2 to +2 pixels)
    - Random brightness/contrast perturbation
    """
    if rng is None:
        rng = np.random.default_rng()

    # Image shape should be (H, W, 1) or (H, W)
    h, w = image.shape[:2]
    aug = image.copy()
    if aug.ndim == 3:
        aug_2d = aug.squeeze(axis=-1)
    else:
        aug_2d = aug

    # 1. Random horizontal flip
    if rng.random() > 0.5:
        aug_2d = np.fliplr(aug_2d)

    # 2. Random rotation & translation affine transform
    angle = rng.uniform(-10.0, 10.0)
    tx = rng.uniform(-2.0, 2.0)
    ty = rng.uniform(-2.0, 2.0)

    center = (w / 2.0, h / 2.0)
    rot_mat = cv2.getRotationMatrix2D(center, angle, scale=1.0)
    rot_mat[0, 2] += tx
    rot_mat[1, 2] += ty

    aug_2d = cv2.warpAffine(
        aug_2d,
        rot_mat,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101
    )

    # 3. Random contrast & brightness perturbation
    contrast_scale = rng.uniform(0.9, 1.1)
    brightness_shift = rng.uniform(-0.05, 0.05)
    aug_2d = np.clip(aug_2d * contrast_scale + brightness_shift, 0.0, 1.0)

    return np.expand_dims(aug_2d.astype(np.float32), axis=-1)
