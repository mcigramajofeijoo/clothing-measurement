import numpy as np
import torch
from cv2.typing import MatLike
from ultralytics import YOLO
from paths import APP_PATH
from app.src.utils import (get_binary_silhouette,
                           resize_mask_to_original_image_size)

SEG_MODEL_PATH = APP_PATH / "src" / "yolo26n-seg.pt"
SEG_MODEL = YOLO(str(SEG_MODEL_PATH))


def stage_segmentation(img: MatLike):
    res = SEG_MODEL(img, classes=[0])[0]

    boxes: np.ndarray = res.boxes.xyxy.cpu().numpy()
    _, y1, _, y2 = boxes[0]
    bbox_height_px = y2 - y1

    mask: torch.Tensor = res.masks.data[0]
    resized_mask: np.ndarray = resize_mask_to_original_image_size(mask, img.shape)

    # binary_silhouette: np.ndarray = get_binary_silhouette(resized_mask)

    return bbox_height_px, resized_mask
