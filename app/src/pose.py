import torch
from cv2.typing import MatLike
from ultralytics import YOLO

from app.src.utils import validate_pose
from paths import APP_PATH

POSE_MODEL_PATH = APP_PATH / "src" / "yolo26n-pose.pt"

POSE_MODEL = YOLO(str(POSE_MODEL_PATH))


def stage_pose(img: MatLike, view: str, testing: bool = False):
    results = POSE_MODEL(img, classes=[0])

    num_people_detected = len(results[0].keypoints.xy)

    if num_people_detected == 0:
        return False, ["No person detected in frame."], None, None

    boxes_xyxy: torch.Tensor = results[0].boxes.xyxy
    keypoints_xy: torch.Tensor = results[0].keypoints.xy[0]
    keypoints_conf: torch.Tensor = results[0].keypoints.conf[0]

    if testing:
        return True, ["OK"], keypoints_xy, boxes_xyxy
    else: 
        ok, messages = validate_pose(keypoints_xy, keypoints_conf, view)

        return ok, messages, keypoints_xy, boxes_xyxy
