from paths import APP_PATH
import cv2
from app.src.pose import stage_pose

SRC_PATH = APP_PATH / "src"

def dummy_capture(view: str ="front"):
    front_img_path = str(SRC_PATH / "dummy-front.jpeg")
    side_img_path = str(SRC_PATH / "dummy-side.jpeg")

    img = cv2.imread(front_img_path if view == "front" else side_img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    ok, messages, keypoints_xy, boxes_xyxy = stage_pose(img, view=view, testing=True)

    return img, img_rgb, keypoints_xy, boxes_xyxy