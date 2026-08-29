import warnings
import time
from dotenv import load_dotenv

load_dotenv()

# INFO: `UserWarning: Momentum is not enabled:` This warning only matters during model training and has no effect on mesh accuracy, speed, or output quality.
warnings.filterwarnings("ignore", category=UserWarning)

import setup.sys_path_manipulation

from notebook.utils import setup_sam_3d_body

from app.src.capture import capture
from app.src.dummy_capture import dummy_capture
from app.src.segmentation import stage_segmentation

if __name__ == "__main__":
    # print("Testing")

    while True:
        try:
            user_height_cm = int(input("Enter your height in centimeters: "))
            if user_height_cm:
                break
        except ValueError as exc:
            print(
                "Invalid height. Make sure you provide your height in centimeters, e.g: 185"
            )
            pass

    captured_frame, captured_frame_rgb, keypoints_xy, boxes_xyxy = dummy_capture(view="front")
    bbox_height_px, binary_s = stage_segmentation(captured_frame)

    start_time = time.time()
    estimator = setup_sam_3d_body(hf_repo_id="facebook/sam-3d-body-dinov3", segmentor_name="sam3", device="cpu")

    outputs = estimator.process_one_image(
        img=captured_frame_rgb, bboxes=boxes_xyxy, masks=binary_s
    )

    latency = time.time() - start_time * 1000
    print(f"Inference Call Latency: {latency}ms")
    print(outputs)
