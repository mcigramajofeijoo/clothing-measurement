import time

import cv2

from app.src.pose import stage_pose
from app.src.utils import (draw_guide_skeleton, draw_skeleton,
                           get_guide_keypoints_in_pixels,
                           validate_pose_matches_guide)


def capture(view: str = "front"):
    cap = cv2.VideoCapture(0)

    ok_pose_start_time = None
    required_ok_seconds = 5

    captured_frame = None
    captured_frame_rgb = None
    captured_keypoints_xy = None
    captured_boxes_xyxy = None

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        clean_frame = frame.copy()

        guide_pixels = get_guide_keypoints_in_pixels(frame.shape, view)
        draw_guide_skeleton(frame, guide_pixels)

        # NOTE: If we already validate if the psoe matches the silhouette, it may not be needed to validate the abduction angles!
        overall_ok = False

        ok, messages, keypoints_xy, boxes_xyxy = stage_pose(frame, view)

        if keypoints_xy is not None:
            draw_skeleton(frame, keypoints_xy)
            keypoints_matches_guide = validate_pose_matches_guide(
                keypoints_xy, guide_pixels, frame.shape[1]
            )

            overall_ok = keypoints_matches_guide

        if overall_ok:
            cv2.putText(
                frame,
                "Stay still",
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )
            if ok_pose_start_time is None:
                ok_pose_start_time = time.time()

            elapsed = time.time() - ok_pose_start_time
            if elapsed >= required_ok_seconds:
                # Pose Captured
                captured_frame = clean_frame
                captured_keypoints_xy = keypoints_xy
                captured_boxes_xyxy = boxes_xyxy
                captured_frame_rgb = cv2.cvtColor(captured_frame, cv2.COLOR_BGR2RGB)  # INFO: Needed for SAM
                break
        else:
            cv2.putText(
                frame,
                f"Please try to align with the silhouette guide.",
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                1,
            )
            if messages:
                cv2.putText(
                    frame,
                    f"Hint: {messages[0]}",
                    (30, 75),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    1,
                )

            ok_pose_start_time = None

        cv2.imshow("Live Camera", frame)

        if cv2.waitKey(1) == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    cv2.waitKey(1) # To force the closing

    return (
        captured_frame,
        captured_frame_rgb,
        captured_keypoints_xy,
        captured_boxes_xyxy,
    )
