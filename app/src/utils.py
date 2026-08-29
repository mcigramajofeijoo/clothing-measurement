import cv2
import numpy as np
import torch
from cv2.typing import MatLike

from .constants import (GUIDE_FRONT_KEYPOINTS_RELATIVE,
                        GUIDE_SIDE_KEYPOINTS_RELATIVE,
                        GUIDE_SKELETON_CONNECTIONS, LEFT_ANKLE, LEFT_EAR,
                        LEFT_ELBOW, LEFT_HIP, LEFT_KNEE, LEFT_SHOULDER,
                        LEFT_WRIST, RIGHT_ANKLE, RIGHT_EAR, RIGHT_ELBOW,
                        RIGHT_HIP, RIGHT_KNEE, RIGHT_SHOULDER, RIGHT_WRIST,
                        SKELETON_CONNECTIONS)

# WARNING: We're transferring every tensor back to CPU whenever we're converting them to Numpy array.
# - More info:
#   In our local environment, the code was running entirely on the CPU, so PyTorch tensors were already in host CPU memory (cpu). When np.asarray() or .numpy() was called, it executed without issue.
#   If we're using GPU, so our YOLO or segmentation model places its output tensors directly on cuda:0 (GPU memory). NumPy cannot interact directly with GPU memory; the tensor must first be transferred back to host CPU memory.

def compute_scale_factor(
    tilt_deg: float, user_height_cm: float, bbox_height_px: float
) -> float:
    """
    Returns how many centimeters of real world height corresponds to 1 pixel in the image.

    Args:
        tilt_deg: The tilt of the camera in degrees.
        user_height_cm: The height of the user in centimeters.
        bbox_height_px: The height of the bounding box in pixels.

    Returns:
        The scale factor in centimeters per pixel.
    """
    tilt_rad = np.radians(tilt_deg)
    effective_height_cm = user_height_cm * np.cos(tilt_rad)

    cm_per_px = effective_height_cm / bbox_height_px
    return cm_per_px


def pixels_to_cm(pixels: float, cm_per_pixel: float) -> float:
    """
    Converts a number of pixels to centimeters using a scale factor.

    Args:
        pixels: The number of pixels.
        cm_per_pixel: The scale factor in centimeters per pixel.

    Returns:
        The number of centimeters.
    """
    return pixels * cm_per_pixel


# –––––––––––––––––––––––––––––––––––––––––––––––––––––
#                       CAPTURE
# –––––––––––––––––––––––––––––––––––––––––––––––––––––
def draw_skeleton(
    frame: MatLike,
    keypoints_xy: torch.Tensor,
    point_color: tuple[int, int, int] = (162, 70, 255),
    line_color: tuple[int, int, int] = (27, 206, 255),
    point_radius: int = 5,
    line_thickness: int = 2,
):
    """
    Draw the full skeleton (points + lines) over the frame (in-place).
    """
    # We draw the lines first so the points are above them
    for idx_a, idx_b in SKELETON_CONNECTIONS:
        x_a, y_a = map(int, keypoints_xy[idx_a])
        x_b, y_b = map(int, keypoints_xy[idx_b])

        point_a = (x_a, y_a)
        point_b = (x_b, y_b)

        cv2.line(frame, point_a, point_b, line_color, line_thickness)

    for x, y in keypoints_xy:
        x, y = int(x), int(y)
        cv2.circle(frame, (x, y), point_radius, point_color, thickness=-1)

    return frame


def get_guide_keypoints_in_pixels(
    frame_shape: tuple[float, float, int], view: str = "front"
):

    height, width = frame_shape[:2]

    if view == "front":
        guide_keypoints = GUIDE_FRONT_KEYPOINTS_RELATIVE
    else:
        guide_keypoints = GUIDE_SIDE_KEYPOINTS_RELATIVE

    guide_pixels = [(x_rel * width, y_rel * height) for x_rel, y_rel in guide_keypoints]

    return np.asarray(guide_pixels, dtype=np.float32)


def draw_guide_skeleton(
    frame: MatLike,
    guide_keypoints_in_pixels: np.ndarray,
    color: tuple[int, int, int] = (255, 255, 255),
    alpha: float = 0.75,
    point_radius: int = 5,
    line_thickness: int = 4,
):
    overlay = frame.copy()

    for idx_a, idx_b in GUIDE_SKELETON_CONNECTIONS:
        x_a, y_a = map(int, guide_keypoints_in_pixels[idx_a])
        x_b, y_b = map(int, guide_keypoints_in_pixels[idx_b])

        point_a = (x_a, y_a)
        point_b = (x_b, y_b)

        cv2.line(overlay, point_a, point_b, color, line_thickness)

    for x, y in guide_keypoints_in_pixels:
        x, y = int(x), int(y)
        cv2.circle(overlay, (x, y), point_radius, color, thickness=-1)

    # dst=frame hace que el resultado de la mezcla se escriba directamente sobre frame (in-place)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, dst=frame)

    return frame


def calculate_euclidean_distance(point_a: torch.Tensor, point_b: torch.Tensor):
    """
    Euclidean distance between 2 points [x, y].
    """
    point_a = np.asarray(point_a.cpu(), dtype=np.float32)
    point_b = np.asarray(point_b.cpu(), dtype=np.float32)

    subtraction = np.subtract(point_a, point_b)

    return np.linalg.norm(subtraction)


def validate_pose_matches_guide(
    keypoints_xy: torch.Tensor,
    guide_keypoints_in_pixels: np.ndarray,
    frame_width: float,
    max_distance_pct: float = 0.1,
) -> bool:
    """
    Verify the user keypoints are reasonably close to the guide keypoints (reference silhouette).

    max_distance_pct: maximum distance allowed, as % of frame width.
    """

    max_distance_px = frame_width * max_distance_pct

    for idx in range(len(guide_keypoints_in_pixels)):
        user_keypoint = keypoints_xy[idx]
        guide_keypoint = guide_keypoints_in_pixels[idx]

        euclidean_distance = calculate_euclidean_distance(user_keypoint, guide_keypoint)
        if euclidean_distance > max_distance_px:
            return False

    return True


# –––––––––––––––––––––––––––––––––––––––––––––––––––––
#                       POSE
# –––––––––––––––––––––––––––––––––––––––––––––––––––––
def calculate_angle(
    vertex: torch.Tensor, point_a: torch.Tensor, point_b: torch.Tensor
) -> float:
    # What is the dtype of these Tensors, is np.float32 as regular?
    vertex = np.asarray(vertex.cpu(), dtype=np.float32)
    point_a = np.asarray(
        point_a.cpu(), dtype=np.float32
    )  # This should be the vertical reference, for convention
    point_b = np.asarray(point_b.cpu(), dtype=np.float32)

    vector_a = np.subtract(point_a, vertex)
    vector_b = np.subtract(point_b, vertex)

    cos_theta = np.dot(vector_a, vector_b) / (
        np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
    )
    angle = np.arccos(cos_theta)

    return np.degrees(angle)


def get_arm_abduction_angle(keypoints_xy: torch.Tensor, side: str) -> float:
    """
    Calculate the arm abduction angle (shoulder-hip vs shoulder-elbow).

    keypoints_xy: tensor/array of shape (17, 2) - the keypoints of ONE person
    side: "right" or "left"
    """
    if side not in ["left", "right"]:
        raise ValueError("Side must be either left or right.")

    if side == "left":
        shoulder_idx, hip_idx, elbow_idx = (
            LEFT_SHOULDER,
            LEFT_HIP,
            LEFT_ELBOW,
        )
    else:
        shoulder_idx, hip_idx, elbow_idx = (
            LEFT_SHOULDER,
            LEFT_HIP,
            LEFT_ELBOW,
        )

    shoulder = keypoints_xy[shoulder_idx]
    hip = keypoints_xy[hip_idx]
    elbow = keypoints_xy[elbow_idx]

    return calculate_angle(vertex=shoulder, point_a=hip, point_b=elbow)


def get_leg_abduction_angle(keypoints_xy: torch.Tensor, side: str) -> float:
    """
    Calculate the abduction angle of the leg, using the hip
    center as vertex, and the shoulder center as vertical reference.
    """
    hip_center = midpoint(keypoints_xy[LEFT_HIP], keypoints_xy[RIGHT_HIP])
    shoulder_center = midpoint(
        keypoints_xy[LEFT_SHOULDER], keypoints_xy[RIGHT_SHOULDER]
    )

    # We cannot use the shoulder_center as reference_point because it's direction is the opposite to the ankle!
    direction_down = hip_center - shoulder_center
    reference_point_bottom = hip_center + direction_down

    ankle_idx = LEFT_ANKLE if side == "left" else RIGHT_ANKLE
    ankle = keypoints_xy[ankle_idx]

    return calculate_angle(
        vertex=hip_center, point_a=reference_point_bottom, point_b=ankle
    )


def check_keypoints_in_frame(
    keypoints_conf: torch.Tensor,
    min_confidence: float = 0.8,
    max_hidden_confidence: float = 0.3,
    view: str = "front",
):
    """
    Verifica que los keypoints tengan la confianza esperada segun la vista.

    Para "front": los 17 deben tener confianza alta.
    Para "side": el lado derecho + nariz deben tener confianza alta,
                 Y el lado izquierdo debe tener confianza BAJA
                 (confirmando que el usuario esta realmente de perfil,
                 no en diagonal).
    """
    FRONT_REQUIRED_KEYPOINTS = list(range(17))
    SIDE_VISIBLE_KEYPOINTS = [
        RIGHT_EAR,
        RIGHT_SHOULDER,
        RIGHT_ELBOW,
        RIGHT_WRIST,
        RIGHT_HIP,
        RIGHT_KNEE,
        RIGHT_ANKLE,
    ]
    SIDE_HIDDEN_KEYPOINTS = [
        LEFT_EAR,
        LEFT_SHOULDER,
        LEFT_ELBOW,
        LEFT_WRIST,
        LEFT_HIP,
        LEFT_KNEE,
        LEFT_ANKLE,
    ]

    # We could return a more helpful message using the keypoints, but for this demo is fine.
    if view == "front":
        checks = [
            keypoints_conf[idx] >= min_confidence for idx in FRONT_REQUIRED_KEYPOINTS
        ]

        if all(checks):
            return True, "All keypoints are visible."
        else:
            return (
                False,
                "Cannot detect required keypoints. Make sure your full body is in the frame.",
            )
    else:
        visible_checks = [
            keypoints_conf[idx] >= min_confidence for idx in SIDE_VISIBLE_KEYPOINTS
        ]
        hidden_checks = [
            keypoints_conf[idx] <= max_hidden_confidence
            for idx in SIDE_HIDDEN_KEYPOINTS
        ]

        if not all(visible_checks):
            return (
                False,
                "Your right side cannot be properly detected. Please make sure to align with the guide silhouette.",
            )
        elif not all(hidden_checks):
            return False, "Turn around further to see your full profile."
        else:
            return True, "Profile view correct."


def midpoint(point_a: torch.Tensor, point_b: torch.Tensor):
    """
    Calculate the midpoint between 2 keypoints.
    """
    point_a = np.asarray(point_a.cpu(), dtype=np.float32)
    point_b = np.asarray(point_b.cpu(), dtype=np.float32)

    return (point_a + point_b) / 2


def check_angle_range(
    angle: float, min_angle: float, max_angle: float, body_part: str
) -> tuple[bool, str]:
    """
    Verify if an angle is inside range.
    """
    if min_angle >= max_angle:
        raise ValueError("'min_angle' must be less than 'max_angle'.")

    if angle < min_angle:
        return False, f"Move your {body_part} a little further away."

    if angle > max_angle:
        return False, f"Bring your {body_part} a little closer."

    return True, "OK"


def validate_pose(keypoints_xy, keypoints_conf, view: str):
    """
    Valida la pose completa: visibilidad + angulos de brazos y piernas.
    Devuelve (es_valida: bool, lista_de_mensajes: list[str])
    """

    keypoints_in_frame, message = check_keypoints_in_frame(keypoints_conf, view=view)
    if not keypoints_in_frame:
        return False, [message]

    checks = [
        check_angle_range(
            get_arm_abduction_angle(keypoints_xy, side="left"), 10, 30, "left arm"
        ),
        check_angle_range(
            get_arm_abduction_angle(keypoints_xy, side="right"), 10, 30, "right arm"
        ),
        check_angle_range(
            get_leg_abduction_angle(keypoints_xy, side="left"), 5, 10, "left leg"
        ),
        check_angle_range(
            get_leg_abduction_angle(keypoints_xy, side="right"), 5, 10, "right leg"
        ),
    ]

    problems = [msg for is_valid, msg in checks if not is_valid]
    if problems:
        return False, problems

    return True, ["OK"]


# –––––––––––––––––––––––––––––––––––––––––––––––––––––
#                       SEGMENTATION
# –––––––––––––––––––––––––––––––––––––––––––––––––––––
def resize_mask_to_original_image_size(
    mask_tensor: torch.Tensor, img_shape: tuple[float, float, int]
):
    # OpenCV works with Numpy arrays, not tensors
    # uint8 is the standard type for images (integer numbers from 0 to 255, even though here we're only using 0 and 1).
    mask_numpy = np.asarray(mask_tensor.cpu(), dtype=np.uint8)

    img_height, img_width, _ = img_shape

    resized_mask = cv2.resize(
        mask_numpy,
        (img_width, img_height),
        interpolation=cv2.INTER_NEAREST,  # CRITIC: This is ESSENTIAL so the mask stays binary after we resize it!
    )

    return resized_mask


def get_binary_silhouette(
    resized_mask: np.ndarray, person_color: int = 255, background_color: int = 0
):
    """
    Convierte la mascara 0/1 a una imagen blanco/negro y la guarda.

    person_color: valor de gris para los pixeles de la persona (0-255)
    background_color: valor de gris para el fondo (0-255)
    """
    silhouette = np.where(resized_mask == 1, person_color, background_color).astype(
        np.uint8
    )
    return silhouette
