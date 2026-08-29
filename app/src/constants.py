# COCO 17-Keypoint Indices
NOSE = 0
LEFT_EYE = 1
RIGHT_EYE = 2
LEFT_EAR = 3
RIGHT_EAR = 4
LEFT_SHOULDER = 5
RIGHT_SHOULDER = 6
LEFT_ELBOW = 7
RIGHT_ELBOW = 8
LEFT_WRIST = 9
RIGHT_WRIST = 10
LEFT_HIP = 11
RIGHT_HIP = 12
LEFT_KNEE = 13
RIGHT_KNEE = 14
LEFT_ANKLE = 15
RIGHT_ANKLE = 16

SKELETON_FACE_CONNECTIONS = [
    # Face Connections
    (NOSE, LEFT_EYE),  # Nose - Left Eye
    (NOSE, RIGHT_EYE),  # Nose - Right Eye
    (LEFT_EYE, RIGHT_EYE),  # Left Eye - Right Eye
    (LEFT_EYE, LEFT_EAR),  # Left Eye - Left Ear
    (RIGHT_EYE, RIGHT_EAR),  # Right Eye - Right Ear
    (LEFT_EAR, LEFT_SHOULDER),  # Left Ear - Left Shoulder
    (RIGHT_EAR, RIGHT_SHOULDER),  # Right Ear - Right Shoulder
]

SKELETON_UPPER_BODY_CONNECTIONS = [
    # Upper Body Connections
    (LEFT_SHOULDER, RIGHT_SHOULDER),  # Left Shoulder - Right Shoulder
    (LEFT_SHOULDER, LEFT_ELBOW),  # Left Shoulder - Left Elbow
    (LEFT_ELBOW, LEFT_WRIST),  # Left Elbow - Left Wrist
    (RIGHT_SHOULDER, RIGHT_ELBOW),  # Right Shoulder - Right Elbow
    (RIGHT_ELBOW, RIGHT_WRIST),  # Right Elbow - Right Wrist
]

SKELETON_TORSO_CONNECTIONS = [
    # Torso Connections
    (LEFT_SHOULDER, LEFT_HIP),  # Left Shoulder - Left Hip
    (RIGHT_SHOULDER, RIGHT_HIP),  # Right Shoulder - Right Hip
    (LEFT_HIP, RIGHT_HIP),  # Left Hip - Right Hip
]

SKELETON_LOWER_BODY_CONNECTIONS = [
    # Lower Body Connections
    (LEFT_HIP, LEFT_KNEE),  # Left Hip - Left Knee
    (LEFT_KNEE, LEFT_ANKLE),  # Left Knee - Left Ankle
    (RIGHT_HIP, RIGHT_KNEE),  # Right Hip - Right Knee
    (RIGHT_KNEE, RIGHT_ANKLE),  # Right Knee - Right Ankle
]

SKELETON_CONNECTIONS = [
    *SKELETON_FACE_CONNECTIONS,
    *SKELETON_UPPER_BODY_CONNECTIONS,
    *SKELETON_TORSO_CONNECTIONS,
    *SKELETON_LOWER_BODY_CONNECTIONS,
]

GUIDE_FRONT_KEYPOINTS_RELATIVE = [
    (0.50, 0.12),  # 0 nose
    (0.52, 0.10),  # 1 left eye
    (0.48, 0.10),  # 2 right eye
    (0.54, 0.11),  # 3 left ear
    (0.46, 0.11),  # 4 right ear
    (0.58, 0.22),  # 5 left shoulder
    (0.42, 0.22),  # 6 right shoulder
    (0.63, 0.38),  # 7 left elbow
    (0.37, 0.38),  # 8 right elbow
    (0.66, 0.52),  # 9 left wrist
    (0.34, 0.52),  # 10 right wrist
    (0.56, 0.55),  # 11 left hip
    (0.44, 0.55),  # 12 right hip
    (0.56, 0.75),  # 13 left knee
    (0.44, 0.75),  # 14 right knee
    (0.56, 0.95),  # 15 left ankle
    (0.44, 0.95),  # 16 right ankle
]

GUIDE_SIDE_KEYPOINTS_RELATIVE = [
    (0.50, 0.11),  # 0 nose
    (0.50, 0.10),  # 1 left eye
    (0.50, 0.10),  # 2 right eye
    (0.50, 0.11),  # 3 left ear
    (0.47, 0.11),  # 4 right ear
    (0.50, 0.23),  # 5 left shoulder
    (0.50, 0.23),  # 6 right shoulder
    (0.51, 0.37),  # 7 left elbow
    (0.51, 0.37),  # 8 right elbow
    (0.51, 0.50),  # 9 left wrist
    (0.51, 0.50),  # 10 right wrist
    (0.50, 0.55),  # 11 left hip
    (0.50, 0.55),  # 12 right hip
    (0.50, 0.75),  # 13 left knee
    (0.50, 0.75),  # 14 right knee
    (0.50, 0.95),  # 15 left ankle
    (0.50, 0.95),  # 16 right ankle
]

GUIDE_SKELETON_CONNECTIONS = [
    *SKELETON_FACE_CONNECTIONS,
    *SKELETON_UPPER_BODY_CONNECTIONS,
    *SKELETON_TORSO_CONNECTIONS,
    *SKELETON_LOWER_BODY_CONNECTIONS,
]
