import cv2
import mediapipe as mp
import json

# Initialize MediaPipe pose module
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Load and process the image
image_path = 'gerakan/Childs.jpg'
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image_results = pose.process(image_rgb)

# Extract landmarks from the image
def extract_landmarks(results):
    if not results.pose_landmarks:
        return None
    landmarks = []
    for lm in results.pose_landmarks.landmark:
        landmarks.append((lm.x, lm.y, lm.z))
    return landmarks

image_landmarks = extract_landmarks(image_results)

# Define mapping between landmark indices and body parts
landmark_labels = {
    0: 'nose',
    1: 'left_eye_inner',
    2: 'left_eye',
    3: 'left_eye_outer',
    4: 'right_eye_inner',
    5: 'right_eye',
    6: 'right_eye_outer',
    7: 'left_ear',
    8: 'right_ear',
    9: 'mouth_left',
    10: 'mouth_right',
    11: 'left_shoulder',
    12: 'right_shoulder',
    13: 'left_elbow',
    14: 'right_elbow',
    15: 'left_wrist',
    16: 'right_wrist',
    17: 'left_pinky',
    18: 'right_pinky',
    19: 'left_index',
    20: 'right_index',
    21: 'left_thumb',
    22: 'right_thumb',
    23: 'left_hip',
    24: 'right_hip',
    25: 'left_knee',
    26: 'right_knee',
    27: 'left_ankle',
    28: 'right_ankle',
    29: 'left_heel',
    30: 'right_heel',
    31: 'left_foot_index',
    32: 'right_foot_index'
}

# Map landmark indices to descriptive labels
descriptive_landmarks = [{'body': landmark_labels.get(idx, 'unknown'), 'coordinates': coord} for idx, coord in enumerate(image_landmarks)]

# Prepare landmark coordinates data
landmark_data = {
    'image_name': image_path,
    'landmarks': descriptive_landmarks
}

# Save landmark coordinates to a variable and print as JSON
landmark_coordinates = json.dumps(landmark_data, indent=4)
print(landmark_coordinates)

# Release resources
pose.close()
