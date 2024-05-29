import cv2
import json
import mediapipe as mp
import numpy as np
from scipy.spatial import distance

# Load JSON data
with open('data_yoga.json') as f:
    data = json.load(f)

image_name = "Camel"
image = None

# Extract landmarks from JSON data
landmarks_from_json = []
the_landmarks = None


# Load the image
for the_data in data:
    if the_data['name'] == image_name:
        for lm in the_data['landmarks']:
            landmarks_from_json.append([lm['coordinates'][0], lm['coordinates'][1]])
        the_landmarks = the_data['landmarks']
        image = cv2.imread(the_data['image_name'])

# Initialize MediaPipe pose module
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Function to calculate similarity between two sets of landmarks
def calculate_similarity(landmarks1, landmarks2):
    if not landmarks1 or not landmarks2:
        return 0
    # Normalize landmarks
    norm_landmarks1 = np.array(landmarks1) - np.mean(landmarks1, axis=0)
    norm_landmarks2 = np.array(landmarks2) - np.mean(landmarks2, axis=0)
    # Calculate the distance between corresponding landmarks
    dists = [distance.euclidean(lm1, lm2) for lm1, lm2 in zip(norm_landmarks1, norm_landmarks2)]
    # Calculate similarity as the inverse of the average distance
    similarity = 1 / (1 + np.mean(dists))
    return similarity * 100

# Draw landmarks and connections on the image
def draw_landmarks(image, landmarks):
    annotated_image = image.copy()
    
    for landmark in landmarks:
        # Extract landmark coordinates
        landmark_x = int(landmark['coordinates'][0] * annotated_image.shape[1])
        landmark_y = int(landmark['coordinates'][1] * annotated_image.shape[0])
        # Draw a circle at the landmark position
        cv2.circle(annotated_image, (landmark_x, landmark_y), 5, (0, 255, 0), -1)
        # Add text with the body part label
        cv2.putText(annotated_image, landmark['body'], (landmark_x, landmark_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1)

    # Draw connections between landmarks
    connections = mp_pose.POSE_CONNECTIONS
    for connection in connections:
        start_idx = connection[0]
        end_idx = connection[1]
        if 0 <= start_idx < len(landmarks) and 0 <= end_idx < len(landmarks):
            start_landmark = landmarks[start_idx]['coordinates']
            end_landmark = landmarks[end_idx]['coordinates']
            start_x = int(start_landmark[0] * annotated_image.shape[1])
            start_y = int(start_landmark[1] * annotated_image.shape[0])
            end_x = int(end_landmark[0] * annotated_image.shape[1])
            end_y = int(end_landmark[1] * annotated_image.shape[0])
            cv2.line(annotated_image, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)

    return annotated_image



# Annotate and display the image
annotated_image = draw_landmarks(image, the_landmarks)
cv2.imshow('Image with Landmarks and Connections', annotated_image)

# Open webcam
cap = cv2.VideoCapture(0)

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the BGR image to RGB
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        
        # Process the image and detect the pose
        results = pose.process(image_rgb)

        # Convert the RGB image back to BGR
        image_rgb.flags.writeable = True
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        # Draw the pose annotation on the image
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                image_bgr,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
            )

            # Extract landmarks
            landmarks_from_webcam = []
            for lm in results.pose_landmarks.landmark:
                landmarks_from_webcam.append([lm.x, lm.y])

            # Calculate similarity
            similarity = calculate_similarity(landmarks_from_json, landmarks_from_webcam)
            cv2.putText(image_bgr, f'Similarity: {similarity:.2f}%', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Display the image
        cv2.imshow('Webcam Pose', image_bgr)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

# Release resources
cap.release()
cv2.destroyAllWindows()
cv2.waitKey(1)
