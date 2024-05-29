from flask import Flask, render_template, Response , request,jsonify,send_file
from flask_socketio import SocketIO
import cv2
import json
import mediapipe as mp
import numpy as np
from scipy.spatial import distance
import base64
import os

app = Flask(__name__)
socketio = SocketIO(app)

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils


image_name = "Camel"
image = None

# Extract landmarks from JSON data
landmarks_from_json = []
the_landmarks = None
dataset = {"name": "", "ket": ""}
similarity = 0
all_data = []

def load_image_and_landmarks(image_name):
    global image, landmarks_from_json, the_landmarks,all_data
    # Load JSON data
    with open('data_yoga.json') as f:
        data = json.load(f)
        all_data = data

    # Load the image and landmarks
    for the_data in data:
        if the_data['name'] == image_name:
            for lm in the_data['landmarks']:
                landmarks_from_json.append([lm['coordinates'][0], lm['coordinates'][1]])
            the_landmarks = the_data['landmarks']
            image = cv2.imread(the_data['image_name'])
            dataset["name"] = the_data['name']
            dataset["ket"] = the_data['ket']

def calculate_similarity(landmarks1, landmarks2):
    if not landmarks1 or not landmarks2:
        return 0
    norm_landmarks1 = np.array(landmarks1) - np.mean(landmarks1, axis=0)
    norm_landmarks2 = np.array(landmarks2) - np.mean(landmarks2, axis=0)
    dists = [distance.euclidean(lm1, lm2) for lm1, lm2 in zip(norm_landmarks1, norm_landmarks2)]
    similarity = 1 / (1 + np.mean(dists))
    return similarity * 100

def draw_landmarks(image, landmarks):
    annotated_image = image.copy()
    for landmark in landmarks:
        landmark_x = int(landmark['coordinates'][0] * annotated_image.shape[1])
        landmark_y = int(landmark['coordinates'][1] * annotated_image.shape[0])
        cv2.circle(annotated_image, (landmark_x, landmark_y), 5, (0, 255, 0), -1)
        # cv2.putText(annotated_image, landmark['body'], (landmark_x, landmark_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

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

def generate_frames():
    global similarity
    cap = cv2.VideoCapture(0)
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image_rgb.flags.writeable = False
            results = pose.process(image_rgb)
            image_rgb.flags.writeable = True
            image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    image_bgr,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=4, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=4, circle_radius=2),
                )

                landmarks_from_webcam = []
                for lm in results.pose_landmarks.landmark:
                    landmarks_from_webcam.append([lm.x, lm.y])

                similarity = calculate_similarity(landmarks_from_json, landmarks_from_webcam)
                cv2.putText(image_bgr, f'Similarity: {similarity:.2f}%', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            ret, buffer = cv2.imencode('.jpg', image_bgr)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

@app.route('/')
def index():
    # create a 2 var, 1 is previous and the other is next, check the previous of the current, if first index, the show the last one, and also check the next of the current, if last index, show the first one
    image_name = request.args.get('image_name', 'Camel')
    previous = None
    next = None
    load_image_and_landmarks(image_name)

    # get the index of the image_name
    current_index = 0
    # loop the all_data and find the index of the image_name
    for index, data in enumerate(all_data):
        if data['name'] == image_name:
            current_index = index
            break

    if current_index == 0:
        previous = all_data[-1]['name']
    else:
        previous = all_data[current_index - 1]['name']

    if current_index == len(all_data) - 1:
        next = all_data[0]['name']
    else:
        next = all_data[current_index + 1]['name']
    

    print(image_name, previous, next)


    annotated_image = draw_landmarks(image, the_landmarks)
    _, buffer = cv2.imencode('.jpg', annotated_image)
    img_str = base64.b64encode(buffer).decode('utf-8')
    return render_template('index2.html', img_str=img_str, previous=previous, next=next)



@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/getdata', methods=['GET'])
def getdata():
    return jsonify(all_data)

@app.route('/similarity', methods=['GET'])
def get_similarity():
    global similarity
    return {'similarity': similarity ,'data' : dataset}   

@app.route('/pose_dataset')
def pose_dataset():
    load_image_and_landmarks("Camel")
    return render_template('pose_dataset.html', data=all_data)

@app.route('/show_image')
def show_image():
    image_path = request.args.get('image_path')
    if image_path and os.path.exists(image_path):
        return send_file(image_path, mimetype='image/jpeg')
    else:
        return "Image not found", 404


if __name__ == '__main__':
    socketio.run(app, debug=True)
