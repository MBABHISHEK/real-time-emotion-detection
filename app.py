from flask import Flask, render_template, Response, jsonify, request
import cv2
import time
import numpy as np
from tensorflow.keras.models import load_model
from collections import Counter

app = Flask(__name__)

# Load the emotion detection model
model = load_model("emotion_detection_model.h5")

# List of emotions corresponding to the labels in the dataset
emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

# Load face cascade classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

camera_active = False  # Track camera state
latest_emotion = "Neutral"  # Store the most recent emotion
emotion_counter = Counter()  # Count detected emotions
start_time = 0  # Track when the camera started

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat')
def chat():
    return render_template('camera.html')

@app.route('/video_feed')
def video_feed():
    global camera_active
    if not camera_active:
        start_camera()  # Start camera on first refresh
    return Response(generate_camera_feed(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_camera', methods=['POST'])
def start_camera():
    global camera_active, start_time, emotion_counter
    camera_active = True
    start_time = time.time()
    emotion_counter.clear()  # Reset emotion counts
    return "Camera started", 200

@app.route('/stop_camera', methods=['POST'])
def stop_camera():
    global camera_active
    camera_active = False
    duration = time.time() - start_time
    most_detected_emotion = emotion_counter.most_common(1)[0][0] if emotion_counter else "Unknown"
    return jsonify({
        "time_taken": f"{duration:.2f} seconds",
        "most_detected_emotion": most_detected_emotion
    }), 200

@app.route('/latest_emotion', methods=['GET'])
def get_latest_emotion():
    global latest_emotion
    return jsonify({"emotion": latest_emotion})

def generate_camera_feed():
    global latest_emotion, emotion_counter, camera_active

    camera = cv2.VideoCapture(0)  # Open video capture
    while camera_active:
        success, frame = camera.read()
        if not success:
            break

        # Convert frame to grayscale for face detection
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the frame
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            # Extract and preprocess the face region for the model
            face_roi = gray_frame[y:y + h, x:x + w]
            face_roi_resized = cv2.resize(face_roi, (48, 48))
            face_roi_resized = face_roi_resized.reshape(1, 48, 48, 1) / 255.0

            # Predict the emotion
            prediction = model.predict(face_roi_resized, verbose=0)
            emotion = emotions[np.argmax(prediction)]

            # Update latest emotion and counter
            latest_emotion = emotion
            emotion_counter[emotion] += 1

            # Draw the bounding box and emotion on the frame
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        # Encode the frame as JPEG
        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    camera.release()

if __name__ == '__main__':
    app.run(debug=True)
