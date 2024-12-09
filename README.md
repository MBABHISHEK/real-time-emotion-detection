# Emotion Detection System

This project is a real-time **Emotion Detection System** using Flask, OpenCV, and TensorFlow. It uses a Convolutional Neural Network (CNN) to detect and classify emotions based on facial expressions captured via a webcam. The detected emotions are displayed along with bounding boxes drawn on the faces in the video feed.

---

## Features

1. **Real-time Emotion Detection:**
   - Detects emotions (`angry`, `disgust`, `fear`, `happy`, `sad`, `surprise`, `neutral`) in live video feeds.
2. **Face Detection:**
   - Uses Haar cascades for face detection.
3. **Emotion Analytics:**
   - Tracks the most common emotion detected during a session and provides session duration.
4. **Web Interface:**
   - A simple web interface to start/stop the camera and view real-time video with emotion annotations.

---

## Installation

### Prerequisites
- Python 3.7+
- Flask
- TensorFlow
- OpenCV
- NumPy
- Pandas
- SciKit-Learn

### Steps to Set Up

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/your-repository/emotion-detection
   cd emotion-detection
2. **Install dependcies :**
     ```bash
     pip install -r requirements.txt
     ```

3. **Download Dataset:**
   Obtain the FER2013 dataset from Kaggle and place it in the project directory as fer2013.csv.


4. **Train the Model (Optional):**
 ```bash
 python train_model.py
```

5. **Run the Application:**
  ```bash
  python app.py
```
Access the application at http://127.0.0.1:5000/.
