from scipy.spatial import distance
from imutils import face_utils
import imutils
import dlib
import cv2
import time
import csv
import numpy as np
from keras.models import load_model

# Load the pre-trained emotion recognition model
net = load_model("facialemotionmodel.h5")

# Initialize dlib's face detector and facial landmark predictor
detect = dlib.get_frontal_face_detector()
predict = dlib.shape_predictor("drowsiness/shape_predictor_68_face_landmarks.dat")

# Emotion labels
emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

def mouth_aspect_ratio(mouth):
    A = distance.euclidean(mouth[3], mouth[9])  # Vertical distance
    B = distance.euclidean(mouth[0], mouth[6])  # Horizontal distance
    mar = A / B
    return mar

def detect_emotion(face_image):
    if face_image is None or face_image.size == 0:
        return "Unknown"
    
    face_image = cv2.resize(face_image, (48, 48))  # Resize to 48x48
    face_image = face_image / 255.0
    face_image = np.expand_dims(face_image, axis=0)
    face_image = np.expand_dims(face_image, axis=-1)
    preds = net.predict(face_image)
    return emotion_labels[np.argmax(preds)]

# Parameters
ear_thresh = 0.22  # EAR threshold for eyes
mar_thresh = 0.5   # MAR threshold for mouth (yawning)
frame_check = 100 # Number of consecutive frames the EAR or MAR should be below or above threshold

# Get the indexes of the facial landmarks for eyes and mouth
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]
(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["mouth"]

# Generate a unique CSV file name based on the current timestamp
csv_file_name = f"drowsiness_yawning_log_{time.strftime('%Y%m%d_%H%M%S')}.csv"

cap = cv2.VideoCapture(0)
flag = 0

# Open CSV file for logging
with open(csv_file_name, mode='w', newline='') as log_file:
    log_writer = csv.writer(log_file)
    log_writer.writerow(["Timestamp", "EAR", "MAR", "Drowsiness Detected", "Yawning Detected", "Emotion"])

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = imutils.resize(frame, width=450)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        subjects = detect(gray, 0)

        for subject in subjects:
            shape = predict(gray, subject)
            shape = face_utils.shape_to_np(shape)

            # Extract left and right eye coordinates
            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]
            mouth = shape[mStart:mEnd]

            # Calculate EAR for both eyes
            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)
            ear = (leftEAR + rightEAR) / 2.0

            # Calculate MAR for mouth
            mar = mouth_aspect_ratio(mouth)

            # Extract face ROI for emotion recognition
            (x, y, w, h) = face_utils.rect_to_bb(subject)
            face_image = gray[y:y+h, x:x+w]

            # Check if face_image is not empty
            if face_image.size > 0:
                emotion = detect_emotion(face_image)
            else:
                emotion = "Unknown"

            # Check if EAR is below the threshold or MAR is above the threshold
            drowsiness_detected = "Yes" if ear < ear_thresh else "No"
            yawning_detected = "Yes"