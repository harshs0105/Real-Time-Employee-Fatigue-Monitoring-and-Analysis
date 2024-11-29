import cv2
import os
import numpy as np
from keras.models import load_model
from pygame import mixer

# Define constants for Haar Cascade files and model paths
HAAR_CASCADE_FACE = 'C:/ALL/Project Final year/eye and mouth ML HAAR/haarcascade_frontalface_alt.xml'
HAAR_CASCADE_LEFT_EYE = 'C:/ALL/Project Final year/eye and mouth ML HAAR/haarcascade_lefteye_2splits.xml'
HAAR_CASCADE_RIGHT_EYE = 'C:/ALL/Project Final year/eye and mouth ML HAAR/haarcascade_righteye_2splits.xml'
MODEL_PATH = 'C:/ALL/Project Final year/eye and mouth ML HAAR/models/eye_state_model.h5'
ALARM_SOUND_PATH = 'C:/ALL/Project Final year/eye and mouth ML HAAR/alarm.wav'

# Initialize the alarm sound
mixer.init()
alarm_sound = mixer.Sound(ALARM_SOUND_PATH)

# Load the Haar Cascade files for face and eye detection
face_detection = cv2.CascadeClassifier(HAAR_CASCADE_FACE)
left_eye_detection = cv2.CascadeClassifier(HAAR_CASCADE_LEFT_EYE)
right_eye_detection = cv2.CascadeClassifier(HAAR_CASCADE_RIGHT_EYE)

# Load the pre-trained model
model = load_model(MODEL_PATH)

# Initialize video capture (webcam)
capture = cv2.VideoCapture(0)

if not capture.isOpened():
    raise IOError("Cannot open webcam")

while True:
    # Read a frame from the webcam
    ret, frame = capture.read()
    
    if not ret:
        print("Failed to grab frame")
        break

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the frame
    faces = face_detection.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    # Iterate through the detected faces
    for (x, y, w, h) in faces:
        # Extract the face region from the frame
        face_region = gray[y:y+h, x:x+w]
        
        # Detect left and right eyes in the face region
        left_eyes = left_eye_detection.detectMultiScale(face_region)
        right_eyes = right_eye_detection.detectMultiScale(face_region)
        
        # Check if both eyes are detected
        if len(left_eyes) > 0 and len(right_eyes) > 0:
            # Extract the eye regions
            left_eye_region = face_region[left_eyes[0][1]:left_eyes[0][1]+left_eyes[0][3], left_eyes[0][0]:left_eyes[0][0]+left_eyes[0][2]]
            right_eye_region = face_region[right_eyes[0][1]:right_eyes[0][1]+right_eyes[0][3], right_eyes[0][0]:right_eyes[0][0]+right_eyes[0][2]]
            
            # Preprocess the eye regions
            left_eye_region = cv2.resize(left_eye_region, (24, 24))
            right_eye_region = cv2.resize(right_eye_region, (24, 24))
            left_eye_region = left_eye_region / 255.0
            right_eye_region = right_eye_region / 255.0
            
            # Reshape for model input (add a batch dimension and channel dimension)
            left_eye_region = np.expand_dims(np.expand_dims(left_eye_region, axis=-1), axis=0)
            right_eye_region = np.expand_dims(np.expand_dims(right_eye_region, axis=-1), axis=0)
            
            # Make predictions for each eye
            left_prediction = model.predict(left_eye_region)
            right_prediction = model.predict(right_eye_region)
            
            # Threshold to determine if eyes are closed (assuming 0 for closed, 1 for open)
            left_eye_closed = left_prediction[0][0] > 0.5
            right_eye_closed = right_prediction[0][0] > 0.5
            
            # If both eyes are closed, sound the alarm
            if left_eye_closed and right_eye_closed:
                alarm_sound.play()
                print("Both eyes are closed!")
            else:
                print("Eyes are open.")
    
    # Display the output
    cv2.imshow('Eye State Detection', frame)
    
    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
capture.release()
cv2.destroyAllWindows()
