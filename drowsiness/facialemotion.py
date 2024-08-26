import cv2
import numpy as np
import dlib
from imutils import face_utils

# Load the pre-trained emotion recognition model (Example: Caffe model)
#net = cv2.dnn.readNetFromCaffe("facialemotionmodel.h5")
from keras.models import load_model

net = load_model("facialemotionmodel.h5")
# Initialize dlib's face detector and shape predictor
detect = dlib.get_frontal_face_detector()
predict = dlib.shape_predictor("drowsiness/shape_predictor_68_face_landmarks.dat")

# Emotion labels (make sure they match the labels of your model)
emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

def detect_emotion(face_image):
    face_image = cv2.resize(face_image, (48, 48))  # Resize to 48x48
    face_image = face_image / 255.0
    face_image = np.expand_dims(face_image, axis=0)
    face_image = np.expand_dims(face_image, axis=-1)
    preds = net.predict(face_image)
    return emotion_labels[np.argmax(preds)]

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    subjects = detect(gray, 0)
    
    for subject in subjects:
        shape = predict(gray, subject)
        shape = face_utils.shape_to_np(shape)

        # Draw face landmarks
        for (x, y) in shape:
            cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

        # Extract face ROI for emotion recognition
        (x, y, w, h) = face_utils.rect_to_bb(subject)
        face_image = gray[y:y+h, x:x+w]
        emotion = detect_emotion(face_image)
        
        # Display the emotion on the frame
        cv2.putText(frame, f"Emotion: {emotion}", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

cv2.destroyAllWindows()
cap.release()
