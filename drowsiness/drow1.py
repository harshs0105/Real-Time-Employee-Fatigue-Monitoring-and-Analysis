from scipy.spatial import distance
from imutils import face_utils
from pygame import mixer
import imutils
import dlib
import cv2
import threading
import time
import csv

# Initialize mixer for alert sound
mixer.init()
mixer.music.load("drowsiness/music.wav")

def play_alert():
    mixer.music.play()

def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Parameters
thresh = 0.25  # EAR threshold
frame_check = 20  # Number of consecutive frames the EAR should be below threshold
alert_cooldown = 10  # Cooldown period in seconds to prevent repeated alerts

# Initialize dlib's face detector and facial landmark predictor
detect = dlib.get_frontal_face_detector()
predict = dlib.shape_predictor("drowsiness/shape_predictor_68_face_landmarks.dat")

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]

cap = cv2.VideoCapture(0)
flag = 0
last_alert_time = 0

# Open CSV file for logging
with open('drowsiness_log.csv', mode='w', newline='') as log_file:
    log_writer = csv.writer(log_file)
    log_writer.writerow(["Timestamp", "EAR", "Drowsiness Detected"])

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

            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]
            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)
            ear = (leftEAR + rightEAR) / 2.0

            # Draw contours around the eyes
            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

            # Check if EAR is below the threshold
            if ear < thresh:
                flag += 1
                if flag >= frame_check:
                    current_time = time.time()
                    if current_time - last_alert_time > alert_cooldown:
                        cv2.putText(frame, "****************ALERT!****************", (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        cv2.putText(frame, "****************ALERT!****************", (10, 325),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        threading.Thread(target=play_alert).start()
                        last_alert_time = current_time
                    log_writer.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), ear, "Yes"])
            else:
                flag = 0
                log_writer.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), ear, "No"])

            # Display the EAR value on the screen
            cv2.putText(frame, f"EAR: {ear:.2f}", (300, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

# Clean up
cv2.destroyAllWindows()
cap.release()
