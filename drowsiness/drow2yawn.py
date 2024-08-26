from scipy.spatial import distance
from imutils import face_utils
from pygame import mixer
import imutils
import dlib
import cv2

# Initialize mixer for alert sound
mixer.init()
mixer.music.load("drowsiness/music.wav")

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

# Parameters
ear_thresh = 0.25  # EAR threshold for eyes
mar_thresh = 0.7   # MAR threshold for mouth (yawning)
frame_check = 20   # Number of consecutive frames the EAR or MAR should be below or above threshold

# Initialize dlib's face detector and facial landmark predictor
detect = dlib.get_frontal_face_detector()
predict = dlib.shape_predictor("drowsiness/shape_predictor_68_face_landmarks.dat")

# Get the indexes of the facial landmarks for eyes and mouth
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]
(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["mouth"]

cap = cv2.VideoCapture(0)
flag = 0

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

        # Draw contours around the eyes and mouth
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        mouthHull = cv2.convexHull(mouth)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [mouthHull], -1, (255, 0, 0), 1)

        # Check if EAR is below the threshold or MAR is above the threshold
        if ear < ear_thresh or mar > mar_thresh:
            flag += 1
            if flag >= frame_check:
                cv2.putText(frame, "****************ALERT!****************", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                if not mixer.music.get_busy():  # Play sound only if not already playing
                    mixer.music.play()
        else:
            flag = 0
            mixer.music.stop()  # Stop the alert sound when the person is not drowsy or yawning

        # Display the EAR and MAR values on the screen
        cv2.putText(frame, f"EAR: {ear:.2f} MAR: {mar:.2f}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

# Clean up
cv2.destroyAllWindows()
cap.release()
