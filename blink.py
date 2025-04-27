import cv2
import dlib
import numpy as np
from scipy.spatial import distance as dist
from imutils import face_utils
import imutils
import time

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Initialize detectors
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Blink detection parameters
EYE_AR_THRESH = 0.25
EYE_AR_CONSEC_FRAMES = 3
COUNTER = 0
TOTAL = 0

# Liveness detection parameters
MOTION_THRESHOLD = 10000  # Adjust based on testing
prev_frame = None
active_frames = 0
REQUIRED_ACTIVE_FRAMES = 10  # Need 10 frames with motion

# Start video stream
vs = cv2.VideoCapture(0)
time.sleep(1.0)

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

while True:
    ret, frame = vs.read()
    if not ret:
        break
        
    frame = imutils.resize(frame, width=800)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Motion detection
    motion_detected = False
    if prev_frame is not None:
        frame_diff = cv2.absdiff(prev_frame, gray)
        _, thresh = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)
        motion_score = np.sum(thresh)
        motion_detected = motion_score > MOTION_THRESHOLD
        
        if motion_detected:
            active_frames = min(active_frames + 1, REQUIRED_ACTIVE_FRAMES)
        else:
            active_frames = max(active_frames - 1, 0)
    
    prev_frame = gray.copy()
    
    # Only detect blinks if live motion is confirmed
    if active_frames >= REQUIRED_ACTIVE_FRAMES:
        rects = detector(gray, 0)
        
        for rect in rects:
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)
            
            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]
            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)
            ear = (leftEAR + rightEAR) / 2.0
            
            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
            
            if ear < EYE_AR_THRESH:
                COUNTER += 1
            else:
                if COUNTER >= EYE_AR_CONSEC_FRAMES:
                    TOTAL += 1
                COUNTER = 0
            
            cv2.putText(frame, f"Blinks: {TOTAL}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, f"EAR: {ear:.2f}", (300, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    else:
        cv2.putText(frame, "Move to activate detection", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # Display liveness status
    liveness_status = f"Liveness: {active_frames}/{REQUIRED_ACTIVE_FRAMES}"
    cv2.putText(frame, liveness_status, (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    cv2.imshow("Improved Blink Detection", frame)
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord("q"):
        break

vs.release()
cv2.destroyAllWindows()