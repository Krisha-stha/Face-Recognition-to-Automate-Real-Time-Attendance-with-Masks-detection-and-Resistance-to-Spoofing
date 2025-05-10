from tkinter import *
from tkinter import ttk
from PIL import Image, ImageTk
from tkinter import messagebox
import mysql.connector
from time import strftime
from datetime import datetime
import cv2
import os
import numpy as np
from scipy.spatial import distance as dist
from imutils import face_utils
import dlib
from keras.applications.mobilenet_v2 import preprocess_input
from keras.utils import img_to_array
from keras.models import load_model
from imutils.video import VideoStream
import imutils


class Face_Recognition:
    def __init__(self,root):
        self.root=root
        self.root.geometry("1530x790+0+0")
        self.root.title("Face Recognition System")
 
        title_lbl = Label(self.root, text = "Face recognition", font = ("times new roman", 35, "bold"),bg = "white", fg = "pink")
        title_lbl.place(x = 0, y = 0, width = 1530, height = 60)

        #second image
        img_bottom = Image.open(r"my_images\details.jpeg")
        img_bottom = img_bottom.resize((1800, 700),Image.ANTIALIAS)
        self.photoimg_bottom=ImageTk.PhotoImage(img_bottom)
 
        f_lbl = Label(self.root,image=self.photoimg_bottom)
        f_lbl.place(x=0,y=60,width=1800,height=700) 
 
        #button
        facedetect = Image.open(r"my_images\facedetector.jpeg")
        facedetect = facedetect.resize((220, 220),Image.ANTIALIAS)
        self.detectface=ImageTk.PhotoImage(facedetect)

        button = Button(self.root,image=self.detectface, cursor="hand2",command=self.face_data)
        button.place(x=500, y=100, width=220, height=220)

        button1 = Button(f_lbl,text="Face Recognition with spoof", command=self.face_recog, cursor="hand2", font = ("times new roman", 18, "bold"),bg = "white", fg = "pink")
        button1.place(x=200, y=300, width=350, height=40)

        #button
        button2 = Button(f_lbl,text="Face Recognition with mask", command=self.face_recogs, cursor="hand2", font = ("times new roman", 18, "bold"),bg = "white", fg = "pink")
        button2.place(x=800, y=300, width=350, height=40)

        #button
        button3 = Button(f_lbl,text="Face Recognition without mask", command=self.face_recog_without_mask, cursor="hand2", font = ("times new roman", 18, "bold"),bg = "white", fg = "pink")
        button3.place(x=200, y=600, width=350, height=40)

        #button
        button4 = Button(f_lbl,text="Mask Detection", command=self.face_mask_detection, cursor="hand2", font = ("times new roman", 18, "bold"),bg = "white", fg = "pink")
        button4.place(x=900, y=600, width=200, height=40)

        # Load mask detector models
        prototxtPath = r"face_detector\deploy.prototxt"
        weightsPath = r"face_detector\res10_300x300_ssd_iter_140000.caffemodel"
        self.faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)
        self.maskNet = load_model("mask_detector.model")

    def eye_aspect_ratio(self, eye):
        A = dist.euclidean(eye[1], eye[5])
        B = dist.euclidean(eye[2], eye[4])
        C = dist.euclidean(eye[0], eye[3])
        ear = (A + B) / (2.0 * C)
        return ear

    def mark_attendance(self, i, r, n, d):
        with open("todaysattendance.csv", "r+", newline="\n") as f:
            myDataList = f.readlines()
            name_list = []
            for line in myDataList:
                entry = line.split((","))
                name_list.append(entry[0])
            if((i not in name_list) and (r not in name_list) and (n not in name_list) and (d not in name_list)):
                now = datetime.now()
                d1 = now.strftime("%d/%m/%Y")
                dtString = now.strftime("%H:%M:%S")
                f.writelines(f"\n{i},{r},{n},{d},{dtString},{d1},Present")

    def face_recog(self):
        # Initialize blink detection variables
        EYE_AR_THRESH = 0.2
        EYE_AR_CONSEC_FRAMES = 1
        COUNTER = 0
        TOTAL = 0
        
        # Initialize detectors
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
        eye_cascade = cv2.CascadeClassifier("haarcascade_eye.xml")
        
        # Load recognizer
        clf = cv2.face.LBPHFaceRecognizer_create()
        clf.read("classifier.xml")
        
        # Get facial landmarks indexes
        (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
        
        video_cap = cv2.VideoCapture(0)
        
        while True:
            ret, img = video_cap.read()
            if not ret:
                break
                
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            
            # Reset blink counter if no faces detected
            if len(faces) == 0:
                TOTAL = 0
                COUNTER = 0
            
            for (x, y, w, h) in faces:
                # Draw rectangle (green for recognized, red for unknown)
                color = (0, 255, 0)  # Default to green
                cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
                
                # Eye detection for anti-spoofing
                roi_gray = gray[y:y+h, x:x+w]
                eyes = eye_cascade.detectMultiScale(roi_gray)
                
                # Blink detection using dlib
                rects = detector(gray, 0)
                for rect in rects:
                    shape = predictor(gray, rect)
                    shape = face_utils.shape_to_np(shape)
                    
                    leftEye = shape[lStart:lEnd]
                    rightEye = shape[rStart:rEnd]
                    leftEAR = self.eye_aspect_ratio(leftEye)
                    rightEAR = self.eye_aspect_ratio(rightEye)
                    ear = (leftEAR + rightEAR) / 2.0
                    
                    if ear < EYE_AR_THRESH:
                        COUNTER += 1
                    else:
                        if COUNTER >= EYE_AR_CONSEC_FRAMES:
                            TOTAL += 1
                        COUNTER = 0
                
                # Only attempt recognition if user has blinked
                if TOTAL >= 1:
                    gray_face = gray[y:y+h, x:x+w]
                    id, confidence = clf.predict(gray_face)
                    
                    if confidence < 60:  # Recognized face
                        conn = mysql.connector.connect(
                            host="localhost", 
                            user="root", 
                            password="krisha123", 
                            database="face_recognition"
                        )
                        my_cursor = conn.cursor()
                        
                        # Get student details
                        my_cursor.execute("SELECT Name FROM student WHERE Student_id=" + str(id))
                        n = my_cursor.fetchone()
                        n = "+".join(n) if n else "Unknown"
                        
                        my_cursor.execute("SELECT Roll FROM student WHERE Student_id=" + str(id))
                        r = my_cursor.fetchone()
                        r = "+".join(r) if r else "Unknown"
                        
                        my_cursor.execute("SELECT Dep FROM student WHERE Student_id=" + str(id))
                        d = my_cursor.fetchone()
                        d = "+".join(d) if d else "Unknown"
                        
                        my_cursor.execute("SELECT Student_id FROM student WHERE Student_id=" + str(id))
                        i = my_cursor.fetchone()
                        i = "+".join(i) if i else "Unknown"
                        
                        # Display info
                        cv2.putText(img, f"ID: {i}", (x, y-75), cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 255, 255), 3)
                        cv2.putText(img, f"Roll: {r}", (x, y-55), cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 255, 255), 3)
                        cv2.putText(img, f"Name: {n}", (x, y-30), cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 255, 255), 3)
                        cv2.putText(img, f"Department: {d}", (x, y-5), cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 255, 255), 3)
                        
                        self.mark_attendance(i, r, n, d)
                    else:  # Unknown face
                        color = (0, 0, 255)  # Red for unknown
                        cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
                        cv2.putText(img, "Unknown Face", (x, y-5), cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 255, 255), 3)
                else:
                    cv2.putText(img, "Blink to verify", (x, y-5), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 0, 255), 2)
                
                # Display blink info
                cv2.putText(img, f"Blinks: {TOTAL}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            cv2.imshow("Face Recognition with Anti-Spoofing", img)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break
        
        video_cap.release()
        cv2.destroyAllWindows()


    def eye_aspect_ratios(self, eye):
        A = dist.euclidean(eye[1], eye[5])
        B = dist.euclidean(eye[2], eye[4])
        C = dist.euclidean(eye[0], eye[3])
        ear = (A + B) / (2.0 * C)
        return ear

    def mark_attendances(self, i, r, n, d):
        with open("todaysattendance.csv", "r+", newline="\n") as f:
            myDataList = f.readlines()
            name_list = []
            for line in myDataList:
                entry = line.split((","))
                name_list.append(entry[0])
            if((i not in name_list) and (r not in name_list) and (n not in name_list) and (d not in name_list)):
                now = datetime.now()
                d1 = now.strftime("%d/%m/%Y")
                dtString = now.strftime("%H:%M:%S")
                f.writelines(f"\n{i},{r},{n},{d},{dtString},{d1},Present")

    def detect_masks(self, frame):
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224), (104.0, 177.0, 123.0))
        self.faceNet.setInput(blob)
        detections = self.faceNet.forward()
        
        faces = []
        locs = []
        preds = []

        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                (startX, startY) = (max(0, startX), max(0, startY))
                (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

                face = frame[startY:endY, startX:endX]
                face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                face = cv2.resize(face, (224, 224))
                face = img_to_array(face)
                face = preprocess_input(face)

                faces.append(face)
                locs.append((startX, startY, endX, endY))

        if len(faces) > 0:
            faces = np.array(faces, dtype="float32")
            preds = self.maskNet.predict(faces, batch_size=32)

        return (locs, preds)

    def face_recogs(self):
        # Initialize blink detection variables
        EYE_AR_THRESH = 0.2
        EYE_AR_CONSEC_FRAMES = 1
        COUNTER = 0
        TOTAL = 0
        mask_detected = False
        
        # Initialize detectors
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
        
        # Load recognizer
        clf = cv2.face.LBPHFaceRecognizer_create()
        clf.read("classifier.xml")
        
        # Get facial landmarks indexes
        (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
        
        video_cap = cv2.VideoCapture(0)
        
        while True:
            ret, img = video_cap.read()
            if not ret:
                break
                
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            
            # Check for masks first
            (mask_locs, mask_preds) = self.detect_masks(img)
            
            # Reset blink counter if no faces detected
            if len(faces) == 0:
                TOTAL = 0
                COUNTER = 0
            
            # Check if any face has a mask
            mask_detected = False
            for (box, pred) in zip(mask_locs, mask_preds):
                (startX, startY, endX, endY) = box
                (mask, withoutMask) = pred
                
                if mask < withoutMask:  # No mask detected
                    mask_detected = True
                    cv2.rectangle(img, (startX, startY), (endX, endY), (0, 0, 255), 2)
                    cv2.putText(img, "Please wear mask", (startX, startY - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
                    break
            
            # If mask is detected, skip face recognition
            if mask_detected:
                cv2.imshow("Face Recognition with Anti-Spoofing", img)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:
                    break
                continue
            
            # If no mask detected, proceed with face recognition
            for (x, y, w, h) in faces:
                # Draw rectangle (green for recognized, red for unknown)
                color = (0, 255, 0)  # Default to green
                cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
                
                # Blink detection using dlib
                rects = detector(gray, 0)
                for rect in rects:
                    shape = predictor(gray, rect)
                    shape = face_utils.shape_to_np(shape)
                    
                    leftEye = shape[lStart:lEnd]
                    rightEye = shape[rStart:rEnd]
                    leftEAR = self.eye_aspect_ratios(leftEye)
                    rightEAR = self.eye_aspect_ratios(rightEye)
                    ear = (leftEAR + rightEAR) / 2.0
                    
                    if ear < EYE_AR_THRESH:
                        COUNTER += 1
                    else:
                        if COUNTER >= EYE_AR_CONSEC_FRAMES:
                            TOTAL += 1
                        COUNTER = 0
                
                # Only attempt recognition if user has blinked
                if TOTAL >= 1:
                    gray_face = gray[y:y+h, x:x+w]
                    id, confidence = clf.predict(gray_face)
                    
                    if confidence < 60:  # Recognized face
                        conn = mysql.connector.connect(
                            host="localhost", 
                            user="root", 
                            password="krisha123", 
                            database="face_recognition"
                        )
                        my_cursor = conn.cursor()
                        
                        # Get student details
                        my_cursor.execute("SELECT Name FROM student WHERE Student_id=" + str(id))
                        n = my_cursor.fetchone()
                        n = "+".join(n) if n else "Unknown"
                        
                        my_cursor.execute("SELECT Roll FROM student WHERE Student_id=" + str(id))
                        r = my_cursor.fetchone()
                        r = "+".join(r) if r else "Unknown"
                        
                        my_cursor.execute("SELECT Dep FROM student WHERE Student_id=" + str(id))
                        d = my_cursor.fetchone()
                        d = "+".join(d) if d else "Unknown"
                        
                        my_cursor.execute("SELECT Student_id FROM student WHERE Student_id=" + str(id))
                        i = my_cursor.fetchone()
                        i = "+".join(i) if i else "Unknown"
                        
                        # Display info
                        cv2.putText(img, f"ID: {i}", (x, y-75), cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 255, 255), 3)
                        cv2.putText(img, f"Roll: {r}", (x, y-55), cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 255, 255), 3)
                        cv2.putText(img, f"Name: {n}", (x, y-30), cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 255, 255), 3)
                        cv2.putText(img, f"Department: {d}", (x, y-5), cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 255, 255), 3)
                        
                        self.mark_attendances(i, r, n, d)
                    else:  # Unknown face
                        color = (0, 0, 255)  # Red for unknown
                        cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
                        cv2.putText(img, "Unknown Face", (x, y-5), cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 255, 255), 3)
                else:
                    cv2.putText(img, "Blink to verify", (x, y-5), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 0, 255), 2)
                
                # Display blink info
                cv2.putText(img, f"Blinks: {TOTAL}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            cv2.imshow("Face Recognition with Anti-Spoofing", img)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break
        
        video_cap.release()
        cv2.destroyAllWindows()

    def detect_mask_again(self, frame):
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224), (104.0, 177.0, 123.0))
        self.faceNet.setInput(blob)
        detections = self.faceNet.forward()
        
        faces = []
        locs = []
        preds = []

        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                (startX, startY) = (max(0, startX), max(0, startY))
                (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

                face = frame[startY:endY, startX:endX]
                face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                face = cv2.resize(face, (224, 224))
                face = img_to_array(face)
                face = preprocess_input(face)

                faces.append(face)
                locs.append((startX, startY, endX, endY))

        if len(faces) > 0:
            faces = np.array(faces, dtype="float32")
            preds = self.maskNet.predict(faces, batch_size=32)

        return (locs, preds)

    def face_recog_without_mask(self):
        # Initialize blink detection variables
        EYE_AR_THRESH = 0.2
        EYE_AR_CONSEC_FRAMES = 1
        COUNTER = 0
        TOTAL = 0
        mask_detected = False
        
        # Initialize detectors
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
        
        # Load recognizer
        clf = cv2.face.LBPHFaceRecognizer_create()
        clf.read("classifier.xml")
        
        # Get facial landmarks indexes
        (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
        
        video_cap = cv2.VideoCapture(0)
        
        while True:
            ret, img = video_cap.read()
            if not ret:
                break
                
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            
            # Check for masks first
            (mask_locs, mask_preds) = self.detect_mask_again(img)
            
            # Reset blink counter if no faces detected
            if len(faces) == 0:
                TOTAL = 0
                COUNTER = 0
            
            # Check if any face has a mask
            mask_detected = False
            for (box, pred) in zip(mask_locs, mask_preds):
                (startX, startY, endX, endY) = box
                (mask, withoutMask) = pred
                
                # CORRECTED LOGIC: Show "Please remove mask" only when mask IS detected
                if mask > withoutMask:  # Mask detected
                    mask_detected = True
                    cv2.rectangle(img, (startX, startY), (endX, endY), (0, 0, 255), 2)
                    cv2.putText(img, "Please remove mask", (startX, startY - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
                    break
            
            # If mask is detected, skip face recognition
            if mask_detected:
                cv2.imshow("Face Recognition with Anti-Spoofing", img)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:
                    break
                continue
            
            # If no mask detected, proceed with face recognition
            for (x, y, w, h) in faces:
                # Draw rectangle (green for recognized, red for unknown)
                color = (0, 255, 0)  # Default to green
                cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
                
                # Blink detection using dlib
                rects = detector(gray, 0)
                for rect in rects:
                    shape = predictor(gray, rect)
                    shape = face_utils.shape_to_np(shape)
                    
                    leftEye = shape[lStart:lEnd]
                    rightEye = shape[rStart:rEnd]
                    leftEAR = self.eye_aspect_ratios(leftEye)
                    rightEAR = self.eye_aspect_ratios(rightEye)
                    ear = (leftEAR + rightEAR) / 2.0
                    
                    if ear < EYE_AR_THRESH:
                        COUNTER += 1
                    else:
                        if COUNTER >= EYE_AR_CONSEC_FRAMES:
                            TOTAL += 1
                        COUNTER = 0
                
                # Only attempt recognition if user has blinked
                if TOTAL >= 1:
                    gray_face = gray[y:y+h, x:x+w]
                    id, confidence = clf.predict(gray_face)
                    
                    if confidence < 60:  # Recognized face
                        conn = mysql.connector.connect(
                            host="localhost", 
                            user="root", 
                            password="krisha123", 
                            database="face_recognition"
                        )
                        my_cursor = conn.cursor()
                        
                        # Get student details
                        my_cursor.execute("SELECT Name FROM student WHERE Student_id=" + str(id))
                        n = my_cursor.fetchone()
                        n = "+".join(n) if n else "Unknown"
                        
                        my_cursor.execute("SELECT Roll FROM student WHERE Student_id=" + str(id))
                        r = my_cursor.fetchone()
                        r = "+".join(r) if r else "Unknown"
                        
                        my_cursor.execute("SELECT Dep FROM student WHERE Student_id=" + str(id))
                        d = my_cursor.fetchone()
                        d = "+".join(d) if d else "Unknown"
                        
                        my_cursor.execute("SELECT Student_id FROM student WHERE Student_id=" + str(id))
                        i = my_cursor.fetchone()
                        i = "+".join(i) if i else "Unknown"
                        
                        # Display info
                        cv2.putText(img, f"ID: {i}", (x, y-75), cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 255, 255), 3)
                        cv2.putText(img, f"Roll: {r}", (x, y-55), cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 255, 255), 3)
                        cv2.putText(img, f"Name: {n}", (x, y-30), cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 255, 255), 3)
                        cv2.putText(img, f"Department: {d}", (x, y-5), cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 255, 255), 3)
                        
                        self.mark_attendances(i, r, n, d)
                    else:  # Unknown face
                        color = (0, 0, 255)  # Red for unknown
                        cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
                        cv2.putText(img, "Unknown Face", (x, y-5), cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 255, 255), 3)
                else:
                    cv2.putText(img, "Blink to verify", (x, y-5), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 0, 255), 2)
                
                # Display blink info
                cv2.putText(img, f"Blinks: {TOTAL}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            cv2.imshow("Face Recognition with Anti-Spoofing", img)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break
        
        video_cap.release()
        cv2.destroyAllWindows()

    def face_mask_detection(self):
        # initialize the video stream
        print("[INFO] starting video stream...")
        vs = VideoStream(src=0).start()

        # loop over the frames from the video stream
        while True:
            # grab the frame from the threaded video stream and resize it
            # to have a maximum width of 400 pixels
            frame = vs.read()
            frame = imutils.resize(frame, width=400)

            # detect faces in the frame and determine if they are wearing a
            # face mask or not
            (locs, preds) = self.detect_masks(frame)

            # loop over the detected face locations and their corresponding
            # locations
            for (box, pred) in zip(locs, preds):
                # unpack the bounding box and predictions
                (startX, startY, endX, endY) = box
                (mask, withoutMask) = pred

                # determine the class label and color we'll use to draw
                # the bounding box and text
                label = "Mask" if mask > withoutMask else "No Mask"
                color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

                # include the probability in the label
                label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

                # display the label and bounding box rectangle on the output
                # frame
                cv2.putText(frame, label, (startX, startY - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
                cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

            # show the output frame
            cv2.imshow("Mask Detection", frame)
            key = cv2.waitKey(1) & 0xFF

            # if the `q` key was pressed, break from the loop
            if key == ord("q"):
                break

        # do a bit of cleanup
        cv2.destroyAllWindows()
        vs.stop()


if __name__ == "__main__":
    root = Tk()
    obj = Face_Recognition(root)
    root.mainloop()
 