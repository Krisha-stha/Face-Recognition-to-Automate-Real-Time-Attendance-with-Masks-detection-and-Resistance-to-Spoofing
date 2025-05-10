from tkinter import *
from tkinter import ttk
from PIL import Image, ImageTk
import mysql.connector
from datetime import datetime
import cv2
import numpy as np
from scipy.spatial import distance as dist
from imutils import face_utils
import dlib
from keras.applications.mobilenet_v2 import preprocess_input
from keras.utils import img_to_array
from keras.models import load_model
from imutils.video import VideoStream
import imutils
import time

class Face_Recognition:
    def __init__(self, root):
        self.root = root
        self.root.geometry("1530x790+0+0")
        self.root.title("Face Recognition System")
        
        # Initialize video stream as None (will be set when needed)
        self.vs = None
        self.current_process = None
        
        # GUI Setup
        self.setup_gui()
        
        # Load models only once
        self.load_models()
        
    def setup_gui(self):
        title_lbl = Label(self.root, text="Face recognition", font=("times new roman", 35, "bold"), 
                         bg="white", fg="pink")
        title_lbl.place(x=0, y=0, width=1530, height=60)

        # Background image
        img_bottom = Image.open(r"my_images\details.jpeg")
        img_bottom = img_bottom.resize((1800, 700), Image.ANTIALIAS)
        self.photoimg_bottom = ImageTk.PhotoImage(img_bottom)
 
        f_lbl = Label(self.root, image=self.photoimg_bottom)
        f_lbl.place(x=0, y=60, width=1800, height=700) 
 
        # Buttons
        button1 = Button(f_lbl, text="Face Recognition with spoof", 
                        command=lambda: self.start_process("spoof"), 
                        cursor="hand2", font=("times new roman", 18, "bold"),
                        bg="white", fg="pink")
        button1.place(x=200, y=300, width=350, height=40)

        button2 = Button(f_lbl, text="Face Recognition with mask", 
                        command=lambda: self.start_process("with_mask"), 
                        cursor="hand2", font=("times new roman", 18, "bold"),
                        bg="white", fg="pink")
        button2.place(x=800, y=300, width=350, height=40)

        button3 = Button(f_lbl, text="Face Recognition without mask", 
                        command=lambda: self.start_process("without_mask"), 
                        cursor="hand2", font=("times new roman", 18, "bold"),
                        bg="white", fg="pink")
        button3.place(x=200, y=600, width=350, height=40)

        button4 = Button(f_lbl, text="Mask Detection", 
                        command=lambda: self.start_process("mask_only"), 
                        cursor="hand2", font=("times new roman", 18, "bold"),
                        bg="white", fg="pink")
        button4.place(x=900, y=600, width=200, height=40)
        
    def load_models(self):
        """Load all required models only once"""
        try:
            # Face detection models
            prototxtPath = r"face_detector\deploy.prototxt"
            weightsPath = r"face_detector\res10_300x300_ssd_iter_140000.caffemodel"
            self.faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)
            
            # Mask detection model
            self.maskNet = load_model("mask_detector.model")
            
            # Blink detection models
            self.detector = dlib.get_frontal_face_detector()
            self.predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
            
            # Face recognizer
            self.clf = cv2.face.LBPHFaceRecognizer_create()
            self.clf.read("classifier.xml")
            
            # Facial landmarks indexes
            (self.lStart, self.lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
            (self.rStart, self.rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
            
        except Exception as e:
            print(f"Error loading models: {e}")
            raise
            
    def start_process(self, process_type):
        """Start the selected face recognition process"""
        if self.current_process:
            self.stop_process()
            
        self.current_process = process_type
        
        if process_type == "spoof":
            self.face_recog()
        elif process_type == "with_mask":
            self.face_recogs()
        elif process_type == "without_mask":
            self.face_recog_without_mask()
        elif process_type == "mask_only":
            self.face_mask_detection()
            
    def stop_process(self):
        """Stop any running process"""
        if self.vs:
            self.vs.stop()
            cv2.destroyAllWindows()
            self.vs = None
        self.current_process = None
        
    def eye_aspect_ratio(self, eye):
        A = dist.euclidean(eye[1], eye[5])
        B = dist.euclidean(eye[2], eye[4])
        C = dist.euclidean(eye[0], eye[3])
        ear = (A + B) / (2.0 * C)
        return ear
        
    def mark_attendance(self, i, r, n, d):
        try:
            with open("todaysattendance.csv", "r+", newline="\n") as f:
                myDataList = f.readlines()
                name_list = [line.split(",")[0] for line in myDataList]
                
                if i not in name_list and r not in name_list and n not in name_list and d not in name_list:
                    now = datetime.now()
                    d1 = now.strftime("%d/%m/%Y")
                    dtString = now.strftime("%H:%M:%S")
                    f.writelines(f"\n{i},{r},{n},{d},{dtString},{d1},Present")
        except Exception as e:
            print(f"Error marking attendance: {e}")
            
    def detect_faces(self, frame):
        """Efficient face detection using DNN"""
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
        self.faceNet.setInput(blob)
        detections = self.faceNet.forward()
        
        faces = []
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:  # Confidence threshold
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                faces.append((startX, startY, endX, endY))
                
        return faces
        
    def detect_masks(self, frame, faces):
        """Detect masks for given face locations"""
        preds = []
        
        for (startX, startY, endX, endY) in faces:
            # Ensure the bounding boxes fall within the frame dimensions
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(frame.shape[1] - 1, endX), min(frame.shape[0] - 1, endY))
            
            # Extract the face ROI, convert it to RGB, resize to 224x224, and preprocess
            face = frame[startY:endY, startX:endX]
            if face.size == 0:  # Skip empty face regions
                continue
                
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)
            preds.append(face)
            
        if preds:
            preds = np.array(preds, dtype="float32")
            return self.maskNet.predict(preds, batch_size=32)
        return []
        
    def face_recog(self):
        """Face recognition with anti-spoofing (blink detection)"""
        EYE_AR_THRESH = 0.2
        EYE_AR_CONSEC_FRAMES = 1
        COUNTER = 0
        TOTAL = 0
        
        try:
            self.vs = VideoStream(src=0).start()
            time.sleep(1.0)  # Warmup camera
            
            while self.current_process == "spoof":
                frame = self.vs.read()
                if frame is None:
                    break
                    
                frame = imutils.resize(frame, width=600)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Detect faces using DNN (more accurate than Haar cascades)
                faces = self.detect_faces(frame)
                
                if not faces:
                    TOTAL = 0
                    COUNTER = 0
                    cv2.imshow("Face Recognition with Anti-Spoofing", frame)
                    if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
                        break
                    continue
                    
                for (startX, startY, endX, endY) in faces:
                    # Draw rectangle (green for recognized, red for unknown)
                    color = (0, 255, 0)
                    cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
                    
                    # Blink detection using dlib
                    rects = self.detector(gray, 0)
                    for rect in rects:
                        shape = self.predictor(gray, rect)
                        shape = face_utils.shape_to_np(shape)
                        
                        leftEye = shape[self.lStart:self.lEnd]
                        rightEye = shape[self.rStart:self.rEnd]
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
                        face_roi = gray[startY:endY, startX:endX]
                        if face_roi.size == 0:  # Skip empty regions
                            continue
                            
                        id, confidence = self.clf.predict(face_roi)
                        
                        if confidence < 60:  # Recognized face
                            try:
                                conn = mysql.connector.connect(
                                    host="localhost", 
                                    user="root", 
                                    password="krisha123", 
                                    database="face_recognition"
                                )
                                my_cursor = conn.cursor()
                                
                                # Get student details in a single query
                                query = f"SELECT Student_id, Roll, Name, Dep FROM student WHERE Student_id={id}"
                                my_cursor.execute(query)
                                result = my_cursor.fetchone()
                                
                                if result:
                                    i, r, n, d = result
                                    i, r, n, d = str(i), str(r), str(n), str(d)
                                    
                                    # Display info
                                    cv2.putText(frame, f"ID: {i}", (startX, startY-75), 
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                                    cv2.putText(frame, f"Roll: {r}", (startX, startY-55), 
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                                    cv2.putText(frame, f"Name: {n}", (startX, startY-30), 
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                                    cv2.putText(frame, f"Department: {d}", (startX, startY-5), 
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                                    
                                    self.mark_attendance(i, r, n, d)
                                else:
                                    color = (0, 0, 255)
                                    cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
                                    cv2.putText(frame, "Unknown Face", (startX, startY-5), 
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                                    
                            except Exception as e:
                                print(f"Database error: {e}")
                                color = (0, 0, 255)
                                cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
                                
                        else:  # Unknown face
                            color = (0, 0, 255)
                            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
                            cv2.putText(frame, "Unknown Face", (startX, startY-5), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                            
                    else:
                        cv2.putText(frame, "Blink to verify", (startX, startY-5), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    
                    # Display blink info
                    cv2.putText(frame, f"Blinks: {TOTAL}", (10, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                cv2.imshow("Face Recognition with Anti-Spoofing", frame)
                
                if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
                    break
                    
        finally:
            self.stop_process()
            
    def face_recogs(self):
        """Face recognition with mask detection"""
        EYE_AR_THRESH = 0.2
        EYE_AR_CONSEC_FRAMES = 1
        COUNTER = 0
        TOTAL = 0
        
        try:
            self.vs = VideoStream(src=0).start()
            time.sleep(1.0)
            
            while self.current_process == "with_mask":
                frame = self.vs.read()
                if frame is None:
                    break
                    
                frame = imutils.resize(frame, width=600)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Detect faces
                faces = self.detect_faces(frame)
                
                if not faces:
                    TOTAL = 0
                    COUNTER = 0
                    cv2.imshow("Face Recognition with Mask", frame)
                    if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
                        break
                    continue
                    
                # Check for masks
                mask_preds = self.detect_masks(frame, faces)
                
                # Check if any face doesn't have a mask
                mask_required = False
                for i, (startX, startY, endX, endY) in enumerate(faces):
                    if i < len(mask_preds):
                        (mask, withoutMask) = mask_preds[i]
                        if mask < withoutMask:  # No mask detected
                            mask_required = True
                            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)
                            cv2.putText(frame, "Please wear mask", (startX, startY-10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
                            break
                
                if mask_required:
                    cv2.imshow("Face Recognition with Mask", frame)
                    if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
                        break
                    continue
                    
                # If all faces have masks, proceed with recognition
                for i, (startX, startY, endX, endY) in enumerate(faces):
                    color = (0, 255, 0)
                    cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
                    
                    # Blink detection
                    rects = self.detector(gray, 0)
                    for rect in rects:
                        shape = self.predictor(gray, rect)
                        shape = face_utils.shape_to_np(shape)
                        
                        leftEye = shape[self.lStart:self.lEnd]
                        rightEye = shape[self.rStart:self.rEnd]
                        leftEAR = self.eye_aspect_ratio(leftEye)
                        rightEAR = self.eye_aspect_ratio(rightEye)
                        ear = (leftEAR + rightEAR) / 2.0
                        
                        if ear < EYE_AR_THRESH:
                            COUNTER += 1
                        else:
                            if COUNTER >= EYE_AR_CONSEC_FRAMES:
                                TOTAL += 1
                            COUNTER = 0
                            
                    if TOTAL >= 1:
                        face_roi = gray[startY:endY, startX:endX]
                        if face_roi.size == 0:
                            continue
                            
                        id, confidence = self.clf.predict(face_roi)
                        
                        if confidence < 60:
                            try:
                                conn = mysql.connector.connect(
                                    host="localhost", 
                                    user="root", 
                                    password="krisha123", 
                                    database="face_recognition"
                                )
                                my_cursor = conn.cursor()
                                query = f"SELECT Student_id, Roll, Name, Dep FROM student WHERE Student_id={id}"
                                my_cursor.execute(query)
                                result = my_cursor.fetchone()
                                
                                if result:
                                    i, r, n, d = result
                                    i, r, n, d = str(i), str(r), str(n), str(d)
                                    
                                    cv2.putText(frame, f"ID: {i}", (startX, startY-75), 
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                                    cv2.putText(frame, f"Roll: {r}", (startX, startY-55), 
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                                    cv2.putText(frame, f"Name: {n}", (startX, startY-30), 
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                                    cv2.putText(frame, f"Department: {d}", (startX, startY-5), 
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                                    
                                    self.mark_attendance(i, r, n, d)
                            except Exception as e:
                                print(f"Database error: {e}")
                                
                    cv2.putText(frame, f"Blinks: {TOTAL}", (10, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                cv2.imshow("Face Recognition with Mask", frame)
                if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
                    break
                    
        finally:
            self.stop_process()
            
    def face_recog_without_mask(self):
        """Face recognition that requires no mask"""
        EYE_AR_THRESH = 0.2
        EYE_AR_CONSEC_FRAMES = 1
        COUNTER = 0
        TOTAL = 0
        
        try:
            self.vs = VideoStream(src=0).start()
            time.sleep(1.0)
            
            while self.current_process == "without_mask":
                frame = self.vs.read()
                if frame is None:
                    break
                    
                frame = imutils.resize(frame, width=600)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Detect faces
                faces = self.detect_faces(frame)
                
                if not faces:
                    TOTAL = 0
                    COUNTER = 0
                    cv2.imshow("Face Recognition without Mask", frame)
                    if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
                        break
                    continue
                    
                # Check for masks
                mask_preds = self.detect_masks(frame, faces)
                
                # Check if any face has a mask
                mask_detected = False
                for i, (startX, startY, endX, endY) in enumerate(faces):
                    if i < len(mask_preds):
                        (mask, withoutMask) = mask_preds[i]
                        if mask > withoutMask:  # Mask detected
                            mask_detected = True
                            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)
                            cv2.putText(frame, "Please remove mask", (startX, startY-10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
                            break
                
                if mask_detected:
                    cv2.imshow("Face Recognition without Mask", frame)
                    if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
                        break
                    continue
                    
                # If no masks detected, proceed with recognition
                for (startX, startY, endX, endY) in faces:
                    color = (0, 255, 0)
                    cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
                    
                    # Blink detection
                    rects = self.detector(gray, 0)
                    for rect in rects:
                        shape = self.predictor(gray, rect)
                        shape = face_utils.shape_to_np(shape)
                        
                        leftEye = shape[self.lStart:self.lEnd]
                        rightEye = shape[self.rStart:self.rEnd]
                        leftEAR = self.eye_aspect_ratio(leftEye)
                        rightEAR = self.eye_aspect_ratio(rightEye)
                        ear = (leftEAR + rightEAR) / 2.0
                        
                        if ear < EYE_AR_THRESH:
                            COUNTER += 1
                        else:
                            if COUNTER >= EYE_AR_CONSEC_FRAMES:
                                TOTAL += 1
                            COUNTER = 0
                            
                    if TOTAL >= 1:
                        face_roi = gray[startY:endY, startX:endX]
                        if face_roi.size == 0:
                            continue
                            
                        id, confidence = self.clf.predict(face_roi)
                        
                        if confidence < 60:
                            try:
                                conn = mysql.connector.connect(
                                    host="localhost", 
                                    user="root", 
                                    password="krisha123", 
                                    database="face_recognition"
                                )
                                my_cursor = conn.cursor()
                                query = f"SELECT Student_id, Roll, Name, Dep FROM student WHERE Student_id={id}"
                                my_cursor.execute(query)
                                result = my_cursor.fetchone()
                                
                                if result:
                                    i, r, n, d = result
                                    i, r, n, d = str(i), str(r), str(n), str(d)
                                    
                                    cv2.putText(frame, f"ID: {i}", (startX, startY-75), 
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                                    cv2.putText(frame, f"Roll: {r}", (startX, startY-55), 
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                                    cv2.putText(frame, f"Name: {n}", (startX, startY-30), 
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                                    cv2.putText(frame, f"Department: {d}", (startX, startY-5), 
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                                    
                                    self.mark_attendance(i, r, n, d)
                            except Exception as e:
                                print(f"Database error: {e}")
                                
                    cv2.putText(frame, f"Blinks: {TOTAL}", (10, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                cv2.imshow("Face Recognition without Mask", frame)
                if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
                    break
                    
        finally:
            self.stop_process()
            
    def face_mask_detection(self):
        """Standalone mask detection"""
        try:
            self.vs = VideoStream(src=0).start()
            time.sleep(1.0)
            
            while self.current_process == "mask_only":
                frame = self.vs.read()
                if frame is None:
                    break
                    
                frame = imutils.resize(frame, width=600)
                
                # Detect faces
                faces = self.detect_faces(frame)
                
                if faces:
                    # Detect masks
                    mask_preds = self.detect_masks(frame, faces)
                    
                    for i, (startX, startY, endX, endY) in enumerate(faces):
                        if i < len(mask_preds):
                            (mask, withoutMask) = mask_preds[i]
                            
                            # Determine label and color
                            label = "Mask" if mask > withoutMask else "No Mask"
                            color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
                            
                            # Include probability
                            label = f"{label}: {max(mask, withoutMask) * 100:.2f}%"
                            
                            # Display
                            cv2.putText(frame, label, (startX, startY-10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
                            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
                
                cv2.imshow("Mask Detection", frame)
                if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
                    break
                    
        finally:
            self.stop_process()
            
    def __del__(self):
        """Destructor to ensure resources are released"""
        self.stop_process()

if __name__ == "__main__":
    root = Tk()
    obj = Face_Recognition(root)
    root.protocol("WM_DELETE_WINDOW", lambda: (obj.stop_process(), root.destroy()))
    root.mainloop()