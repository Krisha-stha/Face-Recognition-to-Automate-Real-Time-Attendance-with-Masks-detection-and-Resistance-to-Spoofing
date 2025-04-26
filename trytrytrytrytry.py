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
import imutils
import mediapipe as mp

class Face_Recognition:
    def __init__(self,root):
        self.root=root
        self.root.geometry("1530x790+0+0")
        self.root.title("Face Recognition System")
        
        # Blink detection variables
        self.blink_threshold = 0.21
        self.closed_frames = 0
        self.blink_count = 0
        self.last_attendance_marked = None  # To prevent duplicate entries
        
        # Initialize MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False, 
            max_num_faces=1, 
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Eye landmarks indices
        self.left_eye = [33, 160, 158, 133, 153, 144]
        self.right_eye = [362, 385, 387, 263, 373, 380]

        title_lbl = Label(self.root, text="Face recognition", font=("times new roman", 35, "bold"), bg="white", fg="pink")
        title_lbl.place(x=0, y=0, width=1530, height=60)

        # First image
        img_top = Image.open(r"my_images\details.jpeg")
        img_top = img_top.resize((650, 700), Image.ANTIALIAS)
        self.photoimg_top = ImageTk.PhotoImage(img_top)

        f_lbl = Label(self.root, image=self.photoimg_top)
        f_lbl.place(x=0, y=60, width=650, height=700) 

        # Second image
        img_bottom = Image.open(r"my_images\details.jpeg")
        img_bottom = img_bottom.resize((950, 700), Image.ANTIALIAS)
        self.photoimg_bottom = ImageTk.PhotoImage(img_bottom)

        f_lbl = Label(self.root, image=self.photoimg_bottom)
        f_lbl.place(x=650, y=60, width=950, height=700) 

        # Button
        button1 = Button(f_lbl, text="Face Recognition", command=self.face_recog, cursor="hand2", 
                        font=("times new roman", 18, "bold"), bg="white", fg="pink")
        button1.place(x=350, y=600, width=200, height=40)

    def get_eye_aspect_ratio(self, landmarks, eye_indices):
        top = (landmarks[eye_indices[1]].y + landmarks[eye_indices[2]].y) / 2
        bottom = (landmarks[eye_indices[4]].y + landmarks[eye_indices[5]].y) / 2
        vertical = abs(top - bottom)
        horizontal = abs(landmarks[eye_indices[0]].x - landmarks[eye_indices[3]].x)
        return vertical / horizontal if horizontal != 0 else 0

    def mark_attendance(self, i, r, n, d):
        # Check if attendance was already marked for this person in this session
        current_id = f"{i}_{r}_{n}_{d}"
        if self.last_attendance_marked == current_id:
            return False
            
        with open("todaysattendance.csv", "r+", newline="\n") as f:
            myDataList = f.readlines()
            name_list = []
            for line in myDataList:
                entry = line.split((","))
                if len(entry) > 0:  # Check if line has content
                    name_list.append(entry[0])
            
            if((i not in name_list) and (r not in name_list) and (n not in name_list) and (d not in name_list)):
                now = datetime.now()
                d1 = now.strftime("%d/%m/%Y")
                dtString = now.strftime("%H:%M:%S")
                f.writelines(f"\n{i},{r},{n},{d},{dtString},{d1},Present")
                self.last_attendance_marked = current_id
                return True
        return False

    def face_recog(self):
        def draw_boundary(img, classifier, scaleFactor, minNeighbors, color, text, clf):
            gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            features = classifier.detectMultiScale(gray_image, scaleFactor, minNeighbors)

            coords = []
            user_data = None

            for (x, y, w, h) in features:
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)
                id, predict = clf.predict(gray_image[y:y + h, x:x + w])
                confidence = int((100 * (1 - predict / 300)))

                try:
                    conn = mysql.connector.connect(host="localhost", user="root", password="krisha123", database="face_recognition")
                    my_cursor = conn.cursor()

                    my_cursor.execute("select Name from student where Student_id=" + str(id))
                    n_result = my_cursor.fetchone()
                    n = "+".join(n_result) if n_result else "Unknown"

                    my_cursor.execute("select Roll from student where Student_id=" + str(id))
                    r_result = my_cursor.fetchone()
                    r = "+".join(r_result) if r_result else "Unknown"

                    my_cursor.execute("select Dep from student where Student_id=" + str(id))
                    d_result = my_cursor.fetchone()
                    d = "+".join(d_result) if d_result else "Unknown"

                    my_cursor.execute("select Student_id from student where Student_id=" + str(id))
                    i_result = my_cursor.fetchone()
                    i = "+".join(i_result) if i_result else "Unknown"

                    if confidence > 77:
                        cv2.putText(img, f"ID:{i}", (x, y - 75), cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 255, 255), 3)
                        cv2.putText(img, f"Roll:{r}", (x, y - 55), cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 255, 255), 3)
                        cv2.putText(img, f"Name:{n}", (x, y - 30), cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 255, 255), 3)
                        cv2.putText(img, f"Department:{d}", (x, y - 5), cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 255, 255), 3)

                        coords = [x, y, w, h]
                        user_data = (i, r, n, d)
                    else:
                        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 3)
                        cv2.putText(img, "Unknown Face", (x, y - 5), cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 255, 255), 3)

                    conn.close()
                except Exception as e:
                    print(f"Database error: {e}")

            return coords, user_data

        def recognize(img, clf, faceCascade):
            # Face recognition
            coords, user_data = draw_boundary(img, faceCascade, 1.1, 10, (255, 255, 255), "Face", clf)
            
            # Blink detection with MediaPipe
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(img_rgb)
            
            blink_detected = False
            
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    lm = face_landmarks.landmark
                    
                    left_ear = self.get_eye_aspect_ratio(lm, self.left_eye)
                    right_ear = self.get_eye_aspect_ratio(lm, self.right_eye)
                    ear = (left_ear + right_ear) / 2
                    
                    # Blink detection logic
                    if ear < self.blink_threshold:
                        self.closed_frames += 1
                    else:
                        if self.closed_frames >= 3:  # Minimum frames for a blink
                            self.blink_count += 1
                            blink_detected = True
                            
                        self.closed_frames = 0
                    
                    # Display blink information
                    cv2.putText(img, f"EAR: {ear:.2f}", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(img, f"Blinks: {self.blink_count}", (10, 60), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                    
                    # Draw eye landmarks
                    mp.solutions.drawing_utils.draw_landmarks(
                        image=img,
                        landmark_list=face_landmarks,
                        connections=self.mp_face_mesh.FACEMESH_CONTOURS,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp.solutions.drawing_styles
                        .get_default_face_mesh_contours_style())
            
            # Mark attendance if face is recognized and blink detected
            if user_data and blink_detected:
                i, r, n, d = user_data
                if self.mark_attendance(i, r, n, d):
                    cv2.putText(img, "ATTENDANCE MARKED", (10, 90), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            return img

        # Initialize face detection
        faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
        clf = cv2.face.LBPHFaceRecognizer_create()
        clf.read("classifier.xml")

        # Reset counters
        self.closed_frames = 0
        self.blink_count = 0
        self.last_attendance_marked = None

        video_cap = cv2.VideoCapture(0)

        while True:
            ret, img = video_cap.read()
            if not ret:
                break
                
            img = recognize(img, clf, faceCascade)
            cv2.imshow("Face Recognition with Spoof Prevention", img)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break

        video_cap.release()
        cv2.destroyAllWindows()
        self.face_mesh.close()

if __name__ == "__main__":
    root = Tk()
    obj = Face_Recognition(root)
    root.mainloop()