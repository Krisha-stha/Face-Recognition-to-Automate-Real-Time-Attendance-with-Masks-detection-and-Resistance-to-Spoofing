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
import time

class Face_Recognition:
    def __init__(self, root):
        self.root = root
        self.root.geometry("1530x790+0+0")
        self.root.title("Face Recognition System")
        
        # Add protocol for window close
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        
        # Video capture object
        self.video_cap = None
        self.is_running = False
        self.prev_gray = None

        title_lbl = Label(self.root, text="Face recognition", font=("times new roman", 35, "bold"), bg="white", fg="pink")
        title_lbl.place(x=0, y=0, width=1530, height=60)

        # First image
        img_top = Image.open(r"my_images\details.jpeg")
        img_top = img_top.resize((650, 700), Image.LANCZOS)
        self.photoimg_top = ImageTk.PhotoImage(img_top)

        f_lbl = Label(self.root, image=self.photoimg_top)
        f_lbl.place(x=0, y=60, width=650, height=700) 

        # Second image
        img_bottom = Image.open(r"my_images\details.jpeg")
        img_bottom = img_bottom.resize((950, 700), Image.LANCZOS)
        self.photoimg_bottom = ImageTk.PhotoImage(img_bottom)

        f_lbl = Label(self.root, image=self.photoimg_bottom)
        f_lbl.place(x=650, y=60, width=950, height=700) 

        # Buttons
        button1 = Button(f_lbl, text="Face Recognition", command=self.face_recog, 
                        cursor="hand2", font=("times new roman", 18, "bold"), 
                        bg="white", fg="pink")
        button1.place(x=350, y=600, width=200, height=40)
        
        stop_btn = Button(f_lbl, text="Stop Recognition", command=self.stop_recognition,
                         cursor="hand2", font=("times new roman", 18, "bold"),
                         bg="white", fg="red")
        stop_btn.place(x=350, y=650, width=200, height=40)

    def on_close(self):
        """Handle window close event"""
        self.stop_recognition()
        self.root.destroy()

    def stop_recognition(self):
        """Stop the face recognition process"""
        self.is_running = False
        if self.video_cap is not None:
            self.video_cap.release()
            self.video_cap = None
        cv2.destroyAllWindows()

    ##################### ATTENDANCE ##################################
    def mark_attendance(self, i, r, n, d):
        with open("todaysattendance.csv", "r+", newline="\n") as f:
            myDataList = f.readlines()
            name_list = []
            for line in myDataList:
                entry = line.split((","))
                name_list.append(entry[0])
            if ((i not in name_list) and (r not in name_list) and 
                (n not in name_list) and (d not in name_list)):
                now = datetime.now()
                d1 = now.strftime("%d/%m/%Y")
                dtString = now.strftime("%H:%M:%S")
                f.writelines(f"\n{i},{r},{n},{d},{dtString},{d1},Present")

    ############################## SPOOF PREVENTION METHODS ################################
    def check_liveness(self, face_roi):
        try:
            # Check if face_roi is valid
            if face_roi is None or face_roi.size == 0:
                return False
                
            # Convert to grayscale and resize to consistent size
            gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (100, 100))  # Fixed size for consistency
            
            # 1. Texture analysis (simplified)
            edges = cv2.Canny(gray, 50, 150)
            texture_score = np.mean(edges)
            
            # 2. Color analysis (more lenient)
            hsv = cv2.cvtColor(cv2.resize(face_roi, (100, 100)), cv2.COLOR_BGR2HSV)
            color_metric = np.std(hsv[:,:,1]) / (np.std(hsv[:,:,0]) + 1e-6)
            
            # 3. Motion analysis (only if we have previous frame)
            motion_magnitude = 1.0  # Default value that assumes motion
            if hasattr(self, 'prev_gray') and self.prev_gray is not None:
                if self.prev_gray.shape == gray.shape:
                    flow = cv2.calcOpticalFlowFarneback(self.prev_gray, gray, None, 
                                                      pyr_scale=0.5, levels=3, 
                                                      winsize=15, iterations=3, 
                                                      poly_n=5, poly_sigma=1.2, flags=0)
                    motion_magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2).mean()
            
            self.prev_gray = gray
            
            # Adjusted thresholds (less strict)
            texture_pass = texture_score > 10  # Lowered from 20
            color_pass = color_metric > 0.2    # Lowered from 0.3
            motion_pass = motion_magnitude > 0.05  # Lowered from 0.1
            
            # Debug output
            print(f"Texture: {texture_score:.1f} ({'PASS' if texture_pass else 'FAIL'}) | "
                  f"Color: {color_metric:.2f} ({'PASS' if color_pass else 'FAIL'}) | "
                  f"Motion: {motion_magnitude:.2f} ({'PASS' if motion_pass else 'FAIL'})")
            
            # Require 2 out of 3 tests to pass
            is_live = sum([texture_pass, color_pass, motion_pass]) >= 2
            
            return is_live
        except Exception as e:
            print(f"Liveness check error: {e}")
            return True  # Fail-safe: assume live if error occurs

    ############################## FACE RECOGNITION ############################################### 
    def face_recog(self):
        def draw_boundary(img, classifier, scaleFactor, minNeighbors, color, text, clf):
            gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            features = classifier.detectMultiScale(gray_image, scaleFactor, minNeighbors)

            coord = []
            
            for (x, y, w, h) in features:
                try:
                    # Ensure coordinates are within image bounds
                    height, width = img.shape[:2]
                    x, y, w, h = max(0, x), max(0, y), min(w, width-x), min(h, height-y)
                    
                    # Skip if width or height is zero
                    if w <= 0 or h <= 0:
                        continue
                        
                    face_roi = img[y:y+h, x:x+w]
                    
                    # Skip if face_roi is empty
                    if face_roi.size == 0:
                        continue
                    
                    # Spoof detection check
                    is_live = self.check_liveness(face_roi)
                    
                    if not is_live:
                        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 3)
                        cv2.putText(img, "SPOOF ATTEMPT", (x, y-10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                        continue
                    
                    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 3)
                    id, predict = clf.predict(gray_image[y:y+h, x:x+w])
                    confidence = int((100*(1-predict/300)))

                    conn = mysql.connector.connect(host="localhost", user="root", 
                                                 password="krisha123", database="face_recognition")
                    my_cursor = conn.cursor()

                    # Fetch student data
                    my_cursor.execute("SELECT Name, Roll, Dep, Student_id FROM student WHERE Student_id=%s", (str(id),))
                    student_data = my_cursor.fetchone()
                    
                    if student_data is None:
                        cv2.putText(img, "Unknown ID", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                        continue
                        
                    n, r, d, i = student_data
                    n = str(n)
                    r = str(r)
                    d = str(d)
                    i = str(i)

                    if confidence > 60: #77
                        cv2.putText(img, f"ID:{i}", (x, y-75), cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 255, 255), 2)
                        cv2.putText(img, f"Roll:{r}", (x, y-55), cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 255, 255), 2)
                        cv2.putText(img, f"Name:{n}", (x, y-30), cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 255, 255), 2)
                        cv2.putText(img, f"Dept:{d}", (x, y-5), cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 255, 255), 2)
                        self.mark_attendance(i, r, n, d)
                    else:
                        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 3)
                        cv2.putText(img, "Unknown Face", (x, y-5), cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 255, 255), 2)
                        
                except Exception as e:
                    print(f"Face processing error: {e}")
                    continue

            return coord

        def recognize(img, clf, faceCascade):
            coord = draw_boundary(img, faceCascade, 1.1, 10, (255, 255, 255), "Face", clf)
            return img

        # Stop any existing recognition first
        self.stop_recognition()

        # Load classifiers
        faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        if faceCascade.empty():
            messagebox.showerror("Error", "Could not load face detector!")
            return

        clf = cv2.face.LBPHFaceRecognizer_create()
        try:
            clf.read("classifier.xml")
        except:
            messagebox.showerror("Error", "Could not load classifier!")
            return

        # Initialize video capture
        self.video_cap = cv2.VideoCapture(0)
        if not self.video_cap.isOpened():
            messagebox.showerror("Error", "Could not open camera!")
            return

        # Initialize variables
        self.prev_gray = None
        self.is_running = True

        try:
            while self.is_running:
                ret, img = self.video_cap.read()
                if not ret:
                    print("Failed to grab frame")
                    break
                    
                img = recognize(img, clf, faceCascade)
                cv2.imshow("Face Recognition - Press Q to quit", img)

                # Check for quit command
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27 or not self.is_running:
                    break

                # Check if OpenCV window was closed
                if cv2.getWindowProperty("Face Recognition - Press Q to quit", cv2.WND_PROP_VISIBLE) < 1:
                    break
        finally:
            self.stop_recognition()

if __name__ == "__main__":
    root = Tk()
    obj = Face_Recognition(root)
    root.mainloop()