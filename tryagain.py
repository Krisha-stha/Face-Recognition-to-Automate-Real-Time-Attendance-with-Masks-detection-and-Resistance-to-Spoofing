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
from collections import deque

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
        self.last_faces = deque(maxlen=10)  # For tracking face count consistency
        
        # Liveness detection parameters
        self.texture_baseline = 12  # More lenient defaults
        self.color_baseline = 15
        self.motion_baseline = 0.05
        self.reflection_baseline = 40
        
        # Temporal smoothing
        self.liveness_history = deque(maxlen=5)  # Stores last 5 liveness results
        self.spoof_cooldown = 0  # Frames to wait after spoof detection
        
        # Load eye cascade for blink detection
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

        title_lbl = Label(self.root, text="Face Recognition", font=("times new roman", 35, "bold"), bg="white", fg="pink")
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
        
        calibrate_btn = Button(f_lbl, text="Calibrate", command=self.calibrate_for_environment,
                             cursor="hand2", font=("times new roman", 12),
                             bg="white", fg="blue")
        calibrate_btn.place(x=350, y=550, width=100, height=30)

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

    def calibrate_for_environment(self, num_frames=30):
        """Calibrate liveness detection for current environment"""
        if self.video_cap is None:
            self.video_cap = cv2.VideoCapture(0)
            if not self.video_cap.isOpened():
                messagebox.showerror("Error", "Could not open camera for calibration!")
                return
        
        print("Calibrating for environment...")
        texture_scores = []
        color_vars = []
        motion_mags = []
        reflection_scores = []
        
        for _ in range(num_frames):
            ret, frame = self.video_cap.read()
            if ret:
                # Process frame
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                
                # Texture
                edges = cv2.Canny(gray, 30, 100)
                texture_scores.append(np.mean(edges))
                
                # Color
                color_vars.append(np.std(hsv[:,:,1]))
                
                # Reflection
                reflection_scores.append(cv2.mean(cv2.inRange(hsv, (0, 0, 220), (180, 30, 255)))[0])
                
                # Motion (needs previous frame)
                if hasattr(self, 'prev_gray') and self.prev_gray is not None:
                    flow = cv2.calcOpticalFlowFarneback(self.prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.1, 0)
                    motion_mags.append(np.sqrt(flow[...,0]**2 + flow[...,1]**2).mean())
                self.prev_gray = gray
        
        # Calculate baseline values (more lenient thresholds)
        if texture_scores:
            self.texture_baseline = max(8, np.median(texture_scores) * 0.6)  # 60% of median
        if color_vars:
            self.color_baseline = max(8, np.median(color_vars) * 0.6)
        if motion_mags:
            self.motion_baseline = max(0.02, np.median(motion_mags) * 0.6)
        if reflection_scores:
            self.reflection_baseline = min(60, np.median(reflection_scores) * 1.5)  # 150% of median
        
        print(f"Calibration complete. Baselines - Texture: {self.texture_baseline:.1f}, "
              f"ColorVar: {self.color_baseline:.1f}, Motion: {self.motion_baseline:.3f}, "
              f"Reflection: {self.reflection_baseline:.1f}")
        
        messagebox.showinfo("Calibration Complete", 
                          f"System calibrated for current environment.\n"
                          f"Texture threshold: {self.texture_baseline:.1f}\n"
                          f"Color variation threshold: {self.color_baseline:.1f}\n"
                          f"Motion threshold: {self.motion_baseline:.3f}\n"
                          f"Reflection threshold: {self.reflection_baseline:.1f}")

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

    ############################## SPOOF PREVENTION ################################
    def check_liveness(self, face_roi):
        try:
            if face_roi is None or face_roi.size == 0:
                return False
                
            # Convert to grayscale and resize
            gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (100, 100))
            
            # 1. Texture Analysis (less sensitive)
            edges = cv2.Canny(gray, 25, 75)  # Lower thresholds
            texture_score = np.mean(edges)
            
            # 2. Color Analysis (less strict)
            hsv = cv2.cvtColor(cv2.resize(face_roi, (100, 100)), cv2.COLOR_BGR2HSV)
            color_variation = np.std(hsv[:,:,1])
            
            # 3. Motion Analysis (more tolerant)
            motion_magnitude = 0
            if hasattr(self, 'prev_gray') and self.prev_gray is not None:
                if self.prev_gray.shape == gray.shape:
                    flow = cv2.calcOpticalFlowFarneback(self.prev_gray, gray, None, 
                                                    pyr_scale=0.5, levels=3, 
                                                    winsize=15, iterations=3, 
                                                    poly_n=5, poly_sigma=1.1, flags=0)
                    motion_magnitude = np.sqrt(flow[...,0]**2 + flow[...,1]**2).mean()
            
            # 4. Blink Detection (only consider if eyes are found)
            eyes = self.eye_cascade.detectMultiScale(gray, 1.1, 5)
            blink_detected = len(eyes) < 2 if len(eyes) > 0 else None
            
            # 5. Reflection Detection (less sensitive)
            reflection_score = cv2.mean(cv2.inRange(hsv, (0, 0, 220), (180, 30, 255)))[0]
            
            # Debug output
            print(f"Texture: {texture_score:.1f} (>{self.texture_baseline:.1f}) | "
                  f"ColorVar: {color_variation:.1f} (>{self.color_baseline:.1f}) | "
                  f"Motion: {motion_magnitude:.3f} (>{self.motion_baseline:.3f}) | "
                  f"Eyes: {len(eyes)} | "
                  f"Reflection: {reflection_score:.1f} (<{self.reflection_baseline:.1f})")
            
            # Adaptive scoring system with weights
            score = 0
            total_possible = 0
            
            # Texture check (weight: 25%)
            texture_pass = texture_score > self.texture_baseline
            score += 0.25 if texture_pass else 0
            total_possible += 0.25
            
            # Color variation check (weight: 20%)
            color_pass = color_variation > self.color_baseline
            score += 0.2 if color_pass else 0
            total_possible += 0.2
            
            # Motion check (weight: 30%)
            motion_pass = motion_magnitude > self.motion_baseline
            score += 0.3 if motion_pass else 0
            total_possible += 0.3
            
            # Blink check (weight: 10%, only if eyes detected)
            if blink_detected is not None:
                blink_pass = not blink_detected
                score += 0.1 if blink_pass else 0
                total_possible += 0.1
            
            # Reflection check (weight: 15%)
            reflection_pass = reflection_score < self.reflection_baseline
            score += 0.15 if reflection_pass else 0
            total_possible += 0.15
            
            # Normalize score
            if total_possible > 0:
                normalized_score = score / total_possible
            else:
                normalized_score = 0
            
            # Store result for temporal smoothing
            self.liveness_history.append(normalized_score >= 0.65)  # 65% threshold
            
            # Use majority voting from history to prevent flipping
            if len(self.liveness_history) >= 3:  # Need at least 3 samples
                is_live = sum(self.liveness_history) >= 2  # At least 2/3 recent frames must be live
            else:
                is_live = normalized_score >= 0.65
            
            # Apply cooldown if we recently detected a spoof
            if self.spoof_cooldown > 0:
                self.spoof_cooldown -= 1
                is_live = False  # Maintain spoof state during cooldown
            
            # If we detect spoof now, set cooldown
            if not is_live:
                self.spoof_cooldown = 5  # 5 frame cooldown
            
            self.prev_gray = gray
            return is_live
        except Exception as e:
            print(f"Liveness check error: {e}")
            return True  # Fail safe - allow if error occurs

    ############################## FACE RECOGNITION ################################
    def face_recog(self):
        def draw_boundary(img, classifier, scaleFactor, minNeighbors, color, text, clf):
            gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            features = classifier.detectMultiScale(gray_image, scaleFactor, minNeighbors)

            coord = []
            
            # Track number of faces detected
            self.last_faces.append(len(features))
            if len(self.last_faces) > 10:
                self.last_faces.popleft()
            
            # Check for suspiciously consistent face counts (photo detection)
            if len(self.last_faces) >= 5:
                face_count_variation = np.std(self.last_faces)
                if face_count_variation < 0.1:
                    cv2.putText(img, "WARNING: Possible photo attack", (20, 50), 
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    return coord

            for (x, y, w, h) in features:
                try:
                    # Ensure coordinates are within image bounds
                    height, width = img.shape[:2]
                    x, y, w, h = max(0, x), max(0, y), min(w, width-x), min(h, height-y)
                    
                    if w <= 0 or h <= 0:
                        continue
                        
                    face_roi = img[y:y+h, x:x+w]
                    
                    if face_roi.size == 0:
                        continue
                    
                    # Spoof detection with temporal smoothing
                    is_live = self.check_liveness(face_roi)
                    
                    # Draw rectangle based on liveness
                    if not is_live:
                        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 3)
                        cv2.putText(img, "SPOOF ATTEMPT", (x, y-10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                        continue
                    
                    # If live, proceed with recognition
                    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 3)
                    id, predict = clf.predict(gray_image[y:y+h, x:x+w])
                    confidence = int((100*(1-predict/300)))

                    conn = mysql.connector.connect(host="localhost", user="root", 
                                                 password="krisha123", database="face_recognition")
                    my_cursor = conn.cursor()

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

                    if confidence > 60:
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

        # Reset tracking variables
        self.prev_gray = None
        self.is_running = True
        self.last_faces = deque(maxlen=10)
        self.liveness_history = deque(maxlen=5)
        self.spoof_cooldown = 0

        try:
            while self.is_running:
                ret, img = self.video_cap.read()
                if not ret:
                    print("Failed to grab frame")
                    break
                    
                img = recognize(img, clf, faceCascade)
                cv2.imshow("Face Recognition - Press Q to quit", img)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27 or not self.is_running:
                    break

                if cv2.getWindowProperty("Face Recognition - Press Q to quit", cv2.WND_PROP_VISIBLE) < 1:
                    break
        finally:
            self.stop_recognition()

if __name__ == "__main__":
    root = Tk()
    obj = Face_Recognition(root)
    root.mainloop()