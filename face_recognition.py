from tkinter import*
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
from scipy.spatial import distance as dist
import dlib
from imutils import face_utils



def eye_aspect_ratio(eye):
    # compute the euclidean distances between the two sets of vertical eye landmarks (x, y)-coordinates
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])

    # compute the euclidean distance between the horizontal eye landmark (x, y)-coordinates
    C = dist.euclidean(eye[0], eye[3])

    # compute the eye aspect ratio
    ear = (A + B) / (2.0 * C)

    # return the eye aspect ratio
    return ear

# Add constants
EYE_AR_THRESH = 0.23
EYE_AR_CONSEC_FRAMES = 3


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")  

(lStart, lEnd) = (42, 48)
(rStart, rEnd) = (36, 42)

#####before spoof

class Face_Recognition:
    def __init__(self,root):
        self.root=root
        self.root.geometry("1530x790+0+0")
        self.root.title("Face Recognition System")
        
        title_lbl = Label(self.root, text = "Face recognition", font = ("times new roman", 35, "bold"),bg = "white", fg = "pink")
        title_lbl.place(x = 0, y = 0, width = 1530, height = 60)

        #first image
        img_top = Image.open(r"my_images\details.jpeg")
        img_top = img_top.resize((650, 700),Image.ANTIALIAS)
        self.photoimg_top=ImageTk.PhotoImage(img_top)

        f_lbl = Label(self.root,image=self.photoimg_top)
        f_lbl.place(x=0,y=60,width=650,height=700) 

        #second image
        img_bottom = Image.open(r"my_images\details.jpeg")
        img_bottom = img_bottom.resize((950, 700),Image.ANTIALIAS)
        self.photoimg_bottom=ImageTk.PhotoImage(img_bottom)

        f_lbl = Label(self.root,image=self.photoimg_bottom)
        f_lbl.place(x=650,y=60,width=950,height=700) 

        #button
        button1 = Button(f_lbl,text="Face Recognition", command=self.face_recog, cursor="hand2", font = ("times new roman", 18, "bold"),bg = "white", fg = "pink")
        button1.place(x=350, y=600, width=200, height=40)

    ############################## ATTENDANCE ##############################
    def mark_attendance(self, i, r, n, d):
        with open("todaysattendance.csv", "r+", newline="\n") as f:
            myDataList=f.readlines()
            name_list = []
            for line in myDataList:
                entry=line.split((","))
                name_list.append(entry[0])
            if((i not in name_list) and (r not in name_list) and (n not in name_list) and (d not in name_list)):
                now = datetime.now()
                d1 = now.strftime("%d/%m/%Y")
                dtString = now.strftime("%H:%M:%S")
                f.writelines(f"\n{i},{r},{n},{d},{dtString},{d1},Present")


    ############################## FACE RECOGNITION ############################## 

    # def face_recog(self):
    #     def draw_boundary(img, classifier, scaleFactor, minNeighbors, color, text, clf):
    #         gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #         features = classifier.detectMultiScale(gray_image,scaleFactor,minNeighbors)

    #         coord = []

    #         for (x,y,w,h) in features:
    #             cv2.rectangle(img, (x,y), (x+w,y+h),(0,255,0),3)
    #             id,predict=clf.predict(gray_image[y:y+h, x:x+w])
    #             confidence = int((100*(1-predict/300)))

    #             conn=mysql.connector.connect(host="localhost", user = "root", password = "krisha123", database="face_recognition")
    #             my_cursor = conn.cursor()

    #             my_cursor.execute("select Name from student where Student_id="+str(id))
    #             n=my_cursor.fetchone()
    #             n="+".join(n)

    #             my_cursor.execute("select Roll from student where Student_id="+str(id))
    #             r=my_cursor.fetchone()
    #             r="+".join(r)

    #             my_cursor.execute("select Dep from student where Student_id="+str(id))
    #             d=my_cursor.fetchone()
    #             d="+".join(d)

    #             my_cursor.execute("select Student_id from student where Student_id="+str(id))
    #             i=my_cursor.fetchone()
    #             i="+".join(i)

    #             if confidence>77:
    #                 cv2.putText(img,f"ID:{i}",(x,y-75), cv2.FONT_HERSHEY_COMPLEX, 0.8,(255,255,255),3)
    #                 cv2.putText(img,f"Roll:{r}",(x,y-55), cv2.FONT_HERSHEY_COMPLEX, 0.8,(255,255,255),3)
    #                 cv2.putText(img,f"Name:{n}",(x,y-30), cv2.FONT_HERSHEY_COMPLEX, 0.8,(255,255,255),3)
    #                 cv2.putText(img,f"Department:{d}",(x,y-5), cv2.FONT_HERSHEY_COMPLEX, 0.8,(255,255,255),3)
    #                 self.mark_attendance(i,r,n,d)
    #             else:
    #                 cv2.rectangle(img,(x,y), (x+w,y+h),(0,0,255),3)
    #                 cv2.putText(img,f"Unknown Face",(x,y-5), cv2.FONT_HERSHEY_COMPLEX, 0.8,(255,255,255),3)
                  
    #             coords=[x,y,w,h]

    #         return coord
        
    #     def recognize(img,clf,faceCascade):
    #         coord = draw_boundary(img,faceCascade,1.1,10,(255,255,255),"Face", clf)
    #         return img
        
    #     faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    #     clf = cv2.face.LBPHFaceRecognizer_create()
    #     clf.read("classifier.xml")

    #     video_cap=cv2.VideoCapture(0)

    #     while True:
    #         ret, img=video_cap.read()
    #         img = recognize(img,clf,faceCascade)
    #         cv2.imshow("Welcome", img)

    #         key = cv2.waitKey(1) & 0xFF
    #         if key == ord('q') or key == 27:  
    #             break

    #     video_cap.release()
    #     cv2.destroyAllWindows()

    #     ################ Add the spoofing part'

    def face_recog(self):
        def draw_boundary(img, classifier, scaleFactor, minNeighbors, color, text, clf):
            gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            features = classifier.detectMultiScale(gray_image, scaleFactor, minNeighbors)

            coords = []

            COUNTER = 0
            TOTAL = 0

            for (x, y, w, h) in features:
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)
                id, predict = clf.predict(gray_image[y:y + h, x:x + w])
                confidence = int((100 * (1 - predict / 300)))

                conn = mysql.connector.connect(host="localhost", user="root", password="krisha123", database="face_recognition")
                my_cursor = conn.cursor()

                my_cursor.execute("select Name from student where Student_id=" + str(id))
                n = "+".join(my_cursor.fetchone())

                my_cursor.execute("select Roll from student where Student_id=" + str(id))
                r = "+".join(my_cursor.fetchone())

                my_cursor.execute("select Dep from student where Student_id=" + str(id))
                d = "+".join(my_cursor.fetchone())

                my_cursor.execute("select Student_id from student where Student_id=" + str(id))
                i = "+".join(my_cursor.fetchone())

                if confidence > 77:
                    cv2.putText(img, f"ID:{i}", (x, y - 75), cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 255, 255), 3)
                    cv2.putText(img, f"Roll:{r}", (x, y - 55), cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 255, 255), 3)
                    cv2.putText(img, f"Name:{n}", (x, y - 30), cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 255, 255), 3)
                    cv2.putText(img, f"Department:{d}", (x, y - 5), cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 255, 255), 3)

                    coords = [x, y, w, h]

                    return coords, (i, r, n, d)

                else:
                    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 3)
                    cv2.putText(img, f"Unknown Face", (x, y - 5), cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 255, 255), 3)

            return coords, None

        def recognize(img, clf, faceCascade):
            coords, user_data = draw_boundary(img, faceCascade, 1.1, 10, (255, 255, 255), "Face", clf)

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            rects = detector(gray, 0)

            for rect in rects:
                shape = predictor(gray, rect)
                shape = face_utils.shape_to_np(shape)

                leftEye = shape[lStart:lEnd]
                rightEye = shape[rStart:rEnd]
                leftEAR = eye_aspect_ratio(leftEye)
                rightEAR = eye_aspect_ratio(rightEye)

                ear = (leftEAR + rightEAR) / 2.0

                # Draw eye contours
                leftEyeHull = cv2.convexHull(leftEye)
                rightEyeHull = cv2.convexHull(rightEye)
                cv2.drawContours(img, [leftEyeHull], -1, (0, 255, 0), 1)
                cv2.drawContours(img, [rightEyeHull], -1, (0, 255, 0), 1)

                # Check for blink
                if ear < EYE_AR_THRESH:
                    self.counter += 1
                else:
                    if self.counter >= EYE_AR_CONSEC_FRAMES:
                        self.total += 1

                        # If blink detected and user recognized
                        if user_data:
                            i, r, n, d = user_data
                            self.mark_attendance(i, r, n, d)

                    self.counter = 0

                    cv2.putText(img, f"Blinks: {self.total}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                if self.total == 0:
                    cv2.putText(img, "ALERT: No blink detected! Possible spoofing!", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            return img

        # Initialize blink counters
        self.counter = 0
        self.total = 0

        faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
        clf = cv2.face.LBPHFaceRecognizer_create()
        clf.read("classifier.xml")

        video_cap = cv2.VideoCapture(0)

        while True:
            ret, img = video_cap.read()
            img = recognize(img, clf, faceCascade)
            cv2.imshow("Welcome", img)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break

        video_cap.release()
        cv2.destroyAllWindows()







        







if __name__ == "__main__":
    root= Tk()
    obj = Face_Recognition(root)
    root.mainloop()