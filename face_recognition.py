from tkinter import*
from tkinter import ttk
from PIL import Image, ImageTk
from tkinter import messagebox
import mysql.connector
import cv2
import os
import numpy as np

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


    ############################## FACE RECOGNITION ############################################### 

    def face_recog(self):
        def draw_boundary(img, classifier, scaleFactor, minNeighbors, color, text, clf):
            gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            features = classifier.detectMultiScale(gray_image,scaleFactor,minNeighbors)

            coord = []

            for (x,y,w,h) in features:
                cv2.rectangle(img, (x,y), (x+w,y+h),(0,255,0),3)
                id,predict=clf.predict(gray_image[y:y+h, x:x+w])
                confidence = int((100*(1-predict/300)))

                conn=mysql.connector.connect(host="localhost", user = "root", password = "krisha123", database="face_recognition")
                my_cursor = conn.cursor()

                my_cursor.execute("select Name from student where Student_id="+str(id))
                n=my_cursor.fetchone()
                n="+".join(n)

                my_cursor.execute("select Roll from student where Student_id="+str(id))
                r=my_cursor.fetchone()
                r="+".join(r)

                my_cursor.execute("select Dep from student where Student_id="+str(id))
                d=my_cursor.fetchone()
                d="+".join(d)

                if confidence>77:
                    cv2.putText(img,f"Roll:{r}",(x,y-55), cv2.FONT_HERSHEY_COMPLEX, 0.8,(255,255,255),3)
                    cv2.putText(img,f"Name:{n}",(x,y-30), cv2.FONT_HERSHEY_COMPLEX, 0.8,(255,255,255),3)
                    cv2.putText(img,f"Department:{d}",(x,y-5), cv2.FONT_HERSHEY_COMPLEX, 0.8,(255,255,255),3)
                else:
                    cv2.rectangle(img,(x,y), (x+w,y+h),(0,0,255),3)
                    cv2.putText(img,f"Unknown Face",(x,y-5), cv2.FONT_HERSHEY_COMPLEX, 0.8,(255,255,255),3)
                  
                coords=[x,y,w,h]

            return coord
        
        def recognize(img,clf,faceCascade):
            coord = draw_boundary(img,faceCascade,1.1,10,(255,255,255),"Face", clf)
            return img
        
        faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
        clf = cv2.face.LBPHFaceRecognizer_create()
        clf.read("classifier.xml")

        video_cap=cv2.VideoCapture(0)

        while True:
            ret, img=video_cap.read()
            img = recognize(img,clf,faceCascade)
            cv2.imshow("Welcome", img)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # <--- Changed this line
                break

        video_cap.release()
        cv2.destroyAllWindows()






        







if __name__ == "__main__":
    root= Tk()
    obj = Face_Recognition(root)
    root.mainloop()