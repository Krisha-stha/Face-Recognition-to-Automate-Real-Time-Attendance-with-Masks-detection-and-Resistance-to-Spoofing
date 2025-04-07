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
        button1 = Button(f_lbl,text="Face Recognition", cursor="hand2", font = ("times new roman", 18, "bold"),bg = "white", fg = "pink")
        button1.place(x=350, y=600, width=200, height=40)


    ############################## FACE RECOGNITION ############################################### 

    def face_recog(self):
        def draw_boundary(img, classifier, scaleFactor, minNeighbors, color, text, cif):
            gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            features = classifier.detectMultiScale(gray_image,scaleFactor,minNeighbors)
        







if __name__ == "__main__":
    root= Tk()
    obj = Face_Recognition(root)
    root.mainloop()