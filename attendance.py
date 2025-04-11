from tkinter import*
from tkinter import ttk
from PIL import Image, ImageTk
from tkinter import messagebox
import mysql.connector
import cv2

class Attendance:
    def __init__(self,root):
        self.root=root
        self.root.geometry("1530x790+0+0")
        self.root.title("Face Recognition System")

        # first image
        img = Image.open(r"my_images\student_image1.jpeg")
        img = img.resize((800, 200),Image.ANTIALIAS)
        self.photoimg=ImageTk.PhotoImage(img)

        f_lbl = Label(self.root,image=self.photoimg)
        f_lbl.place(x=0,y=0,width=800,height=200) 

        # second image
        img1 = Image.open(r"my_images\student_image1.jpeg")
        img1 = img1.resize((800, 200),Image.ANTIALIAS)
        self.photoimg1=ImageTk.PhotoImage(img1)

        # background image
        img3 = Image.open(r"my_images\bgimg.jpeg")
        img3 = img3.resize((1530, 710),Image.ANTIALIAS)
        self.photoimg3=ImageTk.PhotoImage(img3)

        bg_image = Label(self.root,image=self.photoimg3)
        bg_image.place(x=0,y=200,width=1530,height=710) 

        title_lbl = Label(bg_image, text = "Attendance Management System", font = ("times new roman", 35, "bold"),bg = "white", fg = "blue")
        title_lbl.place(x = 0, y = 0, width = 1530, height = 55)

        main_frame=Frame(bg_image, bd=2, bg="white")
        main_frame.place(x=20, y=60, width=1480, height=600)

        # left label frame
        Left_frame = LabelFrame(main_frame, bd=2, bg="white", relief=RIDGE, text="Attendance Details", font=("times new roman", 12, "bold"))
        Left_frame.place(x=10, y=10, width=760, height=580)

        f_lbl = Label(self.root,image=self.photoimg1)
        f_lbl.place(x=800,y=0,width=800,height=180) 

if __name__ == "__main__":
    root= Tk()
    obj = Attendance(root)
    root.mainloop()