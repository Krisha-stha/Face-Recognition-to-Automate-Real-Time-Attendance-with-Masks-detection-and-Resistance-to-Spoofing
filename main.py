from tkinter import*
from tkinter import ttk
from PIL import Image, ImageTk
import os
import tkinter
from student import Student
from train import Train
from face_recognition import Face_Recognition
from attendance import Attendance
from developer import Developer
from help import Help

class Face_Recognition_System:
    def __init__(self,root):
        self.root=root
        self.root.geometry("1530x790+0+0")
        self.root.title("Face Recognition System")

        # first image
        img = Image.open(r"my_images\image2.jpeg")
        img = img.resize((500, 130),Image.ANTIALIAS)
        self.photoimg=ImageTk.PhotoImage(img)

        f_lbl = Label(self.root,image=self.photoimg)
        f_lbl.place(x=0,y=0,width=500,height=130) 

        # second image
        img1 = Image.open(r"my_images\image2.jpeg")
        img1 = img1.resize((500, 130),Image.ANTIALIAS)
        self.photoimg1=ImageTk.PhotoImage(img1)

        f_lbl = Label(self.root,image=self.photoimg1)
        f_lbl.place(x=500,y=0,width=500,height=130) 

        # third image
        img2 = Image.open(r"my_images\image2.jpeg")
        img2 = img2.resize((500, 130),Image.ANTIALIAS)
        self.photoimg2=ImageTk.PhotoImage(img2)

        f_lbl = Label(self.root,image=self.photoimg2)
        f_lbl.place(x=1000,y=0,width=500,height=130) 

        # background image
        img3 = Image.open(r"my_images\bgimg.jpeg")
        img3 = img3.resize((1530, 710),Image.ANTIALIAS)
        self.photoimg3=ImageTk.PhotoImage(img3)

        bg_image = Label(self.root,image=self.photoimg3)
        bg_image.place(x=0,y=130,width=1530,height=710) 

        title_lbl = Label(bg_image, text = "FACE RECOGNITION SYSTEM", font = ("times new roman", 35, "bold"),bg = "pink", fg = "red")
        title_lbl.place(x = 0, y = 0, width = 1530, height = 45)

        # student Details
        student = Image.open(r"my_images\student.jpeg")
        student = student.resize((220, 220),Image.ANTIALIAS)
        self.studentphoto=ImageTk.PhotoImage(student)

        button = Button(bg_image,image=self.studentphoto,command=self.student_details, cursor="hand2")
        button.place(x=200, y=100, width=220, height=220)

        button1 = Button(bg_image,text="Student Details",command=self.student_details, cursor="hand2", font = ("times new roman", 15, "bold"),bg = "white", fg = "pink")
        button1.place(x=200, y=300, width=220, height=40)

        # Detect face button
        facedetect = Image.open(r"my_images\face.jpeg")
        facedetect = facedetect.resize((220, 220),Image.ANTIALIAS)
        self.detectface=ImageTk.PhotoImage(facedetect)

        button = Button(bg_image,image=self.detectface, cursor="hand2",command=self.face_data)
        button.place(x=500, y=100, width=220, height=220)

        button2 = Button(bg_image,text="Face Detector", cursor="hand2", command=self.face_data,font = ("times new roman", 15, "bold"),bg = "white", fg = "pink")
        button2.place(x=500, y=300, width=220, height=40)

         # Attendance button
        attendance = Image.open(r"my_images\attendance.jpeg")
        attendance = attendance.resize((220, 220),Image.ANTIALIAS)
        self.attend=ImageTk.PhotoImage(attendance)

        button = Button(bg_image,image=self.attend, cursor="hand2", command=self.attendance_data)
        button.place(x=800, y=100, width=220, height=220)

        button3 = Button(bg_image,text="Attendance", cursor="hand2", command=self.attendance_data, font = ("times new roman", 15, "bold"),bg = "white", fg = "pink")
        button3.place(x=800, y=300, width=220, height=40)

        # Help button
        helpme = Image.open(r"my_images\help.jpeg")
        helpme = helpme.resize((220, 220),Image.ANTIALIAS)
        self.helppp=ImageTk.PhotoImage(helpme)

        button = Button(bg_image,image=self.helppp, cursor="hand2", command=self.help)
        button.place(x=1100, y=100, width=220, height=220)

        button4 = Button(bg_image,text="Help", cursor="hand2", command=self.help, font = ("times new roman", 15, "bold"),bg = "white", fg = "pink")
        button4.place(x=1100, y=300, width=220, height=40)

        #train
        trainn = Image.open(r"my_images\trainn.jpeg")
        trainn = trainn.resize((220, 220),Image.ANTIALIAS)
        self.trainn=ImageTk.PhotoImage(trainn)

        button = Button(bg_image,image=self.trainn, cursor="hand2", command=self.train_data)
        button.place(x=200, y=380, width=220, height=220)

        button4 = Button(bg_image,text="Train", cursor="hand2", command=self.train_data, font = ("times new roman", 15, "bold"),bg = "white", fg = "pink")
        button4.place(x=200, y=580, width=220, height=40)

        #photos
        photos = Image.open(r"my_images\photos.jpeg")
        photos = photos.resize((220, 220),Image.ANTIALIAS)
        self.photos=ImageTk.PhotoImage(photos)

        button = Button(bg_image,image=self.photos, cursor="hand2", command=self.open_img)
        button.place(x=500, y=380, width=220, height=220)

        button4 = Button(bg_image,text="Photos", cursor="hand2", command=self.open_img, font = ("times new roman", 15, "bold"),bg = "white", fg = "pink")
        button4.place(x=500, y=580, width=220, height=40)

        #develop
        developer = Image.open(r"my_images\develop.jpeg")
        developer = developer.resize((220, 220),Image.ANTIALIAS)
        self.develop=ImageTk.PhotoImage(developer)

        button = Button(bg_image,image=self.develop, cursor="hand2", command = self.developer_data)
        button.place(x=800, y=380, width=220, height=220)

        button4 = Button(bg_image,text="Developer", cursor="hand2", command = self.developer_data,font = ("times new roman", 15, "bold"),bg = "white", fg = "pink")
        button4.place(x=800, y=580, width=220, height=40)

        #exit
        exit = Image.open(r"my_images\exit.jpeg")
        exit = exit.resize((220, 220),Image.ANTIALIAS)
        self.exit=ImageTk.PhotoImage(exit)

        button = Button(bg_image,image=self.exit, cursor="hand2", command=self.iExit)
        button.place(x=1100, y=380, width=220, height=220)

        button4 = Button(bg_image,text="Exit", cursor="hand2", command=self.iExit,font = ("times new roman", 15, "bold"),bg = "white", fg = "pink")
        button4.place(x=1100, y=580, width=220, height=40)

    def open_img(self):
        os.startfile("data")

    def iExit(self):
        self.iExit=tkinter.messagebox.askyesno("Face Recognition", "Are ypu sure you want to exit?", parent = self.root)
        if self.iExit >0:
            self.root.destroy()
        else:
            return

    #Function buttons
    def student_details(self):
        self.new_window=Toplevel(self.root)
        self.app=Student(self.new_window)

    def train_data(self):
        self.new_window=Toplevel(self.root)
        self.app=Train(self.new_window)

    def face_data(self):
        self.new_window=Toplevel(self.root)
        self.app=Face_Recognition(self.new_window)
    
    def attendance_data(self):
        self.new_window=Toplevel(self.root)
        self.app=Attendance(self.new_window)

    def developer_data(self):
        self.new_window=Toplevel(self.root)
        self.app=Developer(self.new_window)

    def help(self):
        self.new_window=Toplevel(self.root)
        self.app=Help(self.new_window)


if __name__ == "__main__":
    root= Tk()
    obj = Face_Recognition_System(root)
    root.mainloop()