from tkinter import*
from tkinter import ttk
from PIL import Image, ImageTk
from tkinter import messagebox
import mysql.connector
import cv2

class Developer:
    def __init__(self,root):
        self.root=root
        self.root.geometry("1530x790+0+0")
        self.root.title("Face Recognition System")

        title_lbl = Label(self.root, text = "Developer", font = ("times new roman", 35, "bold"),bg = "white", fg = "pink")
        title_lbl.place(x = 0, y = 0, width = 1530, height = 45)

        img_top = Image.open(r"my_images\details.jpeg")
        img_top = img_top.resize((1530, 720),Image.ANTIALIAS)
        self.photoimg_top=ImageTk.PhotoImage(img_top)

        f_lbl = Label(self.root,image=self.photoimg_top)
        f_lbl.place(x=5,y=55,width=1530,height=720) 

        # Frame
        main_frame =Frame(f_lbl, bd=2, bg="white")
        main_frame.place(x=1000, y=0, width = 500, height=600)

        img_mine = Image.open(r"my_images\student.jpeg")
        img_mine = img_mine.resize((200, 200),Image.ANTIALIAS)
        self.photoimg_mine=ImageTk.PhotoImage(img_mine)

        f_lbl = Label(main_frame,image=self.photoimg_mine)
        f_lbl.place(x=300,y=0,width=200,height=200) 

        #Developer
        dev_label = Label(main_frame, text="Hello my name is Krisha", font = ("times new roman", 20, "bold"), bg="white")
        dev_label.place(x=0, y=40)

        dev_label = Label(main_frame, text="This is my Production Project", font = ("times new roman", 20, "bold"), bg="white")
        dev_label.place(x=0, y=40)

        img_mine1 = Image.open(r"my_images\student.jpeg")
        img_mine1 = img_mine1.resize((500, 300),Image.ANTIALIAS)
        self.photoimg_mine1=ImageTk.PhotoImage(img_mine1)

        f_lbl = Label(main_frame,image=self.photoimg_mine1)
        f_lbl.place(x=0,y=210,width=500,height=300)




if __name__ == "__main__":
    root= Tk()
    obj = Developer(root)
    root.mainloop()