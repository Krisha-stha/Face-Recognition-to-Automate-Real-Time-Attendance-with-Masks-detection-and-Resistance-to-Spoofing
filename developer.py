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

        title_lbl = Label(self.root, text = "Developer", font = ("times new roman", 35, "bold"),bg = "white", fg = "#191970")
        title_lbl.place(x = 0, y = 0, width = 1530, height = 45)

        img_top = Image.open(r"my_images\image2.jpeg")
        img_top = img_top.resize((1530, 750),Image.ANTIALIAS)
        self.photoimg_top=ImageTk.PhotoImage(img_top)

        f_lbl = Label(self.root,image=self.photoimg_top)
        f_lbl.place(x=5,y=45,width=1530,height=750) 

        # Frame
        main_frame =Frame(f_lbl, bd=2, bg="white")
        main_frame.place(x=400, y=30, width = 750, height=600)

        # img_mine = Image.open(r"my_images\student.jpeg")
        # img_mine = img_mine.resize((200, 200),Image.ANTIALIAS)
        # self.photoimg_mine=ImageTk.PhotoImage(img_mine)

        # f_lbl = Label(main_frame,image=self.photoimg_mine)
        # f_lbl.place(x=300,y=0,width=200,height=200) 

        #Developer
        dev_label1 = Label(main_frame, text="Hello my name is Krisha Shrestha. This is my final product for ", font = ("Calibri", 20, "bold"), bg="white")
        dev_label1.place(x=0, y=35)

        dev_label = Label(main_frame, text="production project.", font = ("Calibri", 20, "bold"), bg="white")
        dev_label.place(x=0, y=70)

        dev_label3 = Label(main_frame, text="This project is a smart attendance system that uses facial ", font = ("Calibri", 20, "bold"), bg="white")
        dev_label3.place(x=0, y=105)

        dev_label4 = Label(main_frame, text="recognition enhanced with mask detection and spoof prevention.", font = ("Calibri", 20, "bold"), bg="white")
        dev_label4.place(x=0, y=140)

        dev_label5 = Label(main_frame, text="It allows for the creation, updating, and deletion of user profiles,", font = ("Calibri", 20, "bold"), bg="white")
        dev_label5.place(x=0, y=175)

        dev_label7 = Label(main_frame, text="where each user is registered by capturing and storing facial ", font = ("Calibri", 20, "bold"), bg="white")
        dev_label7.place(x=0, y=210)

        dev_label8 = Label(main_frame, text="images. The system then trains a recognition model based on this", font = ("Calibri", 20, "bold"), bg="white")
        dev_label8.place(x=0, y=245)

        dev_label9 = Label(main_frame, text="data.", font = ("Calibri", 20, "bold"), bg="white")
        dev_label9.place(x=0, y=280)

        dev_label10 = Label(main_frame, text="During attendance marking, the system performs real-time face ", font = ("Calibri", 20, "bold"), bg="white")
        dev_label10.place(x=0, y=315)

        dev_label11 = Label(main_frame, text="recognition and checks if the person is blinking to ensure the user ", font = ("Calibri", 20, "bold"), bg="white")
        dev_label11.place(x=0, y=350)

        dev_label9 = Label(main_frame, text="is live and not a photo or video (spoof). It also detects if the user ", font = ("Calibri", 20, "bold"), bg="white")
        dev_label9.place(x=0, y=385)

        dev_label9 = Label(main_frame, text="is wearing a mask, ensuring reliable identification under varied ", font = ("Calibri", 20, "bold"), bg="white")
        dev_label9.place(x=0, y=420)

        dev_label9 = Label(main_frame, text="conditions. Only if the face is recognized and blinking is detected, ", font = ("Calibri", 20, "bold"), bg="white")
        dev_label9.place(x=0, y=455)

        dev_label9 = Label(main_frame, text="attendance is recorded successfully.", font = ("Calibri", 20, "bold"), bg="white")
        dev_label9.place(x=0, y=490)


        # img_mine1 = Image.open(r"my_images\student.jpeg")
        # img_mine1 = img_mine1.resize((500, 300),Image.ANTIALIAS)
        # self.photoimg_mine1=ImageTk.PhotoImage(img_mine1)

        # f_lbl = Label(main_frame,image=self.photoimg_mine1)
        # f_lbl.place(x=0,y=210,width=500,height=300)




if __name__ == "__main__":
    root= Tk()
    obj = Developer(root)
    root.mainloop()