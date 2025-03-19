from tkinter import*
from tkinter import ttk
from PIL import Image, ImageTk

class Student:
    def __init__(self,root):
        self.root=root
        self.root.geometry("1530x790+0+0")
        self.root.title("Face Recognition System")
        
        # first image
        img = Image.open(r"C:\Users\Acer\Desktop\FaceRecognition\my_images\student_image1.jpeg")
        img = img.resize((500, 130),Image.ANTIALIAS)
        self.photoimg=ImageTk.PhotoImage(img)

        f_lbl = Label(self.root,image=self.photoimg)
        f_lbl.place(x=0,y=0,width=500,height=130) 

        # second image
        img1 = Image.open(r"C:\Users\Acer\Desktop\FaceRecognition\my_images\student_image1.jpeg")
        img1 = img1.resize((500, 130),Image.ANTIALIAS)
        self.photoimg1=ImageTk.PhotoImage(img1)

        f_lbl = Label(self.root,image=self.photoimg1)
        f_lbl.place(x=500,y=0,width=500,height=130) 

        # third image
        img2 = Image.open(r"C:\Users\Acer\Desktop\FaceRecognition\my_images\student_image1.jpeg")
        img2 = img2.resize((500, 130),Image.ANTIALIAS)
        self.photoimg2=ImageTk.PhotoImage(img2)

        f_lbl = Label(self.root,image=self.photoimg2)
        f_lbl.place(x=1000,y=0,width=500,height=130) 

        # background image
        img3 = Image.open(r"C:\Users\Acer\Desktop\FaceRecognition\my_images\bgimg.jpeg")
        img3 = img3.resize((1530, 710),Image.ANTIALIAS)
        self.photoimg3=ImageTk.PhotoImage(img3)

        bg_image = Label(self.root,image=self.photoimg3)
        bg_image.place(x=0,y=130,width=1530,height=710) 

        title_lbl = Label(bg_image, text = "Student Management System", font = ("times new roman", 35, "bold"),bg = "pink", fg = "red")
        title_lbl.place(x = 0, y = 0, width = 1530, height = 45)



if __name__ == "__main__":
    root= Tk()
    obj = Student(root)
    root.mainloop()