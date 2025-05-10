def __init__(self,root):
    #     self.root=root
    #     self.root.geometry("1530x790+0+0")
    #     self.root.title("Face Recognition System")
 
    #     title_lbl = Label(self.root, text="Face recognition", font=("times new roman", 35, "bold"), bg="white", fg="pink")
    #     title_lbl.place(x=0, y=0, width=1530, height=60)
 
    #     # First image
    #     img_top = Image.open(r"my_images\details.jpeg")
    #     img_top = img_top.resize((650, 700), Image.ANTIALIAS)
    #     self.photoimg_top=ImageTk.PhotoImage(img_top)
 
    #     f_lbl = Label(self.root, image=self.photoimg_top)
    #     f_lbl.place(x=0, y=60, width=650, height=700) 
 
    #     # Second image
    #     img_bottom = Image.open(r"my_images\details.jpeg")
    #     img_bottom = img_bottom.resize((950, 700), Image.ANTIALIAS)
    #     self.photoimg_bottom=ImageTk.PhotoImage(img_bottom)
 
    #     f_lbl = Label(self.root, image=self.photoimg_bottom)
    #     f_lbl.place(x=650, y=60, width=950, height=700) 
 
    #     # Button
    #     button1 = Button(f_lbl, text="Face Recognition", command=self.face_recog, cursor="hand2", 
    #                      font=("times new roman", 18, "bold"), bg="white", fg="pink")
    #     button1.place(x=350, y=600, width=200, height=40)