from tkinter import*
from tkinter import ttk
from PIL import Image, ImageTk
from tkinter import messagebox
import mysql.connector
import cv2
from mtcnn import MTCNN 

class Student:
    def __init__(self,root):
        self.root=root
        self.root.geometry("1530x790+0+0")
        self.root.title("Face Recognition System")

        ###########variables###############
        self.var_dep = StringVar()
        self.var_course = StringVar()
        self.var_year = StringVar()
        self.var_semester = StringVar()
        self.var_std_id = StringVar()
        self.var_std_name = StringVar()
        self.var_div = StringVar()
        self.var_roll = StringVar()
        self.var_gender = StringVar()
        self.var_dob = StringVar()
        self.var_email = StringVar()
        self.var_phone = StringVar()
        self.var_address = StringVar()
        self.var_teacher = StringVar()
        
        # first image
        img = Image.open(r"my_images\studentpageup.jpeg")
        img = img.resize((1800, 130),Image.ANTIALIAS)
        self.photoimg=ImageTk.PhotoImage(img)

        f_lbl = Label(self.root,image=self.photoimg)
        f_lbl.place(x=0,y=0,width=1800,height=130) 

        # # second image
        # img1 = Image.open(r"my_images\studentpageup.jpeg")
        # img1 = img1.resize((500, 130),Image.ANTIALIAS)
        # self.photoimg1=ImageTk.PhotoImage(img1)

        # f_lbl = Label(self.root,image=self.photoimg1)
        # f_lbl.place(x=500,y=0,width=500,height=130) 

        # # third image
        # img2 = Image.open(r"my_images\studentpageup.jpeg")
        # img2 = img2.resize((500, 130),Image.ANTIALIAS)
        # self.photoimg2=ImageTk.PhotoImage(img2)

        # f_lbl = Label(self.root,image=self.photoimg2)
        # f_lbl.place(x=1000,y=0,width=500,height=130) 

        # background image
        img3 = Image.open(r"my_images\studentpageup.jpeg")
        img3 = img3.resize((1530, 710),Image.ANTIALIAS)
        self.photoimg3=ImageTk.PhotoImage(img3)

        bg_image = Label(self.root,image=self.photoimg3)
        bg_image.place(x=0,y=130,width=1530,height=710) 

        title_lbl = Label(bg_image, text = "Student Management System", font = ("Calibri", 35, "bold"),bg = "#447cc4", fg = "#191970")
        title_lbl.place(x = 0, y = 0, width = 1530, height = 45)

        main_frame=Frame(bg_image, bd=2, bg="white")
        main_frame.place(x=20, y=50, width=1480, height=600)

        # left label frame
        Left_frame = LabelFrame(main_frame, bd=2, bg="white", relief=RIDGE, text="Student Details", font=("Calibri", 12, "bold"))
        Left_frame.place(x=10, y=10, width=760, height=580)

        img_left = Image.open(r"my_images\fordetails.jpeg")
        img_left = img_left.resize((720, 130),Image.ANTIALIAS)
        self.photoimg_left=ImageTk.PhotoImage(img_left)

        f_lbl = Label(Left_frame,image=self.photoimg_left)
        f_lbl.place(x=5,y=0,width=720,height=130) 

        #Current Course
        current_course_frame = LabelFrame(Left_frame, bd=2, bg="white", relief=RIDGE, text="Current Course Information", font=("Calibri", 12, "bold"))
        current_course_frame.place(x=10, y=135, width=740, height=150)

        #Department
        dep_label = Label(current_course_frame, text="Department", font = ("Calibri", 12, "bold"), bg="white")
        dep_label.grid(row=0,column=0, padx=10, sticky  =W)

        dep_combo = ttk.Combobox(current_course_frame, textvariable=self.var_dep,font = ("Calibri", 12, "bold"), width = 17,  state="readonly")
        dep_combo["values"] = ("Select Department", "Computer", "Business", "Hospitality")
        dep_combo.current(0)
        dep_combo.grid(row=0,column=1, padx=2, pady=10, sticky  =W)

        # Course
        course_label = Label(current_course_frame, text="Course", font = ("Calibri", 12, "bold"), bg="white")
        course_label.grid(row=0,column=2, padx=10, sticky  =W)

        course_combo = ttk.Combobox(current_course_frame,textvariable=self.var_course,font = ("Calibri", 12, "bold"), width = 17,  state="readonly")
        course_combo["values"] = ("Select Course", "Computing", "BBA", "AI", "Cybersecurity", "Hospitality")
        course_combo.current(0)
        course_combo.grid(row=0,column=3, padx=2, pady=10, sticky = W)

        # year
        year_label = Label(current_course_frame, text="Year", font = ("Calibri", 12, "bold"), bg="white")
        year_label.grid(row=1,column=0, padx=10, sticky  =W)

        year_combo = ttk.Combobox(current_course_frame,textvariable=self.var_year,font = ("Calibri", 12, "bold"), width = 17,  state="readonly")
        year_combo["values"] = ("Select Year", "2020-21", "2021-22", "2022-23", "2023-24", "2024-25")
        year_combo.current(0)
        year_combo.grid(row=1,column=1, padx=2, pady=10, sticky = W)

        # Semester
        semester_label = Label(current_course_frame, text="Semester", font = ("Calibri", 12, "bold"), bg="white")
        semester_label.grid(row=1,column=2, padx=10, sticky  =W)

        semester_combo = ttk.Combobox(current_course_frame,textvariable=self.var_semester,font = ("Calibri", 12, "bold"), width = 17,  state="readonly")
        semester_combo["values"] = ("Select Semester", "1", "2", "3", "4", "5", "6", "7", "8")
        semester_combo.current(0)
        semester_combo.grid(row=1,column=3, padx=2, pady=10, sticky = W)

        #Class Student Information
        class_Student_frame = LabelFrame(Left_frame, bd=2, bg="white", relief=RIDGE, text="Class Student Information", font=("Calibri", 12, "bold"))
        class_Student_frame.place(x=5, y=250, width=740, height=300)

        #Student ID
        studentID_label = Label(class_Student_frame, text="StudentId", font = ("Calibri", 12, "bold"), bg="white")
        studentID_label.grid(row=0,column=0, padx=10, pady=5, sticky  =W)

        studentID_entry = ttk.Entry(class_Student_frame,textvariable=self.var_std_id, width=20,font = ("Calibri", 12, "bold"))
        studentID_entry.grid(row=0, column=1, padx=10, pady=5, sticky = W)

        #Student Name
        studentName_label = Label(class_Student_frame, text="Student Name", font = ("Calibri", 12, "bold"), bg="white")
        studentName_label.grid(row=0,column=2, padx=10, pady=5, sticky  =W)

        studentName_entry = ttk.Entry(class_Student_frame, textvariable=self.var_std_name, width=20,font = ("Calibri", 12, "bold"))
        studentName_entry.grid(row=0, column=3, padx=10, pady=5, sticky = W)

        #Class division
        class_div_label = Label(class_Student_frame, text="Class Divison", font = ("Calibri", 12, "bold"), bg="white")
        class_div_label.grid(row=1,column=0, padx=10, pady=5, sticky  =W)

        # class_div_entry = ttk.Entry(class_Student_frame, textvariable=self.var_div, width=20,font = ("Calibri", 12, "bold"))
        # class_div_entry.grid(row=1, column=1, padx=10, pady=5, sticky = W)

        div_combo = ttk.Combobox(class_Student_frame,textvariable=self.var_div,font = ("Calibri", 12, "bold"), width = 15,  state="readonly")
        div_combo["values"] = ("A", "B", "C")
        div_combo.current(0)
        div_combo.grid(row=1 ,column=1, padx=10, pady=5, sticky = W)


        #Roll No
        roll_no_label = Label(class_Student_frame, text="Roll No", font = ("Calibri", 12, "bold"), bg="white")
        roll_no_label.grid(row=1,column=2, padx=10, pady=5, sticky  =W)

        roll_no_entry = ttk.Entry(class_Student_frame, textvariable=self.var_roll,width=20,font = ("Calibri", 12, "bold"))
        roll_no_entry.grid(row=1, column=3, padx=10, pady=5, sticky = W)

        #Gender
        gender_label = Label(class_Student_frame, text="Gender", font = ("Calibri", 12, "bold"), bg="white")
        gender_label.grid(row=2,column=0, padx=10, pady=5, sticky  =W)

        # gender_entry = ttk.Entry(class_Student_frame, textvariable=self.var_gender, width=20,font = ("Calibri", 12, "bold"))
        # gender_entry.grid(row=2, column=1, padx=10, pady=5, sticky = W)

        gender_combo = ttk.Combobox(class_Student_frame,textvariable=self.var_gender,font = ("Calibri", 12, "bold"), width = 15,  state="readonly")
        gender_combo["values"] = ("Male", "Female", "Other")
        gender_combo.current(0)
        gender_combo.grid(row=2,column=1, padx=10, pady=5, sticky = W)

        #Date Of Birth
        dob_label = Label(class_Student_frame, text="DOB", font = ("Calibri", 12, "bold"), bg="white")
        dob_label.grid(row=2,column=2, padx=10, pady=5, sticky  =W)

        dob_entry = ttk.Entry(class_Student_frame, textvariable=self.var_dob, width=20,font = ("Calibri", 12, "bold"))
        dob_entry.grid(row=2, column=3, padx=10, pady=5, sticky = W)

        #Email
        email_label = Label(class_Student_frame, text="Email", font = ("Calibri", 12, "bold"), bg="white")
        email_label.grid(row=3,column=0, padx=10, pady=5, sticky  =W)

        email_entry = ttk.Entry(class_Student_frame, textvariable=self.var_email, width=20,font = ("Calibri", 12, "bold"))
        email_entry.grid(row=3, column=1, padx=10, pady=5, sticky = W)

        #phone no
        phone_label = Label(class_Student_frame, text="Phone no", font = ("Calibri", 12, "bold"), bg="white")
        phone_label.grid(row=3,column=2, padx=10, pady=5, sticky  =W)

        phone_entry = ttk.Entry(class_Student_frame, textvariable=self.var_phone, width=20,font = ("Calibri", 12, "bold"))
        phone_entry.grid(row=3, column=3, padx=10, pady=5, sticky = W)

        #Address
        address_label = Label(class_Student_frame, text="Address", font = ("Calibri", 12, "bold"), bg="white")
        address_label.grid(row=4,column=0, padx=10, pady=5, sticky  =W)

        address_entry = ttk.Entry(class_Student_frame, textvariable=self.var_address, width=20,font = ("Calibri", 12, "bold"))
        address_entry.grid(row=4, column=1, padx=10, pady=5, sticky = W)

        #Teacher Name
        teacher_label = Label(class_Student_frame, text="Teacher Name", font = ("Calibri", 12, "bold"), bg="white")
        teacher_label.grid(row=4,column=2, padx=10, pady=5, sticky  =W)

        teacher_entry = ttk.Entry(class_Student_frame, textvariable=self.var_teacher, width=20,font = ("Calibri", 12, "bold"))
        teacher_entry.grid(row=4, column=3, padx=10, pady=5, sticky = W)

        #radio buttons
        self.var_radio1 = StringVar()
        radiobtn1 = ttk.Radiobutton(class_Student_frame, variable=self.var_radio1,text = "Take a photo sample", value="Yes")
        radiobtn1.grid(row = 6, column = 0)

        radiobtn2 = ttk.Radiobutton(class_Student_frame, variable=self.var_radio1,text = "No photo sample", value="No")
        radiobtn2.grid(row = 6, column = 1)

        #buttons Frame
        btn_frame = Frame(class_Student_frame, bd=2, relief=RIDGE, bg="white")
        btn_frame.place(x=0, y=200, width=715, height=35)

        #Save
        save_btn=Button(btn_frame, text="Save",command=self.add_data, width = 21, font = ("Calibri", 12, "bold"), bg="#447cc4", fg="white")
        save_btn.grid(row=0, column=0)

        # Update
        update_btn=Button(btn_frame, text="Update", command=self.update_data, width = 21, font = ("Calibri", 12, "bold"), bg="#447cc4", fg="white")
        update_btn.grid(row=0, column=1)

        # Delete
        delete_btn=Button(btn_frame, text="Delete", command=self.delete_data, width = 21, font = ("Calibri", 12, "bold"), bg="#447cc4", fg="white")
        delete_btn.grid(row=0, column=2)

        # Reset
        reset_btn=Button(btn_frame, text="Reset", command=self.reset_data, width = 21, font = ("Calibri", 12, "bold"), bg="#447cc4", fg="white")
        reset_btn.grid(row=0, column=3)
        
        #Photos Frame
        btn_frame1 = Frame(class_Student_frame, bd=2, relief=RIDGE, bg="white")
        btn_frame1.place(x=0, y=235, width=715, height=35)

        #Take a photo sample
        take_photo_btn=Button(btn_frame1, command=self.generate_dataset,text="Take Photo Sample", width = 44, font = ("Calibri", 12, "bold"), bg="#447cc4", fg="white")
        take_photo_btn.grid(row=0, column=0)

        #Take a photo sample
        update_photo_btn=Button(btn_frame1, text="Update Photo Sample", width = 44, font = ("Calibri", 12, "bold"), bg="#447cc4", fg="white")
        update_photo_btn.grid(row=0, column=2)

        # right label frame
        Right_frame = LabelFrame(main_frame, bd=2, bg="white", relief=RIDGE, text="Student Details", font=("Calibri", 12, "bold"))
        Right_frame.place(x=780, y=10, width=660, height=580)

        #Image for right
        img_right = Image.open(r"my_images\fordetails2.jpeg")
        img_right = img_right.resize((720, 130),Image.ANTIALIAS)
        self.photoimg_right=ImageTk.PhotoImage(img_right)

        f_lbl = Label(Right_frame,image=self.photoimg_right)
        f_lbl.place(x=5,y=0,width=720,height=130) 

        #Search System
        search_frame = LabelFrame(Right_frame, bd=2, bg="white", relief=RIDGE, text="Search System", font=("Calibri", 12, "bold"))
        search_frame.place(x=5, y=135, width=720, height=70)

        search_label = Label(search_frame, text="Search By", font = ("Calibri", 15, "bold"), bg="#447cc4", fg= "white")
        search_label.grid(row=0,column=0, padx=10, pady=5, sticky  =W)

        search_combo = ttk.Combobox(search_frame,font = ("Calibri", 12, "bold"), width = 15,  state="readonly")
        search_combo["values"] = ("Select", "Roll_no", "Phone_no")
        search_combo.current(0)
        search_combo.grid(row=0,column=1, padx=2, pady=10, sticky = W)

        search_entry = ttk.Entry(search_frame, width=15,font = ("Calibri", 12, "bold"))
        search_entry.grid(row=0, column=2, padx=10, pady=5, sticky = W)

        search_btn=Button(search_frame, text="Search", width = 12, font = ("Calibri", 12, "bold"), bg="#447cc4", fg="white")
        search_btn.grid(row=0, column=3, padx = 4)

        showAll_btn=Button(search_frame, text="Show All", width = 12, font = ("Calibri", 12, "bold"), bg="#447cc4", fg="white")
        showAll_btn.grid(row=0, column=4, padx = 4)

        ##left##
       
        # table frame
        table_frame = Frame(Right_frame, bd=2, bg="white", relief=RIDGE)
        table_frame.place(x=5, y=210, width=650, height=350)

        scroll_x = ttk.Scrollbar(table_frame, orient=HORIZONTAL)
        scroll_y = ttk.Scrollbar(table_frame, orient=VERTICAL)
        
        self.student_table = ttk.Treeview(table_frame, column = ("dep", "course","year","sem","id","name","div","roll","gender","dob","email","phone","address","teacher","photo"), xscrollcommand=scroll_x.set, yscrollcommand=scroll_y.set)
        
        scroll_x.pack(side=BOTTOM, fill=X)
        scroll_y.pack(side=RIGHT, fill=Y)

        # scroll
        scroll_x.config(command=self.student_table.xview)
        scroll_y.config(command=self.student_table.yview)

        self.student_table.heading("dep", text="Department")
        self.student_table.heading("course", text="Course")
        self.student_table.heading("year", text="Year")
        self.student_table.heading("sem", text="Semester")
        self.student_table.heading("id", text="StudentId")
        self.student_table.heading("name", text="Name")
        self.student_table.heading("div", text="Division")
        self.student_table.heading("dob", text="DOB")
        self.student_table.heading("email", text="Email")
        self.student_table.heading("phone", text="Phone")
        self.student_table.heading("address", text="Address")
        self.student_table.heading("teacher", text="Teacher")
        self.student_table.heading("photo", text="PhotoSampleStatus")
        self.student_table["show"]="headings"

        self.student_table.column("dep",width=100)

        self.student_table.pack(fill=BOTH, expand=1)
        self.student_table.bind("<ButtonRelease>",self.get_cursor)
        self.fetch_data()


    #function declaration
    def add_data(self):
        if self.var_dep.get() == "Select Department" or self.var_std_name.get()=="" or self.var_std_id.get()=="":
            messagebox.showerror("Error", "All fields are required", parent = self.root)
        else:
            try:
                # messagebox.showinfo("success", "Saved!")
                conn=mysql.connector.connect(host="localhost", user = "root", password = "krisha123", database="face_recognition")
                my_cursor = conn.cursor()
                my_cursor.execute("insert into student values(%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",(
                                                                                                            self.var_dep.get(),
                                                                                                            self.var_course.get(),
                                                                                                            self.var_year.get(),
                                                                                                            self.var_semester.get(),
                                                                                                            self.var_std_id.get(),
                                                                                                            self.var_std_name.get(),
                                                                                                            self.var_div.get(),
                                                                                                            self.var_roll.get(),
                                                                                                            self.var_gender.get(),
                                                                                                            self.var_dob.get(),
                                                                                                            self.var_email.get(),
                                                                                                            self.var_phone.get(),
                                                                                                            self.var_address.get(),
                                                                                                            self.var_teacher.get(),
                                                                                                            self.var_radio1.get()
                                                                                                        ))
                conn.commit()
                self.fetch_data()
                conn.close()
                messagebox.showinfo("Success", "Student details has been added successfully", parent = self.root)
            except Exception as es:
                messagebox.showerror("Error", f"Due to: {str(es)}", parent = self.root)

    ###########fetch data#################
    def fetch_data(self):
        conn=mysql.connector.connect(host="localhost", user = "root", password = "krisha123", database="face_recognition")
        my_cursor = conn.cursor()
        my_cursor.execute("select * from student")
        data = my_cursor.fetchall()

        if len(data) != 0:
            self.student_table.delete(*self.student_table.get_children())
            for i in data:
                self.student_table.insert("",END,values=i)
            conn.commit()
        conn.close()


    ##########get cursor######################
    def get_cursor(self, event=""):
        cursor_focus=self.student_table.focus()
        content = self.student_table.item(cursor_focus)
        data = content["values"]

        self.var_dep.set(data[0]),
        self.var_course.set(data[1]),
        self.var_year.set(data[2]),
        self.var_semester.set(data[3]),
        self.var_std_id.set(data[4]),
        self.var_std_name.set(data[5]),
        self.var_div.set(data[6]),
        self.var_roll.set(data[7]),
        self.var_gender.set(data[8]),
        self.var_dob.set(data[9]),
        self.var_email.set(data[10]),
        self.var_phone.set(data[11]),
        self.var_address.set(data[12]),
        self.var_teacher.set(data[13]),
        self.var_radio1.set(data[14]),

    #######update function###########
    def update_data(self):
        if self.var_dep.get() == "Select Department" or self.var_std_name.get()=="" or self.var_std_id.get()=="":
            messagebox.showerror("Error", "All fields are required", parent = self.root)
        else:
            try:
                Update = messagebox.askyesno("Update", "Do you want to update this student details?", parent = self.root)
                if Update>0:
                    conn=mysql.connector.connect(host="localhost", user = "root", password = "krisha123", database="face_recognition")
                    my_cursor = conn.cursor()
                    my_cursor.execute("Update student set Dep=%s, courses = %s, Year = %s, Semester = %s, Name = %s, Division = %s, Roll = %s, Gender = %s, Dob = %s, Email = %s, Phone = %s, Address = %s, Teacher = %s, PhotoSample = %s where Student_id=%s",(
                                                                                                                                                                                                                                                    self.var_dep.get(),
                                                                                                                                                                                                                                                    self.var_course.get(),
                                                                                                                                                                                                                                                    self.var_year.get(),
                                                                                                                                                                                                                                                    self.var_semester.get(),
                                                                                                                                                                                                                                                    self.var_std_name.get(),
                                                                                                                                                                                                                                                    self.var_div.get(),
                                                                                                                                                                                                                                                    self.var_roll.get(),
                                                                                                                                                                                                                                                    self.var_gender.get(),
                                                                                                                                                                                                                                                    self.var_dob.get(),
                                                                                                                                                                                                                                                    self.var_email.get(),
                                                                                                                                                                                                                                                    self.var_phone.get(),
                                                                                                                                                                                                                                                    self.var_address.get(),
                                                                                                                                                                                                                                                    self.var_teacher.get(),
                                                                                                                                                                                                                                                    self.var_radio1.get(),
                                                                                                                                                                                                                                                    self.var_std_id.get()

                                                                                                                                                                                                                                             ))
                else:
                    if not Update:
                        return
                messagebox.showinfo("Success", "Update complete", parent = self.root)
                conn.commit()
                self.fetch_data()
                conn.close()

            except Exception as es:
                messagebox.showerror("Error",f"Due To: {str(es)}", parent = self.root)
                
    #####delete function########
    def delete_data(self):
        if self.var_std_id.get()=="":
            messagebox.showerror("Error", "Student id is required", parent = self.root)

        else:
            try:
                delete = messagebox.askyesno("Delete Info", "Do you want to delete this student?", parent = self.root)
                if delete>0:
                    conn=mysql.connector.connect(host="localhost", user = "root", password = "krisha123", database="face_recognition")
                    my_cursor = conn.cursor()
                    sql="delete from student where Student_id=%s"
                    val =(self.var_std_id.get(),)
                    my_cursor.execute(sql,val)
                else:
                    if not delete:
                        return
                
                conn.commit()
                self.fetch_data()
                conn.close()
                messagebox.showinfo("Delete", "Successfully deleted", parent = self.root)

            except Exception as es:
                messagebox.showerror("Error",f"Due To: {str(es)}", parent = self.root)
                   
    #####reset######
    def reset_data(self):
        self.var_dep.set("Select Department")
        self.var_course.set("Select Course")
        self.var_year.set("Select Year")
        self.var_semester.set("Select Semester")
        self.var_std_id.set("")
        self.var_std_name.set("")
        self.var_div.set("A")
        self.var_roll.set("")
        self.var_gender.set("Male")
        self.var_dob.set("")
        self.var_email.set("")
        self.var_phone.set("")
        self.var_address.set("")
        self.var_teacher.set("")
        self.var_radio1.set("")


    ######################## Generate data set or take photo samples#########################
    # def generate_dataset(self):
    #     if self.var_dep.get() == "Select Department" or self.var_std_name.get()=="" or self.var_std_id.get()=="":
    #         messagebox.showerror("Error", "All fields are required", parent = self.root)
    #     else:
    #         try:
    #             conn=mysql.connector.connect(host="localhost", user = "root", password = "krisha123", database="face_recognition")
    #             my_cursor = conn.cursor()
    #             my_cursor.execute("select * from student")
    #             myresult=my_cursor.fetchall()
    #             id=0
    #             for x in myresult:
    #                 id+=1
    #             my_cursor.execute("Update student set Dep=%s, courses = %s, Year = %s, Semester = %s, Name = %s, Division = %s, Roll = %s, Gender = %s, Dob = %s, Email = %s, Phone = %s, Address = %s, Teacher = %s, PhotoSample = %s where Student_id=%s",(
    #                                                                                                                                                                                                                                                 self.var_dep.get(),
    #                                                                                                                                                                                                                                                 self.var_course.get(),
    #                                                                                                                                                                                                                                                 self.var_year.get(),
    #                                                                                                                                                                                                                                                 self.var_semester.get(),
    #                                                                                                                                                                                                                                                 self.var_std_name.get(),
    #                                                                                                                                                                                                                                                 self.var_div.get(),
    #                                                                                                                                                                                                                                                 self.var_roll.get(),
    #                                                                                                                                                                                                                                                 self.var_gender.get(),
    #                                                                                                                                                                                                                                                 self.var_dob.get(),
    #                                                                                                                                                                                                                                                 self.var_email.get(),
    #                                                                                                                                                                                                                                                 self.var_phone.get(),
    #                                                                                                                                                                                                                                                 self.var_address.get(),
    #                                                                                                                                                                                                                                                 self.var_teacher.get(),
    #                                                                                                                                                                                                                                                 self.var_radio1.get(),
    #                                                                                                                                                                                                                                                 self.var_std_id.get()==id+1

    #                                                                                                                                                                                                                                          ))
    #             conn.commit()
    #             self.fetch_data()
    #             self.reset_data()
    #             conn.close()

    #             #################Load Predefinied data on face frontals from opencv###################################
    #             face_classifier=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

    #             def face_cropped(img):
    #                 gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #                 faces=face_classifier.detectMultiScale(gray, 1.3, 5)
    #                 #scaling factor = 1.3 (by default)
    #                 # Minimim Neighbour = 5

    #                 for (x,y,w,h) in faces:
    #                     face_cropped=img[y:y+h, x:x+w] 
    #                     return face_cropped

    #                 cap = cv2.VideoCapture(0) 
    #                 img_id = 0
    #                 while True:
    #                     ret,my_frame=cap.read()
    #                     if face_cropped(my_frame) is not None:
    #                         img_id+=1
    #                         face = cv2.resize(face_cropped(my_frame),(450,450))
    #                         face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    #                         file_name_path = "data/user."+str(id)+"."+str(img_id)+".jpg"
    #                         cv2.imwrite(file_name_path,face)
    #                         cv2.putText(face,str(img_id),(50,50), cv2.FONT_HERSHEY_COMPLEX,2,(0,255,0),2)
    #                         cv2.imshow("Cropped Face", face)

    #                     if cv2.waitKey(1)==13 or int(img_id)==100:
    #                         break
    #                 cap.release()
    #                 cv2.destroyAllWindows()
    #                 messagebox.showinfo("Result", "Generating data sets completed!")

    #         except Exception as es:
    #             messagebox.showerror("Error",f"Due To: {str(es)}", parent = self.root)

    def generate_dataset(self):
        if self.var_dep.get() == "Select Department" or self.var_std_name.get() == "" or self.var_std_id.get() == "":
            messagebox.showerror("Error", "All fields are required", parent=self.root)
            return

        try:
            conn = mysql.connector.connect(host="localhost", user="root", password="krisha123", database="face_recognition")
            my_cursor = conn.cursor()
            my_cursor.execute("SELECT * FROM student")
            myresult = my_cursor.fetchall()
            id = len(myresult) + 1  # Set ID properly

            my_cursor.execute("""
                UPDATE student 
                SET Dep=%s, courses=%s, Year=%s, Semester=%s, Name=%s, Division=%s, Roll=%s, 
                    Gender=%s, Dob=%s, Email=%s, Phone=%s, Address=%s, Teacher=%s, PhotoSample=%s 
                WHERE Student_id=%s
            """, (
                self.var_dep.get(),
                self.var_course.get(),
                self.var_year.get(),
                self.var_semester.get(),
                self.var_std_name.get(),
                self.var_div.get(),
                self.var_roll.get(),
                self.var_gender.get(),
                self.var_dob.get(),
                self.var_email.get(),
                self.var_phone.get(),
                self.var_address.get(),
                self.var_teacher.get(),
                self.var_radio1.get(),
                self.var_std_id.get()
            ))

            conn.commit()
            self.fetch_data()
            self.reset_data()
            conn.close()

            # Load Haarcascade Face Classifier
            face_classifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

            def face_cropped(img):
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                faces = face_classifier.detectMultiScale(gray, 1.3, 5)

                for (x, y, w, h) in faces:
                    face_cropped = img[y:y+h, x:x+w]
                    return face_cropped
                return None  # Return None if no face is detected

            cap = cv2.VideoCapture(0)  # Open camera
            img_id = 0

            while True:
                ret, my_frame = cap.read()
                if not ret:
                    messagebox.showerror("Error", "Failed to access the camera", parent=self.root)
                    break

                cropped_face = face_cropped(my_frame)
                if cropped_face is not None:
                    img_id += 1
                    face = cv2.resize(cropped_face, (450, 450))
                    face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                    file_name_path = f"data/user.{id}.{img_id}.jpg"
                    cv2.imwrite(file_name_path, face)

                    cv2.putText(face, str(img_id), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 255, 0), 2)
                    cv2.imshow("Cropped Face", face)

                if cv2.waitKey(1) == 13 or img_id == 100:  # Press ENTER (13) to stop
                    break

            cap.release()
            cv2.destroyAllWindows()
            messagebox.showinfo("Result", "Generating data sets completed!", parent=self.root)

        except Exception as es:
            messagebox.showerror("Error", f"Due To: {str(es)}", parent=self.root)



    # def generate_maskset(self):
    #     if self.var_dep.get() == "Select Department" or self.var_std_name.get() == "" or self.var_std_id.get() == "":
    #         messagebox.showerror("Error", "All fields are required", parent=self.root)
    #         return

    #     try:
    #         # Database connection
    #         conn = mysql.connector.connect(
    #             host="localhost",
    #             user="root",
    #             password="krisha123",
    #             database="face_recognition"
    #         )
    #         my_cursor = conn.cursor()
            
    #         # Get the actual student ID from the form
    #         student_id = self.var_std_id.get()

    #         # Update student record
    #         my_cursor.execute("""
    #             UPDATE student 
    #             SET Dep=%s, courses=%s, Year=%s, Semester=%s, Name=%s, Division=%s, Roll=%s, 
    #                 Gender=%s, Dob=%s, Email=%s, Phone=%s, Address=%s, Teacher=%s, PhotoSample=%s 
    #             WHERE Student_id=%s
    #         """, (
    #             self.var_dep.get(),
    #             self.var_course.get(),
    #             self.var_year.get(),
    #             self.var_semester.get(),
    #             self.var_std_name.get(),
    #             self.var_div.get(),
    #             self.var_roll.get(),
    #             self.var_gender.get(),
    #             self.var_dob.get(),
    #             self.var_email.get(),
    #             self.var_phone.get(),
    #             self.var_address.get(),
    #             self.var_teacher.get(),
    #             self.var_radio1.get(),
    #             student_id  # Use the actual student ID here
    #         ))

    #         conn.commit()
    #         self.fetch_data()
    #         self.reset_data()
    #         conn.close()

    #         # Face detection setup
    #         import os
    #         os.makedirs("mask", exist_ok=True)
    #         detector = MTCNN(min_face_size=50, steps_threshold=[0.5, 0.7, 0.9])

    #         def face_cropped(img):
    #             result = detector.detect_faces(img)
    #             if result:
    #                 x, y, w, h = result[0]['box']
    #                 # Adjust for negative coordinates
    #                 x, y = max(0, x), max(0, y)
    #                 return img[y:y+h, x:x+w], (x, y, w, h)
    #             return None, (0, 0, 0, 0)

    #         # Camera capture
    #         cap = cv2.VideoCapture(0)
    #         img_id = 0

    #         while True:
    #             ret, frame = cap.read()
    #             if not ret:
    #                 messagebox.showerror("Error", "Camera error", parent=self.root)
    #                 break

    #             frame = cv2.flip(frame, 1)  # Mirror effect
    #             cropped_face, (x, y, w, h) = face_cropped(frame)
                
    #             if cropped_face is not None:
    #                 img_id += 1
    #                 face = cv2.resize(cropped_face, (224, 224))
    #                 file_name_path = f"mask/user.{student_id}.{img_id}.jpg"
    #                 cv2.imwrite(file_name_path, face)

    #                 # Display feedback
    #                 cv2.putText(frame, f"Saved: {img_id}/50", (10, 30), 
    #                             cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    #                 cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
    #             cv2.imshow("Mask Dataset Collection", frame)
                
    #             # Exit on ESC key or after 50 images
    #             if cv2.waitKey(1) == 27 or img_id >= 50:
    #                 break

    #         cap.release()
    #         cv2.destroyAllWindows()
    #         messagebox.showinfo("Success", f"50 masked face samples saved for student ID: {student_id}", parent=self.root)

    #     except Exception as es:
    #         messagebox.showerror("Error", f"Database/Camera Error: {str(es)}", parent=self.root)

if __name__ == "__main__":
    root= Tk()
    obj = Student(root)
    root.mainloop()