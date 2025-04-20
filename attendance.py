from tkinter import*
from tkinter import ttk
from PIL import Image, ImageTk
from tkinter import messagebox
import mysql.connector
import cv2
import os
import csv
from tkinter import filedialog

#global variable
mydata=[]

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

        img_left = Image.open(r"my_images\details.jpeg")
        img_left = img_left.resize((720, 130),Image.ANTIALIAS)
        self.photoimg_left=ImageTk.PhotoImage(img_left)
        
        f_lbl = Label(Left_frame,image=self.photoimg_left)
        f_lbl.place(x=5,y=0,width=720,height=130) 

        left_inside_frame=Frame(Left_frame, bd=2, relief=RIDGE ,bg="white")
        left_inside_frame.place(x=10, y=135, width=720, height=370)

        ######Labels and entry#######
        #Attendance  ID
        attendanceID_label = Label(left_inside_frame, text="AttendanceId", font = ("times new roman", 12, "bold"), bg="white")
        attendanceID_label.grid(row=0,column=0, padx=10, pady=5, sticky  =W)

        attendanceID_entry = ttk.Entry(left_inside_frame, width=20,font = ("times new roman", 12, "bold"))
        attendanceID_entry.grid(row=0, column=1, padx=10, pady=5, sticky = W)

        #  Roll
        rollLabel = Label(left_inside_frame, text="Roll:", font = ("times new roman", 12, "bold"), bg="white")
        rollLabel.grid(row=0,column=2, padx=10, pady=8, sticky  =W)

        atten_roll = ttk.Entry(left_inside_frame, width=20,font = ("times new roman", 12, "bold"))
        atten_roll.grid(row=0, column=3, padx=10, pady=8, sticky = W)

        #   Name
        nameLabel = Label(left_inside_frame, text="Name:", font = ("times new roman", 12, "bold"), bg="white")
        nameLabel.grid(row=1,column=0, padx=10, pady=8, sticky  =W)

        atten_name = ttk.Entry(left_inside_frame, width=20,font = ("times new roman", 12, "bold"))
        atten_name.grid(row=1, column=1, padx=10, pady=8, sticky = W)

        #   Department
        depLabel = Label(left_inside_frame, text="Department:", font = ("times new roman", 12, "bold"), bg="white")
        depLabel.grid(row=1,column=2, padx=10, pady=8, sticky  =W)

        atten_dep = ttk.Entry(left_inside_frame, width=20,font = ("times new roman", 12, "bold"))
        atten_dep.grid(row=1, column=3, padx=10, pady=8, sticky = W)

        #   time
        timeLabel = Label(left_inside_frame, text="Time:", font = ("times new roman", 12, "bold"), bg="white")
        timeLabel.grid(row=2,column=0, padx=10, pady=8, sticky  =W)

        atten_time = ttk.Entry(left_inside_frame, width=20,font = ("times new roman", 12, "bold"))
        atten_time.grid(row=2, column=1, padx=10, pady=8, sticky = W)

        #   date
        dateLabel = Label(left_inside_frame, text="Date:", font = ("times new roman", 12, "bold"), bg="white")
        dateLabel.grid(row=2,column=2, padx=10, pady=8, sticky  =W)

        atten_date = ttk.Entry(left_inside_frame, width=20,font = ("times new roman", 12, "bold"))
        atten_date.grid(row=2, column=3, padx=10, pady=8, sticky = W)

        #   Attendance
        attendanceLabel = Label(left_inside_frame, text="Attendance Status", font = ("times new roman", 12, "bold"), bg="white")
        attendanceLabel.grid(row=3,column=0, padx=10, pady=5, sticky  =W)

        self.atten_status = ttk.Combobox(left_inside_frame,font = ("times new roman", 12, "bold"), width = 15,  state="readonly")
        self.atten_status["values"] = ("Status", "Present", "Absent")
        self.atten_status.grid(row=3 ,column=1, padx=10, pady=5, sticky = W)
        self.atten_status.current(0)

        #buttons Frame
        btn_frame = Frame(left_inside_frame, bd=2, relief=RIDGE, bg="white")
        btn_frame.place(x=0, y=300, width=715, height=35)

        #Import
        import_btn=Button(btn_frame, text="Import CSV", command=self.importCsv,width = 19, font = ("times new roman", 12, "bold"), bg="blue", fg="white")
        import_btn.grid(row=0, column=0)

        # Export
        export_btn=Button(btn_frame, text="Export CSV", width = 19, font = ("times new roman", 12, "bold"), bg="blue", fg="white")
        export_btn.grid(row=0, column=1)

        # Update
        update_btn=Button(btn_frame, text="Update", width = 19, font = ("times new roman", 12, "bold"), bg="blue", fg="white")
        update_btn.grid(row=0, column=2)

        # Reset
        reset_btn=Button(btn_frame, text="Reset", width = 19, font = ("times new roman", 12, "bold"), bg="blue", fg="white")
        reset_btn.grid(row=0, column=3)

        # right label frame
        Right_frame = LabelFrame(main_frame, bd=2, bg="white", relief=RIDGE, text="Attendance Details", font=("times new roman", 12, "bold"))
        Right_frame.place(x=780, y=10, width=680, height=580)

        table_frame = Frame(Right_frame, bd=2, relief=RIDGE, bg="white")
        table_frame.place(x=5, y=5, width=665, height=455)

        ########scrollbar table###############################
        scroll_x = ttk.Scrollbar(table_frame,orient=HORIZONTAL)
        scroll_Y = ttk.Scrollbar(table_frame,orient=VERTICAL)

        self.AttendanceReportTable=ttk.Treeview(table_frame,column=("id", "roll", "name", "department", "time", "date", "attendance"),xscrollcommand=scroll_x.set,yscrollcommand=scroll_Y.set)

        scroll_x.pack(side=BOTTOM, fill=X)
        scroll_Y.pack(side=RIGHT, fill=Y)

        scroll_x.config(command=self.AttendanceReportTable.xview)
        scroll_Y.config(command=self.AttendanceReportTable.yview)

        self.AttendanceReportTable.heading("id",text="Attendance ID")
        self.AttendanceReportTable.heading("roll",text="Roll")
        self.AttendanceReportTable.heading("name",text="Name")
        self.AttendanceReportTable.heading("department",text="Department")
        self.AttendanceReportTable.heading("time",text="Time")
        self.AttendanceReportTable.heading("date",text="Date")
        self.AttendanceReportTable.heading("attendance",text="Attendance")

        ### removed space
        self.AttendanceReportTable["show"] = "headings"
        self.AttendanceReportTable.column("id", width=100)
        self.AttendanceReportTable.column("roll", width=100)
        self.AttendanceReportTable.column("name", width=100)
        self.AttendanceReportTable.column("department", width=100)
        self.AttendanceReportTable.column("time", width=100)
        self.AttendanceReportTable.column("date", width=100)
        self.AttendanceReportTable.column("attendance", width=100)

        self.AttendanceReportTable.pack(fill=BOTH, expand=1)

        f_lbl = Label(self.root,image=self.photoimg1)
        f_lbl.place(x=800,y=0,width=800,height=180) 


    ########################fetch data###############################

    def fetchData(self, rows):
        self.AttendanceReportTable.delete(*self.AttendanceReportTable.get_children())
        for i in rows:
            self.AttendanceReportTable.insert("", END, values=i)

    def importCsv(self):
        global mydata
        fln = filedialog.askopenfilename(initialdir=os.getcwd(),title="Open CSV", filetypes=(("CSV file", "*.csv"),("All File","*.*")),parent=self.root)
        with open(fln) as myfile:
            csvread= csv.reader(myfile,delimiter=",")
            for i in csvread:
                mydata.append(i)
            self.fetchData(mydata)





        
        

if __name__ == "__main__":
    root= Tk()
    obj = Attendance(root)
    root.mainloop()