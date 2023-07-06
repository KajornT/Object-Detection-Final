#import libraryที่จำเป็น
import tkinter as tk            #pip install tk
from tkinter import filedialog  #pip install tk
import numpy as np              #pip install numpy
import cv2                      #pip install numpy
from tkinter import ttk         #pip install tk

def open_file_dialog():
    file_path = filedialog.askopenfilename()
    if file_path:
        if file_path.lower().endswith(('.jpg', '.jpeg', '.png')):
            detect_image(file_path)
        else:
            process_video(file_path)

def open_camera():
    process_video(0)  # 0 คือเปิดกล้อง

def process_video(file_path):
    #รายชื่อหมวดหมู่ทั้งหมด เรียงตามลำดับ
    CLASSES = ["BACKGROUND", "AEROPLANE", "BICYCLE", "BIRD", "BOAT",
               "BOTTLE", "BUS", "CAR", "CAT", "CHAIR", "COW", "DININGTABLE",
               "DOG", "HORSE", "MOTORBIKE", "PERSON", "POTTEDPLANT", "SHEEP",
               "SOFA", "TRAIN", "TVMONITOR"]
    
    #สีตัวกรอบที่วาดrandomใหม่ทุกครั้ง
    COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 4))

    #โหลดmodelจากโฟลเดอร์
    net = cv2.dnn.readNetFromCaffe("./MobileNetSSD/MobileNetSSD.prototxt", "./MobileNetSSD/MobileNetSSD.caffemodel")

    cap = cv2.VideoCapture(file_path)

    while True:
        #เริ่มอ่านในแต่ละเฟรม
        ret, frame = cap.read()
        if ret:
            (h, w) = frame.shape[:2]
            #ทำpreprocessing
            blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5)
            net.setInput(blob)
            #feedเข้าmodelพร้อมได้ผลลัพธ์ทั้งหมดเก็บมาในตัวแปร detections
            detections = net.forward()

            for i in np.arange(0, detections.shape[2]):
                percent = detections[0, 0, i, 2]
                #กรองเอาเฉพาะค่าpercentที่สูงกว่า0.5 เพิ่มลดได้ตามต้องการ
                if percent > 0.5:
                    class_index = int(detections[0, 0, i, 1])
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")

                    #ส่วนตกแต่งสามารถลองแก้กันได้ วาดกรอบและชื่อ    
                    label = "{} [{:.2f}%]".format(CLASSES[class_index], percent * 100)
                    cv2.rectangle(frame, (startX, startY), (endX, endY), COLORS[class_index], 5)
                    cv2.rectangle(frame, (startX - 1, startY - 30), (endX + 1, startY), COLORS[class_index], cv2.FILLED)
                    y = startY - 15 if startY - 15 > 15 else startY + 15
                    cv2.putText(frame, label, (startX + 20, y + 5), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)

            # ปรับขนาดกรอบให้พอดีกับขนาดของหน้าต่าง
            frame = cv2.resize(frame, (800, 600))

            cv2.imshow("Frame", frame)
            if cv2.waitKey(1) == 27:  # หยุดการลูปซ้ำเมื่อกดปุ่ม 'Esc' (Code ASCII 27)
                break

        else:
            break

    #หลังเลิกใช้แล้วเคลียร์memoryและปิดกล้อง
    cap.release()
    cv2.destroyAllWindows()

def detect_image(file_path):
    #รายชื่อหมวดหมู่ทั้งหมด เรียงตามลำดับ
    CLASSES = ["BACKGROUND", "AEROPLANE", "BICYCLE", "BIRD", "BOAT",
               "BOTTLE", "BUS", "CAR", "CAT", "CHAIR", "COW", "DININGTABLE",
               "DOG", "HORSE", "MOTORBIKE", "PERSON", "POTTEDPLANT", "SHEEP",
               "SOFA", "TRAIN", "TVMONITOR"]

    COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 4))

    net = cv2.dnn.readNetFromCaffe("./MobileNetSSD/MobileNetSSD.prototxt", "./MobileNetSSD/MobileNetSSD.caffemodel")

    frame = cv2.imread(file_path)

    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()

    for i in np.arange(0, detections.shape[2]):
        percent = detections[0, 0, i, 2]
        if percent > 0.5:
            class_index = int(detections[0, 0, i, 1])
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            label = "{} [{:.2f}%]".format(CLASSES[class_index], percent * 100)
            cv2.rectangle(frame, (startX, startY), (endX, endY), COLORS[class_index], 20)
            cv2.rectangle(frame, (startX - 1, startY - 150), (endX + 1, startY + 90), COLORS[class_index], cv2.FILLED)
            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(frame, label, (startX + 20, y + 5), cv2.FONT_HERSHEY_DUPLEX, 5.0, (255, 255, 255), 4)

    # ปรับขนาดกรอบให้พอดีกับขนาดของหน้าต่าง
    frame = cv2.resize(frame, (800, 600))

    cv2.imshow("Image", frame)
    cv2.waitKey(0)


#หน้าต่างเริ่ม
root = tk.Tk()
root.title("Object detection")
root.geometry("700x400")


root.iconbitmap("Assets/icon.ico") #ไอคอน

#รูปLOGO
logo_image = tk.PhotoImage(file="Assets/logo.png")
logo_label = tk.Label(root, image=logo_image)
logo_label.pack(pady=10)

#กำหนดฟังก์ชั่นเปลี่ยนธีม
def toggle_theme():
    if root["bg"] == "white":
        root["bg"] = "#202124"  # Dark mode background color
        button.configure(bg="#202124", fg="white")
        camera_button.configure(bg="#202124", fg="white")
        toggle_button.configure(bg="#202124", fg="white")
    else:
        root["bg"] = "white"  # Light mode background color
        button.configure(bg="white", fg="black")
        camera_button.configure(bg="white", fg="black")
        toggle_button.configure(bg="white", fg="black")


#ปุ่มทั้งหมด
button = tk.Button(root, text="เลือกไฟล์(วีดีโอ, ภาพ)", command=open_file_dialog, bg="white", fg="black")
button.pack(pady=5)

camera_button = tk.Button(root, text="เปิดกล้อง", command=open_camera, bg="white", fg="black")
camera_button.pack(pady=20)

toggle_button = tk.Button(root, text="เปลี่ยนธีม", command=toggle_theme, bg="white", fg="black")
toggle_button.pack()

root.protocol("WM_DELETE_WINDOW", root.destroy)

toggle_theme()

root.mainloop()