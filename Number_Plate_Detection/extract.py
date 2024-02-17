import cv2
from ultralytics import YOLO
import torch
import time
import multiprocessing
import imutils
import numpy as np
import pytesseract

video_path = "ss1.mp4"
# video_path = 'http://172.19.58.221:2204/video_feed_1'

cap = cv2.VideoCapture(video_path)

model = YOLO('yolov8n.pt')
dir_r = "C:/Users/dsilv/OneDrive/Desktop/proj/Raftaar.ai/imgs/"
i = 0
pytesseract.pytesseract.tesseract_cmd = ("C:\\Program Files\\Tesseract-OCR\\tesseract.exe")

def save_plate(img, i):
    if i%4 == 0:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        n_plate_detector = cv2.CascadeClassifier("haarcascade_russian_plate_number.xml")
        detections = n_plate_detector.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=7)

        for (x, y, w, h) in detections:
            number_plate = gray[y:y + h, x:x + w]
            
            text = pytesseract.image_to_string(number_plate, config='--psm 11')
            if len(text) > 6:
                print("Detected license plate Number is:",text)
                save_dir = f'{dir_r}detect_{i}.png'
                cv2.imwrite(save_dir, number_plate)

while cap.isOpened():
    success, frame = cap.read()

    if success:

        results = model.track(frame, persist=True, classes=[1, 2, 3,5,7])
        
        box_pt_tensor = results[0].boxes.xyxy
        box_pt = box_pt_tensor.tolist()
        if len(box_pt):
            for box in box_pt:
                try:
                    # frame = cv2.rectangle(frame, (int(box[0]-box[2]), int(box[1]-box[3])), (int(box[0]+box[2]), int(box[1]+box[3])), (255, 125, 125), 2) 
                    frame = cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 125, 125), 2)

                    save_plate(frame, i)

                except(TypeError, ValueError): # , ValueError
                    pass
        cv2.imshow("YOLOv8 Tracking", frame)

        i+=1

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break


cap.release()
cv2.destroyAllWindows()