import cv2
from Tracking_Algo import YOLOVehicleTracking   
import torch

# Pre saved video path or live camera feed link
video_path = "s1.mp4"
cap = cv2.VideoCapture(video_path)

"""
Below first_line and second_line are the virtual co-ordinates to measure the speed
and distance is the measure of real world distance between both lines in meters[m] 
"""
first_line = [(150, 100), (550, 100)]
second_line = [(100, 260), (600, 260)]
distance = 40

# Declaring tracker
Tracker = YOLOVehicleTracking(first_line=first_line, second_line=second_line, distance=distance)


while cap.isOpened():

    success, frame = cap.read()

    if success:

        frame = Tracker.Tracking(frame=frame)

        cv2.line(frame, first_line[0], first_line[1], (225, 0, 0), 5)
        cv2.line(frame, second_line[0], second_line[1], (225, 0, 0), 5)

        cv2.imshow("Vehicle Tracking", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()