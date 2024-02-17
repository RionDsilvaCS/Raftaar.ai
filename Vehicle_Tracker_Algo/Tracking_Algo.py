import cv2
from ultralytics import YOLO
import time

class YOLOVehicleTracking():
    """
    Here we use YOLOV8 for tracking the vehicles and specifically yolov8n which is
    the small and fastest inferencing model which requires less computational power 
    while compared to other s, l, x models etc.
    """
    def __init__(self, first_line, second_line, distance) -> None:
        
        # Importing yolov model for tracking
        self.model = YOLO('../Models/yolov8n.pt')

        # consider the below car_dir dictionary as storing the information in a database about the vehicls passed
        self.vehicle_dir = {0.0:{
            "s_t": 1,
            "e_t": 2,
            "speed": None
            }
        }   

        """
        Below first_line and second_line are the virtual co-ordinates to measure the speed
        and distance is the measure of real world distance between both lines in meters[m] 
        """
        self.first_line = first_line
        self.second_line = second_line
        self.distance = distance

    def Tracking(self, frame):
        """
        Function to track the vehicls 
        """
        # model tracking classes=1, 2, 3,5,7 constraints the model for cars and bikes
        results = self.model.track(frame, persist=True, classes=[1, 2, 3,5,7])
        veh_id_tensor = results[0].boxes.id
        veh_id =  veh_id_tensor.tolist()
        center_pt_tensor = results[0].boxes.xywh
        center_pt = center_pt_tensor.tolist()

        for i in range(0, len(center_pt)):
            pt = center_pt[i]
            c_id = veh_id[i]

            cord = (int(pt[0]), int(pt[1]))
            cv2.circle(frame, cord, 5, (125, 225, 125), -1) 
            cv2.putText(frame, str(c_id), cord, cv2.FONT_HERSHEY_SIMPLEX,  0.5, (125, 125, 225), 1, cv2.LINE_AA) 

            if c_id not in self.vehicle_dir:
                self.vehicle_dir[c_id] = {
                    "s_t": 0
                }
                
            if cord[1] > self.first_line[1][1] and self.vehicle_dir[c_id]["s_t"] == 0:
                self.vehicle_dir[c_id] = {
                    "s_t": 0,
                    "e_t": 0,
                    "speed": None
                }

                start_time = time.time()
                
                txt = f'{c_id} passed first line'
                cv2.putText(frame, txt, (50, 50), cv2.FONT_HERSHEY_SIMPLEX,  1, (125, 225, 225), 1, cv2.LINE_AA)
                self.vehicle_dir[c_id]["s_t"] = start_time

                print("start ",self.vehicle_dir)

            if cord[1] > self.second_line[1][1] and self.vehicle_dir[c_id]["e_t"] == 0:
                end_time = time.time()
                
                txt = f'{c_id} passed second line'
                cv2.putText(frame, txt, (50, 50), cv2.FONT_HERSHEY_SIMPLEX,  1, (125, 225, 225), 1, cv2.LINE_AA)
                self.vehicle_dir[c_id]["e_t"] = end_time

                try:
                    speed = (self.distance / (self.vehicle_dir[c_id]["e_t"] - self.vehicle_dir[c_id]["s_t"])) * 3.6
                except(ZeroDivisionError):
                    speed = 0

                self.vehicle_dir[c_id]["speed"] = int(speed)
                print("done ", self.vehicle_dir)
        
        return frame