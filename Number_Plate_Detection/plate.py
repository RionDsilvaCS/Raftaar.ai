import cv2

def NumberPlateDetection(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    n_plate_detector = cv2.CascadeClassifier("../Models/haarcascade_russian_plate_number.xml")
    detections = n_plate_detector.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=7)

    for (x, y, w, h) in detections:
        number_plate = gray[y:y + h, x:x + w]
        cv2.imwrite("detected.png", number_plate)


if "__main__" == __name__:

    image = cv2.imread("Cars9.png")
    
    NumberPlateDetection(image=image)

