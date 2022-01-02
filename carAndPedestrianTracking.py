import cv2

img_file = "Assets/carFreeway.jpg"
video = cv2.VideoCapture(0)

classifier_file = "car_detector.xml"
pedestrian_file = "haarcascade_fullbody.xml"

car_tracker = cv2.CascadeClassifier(classifier_file)
pedestrian_tracker = cv2.CascadeClassifier(pedestrian_file)


while True:
    (read_successful, frame) = video.read()

    if read_successful:
        grayscaled_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        break

    cars = car_tracker.detectMultiScale(grayscaled_frame)
    pedestrians = pedestrian_tracker.detectMultiScale(grayscaled_frame)

    for (x, y, w, h) in cars:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
        cv2.putText(frame, "Car", (x,y-5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)

    for (x, y, w, h) in pedestrians:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
        cv2.putText(frame, "Pedestrian", (x,y-5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 255), 1)

    cv2.imshow("Display Video", frame)
    cv2.waitKey(0)

video.release()

"""
# img = cv2.imread(img_file)
# imgBlack_n_White = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# car classifier
car_tracker = cv2.CascadeClassifier(classifier_file)

# detect cars
cars = car_tracker.detectMultiScale(imgBlack_n_White)
# print(cars)
# draw a rectangle
for (x, y, w, h) in cars:
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)
    cv2.putText(img, "Car", (x,y-5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)

# show image
cv2.imshow("Display", img)
cv2.waitKey(0)
"""

print("Code Executed")
