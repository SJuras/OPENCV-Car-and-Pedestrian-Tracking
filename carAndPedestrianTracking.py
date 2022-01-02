import cv2

img_file = "Assets/carFreeway.jpg"
classifier_file = "car_detector.xml"

img = cv2.imread(img_file)
imgBlack_n_White = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

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

print("Code Executed")
