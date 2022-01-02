import cv2

img_file = "Assets/carFreeway.jpg"
classifier_file = "car_detector.xml"

img = cv2.imread(img_file)
imgBlack_n_White = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# car classifier
car_tracker = cv2.CascadeClassifier(classifier_file)

# show image
cv2.imshow("Display", imgBlack_n_White)
cv2.waitKey(0)

print("Code Executed")
