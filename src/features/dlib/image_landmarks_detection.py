# https://www.pyimagesearch.com/2018/01/22/install-dlib-easy-complete-guide/

# import the necessary packages
import csv

import cv2
import dlib
import pandas as pd
from imutils import face_utils

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
p = "../../../models/shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(p)
# load the input image and convert it to grayscale
image = cv2.imread("../../../data/examples/Keanu.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# detect faces in the grayscale image
rects = detector(gray, 0)
# loop over the face detections
for (i, rect) in enumerate(rects):
    # determine the facial landmarks for the face region, then
    # convert the facial landmark (x, y)-coordinates to a NumPy
    # array
    shape = predictor(gray, rect)
    shape = face_utils.shape_to_np(shape)
    # loop over the (x, y)-coordinates for the facial landmarks
    # and draw them on the image
    for (x, y) in shape:
        cv2.circle(image, (x, y), 2, (0, 255, 0), -1)
# show the output image with the face detections + facial landmarks
cv2.imshow("Output", image)
cv2.waitKey(0)
cv2.imwrite("../../../data/processed/examples/dlib.jpg", image)
# cv2.imshow("Output", image)
# cv2.waitKey(0)
