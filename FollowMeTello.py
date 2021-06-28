from djitellopy import Tello
import cv2 as cv
import numpy as np
import argparse
import dlib

me = Tello()
#me.connect()
#me.streamon()
#me.get_frame_read()
desiredFaceWidth = 256
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("lbfmodel.dat")


#Detects and displays a box around faces
def detectAndDisplay(frame):
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    frame_gray = cv.equalizeHist(frame_gray)
    faces = face_cascade.detectMultiScale(frame_gray)
    #For each detected face, draw a box around it.
    for (x,y,w,h) in faces:
        frame = cv.rectangle(frame, (x,y), (x+w,y+h), (255,0,255))
        faceROI = frame_gray[y:y+h, x:x+w]
    cv.imshow('Capture  - Face detection', frame)

#Detects and displays the facial landmarks
#Will be necessary for pose estimation
def displayLandmarks(frame):
    #Resizing the video feed for the sake of dlib optimization
    resize = cv.resize(frame, (320,180))
    gray = cv.cvtColor(resize, cv.COLOR_BGR2GRAY)
    dlibFaces = detector(gray)
    #For each detected face, draw circles on all landmarks.
    for face in dlibFaces:
      points = predictor(gray, face)
      for n in range(68):
           frame = cv.circle(frame, (points.part(n).x*4, points.part(n).y*4), 5, (50,50,255), cv.FILLED)
    cv.imshow('Capture  - Landmark detection', frame)

#Used to calculate where the drone is respective
#to the face in 3D space.
def repositionDrone(cap):
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    frame_gray = cv.equalizeHist(frame_gray)
    faces = face_cascade.detectMultiScale(frame_gray)
    #Finds the center of the frame capture
    capCenter = (cap.get(cv.CAP_PROP_FRAME_WIDTH)/2, cap.get(cv.CAP_PROP_FRAME_HEIGHT)/2)
    center = (capCenter)
    width = 0
    #Finds the center of the face capture
    #along with it's respective width
    for (x,y,w,h) in faces:
        center = (x+w//2, y+h//2)
    for (w) in faces:
        width = max(w)
    
    #Moves the drone in the necessary direction
    if (center[0] > capCenter[0]):
        #me.move_right(20)
        print("moving right")
    if (center[0] < capCenter[0]):
        #me.move_left(20)
        print("moving left")
    if (center[1] > capCenter[1]):
        #me.move_down(20)
        print("moving down")
    if (center[1] < capCenter[1]):
        #me.move_up(20)
        print("moving up")
    if (width > desiredFaceWidth):
        #me.move_backward(20)
        print("moving back")
    if (width < desiredFaceWidth):
        #me.move_forward(20)
        print("moving forward")

#Parsing the haarcascade
parser = argparse.ArgumentParser(description='Code for Cascade Classifier tutorial.')
parser.add_argument('--face_cascade', help='Path to face cascade.', default='D:\Program Files (x86)\OpenCV\opencv\sources\data\haarcascades/haarcascade_frontalface_alt.xml')
args = parser.parse_args()

#Setting up other necessary variables
face_cascade_name = args.face_cascade
face_cascade = cv.CascadeClassifier()

#If we can't load the cascade, end the program.
if not face_cascade.load(cv.samples.findFile(face_cascade_name)):
    print('Error loading face cascade')
    exit(0)

#Create a video capture. If we can't, end the program.
cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()
while True:
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame. Exiting.")
        break
    detectAndDisplay(frame)
    displayLandmarks(frame)
    repositionDrone(cap)
    if cv.waitKey(1) == ord('q'):
        break

cap.release()
cv.destroyAllWindows()


