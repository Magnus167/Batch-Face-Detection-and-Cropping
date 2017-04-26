import cv2
import sys
import glob 

cascPath = "./haarcascade_frontalface_alt.xml"
#cascPath = "./haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)

files=glob.glob("*.JPG")   
for file in files:

    image = cv2.imread(file)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    left = 125      #left from center
    right = 125     #right from center
    top = 175       #top from center   
    bottom = 75     #bottom from center

    for (x, y, w, h) in faces:
      print (x, y, w, h)

    image  = image[y-top:y+h+bottom, x-left:x+w+right]

    cv2.imwrite("cropped_{1}_{0}".format(str(file),str(x)), image)

