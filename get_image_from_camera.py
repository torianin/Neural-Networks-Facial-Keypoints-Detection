import cv2
import sys
from scipy import misc
import numpy as np
import matplotlib.pyplot as plt
from utils import load_model

cascPath = 'haarcascade_frontalface_default.xml'
faceCascade = cv2.CascadeClassifier(cascPath)

video_capture = cv2.VideoCapture(0)

counter = 0

Resized_image_size = (96, 96) 
Min_face_size = (90, 90)

net = load_model('main_net.pickle')

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if(counter == 9 ):
        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=Min_face_size,
            flags=cv2.cv.CV_HAAR_SCALE_IMAGE
        )
        biggest = 0
        biggest_idx = 0
        found_face = False
        # Draw a rectngle around the faces
        for idx, (x, y, w, h) in enumerate(faces):
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            if h > biggest:
                biggest = h
                biggest_idx = idx
                found_face = True
        counter = 0
    
    # Display the resulting frame
    cv2.imshow('Video', frame)
    counter += 1

    k = cv2.waitKey(1) & 0xFF
    if k == ord('q'):
        break
    elif k == ord('s') and found_face:
        x = faces[biggest_idx][0]
        y = faces[biggest_idx][1]
        w = faces[biggest_idx][2]
        h = faces[biggest_idx][3]
        crop_img = frame[y:y+h, x:x+w]
        gray_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
        resized_img = cv2.resize(gray_img, Resized_image_size)
        print resized_img
        X = (resized_img/255.0).astype(np.float32).reshape(-1, 1, 96, 96)
        #print X
        y = (net.predict(X)[0])*48+48
        print y
        plt.imshow(resized_img, cmap=plt.cm.gray)
        plt.scatter(y[0::2], y[1::2], marker='x', s=10)
        plt.show()

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
