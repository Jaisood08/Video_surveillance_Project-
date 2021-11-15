import cv2
import numpy as np
from keras.preprocessing import image


faceCascade= cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

img = cv2.imread("Sharan.jprg")

imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

faces = faceCascade.detectMultiScale(imgGray,1.1,4)

for (x,y,w,h) in faces:
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    crop_img = imgGray[y:y+w,x:x+h]
    crop_img = cv2.resize(crop_img,(48,48))
    img_pixels = image.img_to_array(crop_img)
    img_pixels = np.expand_dims(img_pixels,axis = 0 )
    img_pixels = img_pixels/255
      
cv2.imshow("Result", img)

cap.release()