# -*- coding: utf-8 -*-
"""
Created on Sun Nov  4 19:16:19 2018

@author: Aafreen Dabhoiwala
"""

# importing required libraries
# importing opencv library
import cv2
# importing keras library
from keras.models import model_from_json
import numpy as np

#load created json file for model creation from the disk
json_file = open('model.json', 'r')
#reading json file
loaded_model_json = json_file.read()
json_file.close()
#keras library provides ability to load created json model from disk as a json format
model = model_from_json(loaded_model_json)
#load created weights into new model
model.load_weights("model.h5")
print("Loaded model from disk")
# compiling the model. we will use adam optimizer and categorical_crossentropy as loss,
# because we have more than 2 categories in our target variable.
model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    
#start video capturing
cap = cv2.VideoCapture(0)
# Check if the webcam is opened correctly
if not cap.isOpened():
    raise IOError("Cannot open webcam")
    
# when true
while True:
# capture frame by frame. It has two outputs: img- will get the next frame from the camera.
# ret -  is the return value from the camera frame. it will be either true or false.
    ret, img = cap.read()
    
#defining coordinates for the rectangle.      
    x, y, w, h = 0, 0, 300, 300 
    
    # converting image to gray scale. if this is not done. Opencv will do it under the hood.
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # gaussian blur to remove noise from the image used for image smoothing
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    # changing the threshold value for further image analyis. It has two output: one retval and 
    # another threshold images
    ret, thresh = cv2.threshold(blur, 175, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    newImage = thresh[y:y + h, x:x + w] #saving threshold image into new image
    #resizing the image into 28*28. using opencv to resize image without distortion. 
    newImage = cv2.resize(newImage, (28, 28))
    # converting images to numpy array of images
    newImage = np.array(newImage)
    newImage = newImage.astype('float32')
    # normalizing the image 
    newImage /= 255
    # reshaping numpy array of images. numpy.reshape just change the shape attribute without changing the data (images) at all.
    newImage = newImage.reshape(28, 28, 1)
    # inserting a new axis at the 0th location to expand the size of numpy array of images of 4 from (28,28,1) to (1,28,28,1)
    newImage = np.expand_dims(newImage, axis=0)
    ans = ''
    
    # predicting the model    
    ans = model.predict(newImage).argmax()       
       
    #cv2.rectangle(img, pt1(top left coordinate), pt2(bottom right coordinate), color[, thickness[, lineType[, shift]]]) 
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), 2)
    
    #putText(img, text, org(it is a point representing the bottom left corner text string in the image), fontFace, fontScale, Scalar color, int thickness)
    cv2.putText(img, "CNN : " + str(ans), (10, 320),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # showing the image frame from the video
    cv2.imshow("Frame", img)
    
    # pressing the esc key (27) to freeze the video 
    c = cv2.waitKey(1)
    if c == 27:
        break
    
#destroting all the windows        
    
cv2.destroyAllWindows()
cap.release()