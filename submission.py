#!/usr/bin/env python
# coding: utf-8

# In[33]:


import os
import cv2
import numpy as np 
from keras.models import model_from_json
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.initializers import glorot_uniform
#Reading the model from JSON file
with open('model.json', 'r') as json_file:
    json_savedModel= json_file.read()#load the model architecture 
model_j = tf.keras.models.model_from_json(json_savedModel)
model_j.load_weights('model.h5')
characters='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabdefghnqrt'
def predict(image):
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)


    #binary
    ret,thresh = cv2.threshold(gray,100,255,cv2.THRESH_BINARY_INV)

    #dilation
    kernel = np.ones((3,3), np.uint8)
    kernel1 = np.ones((3,3), np.uint8)
    img_dilation=cv2.dilate(thresh,kernel,iterations=1)
    plt.imshow(img_dilation)
    ctrs, hier = cv2.findContours(img_dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if(len(ctrs)>10):
        img_erosion=cv2.erode(thresh,kernel1,iterations=1)
        #plt.imshow(img_erosion)
        img_dilation=cv2.dilate(img_erosion,kernel,iterations=1)
        ctrs, hier = cv2.findContours(img_dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    m = list()
   #sort contours
    sorted_ctrs = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0])
    pchl = list()
    dp = image.copy()
    for i, ctr in enumerate(sorted_ctrs):
        # Get bounding box
        x, y, w, h = cv2.boundingRect(ctr)
        cv2.rectangle(dp,(x-2,y-2),( x + w + 2, y + h + 2 ),(200,200,0),3)
    
    plt.imshow(dp)
    for i, ctr in enumerate(sorted_ctrs):
        # Get bounding box
        x, y, w, h = cv2.boundingRect(ctr)
        # Getting ROI
        roi = image[max(y-2,0):y+h+2, max(x-2,0):x+w+2]
        roi = cv2.resize(roi, dsize=(28,28), interpolation=cv2.INTER_CUBIC)
        roi = cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY)
        
        roi = np.array(roi)
        t = np.copy(roi)
        t = t / 255.0
        t = 1-t
        t=t.reshape(-1,28,28,1)
        m.append(roi)
        pred = model_j.predict_classes(t)
        if(pred[0]==24):
            pred[0]=0
        if(pred[0]==29):
            pred[0]=7
        if(pred[0]==37):
            pred[0]=6
        if(pred[0]==46):
            pred[0]=7
        if(pred[0]==35):
            pred[0]=2
        if(pred[0]==41):
            pred[0]=9
        pchl.append(pred)
    pcw = list()
    interp = 'bilinear'
    fig, axs = plt.subplots(nrows=len(sorted_ctrs), sharex=True, figsize=(1,len(sorted_ctrs)))
    for i in range(len(pchl)):
        print (pchl[i][0])
        pcw.append(characters[pchl[i][0]])
        axs[i].set_title('-------> predicted letter: '+characters[pchl[i][0]], x=2.5,y=0.24)
        axs[i].imshow(m[i], interpolation=interp)
     
    plt.show()
    
    
    predstring = ''.join(pcw)
    print('Predicted String: '+predstring)  
    answer = predstring # sample needs to be modified
    return answer

def test():
   
    image_paths = ['2BTE.png','2BTX.png','2BVF.png']
    correct_answers = ['2B7E','2B7X','2BVF']
    score = 0

    for i,image_path in enumerate(image_paths):
        image = cv2.imread(image_path) # This input format wont change
        answer = predict(image) # a string is expected

        if correct_answers[i] == answer:
            score += 10
    
    print('The final score of the participant is',score)


if __name__ == "__main__":
    test()

