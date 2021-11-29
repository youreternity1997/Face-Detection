# -*- coding: utf-8 -*-
"""
Created on Mon Nov 29 14:34:10 2021

@author: User
"""
import cv2, os
import glob

faceCascade= cv2.CascadeClassifier("Resources\haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier('Resources\haarcascade_eye.xml')


def detect(images_path, save_dir):
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    
    images = glob.glob(images_path + "\*.jpg")
    for image in images:
        image_name = image.split('\\')[-1]
        
        img = cv2.imread(image)
        imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(imgGray, 
                                             scaleFactor=1.1,
                                             minNeighbors=4,)
         
        for (x,y,w,h) in faces:
            cv2.rectangle(img,(x,y),(x+w,y+h), (255,0,0), 2)
            roi_imgGray = imgGray[y:y+h, x:x+w]
            
            eyes = eye_cascade.detectMultiScale(roi_imgGray,
                                                scaleFactor=1.02,
                                                minNeighbors=3,
                                                minSize=(40,40),)
            for (ex,ey,ew,eh) in eyes:
                img = cv2.rectangle(img,(x+ex,y+ey),(x+ex+ew,y+ey+eh),(0,255,0),2)
            
        #cv2.imshow("Result", img)
        #cv2.waitKey(0)
        save_path = save_dir + '\\' + image_name
        cv2.imwrite(save_path, img)
        print(save_path)
        
if __name__ == '__main__':
    images_path = ".\Resources"
    save_dir = '.\Predicted face'

    detect(images_path, save_dir)