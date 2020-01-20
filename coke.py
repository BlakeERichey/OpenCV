# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 21:52:04 2020

@author: Blake
"""

'''
    Open img. Draw rectangle around coke can. Find red values.
    omit not red values from original image.
    
    Contours - Find a coke can
'''
import cv2
import numpy as np

e1 = cv2.getTickCount()
#LOAD IMAGE
img = cv2.imread('coke.jpg', cv2.IMREAD_COLOR)
img = cv2.resize(img, None, fx=.5, fy=.5, interpolation=cv2.INTER_AREA)
edges = cv2.Canny(img, 100, 250)

kernal = np.ones((5,5), np.uint8)

#FIND RED PIXELS
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
lower_red = np.array([0, 175, 100])
upper_red = np.array([10, 255, 255])
mask = cv2.inRange(hsv, lower_red, upper_red)
lower_red = np.array([160, 175, 100])
upper_red = np.array([179, 255, 255])
mask2 = cv2.inRange(hsv, lower_red, upper_red)
mask = cv2.addWeighted(mask, 1, mask2, 1, 0)
mask = cv2.erode(mask, kernal, iterations=1)
mask = cv2.dilate(mask,kernal,iterations = 1)


#REMOVE NON RED PIXELS FROM ORIGINAL IMAGE
res = cv2.bitwise_and(img, img, mask=mask)

#SMOOTH RESULTS
res = cv2.medianBlur(res, 7)

#FIND CONTOURS
res = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
ret,res = cv2.threshold(res,0,255,cv2.THRESH_BINARY) #to find contoours, need white images
edges = cv2.dilate(edges,kernal,iterations = 1)
for i in range(20):
    edges = cv2.medianBlur(edges, 9)
image, contours, hierarchy = cv2.findContours(res.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

for c in contours:
    x,y,w,h = cv2.boundingRect(c) #get bounding box
    if w*h >= 10000: #if bounding box big enough
        print("wh", w*h)
        img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,255),2)


e2 = cv2.getTickCount()
time = (e2 - e1)/ cv2.getTickFrequency()
print("Time taken:", time)

cv2.imshow("Image", img)
cv2.imshow('Mask',mask)
cv2.imshow("Res", res)
cv2.imshow("Edges", edges)
e2 = cv2.getTickCount()
time = (e2 - e1)/ cv2.getTickFrequency()
print("Time taken:", time)

cv2.imshow("Res", res)
cv2.imshow("Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()