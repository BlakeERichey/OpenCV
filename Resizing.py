# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 21:52:04 2020

@author: Blake
"""

'''
    Open img. Draw rectangle around coke can. Find red values.
    omit not red values from original image.
    
    erode to remove noise
    dilate to recover unintentionally eroded data
    Median Blur to remove noise from res img
'''
import cv2
import numpy as np

e1 = cv2.getTickCount()
#LOAD IMAGE
img = cv2.imread('image716.jpg', cv2.IMREAD_COLOR)
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
#mask = cv2.erode(mask, kernal, iterations=1)
#mask = cv2.dilate(mask,kernal,iterations = 1)
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernal, iterations=2)
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernal, iterations=2)
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernal, iterations=3)
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernal, iterations=3)


#REMOVE NON RED PIXELS FROM ORIGINAL IMAGE
res = cv2.bitwise_and(img, img, mask=mask)

#SMOOTH RESULTS
res = cv2.medianBlur(res, 7)

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