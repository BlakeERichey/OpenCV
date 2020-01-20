# -*- coding: utf-8 -*-
"""
Created on Sun Jan 19 14:26:02 2020

@author: Blake
"""

'''
    Find clear objects
'''

import cv2
import matplotlib.pyplot as plt
import numpy as np

e1 = cv2.getTickCount()
#LOAD IMAGE
img = cv2.imread('wb.jpg', cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, None, fx=.5, fy=.5, interpolation=cv2.INTER_AREA)
img = cv2.GaussianBlur(img, (5,5), 0)

kernel = np.ones((5,5),np.uint8)
sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)
sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)
laplacian = cv2.Laplacian(img,cv2.CV_64F)

abs_sobel64f = np.absolute(sobelx)
sobel_8u = np.uint8(abs_sobel64f)

abs_sobel64f = np.absolute(laplacian)
laplacian_8u = np.uint8(abs_sobel64f)

edges = cv2.Canny(laplacian_8u, 10, 50)
edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8), iterations=10)
edges = cv2.morphologyEx(edges, cv2.MORPH_OPEN, np.ones((3,3), np.uint8), iterations=10)
edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8), iterations=10)
image, contours, hierarchy = cv2.findContours(edges.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)


for c in contours:
  rect = cv2.minAreaRect(c)
  w, h = rect[1]
  box = cv2.boxPoints(rect)
  box = np.int0(box)
  img = cv2.drawContours(img,[box],0,(255,255,255),2)
  # x,y,w,h = cv2.boundingRect(c) #get bounding box
  # if w*h >= 10000: #if bounding box big enough
  #     print("wh", w*h)
  #     img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,255),2)

e2 = cv2.getTickCount()
time = (e2 - e1)/ cv2.getTickFrequency()
print("Time taken:", time)

cv2.imshow("Image", img)
cv2.imshow("Laplacian", laplacian)
cv2.imshow("sobelx", sobelx)
cv2.imshow("sobely", sobely)
cv2.imshow("edges", edges)
cv2.waitKey(0)
cv2.destroyAllWindows()
