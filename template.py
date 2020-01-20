'''
  Attemping to find coke cans by matching to a template
'''

import cv2
import numpy as np
import matplotlib.pyplot as plt

e1 = cv2.getTickCount()
#LOAD IMAGE
img = cv2.imread('coke.jpg', cv2.IMREAD_GRAYSCALE)
# img = cv2.resize(img, None, fx=.5, fy=.5, interpolation=cv2.INTER_AREA)
img2 = img.copy()
template = cv2.imread('template.jpg', cv2.IMREAD_GRAYSCALE)
# template = cv2.resize(img, None, fx=.5, fy=.5, interpolation=cv2.INTER_AREA)
h, w = template.shape

res = cv2.matchTemplate(img,template,cv2.TM_CCORR)
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

#for sqdiff, use min_val and min_loc
top_left = min_loc
top_left = max_loc
bottom_right = (top_left[0]+w, top_left[1]+h)
cv2.rectangle(img, top_left, bottom_right, (255,255,255), 2)

e2 = cv2.getTickCount()
time = (e2 - e1)/ cv2.getTickFrequency()
print("Time taken:", time)

cv2.imshow("Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
