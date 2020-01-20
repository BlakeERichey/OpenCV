'''
  Get histograms of images
'''

import cv2
import matplotlib.pyplot as plt
import numpy as np

e1 = cv2.getTickCount()
#LOAD IMAGE
img = cv2.imread('nacho.jpg', cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, None, fx=.5, fy=.5, interpolation=cv2.INTER_AREA)
# color = ('b','g','r')
# for i,col in enumerate(color):
#     histr = cv2.calcHist([img],[i],None,[256],[0,256])
#     plt.plot(histr,color = col)
#     plt.xlim([0,256])

#calcluate histogram
# hist = cv2.calcHist([img], [0], None, [256], [0,256])
# plt.plot(hist, color='b')
# plt.xlim([0,256])

#calculate equilized histogram
# res = cv2.equalizeHist(img)
# plt.plot(hist, color='r')
# plt.xlim([0,256])

#Tile based equilization
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
res = clahe.apply(img)
hist = cv2.calcHist([res], [0], None, [256], [0,256])
plt.plot(hist, color='b')
plt.xlim([0,256])

e2 = cv2.getTickCount()
time = (e2 - e1)/ cv2.getTickFrequency()
print("Time taken:", time)

cv2.imshow("Image", img)
cv2.imshow("res", res)
plt.show()
cv2.waitKey(0)
cv2.destroyAllWindows()
