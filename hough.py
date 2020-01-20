'''
  Attemping to find coke cans by matching to a template
'''

import cv2
import numpy as np
import matplotlib.pyplot as plt

e1 = cv2.getTickCount()
#LOAD IMAGE
# img = cv2.imread('coke.jpg', cv2.IMREAD_GRAYSCALE)
img = cv2.imread('coke_top.jpg', cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, None, fx=.25, fy=.25, interpolation=cv2.INTER_AREA)
edges = cv2.Canny(img, 100, 200)

#LINE HOUGH
# edges = cv2.Canny(img, 50, 150)
# edges = cv2.morphologyEx(edges.copy(), cv2.MORPH_CLOSE, np.ones((5,5), np.uint8), iterations=2)
# edges = cv2.morphologyEx(edges.copy(), cv2.MORPH_OPEN, np.ones((5,5), np.uint8), iterations=2)
# edges = cv2.Canny(edges, 50, 200)
# min_line_length = 700
# max_line_gap = 3
# lines = cv2.HoughLinesP(edges, 1, np.pi/180, 3, min_line_length, max_line_gap)
# # print('lines', lines)
# for i in range(len(lines)):
#   for x1, y1, x2, y2 in lines[i]:
#     # print("Point:", (x1, y1, x2, y2))
#     cv2.line(img,(x1, y1), (x2,y2), (255,255,255), 2)

#CIRCLE HOUGH
img = cv2.medianBlur(img, 5)
cimg = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 200, param1=200, param2=100, minRadius=0, maxRadius=0)
circles = np.uint16(np.around(circles))
for i in circles[0,:]:
  # draw the outer circle
  cv2.circle(cimg,(i[0],i[1]),i[2],(0,255,0),2)
  # draw the center of the circle
  cv2.circle(cimg,(i[0],i[1]),2,(0,0,255),3)

e2 = cv2.getTickCount()
time = (e2 - e1)/ cv2.getTickFrequency()
print("Time taken:", time)

cv2.imshow("Image", img)
cv2.imshow("Edges", edges)
cv2.imshow("Cimg", cimg)
cv2.waitKey(0)
cv2.destroyAllWindows()
