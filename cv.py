'''
    Open img. Draw rectangle around coke can. Find red values.
    omit not red values from original image
'''
import cv2
import numpy as np

e1 = cv2.getTickCount()
img = cv2.imread('image716.jpg', cv2.IMREAD_COLOR)
img2 = cv2.imread('image740.jpg', cv2.IMREAD_COLOR)

font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(img,'OpenCV',(10,500), font, 4,(255,255,255),2,cv2.LINE_AA)
img = cv2.rectangle(img, (650,240), (1275,575), 10)

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
lower_red = np.array([0, 175, 100])
upper_red = np.array([10, 255, 255])

mask = cv2.inRange(hsv, lower_red, upper_red)

lower_red = np.array([160, 175, 100])
upper_red = np.array([179, 255, 255])

mask2 = cv2.inRange(hsv, lower_red, upper_red)

mask = cv2.addWeighted(mask, 1, mask2, 1, 0)

res = cv2.bitwise_and(img, img, mask=mask)

#can = img[240:575, 650:1275]

#img[600:935, 650:1275] = can

#dst = cv2.addWeighted(img2, 0.7, img, 0.3, 0)
e2 = cv2.getTickCount()
time = (e2 - e1)/ cv2.getTickFrequency()
print("Time taken:", time)

cv2.imshow("Image", img)
cv2.imshow('Mask',mask)
cv2.imshow("Res", res)
k = cv2.waitKey(0)
if k == 27:         # wait for ESC key to exit
    cv2.destroyAllWindows()
elif k == ord('s'): # wait for 's' key to save and exit
    cv2.imwrite('cokegray.png',img)
    cv2.destroyAllWindows()