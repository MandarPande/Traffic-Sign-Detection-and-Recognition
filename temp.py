import cv2
import numpy as np 

img = cv2.imread('practice.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

print(img[:,:,:])