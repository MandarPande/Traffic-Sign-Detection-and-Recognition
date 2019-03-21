import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('practice.png')
out1 = img
'''plt.hist(img.ravel(), 256, [0,256])
plt.show()'''
#--------------PREPROCESSING------------------
#Salt and Pepper Noise Removal
median = cv2.medianBlur(img, 5)
out1 = np.hstack((out1, median))
#Histogram Equalization
median_hsv = cv2.cvtColor(median, cv2.COLOR_BGR2HSV)
equ = cv2.equalizeHist(median_hsv[:, :, 2])
median_contrast_hsv = median_hsv
median_contrast_hsv[:, :, 2] = equ
median_contrast = cv2.cvtColor(median_contrast_hsv, cv2.COLOR_HSV2BGR)
out2 = median_contrast
plt.hist(median_contrast.ravel(), 256, [0,256])
plt.show()

new_img = median_contrast
new_img_hsv = cv2.cvtColor(new_img, cv2.COLOR_BGR2HSV)
'''new_img_hsv_val = new_img_hsv
new_img_hsv_val[:, :, 2]''' 
out2 = np.hstack((out2, new_img_hsv))

#plot H, S, V and histograms for each for all image
cv2.imshow('hue', new_img_hsv[:,:,0])
cv2.imshow('saturaion', new_img_hsv[:,:,1])
cv2.imshow('value', new_img_hsv[:,:,2])
histr_hue = cv2.calcHist([new_img_hsv],[0],None,[256],[0,256])
plt.plot(histr_hue, color = 'r')
plt.xlim([0,256])
histr_sat = cv2.calcHist([new_img_hsv],[1],None,[256],[0,256])
plt.plot(histr_sat, color = 'g')
plt.xlim([0,256])
histr_val = cv2.calcHist([new_img_hsv],[2],None,[256],[0,256])
plt.plot(histr_val, color = 'b')
plt.xlim([0,256])
plt.legend(('hue', 'saturation', 'value'))
plt.show()


#Determine ROI
dimensions = new_img_hsv.shape
height = img.shape[0]
width = img.shape[1]
new_edited_img = np.zeros((height, width, 3), np.uint8)
new_edited_img_hsv = cv2.cvtColor(new_edited_img, cv2.COLOR_BGR2HSV)
for i in range(height):
    for j in range(width):
        hue = new_img_hsv[i, j, 0]
        sat = new_img_hsv[i, j, 1]
        val = new_img_hsv[i, j, 2]
        if (hue/179 < 0.05 or hue/179 > 0.95) and (sat/255 > 0.5) and (val/255 > 0.01):
            new_edited_img_hsv[i, j, 0] = hue
            new_edited_img_hsv[i, j, 1] = sat
            new_edited_img_hsv[i, j, 2] = val


#plot H, S, V and histograms for each for ROI editted
cv2.imshow('hue_edited', new_edited_img_hsv[:,:,0])
cv2.imshow('saturaion_edited', new_edited_img_hsv[:,:,1])
cv2.imshow('value_edited', new_edited_img_hsv[:,:,2])
histr_hue = cv2.calcHist([new_edited_img_hsv],[0],None,[256],[0,256])
plt.plot(histr_hue, color = 'r')
plt.xlim([0,256])
histr_sat = cv2.calcHist([new_edited_img_hsv],[1],None,[256],[0,256])
plt.plot(histr_sat, color = 'g')
plt.xlim([0,256])
histr_val = cv2.calcHist([new_edited_img_hsv],[2],None,[256],[0,256])
plt.plot(histr_val, color = 'b')
plt.xlim([0,256])
plt.legend(('hue', 'saturation', 'value'))
plt.show()

cv2.imshow('out1', out1)
cv2.imshow('out2', out2)
cv2.waitKey(0)
