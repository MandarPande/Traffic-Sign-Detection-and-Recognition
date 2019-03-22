import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('stop_sign_350x350.jpg')
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
#------------IMAGE PROCESSING-------------------
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
new_edited_img = cv2.cvtColor(new_edited_img_hsv, cv2.COLOR_HSV2BGR)
cv2.imshow('new_edited_img', new_edited_img)

#thresholding
new_edited_img_gray = cv2.cvtColor(new_edited_img, cv2.COLOR_BGR2GRAY)
ret,thresh_img = cv2.threshold(new_edited_img_gray,5,255,cv2.THRESH_BINARY)
cv2.imshow('threshold Image', thresh_img)


#pixels 8-connectivity
'''thresh_img_blur = cv2.medianBlur(thresh_img, 5)
cv2.imshow('thresh_blur', thresh_img_blur)
map = [[0 for i in range(width)] for j in range(height)]
index = 1
count = 0
for i in range(height):
    for j in range(width):
        if thresh_img_blur[i,j] != 0:
            if map[i-1][j-1] != 0:
                map[i][j] = map[i-1][j-1]
            elif map[i-1][j] != 0:
                map[i][j] = map[i - 1][j]
            elif map[i -1][j+1] != 0:
                map[i][j] = map[i - 1][j+1]
            elif map[i][j-1] != 0:
                map[i][j] = map[i ][j-1]
            elif map[i ][j +1] != 0:
                map[i][j] = map[i][j +1]
            elif map[i+1][j-1] != 0:
                map[i][j] = map[i+1][j-1]
            elif map[i+1][j] != 0:
                map[i][j] = map[i+1][j]
            elif map[i+1][j+1]!=0:
                map[i][j] = map[i+1][j+1]
            else:
                map[i][j] = index
                index += 1
                count = count + 1

traffic_sign_index = 0
count_dict = {}
for i in range(1, count+1):
    count_dict[i] = 0
for i in range(height):
    for j in range(width):
        if map[i][j] != 0:
            x = map[i][j]
            count_dict[x] = count_dict[x] + 1
            
for i in range(1, count+1):
    if count_dict[i] < 100 or count_dict[i] > 3000:
        count_dict[i] = -1
print(count_dict)

img_copy = img
for i in range(height):
    for j in range(width):
        if map[i][j] == 1:
            img_copy[i, j] = [255,255,255]
cv2.imshow('modified',img_copy)
''''''threshold_img = new_edited_img
for i in range(height):
    for j in range(width):
        if threshold_img[i, j, 0] '''
img_copy = img
thresh_img_blur = cv2.GaussianBlur(thresh_img,(7,7),0)
cv2.imshow('blur threshold', thresh_img_blur)         
output = cv2.connectedComponentsWithStats(thresh_img_blur,  8, cv2.CV_32S)
print(output[0])
print(output[1])
print(output[2])        
'''for i in range(height):
    for j in range(width):
        if output[1][i][j] == 1:
            img_copy[i, j] = [255,255,255]'''
cv2.rectangle(img_copy,(output[2][1][0],output[2][1][1]),(output[2][1][0]+output[2][1][2],output[2][1][1]+output[2][1][3]),(0,255,0),2)            
cv2.imshow('modified',img_copy)

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
