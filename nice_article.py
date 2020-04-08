import cv2
import os
import time
import imutils
import numpy as np
from random import randint
from scanner_func import *


orig = cv2.imread('docs/0.jpg')
img = orig.copy()
img_r = orig.copy()
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = cv2.bilateralFilter(img, 13, 75, 75)

center_img = img[int(img.shape[0]*0.2):int(img.shape[0]*0.8),int(img.shape[1]*0.2):int(img.shape[1]*0.8)]
pre_threshold = np.average(center_img)
img_aver = np.average(img)
thresh = img_aver*0.5+pre_threshold*0.5
print('pre_threshold= '+str(pre_threshold))
print('img_aver= '+str(img_aver))

# Super mega way to find threshold:
img1 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 95, 4)
img = img1
_, img2 = cv2.threshold(img, thresh, 255, cv2.THRESH_BINARY)
img = np.array((img1/2+img2/2), dtype='uint8')

img = cv2.medianBlur(img, 11)
_, img = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY)


cnts, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cnt = sorted(cnts, key = cv2.contourArea, reverse = True)[:1]

##Aprox counturs
ap = 0.0
while len(cnt[0])>20:
    ap += 0.001
    for i, c in enumerate(cnt):
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, ap * peri, True) #0.005
        cnt[i]=approx
img_r = cv2.drawContours(orig.copy(), cnt, -1, (255,0,255), 2)

p = cnt[0]
diag1 = sqrt((p[0][0][0]-p[2][0][0])**2+(p[0][0][1]-p[2][0][1])**2)
diag2 = sqrt((p[1][0][0]-p[3][0][0])**2+(p[1][0][1]-p[3][0][1])**2)

persp_error = max(diag1,diag2)/min(diag1,diag2)
print('persp_error= '+str((persp_error-1)*100)+'%')
#if persp_error>1.05:
#    img_r = orig.copy()

points = find_four_max_counturs(cnt)
img_r = orig.copy()
for p in points:
    img_r = cv2.line(img_r, (p[0][0], p[0][1]), (p[1][0], p[1][1]), (0,0,255), 4)


cv2.imshow('orig', imutils.resize(cv2.cvtColor(orig.copy(), cv2.COLOR_BGR2GRAY), height = 600))
cv2.imshow('trhe', imutils.resize(img, height = 600))
cv2.imshow('Window', imutils.resize(img_r, height = 600))
c = cv2.waitKey(0)
