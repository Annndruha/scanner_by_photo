import cv2
import os
import time
import imutils
import numpy as np
import math
from random import randint
from scanner_func import *


orig = cv2.imread('docs/1.jpg')
img_gray = cv2.cvtColor(orig.copy(), cv2.COLOR_BGR2GRAY)
h_y = img_gray.shape[0]
w_x = img_gray.shape[1]
#cv2.imshow('img_gray', imutils.resize(img_gray, height = 600))
#===================================================================================================================================

img = cv2.GaussianBlur(img_gray.copy(), (3, 3), 0)
img = cv2.Canny(img, 0,255)
img = cv2.GaussianBlur(img, (5, 5), 0)
img = cv2.Canny(img, 0,255)
img = cv2.GaussianBlur(img, (9, 9), 0)
_, img = cv2.threshold(img, 10, 255, cv2.THRESH_BINARY)
#cv2.imshow('Canny', imutils.resize(img, height = 600))
canny = img

#===================================================================================================================================

cnts, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cnt = sorted(cnts, key = cv2.contourArea, reverse = True)[:1]
img_r = cv2.drawContours(orig.copy(), cnt,-1,(255,0,0), 3)
#cv2.imshow('Countur', imutils.resize(img_r, height = 600))

blank_img = np.zeros((h_y,w_x),dtype='uint8')
conturs_map = cv2.drawContours(blank_img, cnt,-1,(255,0,0), 3)
#cv2.imshow('Countur_blank', imutils.resize(conturs_map, height = 600))

#===================================================================================================================================
cdst = cv2.cvtColor(conturs_map, cv2.COLOR_GRAY2BGR)
lines_normal = cv2.HoughLines(conturs_map, 2, np.pi/180, 50, None, 0, 0)
if len(lines_normal)>40: lines_normal = lines_normal[:40]
#blank_img_rgb = cv2.cvtColor(np.zeros((h_y,w_x),dtype='uint8'), cv2.COLOR_GRAY2BGR)
blank_img_rgb = orig.copy()

lines = []
for l in lines_normal:
    A = math.cos(l[0][1]) * l[0][0]
    B = math.sin(l[0][1]) * l[0][0]
    C = -A*A-B*B
    lines.append([A,B,C])

points = []
for i, l1 in enumerate(lines):
    for j, l2 in enumerate(lines):
        if i != j:
            det = (l1[0]*l2[1]-l2[0]*l1[1])
            if abs(det)>10**-8:
                x = -(l1[2]*l2[1]-l2[2]*l1[1])/det
                y = -(l1[0]*l2[2]-l2[0]*l1[2])/det
                if not (np.isinf(x) or np.isinf(y)):
                    if ((0<x<w_x) and (0<y<h_y)):
                        points.append([x,y])
                        blank_img_rgb - cv2.circle(blank_img_rgb,(int(x),int(y)),2,(0,255,0),-1)
points_av = [[],[],[],[]]
for p in points:
    if p[0]<w_x/2 and p[1]<h_y/2:
        points_av[0].append(p)
    elif p[0]<w_x/2 and p[1]>h_y/2:
        points_av[1].append(p)
    elif p[0]>w_x/2 and p[1]<h_y/2:
        points_av[2].append(p)
    elif p[0]>w_x/2 and p[1]>h_y/2:
        points_av[3].append(p)

corners = [[],[],[],[]]
for i, p_av in enumerate(points_av):
    corners[i] = np.array(points_av[i]).T
    corners[i] = (int(np.average(corners[i][0])),int(np.average(corners[i][1])))
    blank_img_rgb - cv2.circle(blank_img_rgb,corners[i],5,(255,127,255),-1)

#for line in lines:
#    rho = line[0][0]
#    theta = line[0][1]

#    a = math.cos(theta)
#    b = math.sin(theta)
#    x0 = a * rho
#    y0 = b * rho

#    LINE_LEN = 3000
#    pt1 = (int(x0 + LINE_LEN*(-b)), int(y0 + LINE_LEN*(a)))
#    pt2 = (int(x0 - LINE_LEN*(-b)), int(y0 - LINE_LEN*(a)))
#    blank_img_rgb = cv2.line(blank_img_rgb, pt1, pt2, (0,0,255), 2, cv2.LINE_AA)
    #blank_img_rgb = cv2.line(blank_img_rgb, pt1, pt2, (0,0,255), 1, cv2.LINE_AA)
    #blank_img_rgb = cv2.line(blank_img_rgb, (0,0), (int(x0),int(y0)), (0,255,0), 1, cv2.LINE_AA)
cv2.imshow('Standard Hough Line Transform', imutils.resize(blank_img_rgb, height = 600))



#===================================================================================================================================
#img_to_find_final_conturs = cv2.cvtColor(blank_img_rgb, cv2.COLOR_BGR2GRAY)
#cnts, _ = cv2.findContours(img_to_find_final_conturs, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#cnt = sorted(cnts, key = cv2.contourArea, reverse = True)[:1]
#img_r = cv2.drawContours(orig.copy(), cnt,-1,(255,0,0), 3)
#cv2.imshow('Countur', imutils.resize(img_r, height = 600))

#===================================================================================================================================
#ap = 0.0
#while len(cnt[0])>4:
#    ap += 0.001
#    for i, c in enumerate(cnt):
#        peri = cv2.arcLength(c, True)
#        approx = cv2.approxPolyDP(c, ap * peri, True) #0.005
#        cnt[i]=approx
#img_aprox = cv2.drawContours(orig.copy(), cnt, -1, (255,0,255), 2)
#cv2.imshow('Countur_aprox', imutils.resize(img_aprox, height = 600))
#===================================================================================================================================

#p = cnt[0]
#diag1 = sqrt((p[0][0][0]-p[2][0][0])**2+(p[0][0][1]-p[2][0][1])**2)
#diag2 = sqrt((p[1][0][0]-p[3][0][0])**2+(p[1][0][1]-p[3][0][1])**2)

#persp_error = max(diag1,diag2)/min(diag1,diag2)
#print('persp_error= '+str((persp_error-1)*100)+'%')
#if persp_error>1.05:
#    img_r = orig.copy()

#points = find_four_max_counturs(cnt)
#img_r = orig.copy()
#for p in points:
#    img_r = cv2.line(img_r, (p[0][0], p[0][1]), (p[1][0], p[1][1]), (0,0,255), 4)

#img_out = four_point_transform(orig.copy(), cnt[0].reshape(4, 2))

#cv2.imshow('Orig', imutils.resize(orig.copy(), height = 600))
#cv2.imshow('Orig_bw', imutils.resize(cv2.cvtColor(orig.copy(), cv2.COLOR_BGR2GRAY), height = 600))


#cv2.imshow('Out', imutils.resize(img_out, height = 600))
c = cv2.waitKey(0)



























#===================================================================================================================================
##scale = 1
##delta = 0
##ddepth = cv2.CV_16S
##img = cv2.GaussianBlur(img_gray.copy(), (5, 5), 0)
##grad_x = cv2.Sobel(img, ddepth, 1, 0, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
##grad_y = cv2.Sobel(img, ddepth, 0, 1, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
##abs_grad_x = cv2.convertScaleAbs(grad_x)
##abs_grad_y = cv2.convertScaleAbs(grad_y)
##img = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)

##_, img = cv2.threshold(img, 20, 255, cv2.THRESH_BINARY)
##img = cv2.GaussianBlur(img, (11, 11), 0)
##_, img = cv2.threshold(img, 20, 255, cv2.THRESH_BINARY)
##cv2.imshow('Sobel', imutils.resize(img, height = 600))
##sobel = img
#===================================================================================================================================
####img = cv2.GaussianBlur(img_gray.copy(), (11, 11), 0)
####img = cv2.resize(img,(int(img.shape[1]/4),int(img.shape[0]/4)))
####img = cv2.GaussianBlur(img, (15, 15), 0)
####img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 95, 5)
#####_, img = cv2.threshold(img, 20, 255, cv2.THRESH_BINARY)
####cv2.imshow('adaptiveThreshold', imutils.resize(img, height = 600))
####adaptive = img

#===================================================================================================================================

#img = np.array((canny/3+sobel/3+adaptive/3), dtype='uint8')
#_, img = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY)
#cv2.imshow('Summ', imutils.resize(img, height = 600))

#===================================================================================================================================