import cv2
import os
import time
import imutils
import numpy as np
from math import sqrt

# import the necessary packages
def order_points(pts):
	# initialzie a list of coordinates that will be ordered
	# such that the first entry in the list is the top-left,
	# the second entry is the top-right, the third is the
	# bottom-right, and the fourth is the bottom-left
	rect = np.zeros((4, 2), dtype = "float32")
	# the top-left point will have the smallest sum, whereas
	# the bottom-right point will have the largest sum
	s = pts.sum(axis = 1)
	rect[0] = pts[np.argmin(s)]
	rect[2] = pts[np.argmax(s)]
	# now, compute the difference between the points, the
	# top-right point will have the smallest difference,
	# whereas the bottom-left will have the largest difference
	diff = np.diff(pts, axis = 1)
	rect[1] = pts[np.argmin(diff)]
	rect[3] = pts[np.argmax(diff)]
	# return the ordered coordinates
	return rect


def four_point_transform(image, pts):
	# obtain a consistent order of the points and unpack them
	# individually
	rect = order_points(pts)
	(tl, tr, br, bl) = rect
	# compute the width of the new image, which will be the
	# maximum distance between bottom-right and bottom-left
	# x-coordiates or the top-right and top-left x-coordinates
	widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
	widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
	maxWidth = max(int(widthA), int(widthB))
	# compute the height of the new image, which will be the
	# maximum distance between the top-right and bottom-right
	# y-coordinates or the top-left and bottom-left y-coordinates
	heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
	heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
	maxHeight = max(int(heightA), int(heightB))
	# now that we have the dimensions of the new image, construct
	# the set of destination points to obtain a "birds eye view",
	# (i.e. top-down view) of the image, again specifying points
	# in the top-left, top-right, bottom-right, and bottom-left
	# order
	dst = np.array([
		[0, 0],
		[maxWidth - 1, 0],
		[maxWidth - 1, maxHeight - 1],
		[0, maxHeight - 1]], dtype = "float32")
	# compute the perspective transform matrix and then apply it
	M = cv2.getPerspectiveTransform(rect, dst)
	warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
	# return the warped image
	return warped


def transform(img_original):
	try:
		img = img_original.copy()
		img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		img = cv2.GaussianBlur(img, (5,5), 1)

		edged = cv2.Canny(img, 100, 255)

		items = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
		#cnts = items[0] if len(items) == 2 else items[1] # Fix opencv deprecated bug again

		img_r = cv2.drawContours(img_original.copy(), cnts, -1, (255,0,255), 3)
		cv2.imshow('sff', edged)
		cv2.waitKey(0)

		cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:5]
		for c in cnts:
			peri = cv2.arcLength(c, True)
			approx = cv2.approxPolyDP(c, 0.02 * peri, True)
			if len(approx) == 4:
				screenCnt = approx
				break


		warped = four_point_transform(img_original, screenCnt.reshape(4, 2))
		return True, warped
	except Exception as err:
		return False, img_original

def transform_better(orig):
	try:
		img = orig.copy()
		img_r = orig.copy()
		img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		img = cv2.bilateralFilter(img, 13, 75, 75)

		a = int(img.shape[0]*0.2)
		b = int(img.shape[0]*0.8)
		c = int(img.shape[1]*0.2)
		d = int(img.shape[1]*0.8)
		center_img = img[a:b,c:d]
		pre_threshold = np.average(center_img)

		# Super mega way to find threshold:
		img1 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 151, 4)
		_, img2 = cv2.threshold(img, pre_threshold*0.6, 255, cv2.THRESH_BINARY)
		img = np.array((img1/2+img2/2), dtype='uint8')
		_, img = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY)

		cnts, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
		cnt = sorted(cnts, key = cv2.contourArea, reverse = True)[:1]

		#Aprox counturs
		ap = 0.0
		while len(cnt[0])>4:
			ap += 0.001
			for i, c in enumerate(cnt):
				peri = cv2.arcLength(c, True)
				approx = cv2.approxPolyDP(c, ap * peri, True) #0.005
				cnt[i]=approx

		p = cnt[0]
		diag1 = sqrt((p[0][0][0]-p[2][0][0])**2+(p[0][0][1]-p[2][0][1])**2)
		diag2 = sqrt((p[1][0][0]-p[3][0][0])**2+(p[1][0][1]-p[3][0][1])**2)

		persp_error = max(diag1,diag2)/min(diag1,diag2)
		
		if persp_error>1.5:
			return False, orig
		else:
			rect = np.array(cnt[0])
			warped = four_point_transform(orig, rect.reshape(4, 2))
			return True, warped
	except:
		return False, orig

def to_A4(img):
	h, w, d = img.shape
	if h>w:
		w2 = (w+h)/2
		h2 = w2*sqrt(2)
		img = cv2.resize(img,(int(w2),int(h2)))
	else:
		h2 = (w+h)/2
		w2 = h2*sqrt(2)
		img = cv2.resize(img,(int(w2),int(h2)))
	return img


def cahnge_contrast_and_brightness(img, cont=0, beta=0):
	try:
		cont = int(cont)/100
		beta = int(beta)
		alpha = 1.0 + cont # Simple contrast control
		new_image = cv2.convertScaleAbs(img.copy(), alpha=alpha, beta=beta)
	except:
		new_image = img.copy()
	return new_image

def gamma_correction(img, gamma):
	try:
		lookUpTable = np.empty((1,256), np.uint8)
		for i in range(256):
			lookUpTable[0,i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)
		res = cv2.LUT(img.copy(), lookUpTable)
		return res
	except:
		return img

def find_four_max_counturs(cnts):
	# Find 4 conturs with max len
	cnt = cnts[0]
	dists=[]
	for i, c in enumerate(cnt):
		x1, y1 = c[0]
		if i+1<len(cnt):
			x2, y2 = cnt[i+1][0]
		else:
			x2, y2 = cnt[0][0]
		dists.append(sqrt((x2-x1)**2+(y2-y1)**2))

	dists.sort(reverse=True)
	dist_min = min(dists[:4])-0.01

	points=[]
	for i, c in enumerate(cnt):
		x1, y1 = c[0]
		if i+1<len(cnt):
			x2, y2 = cnt[i+1][0]
		else:
			x2, y2 = cnt[0][0]
		if sqrt((x2-x1)**2+(y2-y1)**2)>dist_min:
			k = [[x1,y1],[x2,y2]]
			points.append(k)
	return points