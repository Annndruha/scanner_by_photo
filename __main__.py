
import os
import time

import cv2
import numpy as np
import imutils
from fpdf import FPDF

from scanner_func import *


#pdf = FPDF()
imagelist = os.listdir('docs')

for img_name in imagelist:
    img = cv2.imread('docs/'+img_name)
    sucsess, img = transform_better(img)
    if sucsess: img = to_A4(img)

    print(img_name +' '+str(sucsess))
    cv2.imwrite('docs_res/'+img_name, img)


    #pdf.add_page()
    #pdf.image('docs_res/'+img_name, 0,0, 50,50) #,x,y,w,h)

#pdf.output("yourfile.pdf", "F")

#img_original = cv2.imread('docs/4.jpg')
#sucsess, img = transform(img_original)
#if sucsess:
#    img = to_A4(img)
#print(sucsess)

#img = cahnge_contrast_and_brightness(img, 10, 0)
#img = gamma_correction(img, 0.7)



#print(img.shape)
#cv2.imshow("Original", imutils.resize(img_original, height = 650))
#cv2.imshow("Scanned", imutils.resize(img, height = 650))
#cv2.waitKey(0)