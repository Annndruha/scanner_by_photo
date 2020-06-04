# import required libraries
import numpy as np
import cv2
import imutils
import matplotlib.pyplot as plt
from matplotlib import cm
from numpy.random import randn
import os


for dirpath, dirnames, filenames in os.walk('docs'):
    for filename in filenames:
        im = cv2.imread('docs/'+filename)
        grayImage = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

        flat = grayImage.flatten()
        grayMap_y = np.zeros((256,))
        for pix in flat: grayMap_y[pix]+=1

        grayMap_y = grayMap_y/max(grayMap_y)
        print('Image average=')
        print(np.average(grayImage))


        center = grayImage[int(im.shape[0]/3):int(im.shape[0]*2/3), int(im.shape[1]/3):int(im.shape[1]*2/3)]
        grayMap_center = np.zeros((256,))
        flat_c = center.flatten()
        for pix in flat_c: grayMap_center[pix]+=1

        grayMap_center = grayMap_center/max(grayMap_center)
        print('Center average=')
        print(np.average(center))



        fig, ax = plt.subplots(2,1)
        ax[0].set_axis_off()
        cax = ax[0].imshow(grayImage, interpolation='nearest', cmap=cm.gray)
        #fig.colorbar(cax, orientation='horizontal', ticks=[-2,-1])
        ax[1].plot(grayMap_y)
        ax[1].plot(grayMap_center)
        ax[1].set_xticks(np.arange(0, 255, 32))

        plt.savefig('gray_plots/'+filename+'.png', dpi=240, bbox_inches='tight')
#plt.show()