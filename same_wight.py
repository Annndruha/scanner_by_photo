import os
import cv2
import numpy as np
import imutils

PATH_IN = 'docs_out'
PATH_OUT = 'docs_same_shape'

if not os.path.exists(PATH_IN):
    os.makedirs(PATH_IN)

if not os.path.exists(PATH_OUT):
    os.makedirs(PATH_OUT)


imagenames_list = os.listdir(PATH_IN)
images = list()
images_shapes = list()

if len(imagenames_list)<=1:
    print(f"Empty folder or less than 2 files: {PATH_IN}")
else:
    for i, img_name in enumerate(imagenames_list):
        print(f'Open: {img_name}')
        img_orig = cv2.imread(PATH_IN + '/' + img_name)
        images.append((img_name, img_orig))
        images_shapes.append(img_orig.shape[1])

    min_wight_shape = min(images_shapes)

    for img_name, img in images:
        zoom = min_wight_shape/img.shape[1]
        image = imutils.resize(img.copy(), width = int(img.shape[1]*zoom))
        cv2.imwrite(PATH_OUT+'/'+img_name, image)
        print(f'Write: {img_name}')

    print('Done.')