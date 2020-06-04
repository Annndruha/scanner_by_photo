import os
import cv2
import numpy as np
import imutils

from scanner_func import *

print('===Image cutter by @anndruha===')
print('Select corners by LMB')
print('Save image by RMB')
print('To exit press ESC, to skip file SPACE')
print('Change brightness by whell up and down')
print('Change A4 mode by press CMB\n===')

is_A4 = True
exit = False
file_end = False
left_press = False

if not os.path.exists('docs'):
    os.makedirs('docs')

imagelist = os.listdir('docs')
for i, img_name in enumerate(imagelist):
    if exit is False:
        print(f'Open: {img_name}')
        img_orig = cv2.imread('docs/'+img_name)

        contrast = 0
        SCALE = 2
        r_max = max(img_orig.shape)
        for _ in range(10):
            if img_orig.shape[0]/SCALE>700 or img_orig.shape[1]/SCALE>1400:
                SCALE+=1
        
        points = list([[int(img_orig.shape[1]*0.1/SCALE),int(img_orig.shape[0]*0.1/SCALE)], #p1
                        [int(img_orig.shape[1]*0.1/SCALE),int(img_orig.shape[0]*0.9/SCALE)], #p2
                        [int(img_orig.shape[1]*0.9/SCALE),int(img_orig.shape[0]*0.9/SCALE)], #p3
                        [int(img_orig.shape[1]*0.9/SCALE),int(img_orig.shape[0]*0.1/SCALE)]]) #p4

        def brush(event, x, y, flags, param):
            global points, img_orig, img_name, is_A4, file_end, left_press, contrast
            index = None
            if (event == cv2.EVENT_MOUSEMOVE) and left_press:
                min_dist = max(img_orig.shape)
                for i, point in enumerate(points):
                    px, py = point
                    if sqrt((px-x)**2+(py-y)**2)<min_dist:
                        min_dist = sqrt((px-x)**2+(py-y)**2)
                        index = i
                points[index] = [x,y]

            elif event == cv2.EVENT_LBUTTONDOWN:
                left_press = True
                min_dist = max(img_orig.shape)
                for i, point in enumerate(points):
                    px, py = point
                    if sqrt((px-x)**2+(py-y)**2)<min_dist:
                        min_dist = sqrt((px-x)**2+(py-y)**2)
                        index = i
                points[index] = [x,y]

            elif event == cv2.EVENT_LBUTTONUP:
                left_press = False

            elif event == cv2.EVENT_MOUSEWHEEL:
                if flags>0:
                    if contrast < 100+10:
                        contrast +=10
                else:
                    if contrast-10>-100:
                        contrast -=10
            elif event == cv2.EVENT_MBUTTONUP:
                is_A4 = not is_A4
                if is_A4:
                    print('Mode change to A4')
                else:
                    print('Mode change to default')

            elif event == cv2.EVENT_RBUTTONUP:
                poly = np.array(points)
                warped = four_point_transform(img_orig.copy(), SCALE*poly.reshape(4, 2))
                if is_A4:
                    warped = to_A4(warped)
                if not os.path.exists('docs_out'):
                    os.makedirs('docs_out')
                warped = cahnge_contrast_and_brightness(warped, contrast)
                cv2.imwrite(f'docs_out/{img_name}',warped)
                print(f'Cutted image save as docs_out/{img_name}')
                file_end = True

        def draw_poly(image, points):
            poly = np.array(points)
            cv2.polylines(image, [poly], True, (0,255,0), thickness=1)
            return image

        cv2.namedWindow(f'Cut image {img_name}')
        cv2.setMouseCallback(f'Cut image {img_name}',brush)
        while (file_end is not True) and (exit is False):
            image = imutils.resize(img_orig.copy(), height = int(img_orig.shape[0]/SCALE))
            image = cahnge_contrast_and_brightness(image, contrast)
            image = draw_poly(image, points)
            cv2.imshow(f'Cut image {img_name}', image)
            c = cv2.waitKey(10)
            if c == 27:
                exit = True
                break
            if c == 32:
                break
        cv2.destroyAllWindows()
        file_end = False
print('Program end.')
