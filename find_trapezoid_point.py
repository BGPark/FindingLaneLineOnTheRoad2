from cal_camera import get_camera_cal, undistort
from utils import *
import glob
import cv2
import matplotlib.pyplot as plt
from scipy import misc
import numpy as np


TEST_IMAGE_DIR = './test_images'


def find_trapezoid_points(undist_image, filename):
    # To find appropriate position. I picked i=122 according to saved image.
    for i in range(70, 90, 1):
        src = np.float32([[640 - i, 470], [640 + i, 470],
                          [1080, 720], [200, 720]])

        offset_bird = 320
        dst = np.float32([[640 - offset_bird, 0], [640 + offset_bird, 0],
                          [640 + offset_bird, 720], [640 - offset_bird, 720]])

        image_size = (undist_image.shape[1], undist_image.shape[0])
        M = cv2.getPerspectiveTransform(src, dst)

        warped = cv2.warpPerspective(undist_image, M, image_size, flags=cv2.INTER_LINEAR)

        show_image(warped, save=True, save_file_name=(filename.split('/')[-1] + '_' + str(i)+'.png'))



if __name__ == '__main__':
    images = glob.glob(TEST_IMAGE_DIR + '/straight*.jpg')
    mtx, dist = get_camera_cal()

    for filename in images:
        print(filename.split('/')[-1])
        image = misc.imread(filename)
        undist = undistort(image, mtx, dist)
        # show_pair_image(image, 'Original Image', undist, 'Undistorted Image')
        find_trapezoid_points(undist, filename)



