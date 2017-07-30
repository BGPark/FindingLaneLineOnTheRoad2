import cv2
import numpy as np
import glob
import os
import pickle
import matplotlib.pyplot as plt
from utils import show_pair_image


CAMERA_CAL_FILE_NAME = './cal_camera.p'
REF_IMAGE_DIR = './camera_cal'


def get_camera_cal():

    if os.path.isfile(CAMERA_CAL_FILE_NAME):
        # print('cal is ready to load')
        mtx = 0
        dist = 0
        cal_pickle = pickle.load(open(CAMERA_CAL_FILE_NAME, 'rb'))
        mtx = cal_pickle['mtx']
        dist = cal_pickle['dist']
    else:
        mtx, dist = generate_cal()
        cal_pickle = {}
        cal_pickle['mtx'] = mtx
        cal_pickle['dist'] = dist
        pickle.dump(cal_pickle, open(CAMERA_CAL_FILE_NAME, 'wb'))

    return mtx, dist


def generate_cal():
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((6 * 9, 3), np.float32)
    objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

    objpoints = []
    imgpoints = []

    images = glob.glob(REF_IMAGE_DIR + '/*.jpg')

    for idx, fname in enumerate(images):
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)

        # If found, add object points, image points
        if ret:
            objpoints.append(objp)
            imgpoints.append(corners)

    img_size = (img.shape[1], img.shape[0])
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)
    return mtx, dist


def undistort(image, mtx, dist):
    return cv2.undistort(image, mtx, dist, None, mtx)


if __name__ == '__main__':

    images = glob.glob(REF_IMAGE_DIR + '/*.jpg')
    img = cv2.imread(images[9])

    mtx, dist = get_camera_cal()
    undist = undistort(img, mtx, dist)

    show_pair_image(img, 'Original Image', undist, 'Undistorted', 'output_images/undist.png')



