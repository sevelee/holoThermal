#-*- coding: utf-8 -*-
# 根据opencv教程写的代码
# http://docs.opencv.org/3.2.0/dc/dbb/tutorial_py_calibration.html
# start at 2017/08/29

import numpy as np
import cv2
import glob

# termination criteria
# 终止的标准
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*7,3), np.float32)
objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
# 3D points are called object points and 2D image points are called image points.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

images = glob.glob('*.jpg')
for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (7,6), None)

    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)

        # increase the accuracy of conrners
        corners2=cv2.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
        
        imgpoints.append(corners)

        # Draw and display the corners
        cv2.drawChessboardCorners(img, (7,6), corners2, ret)
        
        cv2.imshow('img', img)
        cv2.waitKey(500)

cv2.destroyAllWindows()