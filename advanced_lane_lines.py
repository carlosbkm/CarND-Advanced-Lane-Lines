import os
import numpy as np
import cv2
import glob
import pickle
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from binthreshold import Binthreshold

OUTPUT_IMAGES_FOLDER = 'output_images/'

def obtain_object_image_points(images_directory):

    """
    Given a set of chessboard images contained in a directory, it returns the corners
    :param images_directory:
    :return ret, corners:
    """
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((6 * 9, 3), np.float32)
    objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d points in real world space
    imgpoints = []  # 2d points in image plane.

    # Make a list of calibration images
    images = glob.glob(images_directory + '*.jpg')

    ret, corners = bool, object
    # Step through the list and search for chessboard corners
    for idx, fname in enumerate(images):
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)

        # If found, add object points, image points
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)

            # Draw and display the corners
            cv2.drawChessboardCorners(img, (9, 6), corners, ret)
            output_corners_directory = 'calibration_results/output_corners'
            if not os.path.exists(output_corners_directory):
                os.makedirs(output_corners_directory)
            write_name = output_corners_directory + '/corners_found'+str(idx)+'.jpg'
            cv2.imwrite(write_name, img)
            #cv2.imshow('img', img)
            #cv2.waitKey(500)

    #cv2.destroyAllWindows()
    return objpoints, imgpoints


def undistort_image(img, mtx, dist):
    """
    Takes a test image and applies calibration correction. It saves the output into output_images folder.
    :param img:
    :param mtx:
    :param dist:
    :return dst:
    """
    cv2.imwrite(OUTPUT_IMAGES_FOLDER + 'chessboard_original.jpg', img)
    dst = cv2.undistort(img, mtx, dist, None, mtx)
    cv2.imwrite(OUTPUT_IMAGES_FOLDER + 'chessboard_undistorted.jpg', dst)
    return dst


def calibrate_camera(images_directory):
    """
    Calibrates the camera using the images in a folder.
    It saves the calibration result into calibration_results/wide_dist_pickle.p

    :param images_directory:
    :return mtx, dist:
    """
    img_size = (1280, 720)
    objpoints, imgpoints = obtain_object_image_points(images_directory)
    # Do camera calibration given object points and image points
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)

    # Save the camera calibration result for later use (we won't worry about rvecs / tvecs)
    dist_pickle = {}
    dist_pickle["mtx"] = mtx
    dist_pickle["dist"] = dist
    pickle.dump(dist_pickle, open("calibration_results/wide_dist_pickle.p", "wb"))

    return mtx, dist

def create_threshold_binary (img) :

    return

def apply_perspective_transform () :
    return

def detect_lane_pixels () :
    return

def fit_lane_boundary () :
    return

def determine_lane_curvature () :
    return

def unwarp_image () :
    return

def output_lane_display ():
    return

# -------------- Start Lane Lines pipeline here -----------------------------------------------------------------------

# Camera calibration
mtx, dist = calibrate_camera('camera_cal/')
dist = undistort_image(cv2.imread('camera_cal/calibration1.jpg'), mtx, dist)

# Binary threshold image
test_threshold_img = mpimg.imread('test_images/test5.jpg')
binary_img = Binthreshold.get_combined_threshold(test_threshold_img, 3, OUTPUT_IMAGES_FOLDER)

print("End pipeline")