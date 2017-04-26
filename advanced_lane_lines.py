import os
import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt

def calibrate_camera(images_directory) :
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
    return ret, corners

def correct_distortion () :
    return

def create_threshold_binary () :
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
ret, corners = calibrate_camera('camera_cal/')
print("End pipeline")