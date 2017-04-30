import os
import numpy as np
import cv2
import glob
import pickle
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from binthreshold import Binthreshold

OUTPUT_IMAGES_FOLDER = 'output_images/'
CALIBRATION_OUTPUT = 'calibration_results/'
CALIBRATION_OUTPUT_FILE = 'wide_dist_pickle.p'

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

            # nx = 9
            # implot = plt.imshow(gray, cmap='gray')
            # plt.scatter([corners[0][0][0]], [corners[0][0][1]])
            # plt.scatter([corners[nx - 1][0][0]], [corners[nx - 1][0][1]], c='r')
            # plt.scatter([corners[-1][0][0]], [corners[-1][0][1]], c='g')
            # plt.scatter([corners[-nx][0][0]], [corners[-nx][0][1]], c='y')
            # plt.show()

            # Draw and display the corners
            cv2.drawChessboardCorners(img, (9, 6), corners, ret)
            output_corners_directory = CALIBRATION_OUTPUT + 'output_corners'
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
    cv2.imwrite(OUTPUT_IMAGES_FOLDER + 'distortion_correction/chessboard_original.jpg', img)
    dst = cv2.undistort(img, mtx, dist, None, mtx)
    cv2.imwrite(OUTPUT_IMAGES_FOLDER + 'distortion_correction/chessboard_undistorted.jpg', dst)
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
    pickle.dump(dist_pickle, open(CALIBRATION_OUTPUT + CALIBRATION_OUTPUT_FILE, "wb"))

    return mtx, dist


def apply_perspective_transform(img, dist_filename):
    dist_pickle = pickle.load(open(dist_filename, "rb"))
    mtx = dist_pickle["mtx"]
    dist = dist_pickle["dist"]
    # Pass in your image into this function
    # Write code to do the following steps
    # 1) Undistort using mtx and dist
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    # 2) Convert to grayscale
    gray = cv2.cvtColor(undist, cv2.COLOR_BGR2GRAY)



    height = gray.shape[0]
    upper_limit = 472
    square_width = 720
    square_left_corner = 250
    square_right_corner = square_left_corner + square_width

    # implot = plt.imshow(gray, cmap='gray')
    # plt.scatter([580], [upper_limit])
    # plt.scatter([760], [upper_limit], c='r')
    # plt.scatter([1200], [height], c='y')
    # plt.scatter([180], [height], c='g')
    #
    # plt.show()

    # b) define 4 source points src = np.float32([[,],[,],[,],[,]])
    img_size = (gray.shape[1], gray.shape[0])
    src = np.float32([np.array([(563, upper_limit)], dtype='float32'), np.array([(725, upper_limit)], dtype='float32'),
                      np.array([(1113, height)], dtype='float32'), np.array([(171, height)], dtype='float32')])

    dst = np.float32([np.array([(square_left_corner, 0)], dtype='float32'), np.array([(square_right_corner, 0)], dtype='float32'),
                      np.array([(square_right_corner, height)], dtype='float32'), np.array([(square_left_corner, height)], dtype='float32')])
    # d) use cv2.getPerspectiveTransform() to get M, the transform matrix
    M = cv2.getPerspectiveTransform(src, dst)
    # e) use cv2.warpPerspective() to warp your image to a top-down view
    warped = cv2.warpPerspective(undist, M, img_size)
    plt.imsave(OUTPUT_IMAGES_FOLDER + 'perspective_transform/orginal_image.jpg', img)
    plt.imsave(OUTPUT_IMAGES_FOLDER + 'perspective_transform/orginal_image.jpg', warped)
    return warped, M


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
undistort_image(cv2.imread('camera_cal/calibration1.jpg'), mtx, dist)

# Binary threshold image
test_threshold_img = mpimg.imread('test_images/test5.jpg')
binary_img = Binthreshold.get_combined_threshold(test_threshold_img, 3, OUTPUT_IMAGES_FOLDER)

# Wrap image
warped_image, transform_matrix = apply_perspective_transform(cv2.imread('test_images/straight_lines1.jpg'), CALIBRATION_OUTPUT + 'wide_dist_pickle.p')
plt.imshow(warped_image)

print("End pipeline")