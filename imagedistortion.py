__author__ = 'Carlos'

import os
import numpy as np
import cv2
import glob
import pickle
import matplotlib.pyplot as plt
import paths



class Imagedistortion(object):

    @staticmethod
    def obtain_object_image_points(images_directory, output_folder=None):
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

                if(output_folder is not None):
                    # Draw and display the corners
                    cv2.drawChessboardCorners(img, (9, 6), corners, ret)
                    output_corners_directory = output_folder + 'output_corners'
                    if not os.path.exists(output_corners_directory):
                        os.makedirs(output_corners_directory)
                    write_name = output_corners_directory + '/corners_found'+str(idx)+'.jpg'
                    cv2.imwrite(write_name, img)
                    #cv2.imshow('img', img)
                    #cv2.waitKey(500)
                    #cv2.destroyAllWindows()

        return objpoints, imgpoints


    @staticmethod
    def undistort_image(img, mtx, dist, output_folder):
        """
        Takes a test image and applies calibration correction. It saves the output into output_images folder.
        :param img:
        :param mtx:
        :param dist:
        :return dst:
        """
        dst = cv2.undistort(img, mtx, dist, None, mtx)

        if(output_folder):
            cv2.imwrite(output_folder + 'chessboard_original.jpg', img)
            cv2.imwrite(output_folder + 'chessboard_undistorted.jpg', dst)

        return dst


    @staticmethod
    def calibrate_camera(images_directory, output_folder):
        """
        Calibrates the camera using the images in a folder.
        It saves the calibration result into calibration_results/wide_dist_pickle.p

        :param images_directory:
        :return mtx, dist:
        """
        img_size = (1280, 720)
        objpoints, imgpoints = Imagedistortion.obtain_object_image_points(images_directory, output_folder)
        # Do camera calibration given object points and image points
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)

        # Save the camera calibration result for later use (we won't worry about rvecs / tvecs)
        dist_pickle = {}
        dist_pickle["mtx"] = mtx
        dist_pickle["dist"] = dist
        if(output_folder):
            pickle.dump(dist_pickle, open(output_folder + paths.CALIBRATION_FILENAME, "wb"))

        return mtx, dist


    @staticmethod
    def apply_perspective_transform(img, dist_filename, output_folder):
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
        if(output_folder):
            plt.imsave(output_folder + 'original_image.jpg', img)
            plt.imsave(output_folder + 'perspective_transformed.jpg', warped)
        return warped, M