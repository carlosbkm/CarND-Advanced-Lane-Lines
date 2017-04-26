__author__ = 'Carlos'

import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

class Binthreshold(object):

    @staticmethod
    def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0, 255)):
        # Calculate directional gradient
        sobel = cv2.Sobel(img, cv2.CV_64F, int(orient == 'x'), int(orient == 'y'), ksize=sobel_kernel)
        abs_sobel = np.absolute(sobel)
        scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))

        grad_binary = np.zeros_like(scaled_sobel)
        # Apply threshold
        grad_binary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
        return grad_binary

    @staticmethod
    def mag_thresh(image, sobel_kernel=3, mag_thresh=(0, 255)):
        # Calculate gradient magnitude
        sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

        gradmag = np.sqrt(sobelx ** 2 + sobely ** 2)

        scale_factor = np.max(gradmag) / 255
        gradmag = (gradmag / scale_factor).astype(np.uint8)

        mag_binary = np.zeros_like(gradmag)
        # Apply threshold
        mag_binary[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1
        return mag_binary

    @staticmethod
    def dir_threshold(image, sobel_kernel=3, thresh=(0, np.pi / 2)):
        # Calculate gradient direction
        sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
        # Take the absolute value of the gradient direction,
        # apply a threshold, and create a binary image result
        absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
        dir_binary = np.zeros_like(absgraddir)
        dir_binary[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1
        # Apply threshold
        return dir_binary

    @classmethod
    def get_combined_threshold(image, ksize) :
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply each of the thresholding functions
        gradx = Binthreshold.abs_sobel_thresh(gray, orient='x', sobel_kernel=ksize, thresh=(60, 100))
        grady = Binthreshold.abs_sobel_thresh(gray, orient='y', sobel_kernel=ksize, thresh=(600, 100))
        mag_binary = Binthreshold.mag_thresh(gray, sobel_kernel=ksize, mag_thresh=(80, 200))
        dir_binary = Binthreshold.dir_threshold(gray, sobel_kernel=ksize, thresh=(0, np.pi / 2))

        combined = np.zeros_like(dir_binary)
        combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1
        return image