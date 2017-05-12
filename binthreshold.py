__author__ = 'Carlos'

import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

class Binthreshold(object):

    @staticmethod
    def abs_sobel_thresh(image, orient='x', sobel_kernel=3, thresh=(0, 255)):
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        # Calculate directional gradient
        sobel = cv2.Sobel(gray, cv2.CV_64F, int(orient == 'x'), int(orient == 'y'), ksize=sobel_kernel)
        abs_sobel = np.absolute(sobel)
        scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))

        grad_binary = np.zeros_like(scaled_sobel)
        # Apply threshold
        grad_binary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
        return grad_binary

    @staticmethod
    def mag_thresh(image, sobel_kernel=3, mag_thresh=(0, 255)):
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        # Calculate gradient magnitude
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

        gradmag = np.sqrt(sobelx ** 2 + sobely ** 2)

        scale_factor = np.max(gradmag) / 255
        gradmag = (gradmag / scale_factor).astype(np.uint8)

        mag_binary = np.zeros_like(gradmag)
        # Apply threshold
        mag_binary[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1
        return mag_binary

    @staticmethod
    def hls_select(image, thresh=(0, 255)):
        # 1) Convert to HLS color space
        hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
        # 2) Apply a threshold to the S channel
        s_channel = hls[:,:,2]
        binary_output = np.zeros_like(s_channel)
        binary_output[(s_channel > thresh[0]) & (s_channel <= thresh[1])] = 1
        # 3) Return a binary image of threshold result
        return binary_output

    @staticmethod
    def dir_threshold(image, sobel_kernel=3, thresh=(0, np.pi / 2)):
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        # Calculate gradient direction
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
        # Take the absolute value of the gradient direction,
        # apply a threshold, and create a binary image result
        absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
        dir_binary = np.zeros_like(absgraddir)
        dir_binary[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1
        # Apply threshold
        return dir_binary

    @classmethod
    def get_combined_threshold(cls, image, ksize, output_folder=False):

        # Apply each of the thresholding functions
        gradx = cls.abs_sobel_thresh(image, orient='x', sobel_kernel=ksize, thresh=(80, 255))
        grady = cls.abs_sobel_thresh(image, orient='y', sobel_kernel=ksize, thresh=(80, 255))
        mag_binary = cls.mag_thresh(image, sobel_kernel=ksize, mag_thresh=(80, 255))
        dir_binary = cls.dir_threshold(image, sobel_kernel=ksize, thresh=(0, np.pi / 2))
        hls_binary = cls.hls_select(image, thresh=(120, 255))

        combined = np.zeros_like(dir_binary)
        combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1)) | (hls_binary == 1)] = 1

        if output_folder:
            # binary_converted = np.zeros_like(combined)
            # binary_converted[combined == 1] = 255
            # cv2.imwrite(output_folder + 'threshold_original.jpg', image)
            # cv2.imwrite(output_folder + 'threshold_output.jpg', binary_converted)

            # plt.imsave(output_folder + 'threshold_output_gradx.jpg', gradx, cmap='gray')
            # plt.imsave(output_folder + 'threshold_output_grady.jpg', grady, cmap='gray')
            # plt.imsave(output_folder + 'threshold_output_mag_binary.jpg', mag_binary, cmap='gray')
            # plt.imsave(output_folder + 'threshold_output_dir_binary.jpg', dir_binary, cmap='gray')
            # plt.imsave(output_folder + 'threshold_output_hls_binary.jpg', hls_binary, cmap='gray')
            plt.imsave(output_folder + 'binary_threshold/threshold_output.jpg', combined, cmap='gray')
            plt.imsave(output_folder + 'binary_threshold/threshold_original.jpg', image)

        return combined

if __name__ == "__main__":
    window_width = 20
    window_height = 100
    margin = 10

    warped_image = mpimg.imread('output_images/perspective_transform/perspective_transformed.jpg')

    bin_output = Binthreshold.get_combined_threshold(warped_image, 3, 'output_images/')
    plt.imshow(bin_output)

    print("End pipeline")
