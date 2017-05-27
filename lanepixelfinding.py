__author__ = 'Carlos'

import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.image as mpimg
import cv2
import paths
from line import Line

class Lanepixelfinding(object):

    YM_PER_PIX = 30/720
    XM_PER_PIX = 3.7/700
    LANE_WIDTH = 700
    RADIUS_THRESHOLD_MAX = 2000
    RADIUS_THRESHOLD_MIN = 200
    RADIUS_DIFF_THRESHOLD = 500
    out_img = None

    def __init__(self):
        self.lline = Line()
        self.rline = Line()
        # self.best_line_fit = None

    def find_lane_pixels(self, binary_warped, margin=100, nwindows=9, minpix=50, output_folder=None):

        self.out_img = np.dstack((binary_warped, binary_warped, binary_warped))
        # out_img = out_img *255

        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        left_lane_inds, right_lane_inds = \
            self.__get_lane_inds(binary_warped, nonzerox, nonzeroy, margin, nwindows, minpix)

        # Extract left and right line pixel positions
        lline_x = nonzerox[left_lane_inds]
        lline_y = nonzeroy[left_lane_inds]
        rline_x = nonzerox[right_lane_inds]
        rline_y = nonzeroy[right_lane_inds]

        # Fit a second order polynomial to each, scaling to real values in meters
        left_fit = np.polyfit(lline_y, lline_x, 2)
        right_fit = np.polyfit(rline_y, rline_x, 2)

        # Generate x and y values
        ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

        # if(self.best_line_fit is not None):
        #     if(self.best_line_fit == 'left'):
        #         self.rline.get_coeffs()[0:2] = self.lline.get_coeffs()[0:2]
        #         self.rline.get_coeffs()[2] = self.lline.get_coeffs()[2] + self.LANE_WIDTH
        #     else:
        #         self.lline.get_coeffs()[0:2] = self.rline.get_coeffs()[0:2]
        #         self.lline.get_coeffs()[2] = self.rline.get_coeffs()[2] - self.LANE_WIDTH

        # We return the fitted polynomial scaled in meters
        left_fit_m = np.polyfit(lline_y*self.YM_PER_PIX, lline_x*self.XM_PER_PIX, 2)
        right_fit_m = np.polyfit(rline_y*self.YM_PER_PIX, rline_x*self.XM_PER_PIX, 2)

        left_curverad, right_curverad = self.__find_curvature(left_fit_m, right_fit_m, (binary_warped.shape[0]-1)*self.YM_PER_PIX)

        detected_left = detected_right = self.__correctly_detected(left_curverad, right_curverad)


        detected_left = False if left_fit is None else True
        detected_right = False if right_fit is None else True

        # if left_fit is None: detected_left = False
        # if right_fit is None: detected_right = False

        self.lline.update_values(left_fit, left_fitx, ploty, left_curverad)
        self.rline.update_values(right_fit, right_fitx, ploty, right_curverad)
        # pprint(vars(self.lline))
        # pprint(vars(self.rline))

        self.__plot_and_save(binary_warped, self.lline, self.rline, nonzerox, nonzeroy, left_lane_inds,
                             right_lane_inds, output_folder)

    def __correctly_detected(self, left_curverad, right_curverad):
        if abs(left_curverad - right_curverad) > self.RADIUS_DIFF_THRESHOLD:
            return False
        if(left_curverad > self.RADIUS_THRESHOLD_MAX or left_curverad < self.RADIUS_THRESHOLD_MIN
           or right_curverad > self.RADIUS_THRESHOLD_MAX or right_curverad < self.RADIUS_THRESHOLD_MIN):
            return False
        return True

    def __find_curvature(self, left_fit, right_fit, y_eval):

        left_curverad = ((1 + (2*left_fit[0]*y_eval + left_fit[1])**2)**1.5) / np.absolute(2*left_fit[0])
        right_curverad = ((1 + (2*right_fit[0]*y_eval + right_fit[1])**2)**1.5) / np.absolute(2*right_fit[0])

        return left_curverad, right_curverad

    def __get_lane_inds(self, binary_warped, nonzerox, nonzeroy, margin, nwindows, minpix):
        """
        Returns the lane pixels indexes. It uses sliding window method for the first frame, otherwise we use the fit
        calculated from the previous frame

        :param binary_warped:
        :param nonzerox:
        :param nonzeroy:
        :param nwindows:
        :param minpix:
        :return left_lane_inds, right_lane_inds:
        """
        # if self.lline.detected is True and self.rline.detected is True:
        #     left_lane_inds = ((nonzerox > (self.lline.get_coeffs()[0]*(nonzeroy**2) + self.lline.get_coeffs()[1]*nonzeroy + self.lline.get_coeffs()[2] - margin)) & (nonzerox < (self.lline.get_coeffs()[0]*(nonzeroy**2) + self.lline.get_coeffs()[1]*nonzeroy + self.lline.get_coeffs()[2] + margin)))
        #     right_lane_inds = ((nonzerox > (self.rline.get_coeffs()[0]*(nonzeroy**2) + self.rline.get_coeffs()[1]*nonzeroy + self.rline.get_coeffs()[2] - margin)) & (nonzerox < (self.rline.get_coeffs()[0]*(nonzeroy**2) + self.rline.get_coeffs()[1]*nonzeroy + self.rline.get_coeffs()[2] + margin)))
        # else:
        #     left_lane_inds, right_lane_inds = self.__sliding_window(binary_warped, nonzerox, nonzeroy, margin, nwindows, minpix)

        left_lane_inds, right_lane_inds = self.__sliding_window(binary_warped, nonzerox, nonzeroy, margin, nwindows, minpix)

        return left_lane_inds, right_lane_inds

    def __sliding_window(self, binary_warped, nonzerox, nonzeroy, margin=100, nwindows=9, minpix=50):
        # Assuming you have created a warped binary image called "binary_warped"
        # Take a histogram of the bottom half of the image
        histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)

        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0]//2)
        self.lline.x_base = np.argmax(histogram[:midpoint])
        self.rline.x_base = np.argmax(histogram[midpoint:]) + midpoint

        if(histogram[self.lline.x_base] > histogram[self.rline.x_base]):
            self.best_line_fit = 'left'
        else:
            self.best_line_fit = 'right'

        # Current positions to be updated for each window
        leftx_current = self.lline.x_base
        rightx_current = self.rline.x_base

        # Set height of windows
        window_height = np.int(binary_warped.shape[0]/nwindows)

        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []

        # Step through the windows one by one
        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = binary_warped.shape[0] - (window+1)*window_height
            win_y_high = binary_warped.shape[0] - window*window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin

            if(self.out_img is not None):
                # Draw the windows on the visualization image
                cv2.rectangle(self.out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2)
                cv2.rectangle(self.out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2)
            # Identify the nonzero pixels in x and y within the window
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_left_inds) > minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
        # self.lline.detected = True
        # self.rline.detected = True

        return left_lane_inds, right_lane_inds

    def __plot_and_save(self, binary_warped, lline, rline, nonzerox, nonzeroy,
                      left_lane_inds, right_lane_inds, output_folder=None):

        BLUE = [0, 0, 255]
        RED = [255, 0, 0]
        # Create an output image to draw on and  visualize the result
        self.out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = RED
        self.out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = BLUE

        fig, im = plt.subplots()
        im.imshow(self.out_img)
        im.plot(lline.fitx, lline.fity, color='yellow')
        im.plot(rline.fitx, rline.fity, color='yellow')
        # im.xlim(0, 1280)
        # im.ylim(720, 0)

        if output_folder is not None:
            fig.savefig(output_folder + 'sliding_window_result.jpg')
            # plt.imsave(output_folder + 'sliding_window_result.jpg', im)
            plt.imsave(output_folder + 'original_image.jpg', binary_warped, cmap='gray')

if __name__ == "__main__":
    window_width = 20
    window_height = 100
    margin = 10

    laneF = Lanepixelfinding()

    img = mpimg.imread(paths.OUTPUT_IMAGES_FOLDER + paths.BINARY_OUTPUT + 'threshold_output.jpg')
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    laneF.find_lane_pixels(gray, output_folder=paths.OUTPUT_IMAGES_FOLDER + paths.LANES_OUTPUT)

    print("End pipeline")
