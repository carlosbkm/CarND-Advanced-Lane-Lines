__author__ = 'Carlos'

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import paths

class Lanepixelfinding(object):

    YM_PER_PIX = 30/720
    XM_PER_PIX = 3.7/700

    def __init__(self):
        self.left_fit = self.right_fit = None
        self.best_line_fit = None
        self.left_base = self.right_base = 0
        # self.out_img = None

    def find_lane_pixels(self, binary_warped, margin=100, nwindows=9, minpix=50, output_folder=None):
        #plt.imshow(binary_warped)

        # Create an output image to draw on and  visualize the result
        out_img = np.dstack((binary_warped, binary_warped, binary_warped))
        #plt.imshow(out_img)
        # out_img = out_img *255

        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        left_lane_inds, right_lane_inds = \
            self.__get_lane_inds(binary_warped, nonzerox, nonzeroy, margin, nwindows, minpix, out_img)

        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        # Fit a second order polynomial to each, scaling to real values in meters
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)

        if(self.best_line_fit is not None):
            if(self.best_line_fit == 'left'):
                right_fit[0:2] = left_fit[0:2]
                right_fit[2] = self.right_base
            else:
                left_fit[0:2] = right_fit[0:2]
                left_fit[2] = self.left_base

        self.__plot_and_save(binary_warped, left_fit, right_fit, nonzerox, nonzeroy,
                      left_lane_inds, right_lane_inds, output_folder, out_img)

        # Update left and right fit so that we can use it for the next frame
        self.left_fit = left_fit
        self.right_fit = right_fit
        # We return the fitted polynomial scaled in meters
        left_fit_m = np.polyfit(lefty*self.YM_PER_PIX, leftx*self.XM_PER_PIX, 2)
        right_fit_m = np.polyfit(righty*self.YM_PER_PIX, rightx*self.XM_PER_PIX, 2)

        return left_fit_m, right_fit_m

    def __get_lane_inds(self, binary_warped, nonzerox, nonzeroy, margin, nwindows, minpix, out_img):
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
        if self.left_fit is not None and self.right_fit is not None:
            left_lane_inds = ((nonzerox > (self.left_fit[0]*(nonzeroy**2) + self.left_fit[1]*nonzeroy + self.left_fit[2] - margin)) & (nonzerox < (self.left_fit[0]*(nonzeroy**2) + self.left_fit[1]*nonzeroy + self.left_fit[2] + margin)))
            right_lane_inds = ((nonzerox > (self.right_fit[0]*(nonzeroy**2) + self.right_fit[1]*nonzeroy + self.right_fit[2] - margin)) & (nonzerox < (self.right_fit[0]*(nonzeroy**2) + self.right_fit[1]*nonzeroy + self.right_fit[2] + margin)))
        else:
            left_lane_inds, right_lane_inds = self.__sliding_window(binary_warped, nonzerox, nonzeroy, margin, nwindows, minpix, out_img)

        return left_lane_inds, right_lane_inds

    def __sliding_window(self, binary_warped, nonzerox, nonzeroy, margin=100, nwindows=9, minpix=50, out_img=None):
        # Assuming you have created a warped binary image called "binary_warped"
        # Take a histogram of the bottom half of the image
        histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)

        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0]//2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        if(histogram[leftx_base] > histogram[rightx_base]):
            self.best_line_fit = 'left'
        else:
            self.best_line_fit = 'right'

        # Current positions to be updated for each window
        leftx_current = leftx_base
        rightx_current = rightx_base

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

            if(out_img is not None):
                # Draw the windows on the visualization image
                cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2)
                cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2)
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

        self.left_base = leftx_base + margin//2
        self.right_base = rightx_base + margin//2
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        # rgb = out_img[...,::-1]
        #plt.imshow(out_img)

        # cv2.imwrite(paths.OUTPUT_IMAGES_FOLDER + paths.LANES_OUTPUT + 'pruebecita.jpg', out_img)

        return left_lane_inds, right_lane_inds

    def __plot_and_save(self, binary_warped, left_fit, right_fit, nonzerox, nonzeroy,
                      left_lane_inds, right_lane_inds, output_folder=None, out_img=None):
        # Generate x and y values for plotting
        ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

        BLUE = [0, 0, 255]
        RED = [255, 0, 0]
        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = RED
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = BLUE

        # fig, im = plt.subplots()
        # im.imshow(out_img)
        # im.plot(left_fitx, ploty, color='yellow')
        # im.plot(right_fitx, ploty, color='yellow')
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

    left_fit_m, right_fit_m = laneF.find_lane_pixels(gray, output_folder=paths.OUTPUT_IMAGES_FOLDER + paths.LANES_OUTPUT)

    print("End pipeline")
