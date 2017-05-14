__author__ = 'Carlos'


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import paths
from numpy.linalg import inv


class Drawresult(object):

    @staticmethod
    def draw_on_lane(warped_binary, original_image, transform_matrix, left_fit, right_fit, output_folder=None):

        ploty = np.linspace(0, warped_binary.shape[0]-1, warped_binary.shape[0])
        # ploty = np.arange(11)*warped_binary.shape[0]/10
        # Create an image to draw the lines on
        warp_zero = np.zeros_like(warped_binary).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

        # plt.plot(left_fitx, ploty, color='yellow')
        # plt.plot(right_fitx, ploty, color='yellow')
        # plt.xlim(0, 1280)
        # plt.ylim(720, 0)

        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
        plt.imshow(color_warp)

        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        Minv = inv(transform_matrix)
        newwarp = cv2.warpPerspective(color_warp, Minv, (original_image.shape[1], original_image.shape[0]))
        plt.imshow(newwarp)
        # Combine the result with the original image
        result = cv2.addWeighted(original_image, 1, newwarp, 0.3, 0)
        plt.imshow(result)
        if(output_folder is not None):
            plt.imsave(output_folder + 'result_image.jpg', result)

        return result


if __name__ == "__main__":

    warped = mpimg.imread('output_images/binary_threshold/threshold_original.jpg')
    Drawresult.draw_on_lane(warped)


    print('Finished draw')