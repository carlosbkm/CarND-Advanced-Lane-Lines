__author__ = 'Carlos'


import numpy as np
import matplotlib.image as mpimg
import cv2
from numpy.linalg import inv
from lanepixelfinding import Lanepixelfinding as lane


class Drawresult(object):

    @staticmethod
    def draw_on_lane(warped_binary, original_image, transform_matrix, lline, rline, output_folder=None):

        ploty = np.linspace(0, warped_binary.shape[0]-1, warped_binary.shape[0])
        # ploty = np.arange(11)*warped_binary.shape[0]/10
        # Create an image to draw the lines on
        warp_zero = np.zeros_like(warped_binary).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

        left_fitx = lline.fit_coeffs[0]*ploty**2 + lline.fit_coeffs[1]*ploty + lline.fit_coeffs[2]
        right_fitx = rline.fit_coeffs[0]*ploty**2 + rline.fit_coeffs[1]*ploty + rline.fit_coeffs[2]

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
        #plt.imshow(color_warp)

        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        Minv = inv(transform_matrix)
        newwarp = cv2.warpPerspective(color_warp, Minv, (original_image.shape[1], original_image.shape[0]))
        #plt.imshow(newwarp)
        # Combine the result with the original image
        result = cv2.addWeighted(original_image, 1, newwarp, 0.3, 0)

        text_curvature = 'Curve radius: ' + '{:.0f} m'.format(np.mean([lline.rad_curvature_m,rline.rad_curvature_m]))
        text_curvature_l = 'Left radius: ' + '{:.0f} m'.format(lline.rad_curvature_m)
        text_curvature_r = 'Right radius: ' + '{:.0f} m'.format(rline.rad_curvature_m)
        text_camera_offset = \
            'Offset: ' + '{:.2f} cm'.format(Drawresult.get_camera_offset(warped_binary.shape[1], lline, rline))
        cv2.putText(result, text_camera_offset, (550,625), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (10,25,20),2)
        cv2.putText(result, text_curvature, (550,650), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (10,25,20),2)
        cv2.putText(result, text_curvature_l, (20,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (10,25,20),2)
        cv2.putText(result, text_curvature_r, (1000,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (10,25,20),2)
        #plt.imshow(result)
        if(output_folder is not None):
            cv2.imwrite(output_folder + 'result_image.jpg', result)

        return result

    @staticmethod
    def get_camera_offset(img_width, lline, rline):
        '''
            Returns the offset of the camera from the center of the lane in centimeters
        :param img_width:
        :param lline:
        :param rline:
        :return:
        '''
        camera_position = img_width/2
        lane_center = (rline.x_base - lline.x_base)/2
        return (lane_center - camera_position) * lane.XM_PER_PIX * 100


if __name__ == "__main__":

    warped = mpimg.imread('output_images/binary_threshold/threshold_original.jpg')
    Drawresult.draw_on_lane(warped)

    print('Finished draw')