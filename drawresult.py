__author__ = 'Carlos'


import numpy as np
import matplotlib.image as mpimg
import cv2
from numpy.linalg import inv
from lanepixelfinding import Lanepixelfinding as lane


class Drawresult(object):

    @staticmethod
    def draw_on_lane(warped_binary, original_image, transform_matrix, lline, rline, debug=False, output_folder=None):

        ploty = np.linspace(0, warped_binary.shape[0]-1, warped_binary.shape[0])
        # Create an image to draw the lines on
        warp_zero = np.zeros_like(warped_binary).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

        left_fitx = lline.get_coeff_mean()[0]*ploty**2 + lline.get_coeff_mean()[1]*ploty + lline.get_coeff_mean()[2]
        right_fitx = rline.get_coeff_mean()[0]*ploty**2 + rline.get_coeff_mean()[1]*ploty + rline.get_coeff_mean()[2]

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

        text_curvature = 'Curve radius: ' + '{:.0f} m'.format(np.mean([lline.get_curvature(),rline.get_curvature()]))
        text_curvature_l = 'Left radius: ' + '{:.0f} m'.format(lline.get_curvature())
        text_curvature_r = 'Right radius: ' + '{:.0f} m'.format(rline.get_curvature())
        text_camera_offset = \
            'Offset: ' + '{:.2f} cm'.format(Drawresult.get_camera_offset(warped_binary.shape[1], lline, rline))
        cv2.putText(result, text_camera_offset, (550,625), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (10,25,20),2)
        cv2.putText(result, text_curvature, (550,650), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (10,25,20),2)
        cv2.putText(result, text_curvature_l, (20,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (10,25,20),2)
        cv2.putText(result, text_curvature_r, (1000,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (10,25,20),2)

        # if debug:
        #     Drawresult.write_lane_stats(result, lline, 10, 'LEFT')
        #     Drawresult.write_lane_stats(result, rline, 320, 'RIGHT')

        if(output_folder is not None):
            cv2.imwrite(output_folder + 'result_image.jpg', result)

        return result

    @staticmethod
    def write_lane_stats(image, line, y_pos, title):

        coeffs = 'Coeffs ' + str(line.get_coeff_mean())

        diffpercent = 'Diff percent: ' + str(line.get_diff()) + ' %'
        diffmean = 'Diff mean: ' + str(line.get_coeff_mean()) + ' %'
        threshold = 'Threshold: ' + str(line.PERCENTAGE_THRESHOLD) + ' %'
        fitbuffer = 'Buffer fit length: ' + str(len(line.buffer_coeffs))
        fitx = 'X base: ' + str(line.fitx[0])

        x_pos = 10

        cv2.putText(image, title, (x_pos,y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255),2)
        cv2.putText(image, diffpercent, (x_pos,y_pos + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255),2)
        cv2.putText(image, threshold, (x_pos,y_pos + 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255),2)
        cv2.putText(image, diffmean, (x_pos,y_pos + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255),2)
        cv2.putText(image, fitbuffer, (x_pos,y_pos + 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255),2)
        cv2.putText(image, fitx, (x_pos,y_pos + 125), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255),2)
        cv2.putText(image, 'THRESHOLD', (x_pos,y_pos + 150), cv2.FONT_HERSHEY_SIMPLEX, 0.9, line.threshold_color,2)
        cv2.putText(image, coeffs, (x_pos,y_pos + 200), cv2.FONT_HERSHEY_SIMPLEX, 0.6, line.threshold_color,2)



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
        lane_center = (rline.x_base - lline.x_base)/2 + lline.x_base
        return (lane_center - camera_position) * lane.XM_PER_PIX * 100


if __name__ == "__main__":

    warped = mpimg.imread('output_images/binary_threshold/threshold_original.jpg')
    Drawresult.draw_on_lane(warped)

    print('Finished draw')