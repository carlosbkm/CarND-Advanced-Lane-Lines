from lanepixelfinding import Lanepixelfinding

__author__ = 'Carlos'

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import paths


class Curvaturemeasure(object):

    @staticmethod
    def find_curvature(left_fit, right_fit, y_eval):

        left_curverad = ((1 + (2*left_fit[0]*y_eval + left_fit[1])**2)**1.5) / np.absolute(2*left_fit[0])
        right_curverad = ((1 + (2*right_fit[0]*y_eval + right_fit[1])**2)**1.5) / np.absolute(2*right_fit[0])

        return left_curverad, right_curverad


if __name__ == "__main__":

    laneF = Lanepixelfinding()

    img = mpimg.imread(paths.OUTPUT_IMAGES_FOLDER + paths.BINARY_OUTPUT + 'threshold_output.jpg')
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    left_fit, right_fit = laneF.find_lane_pixels(gray, output_folder=paths.OUTPUT_IMAGES_FOLDER + paths.LANES_OUTPUT)

    left_curverad, right_curverad = Curvaturemeasure.find_curvature(left_fit, right_fit,
                                                                    (gray.shape[0]-1)*Lanepixelfinding.YM_PER_PIX)

    # Now our radius of curvature is in meters
    print(left_curverad, 'm', right_curverad, 'm')
    print('Finished main')