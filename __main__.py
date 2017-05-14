import cv2
import matplotlib.pyplot as plt
from binthreshold import Binthreshold
from lanepixelfinding import Lanepixelfinding
from imagedistortion import Imagedistortion
from curvaturemeasure import Curvaturemeasure
from drawresult import Drawresult
import paths


# -------------- Start Lane Lines pipeline here -----------------------------------------------------------------------
if __name__ == "__main__":

    # Camera calibration
    mtx, dist = Imagedistortion.calibrate_camera(paths.CALIBRATION_SOURCE, paths.CALIBRATION_OUTPUT)
    Imagedistortion.undistort_image(cv2.imread(paths.CALIBRATION_SOURCE + 'calibration1.jpg'), mtx, dist,
                                    paths.OUTPUT_IMAGES_FOLDER + paths.DISTORTION_OUTPUT)

    # Wrap image
    warped_image, transform_matrix = \
        Imagedistortion.apply_perspective_transform(cv2.imread('test_images/straight_lines1.jpg'),
                                                    paths.CALIBRATION_OUTPUT + 'wide_dist_pickle.p',
                                                    paths.OUTPUT_IMAGES_FOLDER + paths.PERSPECTIVE_OUTPUT)
    plt.imshow(warped_image)

    # Binary threshold image
    #test_threshold_img = mpimg.imread('test_images/test5.jpg')
    binary_img = Binthreshold.get_combined_threshold(warped_image, 3, paths.OUTPUT_IMAGES_FOLDER + paths.BINARY_OUTPUT)

    laneFind = Lanepixelfinding()
    left_fit, right_fit = laneFind.find_lane_pixels(binary_img, output_folder=paths.OUTPUT_IMAGES_FOLDER + paths.LANES_OUTPUT)
    left_curverad, right_curverad = Curvaturemeasure.find_curvature(left_fit, right_fit,
                                                                    (binary_img.shape[0]-1)*Lanepixelfinding.YM_PER_PIX)

    # Now our radius of curvature is in meters
    print(left_curverad, 'm', right_curverad, 'm')

    result = Drawresult.draw_on_lane(binary_img, transform_matrix, left_fit, right_fit, paths.OUTPUT_IMAGES_FOLDER +
                                     paths.DRAW_OUTPUT)

    print("End pipeline")