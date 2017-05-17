import cv2
import matplotlib.pyplot as plt
from binthreshold import Binthreshold
from lanepixelfinding import Lanepixelfinding
from imagedistortion import Imagedistortion
from drawresult import Drawresult
import paths


# -------------- Start Lane Lines pipeline here -----------------------------------------------------------------------
if __name__ == "__main__":

    # Camera calibration
    mtx, dist = Imagedistortion.calibrate_camera(paths.CALIBRATION_SOURCE, paths.CALIBRATION_OUTPUT)
    Imagedistortion.undistort_image(cv2.imread(paths.CALIBRATION_SOURCE + 'calibration1.jpg'), mtx, dist,
                                    paths.OUTPUT_IMAGES_FOLDER + paths.DISTORTION_OUTPUT)

    # Wrap image
    original_image = cv2.imread('frame_analysis/problem_frames/frame67.jpg')
    # original_image = cv2.imread('test_images/test2.jpg')
    warped_image, transform_matrix = \
        Imagedistortion.apply_perspective_transform(original_image,
                                                    paths.CALIBRATION_OUTPUT + 'wide_dist_pickle.p',
                                                    paths.OUTPUT_IMAGES_FOLDER + paths.PERSPECTIVE_OUTPUT)
    plt.imshow(warped_image)

    # Binary threshold image
    #test_threshold_img = mpimg.imread('test_images/test5.jpg')
    binary_img = Binthreshold.get_combined_threshold(warped_image, 3, paths.OUTPUT_IMAGES_FOLDER + paths.BINARY_OUTPUT)

    laneFind = Lanepixelfinding()
    laneFind.find_lane_pixels(binary_img, output_folder=paths.OUTPUT_IMAGES_FOLDER + paths.LANES_OUTPUT)
    # left_curverad, right_curverad = Curvaturemeasure.find_curvature(left_fit_m, right_fit_m,
    #                                                                 (binary_img.shape[0]-1)*Lanepixelfinding.YM_PER_PIX)

    # Now our radius of curvature is in meters
    print(laneFind.lline.rad_curvature_m, 'm', laneFind.rline.rad_curvature_m, 'm')

    result = Drawresult.draw_on_lane(binary_img, original_image, transform_matrix, laneFind.lline, laneFind.rline, paths.OUTPUT_IMAGES_FOLDER +
                                     paths.DRAW_OUTPUT)

    print("End pipeline")