__author__ = 'Carlos'

import cv2
from binthreshold import Binthreshold
from lanepixelfinding import Lanepixelfinding
from imagedistortion import Imagedistortion
from drawresult import Drawresult
import paths
from moviepy.editor import VideoFileClip


def process_image(image):
    result = lane_find_pipeline(image)
    return result


def lane_find_pipeline(image):

    # 1. Wrap image
    warped_image, transform_matrix = \
        Imagedistortion.apply_perspective_transform(image,
                                                    paths.CALIBRATION_OUTPUT + 'wide_dist_pickle.p',
                                                    paths.OUTPUT_IMAGES_FOLDER + paths.PERSPECTIVE_OUTPUT)

    # 2. Binary threshold image
    #test_threshold_img = mpimg.imread('test_images/test5.jpg')
    binary_img = Binthreshold.get_combined_threshold(warped_image, 5, paths.OUTPUT_IMAGES_FOLDER + paths.BINARY_OUTPUT)


    laneFind.find_lane_pixels(binary_img, output_folder=paths.OUTPUT_IMAGES_FOLDER + paths.LANES_OUTPUT)

    result = Drawresult.draw_on_lane(binary_img, image, transform_matrix, laneFind.lline, laneFind.rline, paths.OUTPUT_IMAGES_FOLDER +
                                     paths.DRAW_OUTPUT)
    return result

laneFind = Lanepixelfinding()
# Camera calibration
mtx, dist = Imagedistortion.calibrate_camera(paths.CALIBRATION_SOURCE, paths.CALIBRATION_OUTPUT)
Imagedistortion.undistort_image(cv2.imread(paths.CALIBRATION_SOURCE + 'calibration1.jpg'), mtx, dist,
                                paths.OUTPUT_IMAGES_FOLDER + paths.DISTORTION_OUTPUT)

output_filename = 'project_video_result.mp4'
clip1 = VideoFileClip("project_video.mp4")
video_output = clip1.fl_image(process_image) #NOTE: this function expects color images!!
video_output.write_videofile(output_filename, audio=False)

print("End pipeline")