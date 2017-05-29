## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./examples/undistort_output.png "Undistorted"
[image2]: ./test_images/test1.jpg "Road Transformed"
[image3]: ./examples/binary_combo_example.jpg "Binary Example"
[image4]: ./examples/warped_straight_lines.jpg "Warp Example"
[image5]: ./examples/color_fit_lines.jpg "Fit Visual"
[image6]: ./examples/example_output.jpg "Output"
[video1]: ./project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the file "imagedistortion.py". I created this file to put together all the functions required for camera calibration.

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection:

<img src="https://github.com/carlosbkm/CarND-Advanced-Lane-Lines/blob/master/calibration_results/output_corners/corners_found11.jpg?raw=true" alt="objpoints" width="400"/>

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result:

Original image
<img src="https://github.com/carlosbkm/CarND-Advanced-Lane-Lines/blob/master/output_images/distortion_correction/chessboard_original.jpg?raw=true" alt="chessborard original"/>

Undistorted image
<img src="https://github.com/carlosbkm/CarND-Advanced-Lane-Lines/blob/master/output_images/distortion_correction/chessboard_undistorted.jpg?raw=true" alt="chessboard undistorted"/>

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
<img src="https://github.com/carlosbkm/CarND-Advanced-Lane-Lines/blob/master/test_images/test3.jpg?raw=true" />

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

The code for this section can be found in the binthreshold.py file.

After trying different options, I used a combination of color gradient in X axis and HLS. This proved to work the best for me. That is implemented in the get_combined_threshold method of the file. In the image below can be seen the result of every filter separately:

<img src="https://github.com/carlosbkm/CarND-Advanced-Lane-Lines/blob/master/writeup_images/gradient_combinations.png?raw=true" />

However, some frames still caused undesired effects, due to the dashed line not containing enough information for the polynomial to fit the coefficients. I found that growing the stroke of the thresholded image helped a lot to get rid of this. I used cv2.dilate to achieve that (line 78 of binthreshold.py). The application of the combined thresholds and the dilate method gives this result:

<img src="https://github.com/carlosbkm/CarND-Advanced-Lane-Lines/blob/master/writeup_images/dilated_example.png?raw=true" />


#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform is included in the file `imagedistortion.py`. The function `apply_perspective_transform` (line 113) takes as input an image and a source and destination folder, and returns the warped image and the transformation matrix which will be used later to unwarp the image.

I chose the hardcode the source and destination points in the following manner:

```python
        height = image.shape[0]
        upper_limit = 472
        square_width = 720
        square_left_corner = 250
        square_right_corner = square_left_corner + square_width
        src = np.float32([np.array([(563, upper_limit)], dtype='float32'), np.array([(725, upper_limit)], dtype='float32'),
                          np.array([(1113, height)], dtype='float32'), np.array([(171, height)], dtype='float32')])

        dst = np.float32([np.array([(square_left_corner, 0)], dtype='float32'), np.array([(square_right_corner, 0)], dtype='float32'),
                          np.array([(square_right_corner, height)], dtype='float32'), np.array([(square_left_corner, height)], dtype='float32')])
```

The result of this step can be found in `.output_images/perspective_transform`

Original image:
<img src="https://github.com/carlosbkm/CarND-Advanced-Lane-Lines/blob/using-line-class/output_images/perspective_transform/original_image.jpg?raw=true"/>

Warped image:
<img src="https://github.com/carlosbkm/CarND-Advanced-Lane-Lines/blob/using-line-class/output_images/perspective_transform/perspective_transformed.jpg?raw=true" />
<img src="" />


#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Then I fit my lane lines with a 2nd order polynomial like this:

![alt text][image5]

I implemented mi solution in the file `lanepixelfinding.py`. 

To find the hot pixels of the lane lines, I use a sliding window algorithm to scan the image (line 133: `__sliding_window`). 
First, I got the histogram of the binary image. Then, I splitted the image by half and got the maximum histogram values for each side. That was the base x position for my window to start scanning. Then I run 9 windows along the image and got the lane indices for the hot pixels. Then, in line 29: `find_lines` I fit a second order polynomial to the non zero values of the positions found. 

In line 186: `__plot_and_save`, I get the x values fit using the previously calculated polynomial and plot the lines found. And example of the result obtained is at `.output_images/lane_detection/`:

Binary image
<img src="https://github.com/carlosbkm/CarND-Advanced-Lane-Lines/blob/using-line-class/output_images/lane_detection/original_image.jpg?raw=true"/>

Sliding window result:
<img src="https://github.com/carlosbkm/CarND-Advanced-Lane-Lines/blob/using-line-class/output_images/lane_detection/sliding_window_result.jpg?raw=true"/>

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I calculated the curvature in `lanepixelfinding.py` line 104:
```python
    def __find_curvature(self, left_fit, right_fit, y_eval):

        left_curverad = ((1 + (2*left_fit[0]*y_eval + left_fit[1])**2)**1.5) / np.absolute(2*left_fit[0])
        right_curverad = ((1 + (2*right_fit[0]*y_eval + right_fit[1])**2)**1.5) / np.absolute(2*right_fit[0])

        return left_curverad, right_curverad
```
For getting the curvature in meters, I used the following equilavence meters/pixels:
*YM_PER_PIX = 30/720
*XM_PER_PIX = 3.7/700

And applied it in line 43.

For calculating the position of the vehicle, I considered that the camera is fixed in the center of the image, and calculate the difference with the middle of the lane. The code is in the method `get_camera_offset`of the file `drawresult.py`:
```python
        camera_position = img_width/2
        lane_center = (rline.x_base - lline.x_base)/2 + lline.x_base
        return (lane_center - camera_position) * lane.XM_PER_PIX * 100
```

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in the file `drawresult.py` in the method `draw_on_lane`. Here is an example of my result on a test image, which can be also found in `.output_images/draw_lane/`:

<img src="https://github.com/carlosbkm/CarND-Advanced-Lane-Lines/blob/using-line-class/output_images/draw_lane/result_image.jpg?raw=true"/>

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)

---



### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  
