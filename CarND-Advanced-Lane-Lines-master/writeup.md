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

[image1]: ./output_images/undistorted_chess.png "Undistorted"
[image2]: ./output_images/undistorted_test1.jpg "Road Transformed"
[image3]: ./output_images/straight_lines1_thresh.png "Binary Example"
[image4]: ./output_images/straight_lines1_warp.png "Warp Example"
[image5]: ./output_images/histogram.png "Histogram"
[image5a]: ./output_images/fitted_windows.png "Fitted windows"
[image6]: ./output_images/pipeline_images.png "Output"
[video1]: /project_video_out.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the second code cell of the IPython notebook located in "project.ipynb".

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection. 

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image2]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image (thresholding steps are in cell #5 of the IPython notebook).  Here's an example of my output for this step:

![alt text][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `warper()`, which appears in the 5th code cell of the IPython notebook.  The `warper()` function takes as inputs an image (`img`).  I chose the hardcode the source and destination points inside the function in the following manner:

```python
src = np.float32(
        [[(img_size[0] / 2) - 60, img_size[1] / 2 + 100],
        [((img_size[0] / 6) - 5), img_size[1]],
        [(img_size[0] * 5 / 6) + 45, img_size[1]],
        [(img_size[0] / 2 + 60), img_size[1] / 2 + 100]])
    dst = np.float32(
        [[(img_size[0] / 4), 0],
        [(img_size[0] / 4), img_size[1]],
        [(img_size[0] * 3 / 4), img_size[1]],
        [(img_size[0] * 3 / 4), 0]])
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 585, 460      | 320, 0        | 
| 208, 720      | 320, 720      |
| 1111, 720     | 960, 720      |
| 700, 460      | 960, 0        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?
In order to identify lane line lines I used the histogram method where I tried to identify the pixels in the image with the highest concentration of pixels in the x axis. One example of a histogram that corresponds to the `straight_lines1.jpg` is the following:

![alt text][image5]

Following the methodology described in the lectures I used the points in the x axis with the highest concentraion of pixels based on the histogram and searched moving upwards in the y axis for pixels  using search windows / rectangles. Having the pixels that belonged to each of the left and right lines I used a second order `numpy.polyfit()`function to find the best fit. This process is part of the function `get_lanes_before_fit()`which is located in the 10th cell of the IPYthon notebook.
For each subsequent frame, I did not calculate I used the previous values for the left and right fitted lines and used that as as a base for the calculation of the lines. This is implemented in the function `get_lanes_after_fit()` which is located in the 10th cell of the IPYthon notebook.
The results on two subsequent frames can be seen in the next imeg:
![alt text][image5a]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

The radius of the curvature of the lane in meters is calculated in function `curvature()` which is located in the 10th cell of the IPYthon notebook. The function calculates the curvature of each lane as well as the distance of the car from the center of the image.

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in the function `draw_lane()` which is located in the 10th cell of the IPython notebook..  Here is an example of my result on a test image:

![alt text][image6]
---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video_out.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The main issues I faced were during specific frames of the video where the lines were not detected properly and this resulted in loss of the lane. In order to get past this, I used the Line() class and stored the previous values of the fitted lines and extracted a best fit value from the past four values. This alleviated most of the problems but in some frames I still faced the issue of the two lines merging. This was due to the fact that the road was not clear for subsequent frames and the histogram values were quite noisy. In order to avoid getting fitted lines that merged, I filtered the fitted values based on their place on the x axis. (function `pipeline()`in the 14th cell of the notebook, lines 22-24). I cannot say that I am quite happy with this approach as I think that it could be avoided with better thresholding and filtering of the images.
