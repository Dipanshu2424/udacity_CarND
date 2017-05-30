# **Finding Lane Lines on the Road** 

[//]: # (Image References)

[image1]: /home/emarrig/repos/udacity_CarND/CarND-LaneLines-P1/test_images_output/solidWhiteCurve_gray.png "Grayscale"
[image2]: /home/emarrig/repos/udacity_CarND/CarND-LaneLines-P1/test_images_output/solidWhiteCurve_masked.png "Masked"
[image3]: /home/emarrig/repos/udacity_CarND/CarND-LaneLines-P1/test_images_output/solidWhiteCurve_hough.png "Hough"
[image4]: /home/emarrig/repos/udacity_CarND/CarND-LaneLines-P1/test_images_output/solidWhiteCurve_final.png "Final"


### Reflection

***


### 1. Describe your pipeline. As part of the description, explain how you modified the draw_lines() function.

My pipeline consisted of 5 steps: 

* **Step 1**: Grayscale conversion from colour image to grayscale as seen in the image below.

![alt text][image1]


* **Step 2**: Canny edge detection (with gaussian blurring being applied before the edge detection).

* **Step 3**: Masking. This step was required in order to remove edges that were not relevant to the two lanes as Canny returns all edges that it can detect in an image. The following images shows the results after steps 2 and 3.

![alt text][image2]

* **Step 4**: Hough transformation. The masked image was used as input and by selecting the right parameters, it was possible to remove most of the noise in the middle of the image and end up with the two lanes as can be seen from the next image.
![alt text][image3]

* **Step 5**: The final step was to combine the original image and the output from the hough transformation and get the result that clearly marks the two lanes.
![alt text][image4]


In order to draw a single line on the left and right lanes, I modified the draw_lines() function by finding the slopes of the left and right lines and average them. Then in order to draw two lines (one left and one right) I took one of the points from the detected lines and extrapolated it to the edge of my masking area in order to make sure the lines appear for the best part of the viewpoint. 

### 2. Identify potential shortcomings with your current pipeline


One potential shortcoming would be what would happen when there is a difficulty in detecting edges, for example when the color of the road is very similar to the lane colour. Shade from trees or other objects could potentially affect the edge detection. 

Another shortcoming could be if the lines are disappearing or if they have bigger curvature. Then a straight line will not work that well.


### 3. Suggest possible improvements to your pipeline

At the moment it seems that the lines can become a bit jumpy. One possible improvement would be to have some small "memory" and average the past two three slopes in order to have more stability.
