# **Finding Lane Lines on the Road** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file. But feel free to use some other method and submit a pdf if you prefer.

---

**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report


[//]: # (Image References)

[image1]: ./test_images_output/solidWhiteCurve_gray.png "Grayscale"
[image2]: ./test_images_output/solidWhiteCurve_masked.png "Masked"
[image3]: ./test_images_output/solidWhiteCurve_hough.png "Hough"
[image4]: ./test_images_output/solidWhiteCurve_final.png "Final"
---

### Reflection

### 1. Describe your pipeline. As part of the description, explain how you modified the draw_lines() function.

My pipeline consisted of 5 steps. 

* **Step 1**: Grayscale conversion

![alt text][image1]

* **Step 2**: Gaussian blurring

* **Step 3**: Canny

* **Step 4**: Masking edges

![alt text][image2]

* **Step 5**: Hough
![alt text][image3]

* **Step 6**: combine image from hough transformation with the original image. 
![alt text][image4]


In order to draw a single line on the left and right lanes, I modified the draw_lines() function by finding the slopes of the left and right lines and average them.

### 2. Identify potential shortcomings with your current pipeline


One potential shortcoming would be what would happen when ... 

Another shortcoming could be ...


### 3. Suggest possible improvements to your pipeline

A possible improvement would be to ...

Another potential improvement could be to ...
