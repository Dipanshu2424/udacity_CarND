# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from IPython.display import HTML
from utils import *
import numpy as np
import pickle
from scipy.ndimage.measurements import label
import matplotlib.pyplot as plt

# Retrieve the model and the scaler
X_scaler = pickle.load(open("scaler.pkl", "rb"))
model = pickle.load(open("model_svc.pkl", "rb"))
print(model)

color_space = 'YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 9  # HOG orientations
pix_per_cell = 8 # HOG pixels per cell
cell_per_block = 2 # HOG cells per block
hog_channel = 'ALL' # Can be 0, 1, 2, or "ALL"
spatial_size = (16, 16) # Spatial binning dimensions
hist_bins = 16    # Number of histogram bins
spatial_feat = True # Spatial features on or off
hist_feat = True # Histogram features on or off
hog_feat = True # HOG features on or off
heat_len = 6
heat_list = []

# Parameters for window search
ystart = 360
ystop = 680
scale = 1.5


def process_image(image):
	draw_image = np.copy(image)

	hot_windows = find_cars(image, ystart, ystop, scale, model, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)
	#out_img = draw_boxes(draw_image, hot_windows, color=(0, 0, 255), thick=6)

	heat = np.zeros_like(image[:,:,0]).astype(np.float)

	# Add heat to each box in box list
	heat = add_heat(heat, hot_windows)
	heat_list.append(heat)
	
	# Apply threshold to help remove false positives
	totalsum = np.zeros_like(heat)
	if len(heat_list) < 6:
		totalsum = np.sum(heat_list, 0) / float(len(heat_list))
		heat_t = apply_threshold(totalsum, 1)
	else:
		totalsum = np.sum(heat_list[-6:], 0) / 6.0
		#print(totalsum.shape)
		heat_t = apply_threshold(totalsum, 1)

	# Visualize the heatmap when displaying    
	heatmap = np.clip(heat_t, 0, 255)

	# Find final boxes from heatmap using label function
	labels = label(heatmap)
	draw_img = draw_labeled_bboxes(draw_image, labels)

	# Remove comments to print heatmaps
	#if labels[1] > 0:
	#	fig = plt.figure()
	#	plt.subplot(122)
	#	plt.imshow(labels[0], cmap='gray')
	#	plt.title('Heatmap')
	#	plt.subplot(121)
	#	plt.title('Car Positions')
	#	plt.imshow(draw_img)
	#	filename = './output_images/image50' + str(len(heat_list)) + '.png'
	#	plt.savefig(filename)

	return draw_image

video_out = 'project_video_out_svc3.mp4'

clip1 = VideoFileClip("project_video.mp4")
project_clip = clip1.fl_image(process_image) 
project_clip.write_videofile(video_out, audio=False)
