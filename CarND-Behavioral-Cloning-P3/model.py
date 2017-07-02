import csv
import cv2
import numpy as np 
from keras.models import Sequential 
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers import Conv2D, MaxPooling2D, Dropout
import sklearn


lines = []
with open('../data/behavioral2/driving_log2.csv') as csvfile:
	reader = csv.reader(csvfile)
	for line in reader:
		lines.append(line)

car_images = []
steering_angles = []

for line in lines:
	steering_center = float(line[3])

	# create adjusted steering measurements for the side camera images
	correction = 0.29 # this is a parameter to tune
	steering_left = steering_center + correction
	steering_right = steering_center - correction
	
	img_center = cv2.imread(line[0])
	img_left = cv2.imread(line[1])
	img_right = cv2.imread(line[2])

	car_images.extend([img_center, img_left, img_right])
	steering_angles.extend([steering_center, steering_left, steering_right])



X_train = np.array(car_images)
y_train = np.array(steering_angles)

input_shape = (160, 320, 3)
nb_epoch = 5

# Define the CNN in keras
model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=input_shape))
model.add(Cropping2D(cropping=((50,30), (0,0)), input_shape=input_shape))

# Convolutional layer 1
model.add(Conv2D(16, kernel_size=(5, 5), activation='relu', input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Convolutional layer 2
model.add(Conv2D(16, (5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Convolutional layer 1
model.add(Conv2D(16, (5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.75))

# Fully connected layer
model.add(Flatten())
model.add(Dense(100))
model.add(Dropout(0.5))

# Output
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=nb_epoch, batch_size=128)
model.save('model.h5')