import csv
import cv2
import numpy as np 
from keras.models import Sequential 
from keras.layers import Flatten, Dense, Lambda
from keras.layers import Conv2D, MaxPooling2D, Dropout

lines = []
with open('../data/behavioral/data/driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)
	for line in reader:
		lines.append(line)

images = []
measurements = []
for line in lines:
	source_path = line[0]
	filename = source_path.split('/')[-1]
	current_path = '../data/behavioral/data/IMG/' + filename
	image = cv2.imread(current_path)
	measurement = float(line[3])
	images.append(image)
	measurements.append(measurement)
	image_flipped = np.fliplr(image)
	measurement_flipped = -measurement
	images.append(image_flipped)
	measurements.append(measurement_flipped)

X_train = np.array(images)
y_train = np.array(measurements)
input_shape = (160, 320, 3)
nb_epoch = 3

model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=input_shape))
#model.add(Flatten(input_shape=(160, 320, 3)))

model.add(Conv2D(16, kernel_size=(5, 5),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(16, (5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1))


model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=nb_epoch)

model.save('model.h5')

