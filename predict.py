from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.optimizers import Adam
from keras.losses import categorical_crossentropy
import cv2
import numpy as np
import sys
import os
import json

# Suppresses messages from TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

###################################################
# PARAMETERS
###################################################

IMG_WIDTH = 50
IMG_HEIGHT = 50
IMG_COLOR_DEPTH = 1
INPUT_SHAPE = IMG_HEIGHT, IMG_WIDTH, IMG_COLOR_DEPTH
NUM_CLASSES = 2
BATCH_SIZE = 1
EPOCHS = 10
WEIGHTPATH = "weights.hdf5"

###################################################

# Read JSON file with classes
def read_classes ():
	with open ('classes.json') as json_file:
		classes = json.load (json_file)
		keys = list (classes.keys ())
	return keys

# Create 7layers model
def create_model (input_shape, num_classes):
	model = Sequential ()
	model.add (Conv2D (32, kernel_size = (3, 3), strides = (1, 1), activation = 'relu', input_shape = INPUT_SHAPE))
	model.add (MaxPooling2D (pool_size = (2, 2)))
	model.add (Conv2D (64, kernel_size = (3, 3), activation = 'relu'))
	model.add (MaxPooling2D (pool_size = (2, 2)))
	model.add (Flatten ())
	model.add (Dense (512, activation = 'relu'))
	model.add (Dense (NUM_CLASSES, activation = 'softmax'))

	model.compile (loss = categorical_crossentropy,
		   optimizer = Adam (),
		   metrics = ['accuracy'])
	return model

# Load weights to model
def load_weights ():
	if os.path.exists (WEIGHTPATH):
		model.load_weights (WEIGHTPATH)
	else:
		exit (">>> File with weights no exist!")

# Load image for predict
def load_image ():
	img_name = sys.argv[1]
	img = cv2.imread (img_name, cv2.IMREAD_GRAYSCALE)
	return img

# Predict class
def predict (image):
	img_in = cv2.resize (image, dsize = (IMG_HEIGHT, IMG_WIDTH), interpolation = cv2.INTER_CUBIC)
	img_in = img_in.astype ('float') / 255
	img_in = np.expand_dims (img_in, axis = 0)
	img_in = np.expand_dims (img_in, axis = 3)
	result = model.predict (img_in, batch_size = BATCH_SIZE, verbose = 1)
	class_name = classes[np.argmax (result)]
	return class_name

# Draw loaded image
def draw_image (image):
	window_name = "Test"
	cv2.namedWindow (window_name)
	cv2.moveWindow (window_name, 0, 30)
	ratio = image.shape[1] / image.shape[0]
	image = cv2.resize (image, (int (500 * ratio), 500))
	cv2.imshow (window_name, image)
	cv2.waitKey (0)

classes = read_classes ()
model = create_model (INPUT_SHAPE, NUM_CLASSES)
load_weights ()
image = load_image ()
result = predict (image)
print (result)
draw_image (image)
