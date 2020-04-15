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
import mss
from screeninfo import get_monitors

# Suppresses messages from TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

####################################################
# PARAMETERS
####################################################

IMG_WIDTH = 50
IMG_HEIGHT = 50
IMG_COLOR_DEPTH = 1
INPUT_SHAPE = IMG_HEIGHT, IMG_WIDTH, IMG_COLOR_DEPTH
NUM_CLASSES = 2
BATCH_SIZE = 1
WEIGHTPATH = "weights.hdf5"

####################################################

desk_width = get_monitors ()[0].width
desk_height = get_monitors ()[0].height

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
	# ~ model.add (Conv2D (128, kernel_size = (3, 3), activation='relu'))
	# ~ model.add (MaxPooling2D (pool_size = (2, 2)))
	# ~ model.add (Conv2D (256, kernel_size = (3, 3), activation='relu'))
	# ~ model.add (MaxPooling2D (pool_size = (2, 2)))
	model.add (Flatten ())
	model.add (Dense (512, activation = 'relu'))
	# ~ model.add (Dense (NUM_CLASSES, activation = 'sigmoid'))
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

# Predict class
def predict (image):
	img_in = cv2.resize (image, dsize = (IMG_HEIGHT, IMG_WIDTH), interpolation = cv2.INTER_CUBIC)
	img_in = img_in.astype ('float') / 255
	img_in = np.expand_dims (img_in, axis = 0)
	img_in = np.expand_dims (img_in, axis = 3)
	result = model.predict (img_in, batch_size=BATCH_SIZE, verbose=0)
	print (result)
	class_name = classes[np.argmax (result)]
	return class_name

# Capture and draw desktop
def scan_desktop ():
	# Set screen capture
	shift = 115
	monitor = {"top": shift, "left": 0, "width": desk_width // 2, "height": desk_height - shift}

	# Set scan
	startx = 0
	starty = 0
	cropx = 159
	cropy = 150

	# Set preview window
	window_name = "Preview"
	cv2.namedWindow (window_name)
	cv2.moveWindow (window_name, desk_width // 2, 30)

	# Capture screen
	with mss.mss () as sct:
		while "Screen capturing":
			# Get raw pixels from the screen, save it to a Numpy array
			img = np.array (sct.grab (monitor))
			# Yaxis scan
			while starty + cropy < desk_height:
				# Xaxis scan
				while startx + cropx < desk_width // 2:
					crop_img = img[starty : starty + cropy, startx : startx + cropx]
					crop_img = cv2.cvtColor (crop_img, cv2.COLOR_RGB2GRAY)
					result = predict (crop_img)
					print (">>> result", result)
					if result != "mom":
						img = cv2.rectangle (img, (startx, starty), (startx + cropx, starty + cropy), (255, 0, 0), -1)
					# Display the picture
					cv2.imshow (window_name, img)
					# Press "q" to quit
					if cv2.waitKey (25) & 0xFF == ord ("q"):
						cv2.destroyAllWindows ()
						exit ("END")
					startx += cropx
				startx = 0
				starty += cropy
			starty = 0

classes = read_classes ()
model = create_model (INPUT_SHAPE, NUM_CLASSES)
load_weights ()
# ~ image = load_image ()
# ~ result = predict (image)
# ~ print (result)
scan_desktop ()

