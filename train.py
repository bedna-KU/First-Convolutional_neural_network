from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from keras.optimizers import Adam
from keras.losses import categorical_crossentropy
from keras.callbacks import ModelCheckpoint
from keras.utils.vis_utils import plot_model
from keras.preprocessing.image import ImageDataGenerator
from termcolor import cprint
import os
import argparse
import json

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

###################################################
# PARAMETERS
###################################################

IMG_WIDTH = 50
IMG_HEIGHT = 50
IMG_COLOR_DEPTH = 1
INPUT_SHAPE = IMG_HEIGHT, IMG_WIDTH, IMG_COLOR_DEPTH
NUM_CLASSES = 2
BATCH_SIZE = 32
EPOCHS = 20
WEIGHTPATH = "weights.hdf5"

###################################################

# Set input args
ap = argparse.ArgumentParser ()
ap.add_argument ("-i", "--input", type = str, required = True,
	help = "directory to the dataset")
args = vars (ap.parse_args ())

# Input directory
input = args["input"]

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

def load_dataset ():
	cprint ("Load dataset", 'yellow', attrs = ['bold'])
	datagen = ImageDataGenerator (rescale = 1. / 255)

	cprint ("Train", 'yellow', attrs = ['bold'])
	# Load training dataset
	train_data = datagen.flow_from_directory ('data/train/',
											target_size = (IMG_HEIGHT, IMG_WIDTH),
											color_mode = 'grayscale',
											class_mode = 'categorical',
											batch_size = BATCH_SIZE)

	cprint ("Test", 'yellow', attrs = ['bold'])
	# Load test dataset
	test_data = datagen.flow_from_directory ('data/test/',
										  target_size = (IMG_HEIGHT, IMG_WIDTH),
										  color_mode = 'grayscale',
										  class_mode = 'categorical',
										  batch_size = BATCH_SIZE)

	# confirm the iterator works
	batchX, batchy = train_data.next()
	print ('Batch shape=%s, min=%.3f, max=%.3f' % (batchX.shape, batchX.min(), batchX.max()))

	classes = train_data.class_indices
	cprint (train_data.class_indices, 'yellow', attrs = ['bold'])
	json.dump (classes, open ("classes.json", 'w'))

	return train_data, test_data

def train_model ():
	cprint ("Start learning", 'yellow', attrs = ['bold'])
	tst = model.fit_generator (train_data,
                    validation_data = test_data,
                    steps_per_epoch = train_data.n // BATCH_SIZE,
                    validation_steps = (test_data.n // BATCH_SIZE),
                    epochs = EPOCHS,
                    callbacks = callbacks)

checkpoint = ModelCheckpoint (WEIGHTPATH,
							  monitor = 'loss',
							  verbose = 1,
							  save_best_only = True,
							  save_weights_only = True,
							  mode = 'min')

callbacks = [checkpoint]

model = create_model (INPUT_SHAPE, NUM_CLASSES)
model.summary ()
plot_model (model, to_file = 'model_plot.png', show_shapes = True, show_layer_names = True)
train_data, test_data = load_dataset ()
train_model ()
