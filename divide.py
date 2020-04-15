#!/usr/bin/env python3
import argparse
import random
import os
from shutil import copyfile

# Input args
ap = argparse.ArgumentParser ()
ap.add_argument ("-t", "--test", type = int, choices = range (1, 100), default = 20, required = False,
	help = "Percent of test images")
ap.add_argument ("-i", "--input", type = str, required = True,
	help = "Directory for input images")
ap.add_argument ("-o", "--output", type = str, required = True,
	help = "Output directory")
args = vars (ap.parse_args ())

# Args parse
# Percent of test images
test = args["test"]
# Input directory
input = args["input"]
# Output directory
output = args["output"]
if not os.path.exists (os.path.join (output, "train", input)):
	os.makedirs (os.path.join (output, "train", input))
if not os.path.exists (os.path.join (output, "test", input)):
	os.makedirs (os.path.join (output, "test", input))

# Returns the number of test images by percentage
def percentage (percent, whole):
  return int ((percent * whole) / 100.0)

# Read only images in directory
def read_images_in_dir (path):
	images = []
	valid_images = [".jpg", ".jpeg", ".png"]
	for f in os.listdir (path):
		ext = os.path.splitext (f)[1]
		if ext.lower () in valid_images:
			images.append (os.path.join (path, f))

	return images

# Divide the images into train and test
def divide ():
	# List images in directory
	images = read_images_in_dir (input)

	# Count of images for test
	test_count = percentage (test, len (images))

	# If images for test is lower than one set one
	if test_count < 1:
		test_count = 1

	# Random copy test images
	for i in range (test_count):
		item = images.pop (random.randrange (len (images)))
		print (item)
		copyfile (item, os.path.join (output, "test", item))

	# Copy train images
	for item in images:
		print (item)
		copyfile (item, os.path.join (output, "train", item))

divide ()
