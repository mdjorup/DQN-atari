# This file contains all of the preprocessing functions for the frames

#input should be a list of n images, and they need the following things to happen to them
#1. input: list of n images, each of shape (210, 160, 3)
#2. convert each image to grayscale - new shape: (210, 160)
#3. scale down each image to (110, 84) 
#4. crop each image to only include the playing area (84, 84)
#5. stack each image together and return - (84, 84, n)

#try possibly vectorized mapping - stacking images first then applying the functions

def convert_grayscale(image):
  pass


def reshape(image, new_shape):
  pass


def crop_to_playing_area(image):
  pass


def phi(images):
  pass
