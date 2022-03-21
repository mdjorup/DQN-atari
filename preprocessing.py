# This file contains all of the preprocessing functions for the frames

#input should be a list of n images, and they need the following things to happen to them
#1. input: list of n images, each of shape (210, 160, 3)
#2. convert each image to grayscale - new shape: (210, 160)
#3. scale down each image to (110, 84) 
#4. crop each image to only include the playing area (84, 84)
#5. stack each image together and return - (84, 84, n)

#try possibly vectorized mapping - stacking images first then applying the functions
# tf.vectorized_map()


# import numpy as np

# import tensorflow as tf
# import time 
# test_images = np.zeros((4, 210, 160, 3))


# t0 = time.time()
# converted = tf.image.rgb_to_grayscale(test_images)
# print("Seconds:", time.time()-t0)


# t1 = time.time()
# converted2 = tf.image.rgb_to_grayscale(test_images)
# print("Seconds:", time.time()-t1)

# converted3 = np.resize(converted2, (4, 110, 84, 1))

# print(converted3.shape)

# converted2 = tf.image.rgb_to_grayscale(test_images)



def convert_grayscale(image):
  return tf.image.rgb_to_grayscale(image)


def reshape(image, new_shape):
  pass


def crop_to_playing_area(image):
  pass


def process_single_image(image):
  return crop_to_playing_area(reshape(convert_grayscale(image), (110, 84)))
# if images length is less than the desired amount, then it needs to repeat images


def phi(images, length=4):

  pass
