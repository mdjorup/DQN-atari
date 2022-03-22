import numpy as np
import tensorflow as tf

def process_single_frame(frame):
    # x should be a single numpy array of shape (210, 160, 3)
    grayscale = tf.image.rgb_to_grayscale(frame)
    resize = tf.image.resize(grayscale, [110, 84])
    cropped = tf.image.crop_to_bounding_box(resize, 17, 0, 84, 84)
    return cropped.numpy() # output shape (84, 84, 1)


class StateProcessor:

    def __init__(self, initial_frame, length=4):
        x = process_single_frame(initial_frame)
        self.state = np.repeat([x], length, axis=0)


    def insert_frame(self, frame):
        x = process_single_frame(frame)
        appended = np.append(self.state, [x], axis=0)
        self.state = np.remove(appended, 0, 0)
    

    def get_state(self):
        return self.state