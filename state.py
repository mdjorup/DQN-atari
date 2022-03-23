import numpy as np
import tensorflow as tf

def process_frames(frames):
    # x should be a single numpy array of shape (len, 210, 160, 3)
    grayscale = tf.image.rgb_to_grayscale(frames)
    resize = tf.image.resize(grayscale, [110, 84])
    cropped = tf.image.crop_to_bounding_box(resize, 17, 0, 84, 84)
    reshaped = np.reshape(cropped, (-1, 84, 84))
    stacked = np.stack(reshaped, axis=2)
    return stacked


class StateProcessor:

    # state initialization takes ~ 4ms
    def __init__(self, initial_frame, length=4):
        initial_frame = np.reshape(initial_frame, (-1, 210, 160, 3))
        frames = np.repeat(initial_frame, 4, axis=0)
        self.state = process_frames(frames)
        self.length = length


    def insert_frames(self, frames):
        initial_frames = np.reshape(frames, (-1, 210, 160, 3))
        processed = process_frames(initial_frames)
        self.state = processed
            

    def get_state(self):
        return self.state