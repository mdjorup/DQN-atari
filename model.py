from keras import models, layers

def build_model(shape, num_actions):
    model = models.Sequential()
    model.add(layers.Conv2D(16, 8, (4, 4), activation='relu', input_shape=shape))
    model.add(layers.Conv2D(32, 4, (2, 2), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(num_actions))
    return model


# model = build_model((84, 84, 4), 4)
# print(model.summary())