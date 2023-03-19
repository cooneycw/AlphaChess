import tensorflow as tf
from tensorflow import keras
from keras.layers import Input, Conv2D, BatchNormalization, Activation, Add, Flatten, Dense


def residual_block(x, filters):
    y = Conv2D(filters, kernel_size=3, padding='same')(x)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    y = Conv2D(filters, kernel_size=3, padding='same')(y)
    y = BatchNormalization()(y)
    y = Add()([x, y])
    y = Activation('relu')(y)
    return y


# Define the neural network
def create_network(config):
    # Input layer
    inputs = Input(shape=(config.board_size, config.board_size, config.num_channels))

    # Residual blocks
    x = Conv2D(256, kernel_size=3, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    for i in range(4):
        x = residual_block(x, 256)

    # Value head
    v = Conv2D(1, kernel_size=1, padding='same')(x)
    v = BatchNormalization()(v)
    v = Activation('relu')(v)
    v = Flatten()(v)
    v = Dense(256, activation='relu')(v)
    v = Dense(1, activation='tanh', name='value')(v)

    # Policy head
    p = Conv2D(2, kernel_size=1, padding='same')(x)
    p = BatchNormalization()(p)
    p = Activation('relu')(p)
    p = Flatten()(p)
    p = Dense(config.action_space_size, activation='softmax', name='policy')(p)

    model = tf.keras.Model(inputs=inputs, outputs=[p, v])

    # Initialize the weights of the model
    model.build(input_shape=(None, config.board_size, config.board_size, config.num_channels))
    return model
