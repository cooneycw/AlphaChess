import tensorflow as tf
from tensorflow import keras
from keras.regularizers import l1_l2
from keras.layers import Input, Conv2D, BatchNormalization, Activation, Add, Flatten, Dense, Dropout


def residual_block(x, filters, block_idx, l1=0.00001, l2=0.000001, dropout_rate=0.5):
    y = Conv2D(filters, kernel_size=3, padding='same', kernel_regularizer=l1_l2(l1=l1, l2=l2), name=f'res_conv1_block{block_idx}')(x)
    y = BatchNormalization(name=f'res_bn1_block{block_idx}')(y)
    y = Activation('relu', name=f'res_relu1_block{block_idx}')(y)
    y = Dropout(dropout_rate)(y)
    y = Conv2D(filters, kernel_size=3, padding='same', kernel_regularizer=l1_l2(l1=l1, l2=l2), name=f'res_conv2_block{block_idx}')(y)
    y = BatchNormalization(name=f'res_bn2_block{block_idx}')(y)
    y = Add(name=f'res_add_block{block_idx}')([x, y])
    y = Activation('relu', name=f'res_relu2_block{block_idx}')(y)
    return y


# def residual_block(x, filters, block_idx):
#     y = Conv2D(filters, kernel_size=3, padding='same', name=f'res_conv1_block{block_idx}')(x)
#     y = BatchNormalization(name=f'res_bn1_block{block_idx}')(y)
#     y = Activation('relu', name=f'res_relu1_block{block_idx}')(y)
#     y = Conv2D(filters, kernel_size=3, padding='same', name=f'res_conv2_block{block_idx}')(y)
#     y = BatchNormalization(name=f'res_bn2_block{block_idx}')(y)
#     y = Add(name=f'res_add_block{block_idx}')([x, y])
#     y = Activation('relu', name=f'res_relu2_block{block_idx}')(y)
#     return y
#

def create_network(config, l1=0.0001, l2=0.00001, dropout_rate=0.5):
    # Input layer
    inputs = Input(shape=(config.board_size, config.board_size, config.num_channels), name='input')

    # Residual blocks
    x = Conv2D(256, kernel_size=3, padding='same', name='conv1', kernel_regularizer=l1_l2(l1=l1, l2=l2))(inputs)
    x = BatchNormalization(name='bn1')(x)
    x = Activation('relu', name='relu1')(x)
    for i in range(20):
        x = residual_block(x, 256, block_idx=i+1)

    # Value head
    v = Conv2D(1, kernel_size=1, padding='same', name='value_conv', kernel_regularizer=l1_l2(l1=l1, l2=l2))(x)
    v = BatchNormalization(name='value_bn')(v)
    v = Activation('relu', name='value_relu')(v)
    v = Flatten(name='value_flatten')(v)
    v = Dense(256, activation='relu', kernel_initializer='lecun_uniform', name='value_dense1', kernel_regularizer=l1_l2(l1=l1, l2=l2))(v)
    v = Dense(1, activation='tanh', kernel_initializer='lecun_uniform', name='value', kernel_regularizer=l1_l2(l1=l1, l2=l2))(v)

    # Policy head
    p = Conv2D(2, kernel_size=1, padding='same', name='policy_conv', kernel_regularizer=l1_l2(l1=l1, l2=l2))(x)
    p = BatchNormalization(name='policy_bn')(p)
    p = Activation('relu', name='policy_relu')(p)
    p = Flatten(name='policy_flatten')(p)
    p = Dense(config.action_space_size, activation='softmax', name='policy', kernel_regularizer=l1_l2(l1=l1, l2=l2))(p)

    model = tf.keras.Model(inputs=inputs, outputs=[p, v])

    # Initialize the weights of the model
    model.build(input_shape=(None, config.board_size, config.board_size, config.num_channels))
    return model


# Define the neural network
# def create_network(config):
#     # Input layer
#     inputs = Input(shape=(config.board_size, config.board_size, config.num_channels), name='input')
#
#     # Residual blocks
#     x = Conv2D(256, kernel_size=3, padding='same', name='conv1')(inputs)
#     x = BatchNormalization(name='bn1')(x)
#     x = Activation('relu', name='relu1')(x)
#     for i in range(12):
#         x = residual_block(x, 256, block_idx=i+1)
#
#     # Value head
#     v = Conv2D(1, kernel_size=1, padding='same', name='value_conv')(x)
#     v = BatchNormalization(name='value_bn')(v)
#     v = Activation('relu', name='value_relu')(v)
#     v = Flatten(name='value_flatten')(v)
#     v = Dense(256, activation='relu', kernel_initializer='lecun_uniform', name='value_dense1')(v)
#     v = Dense(1, activation='tanh', kernel_initializer='lecun_uniform', name='value')(v)
#
#     # Policy head
#     p = Conv2D(2, kernel_size=1, padding='same', name='policy_conv')(x)
#     p = BatchNormalization(name='policy_bn')(p)
#     p = Activation('relu', name='policy_relu')(p)
#     p = Flatten(name='policy_flatten')(p)
#     p = Dense(config.action_space_size, activation='softmax', name='policy')(p)
#
#     model = tf.keras.Model(inputs=inputs, outputs=[p, v])
#
#     # Initialize the weights of the model
#     model.build(input_shape=(None, config.board_size, config.board_size, config.num_channels))
#     return model
