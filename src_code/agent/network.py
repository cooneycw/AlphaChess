import tensorflow as tf
from tensorflow import keras
from keras.regularizers import l1_l2
from keras.layers import Input, Conv2D, BatchNormalization, Activation, Add, Flatten, Dense, Dropout


def leaky_relu(x, alpha=0.2):
    return tf.maximum(x, alpha * x)


def residual_block(x, filters, block_idx, dropout_rate):
    y = Conv2D(filters, kernel_size=3, padding='same', name=f'res_conv1_block{block_idx}')(x)
    y = BatchNormalization(name=f'res_bn1_block{block_idx}')(y)
    y = Activation(leaky_relu, name=f'res_leakyrelu1_block{block_idx}')(y)
    y = Dropout(dropout_rate)(y)
    y = Conv2D(filters, kernel_size=3, padding='same', name=f'res_conv2_block{block_idx}')(y)
    y = BatchNormalization(name=f'res_bn2_block{block_idx}')(y)
    y = Add(name=f'res_add_block{block_idx}')([x, y])
    y = Activation(leaky_relu, name=f'res_leakyrelu2_block{block_idx}')(y)
    return y


def create_network(config, network_type, dropout_rate=0.5):
    # Input layer
    inputs = Input(shape=(config.board_size, config.board_size, config.num_channels), name='input')

    # Residual blocks
    x = Conv2D(256, kernel_size=3, padding='same', name='conv1')(inputs)
    x = BatchNormalization(name='bn1')(x)
    x = Activation(leaky_relu, name='leakyrelu1')(x)
    for i in range(6):
        blk_idx = i + 1
        x = residual_block(x, 256, blk_idx, dropout_rate)

    # Value head
    if network_type == 'value':
        v = Conv2D(1, kernel_size=1, padding='same', name='value_conv')(x)
        v = BatchNormalization(name='value_bn')(v)
        v = Activation(leaky_relu, name='value_leakyrelu')(v)
        v = Flatten(name='value_flatten')(v)
        v = Dense(256, activation='relu', kernel_initializer='glorot_uniform', name='value_dense1')(v)
        v = Dense(1, activation='tanh', kernel_initializer='glorot_uniform', name='value_dense2')(v)
        model = tf.keras.Model(inputs=inputs, outputs=v)

    elif network_type == 'policy':
        # Policy head
        p = Conv2D(2, kernel_size=1, padding='same', name='policy_conv')(x)
        p = BatchNormalization(name='policy_bn')(p)
        p = Activation(leaky_relu, name='policy_leakyrelu')(p)
        p = Flatten(name='policy_flatten')(p)
        p = Dense(config.action_space_size, kernel_initializer='glorot_uniform', activation='softmax', name='policy_dense')(p)
        model = tf.keras.Model(inputs=inputs, outputs=p)

    # Initialize the weights of the model
    model.build(input_shape=(None, config.board_size, config.board_size, config.num_channels))
    return model
