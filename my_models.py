import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPool2D, Input, Dense,\
BatchNormalization, Flatten, GlobalAveragePooling2D, Dropout, Activation, add, AveragePooling2D

def mnist_model():
    
    my_input = Input(shape=(28,28,1))
    
    x = Conv2D(32, (3,3), activation='relu')(my_input)
    x = MaxPool2D()(x)
    x = BatchNormalization()(x)
    
    x = Conv2D(64, (3,3), activation='relu')(x)
    x = MaxPool2D()(x)
    x = BatchNormalization()(x)
    
    x = Conv2D(128, (3,3), activation='relu')(x)
    x = MaxPool2D()(x)
    x = BatchNormalization()(x)
    
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dense(10, activation='softmax')(x)
    
    model = tf.keras.Model(inputs = my_input, outputs = x)
    
    return model


def cifar10_model():
    my_input = Input(shape=(32,32,3))
    
    x = Conv2D(64, (3,3), activation='relu')(my_input)
    x = BatchNormalization()(x)
    x = Conv2D(64, (3,3), activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPool2D()(x)
    x = Dropout(0.2)(x)
    
    
    x = Conv2D(128, (3,3), activation='relu')(x)
    x = BatchNormalization()(x)
    x = Conv2D(128, (3,3), activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPool2D()(x)
    x = Dropout(0.3)(x)
    
    x = Conv2D(256, (3,3), activation='relu')(x)
    x = BatchNormalization()(x)
    x = Conv2D(256, (3,3), activation='relu')(x)
    x = BatchNormalization()(x)
#     x = MaxPool2D()(x)
    x = Dropout(0.4)(x)
    
#     x = Conv2D(256, (3,3), activation='relu')(x)
#     x = BatchNormalization()(x)
#     x = Conv2D(256, (3,3), activation='relu')(x)
#     x = BatchNormalization()(x)
#     x = MaxPool2D()(x)
#     x = Dropout(0.3)(x)
    
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(10, activation='softmax')(x)
    
    model = tf.keras.Model(inputs = my_input, outputs = x)
    
    return model


def residual_block(inputs, channels, strides=(1, 1)):
    net = BatchNormalization(momentum=0.9, epsilon=1e-5)(inputs)
    net = Activation('relu')(net)

    if strides == (1, 1):
        shortcut = inputs
    else:
        shortcut = Conv2D(channels, (1, 1), strides=strides)(net)

    net = Conv2D(channels, (3, 3), padding='same', strides=strides)(net)
    net = BatchNormalization(momentum=0.9, epsilon=1e-5)(net)
    net = Activation('relu')(net)
    net = Conv2D(channels, (3, 3), padding='same')(net)

    net = add([net, shortcut])
    return net

def bottle_neck_residual_block(inputs, channels, strides=(1, 1)):
    net = BatchNormalization(momentum=0.9, epsilon=1e-5)(inputs)
    net = Activation('relu')(net)

    if strides == (1, 1):
        shortcut = inputs
    else:
        shortcut = Conv2D(channels, (1, 1), strides=strides)(net)

    net = Conv2D(channels, (3, 3), padding='same', strides=strides)(net)
    net = BatchNormalization(momentum=0.9, epsilon=1e-5)(net)
    net = Activation('relu')(net)
    net = Conv2D(channels, (3, 3), padding='same')(net)

    net = add([net, shortcut])
    return net


def ResNet(inputs, stack_n = 18):
    net = Conv2D(16, (3, 3), padding='same')(inputs)

    for i in range(stack_n):
        net = residual_block(net, 16)

    net = residual_block(net, 32, strides=(2, 2))
    for i in range(stack_n - 1):
        net = residual_block(net, 32)

    net = residual_block(net, 64, strides=(2, 2))
    for i in range(stack_n - 1):
        net = residual_block(net, 64)

    net = BatchNormalization(momentum=0.9, epsilon=1e-5)(net)
    net = Activation('relu')(net)
    net = AveragePooling2D(8, 8)(net)
    net = Flatten()(net)
    output = Dense(10, activation='softmax')(net)
#     model = tf.keras.Model(inputs = Input(shape=(32,32,3)), outputs = output)
    return output
    
    