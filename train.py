import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input
from tqdm import tqdm
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from data_aug import color_normalize, images_augment, display_some_examples
from my_models import cifar10_model, residual_block, ResNet

# training params
batch_size = 128
train_num = 45000
iterations_per_epoch = int(train_num / batch_size)

# test config
test_batch_size = 200
test_num = 5000
test_iterations = int(test_num / test_batch_size)


weight_decay = 1e-4

def accuracy(y_true, y_pred):
    correct_num = tf.equal(tf.argmax(y_true, -1), tf.argmax(y_pred, -1))
    accuracy = tf.reduce_mean(tf.cast(correct_num, dtype=tf.float32))
    return accuracy

def cross_entropy(y_true, y_pred):
    cross_entropy = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
    return tf.reduce_mean(cross_entropy)

def l2_loss(model, weights=weight_decay):
    variable_list = []
    for v in model.trainable_variables:
        if 'kernel' in v.name:
            variable_list.append(tf.nn.l2_loss(v))
    return tf.add_n(variable_list) * weights


@tf.function
def train_step(model, optimizer, x, y):
    with tf.GradientTape() as tape:
        prediction = model(x, training=True)
        ce = cross_entropy(y, prediction)
        l2 = l2_loss(model)
        loss = ce + l2
        gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return ce, prediction

@tf.function
def test_step(model, x, y):
    prediction = model(x, training=False)
    ce = cross_entropy(y, prediction)
    return ce, prediction

def train(model, optimizer, images, labels):
    sum_loss = 0
    sum_accuracy = 0

    # random shuffle
    seed = np.random.randint(0, 65536)
    np.random.seed(seed)
    np.random.shuffle(images)
    np.random.seed(seed)
    np.random.shuffle(labels)

    for i in tqdm(range(iterations_per_epoch)):
        x = images[i * batch_size: (i + 1) * batch_size, :, :, :]
        y = labels[i * batch_size: (i + 1) * batch_size, :]
        x = images_augment(x)

        loss, prediction = train_step(model, optimizer, x, y)
        sum_loss += loss
        sum_accuracy += accuracy(y, prediction)

    print('ce_loss:%f, l2_loss:%f, accuracy:%f' %
          (sum_loss / iterations_per_epoch, l2_loss(model), sum_accuracy / iterations_per_epoch))

def test(model, images, labels):
    sum_loss = 0
    sum_accuracy = 0

    for i in tqdm(range(test_iterations)):
        x = images[i * test_batch_size: (i + 1) * test_batch_size, :, :, :]
        y = labels[i * test_batch_size: (i + 1) * test_batch_size, :]

        loss, prediction = test_step(model, x, y)
        sum_loss += loss
        sum_accuracy += accuracy(y, prediction)

    print('test, loss:%f, accuracy:%f' %
          (sum_loss / test_iterations, sum_accuracy / test_iterations))

# (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
# x_train = x_train.astype("float32") / 255
# x_test = x_test.astype("float32") / 255
# x_train, x_test = color_normalize(x_train, x_test)
# x_train = images_augment(x_train)
# x_test = images_augment(x_test)
# # x_train = np.expand_dims(x_train, axis=-1)
# # x_test = np.expand_dims(x_test, axis=-1)

# y_train = tf.keras.utils.to_categorical(y_train, 10)
# y_test = tf.keras.utils.to_categorical(y_test, 10)

# print('x_train shape:', x_train.shape)
# print('y_train shape:', y_train.shape)
# print('x_test shape:', x_test.shape)
# print('y_test shape:', y_test.shape)

# # output = resnet(inputs = Input(shape=(32,32,3)))
# # model = tf.keras.Model(inputs = Input(shape=(32,32,3)), outputs = output)
# # model = resnet(Input(shape=(32,32,3)))
# img_input = Input(shape=(32, 32, 3))
# output = ResNet(inputs = img_input, stack_n = 56)
# model = tf.keras.Model(img_input, output)
# model.summary()


# adam = tf.optimizers.Adam(learning_rate=5e-3, amsgrad='False')
# batch_size = 128
# train_num = x_train.shape[0]
# iterations_per_epoch = int(train_num / batch_size)
# learning_rate = [0.01, 0.001, 0.0001]
# boundaries = [80 * iterations_per_epoch, 120 * iterations_per_epoch]
# learning_rate_schedules = tf.optimizers.schedules.PiecewiseConstantDecay(boundaries, learning_rate)
# optimizer = tf.optimizers.SGD(learning_rate=learning_rate_schedules, momentum=0.9, nesterov=True)
# model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics='accuracy')


# # In[ ]:


# # history = model.fit(x_train, y_train, batch_size=128, epochs=50, validation_split=0.2)
# # Data Augmentation
# if True:
#     x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size = 0.1)
#     datagen = ImageDataGenerator(
# #         rescale=1./255,
# #         rotation_range=20,
#         width_shift_range=0.2,
#         height_shift_range=0.2,
# #         channel_shift_range=50,
#         horizontal_flip=True)
#     testgen = ImageDataGenerator()
# #         rescale=1./255)

#     # フィット
#     datagen.fit(x_train)
#     testgen.fit(x_test)
#     print("x_train.shape= ", x_train.shape)
#     print("x_test.shape= ", x_test.shape)
#     history = model.fit(datagen.flow(x_train, y_train, batch_size=batch_size),
#                         steps_per_epoch=len(x_train) / batch_size,
#                         validation_data=testgen.flow(x_val, y_val), epochs=200).history

#     # with open("history.dat", "wb") as fp:
#     #     pickle.dump(history, fp)


# # In[ ]:


# # evaluate on test data, return [loss, metrics]
# model.evaluate(x_test, y_test, batch_size = 128)


# # In[ ]:


# # return predicted lables
# for i in range(10):
#     rand_idx = np.random.randint(y_test.shape[0])
#     y_pred = model.predict(x_test[rand_idx:rand_idx+1,:,:,:])
#     print(rand_idx)
#     print("True label: ", np.argmax(y_test[rand_idx:rand_idx+1,:], axis=1))
#     print("predicted label: ", np.argmax(y_pred, axis=-1))


# # In[ ]:





# # In[ ]:





# # In[ ]:




