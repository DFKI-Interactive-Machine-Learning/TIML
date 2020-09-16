import os
import argparse

import tensorflow as tf
# To list available GPUs
from tensorflow.python.client import device_lib

from keras.models import Sequential
from keras.layers import Dense


args_parser = argparse.ArgumentParser(
    description='Stress the GPUs through a Keras dummy training.')

args_parser.add_argument('gpu_num', metavar='<gpu_number>', type=str,
                         help="The GPU number (0, 1, 2, ...)")

# This possibly stops the execution if the arguments are not correct
args = args_parser.parse_args()


print("Using GPU {}".format(args.gpu_num))
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_num


print("Listing devices...")
print(device_lib.list_local_devices())

##############################################################################################

# Parameters to control the load on the GPU
num_data_points = 16
NUM_UNITS_PER_LAYER = 4096
num_layers = 20
num_epochs = 200

SAMPLE_SIZE = 4096  # 1024  # 450*450*3
N_OUTPUTS = 8

x_shape = [num_data_points, SAMPLE_SIZE]
y_shape = [num_data_points, N_OUTPUTS]

# Defining the random variables
print("Initializing random input/output...")
X = tf.random.uniform(shape=x_shape)
Y = tf.random.uniform(shape=y_shape, dtype=tf.dtypes.int32, maxval=100)

tf.global_variables_initializer()

# Defining the keras model
print("Defining Keras model...")
model = Sequential()
model.add(Dense(units=NUM_UNITS_PER_LAYER, activation='relu', input_shape=(SAMPLE_SIZE,)))

for i in range(num_layers):
    model.add(Dense(units=NUM_UNITS_PER_LAYER, activation='relu'))

model.add(Dense(units=N_OUTPUTS))


# compiling the keras model
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])


print("Model summary...")
print(model.summary())

while True:
    print("Training...")
    # fitting the Keras model on the random dataset
    model.fit(X, Y, epochs=num_epochs, steps_per_epoch=30)

# print("All done.")
