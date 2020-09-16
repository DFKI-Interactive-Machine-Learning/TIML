# Simple file to print out the devices available from TensorFlow
# From: https://www.tensorflow.org/guide/using_gpu

import tensorflow as tf

# Creates a graph.
a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
c = tf.matmul(a, b)
# Creates a session with log_device_placement set to True.
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
# Runs the op.
print(sess.run(c))

print("==" * 20)

# From https://stackoverflow.com/questions/51306862/how-to-use-tensorflow-gpu

from tensorflow.python.client import device_lib

print(device_lib.list_local_devices())
