import tensorflow as tf
import os

os.environ["CUDA_VISIBLE_DEVICES"]="0"

print(tf.reduce_sum(tf.random.normal([1000, 1000])))