#import numpy as np

#action=input().split()

#action=np.array([float(action[0]),float(action[1])])

#print(action)
import tensorflow as tf
import os


os.environ["CUDA_VISIBLE_DEVICES"]="0"

print(tf.reduce_sum(tf.random.normal([1000, 1000])))
