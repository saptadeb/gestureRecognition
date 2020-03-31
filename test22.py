#! /usr/bin/env python

""" RNN creation and loading to class
class is instant. in main()
just call osess.run(predSM) in callback
find out how to clear RNN activities after a determined period (''state'')
create 4 RNN instances
"""
import roslib
import rospy
import sys, time
import time, random
import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
import argparse
import json
import os
from collections import OrderedDict
from std_msgs.msg import Float32MultiArray
from std_msgs.msg import MultiArrayDimension
from rospy.numpy_msg import numpy_msg
import matplotlib.pyplot as plt
from std_msgs.msg import String

alldata = np.zeros([1,625])                    #defining the array which will be passed for classfctn
i=0
c=0

class dasdas(object):
    """docstring for ClassName"""
    def __init__(self, gg):
        print gg.batch_size
        

class initGraph(object):
        # Training Parameters
        learning_rate = 0.0001
        testing_iters = 200
        batch_size = 1
        display_step = 4

        # Network Parameters
        n_steps = 1  #1 No. of frames
        n_input = 625  #625 No. of descriptors

        n_hidden = 20  
        n_classes = 4  
        n_layers = 3

        # tf Graph input
        x = tf.placeholder("float", [None, n_steps, n_input])

        # Define weights & biases
        weights = {'out': tf.Variable(tf.random_normal([n_hidden, n_classes]))}
        biases = {'out': tf.Variable(tf.random_normal([n_classes]))}        

if __name__ == '__main__':

    qqq = initGraph()
    dasdas(gg=qqq)
    #print(qqq.n_hidden)