#! /usr/bin/env python

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



def lstm_callback(hist):
    def RNN(x, weights, biases, n_steps, n_layers, n_hidden, outputFrame):
        x = tf.unstack(x, n_steps, 1)
        multi_lstm_cell = rnn.MultiRNNCell([rnn.BasicLSTMCell(n_hidden, forget_bias=0.0) for _ in range(n_layers)])
        outputs, states = rnn.static_rnn(multi_lstm_cell, x, dtype=tf.float32)
        return tf.matmul(outputs[outputFrame], weights['out']) + biases['out']

    # Training Parameters
    learning_rate = 0.0001
    training_iters = 200
    batch_size = 1
    display_step = 4
    data=hist.data;
    #print data.length

    # Network Parameters
    n_input = 625 
    n_steps = 1   

    n_hidden = 20  
    n_classes = 4   
    n_layers = 3
    time_step_counter=-41

    # tf Graph input
    x = tf.placeholder("float", [None, n_steps, n_input])
    y = tf.placeholder("float", [None, n_classes])

    # Define weights & biases
    weights = {'out': tf.Variable(tf.random_normal([n_hidden, n_classes]))}
    biases = {'out': tf.Variable(tf.random_normal([n_classes]))}

    pred = RNN(x, weights, biases, n_steps, n_layers, n_hidden, 0)
    otp=tf.argmax(pred,axis=1)
    # Defining loss and optimizer (Adam optimizer most preferable...need to check out others)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=pred, labels=y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    # Evaluate model
    correct_pred = tf.equal(tf.argmax(pred,axis=1), tf.argmax(y,axis=1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    # 'Saver' op to save and restore all the variables
    saver = tf.train.Saver()

    # Training Variables
    print"Data shape is ", data.shape;
    acc_test=[]

    with tf.Session() as sess:
        saver.restore(sess,"./model.ckpt") ;
       
        for i in range (0,training_iters):
            label_y = []
            data_x = [] ;
                
            if((i%100)==0):
                batch_x_test = data;
                
                # Calculate batch accuracy
                otptclss = sess.run([otp], feed_dict={x: batch_x_test})
                        
                print("Iter " + str(i) + ", outputclass= " + \
                      "{:.0f}".format(otptclss))
                print("##############################################################")


if __name__ == '__main__':
    rospy.init_node('lstm', anonymous=True)

    # TODO: Create Subscribers
    arr_sub = rospy.Subscriber("/arr",numpy_msg(Float32MultiArray),lstm_callback)

    # # TODO: Create Publishers
    # pcl_objects_pub = rospy.Publisher("/pcl_objects", PointCloud2, queue_size=1)
    # pcl_table_pub   = rospy.Publisher("/pcl_table",   PointCloud2, queue_size=1)
    # pcl_cluster_pub = rospy.Publisher("/pcl_cluster", PointCloud2, queue_size=1)

    # object_markers_pub   = rospy.Publisher("/object_markers", Marker, queue_size=1)
    # detected_objects_pub = rospy.Publisher("/detected_objects", DetectedObjectsArray, queue_size=1)

    # # Initialize color_list
    # get_color_list.color_list = []

    # TODO: Spin while node is not shutdown
    while not rospy.is_shutdown():
        rospy.spin()