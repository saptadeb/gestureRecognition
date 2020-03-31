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
import matplotlib.pyplot as plt



alldata = np.zeros([40,625])
i=0
c=0

def gest(dataa):
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
    data=dataa;

    # Network Parameters
    n_input = data.shape[1]  
    n_steps = data.shape[0]    

    n_hidden = 20  
    n_classes = 4  
    n_layers = 3

    # tf Graph input
    x = tf.placeholder("float", [None, n_steps, n_input])

    # Define weights & biases
    weights = {'out': tf.Variable(tf.random_normal([n_hidden, n_classes]))}
    biases = {'out': tf.Variable(tf.random_normal([n_classes]))}

    pred = RNN(x, weights, biases, n_steps, n_layers, n_hidden, 39)
    otp=tf.argmax(pred,axis=1)
    predSM = tf.nn.softmax(pred)

    saver = tf.train.Saver()

    acc_test=[]

    batch_x_test=np.zeros([1,40,625])

    with tf.Session() as sess:
        saver.restore(sess,"./model.ckpt") ;
       
        for i in range (0,training_iters):
            label_y = []
            data_x = [] ;
                
            if((i%100)==0):
                batch_x_test [0,:,:]= data;
                predsm,otptclss = sess.run([predSM,otp], feed_dict={x: batch_x_test})
                return predsm, otptclss

def lstm_callback(hist):
    global i, alldata, c
    #print "frame", i
    data=hist.data
    shape=hist.layout.dim[0].size
    alldata[i,:]=data
    #print alldata.shape
    i+=1
    if i > 39:
        c+=1
        qt,op = gest(alldata)
        p = np.array([qt[0,0],qt[0,1],qt[0,2],qt[0,3]], dtype=np.float32)
        print "OP ", c, i, " = ", p, op
        bar(p)
        i=0
        tf.reset_default_graph()
        #alldata[:,:]=0
 
def bar(r):
    fig, ax = plt.subplots()
    ind = np.arange(1,5)
    width = 0.3
    colours = ['r','b','g','c']

    p0, p1, p2, p3 = plt.bar(ind, r, color=colours)

    plt.ylabel('Performance Accuracy')
    plt.xticks(ind,("Class 1", "Class 2", "Class 3", "Class 4"))

    plt.show()

if __name__ == '__main__':
    rospy.init_node('lstm', anonymous=True)

    # TODO: Create Subscribers
    arr_sub = rospy.Subscriber("/arr",numpy_msg(Float32MultiArray),lstm_callback)

    # TODO: Spin while node is not shutdown
    while not rospy.is_shutdown():
        rospy.spin()