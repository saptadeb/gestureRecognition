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

def gest(dataa):
    def RNN(x, weights, biases, n_steps, n_layers, n_hidden, outputFrame):
        x = tf.unstack(x, n_steps, 1)
        multi_lstm_cell = rnn.MultiRNNCell([rnn.BasicLSTMCell(n_hidden, forget_bias=0.0, reuse=tf.AUTO_REUSE) for _ in range(n_layers)])
        outputs, states = rnn.static_rnn(multi_lstm_cell, x, dtype=tf.float32)
        print len(weights), len(biases)
        return tf.matmul(outputs[outputFrame], weights['out']) + biases['out']

    # Training Parameters
    learning_rate = 0.0001
    testing_iters = 200
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

    pred = RNN(x, weights, biases, n_steps, n_layers, n_hidden, 0)    #changing the outputFrame to 0
    otp=tf.argmax(pred,axis=1)
    predSM = tf.nn.softmax(pred)

    saver = tf.train.Saver()

    batch_x_test=np.zeros([1,1,625])                     #changed size of frames from 40 to 1

    with tf.Session() as sess:
        saver.restore(sess,"./model.ckpt") ;
       
        for ti in range (0,testing_iters):                
            if((ti%100)==0):
                batch_x_test [0,:,:]= data;
                predsm,otptclss = sess.run([predSM,otp], feed_dict={x: batch_x_test})
                return predsm, otptclss

def lstm_callback(hist):                     #after this callback is finished it returns to the subscriber & sends another frame
    global i, alldata, c
    #print "frame", i
    data=hist.data
    shape=hist.layout.dim[0].size            #625
    alldata[0,:]=data                        #removed i (from the orgnl code), alldata_shape = [1,625]
    #print alldata.shape
    i+=1
    kb=None
    qt,op = gest(alldata)
    p = np.array([qt[0,0],qt[0,1],qt[0,2],qt[0,3]], dtype=np.float32)
    #print "OP ", i, " = ", p, op+1
    print "Gesture Class -- ", op+1
    '''if op==[0]:
        kb='i'
        print "class one", kb
    elif op==[1]:
        kb=','
        print "class two", kb
    '''
    i=0
    pubb.publish(kb)
    tf.reset_default_graph()
    #alldata[:,:]=0
 
if __name__ == '__main__':
    rospy.init_node('lstm', anonymous=True)

    # TODO: Create Subscribers
    #print "1"
    arr_sub = rospy.Subscriber('/arr',numpy_msg(Float32MultiArray),lstm_callback)
    #print "2"
    pubb = rospy.Publisher('/keybind', String, queue_size=1)
    #print "3"
    # TODO: Spin while node is not shutdown
    while not rospy.is_shutdown():
        #print "in"
        rospy.spin()
    #print "4"