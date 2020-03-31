#! /usr/bin/env python

""" RNN creation and loading to class
class is instant. in main()
just call sess.run(predSM) in callback
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


# in calllback: output dimension of received data
# copy received data (numpy array) to global variable and set a flag
# execute should only be done in main loop when flag is set
# message should be published in main()

alldata = np.zeros([1,625])            #defining the array which will be passed for classfctn
i=0
c=0
g = tf.Graph()

class getconfigs(object):
    learning_rate = 0.0001
    batch_size = 1
    display_step = 4
    n_steps = 1            #1 No. of frames
    n_input = 625        #625 No. of descriptors
    n_hidden = 20  
    n_classes = 4  
    n_layers = 3

class initGraph(object):
    def __init__(self, config):

            #self.weights = {'out': tf.Variable(tf.random_normal([config.n_hidden, config.n_classes]))}
            #self.biases = {'out': tf.Variable(tf.random_normal([config.n_classes]))}
            x = tf.placeholder(tf.float32, [1, 1, 625])
            x = tf.unstack(x,config.n_steps, 1)
            self.x = x;
            self.weights = {'out': tf.Variable(tf.random_normal([config.n_hidden, config.n_classes]))}
            self.biases = {'out': tf.Variable(tf.random_normal([config.n_classes]))}    

            cell = rnn.BasicLSTMCell(config.n_hidden, forget_bias=0.0, reuse=tf.AUTO_REUSE)
            self.multi_lstm_cell = rnn.MultiRNNCell([cell for _ in range(config.n_layers)]) ##creates the graph
            self.initial_state = states = cell.zero_state(config.batch_size, tf.float32)
            output = []
            self.outputs, self.states = rnn.static_rnn(self.multi_lstm_cell, x, dtype=tf.float32)
            tf.global_variables_initializer().run() ;
            outputFrame = 0;
            self.pred = (tf.matmul(outputs[outputFrame], gph.weights['out']) + gph.biases['out'])
            self.otp = tf.argmax(pred,axis=1)
            self.predSM = tf.nn.softmax(pred)
        


    def execute(x,config, outputFrame):
        return self.pred.run(feed_dict={self.x:x}) ;



def lstm_callback(hist):              #after this callback is finished it returns to the subscriber & sends another frame
    # check whether we xan accept or skip this message
    data=hist.data
    shape=hist.layout.dim[0].size            #625
    alldata[0,:]=data                         
    batch_x_test = np.zeros([1,1,625])                 #changed size of frames from 40 to 1
    batch_x_test[0,:,:] = alldata
    x=batch_x_test
    x = np.float32(x)
    #i+=1
    kb=None
    #x = tf.placeholder("float", [None, config.n_steps, config.n_input])
    print x.dtype, "callback",model.weights.items(), model.biases.items()
    result = model.execute(x) ;
    print result ;

    #qt, op = sess.run([predSM,otp], feed_dict={x: batch_x_test})
    #p = np.array([qt[0,0],qt[0,1],qt[0,2],qt[0,3]], dtype=np.float32)
    #print "OP ", i, " = ", p, op+1
    print "Gesture Class -- ", op+1
    '''if op==[0]:
        kb='i'
        print "class one", kb
    elif op==[1]:
        kb=','
        print "class two", kb
    '''
    #i=0
    pubb.publish(kb)



 
if __name__ == '__main__':
    rospy.init_node('lstm', anonymous=True)

    global config, model, x


    pubb = rospy.Publisher('/keybind', String, queue_size=1)

    with tf.Session() as sess:
        config = getconfigs()
        model = initGraph(config=config)
        saver = tf.train.Saver()
        saver.restore(sess,"./model.ckpt")
        arr_sub = rospy.Subscriber('/arr',numpy_msg(Float32MultiArray),lstm_callback)


        while not rospy.is_shutdown():
          rospy.spinOnce()

          # check flag
          # if set: process received hist, block callback, call modelexecute()
