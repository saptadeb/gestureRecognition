# to prepend (add before) and append (add after) nframes (N number of frames) to four types of data ---
# append (CASE=0) or prepend (CASE=1) zeros, prepend first frame (CASE=2) or append last frame (CASE=3)

# command to run this
# python rnn_lstm_test.py --db all_data.npy --labels all_labels.npy --bs 10 --layers 3 --cells 20 --iters 200 --output 39 --fracTest 0.2 --load ./model.ckpt --nframes 0 --case 0

import time, random
import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
import argparse
import json
import os
from collections import OrderedDict

def RNN(x, weights, biases, n_steps, n_layers, n_hidden, outputFrame):
    x = tf.unstack(x, n_steps, 1)
    multi_lstm_cell = rnn.MultiRNNCell([rnn.BasicLSTMCell(n_hidden, forget_bias=0.0) for _ in range(n_layers)])
    outputs, states = rnn.static_rnn(multi_lstm_cell, x, dtype=tf.float32)
    return tf.matmul(outputs[outputFrame], weights['out']) + biases['out']

parser = argparse.ArgumentParser()
parser.add_argument("--db", help="file with train/test data as npy saved array",type=str)
parser.add_argument("--labels", help="file with train/test labels as npy saved array",type=str)
parser.add_argument("--bs", help="batch_size",type=int)
parser.add_argument("--layers", help="Number of LSTM layers in the memory block", type=int)
parser.add_argument("--cells", help="Number of Cells in each LSTM layer",type=int)
parser.add_argument("--iters", help="number of training iterations",type=int)
parser.add_argument("--output", help="frame for which classif. output is computed",type=int)
parser.add_argument("--fracTest", help="frqction of test samples",type=float, default=0.2) ;
parser.add_argument("--load", help="name of tf checkpoint to load previously trained net state",type=str,default="")
parser.add_argument("--nframes", help="no. of frames to append",type=int,default=0)
parser.add_argument("--case", help="append (CASE=0) or prepend (CASE=1) zeros, prepend first frame (CASE=2) or append last frame (CASE=3)",type=int,default=0)

args = parser.parse_args()

# Training Parameters
learning_rate = 0.0001
training_iters = args.iters
batch_size = args.bs
display_step = 4
f2app = args.nframes
typecas = args.case
alldata = np.load(args.db) ;
alllabels = np.load(args.labels) ;
print "DATA", alldata.shape

# splitting data into train/test 
fracTest = args.fracTest ;
n = alldata.shape[0] ;
sep = int(n*fracTest) ;
indices = range(0,n) ;
random.shuffle(indices) ;
indTest = indices[0:sep] ;
indTrain = indices[sep:] ;
testdata = alldata[indTest] ;
testlabels = alllabels[indTest] ;
data = alldata[indTrain] ;
labels = alllabels[indTrain] ;
print alllabels.min(), alllabels.max();


# Network Parameters
n_input = data.shape[2]          # data is (img feature shape : 625 descriptors * 40 frames)
n_steps = data.shape[1]+f2app    # timesteps
print "n_steps", n_steps
ouputFrame = args.output ;


n_hidden = args.cells  # hidden layer num of features
n_classes = 4   # gesture recognition total classes (1-4 classes)
n_layers = args.layers
time_step_counter=-41

# tf Graph input
x = tf.placeholder("float", [None, n_steps, n_input])
y = tf.placeholder("float", [None, n_classes])

# Define weights & biases
weights = {'out': tf.Variable(tf.random_normal([n_hidden, n_classes]))}
biases = {'out': tf.Variable(tf.random_normal([n_classes]))}

pred = RNN(x, weights, biases, n_steps, n_layers, n_hidden, args.output)

# Defining loss and optimizer (Adam optimizer most preferable...need to check out others)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred,axis=1), tf.argmax(y,axis=1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# 'Saver' op to save and restore all the variables
saver = tf.train.Saver()

#### Training Variables
print"Train data shape is ", data.shape;
print"Test data shape is ", testdata.shape;
acc_test=[]

with tf.Session() as sess:
	# Initializing the variables
    if args.load=="":
        init = tf.global_variables_initializer()
        sess.run(init)
    else:
        saver.restore(sess,args.load) ;
    
    #########################################
    ######          Training Loop      ######
    #########################################
   
    for i in range (0,training_iters):
        if((i%100)==0):
            if (typecas==0): ## to prepend zeros ##
                z = np.zeros([testdata.shape[0], f2app, testdata.shape[2]]) ; 
                batch_x_test = np.append(z, testdata, axis=1);
            elif (typecas==1): ## to append zeros ##
                z = np.zeros([testdata.shape[0], f2app, testdata.shape[2]]) ; 
                batch_x_test = np.append(testdata, z, axis=1);
            elif (typecas==2): ## to prepend first frame ##
                z = np.zeros([testdata.shape[0], f2app, testdata.shape[2]])
                g = testdata[:,0,:];
                for q in range (0, f2app):
                    z[:,q,:] = g;
                batch_x_test = np.append(z, testdata, axis=1);
            elif (typecas==3): ## to append last frame ##
                z = np.zeros([testdata.shape[0], f2app, testdata.shape[2]])
                g = testdata[:,39,:];
                for q in range (0, f2app):
                    z[:,q,:] = g;
                batch_x_test = np.append(testdata, z, axis=1);

            print "shape of test data", batch_x_test.shape;

            batch_y_test = np.zeros([testdata.shape[0],n_classes]) ;
            labels_y = np.ravel(testlabels.astype(np.int32)) ;
            starts = np.arange(0,testdata.shape[0])*n_classes+labels_y ;
            batch_y_test.ravel()[starts] = 1 ;        
            batch_y_test=batch_y_test.reshape(testdata.shape[0],n_classes);
            
            # Calculate batch accuracy
            _pred, acc = sess.run([pred, accuracy], feed_dict={x: batch_x_test, y: batch_y_test})
                    
            # Calculate batch loss
            loss = sess.run(cost, feed_dict={x: batch_x_test, y: batch_y_test})
            print("Iter " + str(i) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc))
            print("##############################################################")