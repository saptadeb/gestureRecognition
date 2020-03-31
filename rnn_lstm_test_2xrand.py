# experiment is for the query --- 
# is classification dependent on what is classified before? for each test sample, prepend another 
# randomly chosen test sample and let the net run on the extended sample of 2X frames. Evaluating at the end of sample 1 (frameX) 
# and again at the end of sample 2 (frame 2X). Is classification on sample 2 as good as the classification on normal test set?

# random dataset of 40 frames prepended before the sample of 40 frames (in the testing phase), making it a 80 frame sample.
# classification done in two phases, first after the 40 frames, and after 80 frames to calculate the accuracy  

# command to run this
# python rnn_lstm_test.py --db all_data.npy --labels all_labels.npy --bs 10 --layers 3 --cells 20 --iters 200 --output 39 --fracTest 0.2 --load ./model.ckpt

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
parser.add_argument("--bs", help="batch_size", type = int)
parser.add_argument("--layers", help="Number of LSTM layers in the memory block", type = int)
parser.add_argument("--cells", help="Number of Cells in each LSTM layer", type= int)
parser.add_argument("--iters", help="number of training iterations",type=int)
parser.add_argument("--output", help="frame for which classif. output is computed",type=int)
parser.add_argument("--fracTest", help="frqction of test samples",type=float, default=0.2)
parser.add_argument("--load", help="name of tf checkpoint to load previously trained net state",type=str,default = "")

args = parser.parse_args()

# Training Parameters
learning_rate = 0.0001
training_iters = args.iters
batch_size = args.bs
display_step = 4
alldata=np.load(args.db) 
alllabels=np.load(args.labels) 
print "DATA", alldata.shape

# splitting data into train/test 
fracTest=args.fracTest 
n = alldata.shape[0] 
sep = int(n*fracTest) 
indices=range(0,n) 
random.shuffle(indices) 
indTest=indices[0:sep] 
indTest2=indices[sep:sep*2] 
testdata=alldata[indTest] 
testlabels=alllabels[indTest] 
testdata2=alldata[indTest2] 
testlabels2=alllabels[indTest2] 
# print alllabels.min(), alllabels.max()


# Network Parameters
n_input = testdata.shape[2]   # data is (img feature shape : 625 descriptors * 40 frames)
# n_steps = data.shape[1]     #### UNCOMMENT for the first phase ####
# n_steps = 80                #### UNCOMMENT for the second phase ###
ouputFrame = args.output 
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
#print pred.shape

# Defining loss and optimizer (Adam optimizer most preferable...need to check out others)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred,axis=1), tf.argmax(y,axis=1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# 'Saver' op to save and restore all the variables
saver = tf.train.Saver()

# Test data shapes
print"Test data shape is ", testdata.shape
print"Test data2 shape is ", testdata2.shape
acc_test=[]

with tf.Session() as sess:
	# Initializing the variables
    if args.load=="":
        init = tf.global_variables_initializer()
        sess.run(init)
    else:
        saver.restore(sess,args.load) 
    
    #########################################
    ######          Training Loop      ######
    #########################################
   
    for i in range (0,training_iters):
        if((i%100)==0):

            #### UNCOMMENT for the first phase ####

            # batch_x_test = testdata2
            # print "shape of test data", batch_x_test.shape;

            # batch_y_test = np.zeros([testdata2.shape[0],n_classes]) ;
            # labels_y = np.ravel(testlabels2.astype(np.int32)) ;
            # print labels_y.shape
            # starts = np.arange(0,testdata2.shape[0])*n_classes+labels_y ;
            # batch_y_test.ravel()[starts] = 1 ;        
            # batch_y_test=batch_y_test.reshape(testdata2.shape[0],n_classes);
            # # print batch_y_test
            
            # # Calculate batch accuracy
            # _pred, acc = sess.run([pred, accuracy], feed_dict={x: batch_x_test, y: batch_y_test})
                    
            # # Calculate batch loss
            # loss = sess.run(cost, feed_dict={x: batch_x_test, y: batch_y_test})
            # print("Iter ONE " + str(i) + ", Minibatch Loss= " + \
            #       "{:.6f}".format(loss) + ", Training Accuracy= " + \
            #       "{:.5f}".format(acc))
            # print("##############################################################")

            #### UNCOMMENT for the second phase ####

            # batch_x_test2 = np.append(testdata2, testdata, axis=1); ## for append
            # print "shape of test data", batch_x_test2.shape;

            # batch_y_test2 = np.zeros([testdata.shape[0],n_classes]) ;
            # labels_y2 = np.ravel(testlabels.astype(np.int32)) ;
            # print labels_y2.shape
            # starts2 = np.arange(0,batch_x_test2.shape[0])*n_classes+labels_y2 ;
            # batch_y_test2.ravel()[starts2] = 1 ;        
            # batch_y_test2=batch_y_test2.reshape(batch_x_test2.shape[0],n_classes);
                        
            # #Calculate batch accuracy
            # _pred, acc = sess.run([pred, accuracy], feed_dict={x: batch_x_test2, y: batch_y_test2})

            # # Calculate batch loss
            # loss = sess.run(cost, feed_dict={x: batch_x_test2, y: batch_y_test2})
            # print("Iter TWO" + str(i) + ", Minibatch Loss= " + \
            #       "{:.6f}".format(loss) + ", Training Accuracy= " + \
            #       "{:.5f}".format(acc))
            # print("##############################################################")