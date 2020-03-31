#Goal: to test whether LSTM can handle gesture onset delays that are unknown. Because in an application, it is not known when the gesture starts. The user has to input an integer N (0-20). The system will decide a random integer between 0-N, and at that index the data has to be embedded. then.. separate train test.. train and test. (ZERO FRAMES)

# command to run this
# python test10.py --db all_data.npy --labels all_labels.npy --bs 10 --layers 3 --cells 20 --iters 10000 --output 39 --fracTest 0.2 --nframes 0 --save ./1001

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
parser.add_argument("--nframes", help="no. of frames to append, max-20",type=int,default=0)
parser.add_argument("--save", help="name of tf checkpoint to save trained net state",type=str,default = "./model.ckpt")

args = parser.parse_args()

# Training Parameters
learning_rate = 0.0001
training_iters = args.iters
batch_size = args.bs
display_step = 4
f2app = args.nframes ## N
alldataa = np.load(args.db) ;
alllabels = np.load(args.labels) ;
print "DATA", alldataa.shape

alldata = np.zeros([alldataa.shape[0], alldataa.shape[1]+20, alldataa.shape[2]])

for ad in range(alldataa.shape[0]):
    for i in range(alldataa.shape[1]):
        alldata[ad,i+random.randint(0,f2app),:] = alldataa[ad,i,:]
print "DATA", alldata.shape
np.save(args.save, alldata)

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
n_steps = alldataa.shape[1]+20   # timesteps
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
        label_y = []
        data_x = [] ;

        # draw random minibatch from data
        rand_n = np.random.random_integers(0, len(data)-batch_size) ;
        data_x = data[rand_n:rand_n+batch_size] ;
        data_y = np.zeros([batch_size,n_classes]) ;
        labels_y = np.ravel(labels[rand_n:rand_n+batch_size].astype(np.int32)) ;
        starts = np.arange(0,batch_size)*n_classes+labels_y ;
        data_y.ravel()[starts] = 1 ;        
        data_y=data_y.reshape(batch_size,n_classes);
        batch_x = np.array(data_x)                
        batch_y = np.array(data_y)
        
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})    
        if((i%100)==0):
            batch_x_test = testdata ;
            batch_y_test = np.zeros([testdata.shape[0],n_classes]) ;
            labels_y = np.ravel(testlabels.astype(np.int32)) ;
            starts = np.arange(0,testdata.shape[0])*n_classes+labels_y ;
            batch_y_test.ravel()[starts] = 1 ;        
            batch_y_test=batch_y_test.reshape(testdata.shape[0],n_classes);
            # print batch_y_test
            
            # Calculate batch accuracy
            _pred, acc = sess.run([pred, accuracy], feed_dict={x: batch_x_test, y: batch_y_test})
                    
            # Calculate batch loss
            loss = sess.run(cost, feed_dict={x: batch_x_test, y: batch_y_test})
            print("Iter " + str(i) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc))
            print("##############################################################")
    print("Model saved in file: %s" % args.save)
    save_path = saver.save(sess, "./"+args.save)