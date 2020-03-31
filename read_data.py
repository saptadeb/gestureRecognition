#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri May  5 11:29:16 2017

@author: admin
"""

import os,sys
#from pathlib import Path
import numpy as np


data=[]
# arr is organized as follows: [sampleIDX, frameIDX, histogramIDX]
# samples go from 0 to 80
path=sys.argv[1]
nrClasses=4;

# determine numberr of classes: 4
# det nr of persons
dirs = [f for f in os.listdir(path) if f[0]=='p'] ;
nrPersons=len(dirs)
print "Nr of persons", len(dirs)

# det nr of recorded gestures per person
nrGestures = len([f for f in os.listdir(os.path.join(path,"p1")) if f[0]=='g'])-1
print "nr of gestures", nrGestures

# det nr of frames per gesture
nrFrames = len([f for f in os.listdir(os.path.join(path,"p1", "g0","cropped")) ])
print "nrFrames=", nrFrames ;
 
arr=np.zeros((nrPersons*nrGestures,nrFrames,625))
labels=np.zeros((nrPersons*nrGestures,1)) ;


# iteration over classes
index=0;
for p in range(0,nrPersons):
        counter=0
        # iteration over samples
        
        for g in xrange(0,nrGestures):
            labels[index,0] = g//10 ;
            # iteration over frames
            j=0
            for fr in xrange(0,nrFrames):
                fname=os.path.join(path,"p"+str(p), "g"+str(g), "cropped", "p"+str(p)+"-g"+str(g)+"-s"+str(fr)+".txt") ;
                line=file(fname).readline() ;


                st=[float(x) for x in line.split(",") if x.find ("g")==-1]
                arr[index,j,:]=(st[0:625])
                j+=1;
            index+=1
                
          
print ("Train")
np.save('all_data', arr) ;
np.save('all_labels', labels) ;
#a=np.array(data)
#a=np.reshape(a,(300*4,40,625))

#np.save('test_data', arr)
#b=np.load('test_data.npy')
#print b

#print (np.equal(arr,b))


"""
for i in range (0,80):
    for j in range (0,40):
        for k in range (0,625):
            print (a[i,j,k],"   -   ", arr[i,j,k])
            if (a[i,j,k]==arr[i,j,k]):
                continue
            else:
                print "NOT"
                break
            
            print "HI"
"""         


"""
if os.path.exists('/home/admin/rnn&lstm_gesture_recog/max_mins/mins/class1.npy')==True:
    overall_class = list(np.load('/home/admin/rnn&lstm_gesture_recog/max_mins/mins/class3.npy'))
print (np.array(overall_class)) 
""" 
    
