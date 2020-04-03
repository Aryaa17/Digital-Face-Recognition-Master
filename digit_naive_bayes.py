#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 23 05:08:16 2018

@author: ananya
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.measurements import label
import scipy.stats
import time

#Preprocessing training data
start_idx = 0
end_idx = 0
total_train_samples = 5000
x_percent = int(input('Input the Percent of Data set you want to train your system on: '))
num_instance = int((total_train_samples * x_percent)/100)

H=28
W=28
start_time = time.time()
X_train=[[] for k in range(num_instance)]
with open('digitdata/trainingimages') as f :
    
    for k in range(num_instance):
        #count_hash = 0
        #count_plus = 0
        datainstance=[[] for k in range(H)]
        for j in range(H):
            line=f.readline()
            l=[]
            for ch in range(len(line)-1):
                
                ## if you try to convert to numbers 
                ## do it here
                if line[ch] == ' ':
                    l.append(0)
                if line[ch] == '+':
                    l.append(1)
                    #count_plus += 1
                if line[ch] == '#':
                   l.append(2)
                   #count_hash += 1
                   
            datainstance[j]=l
            #print(line)
        
        #print(count_hash)
        #print(count_plus)
        #print("___________________________________")
        X_train[k]=datainstance
        

X_train=np.array(X_train)
# save as numpy format


# read train targets
Y_train=np.loadtxt('digitdata/traininglabels')
end_idx = num_instance
Y_train = np.array(Y_train[start_idx:end_idx])

#time taken to preprocess the training data
prep_train_time = time.time() - start_time
print('time taken to preprocess the training data: ')
print(x_percent, 'percent of data took ', prep_train_time)


#Preprocessing test data
############################################################
num_instance=1000
start_time = time.time()
X_test=[[] for k in range(num_instance)]
with open('digitdata/testimages') as f :
    
    for k in range(num_instance):
        #count_hash = 0
        #count_plus = 0
        datainstance=[[] for k in range(H)]
        for j in range(H):
            line=f.readline()
            l=[]
            for ch in range(len(line)-1):
                
                ## if you try to convert to numers 
                ## do it here
                if line[ch] == ' ':
                    l.append(0)
                if line[ch] == '+':
                    l.append(1)
                    #count_plus += 1
                if line[ch] == '#':
                   l.append(2)
                   #count_hash += 1
                   
            datainstance[j]=l
            #print(line)
        #print(count_hash)
        #print(count_plus)
        #print("___________________________________")
        #print(datainstance)
        X_test[k]=datainstance
        

X_test=np.array(X_test)
Y_test=np.loadtxt('digitdata/testlabels')

#time taken to preprocess the test data
prep_test_time = time.time() - start_time
print('time taken to preprocess the test data: ')
print('data took ', prep_test_time)

################################
(m, n, p) = X_train.shape
print(m)
print(n)
print(p)


def countPlus(sample):
    m,n=sample.shape
    return sum(sum(sample==2))

def countX(sample):
    m,n=sample.shape   
    return sum(sum(sample==1))

def countZero(sample):
    m,n=sample.shape   
    return sum(sum(sample==0))



def countTop(sample):
    m,n = sample.shape
    count_zero = 0
    count_X = 0
    count_Plus = 0
    for i in range(int(m/2)):
        for j in range(n):
            if(sample[i][j]==0):
                count_zero = count_zero+1
            if(sample[i][j]==1):
                count_X = count_X+1
            if(sample[i][j]==2):
                count_Plus = count_Plus+1
    return count_zero, count_X, count_Plus



def countBottom(sample):
    m,n = sample.shape
    count_zero = 0
    count_X = 0
    count_Plus = 0
    for i in range(int(m/2)):
        for j in range(n):
            if(sample[int(m/2)+i][j]==0):
                count_zero = count_zero+1
            if(sample[i][j]==1):
                count_X = count_X+1
            if(sample[i][j]==2):
                count_Plus = count_Plus+1
    return count_zero, count_X, count_Plus




def countLeft(sample):
    m,n = sample.shape
    count_zero = 0
    count_X = 0
    count_Plus = 0
    for i in range(m):
        for j in range(int(n/2)):
            if(sample[i][j]==0):
                count_zero = count_zero+1
            if(sample[i][j]==1):
                count_X = count_X+1
            if(sample[i][j]==2):
                count_Plus = count_Plus+1
    return count_zero, count_X, count_Plus



def countRight(sample):
    m,n = sample.shape
    count_zero = 0
    count_X = 0
    count_Plus = 0
    for i in range(m):
        for j in range(int(n/2)):
            if(sample[i][int(n/2)+j]==0):
                count_zero = count_zero+1
            if(sample[i][j]==1):
                count_X = count_X+1
            if(sample[i][j]==2):
                count_Plus = count_Plus+1
    return count_zero, count_X, count_Plus




def reducegreysapce(sample):
    m,n = sample.shape
    begin = np.empty([m], dtype= int)
    ending = np.empty([m], dtype= int)
    #for i in range(m):
        #begin[i]= 28
    Counter1= True
    Counter2 = True
    Counter11= True
    Counter21= True
    column_begin = 0
    column_end = 0
    for i in range(m):
        Counter1 = True
        Counter2 = True
        for j in range(n):
            if (sample[i][j]!=0 and Counter1 == True):
                begin[i]=j
                Counter1 = False
                
            if (sample[i][j]!=0 and Counter11 == True):
                column_begin = i
                Counter11 = False
                
            if (sample[i][n-j-1]!=0 and Counter2 == True):
                 ending[i]=j
                 Counter2 = False
            
            if (sample[n-1-i][j]!=0 and Counter21 == True):
                column_end = n-1-i
                Counter21 = False
                
    #print(begin)
    #print(ending)
    initial_counter= np.amin(begin)
    #print(initial_counter)
    end_counter = np.amax(ending)
    #print(end_counter)
    new_sample=[[] for k in range(m)]
    for i in range(m):
        new_sample[i] = sample[i][initial_counter:end_counter]
    new_sample = np.array(new_sample) 
    #print(new_sample)
    m1, n1 = new_sample.shape
    l = (column_end-column_begin)/(end_counter-initial_counter)
    return new_sample, l

def connectedComponent(sample):
    m,n = sample.shape
    new_sample = [[] for k in range(m)]
    for i in range(m):
        l = []
        for j in range(n):
            if (sample[i][j]==0):
                l.append(1)
            else:
                l.append(0)
        new_sample[i] = l
    sample = np.array(new_sample)
    #print(new_sample)
    structure = np.ones((3, 3), dtype=np.int)  # this defines the connection filter
    labeled, ncomponents = label(new_sample, structure)
    #print(ncomponents)
    if (ncomponents<1):
        ncomponents = 1
    return ncomponents    
    
def computeFea(sample):
    m,n=sample.shape
    f1=countX(sample)
    f2=countPlus(sample)
    [new_sample, l] = reducegreysapce(sample)
    f3 = countZero(new_sample)
    [a1, b1, c1]= countTop(sample)
    #[a2, b2, c2]= countBottom(sample)
    #[a3, b3, c3]= countLeft(sample)
    [a4, b4, c4]= countRight(sample)
    f4 = (b1+c1)/(a1+b1+c1)
    #f5 = (b2+c2)/(a2+b2+c2)
    #f6 = (b3+c3)/(a3+b3+c3)
    f7 = (b4+c4)/(a4+b4+c4)
    f8 =l
    #f9 = connectedComponent(sample)
    #xx=[f1, f2, f3, f4, f5]
    xx=[f1, f2, f3, f4, f7, f8]
    #xx=sample.reshape(28*28,1)
    #xx=list(xx)
    return xx

def createFeatureMatrix (XX):
        
    X_fea=[]   
    for idx, row in enumerate(XX):
        X_fea.append(computeFea(row))
    return np.array(X_fea)

##############################################


start_time = time.time()
X_TrainFea = createFeatureMatrix(X_train)
#X_TestFea = createFeatureMatrix(X_test)

m1, n1 =  X_TrainFea.shape

all_classes, counts=np.unique(Y_train, return_counts=True)
NumClass=len(all_classes)
priors = counts/sum(counts)

means=np.zeros((NumClass,n1))
stds = np.zeros((NumClass, n1))
for t in all_classes:
    t=int(t)
    subset=X_TrainFea[Y_train==t]
    for fea in range(n1):
        means[t,fea]=np.mean(subset[:,fea])
        stds[t, fea]=np.std(subset[:,fea])
        
training_time = time.time() - start_time

###########################################
#Checking the dataset against test data
X_TestFea = createFeatureMatrix(X_test)
Y_pred=np.array([-100 for k in range(len(X_TestFea))])

for idx, row in enumerate(X_TestFea):
    Prob=np.zeros(NumClass)
    for i,c in enumerate(all_classes):
        prob = 1
        for fea in range(n1):
            prob = prob*scipy.stats.norm(means[i, fea], stds[i, fea]).pdf(row[fea])
            Prob[i]=prob
        
    predicted_class_index=np.argmax(Prob)  
    Y_pred[idx]=all_classes[predicted_class_index]
    
    
accuracy = sum(sum([Y_test == Y_pred]))
mse=np.sqrt(sum(sum([Y_test != Y_pred])))/len(Y_pred)
###############
# Calculating standard deviation
result_arr = np.ones(len(Y_test) - accuracy)
zero_arr = np.zeros(accuracy)
result_arr = np.concatenate((result_arr, zero_arr))

result_mean=np.mean(result_arr)
result_std=np.std(result_arr)

print('################### STATISTICS #####################')
print('Accuracy: ', accuracy, 'out of ', len(Y_test), 'percentage: ', (accuracy/len(Y_test))*100, '%' )

print('time taken to preprocess the training data: ')
print(x_percent, 'percent of data took ', prep_train_time)

print('time taken to preprocess the test data: ')
print('data took ', prep_test_time)

print('time taken to train: ')
print(x_percent, 'percent of data took ', training_time)

print('\n\n\n')
print('-------------nice format------------')
print('Number of test data points: ', len(Y_test))
#print('Total number of training points', total_train_samples)
print('Percentage of training data used: ', x_percent, '%')
print('Number of training data points: ', len(Y_train))
print('Accuracy: ', (accuracy/len(Y_test))*100, '%' )
print('Prediction Error: ', (100 - ((accuracy/len(Y_test))*100)), '%' )
print('Training time: ', training_time, 'seconds')
print('standard Deviation', result_std)

# another way of counting plus and hash

"""
count_x1 = 0
count_x2 = 0

for i in range(H):
    for j in range(W):
        #print(X_test[0, i, j])
        if X_train[1, i, j] == 1:
            count_x1 += 1
        if X_train[1, i, j] == 2:
            count_x2 += 1
print(count_x1)
print(count_x2)
"""
