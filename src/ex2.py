# -*- coding: utf-8 -*-
"""
Created on Sun Jan  3 15:32:16 2021

@author: pchrk
"""

import numpy as np
import random
import naivebayes
import LogReg
from print_results import print_results
  
        
bow=open("labeledBow.feat", 'r')
vocabulary=open("imdb.vocab",'r')
example_size=5000

bowlist1=bow.readlines()[:example_size]
bow.seek(0)
bowlist2=bow.readlines()[12500:12500+example_size]
bowlist=bowlist1+bowlist2
num_in_dict=[]
y0=np.zeros(example_size)
y1=np.ones(example_size)
y_vec=np.concatenate((y1,y0))
for item in bowlist:
    words_freq = item.split(' ')[1:]
    num_in_dict.append([ w.split(":")[0] for w in words_freq])
#print(y_vec)


#yperparametroi m,n
LOWER_LIMIT=48
UPPER_LIMIT=1000
x_vector = [ [0]*(UPPER_LIMIT-LOWER_LIMIT) for _ in range(2*example_size) ] 
n=-1
for sentence in num_in_dict:
    n+=1
    #print(sentence)
    for elem in sentence:
        if int(elem)>=UPPER_LIMIT:
            break
        elif int(elem)>=LOWER_LIMIT:
            #print(n,elem)
            x_vector[n][int(elem)-LOWER_LIMIT]=1

#tha data to train the model on are the x_vecotr with 0,1 for the words
#and the y_train with 1 for positive and 0 for negative reviews
x_vec=np.array(x_vector)


def split_data(data, prob):
    results = [], []
    random.seed(0)
    for row in data:
        results[0 if random.random() < prob else 1].append(row)
    return results

def train_test_split(x, y, test_pct):
    data = zip(x, y) # pair corresponding values
    train, test = split_data(data, 1 - test_pct) # split the data set of pairs
    x_train, y_train = zip(*train) # magical un-zip trick
    x_test, y_test = zip(*test)
    return x_train, x_test, y_train, y_test

  

x_train, x_test, y_train, y_test=train_test_split(x_vec,y_vec,0.2)
x_train=np.array(x_train)
y_train=np.array(y_train)
x_test=np.array(x_test)
y_test=np.array(y_test)

#print(np.shape(x_train),np.shape(y_train))


#paketo gia na ektypvneis kai na ekpaideyeis logistic regression

regressor=LogReg.LogisticRegression(0.005,5000)
regressor.fit(x_train,y_train)
predictions=regressor.predict(x_test)
predictions=np.array(predictions)


print_results(y_test,predictions)



'''
#paketo gia na ekpaideyeis kai na ektypvneis bayes

regressor=naivebayes.NaiveBayes()
regressor.calculate_class_probabilities(y_train)
regressor.calculate_conditional_probabilities(x_train,y_train)
#predictions contain the probability for both classes
predictions=regressor.predict(x_test)
print_results(y_test,predictions)

'''









