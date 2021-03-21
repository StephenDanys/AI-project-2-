# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 17:18:53 2021

@author: pchrk
"""
import numpy as np
  
def accuracy(ytrue,yhat):
    accuracy =np.sum(yhat==ytrue)/len(ytrue)
    return accuracy

#true positive reviews/total predicted positive reviews
#total predicted positive reviews are the 1 in the yhat vector
def precision_for_positive_class(ytrue,yhat):
    pfpc=np.sum((yhat==ytrue)&(yhat==1))/np.count_nonzero(yhat)
    return pfpc

def recall_for_positive_class(ytrue,yhat):
    rfpc=np.sum((yhat==ytrue)&(yhat==1))/np.count_nonzero(ytrue)
    return rfpc

def precision_for_negative_class(ytrue,yhat):
    pfnc=np.sum((yhat==ytrue)&(yhat==0))/(len(yhat)-np.count_nonzero(yhat))
    return pfnc

def recall_for_negative_class(ytrue,yhat):
    rfnc=np.sum((yhat==ytrue)&(yhat==0))/(len(ytrue)-np.count_nonzero(ytrue))
    return rfnc

def macro_recall(recall1,recall2):
    return (recall1+recall2)/2
def macro_precision(precision1,precision2):
    return (precision1+precision2)/2

def F1(recall,precision):
    return (2*precision*recall)/(precision+recall)


def print_results(y_test,final_pred):
    print("accuracy:",accuracy(y_test,final_pred))
    print("positive precision",precision_for_positive_class(y_test,final_pred))
    print("positive recall",recall_for_positive_class(y_test,final_pred))
    print("negative precision",precision_for_negative_class(y_test,final_pred))
    print("negative recall",recall_for_negative_class(y_test,final_pred))
    print("F1:",F1(macro_recall(recall_for_positive_class(y_test,final_pred),recall_for_negative_class(y_test,final_pred)),macro_precision(precision_for_positive_class(y_test,final_pred),precision_for_negative_class(y_test,final_pred))))