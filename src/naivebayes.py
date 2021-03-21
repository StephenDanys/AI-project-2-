# -*- coding: utf-8 -*-
"""
Created on Mon Jan  4 10:19:42 2021

@author: pchrk
"""
import numpy as np
class NaiveBayes():
    
    def calculate_class_probabilities(self,y):
        prob_positive=np.divide(np.count_nonzero(y),y.shape[0])
        prob_negative=1-prob_positive
        self.class_prob=np.array([prob_positive,prob_negative])
        print("calculated class probabilities are ", self.class_prob)
    
    #we take a column of a feature transpose it and calculate
    #the conditional prob.this is possible because only 0,1 
    #in both x,y
    #it returns are the count of each feature so we have to div by 
    def calculate_conditional_probability_i(self,x_col,y,total_pos,total_neg):
        #count of feature when review positive
        feature_count_pos=0 
        #count of feature when review negative
        feature_count_neg=0
    #find count of feature|negative
        for x_i,y_i in zip(x_col,y):
            if x_i>0 and y_i>0:
                feature_count_pos=feature_count_pos+1
            elif x_i>0 and y_i==0:
                feature_count_neg=feature_count_neg+1
       # print(feature_count_pos/total_pos,feature_count_neg/total_neg)
        return feature_count_pos/total_pos,feature_count_neg/total_neg
    
    
        
    def calculate_conditional_probabilities(self,x,y):
        #2 conditional prob for each feature
        # gives 2 rows and #feature columns
        samples,features=x.shape
        total_pos=np.count_nonzero(y)
        total_neg=samples-total_pos
        self.conditional_prob=np.zeros((2,features))

        for feature in range(features):
            x_col=x[0:samples,feature]
            self.conditional_prob[0][feature],self.conditional_prob[1][feature]=self.calculate_conditional_probability_i(x_col,y,total_pos,total_neg)
            
        #print(self.conditional_prob)
        return self.conditional_prob
    
    #classify a vector x with the same number of features
    #returns a vector with the predictions for the cvector x
    """
    predictions 0 col=positive
    predictions 1 col=negative 
    
    
    """
    
    
    def predict(self,x):
        predictions=np.zeros((x.shape[0],2))
        
        
        for rows_id,x_i in enumerate(x):
            for idx,feature in enumerate(x_i):
                if (feature==1):
                    predictions[rows_id][1]+=np.log(self.conditional_prob[0][idx])
                    predictions[rows_id][0]+=np.log(self.conditional_prob[1][idx])
                else:
                    predictions[rows_id][1]+=np.log((1-self.conditional_prob[0][idx]))
                    predictions[rows_id][0]+=np.log((1-self.conditional_prob[1][idx]))
        
        #multiply each class with the probability of the class
        predictions=np.exp(predictions)
        predictions[:,0]=predictions[:,0]*self.class_prob[0]
        predictions[:,1]=predictions[:,1]*self.class_prob[1]
        #print(predictions,predictions.shape)
        final_pred=np.zeros(predictions.shape[0])
        i=0
        for pred in predictions:
            final_pred[i]=np.argmax(pred)
            i=i+1
        return final_pred
        #final_pred contain the predicted class for each review
        #predictions[0] is likelihood of review being positive
        #predictions[1] is likelihood of review being negative
        #return predictions
    
    
    
    
    
"""class NaiveBayes():
    

          
        
    def calculate_class_probabilities(self,y):
        prob_positive=np.divide(np.count_nonzero(y),y.shape()[0])
        prob_negative=1-prob_positive
        self.class_prob=np.array([prob_positive,prob_negative])
        print("changed class probabilities")
    
    #we take a column of a feature transpose it and calculate
    #the conditional prob.this is possible because only 0,1 
    #in both x,y
    
    def calculate_conditional_probability_i(self,x_col,y):
        ar=np.dot(np.transpose(x_col),y)
        return ar
        
    def calculate_conditional_probabilities(self,x,y):
        #2 conditional prob for each feature
        # gives 2 rows and #feature columns
        samples,features=x.shape
        self.conditional_prob=np.zeros((2,features))

        for feature in range(features):
            x_col=x[0:samples,feature]
            self.conditional_prob[0][feature]=np.divide( self.calculate_conditional_probability_i(x_col,y),features)
            self.conditional_prob[1][feature]=1-self.conditional_prob[0][feature]
    
        print("changed conditional _probabilities")
        print("conditional probabilities are:", self.conditional_prob)
    #classify a vector x with the same number of features
    #returns predictions vector with the predictions for the cvector x as percentage
    def predict(self,x):
        y_hat=np.dot(x,np.transpose(self.conditional_prob))
        #predictions=[1 if i>0.5 else 0 for i in y_hat[0]]
        return y_hat
"""

















