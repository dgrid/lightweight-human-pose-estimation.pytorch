#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 16 12:42:47 2019

@author: takatoyamada
"""
from django.core.management.base import BaseCommand
import pickle
import MeCab
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer 

vectorizer = CountVectorizer(min_df = 60)
mecab = MeCab.Tagger('mecabrc')

class Command(BaseCommand):
    def handle(self, *args, **options):
        with open('./app/python/pickle/words_list.pickle', mode = 'rb')as f:
            words_list = pickle.load(f)
        X = vectorizer.fit_transform(words_list).toarray()
        y = make_label()
        (X_train, X_test, y_train, y_test) = train_test_split(X,y,test_size=0.3)
        train(X_train,y_train)
        valid(X_train,y_train,X_test,y_test)

def make_label():
    for i in range(8):
        num = i+1
        y_ = np.full(100,num)
        if num ==1:
            y = y_
        else:
            y = np.concatenate((y,y_)) 
    return y     

def train(X_train,y_train):
    mu_lists =[]
    cov_lists = []
    for i in range(8):
        num = i+1
        X_i = X_train[np.where(y_train==num)]
        cov = np.cov(X_i.T)   
        mu = np.mean(X_i.T,axis = 1).reshape(-1,1)
        mu_lists.append(mu)
        cov_lists.append(cov)
    with open('./app/python/pickle/mu_lists.pickle', mode = 'wb')as f:
        pickle.dump(mu_lists,f)
    with open('./app/python/pickle/cov_lists.pickle', mode = 'wb')as f:
        pickle.dump(cov_lists,f)

def np_log(x):
    return np.log(np.clip(a=x,a_min=1e-10,a_max=x))

def calcu_Pc(X,y):
    P = np.zeros((1,8))
    for i in range(8):
        num = i+1
        P[0][i] = np.count_nonzero(y==num)/len(y)
    Pc = np_log(P) 
    return Pc

def valid(X_train,y_train,X_valid,y_valid):
    with open('./app/python/pickle/mu_lists.pickle', mode = 'rb')as f:
        mu_lists = pickle.load(f)
    with open('./app/python/pickle/cov_lists.pickle', mode = 'rb')as f:
        cov_lists = pickle.load(f)
    m = len(y_valid)
    alpha = 0.1
    predicts = np.zeros((1,m))
    log_Pc = calcu_Pc(X_train,y_train)
    for i in range(m):  
        x = X_valid[i]
        g = np.zeros((1,8))
        for j in range(8):
            cov_mat = cov_lists[j]
            mu = mu_lists[j]
            ic = np.linalg.inv(cov_mat+alpha*np.eye(cov_mat.shape[0]))
            W = -1/2*ic
            w = np.dot(ic,mu)
            w0 = np.dot(mu.T,np.dot(W,mu))+log_Pc[0][j]#-1/2*np.log(np.linalg.det(ic))
            g[0][j] = np.dot(x.T,np.dot(W,x))+np.dot(w.T,x)+w0
        pre = np.argmax(g)
        predicts[0][i] = pre+1
    accu = np.count_nonzero(predicts[0] == y_valid)/m
    print("<improved method>")
    print("accuracy rate:%s"%str(accu))
    