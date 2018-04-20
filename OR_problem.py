# -*- coding: utf-8 -*-
"""
Created on Fri Apr 20 19:42:02 2018

@author: louis
"""
import numpy as np

data = np.array([[1,0,0],[1,1,0],[1,0,1],[1,1,1]]) # 4x3 array

y = [-1,1,1,1]

w = np.array([1, -1, 1]) # 1x3 array

grad = np.zeros((3,), np.int32)


def f(w, x): # linear classification function
  return np.sign(w.T.dot(x))
stop = False
epoch = 1
print('Epoch %d: Gradient Vector=[%d %d %d], Initial Weight w = [%d %d %d]'%(0, grad[0], grad[1], grad[2], w[0], w[1], w[2]))
    
    
while not stop: # repeat the training until no misclassification
   grad = [0, 0, 0]
   stop = True
   for i, x in enumerate(data): # input sample by sample
      if f(w,x)!=y[i]: stop = False # misclassify the sample
      grad += (y[i] - f(w,x))*x # accumulate the gradient vec.
   w += grad # batch update on w
   print('Epoch %d: Gradient Vector=[%d %d %d], New Weight w = [%d %d %d]'%(epoch, grad[0], grad[1], grad[2], w[0], w[1], w[2]))
   epoch += 1
