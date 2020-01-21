#!/usr/bin/python3
# -*- coding: UTF-8 -*-

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import models
from scipy.interpolate import interp1d

def history_map(loss,val_loss,N=100,norm=True,partial=1.0):
   """
   Get N=100 points along the training history and normalize if requested
   """
   loss = loss[:int(len(loss)*partial)]
   val_loss = val_loss[:int(len(val_loss)*partial)]
   Loss = interp1d(range(len(loss)), loss)
   VLoss = interp1d(range(len(val_loss)), val_loss)
   x = 1/(np.linspace(1,N+1,N)**0.05)
   x = (x-np.min(x))/(np.max(x)-np.min(x))
   x *= len(loss) - 1
   # x *= loss.shape[0]-1
   x = np.sort(x)
   X,Y,Yv = [],[],[]
   for ix in x:
      X.append(ix)
      Y.append(Loss(ix))
      Yv.append(VLoss(ix))
   Y = np.array(Y)
   Yv = np.array(Yv)
   if norm:
      y1 = max([np.max(Y),np.max(Yv)])
      y0 = min([np.min(Y),np.min(Yv)])
      Y  = (Y-y0)/(y1-y0)
      Yv = (Yv-y0)/(y1-y0)
   return Y,Yv


class StopOnConvergence(tf.keras.callbacks.Callback):
   """
   Stops training if the std of the loss in the last N epochs is < threshold
   of if a file STOP is present in the folder
   """
   def __init__(self,N=30,threshold=0.05,fmodel='converger.h5'):
      self.N = N
      self.thres = threshold
      self.loss = []
      self.val_loss = []
      self.converger = models.load_model(fmodel)
   def on_epoch_end(self, epoch, logs={}):
      stop = False
      N = self.N
      thres = self.thres
      self.loss.append(logs.get('loss'))
      # self.loss = self.loss[int(-1.2*N):]
      self.val_loss.append(logs.get('val_loss'))
      # self.val_loss = self.val_loss[int(-1.2*N):]
      if len(self.loss) > N:
         # Check for convergence
         std = np.std(self.loss[-N:])
         if std < thres:
            print('\n\nConvergence?')
            stop = True
         # Check for overfit
         l,lv = history_map(self.loss, self.val_loss)
         inp = np.expand_dims(np.concatenate((l,lv)), axis=0)
         prediction = self.converger.predict(inp)
         i = np.argmax(prediction)
         if i == 1:
            print('\n\nOverfitting?')
            stop = True
      if os.path.isfile('STOP'):
         print('\n\nExternal stop')
         stop = True
      self.model.stop_training = stop
