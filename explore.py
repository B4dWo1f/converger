#!/usr/bin/python3
# -*- coding: UTF-8 -*-

import os
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from matplotlib import gridspec


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


def read_map(fname,N=100,norm=True,partial=1.0):
   """
   read the data and map it to N=100 points
   if norm == True: normalize the data
   """
   loss, val_loss = np.loadtxt(fname,unpack=True)
   return history_map(loss,val_loss,N=N,norm=norm,partial=partial)


if __name__ == '__main__':
   fol = 'data'

   fig = plt.figure()
   gs = gridspec.GridSpec(2, 1)
   ax1 = plt.subplot(gs[0,0])
   ax2 = plt.subplot(gs[1,0])

   convs = os.popen(f'ls {fol}/conv*.dat').read().strip().splitlines()
   for i in range(len(convs)):
      f = convs[i]
      print(i,f)
      Y,Yv = read_map(f)
      ax1.plot(Y,f'C{i%9}')
      ax1.plot(Yv,f'C{i%9}--')

   print('-'*80)
   overs = os.popen(f'ls {fol}/overfit*.dat').read().strip().splitlines()
   for i in range(len(overs)):
      f = overs[i]
      print(i,f)
      Y,Yv = read_map(f)
      ax2.plot(Y,f'C{i%9}')
      ax2.plot(Yv,f'C{i%9}--')
   fig.tight_layout()
   plt.show()
