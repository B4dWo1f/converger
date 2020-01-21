#!/usr/bin/python3
# -*- coding: UTF-8 -*-

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow.keras import models
from tensorflow.keras.layers import Dense
import tf_helper
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from random import randint, choice, shuffle
import explore



fmodel = 'converger.h5'
classes = ['Converged', 'Overfitted']

conv = os.popen('ls data/conv*.dat').read().strip().splitlines()
over = os.popen('ls data/over*.dat').read().strip().splitlines()

tests = [choice(conv), choice(over)]

files = conv+over

for rm in tests:
   files.remove(rm)

inps, outs = [],[]
for f in files:
   l,lv = explore.read_map(f)
   name = f.split('/')[-1].split('.')[0]
   if name.startswith('conv'): outs.append(0)  #np.array((1,0)))
   elif name.startswith('overfit'): outs.append(1)  #np.array((0,1)))
   else: continue
   inps.append(np.concatenate((l,lv)))


inps = np.array(inps)
outs = np.array(outs)
inp_shape = inps[0].shape

model = models.Sequential()
model.add(Dense(300, activation='tanh',input_shape=inp_shape))
model.add(Dense(100, activation='tanh'))
model.add(Dense(50, activation='tanh'))
model.add(Dense(10, activation='tanh'))
model.add(Dense(2, activation='softmax'))

Stopper = tf_helper.StopOnConvergence(50,5e-4)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
                                metrics=['accuracy'])

history = model.fit(inps, outs, validation_split=0.1, epochs=500, verbose=2,
                                callbacks=[Stopper])
# model.save(fmodel)

print('done')
err = history.history['loss']
# fig, ax = plt.subplots()
fig = plt.figure()
gs = gridspec.GridSpec(2, 1)
fig.subplots_adjust(wspace=0.,hspace=0.15)
ax1 = plt.subplot(gs[0])  # converged
ax2 = plt.subplot(gs[1], sharex=ax1)  # overfitted
ax1.plot(err)
X = list(range(len(err)))
Y = [0 for _ in X]
for i in range(len(X)):
   if i > 50: Y[i] = np.std(err[i-50:i])
ax2.plot(X,Y)
try:
   val_err = history.history['val_loss']
   ax1.plot(val_err)
except KeyError: pass


print('    -------- Test --------')
inds = [0,1]
shuffle(inds)
conv = tests[inds[0]]
over = tests[inds[1]]
part = 0.2

fig = plt.figure()
gs = gridspec.GridSpec(1, 2)
fig.subplots_adjust(wspace=0.,hspace=0.15)
ax1 = plt.subplot(gs[0])  # converged
ax2 = plt.subplot(gs[1], sharey=ax1)  # overfitted

for fname,ax in zip(tests,[ax1,ax2]):
   Fl,Flv = explore.read_map(fname)
   l,lv = explore.read_map(fname, partial=part)
   inp = np.expand_dims(np.concatenate((l,lv)), axis=0)
   prediction = model.predict(inp)
   i = np.argmax(prediction)
   fname = fname.split('/')[-1][:4]
   print(f'{fname}:',prediction[0])
   ax.plot(l)
   ax.plot(lv)
   ax.plot(Fl,'C0--')
   ax.plot(Flv,'C1--')
   if not fname in classes[i].lower(): color = 'red'
   else: color = 'black'
   ax.set_title(f'{classes[i]} ({prediction[0][i]*100:.2f}%)',color=color)

# Fl,Flv = explore.read_map(over)
# l,lv = explore.read_map(over, partial=part)
# ax2.plot(x,l)
# ax2.plot(x,lv)
# ax2.plot(Fl,'C0--')
# ax2.plot(Flv,'C1--')
# inp = np.expand_dims(np.concatenate((l,lv)), axis=0)
# prediction = model.predict(inp)
# i = np.argmax(prediction)
# print('over:',prediction[0])
# ax2.set_title(f'{classes[i]} ({prediction[0][i]*100:.2f}%)')

plt.show()
