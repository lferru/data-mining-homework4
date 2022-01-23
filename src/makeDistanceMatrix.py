'''
Luis Ferrufino
CS 484-002
4/13/20
makeDistanceMatrix.py
'''

import numpy as np

dist = np.zeros( (498, 498) )
neighbs = np.zeros( (498, 498) )

a = np.load('./baselineA.npy')
b = np.load('./baselineB.npy')
c = np.load('./baselineC.npy')
d = np.load('./baselineD.npy')
m = np.load('./baselineM.npy')

#we need to remove 2 samples whose corresponding files were empty:

d = np.delete(d, obj=68, axis=0)
m = np.delete(m, obj=58, axis=0)

baseline = np.concatenate( (a, b, c, d, m) )
labels = np.array( ['A'] * 100 + ['B'] * 100 + ['C'] * 100 + ['D'] * 99 + ['M'] * 99 )

for i in range(0, 498):

  for j in range(0, 498):

    dist[i][j] = np.linalg.norm(baseline[i] - baseline[j])
  print('Pass ' + str(i) + ' out of 497')

np.save(arr=dist, file='./distMatrix.npy')
np.save(arr=labels, file='./labelVector.npy')
