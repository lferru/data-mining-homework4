'''
Luis Ferrufino
CS 484-002
4/13/20
extractFeatures.py
'''

import numpy as np

baseline = np.zeros((100, 20000), dtype=np.cdouble)
modes = ['A', 'B', 'C', 'D', 'M']

for letter in modes:

  for i in range(0, 100):

    newSample = np.genfromtxt('./base/Mode' + letter + '/File' + str(i) + '.txt')

    if newSample.size == 0:
      print("REPORT: File" + str(i) + '.txt in folder Mode' + letter + ' is empty') 
      continue
    baseline[i] = np.fft.fft(newSample)
    #baseline[i] /= np.linalg.norm(baseline[i])
    print('Pass ' + letter + ', ' + str(i))

  np.save('./baseline' + letter + '.npy', baseline)
