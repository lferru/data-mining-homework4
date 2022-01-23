'''Luis Ferrufino
CS 484-002
final.py
4/16/20
HW#4
G#00997076
'''

import numpy as np

trainDist = np.load('./distMatrix.npy')
trainDist = np.delete(trainDist, list(range(399,498)), axis = 0) #throw away the rows corresponding to mode M
trainDist = np.delete(trainDist, list(range(399,498)), axis = 1) #throw away the cols corresponding to mode M
aData = np.load('./baselineA.npy')
bData = np.load('./baselineB.npy')
cData = np.load('./baselineC.npy')
dData = np.load('./baselineD.npy') 
dData = np.delete(dData, 68, 0) # remove empty entry
trainData = np.concatenate( (aData, bData, cData, dData) )

k = 67

testData = np.zeros((499, 20000), dtype=np.cdouble)

#obtain test data:

print("(1/12) Extracting test samples...")

for i in range(1, 500):

  newSample = np.genfromtxt('./TestWT/Data' + str(i) + '.txt')

  testData[i - 1] = np.fft.fft(newSample)
print('>Done')
print('(2/12) Computing distance matrix for test and training sets...')

testDist = np.zeros((testData.shape[0], trainDist.shape[1])) #the distance from the test points to the training points

for i in range(testDist.shape[0]):

  for j in range(testDist.shape[1]):

    testDist[i][j] = np.linalg.norm(testData[i] - trainData[j])

print('>Done')
print('(3/12) Computing neighbour matrix for training set...')

#compute neighbour matrices for each set:

trainNeighb = np.zeros(trainDist.shape, dtype=int)

for i in range(trainNeighb.shape[0]):

  trainNeighb[i] = np.argsort(trainDist[i])

print('>Done')
print('(4/12) Computing neighbour matrix for test set...')

testNeighb = np.zeros(testDist.shape, dtype=int)

for i in range(testNeighb.shape[0]):

  testNeighb[i] = np.argsort(testDist[i])

print('>Done')
print('(5/12) Computing training reachability matrix...')

#compute strangeness training list:

trainRd = np.zeros(trainDist.shape)

for i in range(trainDist.shape[0]):

  for j in range(trainDist.shape[1]):

    trainRd[i][j] = max( trainDist[i][trainNeighb[i][k]], trainDist[i][j] )

print('>Done')
print('(6/12) Computing training local reachability density vector...')

trainLrd = np.zeros(trainDist.shape[0])

for i in range(trainLrd.size):

  summation = 0

  for j in range(1, k + 1):

    summation += trainRd[i][trainNeighb[i][j]]

  trainLrd[i] = k / summation
print('>Done')
print('(7/12) Computing the training local outlier factor vector...')

trainLof = np.zeros(trainDist.shape[0])

for i in range(trainLof.size):

  summation = 0

  for j in range(1, k + 1):

    summation += trainLrd[trainNeighb[i][j]] / trainLrd[i]
  trainLof[i] = summation / k
trainLof = np.sort(trainLof)
print('>Done')

#compute strangeness test list:
print('(8/12) Computing the test reachability distance matrix...')
testRd = np.zeros(testDist.shape)

for i in range(testDist.shape[0]):

  for j in range(testDist.shape[1]):

    testRd[i][j] = max( testDist[i][testNeighb[i][k]], testDist[i][j] )
print('>Done')
print('(9/12) Computing test local reachability density vector...')
testLrd = np.zeros(testDist.shape[0])

for i in range(testLrd.size):

  summation = 0

  for j in range(1, k + 1):

    summation += testRd[i][testNeighb[i][j]]

  testLrd[i] = k / summation
print('>Done')
print('(10/12) Computing test local outlier factor matrix...')
testLof = np.zeros(testDist.shape[0])

for i in range(0, testLof.size):

  summation = 0

  for j in range(1, k + 1):

    summation += trainLrd[testNeighb[i][j]] / testLrd[i]

  testLof[i] = summation / k
print('>Done')
#compute p-values:
print('(11/12) Computing p-values...')
pValues = np.zeros(testLof.shape)

for i in range(pValues.size):

  pValues[i] = np.where(trainLof>=testLof[i])[0].size / ( trainLof.size + 1 )
print('>Done')
print('(12/12) Saving p-values...')
#save p-values as a file:

f = open('./p-values.txt', 'w+')

for i in range(pValues.size):

  f.write( str(pValues[i]) + '\n' )
f.close()
print('>Done')
print('>>Complete')
