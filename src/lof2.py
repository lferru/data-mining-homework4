'''
Luis Ferrufino
G#00997076
HW#4
4/15/20
lof.py
'''

import random as rnd
import numpy as np
import matplotlib.pyplot as plt

dist = np.load('./distMatrix.npy')
kRange = range(1, 81)
avgAuc = np.zeros(len(kRange))
numRuns = 100 # number of trials we run each test for k

for k in kRange:

  for j in range(1, numRuns + 1):

    #create two random sets of indeces
    trainIndeces = []
    for i in range(50):
      indexA = rnd.randint(0, 99)
      while trainIndeces.count(indexA) > 0:
        indexA = rnd.randint(0, 99)
      indexB = rnd.randint(100, 199)
      while trainIndeces.count(indexB) > 0:
        indexB = rnd.randint(100, 199)
      indexC = rnd.randint(200, 299)
      while trainIndeces.count(indexC) > 0:
        indexC = rnd.randint(200, 299)
      indexD = rnd.randint(300, 398)
      while trainIndeces.count(indexD) > 0:
        indexD = rnd.randint(300, 398)

      trainIndeces.append(indexA)
      trainIndeces.append(indexB)
      trainIndeces.append(indexC)
      trainIndeces.append(indexD)

    testIndeces = []

    for i in range(36):
      indexA = rnd.randint(0, 99)
      while trainIndeces.count(indexA) > 0 or testIndeces.count(indexA) > 0:
        indexA = rnd.randint(0, 99)
      indexB = rnd.randint(100, 199)
      while trainIndeces.count(indexB) > 0 or testIndeces.count(indexB) > 0:
        indexB = rnd.randint(100, 199)
      indexC = rnd.randint(200, 299)
      while trainIndeces.count(indexC) > 0 or testIndeces.count(indexC) > 0:
        indexC = rnd.randint(200, 299)
      indexD = rnd.randint(300, 398)
      while trainIndeces.count(indexD) > 0 or testIndeces.count(indexD) > 0:
        indexD = rnd.randint(300, 398)
      indexM = rnd.randint(399, 497)
      while trainIndeces.count(indexM) > 0 or testIndeces.count(indexM) > 0:
        indexM = rnd.randint(399, 497)

      testIndeces.append(indexA)
      testIndeces.append(indexB)
      testIndeces.append(indexC)
      testIndeces.append(indexD)
      testIndeces.append(indexM)

    #use these to create a new training set and test set:
    notTests = [x for x in list(range(0,498)) if x not in testIndeces]
    notTrains = [x for x in list(range(0,498)) if x not in trainIndeces]

    testDist = np.delete(dist, notTests, 0)
    testDist = np.delete(testDist, notTrains, 1)
    trainDist = np.delete(dist, notTrains, 0)
    trainDist = np.delete(trainDist, notTrains, 1)

    #now create the neighbours matrix for each set:

    trainNeighb = np.zeros(trainDist.shape)
    
    for i in range(0, trainNeighb.shape[0]):

      temp = np.argsort(trainDist[i])

      for l in range(0, temp.size):
        
        trainNeighb[i][l] = trainIndeces[temp[l]]
      
    testNeighb = np.zeros(testDist.shape)

    for i in range(0, testNeighb.shape[0]):

      temp = np.argsort(testDist[i])

      for l in range(0, temp.size):

        testNeighb[i][l] = trainIndeces[temp[l]]

    #now create the strangeness training list by calculating the LOF between points in the training set:

    trainRd = np.zeros(trainDist.shape) #the reachability distances between points

    for i in range(0, trainDist.shape[0]):

      for l in range(0, trainDist.shape[1]):

        trainRd[i][l] = max( dist[trainIndeces[i]][int(trainNeighb[i][k])], dist[trainIndeces[i]][trainIndeces[l]] )

    trainLrd = np.zeros(trainDist.shape[0]) # the lrd (local reachability density) for each point

    for i in range(0, trainLrd.size):

      summation = 0

      for l in range(1, k + 1):
 
        summation += trainRd[i][trainIndeces.index(int(trainNeighb[i][l]))]

      trainLrd[i] = k / summation

    trainLof = np.zeros(trainDist.shape[0])

    for i in range(0, trainLof.size):

      summation = 0

      for l in range(1, k + 1):

        summation += trainLrd[trainIndeces.index(int(trainNeighb[i][l]))] / trainLrd[i]

      trainLof[i] = summation / k
    
    trainLof = np.sort(trainLof) #this is now my strangeness training list

    #in a very similar manner, we compute the strangeness test list:

    testRd = np.zeros(testDist.shape) #reach-dist

    for i in range(0, testDist.shape[0]):

      for l in range(0, testDist.shape[1]):

        testRd[i][l] = max( dist[testIndeces[i]][int(testNeighb[i][k])], dist[testIndeces[i]][trainIndeces[l]] )

    testLrd = np.zeros(testDist.shape[0]) #lrd

    for i in range(0, testLrd.size):

      summation = 0

      for l in range(1, k + 1):

        summation += testRd[i][trainIndeces.index(int(testNeighb[i][l]))]

      testLrd[i] = k / summation

    testLof = np.zeros(testDist.shape[0]) #WON'T be sorted; will become my strangeness test list

    for i in range(0, testLof.size):

      summation = 0

      for l in range(1, k + 1):

        summation += trainLrd[trainIndeces.index(int(testNeighb[i][l]))] / testLrd[i]
      
      testLof[i] = summation / k

    #next we calculate the p-values:

    pValues = np.zeros(testLof.shape)

    for i in range(0, pValues.size):

      pValues[i] = np.where(trainLof>=testLof[i])[0].size / ( trainLof.size + 1 ) # b / ( N + 1 )
    
    #finally, we compute the ROC's AUC:
    
    clDelta = 0.05 #intervals for confidence level
    numIntervals = int(1 / clDelta + 1)
    cl = 0
    tpr = np.zeros(numIntervals) # y-coördinates (true positive rate)
    fpr = np.zeros(numIntervals) # x-coördinates (false positive rate)
    labels = np.load('./labelVector.npy')
    normalLabels = ['A', 'B', 'C', 'D']

    for i in range(numIntervals):
    
      predictions = np.zeros(pValues.shape, '<U1')

      for l in range(pValues.size):
      
        predictions[l] = 'M' if pValues[l] < ( 1 - cl ) else 'N' # N for normal
      tp = 0 #true positive
      fp = 0 #false positive
      tn = 0 #true negative
      fn = 0 #false negative

      for l in range(pValues.size): #here, an anomaly is a positive, and a normal point is a negative

        if ( normalLabels.count(labels[testIndeces[l]]) > 0 and predictions[l] == 'N' ):
          tn += 1
        elif ( labels[testIndeces[l]] == 'M' and predictions[l] == 'M' ):
          tp += 1
        elif ( normalLabels.count(labels[testIndeces[l]]) > 0 and predictions[l] == 'M' ):
          fp += 1
        else:
          fn += 1

      tpr[i] = tp / ( tp + fn )
      fpr[i] = fp / ( tn + fp )

      cl += clDelta
    
    order = np.argsort(fpr)
    tpr = tpr[order]
    fpr = fpr[order]
    auc = np.trapz(y=tpr, x=fpr)
    print('Pass: k=' + str(k) + ' , j=' + str(j) + ' with ' + str(auc))
    avgAuc[k - 1] += auc
  avgAuc[k - 1] /= numRuns
  print('*During k=' + str(k) + ', the average auc was ' + str(avgAuc[k - 1]))

print('**The best k values were ' + str(np.argsort(avgAuc) + 1))

#generate a graph:

plt.plot(list(range(1,81)), avgAuc.tolist()) 
plt.ylabel('AUC')
plt.xlabel('K-VALUES')
plt.show()
