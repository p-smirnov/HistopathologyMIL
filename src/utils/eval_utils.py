import numpy as np

## https://stackoverflow.com/questions/2148543/how-to-write-a-confusion-matrix
def compute_confusion_matrix(true, pred):
  '''Computes a confusion matrix using numpy for two np.arrays
  true and pred.
  Results are identical (and similar in computation time) to: 
  "from sklearn.metrics import confusion_matrix"
  However, this function avoids the dependency on sklearn.'''
  K = len(np.unique(true)) # Number of classes 
  result = np.zeros((K, K))
  for i in range(len(true)):
    result[true[i]][pred[i]] += 1
  return result
