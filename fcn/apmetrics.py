from sklearn.metrics import average_precision_score
import numpy as np

class APMetrics(object):
    def __init__(self):
        self.y_true = []
        self.y_score = []
        return
    def get_ap(self):
        y_true = np.concatenate(self.y_true, axis = 0).ravel()
        y_score = np.concatenate(self.y_score, axis = 0).ravel()
        return average_precision_score(y_true, y_score)
        