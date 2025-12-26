# understand linear regression and see if you can create any functions for that
# understand random forest algorithm and see if you can create any functions for that
# understand gradient boosting algorithm and see if you can create any functions for that
import math
import numpy as np


def get_mae(y_testing, y_prediction):
    return np.mean(np.abs(y_testing - y_prediction))

def get_mae2(y_testing, y_prediction):
    return np.sum(np.abs(y_testing - y_prediction))/len(y_testing)

def get_mae3(y_testing, y_prediction):
    y_testing = np.asarray(y_testing)
    y_prediction = np.asarray(y_prediction)

    if len(y_testing) != len(y_prediction):
        raise  ValueError("Number of testing samples does not match number of prediction samples.")

    length = len(y_testing)

    if length == 0:
        return 0

    total_sum = 0

    for i in range(length):
        total_sum += math.abs(y_testing[i] - y_prediction[i])

    return total_sum/length

def get_r2_score(y_testing, y_prediction):
    y_testing = np.asarray(y_testing)
    y_prediction = np.asarray(y_prediction)

    if len(y_testing) != len(y_prediction):
        raise ValueError("Number of testing samples does not match number of prediction samples.")

    if len(y_testing) == 0:
        return 0

    sse = np.sum((y_testing - y_prediction)**2)
    sst = np.sum((y_testing - np.mean(y_testing))**2)

    if sst == 0:
        return 0.0

    r2_score = 1 - sse / sst
    return r2_score