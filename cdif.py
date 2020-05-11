
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib as mpl
import hyplog as log
import numpy as np
import tensorflow as tf

def PRI2(time, num, ii):
    if time.shape[0] < num:
        print("ERROR:Parameter 'testnum' is too large for arrival time recorded!")
        return
    else:
        time = time[0:num]
        timed = difference(time)
        timemode = stats.mode(timed)[0]
        for i in range(num - 1):
            if (timemode - ii)<timed[i] and timed[i]<(timemode + ii):
                time[i] = 0
                time[i + 1] = 0
        time = time.ravel()[np.flatnonzero(time)]
        timed = difference(time)
        timemode2 = stats.mode(timed)[0]
        return timemode[0], timemode2[0]

def PRI(time, num):
    if time.shape[0] < num:
        print("ERROR:Parameter 'testnum' is too large for arrival time recorded!")
        return
    else:
        time = time[0:num]
        # print(time.shape)
        timed = difference(time)
        timemode = stats.mode(timed)[0]
        return timemode

def difference(time):
    num = time.shape[0]
    timed = np.zeros(num - 1)
    for i in range(num - 1):
        timed[i] = time[i + 1] - time[i]
    return timed