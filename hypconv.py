# import scipy.io as scio
# import matplotlib.pyplot as plt
# import matplotlib as mpl
# import hyplog as log
import numpy as np
import tensorflow as tf
def des():
    print('卷积层实验性函数')

def mod2(x):
    a = tf.square(x)
    b = tf.reduce_sum(a)
    return b

def squash(x):
    a = (mod2(x)**0.5)/(1+mod2(x))
    return a*x
def hypconv(signal,core):
    conv=np.convolve(signal,core,'valid')
    return conv
    
def maxpool(signal,n):
    total=signal.size
    num=total//n+1
    maxlist=np.zeros(num)
    for x in range(num):
        maxvalue=signal[x*n]
        for y in range(n):
            if x*n+y>=total:
                break
            if signal[x*n+y]>maxvalue:
                maxvalue=signal[x*n+y]
        maxlist[x]=maxvalue
    return maxlist
    
def Relu(signal):
    total=signal.size
    output=np.zeros(total)
    for x in range(total):
        if signal[x]<0:
            output[x]=0
        else:
            output[x]=signal[x]
    return output
    
def returnone(signal):
    total=signal.size
    differ=(max(signal)-min(signal))
    if differ==0:
        out=signal/max(signal)
    else:
        out=(signal-min(signal)*np.ones(total))/(max(signal)-min(signal))
    return out

def btoc(b):
    x = tf.reduce_sum(b) + 1e-10
    c = tf.exp(b)/x
    return c


def one_hot(labels, Label_class):
    if Label_class<labels[0]:
        print('One_hot is out of range.')
    else:
        one_hot_label = np.array([[int(i == int(labels[j])) for i in range(Label_class)] for j in range(len(labels))])
        return one_hot_label[0]

def softmax(x):
    mx = x - np.min(x) * np.ones(x.shape)
    # print(mx)
    x_exp = np.exp(mx)
    x_sum = np.sum(x_exp, keepdims = False)
    s = x_exp / x_sum
    return s