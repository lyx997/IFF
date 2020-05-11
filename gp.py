# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 12:50:16 2019

@author: Eatapple
"""

import scipy.io as scio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import hyplog as log
import hypconv as convt
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN
import winsound
import os
import conv_n as vn
import caps_n as cn
import cdif

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
mpl.rcParams['font.sans-serif'] = [u'SimHei']
mpl.rcParams['axes.unicode_minus'] = False

trainstate= 0
ctrainstate = 0
loaddir = r'E:\gpproject\project\history\test2019_06_04_13_24_05\models'
cloaddir = r'E:\gpproject\project\history\test2019_06_04_10_54_08\cmodels'
epoch=501
cepoch = 501
dataFiles = 'E://gpproject//MATLAB//data.mat'

starttime = log.startlog()
log.pathcreate(trainstate,ctrainstate)

data = scio.loadmat(dataFiles)

testnum = 80
log.record("信号加载完成")
if trainstate:
    vn.train(epoch, testnum, data)

output3_value_embedded = vn.test(testnum, data, trainstate, loaddir)

########################################################################################################################
eps=0.3
ms=20

'''
eps： DBSCAN算法参数，即我们的ϵϵ-邻域的距离阈值，和样本距离超过ϵϵ的样本点不在ϵϵ-邻域内。
默认值是0.5.一般需要通过在多组值里面选择一个合适的阈值。
eps过大，则更多的点会落在核心对象的ϵϵ-邻域，此时我们的类别数可能会减少,本来不应该是一类的样本也会被划为一类。
反之则类别数可能会增大，本来是一类的样本却被划分开。
min_samples： DBSCAN算法参数，即样本点要成为核心对象所需要的ϵϵ-邻域的样本数阈值。
默认值是5. 一般需要通过在多组值里面选择一个合适的阈值。通常和eps一起调参。
在eps一定的情况下，min_samples过大，则核心对象会过少，此时簇内部分本来是一类的样本可能会被标为噪音点，
类别数也会变多。反之min_samples过小的话，则会产生大量的核心对象，可能会导致类别数过少。
'''
dbscan = DBSCAN(eps=eps, min_samples=ms)
dbscan.fit(output3_value_embedded)
label_pred = dbscan.labels_
# while(label_pred.max()!=4):
#     print(label_pred.max())
#     if label_pred.max()>4:
#         eps=eps*1.05
#         ms=ms+1
#         print(eps,ms)
#         dbscan = DBSCAN(eps=eps, min_samples=ms)
#         dbscan.fit(output3_value_embedded)
#         label_pred = dbscan.labels_
#     if label_pred.max()<4:
#         eps = eps * 0.95
#         if ms>2:
#             ms=ms-1
#         print(eps,ms)
#         dbscan = DBSCAN(eps=eps, min_samples=ms)
#         dbscan.fit(output3_value_embedded)
#         label_pred = dbscan.labels_
# print(label_pred.max())

points = TSNE(n_components=2, init='pca').fit_transform(output3_value_embedded)

xn = points[label_pred == -1]
x0 = points[label_pred == 0]
x1 = points[label_pred == 1]
x2 = points[label_pred == 2]
x3 = points[label_pred == 3]
x4 = points[label_pred == 4]
log.record("聚类完成")
########################################################################################################################
if ctrainstate:
    cn.train(cepoch, data)
predict_value_embedded = cn.test(testnum, data, xn, label_pred, ctrainstate, cloaddir)

mix = np.zeros([2, predict_value_embedded.shape[0]])
mix[0] = predict_value_embedded.argmax(axis=1)
for n in range(predict_value_embedded.shape[0]):
    predict_value_embedded[n][int(mix[0][n])] = predict_value_embedded[n][predict_value_embedded.argmin(axis=1)[n]]
mix[1] = predict_value_embedded.argmax(axis=1)
log.record("混叠信号处理完成")
########################################################################################################################
priout = np.zeros(6)
for i in range(4):
    exec("num = x"+str(i)+".shape[0]")
    priout[i] = cdif.PRI(data["time"][i], num)[0]
priout[4], priout[5] = cdif.PRI2(data["time"][4], x4.shape[0], 5)#testnum只是暂时测试使用，具体按照该聚类实际数目而定
# print(priout)
priout = 1e6//priout
log.record("脉冲重复周期为" + str(priout))
time = data["time"][4][0:x4.shape[0]]
a = max(label_pred)
for i in range(time.shape[0]):
    if time[i] == 0:
        label_pred[label_pred.shape[0]-time.shape[0]+i] = a + 1
# label_pred[time == 0] = max(label_pred) + 1
x41 = x4[time == 0]
x42 = x4[time > 0]
data = scio.loadmat(dataFiles)#data因为cdif部分数据被置零，需要重新加载
log.record("累计插值计算完成")
########################################################################################################################
errors = 0
bingorate = 0
for _count in range(5 * testnum):
    if label_pred[_count - 1] == -1:
        errors += 1
for _count in range(testnum):
    if label_pred[_count + 5 * testnum - 1] != -1:
        errors += 1
bingorate = 1 - errors / (6 * testnum)
color = ['red', 'green', 'blue', 'yellow', 'cyan', 'black', 'magenta']

fig1 = plt.figure('origion')
plt.scatter(points[testnum * 0:(testnum * 1 - 1), 0], points[testnum * 0:(testnum * 1 - 1), 1], c=color[0])  # 窄带线性调频
plt.scatter(points[testnum * 1:(testnum * 2 - 1), 0], points[testnum * 1:(testnum * 2 - 1), 1], c=color[1])  # 宽带线性调频
plt.scatter(points[testnum * 2:(testnum * 3 - 1), 0], points[testnum * 2:(testnum * 3 - 1), 1], c=color[2])  # 固定频率
plt.scatter(points[testnum * 3:(testnum * 4 - 1), 0], points[testnum * 3:(testnum * 4 - 1), 1], c=color[3])  # SFM
plt.scatter(points[testnum * 4:(testnum * 5 - 1), 0], points[testnum * 4:(testnum * 5 - 1), 1], c=color[4])  # 巴克码
for n in range(testnum):
    plt.scatter(points[testnum * 5 + n, 0], points[testnum * 5 + n, 1], c=color[data["mark"][0][n]-1], marker='o', s=20)
    plt.scatter(points[testnum * 5 + n, 0], points[testnum * 5 + n, 1], c=color[data["mark"][1][n]-1], marker='o', s=10)# 混叠
plt.savefig(log.picspath+"a原始信号")
plt.show()

fig2 = plt.figure('聚类')
plt.scatter(xn[:, 0], xn[:, 1], c=color[5], marker='x', label='label-1')
plt.scatter(x0[:, 0], x0[:, 1], c=color[0], marker='o', label='label0')
plt.scatter(x1[:, 0], x1[:, 1], c=color[1], marker='*', label='label1')
plt.scatter(x2[:, 0], x2[:, 1], c=color[2], marker='+', label='label2')
plt.scatter(x3[:, 0], x3[:, 1], c=color[3], marker='v', label='label3')
plt.scatter(x4[:, 0], x4[:, 1], c=color[4], marker='s', label='label4')
plt.savefig(log.picspath+"b聚类结果")
plt.show()

fig3 = plt.figure('混叠')
plt.scatter(x0[:, 0], x0[:, 1], c=color[0], marker='o', label='label0')
plt.scatter(x1[:, 0], x1[:, 1], c=color[1], marker='*', label='label1')
plt.scatter(x2[:, 0], x2[:, 1], c=color[2], marker='+', label='label2')
plt.scatter(x3[:, 0], x3[:, 1], c=color[3], marker='v', label='label3')
plt.scatter(x4[:, 0], x4[:, 1], c=color[4], marker='s', label='label4')
for n in range(predict_value_embedded.shape[0]):
    plt.scatter(xn[n, 0], xn[n, 1], c=color[int(mix[1][n])], marker='o', s=20, label='label-1')
    plt.scatter(xn[n, 0], xn[n, 1], c=color[int(mix[0][n])], marker='o', s=10, label='label-1')
plt.savefig(log.picspath+"c混叠信号处理结果")
plt.show()

fig4 = plt.figure('PRI')
plt.scatter(x0[:, 0], x0[:, 1], c=color[0], marker='o', label='label0')
plt.scatter(x1[:, 0], x1[:, 1], c=color[1], marker='*', label='label1')
plt.scatter(x2[:, 0], x2[:, 1], c=color[2], marker='+', label='label2')
plt.scatter(x3[:, 0], x3[:, 1], c=color[3], marker='v', label='label3')
plt.scatter(x41[:, 0], x41[:, 1], c=color[4], marker='s', label='label4')
plt.scatter(x42[:, 0], x42[:, 1], c=color[5], marker='s', label='label5')
for n in range(predict_value_embedded.shape[0]):
    plt.scatter(xn[n, 0], xn[n, 1], c=color[int(mix[1][n])], marker='o', s=20, label='label-1')
    plt.scatter(xn[n, 0], xn[n, 1], c=color[int(mix[0][n])], marker='o', s=10, label='label-1')
plt.savefig(log.picspath+"综合分析结果")
plt.show()

log.record("结果图已输出")
# print("信号聚类准确率:%.2f%%" % (bingorate * 100))
log.record("信号聚类准确率:" + str(bingorate * 100) +"%")

########################################################################################################################
mean = np.zeros([6])
for i in range(label_pred.shape[0]):
    _s = i//testnum + 1
    _sam = i%testnum + 1
    if label_pred[i] >= 0:
        mean[label_pred[i]] = mean[label_pred[i]] + np.abs(data['signal' + str(_s) + '_' + str(_sam)][0][0:129]).mean()
for i in range(mean.shape[0]):
    mean[i] = mean[i] / np.count_nonzero([label_pred == i])
# print(mean)
log.record("平均电压为" + str(mean))
########################################################################################################################
log.stoplog(starttime)
mp3 = r'E:\gpproject\python\陈雪凝 - 绿色.wav'
winsound.PlaySound(mp3, winsound.SND_ASYNC)
input('press<enter>')
winsound.PlaySound(None, winsound.SND_ASYNC)