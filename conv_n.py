import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import hyplog as log
import hypconv as convt
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN

x = tf.placeholder(tf.float32, [1, 300, 1])  # shape = [batch, in_width, in_channels]
label = tf.placeholder(tf.float32, [1])
# core1 = tf.placeholder(tf.float32,[7,1,1],name='x-input')#[filter_width, in_channels, out_channels]
core1 = tf.Variable(tf.random_normal([7, 1, 200]), dtype=tf.float32)
# core2 = tf.placeholder(tf.float32,[5,1,1],name='x-input')#[filter_width, in_channels, out_channels]
core2 = tf.Variable(tf.random_normal([5, 200, 100]), dtype=tf.float32)
core3 = tf.Variable(tf.random_normal([3, 100, 10]), dtype=tf.float32)
Weights = tf.Variable(tf.random_normal([10, 10]), dtype=tf.float32)
Weights2 = tf.Variable(tf.random_normal([10, 1]), dtype=tf.float32)
Weights3 = tf.Variable(tf.random_normal([10, 1]), dtype=tf.float32)
biases = tf.Variable(tf.zeros([10, 10]), dtype=tf.float32)
biases2 = tf.Variable(tf.zeros([10, 1]), dtype=tf.float32)
biases3 = tf.Variable(tf.zeros([10, 1]), dtype=tf.float32)
conv1d_1 = tf.nn.conv1d(x, core1, 1, data_format="NWC", padding='SAME')
h_conv1 = tf.nn.tanh(tf.nn.relu(conv1d_1))
h_pool1 = tf.nn.pool(h_conv1, window_shape=[5], strides=[5], pooling_type="MAX", padding='SAME')

conv1d_2 = tf.nn.conv1d(h_pool1, core2, 1, data_format="NWC", padding='SAME')
h_conv2 = tf.nn.tanh(tf.nn.relu(conv1d_2))
h_pool2 = tf.nn.pool(h_conv2, window_shape=[3], strides=[3], pooling_type="MAX", padding='SAME')

conv1d_3 = tf.nn.conv1d(h_pool2, core3, 1, data_format="NWC", padding='SAME')
h_conv3 = tf.nn.tanh(tf.nn.relu(conv1d_3))
h_pool3 = tf.nn.pool(h_conv3, window_shape=[2], strides=[2], pooling_type="MAX", padding='SAME')
# predict = s_aver(tf.matmul(tf.reshape(h_pool2,[10,10]),Weights) + biases)
output1 = tf.matmul(tf.reshape(h_pool3, [10, 10]), Weights) + biases
output2 = tf.nn.tanh(output1)
output3 = tf.reshape((tf.matmul(output2, Weights2) + biases2), [1, 10])
predict = tf.matmul(output3, Weights3) + biases3

# print(tf.shape(output3))
loss = tf.reduce_mean(tf.square(label - predict))
train_step = tf.train.GradientDescentOptimizer(0.0005).minimize(loss)

saver = tf.train.Saver(max_to_keep=1010)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True


def train(epoch, testnum, data):
    output3_value_embedded = np.zeros([6 * testnum, 10])
    interval = 20
    dbscan = DBSCAN(eps=0.5, min_samples=testnum / 2)
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        writer = tf.summary.FileWriter(log.path, sess.graph)
        for route in range(epoch):
            for _s in range(5):
                for _sam in range(100):
                    sig=convt.returnone(data['signal' + str(_s + 1) + '_' + str(_sam + 1)][0])
                    sign = sig.astype(np.float32).reshape(1, 300, 1)
                    sess.run(train_step, feed_dict={x: sign, label: [_s]})
            if route % interval == 0:
                print(route)
                saver.save(sess,log.modelspath+'testmodel.ckpt',global_step=route)
                for _s in range(6):
                     for _sam in range(testnum):
                        sig=convt.returnone(data['signal' + str(_s + 1) + '_' + str(_sam + 1)][0])
                        sign = sig.astype(np.float32).reshape(1, 300, 1)
                        output3_value = sess.run(output3, feed_dict={x: sign})
                        output3_value_embedded[_s * testnum + _sam] = output3_value
                dbscan.fit(output3_value_embedded)
                label_pred = dbscan.labels_
                errors = 0
                bingorate = 0
                for _count in range(5 * testnum):
                    if label_pred[_count - 1] == -1:
                        errors += 1
                for _count in range(testnum):
                    if label_pred[_count + 5 * testnum - 1] != -1:
                        errors += 1
                bingorate = 1 - errors / (6 * testnum)
                print("信号聚类准确率:" + str(bingorate * 100) + "%")
                points = TSNE(n_components=2, init='pca').fit_transform(output3_value_embedded)
                plt.figure(str(route/interval))
                plt.scatter(points[testnum * 0:(testnum * 1 - 1), 0], points[testnum * 0:(testnum * 1 - 1), 1],
                            c='red')  # 窄带线性调频
                plt.scatter(points[testnum * 1:(testnum * 2 - 1), 0], points[testnum * 1:(testnum * 2 - 1), 1],
                            c='green')  # 宽带线性调频
                plt.scatter(points[testnum * 2:(testnum * 3 - 1), 0], points[testnum * 2:(testnum * 3 - 1), 1],
                            c='blue')  # 固定频率
                plt.scatter(points[testnum * 3:(testnum * 4 - 1), 0], points[testnum * 3:(testnum * 4 - 1), 1],
                            c='yellow')  # SFM
                plt.scatter(points[testnum * 4:(testnum * 5 - 1), 0], points[testnum * 4:(testnum * 5 - 1), 1],
                            c='cyan')  # 巴克码
                plt.scatter(points[testnum * 5:(testnum * 6 - 1), 0], points[testnum * 5:(testnum * 6 - 1), 1],
                            c='black', marker='x')  # 混叠
                plt.savefig(log.picspath+'test' + str(route) + '.png')
                plt.show()
    print("CNN has trained for %d epochs."% (epoch))
    log.record("卷积神经网络已训练了" + str(epoch) + "个周期")
    return

def test(testnum, data, trainstate, loaddir=None):
    output3_value_embedded = np.zeros([6 * testnum, 10])
    points = np.zeros([6 * testnum, 2])
    with tf.Session(config=config) as sess:
        if trainstate:
            saver.restore(sess, tf.train.latest_checkpoint(log.modelspath))
        else:
            if loaddir == None:
                print("请输入卷积神经网络模型！！！")
                return
            else:
                saver.restore(sess, tf.train.latest_checkpoint(loaddir))
        for _s in range(6):
            for _sam in range(testnum):
                sig=convt.returnone(data['signal' + str(_s + 1) + '_' + str(_sam + 101)][0])
                sign = sig.astype(np.float32).reshape(1, 300, 1)
                output3_value = sess.run(output3, feed_dict={x: sign})
                output3_value_embedded[_s * testnum + _sam] = output3_value
    log.record("卷积神经网络测试完成")
    return output3_value_embedded