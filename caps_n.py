import numpy as np
import tensorflow as tf
import hyplog as log
import hypconv as convt
import math


x = tf.placeholder(tf.float32, [1, 300, 1])  # shape = [batch, in_width, in_channels]
label = tf.placeholder(tf.float32, [5, ])
core1 = tf.Variable(tf.random_normal([7, 1, 200]), dtype=tf.float32)
# core2 = tf.placeholder(tf.float32,[5,1,1],name='x-input')#[filter_width, in_channels, out_channels]

core2 = tf.Variable(tf.random_normal([5, 200, 20]), dtype=tf.float32)
W = tf.Variable(tf.random_normal([20, 6, 10, 10]), dtype=tf.float32)
C = tf.Variable(tf.random_normal([6, 20, 1]), dtype=tf.float32)
B = tf.Variable(tf.zeros([6, 20, 1]), dtype=tf.float32)
Weights = tf.Variable(tf.random_normal([6, 1, 1]), dtype=tf.float32)
conv1d_1 = tf.nn.conv1d(x, core1, 1, data_format="NWC", padding='SAME')
h_conv1 = tf.nn.tanh(tf.sigmoid(conv1d_1))
h_pool1 = tf.nn.pool(h_conv1, window_shape=[5], strides=[5], pooling_type="MAX", padding='SAME')

conv1d_2 = tf.nn.conv1d(h_pool1, core2, 1, data_format="NWC", padding='SAME')
h_conv2 = tf.nn.tanh(tf.sigmoid(conv1d_2))
h_pool2 = tf.nn.pool(h_conv2, window_shape=[6], strides=[6], pooling_type="MAX", padding='SAME')
input = tf.tile(tf.reshape(h_pool2, [20, 1, 1, 10]), [1, 6, 1, 1])
# ch_pool2(1,length,number)
# # cU(number,1,length)
U = tf.reshape(tf.matmul(input, W), [6, 20, 10])
for r in range(5):
    S = tf.reshape(tf.reduce_sum(tf.multiply(tf.tile(C, [1, 1, 10]),U), 1),[6, 10, 1])
    V = convt.squash(S)
    B = B + tf.matmul(U,V)
    C = convt.btoc(B)
S = tf.reshape(tf.reduce_sum(tf.multiply(tf.tile(C, [1, 1, 10]), U), 1), [6, 10, 1])
V = convt.squash(S)
output = tf.reshape(tf.reduce_sum(tf.matmul(V, Weights), [0, 2]), [2, 5])
output1 = tf.slice(output, [0, 0], [1, 5])
output2 = tf.slice(output, [1, 0], [1, 5])
predict = output1 + output2
aloss = tf.reduce_sum(tf.square(label - predict))
loss = tf.reduce_sum(tf.sigmoid(tf.square(label - predict)))

# loss = tf.log(tf.clip_by_value(aloss, 1e-8, tf.reduce_max(aloss)))
# train_step = tf.train.GradientDescentOptimizer(0.00005).minimize(aloss)
# train_step = tf.train.AdamOptimizer(0.0001).minimize(loss)
# train_step = tf.train.AdagradOptimizer(0.000001).minimize(loss)
# train_step = tf.train.AdadeltaOptimizer(0.000001).minimize(loss)
train_step = tf.train.MomentumOptimizer(0.0001, 0.5).minimize(loss)

saver = tf.train.Saver(max_to_keep=1010)
config = tf.ConfigProto()
config.gpu_options.allow_growth = True


def train(epoch, data):
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        writer = tf.summary.FileWriter(log.path, sess.graph)
        for route in range(epoch):
            for _sam in range(200):
                sig = convt.returnone(data['signal6_' + str(_sam + 1)][0])
                sign = sig.astype(np.float32).reshape(1, 300, 1)
                labelofc = 0.5 * convt.one_hot([data["mark"][0][_sam]], 5) + 0.5 * convt.one_hot(
                    [data["mark"][1][_sam]], 5)
                sess.run(train_step, feed_dict={x: sign, label: labelofc})
            if route % 20 == 0:
                loss_value, predict_value, Weights_value = sess.run([aloss,loss ,input], feed_dict={x: sign, label: labelofc})
                if math.isnan(loss_value):
                    print("胶囊神经网络训练失败：梯度爆炸，请调整学习率并使用合适的优化器")
                    log.record("胶囊神经网络训练失败：梯度爆炸，请调整学习率并使用合适的优化器")
                    print("胶囊神经网络已训练了" + str(epoch) + "个周期")
                    log.record("胶囊神经网络已训练了" + str(epoch) + "个周期")
                    return
                print(route,loss_value)
                # print(Weights_value)
                saver.save(sess, log.cmodelspath + 'ctestmodel.ckpt', global_step=route)
    print("Capsules-net has trained for %d epochs."% (epoch))
    log.record("胶囊神经网络已训练了" + str(epoch) + "个周期")
    return


def test(testnum, data, xn, label_pred, trainstate, loaddir=None):
    predict_value_embedded = np.zeros([xn.size // 2, 5])
    with tf.Session(config=config) as sess:
        if trainstate:
            saver.restore(sess, tf.train.latest_checkpoint(log.cmodelspath))
        else:
            if loaddir == None:
                print("请输入胶囊神经网络模型！！！")
                return
            else:
                saver.restore(sess,tf.train.latest_checkpoint(loaddir))
        sorter = 0
        for _s in range(6):
            for _sam in range(testnum):
                counter = _s * testnum + _sam + 1
                if label_pred[counter - 1] == -1:
                    sorter = sorter + 1
                    sig = convt.returnone(data['signal' + str(_s + 1) + '_' + str(_sam + 1)][0])
                    sign = sig.astype(np.float32).reshape(1, 300, 1)
                    predict_value = sess.run(predict, feed_dict={x: sign})
                    predict_value_embedded[sorter - 1] = convt.softmax(predict_value[0])
    log.record("胶囊神经网络测试完成")
    return predict_value_embedded