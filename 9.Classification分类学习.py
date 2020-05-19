'''
出了点问题，loss一直是nan 为什么

原因其实在于交叉熵中的 y_truth * log(y_predict)
log(0) * 0的时候， 则会出现NaN，
一旦出现这个情况，网络训练结果必然完蛋
解决办法其实也很简单：
在交叉熵的公式中加入一个比较小的数：1e-10
'''


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 加载手写体数据集
mint = tf.keras.datasets.mnist
# x_.shape :(60000, 28, 28) y_.shape(60000,)
(x_,y_),(x_1,y_1) = mint.load_data()
# 更改维度形状以适应数据输入
#   训练集
x = x_.reshape(-1,784)
y = tf.one_hot(y_,10)
#   测试集
x_test = x_1.reshape(-1,784)
y_test = tf.one_hot(y_1,10)

# x,y = x_[100:105,:],y_[100:105,:]
# sess = tf.Session()
# a = sess.run(y_)
# print(a[:10,:])

def add_layer(inputs,in_size,out_size,activation_function=None):
    Weights = tf.Variable(tf.random_normal([in_size,out_size]))
    biaes = tf.Variable(tf.zeros([1,out_size]) + 0.1)

    Wx_plus_b = tf.matmul(inputs,Weights) + biaes

    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs

def compute_accuracy(v_xs,v_ys):
    global prediction
    y_pre = sess.run(prediction,feed_dict={xs:v_xs})
    correct_prediction = tf.equal(tf.argmax(y_pre,1),tf.argmax(v_ys,1))
    # tf.cast 张量数据类型转换
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    result = sess.run(accuracy,feed_dict={xs:v_xs,ys:v_ys})
    return result

# 28 * 28 = 784
xs = tf.placeholder(tf.float32,[None,784])
ys = tf.placeholder(tf.float32,[None,10])

# add output layer
layer1 = add_layer(xs,784,20,activation_function=tf.nn.relu)
layer2 = add_layer(layer1,20,20,activation_function=tf.nn.relu)
prediction = add_layer(layer2,20,10,activation_function=tf.nn.softmax)

# loss
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction+1e-8),
                                              reduction_indices=[1]))
train_step = tf.train.AdamOptimizer().minimize(cross_entropy)

sess = tf.Session()
sess.run(tf.initialize_all_variables())


for epoch in range(10):
    for i in range(1000):
        y_one_hot1 = sess.run(y)

        batch_xs,batch_ys = x[i*60:60 * i + 60,:],y_one_hot1[i*60:60*i + 60,:]
        sess.run(train_step,feed_dict={xs:batch_xs,ys:batch_ys})
        # print(sess.run(cross_entropy,feed_dict={xs:batch_xs,ys:batch_ys}))

        if i % 500 == 0:
            y_one_hot2 = sess.run(y_test)
            print(compute_accuracy(
                x_test[:, :], y_one_hot2[:, :]
            ))







