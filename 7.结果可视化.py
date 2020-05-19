import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def add_layer(inputs,in_size,out_size,activation_function=None):
    Weights = tf.Variable(tf.random_normal([in_size,out_size]))
    biaes = tf.Variable(tf.zeros([1,out_size]) + 0.1)

    Wx_plus_b = tf.matmul(inputs,Weights) + biaes

    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs

# -1 到 1 的范围内有300个单位    加一个维度变成300行1列
x_data = np.linspace(-1,1,300)[:,np.newaxis]
#   加上一些噪点
noise = np.random.normal(0,0.05,x_data.shape)
y_data = np.square(x_data) - 0.5 + noise

# 我们想构造的网络的输入层和输出层都只有一个节点
xs = tf.placeholder(tf.float32,[None,1])
ys = tf.placeholder(tf.float32,[None,1])

# 输入层
layer1 = add_layer(xs,1,10,activation_function=tf.nn.relu)
# 输出层
prediction = add_layer(layer1,10,1,activation_function=None)

# tf.reduce_sum()对所有样本的求和  tf.reduce_mean()求平均值
#   reduction_indices 表示函数的处理维度
loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction),
                     reduction_indices=[1]))

train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)

    # 创建一个画板
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.scatter(x_data,y_data)
    # 连续的显示所需要的函数
    plt.ion()
    plt.show()

    for i in range(1000):
        sess.run(train_step,feed_dict={xs:x_data,ys:y_data})

        if i % 50 == 0:
            # print(sess.run(loss,feed_dict={xs:x_data,ys:y_data}))

            try:
                # 去掉上一次画的线
                ax.lines.remove(lines[0])
            except Exception:
                pass

            prediction_value = sess.run(prediction,feed_dict={xs:x_data})
            #                                   红色的线，宽度为5
            lines = ax.plot(x_data,prediction_value,'r-',lw=5)
            plt.pause(0.1)


















