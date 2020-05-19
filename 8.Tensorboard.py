import tensorflow as tf
import numpy as np

def add_layer(inputs,in_size,out_size,n_layer,activation_function=None):
    layer_name = 'layer%s' % n_layer
    with tf.name_scope(layer_name):
        with tf.name_scope('weights'):
            Weights = tf.Variable(tf.random_normal([in_size,out_size]),name='W')
            tf.summary.histogram(layer_name+'/weights',Weights)
        with tf.name_scope('biaes'):
            biaes = tf.Variable(tf.zeros([1,out_size]) + 0.1,name='b')
            tf.summary.histogram(layer_name + '/biaes', biaes)
        with tf.name_scope('Wx_plus_b'):
            Wx_plus_b = tf.matmul(inputs,Weights) + biaes

        if activation_function is None:
            outputs = Wx_plus_b
        else:
            outputs = activation_function(Wx_plus_b)
        tf.summary.histogram(layer_name + '/outputs', outputs)
        return outputs

# -1 到 1 的范围内有300个单位    加一个维度变成300行1列
x_data = np.linspace(-1,1,300)[:,np.newaxis]
#   加上一些噪点
noise = np.random.normal(0,0.05,x_data.shape)
y_data = np.square(x_data) - 0.5 + noise

# 命名框架
with tf.name_scope('inputs'):
    xs = tf.placeholder(tf.float32,[None,1],name = 'x_input')
    ys = tf.placeholder(tf.float32,[None,1],name = 'y_input')

# 输入层
layer1 = add_layer(xs,1,10,n_layer=1,activation_function=tf.nn.relu)
# 输出层
prediction = add_layer(layer1,10,1,n_layer=2,activation_function=None)

# tf.reduce_sum()对所有样本的求和  tf.reduce_mean()求平均值
#   reduction_indices 表示函数的处理维度
with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction),
                         reduction_indices=[1]))
    # loss 的tensorboard 使用和上面的不太一样
    tf.summary.scalar('loss',loss)

with tf.name_scope('train'):
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)


init = tf.initialize_all_variables()

with tf.Session() as sess:

    # 把所有的summary合并到一起
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter('logs/', sess.graph)


    sess.run(init)
    for i in range(1000):
        sess.run(train_step,feed_dict={xs:x_data,ys:y_data})

        if i % 50 == 0:
            result = sess.run(merged,feed_dict={xs:x_data,ys:y_data})
            writer.add_summary(result,i)
            # print(sess.run(loss,feed_dict={xs:x_data,ys:y_data}))









