'''
Created on May 20 14:37 2020
@author : zcx
'''

'''
简单实现LeNet-5网络
'''
import tensorflow as tf
import numpy as np
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# X_test = x_test.reshape(x_test.shape[0],28,28,1).astype('float32') / 255.0
# Y_test = tf.one_hot(y_test,10)
# print(X_test[:1,:,:,:])
# sess = tf.Session()
# a = sess.run(Y_test)
# print(a[:10,:])
# print(a.shape)
# raise IOError

def create_placeholder():
    # 创建输入输出的占位符变量模块
    # 论文中网络输入是32 * 32, 但我们使用的数据集 mnist 图像的输入为28 * 28 ,后面会使用 pad 填补成 32 * 32
    X = tf.placeholder(tf.float32,shape=(None,28*28))
    Y = tf.placeholder(tf.float32,shape=(None,10))
    keep_prob = tf.placeholder(tf.float32)
    return X,Y,keep_prob

def initialize_parameters():
    '''
    初始化各层参数
    :return:
    '''
    W1 = tf.get_variable('W1',[5,5,1,6],initializer=tf.contrib.layers.xavier_initializer())
    b1 = tf.get_variable('b1',[6],initializer=tf.zeros_initializer())

    W2 = tf.get_variable('W2',[5,5,6,16],initializer=tf.contrib.layers.xavier_initializer())
    b2 = tf.get_variable('b2',[16],initializer=tf.zeros_initializer())

    W3 = tf.get_variable('W3',[5,5,16,120],initializer=tf.contrib.layers.xavier_initializer())
    b3 = tf.get_variable('b3',[120],initializer=tf.zeros_initializer())

    W4 = tf.get_variable('W4',[120,84],initializer=tf.contrib.layers.xavier_initializer())
    b4 = tf.get_variable('b4',[84],initializer=tf.zeros_initializer())

    W5 = tf.get_variable('W5',[84,10],initializer=tf.contrib.layers.xavier_initializer())
    b5 = tf.get_variable('b5',[10],initializer=tf.zeros_initializer())

    para = {
        'W1':W1,
        'b1':b1,
        'W2':W2,
        'b2':b2,
        'W3':W3,
        'b3':b3,
        'W4':W4,
        'b4':b4,
        'W5':W5,
        'b5':b5,
    }
    return para

def forward_propagation(X,para,dropout):
    '''
    创建 LeNet-5的前向计算
    :param X:
    :param para:
    :param dropout:
    :return:
    '''
    #todo:这里的前向计算还需要仔细看看
    X = tf.reshape(X,[-1,28,28,1])
    X = tf.pad(X,[[0,0],[2,2],[2,2],[0,0]])

    c1 = tf.nn.conv2d(X,para['W1'],strides=[1,1,1,1],padding='VALID') + para['b1']
    p2 = tf.nn.max_pool(c1,ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID')

    c3 = tf.nn.conv2d(p2,para['W2'],strides=[1,1,1,1],padding='VALID') + para['b2']
    p4 = tf.nn.max_pool(c3,ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID')

    c5 = tf.nn.conv2d(p4,para['W3'],strides=[1,1,1,1],padding='VALID') + para['b3']
    c5 = tf.contrib.layers.flatten(c5)

    f6 = tf.nn.tanh(tf.add(tf.matmul(c5,para['W4']),para['b4']))
    f7 = tf.nn.tanh(tf.add(tf.matmul(f6,para['W5']),para['b5']))
    # tf.nn.dropout 一般用在全连接层
    f7 = tf.nn.dropout(f7,dropout)

    return f7



def lenet_model():
    '''
    创建模型优化计算函数
    :return:
    '''
    X,Y,keep_prob = create_placeholder()
    para = initialize_parameters()
    f7 = forward_propagation(X,para,keep_prob)
    prediction = tf.nn.softmax(f7)

    # 计算的是没有正则化的损失
    loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=f7,labels=Y))
    # 使用了l2正则化的损失
    l2_loss = tf.contrib.layers.apply_regularization(regularizer=tf.contrib.layers.l2_regularizer(l2_lambda),weights_list=tf.trainable_variables())
    final_loss = loss_op + l2_loss
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(final_loss)

    correct_pred = tf.equal(tf.argmax(prediction,1),tf.argmax(Y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred,tf.float32))
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        # todo:这里经过了修改，数据集不一样
        X_test = x_test.reshape(x_test.shape[0], 784).astype('float32') / 255.0
        Y_test = tf.one_hot(y_test, 10)
        Y_test_one_hot = sess.run(Y_test)

        # X_test = mnist.test.images[0:10000]
        # Y_test = mnist.test.images[0:10000]
        X_train = x_train.reshape(x_train.shape[0], 784).astype('float32') / 255.0
        Y_train = tf.one_hot(y_train, 10)
        Y_train_one_hot = sess.run(Y_train)

        for epoch in range(epoches):
            for step in range(1,num_steps+1):

                # batch_x,batch_y = mnist.train.next(batch_size)

                batch_x= X_train[batch_size*step:batch_size*step+batch_size,:]
                batch_y = Y_train_one_hot[batch_size*step:batch_size*step+batch_size,:]

                sess.run(train_op,feed_dict={X:batch_x,Y:batch_y,keep_prob:dropout})

                if step % display_step == 0 or step == 1:
                    pre,loss,acc = sess.run([prediction,loss_op,accuracy],feed_dict={X:batch_x,Y:batch_y,keep_prob:dropout})
                    print('Step ' + str(step) + \
                          ', Minibatch Loss= '+ '{:.4f}'.format(loss) + \
                          ', Training Accuracy= ' + '{:.3f}'.format(acc))

                if step % test_step == 0:
                    print('Test Step ' + str(step)+ ': Accuracy:',
                          sess.run(accuracy,feed_dict={X:X_test,Y:Y_test_one_hot,keep_prob:1.0}))



if __name__ == '__main__':

    batch_size = 128
    learning_rate = 0.001
    display_step = 10
    test_step = 150
    # 数据集不够大，不能设置10000 ,数据集只有60000张，batch_size 为 128 ,那么num_steps最大只能460左右
    epoches = 10
    num_steps = 460
    dropout = 0.5
    l2_lambda = 0.0001
    lenet_model()










