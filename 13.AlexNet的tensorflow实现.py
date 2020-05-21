'''
Created on May 21 16:30 2020
@author :zcx
'''
import tensorflow as tf
import numpy as np

'''
基本步骤和方法：
    1.定义创建输入输出的占位符变量模块
    2.初始化各层参数模块
    3.创建前向传播模块
    4.定义模型优化迭代模型
    5.最后设置输入数据
'''

def conv(x,filter_height,filter_width,num_filters,stride_y,stride_x,name,padding='SAME',groups=1):
    '''
    定义卷积过程
    '''

    # 获取输入通道数
    input_channels = int(x.get_shape()[-1])

    # 为卷积创建lambda函数
    convolve = lambda i,k: tf.nn.conv2d(i,k,strides=[1,stride_y,stride_x,1],
                                        padding=padding)

    with tf.variable_scope(name) as scope:
        # 为卷积层的权重和偏置创建 tf 变量
        weights = tf.get_variable('weights',shape=[filter_height,
                                                   filter_width,
                                                   input_channels/groups,
                                                   num_filters])
        biases = tf.get_variable('biases',shape=[num_filters])

    if groups == 1:
        conv = convolve(x,weights)
        # 在多组的情况下，分割输入和权重
    else:
        #todo:???
        # 将输入和权重分开并分别卷积
        input_groups = tf.split(axis=3,num_or_size_splits=groups,value=x)
        weight_group = tf.split(axis=3,num_or_size_splits=groups,value=weights)

        output_groups = [convolve(i,k) for i,k in zip(input_groups,weight_group)]
        # 再次将卷积输出连接到一起
        conv = tf.concat(axis=3,values=output_groups)

    # 增加偏置
    bias = tf.reshape(tf.nn.bias_add(conv,biases),tf.shape(conv))

    # relu 激活
    relu_result = tf.nn.relu(bias,name=scope.name)

    return relu_result

def fc(x,num_in,num_out,name,relu=True):
    '''定义全连接层'''
    with tf.variable_scope(name) as scope:
        #todo ???
        weights = tf.get_variable('weights',shape=[num_in,num_out],trainable=True)
        biases = tf.get_variable('biases',shape=[num_out],trainable=True)

        act = tf.nn.xw_plus_b(x,weights,biases,name=scope.name)
        if relu:
            relu = tf.nn.relu(act)
            return relu
        else:
            return act

def max_pool(x,filter_height,filter_width,stride_y,stride_x,name,padding='SAME'):
    '''定义最大池化过程'''
    return tf.nn.max_pool(x,ksize=[1,filter_height,filter_width,1],strides=[1,stride_y,stride_x,1],padding=padding,name=name)

def lrn(x,radius,alpha,beta,name,bias=1.0):
    '''定义LRN'''
    return tf.nn.local_response_normalization(x,depth_radius=radius,alpha=alpha,beta=beta,bias=bias,name=name)

def dropout(x,keep_prob):
    '''定义dropout操作'''
    return tf.nn.dropout(x,keep_prob)


# 以上关于搭建AlexNet的各个组件我们都已准备好，下面我们利用这些组件创建一个AlexNet类来实现AlexNet
class AlexNet(object):
    def __init__(self,x,keep_prob,num_classes,skip_layer,weights_path='DEFAULT'):
        # 将输入参数解析为类变量
        self.X = x
        self.NUM_CLASSES = num_classes
        self.KEEP_PROB = keep_prob
        self.SKIP_LAYER = skip_layer

        if weights_path == 'DEFAULT':
            self.WEIGHTS_PATH = 'bvlc_alexnet.npy'
        else:
            self.WEIGHTS_PATH = weights_path

        def create(self):
            # 第一层 ：conv(w ReLu) -> Lrn -> Pool
            conv1 = conv(self.X,11,11,96,4,4,padding='VALID',name='conv1')
            norm1 = lrn(conv1,2,1e-04,0.75,name='norm1')
            pool1 = max_pool(norm1,3,3,2,2,padding='VALID',name='pool1')

            # 第二层 ：conv(w ReLu) -> Lrn -> Pool with 2 groups
            conv2 = conv(pool1,5,5,256,1,1,groups=2,name='conv2')
            norm2 = lrn(conv2,2,1e-04,0.75,name='norm2')
            pool2 = max_pool(norm2,3,3,2,2,padding='VALID',name='pool2')

            # 第三层：conv(w ReLu)
            conv3 = conv(pool2,3,3,384,1,1,name='conv3')

            # 第四层：conv(w Relu) 分成两组
            conv4 = conv(conv3,3,3,384,1,1,groups=2,name='conv4')

            # 第五层：conv(w ReLu) -> Pool 分成两组
            conv5 = conv(conv4,3,3,256,1,1,groups=2,name='conv5')
            pool5 = max_pool(conv5,3,3,2,2,padding='VALID',name='pool5')

            # 第六层：Flatten -> FC (w ReLu) -> Dropout
            flattened = tf.reshape(pool5,[-1,6*6*256])
            fc6 = fc(flattened,6*6*256,4096,name='fc6')
            dropout6 = dropout(fc6,self.KEEP_PROB)

            # 第7层：FC(w ReLu) -> Dropout
            fc7 = fc(dropout6,4096,4096,name='fc7')
            dropout7 = dropout(fc7,self.KEEP_PROB)

            # 第8层：FC and 返回未标记的激活
            self.fc8 = fc(dropout7,4096,self.NUM_CLASSES,relu=False,name='fc8')


        def load_initial_weights(self,session):

            # 加载权重
            weights_dict = np.load(self.WEIGHTS_PATH,encoding='bytes').item()

            # 在weights_dict中存储的所有图层名上循环
            for op_name in weights_dict:
                # 检查层是否需要从头开始训练
                if op_name not in self.SKIP_LAYER:
                    with tf.variable_scope(op_name,reuse=True):

                        # 为相应的tf变量分配权重/偏差
                        for data in weights_dict[op_name]:

                            # 偏差
                            if len(data.shape == 1):
                                var = tf.get_variable('biases',trainable=False)
                                session.run(var.assign(data))

                            # 权重
                            else:
                                var = tf.get_variable('weights',trainable=False)
                                session.run(var.assign(data))

'''
    在上述代码中，我们利用了之前定义的各个组件封装了前向计算过程，从http://www.cs.toronto.edu/~guerzhoy/tf_alexnet/上导入
预训练好的模型权重，这样一来，我们就将AlexNet基本搭建好了
'''

#todo：未完待续，输入什么数据集试试好呢？

















