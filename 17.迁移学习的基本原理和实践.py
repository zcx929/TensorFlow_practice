'''
Created on May 27 16:16 2020
@author :zcx
'''

'''
    在实际操作中使用迁移学习。
    keras 为我们提供了经典网络在ImageNet上为我们训练好的预训练模型，预训练模型的基本信息图片中查看
    
    这里以VGG16网络预训练为例对手写数字数据集mnist进行迁移学习任务，试验代码如下
'''
from keras.models import Model
from keras.layers import Dense,Flatten,Dropout
from keras import datasets
from keras.applications.vgg16 import VGG16
from keras.optimizers import SGD,Adam
from keras.datasets import mnist
import numpy as np
import cv2

# 查看VGG16 预训练模型的基本信息
model_vgg = VGG16(include_top=False,weights='imagenet',input_shape=(224,224,3))
# x = Flatten(name='Flatten')(model_vgg.output)
# x = Dense(10,activation='softmax')(x)
#
# model_vgg_mnist = Model(inputs=model_vgg.input,outputs=x,name='vgg16')
# model_vgg_mnist.summary()

# 冻结预训练模型的卷积和池化层，仅修改全连接层
for layers in model_vgg.layers:
    layers.trainable = False
    model = Flatten()(model_vgg.output)

model = Dense(10,activation='softmax')(model)
model_vgg_mnist_pretrain = Model(inputs=model_vgg.input,outputs=model,name='vgg16_pretrain')

sgd = SGD(lr=0.05,decay=1e-5)
model_vgg_mnist_pretrain.compile(optimizer=sgd,loss='categorical_crossentropy',
                                 metrics=['accuracy'])


(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
# 然后转换mnist训练数据的输入大小以适应 VGG16 的输入
X_train = [cv2.cvtColor(cv2.resize(i,(224,224)),cv2.COLOR_GRAY2BGR) for i in X_train]
X_train = np.concatenate([arr[np.newaxis] for arr in X_train]).astype('float32')

X_test = [cv2.cvtColor(cv2.resize(i,(224,224)),cv2.COLOR_GRAY2BGR) for i in X_test]
X_test = np.concatenate([arr[np.newaxis] for arr in X_test]).astype('float32')

X_train /= 255
X_test /= 255
np.where(X_train[0] != 0)

def train_y(y):
    y_one = np.zeros(10)
    y_one[y] = 1
    return y_one

y_train_one = np.array([train_y(Y_train[i]) for i in range(len(Y_train))])
y_test_one = np.array([train_y(Y_test[i]) for i in range(len(Y_test))])

model_vgg_mnist_pretrain.fit(X_train,y_train_one,validation_data=(X_test,y_test_one),
                             epochs=10,batch_size=128)








