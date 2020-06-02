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
import time
import matplotlib.pyplot as plt
from keras.utils import np_utils

epochs = 1
ishape = 48

# 查看VGG16 预训练模型的基本信息
model_vgg = VGG16(include_top=False,weights='imagenet',input_shape=(ishape,ishape,3))
# x = Flatten(name='Flatten')(model_vgg.output)
# x = Dense(10,activation='softmax')(x)

# model_vgg_mnist = Model(inputs=model_vgg.input,outputs=x,name='vgg16')
# model_vgg_mnist.summary()

# 冻结预训练模型的卷积和池化层，仅修改全连接层
for layers in model_vgg.layers:
    layers.trainable = False

model = Flatten()(model_vgg.output)
model = Dense(10,activation='softmax')(model)
model_vgg_mnist_pretrain = Model(inputs=model_vgg.input,outputs=model,name='vgg16_pretrain')
model_vgg_mnist_pretrain.summary()

sgd = SGD(lr=0.05,decay=1e-5)
model_vgg_mnist_pretrain.compile(optimizer=sgd,loss='categorical_crossentropy',
                                 metrics=['accuracy'])


# 然后转换mnist训练数据的输入大小以适应 VGG16 的输入
print('[INFO] loading dataset......')
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

X_train = [cv2.cvtColor(cv2.resize(i,(ishape,ishape)),cv2.COLOR_GRAY2BGR) for i in X_train]
X_test = [cv2.cvtColor(cv2.resize(i,(ishape,ishape)),cv2.COLOR_GRAY2BGR) for i in X_test]
# np.newaxis 在使用和功能上等价于 None
X_train = np.concatenate([arr[np.newaxis] for arr in X_train]).astype('float32')
X_test = np.concatenate([arr[np.newaxis] for arr in X_test]).astype('float32')
# 归一化
X_train /= 255.0
X_test /= 255.0
# one-hot
Y_train = np_utils.to_categorical(Y_train,10)
Y_test = np_utils.to_categorical(Y_test,10)

# 模型拟合
print('[INFO] Fitting model......')
log = model_vgg_mnist_pretrain.fit(X_train,Y_train,validation_data=(X_test,Y_test),
                             epochs=epochs,batch_size=64)

# 模型评估
print('[INFO] Evaluating model......')
score = model_vgg_mnist_pretrain.evaluate(X_test, Y_test, verbose=0)

#保存权重
print('[INFO] Saving weights......')
model_vgg_mnist_pretrain.save_weights('transfer_learning_weights.h5', overwrite=True)


print('Test loss:', score[0])
print('Test accuracy:', score[1])
plt.figure('acc')
plt.subplot(2, 1, 1)
plt.plot(log.history['accuracy'], 'r--', label='Training Accuracy')
plt.plot(log.history['val_accuracy'], 'r-', label='Validation Accuracy')
plt.legend(loc='best')
plt.xlabel('Epochs')
plt.axis([0, epochs, 0.9, 1])
plt.figure('loss')
plt.subplot(2, 1, 2)
plt.plot(log.history['loss'], 'b--', label='Training Loss')
plt.plot(log.history['val_loss'], 'b-', label='Validation Loss')
plt.legend(loc='best')
plt.xlabel('Epochs')
plt.axis([0, epochs, 0, 1])
plt.show()


# 绘制训练过程曲线
# model_id = np.int64(time.strftime('%Y%m%d%H%M', time.localtime(time.time())))
# fig = plt.figure()#新建一张图
# plt.plot(history.history['accuracy'],label='training acc')
# plt.plot(history.history['val_accuracy'],label='val acc')
# plt.title('model accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.legend(loc='lower right')
# fig.savefig('VGG16'+str(model_id)+'acc.png')
# fig = plt.figure()
# plt.plot(history.history['loss'],label='training loss')
# plt.plot(history.history['val_loss'], label='val loss')
# plt.title('model loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(loc='upper right')
# fig.savefig('VGG16'+str(model_id)+'loss.png')






# import numpy as np
# import gc
#
# from keras.models import Sequential, Model
# from keras.layers import Input, Dense, Dropout, Flatten
# from keras.layers.convolutional import Conv2D, MaxPooling2D
# from keras.applications.vgg16 import VGG16
# from keras.optimizers import SGD
# import matplotlib.pyplot as plt
# import os
# from keras.datasets import mnist
# import cv2
# import h5py as h5py
#
#
# def tran_y(y):
#     y_ohe = np.zeros(10)
#     y_ohe[y] = 1
#     return y_ohe
#
#
# epochs = 1
#
# # 如果硬件配置较高，比如主机具备32GB以上内存，GPU具备8GB以上显存，可以适当增大这个值。VGG要求至少48像素
# ishape = 48
# (X_train, y_train), (X_test, y_test) = mnist.load_data()
#
# X_train = [cv2.cvtColor(cv2.resize(i, (ishape, ishape)), cv2.COLOR_GRAY2BGR) for i in X_train]
# X_train = np.concatenate([arr[np.newaxis] for arr in X_train]).astype('float32')
# X_train /= 255.0
#
# X_test = [cv2.cvtColor(cv2.resize(i, (ishape, ishape)), cv2.COLOR_GRAY2BGR) for i in X_test]
# X_test = np.concatenate([arr[np.newaxis] for arr in X_test]).astype('float32')
# X_test /= 255.0
#
# y_train_ohe = np.array([tran_y(y_train[i]) for i in range(len(y_train))])
# y_test_ohe = np.array([tran_y(y_test[i]) for i in range(len(y_test))])
# y_train_ohe = y_train_ohe.astype('float32')
# y_test_ohe = y_test_ohe.astype('float32')
#
# model_vgg = VGG16(include_top=False, weights='imagenet', input_shape=(ishape, ishape, 3))
# for layer in model_vgg.layers:
#     layer.trainable = False
# model = Flatten()(model_vgg.output)
# model = Dense(4096, activation='relu', name='fc1')(model)
# model = Dense(4096, activation='relu', name='fc2')(model)
# model = Dropout(0.5)(model)
# model = Dense(10, activation='softmax', name='prediction')(model)
# model_vgg_mnist_pretrain = Model(model_vgg.input, model, name='vgg16_pretrain')
# model_vgg_mnist_pretrain.summary()
# sgd = SGD(lr=0.05, decay=1e-5)
# model_vgg_mnist_pretrain.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
# log = model_vgg_mnist_pretrain.fit(X_train, y_train_ohe, validation_data=(X_test, y_test_ohe), epochs=epochs,
#                                    batch_size=64)
#
# score = model_vgg_mnist_pretrain.evaluate(X_test, y_test_ohe, verbose=0)
# print('Test loss:', score[0])
# print('Test accuracy:', score[1])
#
# plt.figure('acc')
# plt.subplot(2, 1, 1)
# plt.plot(log.history['accuracy'], 'r--', label='Training Accuracy')
# plt.plot(log.history['val_accuracy'], 'r-', label='Validation Accuracy')
# plt.legend(loc='best')
# plt.xlabel('Epochs')
# plt.axis([0, epochs, 0.9, 1])
# plt.figure('loss')
# plt.subplot(2, 1, 2)
# plt.plot(log.history['loss'], 'b--', label='Training Loss')
# plt.plot(log.history['val_loss'], 'b-', label='Validation Loss')
# plt.legend(loc='best')
# plt.xlabel('Epochs')
# plt.axis([0, epochs, 0, 1])
#
# plt.show()
# os.system("pause")

















