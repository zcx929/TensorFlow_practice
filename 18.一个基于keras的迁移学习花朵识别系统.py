'''
Created on May 27 19:07 2020
@author :zcx
'''

'''
    使用的实验数据https://pan.baidu.com/s/1jIMOc1S
'''
import os
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import time
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import keras
from keras.models import Model
from keras.layers import Dense,Activation,Flatten,Dropout
from keras.utils import np_utils
from keras.applications.resnet50 import ResNet50

first_resize_photo = 0

# 提取数据标签
#   数据没有单独给出标签文件，需要我们自行通过文件夹提取每张图片的标签，建立标签文件
def tranverse_images(path):
    labels = pd.DataFrame()
    first_dir_file = [file for file in os.listdir(path)]
    for item in first_dir_file:
        flower = [image for image in os.listdir(path+item)]
        labels_data = pd.DataFrame({'flower':flower,'labels':item})
        labels = pd.concat((labels,labels_data))

    return labels

labels = tranverse_images('../flower_photos/')
# print(labels.head())



# 图片预处理（缩放）
#   通过实验可知每张图像像素大小并不一致，所以在搭建模型之前，我们需要对图片进行整体缩放为统一尺寸，我借助opencv实现图片缩放
#   因为我们的迁移学习策略采用的是ResNet50作为预训练模型，我们这里将图片缩放大小为224*224*3
def resize_image(path1,path2):
    images = []
    img_dir_file = [file for file in os.listdir(path1)]
    for item in img_dir_file:
        print('item',item)
        imges = [os.path.join(item,image) for image in os.listdir(path1 + item)]
        images = imges + images

    print(images)
    total = 0
    start_time = time.time()

    for img in images:
        img1 = cv2.imread(os.path.join(path1,img))
        img1 = cv2.resize(img1,(224,224))
        total += 1
        print('now is resizing {} image.'.format(total))
        img = img.split('\\')[-1]
        cv2.imwrite(os.path.join(path2,img),img1)
    print('all images are resized,all resized image is {}'.format(total))
    end_time = time.time()
    print('the resize time is {}'.format(end_time - start_time))

if first_resize_photo:
    resize_image(path1='../flower_photos/',path2='../resize_flower_photos/')



# 准备训练数据
#   处理好的图片无法直接拿来训练，我们需要将其转化为numpy数组的形式，另外标签也需要进一步的处理
#   将图片转化为数组
def image2array(labels,path):
    lst_imgs = [l for l in labels['flower']]
    return np.array([np.array(Image.open(path+img),np.float32) for img in lst_imgs])

X = image2array(labels,'../resize_flower_photos/')
print(X.shape)      # (3670,224,224,3)
# np.save('../X_train.npy',X)


# 然后我们再来处理标签数据，标签变量都是分类值，我们先需要将其进行硬编码 LabelEncoder,然后借助keras 将其 one-hot 处理
# lbl = LabelEncoder().fit(list(labels['labels'].values))
# labels['code_labels'] = pd.DataFrame(lbl.transform(list(labels['labels'].values)))
# print(labels.head())

# 然后我们再来处理标签数据，标签变量都是分类值，我们先需要将其进行硬编码 LabelEncoder,即将不同标签的图片用数字来代替
labelencoder = LabelEncoder()
labels['code_labels'] = labelencoder.fit_transform(labels['labels'])
# print(labels.tail())
# print(type(labels['code_labels'].values))     # ndarray 类型
# print(labels['code_labels'].values.shape)     # (3670,)
y = labels['code_labels'].values



# 训练数据的预处理准备妥当，下面使用sklearn 划分一下数据集
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=42)
# print(X_train.shape,X_test.shape,y_train.shape,y_test.shape)

X_train /= 255
X_test /= 255
# keras 内置方法转化为one-hot 编码
y_train = np_utils.to_categorical(y_train,5)
y_test = np_utils.to_categorical(y_test,5)
# print(X_train.shape,X_test.shape,y_train.shape,y_test.shape)

def flower_model(X_train,y_train):
    base_model = ResNet50(include_top=False,weights='imagenet',input_shape=(224,224,3))
    for layers in base_model.layers:
        layers.trainable = False

    model = Flatten()(base_model.output)
    model = Dense(128,activation='relu')(model)
    model = Dropout(0.5)(model)
    model = Dense(5,activation='softmax')(model)

    model = Model(inputs=base_model.input,outputs=model)
    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

    return model


model = flower_model(X_train,y_train)
print('[INFO] training model')
history = model.fit(X_train,y_train,batch_size=128,epochs=10)

print('[INFO] saving model and weights')
#保存模型（不含权重）
model_json=model.to_json()
open('model_architecture.json','w').write(model_json)
#保存权重
model.save_weights('keras_transfer_flower_weights.h5', overwrite=True)


model.evaluate(X_test,y_test,verbose=1)

model_id = np.int64(time.strftime('%Y%m%d%H%M', time.localtime(time.time())))
fig = plt.figure()#新建一张图
plt.plot(history.history['accuracy'],label='training acc')
plt.plot(history.history['val_accuracy'],label='val acc')
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(loc='lower right')
fig.savefig('ResNet'+str(model_id)+'acc.png')
fig = plt.figure()
plt.plot(history.history['loss'],label='training loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(loc='upper right')
fig.savefig('ResNet'+str(model_id)+'loss.png')












