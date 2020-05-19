import tensorflow as tf
import numpy as np

# create data
# 生成100个随机数列    tf中的数据大部分以float32的形式存在
x_data = np.random.rand(100).astype(np.float32)
y_data = x_data * 0.1 + 0.3

## create tensorflow structure start ##
#                   tr.random_uniform 用随机数列的方式生成了权重参数
#                           一维，范围从-1.0 到 1.0
Weights = tf.Variable(tf.random_uniform([1],-1.0,1.0))
biaes = tf.Variable(tf.zeros([1]))

y = Weights * x_data + biaes

loss = tf.reduce_mean(tf.square(y-y_data))
# 建议一个优化器   0.5是学习速率
optimizer = tf.train.GradientDescentOptimizer(0.5)
# 优化器用来减小loss这个误差
train = optimizer.minimize(loss)

# 创建完上面的结构之后，这里的初始化才能让这个静态图活动起来
init = tf.initialize_all_variables()

## create tensorflow structure end ##

sess = tf.Session()
# sess就像一个指针，指向了处理的地方
sess.run(init)

for step in range(201):
    sess.run(train)
    if step % 20 == 0:
        print(step,sess.run(Weights),sess.run(biaes))



print('helloworld')











