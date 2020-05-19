import tensorflow as tf

state = tf.Variable(0,name='counter')
# print(state)
one = tf.constant(1)

new_value = tf.add(state,one)
# 把 new_value 加载到 state 上，现在 state 的状态为 new_value
update = tf.assign(state,new_value)

# 只要出现Variable都必须有初始化,都此时还没有激活初始化
init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)
    for _ in range(3):
        sess.run(update)
        # 直接print(state)是没有用的，必须要sess这个指针指向state run一下才行
        print(sess.run(state))











