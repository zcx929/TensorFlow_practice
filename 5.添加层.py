import tensorflow as tf

def add_layer(inputs,in_size,out_size,activation_function=None):
    Weights = tf.Variable(tf.random_normal([in_size,out_size]))
    biaes = tf.Variable(tf.zeros([1,out_size]) + 0.1)

    Wx_plus_b = tf.matmul(inputs,Weights) + biaes

    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs









