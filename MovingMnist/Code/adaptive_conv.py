import tensorflow as tf

def kernel_G_extract(input, name = None):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        print 'en1',input.get_shape() # should be (self.batch_size, 64, 64 3)
        output1 = tf.layers.conv2d(inputs = input, filters = 4, kernel_size = (5,5), strides=(2, 2), padding='same', use_bias=False, kernel_initializer=tf.glorot_uniform_initializer(),  name = 'c1')
        output3 = tf.nn.leaky_relu(output1, alpha=0.2, name='l1')
        print 'en2',output3.get_shape()

        output4 = tf.layers.conv2d(inputs = output3, filters = 8, kernel_size = (5,5), strides=(2, 2), padding='same', use_bias=False, kernel_initializer=tf.glorot_uniform_initializer(),  name = 'c2')
        output6 = tf.nn.leaky_relu(output4, alpha=0.2, name='l2')
        print 'en3',output6.get_shape()

        output7 = tf.layers.conv2d(inputs = output6, filters = 16, kernel_size = (5,5), strides=(2, 2), padding='same', use_bias=False, kernel_initializer=tf.glorot_uniform_initializer(),  name = 'c3')
        output9 = tf.nn.leaky_relu(output7, alpha=0.2, name='l3')
        print 'en4',output9.get_shape()

        output10 = tf.layers.conv2d(inputs = output9, filters = 32, kernel_size = (5,5), strides=(2, 2), padding='same', use_bias=False, kernel_initializer=tf.glorot_uniform_initializer(),  name = 'c4')
        output11 = tf.nn.leaky_relu(output10, alpha=0.2, name='l4')
        print 'en5',output11.get_shape()

        output12 = tf.layers.conv2d(inputs = output11, filters = 64, kernel_size = (5,5), strides=(2, 2), padding='same', use_bias=False, kernel_initializer=tf.glorot_uniform_initializer(),  name = 'c5')
        output13 = tf.nn.leaky_relu(output12, alpha=0.2, name='l5')
        print 'en6',output13.get_shape()

        output14 = tf.layers.conv2d(inputs = output13, filters = 121, kernel_size = (5,5), strides=(2, 2), padding='same', use_bias=False, kernel_initializer=tf.glorot_uniform_initializer(),  name = 'c6')
        print 'en7',output14.get_shape()
        output = tf.reshape(output14, [-1, 121])
        return output

class kernel_G(object):
    def __init__(self, name = None):
        self.name = name
    def forward(self, input):
        input1, input2 = input
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            f1 = kernel_G_extract(input1, name = 'kge1')
            f2 = kernel_G_extract(input2, name = 'kge1')
            output = tf.layers.dense(f1 - f2, 121, use_bias = True, kernel_initializer = tf.glorot_uniform_initializer(), name = 'FC1')
            output = tf.nn.softmax(output, axis = -1)
            output = tf.reshape(output, [-1, 11, 11])
            kernel = tf.expand_dims(output, axis = -1)
            kernel = tf.transpose(kernel, [1,2,3,0])
            print 'ker,', kernel.get_shape()
            return kernel


def adap_conv2d_warp(inputs, hidden_state, filter, stride, padding, name = 'adap_conv2d_warp'):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        output1 = tf.nn.conv2d(input = inputs, filter = filter, strides = stride, padding = padding, name = 'c1')
        print 'name',output1.get_shape()
        return output1, hidden_state

class adap_conv(object):
    def __init__(self, name = None):
        self.name = name
    def forward(self, input, hidden_state, kernel = None):
        with tf.variable_scope(self.name, reuse = tf.AUTO_REUSE):
            h = input #shape should be (1, 64, 64, 32), kernel shape should be (4, 4, 32, 32)
            print 'h', h.get_shape()
            print 'kernel[0]', kernel[0].get_shape()
            out1, h1 = adap_conv2d_warp(inputs = h, hidden_state = tf.expand_dims(hidden_state[0], axis = 0), filter = kernel[0], stride = (1,1,1,1), padding = "SAME", name = 'adw1')
            out2, h2 = adap_conv2d_warp(inputs = h, hidden_state = tf.expand_dims(hidden_state[1], axis = 0), filter = kernel[1], stride = (1,1,1,1), padding = "SAME", name = 'adw2')
            out3, h3 = adap_conv2d_warp(inputs = h, hidden_state = tf.expand_dims(hidden_state[2], axis = 0), filter = kernel[2], stride = (1,1,1,1), padding = "SAME", name = 'adw3')
            out4, h4 = adap_conv2d_warp(inputs = h, hidden_state = tf.expand_dims(hidden_state[3], axis = 0), filter = kernel[3], stride = (1,1,1,1), padding = "SAME", name = 'adw4')
            out5, h5 = adap_conv2d_warp(inputs = h, hidden_state = tf.expand_dims(hidden_state[4], axis = 0), filter = kernel[4], stride = (1,1,1,1), padding = "SAME", name = 'adw5')
            return tf.stack([out1, out2, out3,out4, out5], axis = -1), tf.stack([h1, h2, h3, h4, h5], axis = 1)# (1, 64, 64, 32, 5)