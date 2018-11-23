import tensorflow as tf

def kernel_G_extract(input, name = None):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE): 
        output = tf.layers.flatten(input)
        output = tf.layers.dense(output, 256, use_bias = False, kernel_initializer = tf.glorot_uniform_initializer(), name = 'FC1')
        output = tf.nn.leaky_relu(output, alpha=0.2)
        output = tf.nn.tanh(output)
        # tmp_out = output
        print 'en1', output.get_shape()

        output = tf.reshape(output, [1, 16, 16, 1])
        output = tf.image.resize_bicubic(output, size = [output.get_shape()[1] * 2 + 2, output.get_shape()[2] * 2 + 2])
        output = tf.layers.conv2d(inputs = output, filters = 1, kernel_size = (3,3), strides=(1, 1), padding='valid', use_bias=False, kernel_initializer=tf.glorot_uniform_initializer(),  name = 'c4')
        # output = tf.nn.leaky_relu(output, alpha=0.2)
        output = tf.nn.tanh(output)
        print 'en5',output.get_shape()

        output = tf.reshape(output, [-1, 256])
        output = tf.layers.dense(output, 256, use_bias = False, kernel_initializer = tf.glorot_uniform_initializer(), name = 'FC2')
        output = tf.nn.tanh(output)
        output = tf.reshape(tf.transpose(output, [1, 0]), [1, 32, 32, 1])

        output = tf.image.resize_bicubic(output, size = [output.get_shape()[1] * 2 + 2, output.get_shape()[2] * 2 + 2])
        output = tf.layers.conv2d(inputs = output, filters = 1, kernel_size = (3,3), strides=(1, 1), padding='valid', use_bias=False, kernel_initializer=tf.glorot_uniform_initializer(),  name = 'c5')
        # output = tf.nn.leaky_relu(output, alpha=0.2)
        output = tf.nn.tanh(output)
        print 'en6',output.get_shape()

        output = tf.reshape(output, [-1, 256])
        output = tf.layers.dense(output, 256, use_bias = False, kernel_initializer = tf.glorot_uniform_initializer(), name = 'FC3')
        output = tf.nn.tanh(output)
        output = tf.reshape(tf.transpose(output, [1, 0]), [1, 64, 64, 1])

        output = tf.image.resize_bicubic(output, size = [output.get_shape()[1] * 2 + 2, output.get_shape()[2] * 2 + 2])
        output = tf.layers.conv2d(inputs = output, filters = 1, kernel_size = (3,3), strides=(1, 1), padding='valid', use_bias=False, kernel_initializer=tf.glorot_uniform_initializer(),  name = 'c6')
        # output = tf.nn.leaky_relu(output, alpha=0.2)
        output = tf.nn.tanh(output)
        print 'en7',output.get_shape()

        output = tf.reshape(output, [-1, 256])
        output = tf.layers.dense(output, 256, use_bias = False, kernel_initializer = tf.glorot_uniform_initializer(), name = 'FC4')
        output = tf.nn.tanh(output)
        output = tf.reshape(tf.transpose(output, [1, 0]), [1, 128, 128, 1])

        output = tf.image.resize_bicubic(output, size = [output.get_shape()[1] * 2 + 2, output.get_shape()[2] * 2 + 2])
        output = tf.layers.conv2d(inputs = output, filters = 1, kernel_size = (3,3), strides=(1, 1), padding='valid', use_bias=False, kernel_initializer=tf.glorot_uniform_initializer(),  name = 'c7')
        output = tf.squeeze(tf.squeeze(output, axis = 0), axis = -1)
        output = 0.1 * tf.nn.tanh(output)
        # output = tf.nn.softmax(output, axis = 0)
        # output = 0.1 * output / tf.reduce_max(output)
        # output = tf.reshape(tf.nn.l2_normalize(tf.reshape(output, [1, 256 * 256]), axis = -1), [256, 256])
        print 'en8',output.get_shape()
        return output#, tmp_out
        
class kernel_G(object):
    def __init__(self, name = None):
        self.name = name
    def forward(self, input):
        input1, input2 = input
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            kernel = kernel_G_extract(input1 - input2, name = 'kge1')
            print 'ker,', kernel.get_shape()
            return kernel#, tmp_out

def adap_conv2d_warp(inputs, hidden_state, filter, stride, padding, name = 'adap_conv2d_warp'):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        output1 = tf.matmul(a = inputs, b = filter)
        print 'name',output1.get_shape()
        return output1, hidden_state

class feat_D(object):
    def __init__(self, name = None):
        self.name = name
    def forward(self, input):
        with tf.variable_scope(self.name, reuse = tf.AUTO_REUSE):
            output = tf.layers.dense(input, 512, use_bias = False, kernel_initializer = tf.glorot_uniform_initializer(), name = 'FC1')
            output = tf.nn.leaky_relu(output, alpha=0.2)
            output = tf.layers.dense(input, 512, use_bias = False, kernel_initializer = tf.glorot_uniform_initializer(), name = 'FC2')
            output = tf.nn.leaky_relu(output, alpha=0.2)
            output = tf.layers.dense(input, 512, use_bias = False, kernel_initializer = tf.glorot_uniform_initializer(), name = 'FC3')
            output = tf.nn.leaky_relu(output, alpha=0.2)
            output = tf.layers.dense(input, 512, use_bias = False, kernel_initializer = tf.glorot_uniform_initializer(), name = 'FC4')
            output = tf.nn.leaky_relu(output, alpha=0.2)
            output = tf.layers.dense(input, 1, use_bias = False, kernel_initializer = tf.glorot_uniform_initializer(), name = 'FC5')
            return output

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