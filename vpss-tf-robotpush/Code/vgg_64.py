import tensorflow as tf

def _conv2d(input, out_channel, name = None):
    with tf.variable_scope(name, reuse = tf.AUTO_REUSE):
        output1 = tf.layers.conv2d(inputs = input, filters = out_channel, kernel_size = (3,3), strides=(1, 1), padding='same', use_bias=True, kernel_initializer=tf.random_normal_initializer(0.0,0.02), bias_initializer=tf.zeros_initializer(), name = 'c1')
        output2 = tf.layers.batch_normalization(output1, training = True, momentum=0.9, gamma_initializer = tf.random_normal_initializer(1.0,0.02), beta_initializer = tf.zeros_initializer(), name = 'b1')
        output3 = tf.nn.leaky_relu(output2, alpha=0.2, name='l1')
        return output3

def final_conv2d(input, out_channel, name = None):
    with tf.variable_scope(name, reuse = tf.AUTO_REUSE):
        output1 = tf.layers.conv2d(inputs = input, filters = out_channel, kernel_size = (4,4), strides=(1, 1), padding='valid', use_bias=True, kernel_initializer=tf.random_normal_initializer(0.0,0.02), bias_initializer=tf.zeros_initializer(), name = 'c1')
        output2 = tf.layers.batch_normalization(output1, training = True, momentum=0.9, gamma_initializer = tf.random_normal_initializer(1.0,0.02), beta_initializer = tf.zeros_initializer(), name = 'b1')
        output3 = tf.nn.tanh(output2, name='t1')
        # output3 = tf.nn.l2_normalize(output2, axis = -1, name='t1')
        return tf.squeeze(tf.squeeze(output2, axis = -2), axis = -2), output3

def init_conv2d(input, out_channel, name = None):
    with tf.variable_scope(name, reuse = tf.AUTO_REUSE):
        output1 = tf.layers.conv2d_transpose(inputs = input, filters = out_channel, kernel_size = (4,4), strides=(1, 1), padding='valid', use_bias=True, kernel_initializer=tf.random_normal_initializer(0.0,0.02), bias_initializer=tf.zeros_initializer(), name = 'c1')
        output2 = tf.layers.batch_normalization(output1, training = True, momentum=0.9, gamma_initializer = tf.random_normal_initializer(1.0,0.02), beta_initializer = tf.zeros_initializer(), name = 'b1')
        output3 = tf.nn.leaky_relu(output2, alpha=0.2, name='l1')
        return output3

def max_pool(input, name = None):
    with tf.variable_scope(name, reuse = tf.AUTO_REUSE):
        output = tf.layers.max_pooling2d(inputs = input, pool_size = (2, 2), strides = (2, 2), name = 'max-1')
        return output

def up_pool(input, name = None):
    with tf.variable_scope(name, reuse = tf.AUTO_REUSE):
        output = tf.image.resize_nearest_neighbor(input, [input.get_shape()[1] * 2, input.get_shape()[2] * 2])
        return output 

class encoder_G(object):
    def __init__(self, name = None):
        self.name = name
    def forward(self, input):
        with tf.variable_scope(self.name, reuse = tf.AUTO_REUSE):
            print 'en-input', input.get_shape()
            output1 = _conv2d(_conv2d(input, 32, name = 'conv1'), 32, name = 'conv2')
            print 'en-output1', output1.get_shape()
            output2 = _conv2d(_conv2d(max_pool(output1, name = 'max-1'), 64, name = 'conv3'), 64, name = 'conv4')
            print 'en-output2', output2.get_shape()
            output3 = _conv2d(_conv2d(_conv2d(max_pool(output2, name = 'max-2'), 128, name = 'conv5'), 128, name = 'conv6'), 128, name = 'conv7')
            print 'en-output3', output3.get_shape()
            output4 = _conv2d(_conv2d(_conv2d(max_pool(output3, name = 'max-3'), 256, name = 'conv8'), 256, name = 'conv9'), 256, name = 'conv10')
            print 'en-output4', output4.get_shape()
            prev, output5 = final_conv2d(max_pool(output4, name = 'max-4'), 256, name = 'conv11')
            print 'en-output5', output5.get_shape()
            return prev, tf.squeeze(tf.squeeze(output5, axis = -2), axis = -2), [output1, output2, output3, output4]
            
class decoder_G(object):
    def __init__(self, name):
        self.name = name
    def forward(self, vec, skip):
        # vec, skip = input
        vec = tf.expand_dims(tf.expand_dims(vec, axis = -2), axis = -2)
        with tf.variable_scope(self.name, reuse = tf.AUTO_REUSE):
            print 'de-input', vec.get_shape()
            output1 = init_conv2d(vec, 256, name = 'convt1')
            print 'de-output1', output1.get_shape()
            output2 = _conv2d(_conv2d(_conv2d(tf.concat([up_pool(output1, name = 'up-1'), skip[-1]], axis = -1), 256, name = 'convt2'), 256, name = 'convt3'), 128, name = 'convt4')
            print 'de-output2', output2.get_shape()
            output3 = _conv2d(_conv2d(_conv2d(tf.concat([up_pool(output2, name = 'up-2'), skip[-2]], axis = -1), 128, name = 'convt5'), 128, name = 'convt6'), 64, name = 'convt7')
            print 'de-output3', output3.get_shape()
            output4 = _conv2d(_conv2d(tf.concat([up_pool(output3, name = 'up-3'), skip[-3]], axis = -1), 64, name = 'convt8'), 32, name = 'convt9')
            print 'de-output4', output4.get_shape()
            output5 = _conv2d(tf.concat([up_pool(output4, name = 'up-4'), skip[-4]], axis = -1), 32, name = 'convt10')
            print 'de-output5', output5.get_shape()
            output6 = tf.nn.sigmoid(tf.layers.conv2d_transpose(inputs = output5, filters = 3, kernel_size = (3,3), strides=(1, 1), padding='same', use_bias=True, kernel_initializer=tf.random_normal_initializer(0.0,0.02), bias_initializer=tf.zeros_initializer(), name = 'c1'))
            print 'de-output6', output6.get_shape()
            return output6

class encoder_D(object):
    def __init__(self, dim, nc = 1, name = None):
        self.dim = dim
        self.nf = 64
        self.name = name
        self.nc = 1
    
    def forward(self, input):
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            print 'de1',input.get_shape()
            output1 = tf.layers.conv2d(inputs = input, filters = self.nf, kernel_size = (4,4), strides=(2, 2), padding='same', use_bias=True, kernel_initializer=tf.random_normal_initializer(0.0,0.02), bias_initializer=tf.zeros_initializer(), name = 'c1')
            # output2 = tf.layers.batch_normalization(output1, training = True, momentum=0.1, gamma_initializer = tf.random_normal_initializer(1.0,0.02), beta_initializer = tf.zeros_initializer(), name = 'b1')
            output3 = tf.nn.leaky_relu(output1, alpha=0.2, name='l1')
            # output3 = tf.nn.tanh(output2)
            print 'de2',output3.get_shape()

            output4 = tf.layers.conv2d(inputs = output3, filters = self.nf * 2, kernel_size = (4,4), strides=(2, 2), padding='same', use_bias=True, kernel_initializer=tf.random_normal_initializer(0.0,0.02), bias_initializer=tf.zeros_initializer(), name = 'c2')
            # output5 = tf.layers.batch_normalization(output4, training = True, momentum=0.1, gamma_initializer = tf.random_normal_initializer(1.0,0.02), beta_initializer = tf.zeros_initializer(), name = 'b2')
            output6 = tf.nn.leaky_relu(output4, alpha=0.2, name='l2')
            # output6 = tf.nn.tanh(output5)
            print 'de3',output6.get_shape()

            output7 = tf.layers.conv2d(inputs = output6, filters = self.nf * 4, kernel_size = (4,4), strides=(2, 2), padding='same', use_bias=True, kernel_initializer=tf.random_normal_initializer(0.0,0.02), bias_initializer=tf.zeros_initializer(), name = 'c3')
            # output8 = tf.layers.batch_normalization(output7, training = True, momentum=0.1, gamma_initializer = tf.random_normal_initializer(1.0,0.02), beta_initializer = tf.zeros_initializer(), name = 'b3')
            output9 = tf.nn.leaky_relu(output7, alpha=0.2, name='l3')
            # output9 = tf.nn.tanh(output8)
            print 'de4',output9.get_shape()

            output10 = tf.layers.conv2d(inputs = output9, filters = self.nf * 8, kernel_size = (4,4), strides=(2, 2), padding='same', use_bias=True, kernel_initializer=tf.random_normal_initializer(0.0,0.02), bias_initializer=tf.zeros_initializer(), name = 'c4')
            # output11 = tf.layers.batch_normalization(output10, training = True, momentum=0.1, gamma_initializer = tf.random_normal_initializer(1.0,0.02), beta_initializer = tf.zeros_initializer(), name = 'b4')
            output12 = tf.nn.leaky_relu(output10, alpha=0.2, name='l4')
            # output12 = tf.nn.tanh(output11)
            print 'de5',output12.get_shape()

            output13 = tf.layers.conv2d(inputs = output12, filters = self.dim, kernel_size = (4,4), strides=(1, 1), padding='valid', use_bias=True, kernel_initializer=tf.random_normal_initializer(0.0,0.02), bias_initializer=tf.zeros_initializer(), name = 'c5')
            # output14 = tf.layers.batch_normalization(output13, training = True, momentum=0.1, gamma_initializer = tf.random_normal_initializer(1.0,0.02), beta_initializer = tf.zeros_initializer(), name = 'b5')
            output15 = tf.nn.leaky_relu(output13, alpha=0.2, name='l5')
            # output15 = tf.nn.tanh(output14)
            print 'de6',output15.get_shape()

            # output17 = tf.layers.conv2d(inputs = output15, filters = self.dim, kernel_size = (2,2), strides=(1, 1), padding='valid', use_bias=True, kernel_initializer=tf.random_normal_initializer(0.0,0.02), bias_initializer=tf.zeros_initializer(), name = 'c6')
            # output18 = tf.layers.batch_normalization(output17, training = True, momentum=0.1, gamma_initializer = tf.random_normal_initializer(1.0,0.02), beta_initializer = tf.zeros_initializer(), name = 'b6')
            # output19 = tf.nn.leaky_relu(output18, alpha=0.2, name='l6')
            # output15 = tf.nn.tanh(output14, name='l5')
            output16 = tf.layers.dense(inputs = tf.reshape(output15, [-1, self.dim]), units = 1, use_bias=True, kernel_initializer=tf.random_normal_initializer(0.0,0.02), bias_initializer=tf.zeros_initializer())
            print 'de7',output16.get_shape()
            return output16

class feat_D(object):
    def __init__(self):
        pass
    def froward(self):
        pass