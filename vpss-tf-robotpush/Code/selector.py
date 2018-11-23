import tensorflow as tf
from tensorflow.contrib.layers.python import layers as tf_layers

def conv(batch_input, out_channels, stride,scope_name):
    with tf.variable_scope("conv-" + scope_name):
        in_channels = batch_input.get_shape()[3]
        filter = tf.get_variable("filter", [3, 3, in_channels, out_channels], dtype=tf.float32, initializer=tf.random_normal_initializer(0, 0.02))
        padded_input = tf.pad(batch_input, [[0, 0], [1, 1], [1, 1], [0, 0]], mode="CONSTANT")
        conved = tf.nn.conv2d(padded_input, filter, [1, stride, stride, 1], padding="VALID")
        return conved

def deconv(batch_input, out_channels, stride, scope_name):
    with tf.variable_scope("deconv-" + scope_name):
        batch, in_height, in_width, in_channels = [int(d) for d in batch_input.get_shape()]
        filter = tf.get_variable("filter", [3, 3, out_channels, in_channels], dtype=tf.float32, initializer=tf.random_normal_initializer(0, 0.02))
        deconved = tf.nn.conv2d_transpose(batch_input, filter, [batch, in_height * stride, in_width * stride, out_channels], [1, stride, stride, 1], padding="SAME")
        return deconved


class Selector(object):
    def __init__(self, name):
        self.conLSTM_kernel = [3,3]
        self.name = name
        self.cell_dict = dict()
    def _convLSTM(self,ins, ph = 'ENC', scope_name = 'convLSTM', output_height = 64, output_width = 64, output_channel = 3, initial_state=None, scope_reuse = False):
        with tf.variable_scope(scope_name, reuse = tf.AUTO_REUSE):
            # Create a placeholder for videos.
            print scope_name,ins.get_shape()
            if ph == 'ENC':
                ins = conv(ins, output_channel, stride=2, scope_name = scope_name)
            else:
                ins = deconv(ins, output_channel, stride=2, scope_name = scope_name)
            if initial_state == None:
                self.cell_dict[scope_name] = tf.contrib.rnn.ConvLSTMCell(conv_ndims = 2, input_shape = ins.get_shape().as_list()[1:], output_channels = output_channel, kernel_shape = [3,3], use_bias=True, name = scope_name)
                outputs, state = self.cell_dict[scope_name](ins, self.cell_dict[scope_name].zero_state(ins.get_shape()[0], dtype = tf.float32))
            else:
                outputs, state = self.cell_dict[scope_name](ins, initial_state)
            print scope_name,outputs.get_shape()
            outputs = tf_layers.layer_norm(outputs, scope=scope_name)
            return outputs, state

    def forward(self, input):
        print 'input_images',input.get_shape()
        self.ins = input
        self.ens1, self.ens2, self.ens3, self.ens4 = None, None, None, None
        self.des1, self.des2, self.des3, self.des4 = None, None, None, None
        reuse_flag = False
        with tf.variable_scope(self.name, reuse = tf.AUTO_REUSE):
            self.eni0 = input
            self.eni1,self.ens1 = self._convLSTM(self.eni0, initial_state = self.ens1,ph = 'ENC',scope_name = 'ConE1', scope_reuse=reuse_flag, output_height = 32, output_width = 32, output_channel =  64)
            self.eni2,self.ens2 = self._convLSTM(self.eni1, initial_state = self.ens2,ph = 'ENC',scope_name = 'ConE2', scope_reuse=reuse_flag, output_height = 16, output_width = 16, output_channel = 128)
            self.eni3,self.ens3 = self._convLSTM(self.eni2, initial_state = self.ens3,ph = 'ENC',scope_name = 'ConE3', scope_reuse=reuse_flag, output_height =  8, output_width =  8, output_channel = 256)
            self.eni4,self.ens4 = self._convLSTM(self.eni3, initial_state = self.ens4,ph = 'ENC',scope_name = 'ConE4', scope_reuse=reuse_flag, output_height =  4, output_width =  4, output_channel = 512)
            self.dei1,self.des1 = self._convLSTM(self.eni4, initial_state = self.des1,ph = 'DEC',scope_name = 'ConD1', scope_reuse=reuse_flag, output_height =  8, output_width =  8, output_channel = 256)
            self.dei2,self.des2 = self._convLSTM(tf.concat([self.dei1, self.eni3], axis = -1), initial_state = self.des2, ph = 'DEC',scope_name = 'ConD2', scope_reuse=reuse_flag, output_height = 16, output_width = 16, output_channel = 128)
            self.dei3,self.des3 = self._convLSTM(tf.concat([self.dei2, self.eni2], axis = -1), initial_state = self.des3, ph = 'DEC',scope_name = 'ConD3', scope_reuse=reuse_flag, output_height = 32, output_width = 32, output_channel =  64)
            self.dei4,self.des4 = self._convLSTM(tf.concat([self.dei3, self.eni1], axis = -1), initial_state = self.des4, ph = 'DEC',scope_name = 'ConD4', scope_reuse=reuse_flag, output_height = 64, output_width = 64, output_channel =  16)
            output = tf.nn.sigmoid(conv(self.dei4, 3, stride=1, scope_name = 'fin_conv'))
            return output

if __name__ == '__main__':
    test_array = tf.zeros([4, 12, 64, 64, 3])
    myAuto = Selector('selector')
    myAuto.forward(test_array)