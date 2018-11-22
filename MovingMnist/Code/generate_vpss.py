import argparse
import os, datetime
import random
import itertools
import progressbar
import numpy as np
import tensorflow as tf
import scipy.io as io
import adaptive_conv as adaptive_conv
from tensorflow.python import pywrap_tensorflow

parser = argparse.ArgumentParser()
parser.add_argument('--lr', default=0.0002, type=float, help='learning rate')
parser.add_argument('--log_dir', default='../Log', help='base directory to save logs')
parser.add_argument('--beta1', default=0.9, type=float, help='momentum term for adam')
parser.add_argument('--batch_size', default=8, type=int, help='batch size')
parser.add_argument('--optimizer', default='adam', help='optimizer to train with')
parser.add_argument('--niter', type=int, default=100, help='number of epochs to train for')
parser.add_argument('--seed', default=1, type=int, help='manual seed')
parser.add_argument('--epoch_size', type=int, default=600, help='epoch size')
parser.add_argument('--image_width', type=int, default=64, help='the height / width of the input image to network')
parser.add_argument('--channels', default=1, type=int)
parser.add_argument('--n_past', type=int, default=2, help='number of frames to condition on')
parser.add_argument('--n_future', type=int, default=10, help='number of frames to predict during training')
parser.add_argument('--g_dim', type=int, default=128, help='dimensionality of encoder output vector and decoder input vector')
parser.add_argument('--num_digits', type=int, default=1, help='number of digits for moving mnist')

opt = parser.parse_args()
now = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
opt.log_dir = '%s/generate-%s' % (opt.log_dir, now)

if not os.path.exists('%s/gen/' % opt.log_dir):
    os.makedirs('%s/gen/' % opt.log_dir)
print(opt)

adap_conv = adaptive_conv.adap_conv(name = 'adap_conv')
kernel_G1  = adaptive_conv.kernel_G(name = 'kernel_G1')
kernel_G2  = adaptive_conv.kernel_G(name = 'kernel_G2')
kernel_G3  = adaptive_conv.kernel_G(name = 'kernel_G3')
kernel_G4  = adaptive_conv.kernel_G(name = 'kernel_G4')
kernel_G5  = adaptive_conv.kernel_G(name = 'kernel_G5')
# ---------------- optimizers ----------------
if opt.optimizer == 'adam':
    opt.optimizer = tf.train.AdamOptimizer
elif opt.optimizer == 'rmsprop':
    opt.optimizer = tf.train.RMSPropOptimizer

generator_optimizer = opt.optimizer(learning_rate = opt.lr, beta1 = opt.beta1, beta2 = 0.999)
diccriminator_optimizer = opt.optimizer(learning_rate = opt.lr * 0.1, beta1 = opt.beta1, beta2 = 0.999)

train_x = tf.placeholder(dtype = tf.float32, shape = (opt.batch_size, opt.n_past + opt.n_future, opt.image_width, opt.image_width, opt.channels))

def _predict(input_frame, hidden_state):
    with tf.variable_scope("forward_net", reuse=tf.AUTO_REUSE):
        output = tf.layers.conv2d(input_frame, filters=32, kernel_size=[3,3], strides=[1,1], padding="same", activation=tf.nn.leaky_relu)
        output = tf.layers.conv2d(output, filters=32, kernel_size=[3,3], strides=[2,2], padding="same", activation=tf.nn.leaky_relu)
        output = tf.layers.conv2d(output, filters=64, kernel_size=[3,3], strides=[1,1], padding="same", activation=tf.nn.leaky_relu)
        output = tf.layers.conv2d(output, filters=64, kernel_size=[3,3], strides=[1,1], padding="same", activation=tf.nn.leaky_relu)

        output = tf.layers.conv2d(output, filters=128, kernel_size=[3,3], strides=[1,1], padding="same", activation=tf.nn.leaky_relu)        
        hidden = tf.layers.conv2d(hidden_state, filters=128, kernel_size=[3,3], strides=[1,1], padding="same", activation=tf.nn.leaky_relu)
        hidden = tf.layers.conv2d(hidden, filters=128, kernel_size=[3,3], strides=[1,1], padding="same", activation=tf.nn.leaky_relu)
        output = output + hidden
        hidden_state = output

        output = tf.layers.conv2d(output, filters=64, kernel_size=[3,3], strides=[1,1], padding="same", activation=tf.nn.leaky_relu)
        output = tf.layers.conv2d(output, filters=64, kernel_size=[3,3], strides=[1,1], padding="same", activation=tf.nn.leaky_relu)
        output = tf.image.resize_nearest_neighbor(output, size=(2*output.get_shape()[1].value, 2*output.get_shape()[2].value), align_corners=False)
        output = tf.layers.conv2d(output, filters=64, kernel_size=[3,3], strides=[1,1], padding="same", activation=tf.nn.leaky_relu)
        output = tf.layers.conv2d(output, filters=64, kernel_size=[3,3], strides=[1,1], padding="same", activation=tf.nn.leaky_relu)

        output = tf.layers.conv2d(output, filters=128, kernel_size=[1,1], strides=[1,1], padding="same", activation=tf.nn.leaky_relu)
        dynamic_filters = tf.layers.conv2d(output, filters=81, kernel_size=[1,1], padding="same")
        dynamic_filters = tf.nn.softmax(dynamic_filters, dim=-1)

        input_frame_transformed = tf.extract_image_patches(
            input_frame, [1, 9, 9, 1],
            strides=[1,1,1,1], rates=[1,1,1,1], padding="SAME")

        prediction = tf.reduce_sum(dynamic_filters * input_frame_transformed, -1, keep_dims=True)
    return prediction, hidden_state

pred_array = []
gen0_array = []
gen1_array = []
gen2_array = []
gen3_array = []
gen4_array = []
d_loss_real = 0
d_loss_fake = 0
g_loss = 0
kernel_array = []
final_pred = None
hidden_state = tf.zeros((opt.batch_size, 5, opt.image_width, opt.image_width, 32))
dfn_hidden_state = tf.zeros((opt.batch_size, opt.image_width / 2, opt.image_width / 2, 128))
prediction, dfn_hidden_state = _predict(train_x[:,0], dfn_hidden_state)
prediction = tf.clip_by_value(prediction, np.finfo(np.float32).eps, 1-np.finfo(np.float32).eps)
for i in range(2, opt.n_past+opt.n_future):
    print i
    if i == 2:
        h = train_x[:,i-1]
    else:
        h = final_pred
    h_hat = []
    hidden_hat = []
    for bs in range(opt.batch_size):
        if i == 2:
            inputs1, inputs2 = train_x[bs, i-1], train_x[bs, i-2]
        elif i == 3:
            inputs1, inputs2 = pred_array[-1][bs], train_x[bs, i-2]
        else:
            inputs1, inputs2 = pred_array[-1][bs], pred_array[-2][bs]
        x_n = [tf.expand_dims(inputs1, axis = 0), tf.expand_dims(inputs2, axis = 0)]
        kernel = [kernel_G1.forward(x_n),kernel_G2.forward(x_n),kernel_G3.forward(x_n),kernel_G4.forward(x_n),kernel_G5.forward(x_n)]
        out, hidden = adap_conv.forward(input = tf.expand_dims(h[bs], axis = 0), hidden_state = hidden_state[bs], kernel = kernel)
        h_hat.append(out)
        hidden_hat.append(hidden)
    h_hat = tf.concat(h_hat, axis = 0)
    hidden_state = tf.concat(hidden_hat, axis = 0)
    print 'h,', h.get_shape()
    print 'h_hat,', h_hat.get_shape()
    out = tf.unstack(h_hat, axis = -1) 
    x_pred_array = [out[0],out[1],out[2],out[3],out[4]]
    if i == 2:
        inputs1 = train_x[:, 1]
    else:
        inputs1 = pred_array[-1]
    prediction, dfn_hidden_state = _predict(inputs1, dfn_hidden_state)
    prediction = tf.clip_by_value(prediction, np.finfo(np.float32).eps, 1-np.finfo(np.float32).eps)
    mse_array = [tf.reduce_mean(tf.squared_difference(cur_pred, prediction), axis = [-3,-2,-1], keepdims = True) for cur_pred in x_pred_array]
    print 'mse_array[0],', mse_array[0].get_shape()
    mse_tensor = tf.concat(axis = -1, values = mse_array) #(batch_size, 5)
    print 'mse_tensor', mse_tensor.get_shape()
    mask = tf.one_hot(indices = tf.argmin(mse_tensor, axis = -1), depth = 5, axis = -1, on_value = 1.0, off_value = 0.0) #(batch_size, 5)
    
    mask = tf.split(mask, 5, axis = -1)
    mask = tf.stop_gradient(mask)
    print 'mask', mask.get_shape()
    final_pred = tf.add_n([x_pred_array[k] * tf.tile(mask[k], [1, 64, 64, 1]) for k in range(5)])
    print 'final_pred', final_pred.get_shape()
    pred_array.append(final_pred)
    gen0_array.append(x_pred_array[0])
    gen1_array.append(x_pred_array[1])
    gen2_array.append(x_pred_array[2])
    gen3_array.append(x_pred_array[3])
    gen4_array.append(x_pred_array[4])
import moving_mnist as mnist
test_mnist = mnist.mnist(False, batch_size = opt.batch_size, seq_len = (opt.n_past + opt.n_future), num_digits = opt.num_digits, image_size = opt.image_width)
def get_training_batch():
    return test_mnist.getbatch()

import scipy.misc as misc

config = tf.ConfigProto()  
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
saver = tf.train.Saver()
sess.run(tf.global_variables_initializer())

params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
saver_params={}
reader = pywrap_tensorflow.NewCheckpointReader('../PretrainedModels/model.ckpt-0')  
var_to_shape_map = reader.get_variable_to_shape_map()  
checkpoint_keys=var_to_shape_map.keys()
for v in params:
    v_name=v.name.split(':')[0]
    if v_name in checkpoint_keys:
        saver_params[v_name] = v
        print 'For params: ',v_name
saver_res1=tf.train.Saver(saver_params)
saver_res1.restore(sess,'../PretrainedModels/model.ckpt-0')

for epoch in range(opt.niter):
    print epoch

    train_batch = get_training_batch()
    pred_array_eval, gen0_array_eval, gen1_array_eval, gen2_array_eval, gen3_array_eval, gen4_array_eval \
    = sess.run([pred_array, gen0_array, gen1_array, gen2_array, gen3_array, gen4_array], feed_dict = {train_x:train_batch})
    for bs in range(opt.batch_size):
        if not os.path.exists(opt.log_dir+'/gen/'+str(epoch * opt.batch_size + bs)):
            os.mkdir(opt.log_dir+'/gen/'+str(epoch * opt.batch_size + bs))
        for i in range((opt.n_past+opt.n_future-2)):
            gen = gen0_array_eval[i][bs,:,:,0]
            gen = np.stack([gen,gen,gen], axis = -1)
            misc.imsave(opt.log_dir+'/gen/'+str(epoch * opt.batch_size + bs)+'/gen0_'+str(i)+'.png', np.cast[np.uint8](gen*255.0))
            gen = gen1_array_eval[i][bs,:,:,0]
            gen = np.stack([gen,gen,gen], axis = -1)
            misc.imsave(opt.log_dir+'/gen/'+str(epoch * opt.batch_size + bs)+'/gen1_'+str(i)+'.png', np.cast[np.uint8](gen*255.0))
            gen = gen2_array_eval[i][bs,:,:,0]
            gen = np.stack([gen,gen,gen], axis = -1)
            misc.imsave(opt.log_dir+'/gen/'+str(epoch * opt.batch_size + bs)+'/gen2_'+str(i)+'.png', np.cast[np.uint8](gen*255.0))
            gen = gen3_array_eval[i][bs,:,:,0]
            gen = np.stack([gen,gen,gen], axis = -1)
            misc.imsave(opt.log_dir+'/gen/'+str(epoch * opt.batch_size + bs)+'/gen3_'+str(i)+'.png', np.cast[np.uint8](gen*255.0))
            gen = gen4_array_eval[i][bs,:,:,0]
            gen = np.stack([gen,gen,gen], axis = -1)
            misc.imsave(opt.log_dir+'/gen/'+str(epoch * opt.batch_size + bs)+'/gen4_'+str(i)+'.png', np.cast[np.uint8](gen*255.0))

            gen = pred_array_eval[i][bs,:,:,0]
            gen = np.stack([gen,gen,gen], axis = -1)
            misc.imsave(opt.log_dir+'/gen/'+str(epoch * opt.batch_size + bs)+'/fin_'+str(i)+'.png', np.cast[np.uint8](gen*255.0))

            ori = train_batch[bs,i+2,:,:,0]
            ori = np.stack([ori,ori,ori], axis = -1)
            misc.imsave(opt.log_dir+'/gen/'+str(epoch * opt.batch_size + bs)+'/ori_'+str(i)+'.png', np.cast[np.uint8](ori*255.0))

import imageio as imageio
psnr_array = np.zeros((opt.batch_size * opt.niter,10))
os.mkdir(opt.log_dir+'/GIF')
for itr in range(opt.batch_size * opt.niter):
    images = []
    for i in range(10):
        ori = misc.imread(opt.log_dir+'/gen/'+str(itr)+'/ori_'+str(i)+'.png')
        img = 0
        for j in ([0,1,2,3,4]):
            tmp = misc.imread(opt.log_dir+'/gen/'+str(itr)+'/gen'+str(j)+'_'+str(i)+'.png') 
            img = img + tmp
        final_pred = misc.imread(opt.log_dir+'/gen/'+str(itr)+'/fin_'+str(i)+'.png')
        images.append(np.concatenate([ori, img, final_pred], axis = 1))
    imageio.mimsave(opt.log_dir+'/GIF/'+str(itr)+'.gif', images, duration=0.2)
        