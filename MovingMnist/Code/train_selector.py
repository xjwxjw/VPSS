import argparse
import os, datetime
import random
import utils
import itertools
import progressbar
import numpy as np
import tensorflow as tf
import scipy.io as io
import adaptive_conv as adaptive_conv
from tensorflow.python import pywrap_tensorflow
import dcgan_64 as model

parser = argparse.ArgumentParser()
parser.add_argument('--lr', default=0.0002, type=float, help='learning rate')
parser.add_argument('--log_dir', default='../Log', help='base directory to save logs')
parser.add_argument('--beta1', default=0.9, type=float, help='momentum term for adam')
parser.add_argument('--batch_size', default=8, type=int, help='batch size')
parser.add_argument('--optimizer', default='adam', help='optimizer to train with')
parser.add_argument('--niter', type=int, default=302, help='number of epochs to train for')
parser.add_argument('--seed', default=1, type=int, help='manual seed')
parser.add_argument('--epoch_size', type=int, default=600, help='epoch size')
parser.add_argument('--image_width', type=int, default=64, help='the height / width of the input image to network')
parser.add_argument('--channels', default=1, type=int)
parser.add_argument('--n_past', type=int, default=2, help='number of frames to condition on')
parser.add_argument('--n_future', type=int, default=10, help='number of frames to predict during training')
parser.add_argument('--model', default='dcgan', help='model type (dcgan | vgg)')
parser.add_argument('--restore_dir', default='../Log/movingmnist/model.ckpt-300', help='pretrained model path')

opt = parser.parse_args()
opt.log_dir = '%s/%s' % (opt.log_dir, 'movingmnist')

if not os.path.exists('%s/gen/' % opt.log_dir):
    os.makedirs('%s/gen/' % opt.log_dir)
print(opt)


discriminator = model.encoder_D(opt.g_dim, opt.channels, name = 'discriminator')
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
pred_optimizer = opt.optimizer(learning_rate = opt.lr * 0.5, beta1 = opt.beta1, beta2 = 0.999)

def selector(input_frame, hidden_state):
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

train_x = tf.placeholder(dtype = tf.float32, shape = (opt.batch_size, opt.n_past + opt.n_future, opt.image_width, opt.image_width, opt.channels))

mse = 0
kld = 0
pred_array = []
gen0_array = []
gen1_array = []
gen2_array = []
gen3_array = []
gen4_array = []
d_loss_real = 0
d_loss_fake = 0
g_loss = 0
final_pred = None
sel_hidden_state = tf.zeros((opt.batch_size, opt.image_width / 2, opt.image_width / 2, 128))
prediction, sel_hidden_state = selector(train_x[:,0], sel_hidden_state)
prediction = tf.clip_by_value(prediction, np.finfo(np.float32).eps, 1-np.finfo(np.float32).eps)
hidden_state = tf.zeros((opt.batch_size, 5, opt.image_width, opt.image_width, 32))
skip = None
selector_pred = []
bce_loss = 0
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
    out = tf.unstack(h_hat, axis = -1) #kernel_fake.forward(h) 
    x_pred_array = [out[0],out[1],out[2],out[3],out[4]]
    
    if i == 2:
        cur_sel = train_x[:, 1]
    else:
        cur_sel = pred_array[-1]
    prediction, sel_hidden_state = selector(cur_sel, sel_hidden_state)
    prediction = tf.clip_by_value(prediction, np.finfo(np.float32).eps, 1-np.finfo(np.float32).eps)
    mse_array = [tf.reduce_mean(tf.squared_difference(cur_pred, train_x[:,i]), axis = [-3,-2,-1], keepdims = True) for cur_pred in x_pred_array]
    print 'mse_array[0],', mse_array[0].get_shape()
    mse_tensor = tf.concat(axis = -1, values = mse_array) #(batch_size, 5)
    print 'mse_tensor', mse_tensor.get_shape()

    mask = tf.one_hot(indices = tf.argmin(mse_tensor, axis = -1), depth = 5, axis = -1, on_value = 1.0, off_value = 0.0) #(batch_size, 5)
    
    mask = tf.split(mask, 5, axis = -1)
    mask = tf.stop_gradient(mask)
    print 'mask', mask.get_shape()
    final_pred = tf.add_n([x_pred_array[k] * tf.tile(mask[k], [1, 64, 64, 1]) for k in range(5)])
    print 'final_pred', final_pred.get_shape()
    selector_pred.append(prediction)
    bce_loss += tf.reduce_mean(tf.reduce_mean((0 - train_x[:,i]*tf.log(prediction) - (1-train_x[:,i])*tf.log(1-prediction)), [1,2,3]), -1)

    pred_array.append(final_pred)
    gen0_array.append(x_pred_array[0])
    gen1_array.append(x_pred_array[1])
    gen2_array.append(x_pred_array[2])
    gen3_array.append(x_pred_array[3])
    gen4_array.append(x_pred_array[4])

all_vars = tf.trainable_variables()
pred_variables = [var for var in all_vars if var.name.startswith('forward_net')]
pre_op = pred_optimizer.minimize(bce_loss, var_list = pred_variables) 

import moving_mnist as mnist
train_mnist = mnist.mnist(True, batch_size = opt.batch_size, seq_len = (opt.n_past + opt.n_future), num_digits = opt.num_digits, image_size = opt.image_width)

def get_training_batch():
    return train_mnist.getbatch()

import scipy.misc as misc

config = tf.ConfigProto()  
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
saver = tf.train.Saver()
sess.run(tf.global_variables_initializer())

params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
saver_params={}
reader = pywrap_tensorflow.NewCheckpointReader(opt.restore_dir)  
var_to_shape_map = reader.get_variable_to_shape_map()  
checkpoint_keys=var_to_shape_map.keys()
for v in params:
    v_name=v.name.split(':')[0]
    if v_name in checkpoint_keys:
        saver_params[v_name] = v
        print 'For params: ',v_name
saver_res2=tf.train.Saver(saver_params)
saver_res2.restore(sess,opt.restore_dir)

for epoch in range(opt.niter):
    progress = progressbar.ProgressBar(maxval=opt.epoch_size).start()

    train_batch = get_training_batch()
    pred_array_eval, gen0_array_eval, gen1_array_eval, gen2_array_eval, gen3_array_eval, gen4_array_eval, selector_pred_eval \
    = sess.run([pred_array, gen0_array, gen1_array, gen2_array, gen3_array, gen4_array, selector_pred], feed_dict = {train_x:train_batch})
    if not os.path.exists(opt.log_dir+'/gen/'+str(epoch)):
        os.mkdir(opt.log_dir+'/gen/'+str(epoch))
    for i in range((opt.n_past+opt.n_future-2)):
        if not os.path.exists(opt.log_dir+'/gen/'+str(epoch)+'/'+str(i)):
            os.mkdir(opt.log_dir+'/gen/'+str(epoch)+'/'+str(i))
        pred = selector_pred_eval[i][-1,:,:,0]
        pred = np.stack([pred,pred,pred], axis = -1)
        misc.imsave(opt.log_dir+'/gen/'+str(epoch)+'/'+str(i)+'/pred_'+str(i)+'.png', np.cast[np.uint8](pred*255.0))

        gen = gen0_array_eval[i][-1,:,:,0]
        gen = np.stack([gen,gen,gen], axis = -1)
        misc.imsave(opt.log_dir+'/gen/'+str(epoch)+'/'+str(i)+'/gen0_'+str(i)+'.png', np.cast[np.uint8](gen*255.0))
        gen = gen1_array_eval[i][-1,:,:,0]
        gen = np.stack([gen,gen,gen], axis = -1)
        misc.imsave(opt.log_dir+'/gen/'+str(epoch)+'/'+str(i)+'/gen1_'+str(i)+'.png', np.cast[np.uint8](gen*255.0))
        gen = gen2_array_eval[i][-1,:,:,0]
        gen = np.stack([gen,gen,gen], axis = -1)
        misc.imsave(opt.log_dir+'/gen/'+str(epoch)+'/'+str(i)+'/gen2_'+str(i)+'.png', np.cast[np.uint8](gen*255.0))
        gen = gen3_array_eval[i][-1,:,:,0]
        gen = np.stack([gen,gen,gen], axis = -1)
        misc.imsave(opt.log_dir+'/gen/'+str(epoch)+'/'+str(i)+'/gen3_'+str(i)+'.png', np.cast[np.uint8](gen*255.0))
        gen = gen4_array_eval[i][-1,:,:,0]
        gen = np.stack([gen,gen,gen], axis = -1)
        misc.imsave(opt.log_dir+'/gen/'+str(epoch)+'/'+str(i)+'/gen4_'+str(i)+'.png', np.cast[np.uint8](gen*255.0))

        gen = pred_array_eval[i][-1,:,:,0]
        ori = train_batch[-1,i,:,:,0]
        ori = np.stack([ori, 0 * gen ,gen], axis = -1)
        misc.imsave(opt.log_dir+'/gen/'+str(epoch)+'/'+str(i)+'/ori_'+str(i)+'.png', np.cast[np.uint8](ori*255.0))
        ori = train_batch[-1,i+1,:,:,0]
        ori = np.stack([ori, 0 * gen ,gen], axis = -1)
        misc.imsave(opt.log_dir+'/gen/'+str(epoch)+'/'+str(i)+'/ori_'+str(i+1)+'.png', np.cast[np.uint8](ori*255.0))
        ori = train_batch[-1,i+2,:,:,0]
        ori = np.stack([ori, 0 * gen ,gen], axis = -1)
        misc.imsave(opt.log_dir+'/gen/'+str(epoch)+'/'+str(i)+'/ori_'+str(i+2)+'.png', np.cast[np.uint8](ori*255.0))
        
    for i in range(opt.epoch_size):
        progress.update(i+1)
        train_batch_ = get_training_batch()
        bce_eval, _ = sess.run([bce_loss, pre_op], feed_dict = {train_x:train_batch})
        print 'gen ', bce_eval
    progress.finish()
    utils.clear_progressbar()
    if epoch % 10 == 0:
        saver.save(sess, opt.log_dir + "/model.ckpt", global_step=epoch)

    
        
