import argparse
import os, datetime
import random
import itertools
import progressbar
import numpy as np
import tensorflow as tf
import scipy.io as io
import adaptive_conv_robotpush as adaptive_conv
from RobotPushData import build_tfrecord_input_varilength
import selector as selector
from tensorflow.python import pywrap_tensorflow

parser = argparse.ArgumentParser()
parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
parser.add_argument('--beta1', default=0.9, type=float, help='momentum term for adam')
parser.add_argument('--batch_size', default=16, type=int, help='batch size')
parser.add_argument('--log_dir', default='../Log', help='base directory to save logs')
parser.add_argument('--optimizer', default='adam', help='optimizer to train with')
parser.add_argument('--niter', type=int, default=302, help='number of epochs to train for')
parser.add_argument('--seed', default=1, type=int, help='manual seed')
parser.add_argument('--epoch_size', type=int, default=600, help='epoch size')
parser.add_argument('--image_width', type=int, default=64, help='the height / width of the input image to network')
parser.add_argument('--channels', default=3, type=int)
parser.add_argument('--dataset', default='robotpush', help='dataset to train with')
parser.add_argument('--n_past', type=int, default=2, help='number of frames to condition on')
parser.add_argument('--n_future', type=int, default=10, help='number of frames to predict during training')

opt = parser.parse_args()
now = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
opt.log_dir = '%s/%s' % (opt.log_dir, 'robotpush')

if not os.path.exists('%s/gen/' % opt.log_dir):
    os.makedirs('%s/gen/' % opt.log_dir)
if not os.path.exists('%s/plots/' % opt.log_dir):
    os.makedirs('%s/plots/' % opt.log_dir)
print(opt)

import vgg_64 as model

encoder = model.encoder_G(name = 'encoder')
decoder = model.decoder_G(name = 'decoder')
discriminator = model.encoder_D(opt.g_dim, opt.channels, name = 'discriminator_new')
selector = selector.Selector(name = 'selector')
adap_conv = adaptive_conv.adap_conv(name = 'adap_conv')
kernel_G1  = adaptive_conv.kernel_G(name = 'kernel_G1')
kernel_G2  = adaptive_conv.kernel_G(name = 'kernel_G2')
kernel_G3  = adaptive_conv.kernel_G(name = 'kernel_G3')
kernel_G4  = adaptive_conv.kernel_G(name = 'kernel_G4')
kernel_G5  = adaptive_conv.kernel_G(name = 'kernel_G5')

feat_D  = adaptive_conv.feat_D(name = 'feat_D')
# ---------------- optimizers ----------------
if opt.optimizer == 'adam':
    opt.optimizer = tf.train.AdamOptimizer
    generator_optimizer = opt.optimizer(learning_rate = opt.lr, beta1 = opt.beta1, beta2 = 0.999)
    diccriminator_optimizer = opt.optimizer(learning_rate = opt.lr, beta1 = opt.beta1, beta2 = 0.999)
elif opt.optimizer == 'rmsprop':
    opt.optimizer = tf.train.RMSPropOptimizer
    generator_optimizer = opt.optimizer(learning_rate = opt.lr)
    diccriminator_optimizer = opt.optimizer(learning_rate = opt.lr)



TOTAL_LENGTH = 12
input_x = build_tfrecord_input_varilength(length = TOTAL_LENGTH, batch_size = opt.batch_size, training = True)
print 'input_x,',input_x.get_shape()
train_x = input_x
print 'train_x,',train_x.get_shape()

msex_loss = 0
msef_loss = 0
mse_sel_loss = 0
kld = 0
pred_array = []
gen0_array = []
gen1_array = []
gen2_array = []
gen3_array = []
gen4_array = []
# h_array = []
dx_loss_real = 0
dx_loss_fake = 0
gx_loss = 0
df_loss_real = 0
df_loss_fake = 0
gf_loss = 0

final_pred = None
hidden_state = tf.zeros((opt.batch_size, 5, opt.image_width, opt.image_width, 32))
sel_pred = selector.forward(train_x[:,0])
sel_array = []
sel_array.append(sel_pred)
mse_sel_loss += tf.reduce_mean(tf.abs(sel_pred - train_x[:,1]))
for i in range(2, opt.n_past+opt.n_future):
    print i
    if i == 2:
        inputs1, inputs2 = train_x[:, i-1], train_x[:, i-2]
    elif i == 3:
        inputs1, inputs2 = pred_array[-1], train_x[:, i-2]
    else:
        inputs1, inputs2 = pred_array[-1], pred_array[-2]
    prev, h, skip = encoder.forward(inputs1)
    _, h_prev, _ = encoder.forward(inputs2)

    h_hat = []
    hidden_hat = []
    for bs in range(opt.batch_size):
        h_n = [tf.expand_dims(h[bs], axis = 0), tf.expand_dims(h_prev[bs], axis = 0)]
        kernel = [kernel_G1.forward(h_n),kernel_G2.forward(h_n),kernel_G3.forward(h_n),kernel_G4.forward(h_n),kernel_G5.forward(h_n)]
        out_res, hidden = adap_conv.forward(input = tf.expand_dims(h[bs], axis = 0), hidden_state = hidden_state[bs], kernel = kernel)
        h_hat.append(out_res)
        hidden_hat.append(hidden)
    h_hat = tf.concat(h_hat, axis = 0)
    hidden_state = tf.concat(hidden_hat, axis = 0)
    print 'h,', h.get_shape()
    print 'h_hat,', h_hat.get_shape()
    out = tf.unstack(h_hat, axis = -1) 
    h_pred = [tf.tanh(out[0] + prev), tf.tanh(out[1] + prev), tf.tanh(out[2] + prev), tf.tanh(out[3] + prev), tf.tanh(out[4] + prev)]
    x_pred_array = [decoder.forward(h_pred[0], skip), decoder.forward(h_pred[1], skip), decoder.forward(h_pred[2], skip), decoder.forward(h_pred[3], skip), decoder.forward(h_pred[4], skip)]
    sel_pred = selector.forward(inputs1)
    sel_array.append(sel_pred)
    mse_array = [tf.reduce_mean(tf.squared_difference(cur_pred, sel_pred), axis = [-3,-2,-1], keepdims = True) for cur_pred in x_pred_array]
    print 'mse_array[0],', mse_array[0].get_shape()
    mse_tensor = tf.concat(axis = -1, values = mse_array) #(batch_size, 5)
    print 'mse_tensor', mse_tensor.get_shape()
    mask = tf.one_hot(indices = tf.argmin(mse_tensor, axis = -1), depth = 5, axis = -1, on_value = 1.0, off_value = 0.0) #(batch_size, 5)
    
    mask = tf.split(mask, 5, axis = -1)
    mask = tf.stop_gradient(mask)
    print 'mask', mask.get_shape()
    final_pred = tf.add_n([x_pred_array[k] * tf.tile(mask[k], [1, 64, 64, 3]) for k in range(5)])
    final_feat = tf.add_n([h_pred[k] * tf.tile(tf.squeeze(tf.squeeze(mask[k], axis = -1), axis = -1), [1, 256]) for k in range(5)])
    print 'final_pred', final_pred.get_shape()
    mse_sel_loss += tf.reduce_mean(tf.abs(sel_pred - train_x[:,i]))
    pred_array.append(final_pred)
    gen0_array.append(x_pred_array[0])
    gen1_array.append(x_pred_array[1])
    gen2_array.append(x_pred_array[2])
    gen3_array.append(x_pred_array[3])
    gen4_array.append(x_pred_array[4])
dx_loss = 0.5 * dx_loss_real + 0.5 * dx_loss_fake 
df_loss = 0.5 * df_loss_real + 0.5 * df_loss_fake
all_vars = tf.trainable_variables()
selector_vars = [var for var in all_vars if var.name.startswith('selector')]
mse_sel_op = diccriminator_optimizer.minimize(1000 * mse_sel_loss, var_list = selector_vars) 

import scipy.misc as misc

def clear_progressbar():
    # moves up 3 lines
    print("\033[2A")
    # deletes the whole line, regardless of character position
    print("\033[2K")
    # moves up two lines again
    print("\033[2A")


config = tf.ConfigProto()  
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
saver = tf.train.Saver()
sess.run(tf.global_variables_initializer())


params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
saver_params={}
reader = pywrap_tensorflow.NewCheckpointReader(opt.log_dir+'/model.ckpt-300')  
var_to_shape_map = reader.get_variable_to_shape_map()  
checkpoint_keys=var_to_shape_map.keys()
for v in params:
    v_name=v.name.split(':')[0]
    if v_name in checkpoint_keys:
        saver_params[v_name] = v
        print 'For params: ',v_name
saver_res1=tf.train.Saver(saver_params)
saver_res1.restore(sess,opt.log_dir+'/model.ckpt-300')
wu = 0
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess, coord)
train_idx_ = np.random.choice(TOTAL_LENGTH-(opt.n_past+opt.n_future - 1), opt.batch_size)
print 'hh'
train_x_eval = sess.run(train_x)
print 'hh'
for epoch in range(opt.niter):
    epoch_mse = 0
    epoch_kld = 0
    progress = progressbar.ProgressBar(maxval=opt.epoch_size).start()
    train_idx_ = np.random.choice(TOTAL_LENGTH-(opt.n_past+opt.n_future - 1), opt.batch_size)
    pred_array_eval, gen0_array_eval, gen1_array_eval, gen2_array_eval, gen3_array_eval, gen4_array_eval, train_x_eval, sel_array_eval \
     = sess.run([pred_array, gen0_array, gen1_array, gen2_array, gen3_array, gen4_array, train_x, sel_array])
    if not os.path.exists(opt.log_dir+'/gen/'+str(epoch)):
        os.mkdir(opt.log_dir+'/gen/'+str(epoch))
    for i in range((opt.n_past+opt.n_future-2)):

        gen = pred_array_eval[i][-1,:,:,:]
        # gen = np.stack([gen,gen,gen], axis = -1)
        misc.imsave(opt.log_dir+'/gen/'+str(epoch)+'/fin_'+str(i)+'.png', np.cast[np.uint8](gen*255.0))

        sel = sel_array_eval[i+1][-1,:,:,:]
        # gen = np.stack([gen,gen,gen], axis = -1)
        misc.imsave(opt.log_dir+'/gen/'+str(epoch)+'/sel_'+str(i)+'.png', np.cast[np.uint8](sel*255.0))

        ori = train_x_eval[-1,i+2,:,:,:]
        # ori = np.stack([ori,0 * gen ,gen], axis = -1)
        misc.imsave(opt.log_dir+'/gen/'+str(epoch)+'/ori_'+str(i+2)+'.png', np.cast[np.uint8](ori*255.0))
        
    for i in range(opt.epoch_size):
        progress.update(i+1)

        train_idx_ = np.random.choice(TOTAL_LENGTH-(opt.n_past+opt.n_future - 1), opt.batch_size)
        m1, _ = sess.run([mse_sel_loss, mse_sel_op])
        print 'mse ', m1

    progress.finish()
    clear_progressbar()
    if epoch % 10 == 0:
        saver.save(sess, opt.log_dir + "/model.ckpt", global_step=epoch)
coord.request_stop()
coord.join(threads)

    
        