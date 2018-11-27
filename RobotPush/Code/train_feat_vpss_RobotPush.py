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
parser.add_argument('--n_past', type=int, default=1, help='number of frames to condition on')
parser.add_argument('--n_future', type=int, default=1, help='number of frames to predict during training')

opt = parser.parse_args()
opt.log_dir = '%s/%s' % (opt.log_dir, 'robotpush')

if not os.path.exists('%s/gen/' % opt.log_dir):
    os.makedirs('%s/gen/' % opt.log_dir)
if not os.path.exists('%s/plots/' % opt.log_dir):
    os.makedirs('%s/plots/' % opt.log_dir)
print(opt)

import vgg_64 as model

encoder = model.encoder_G(name = 'encoder')
decoder = model.decoder_G(name = 'decoder')
discriminator = model.encoder_D(opt.g_dim, opt.channels, name = 'discriminator')
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
elif opt.optimizer == 'rmsprop':
    opt.optimizer = tf.train.RMSPropOptimizer

generator_optimizer = opt.optimizer(learning_rate = opt.lr, beta1 = opt.beta1, beta2 = 0.999)
diccriminator_optimizer = opt.optimizer(learning_rate = 0.1 * opt.lr, beta1 = opt.beta1, beta2 = 0.999)



TOTAL_LENGTH = 20
input_x = build_tfrecord_input_varilength(length = TOTAL_LENGTH, batch_size = opt.batch_size, training = True)
# train_x = tf.transpose(train_x, [0,1,2,3,4])
print 'input_x,',input_x.get_shape()
train_idx = tf.placeholder(dtype = tf.int32, shape = (opt.batch_size,2))
train_x_array = []
for i in range(opt.batch_size):
    tmp = tf.stack([tf.expand_dims(input_x[i,train_idx[i,0]], axis = 0), tf.expand_dims(input_x[i,train_idx[i,1]], axis = 0)], axis = 1)
    train_x_array.append(tmp)
train_x = tf.concat(train_x_array, axis = 0)
print 'train_x,',train_x.get_shape()
train_x.set_shape((opt.batch_size, opt.n_past + opt.n_future, opt.image_width, opt.image_width, opt.channels))
print 'train_x,',train_x.get_shape()

mse = 0
kld = 0
pred_array = []
gen0_array = []
gen1_array = []
gen2_array = []
gen3_array = []
gen4_array = []
# h_array = []
d_loss_real = 0
d_loss_fake = 0
g_loss = 0
mse_loss = 0

final_pred = None
hidden_state = tf.zeros((opt.batch_size, 5, opt.image_width, opt.image_width, 32))
for i in range(1):
    p0, h0, s0 = encoder.forward(train_x[:,0])
    p1, h1, s1 = encoder.forward(train_x[:,1])
    x0 = decoder.forward(h0, s1)# should be fake train_x[:,0]
    x1 = decoder.forward(h1, s0)# shoula be fake train_x[:,1]
    real_logits = discriminator.forward(tf.concat([train_x[:,1], train_x[:,0]], axis = -1))
    fake1_logits =  discriminator.forward(tf.concat([x0, x1], axis = -1)) 
    d_loss_real += tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = real_logits, labels = tf.ones_like(real_logits))) 
    d_loss_fake += tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = fake1_logits, labels = tf.zeros_like(fake1_logits))) #+ \
    g_loss += 1 * tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = fake1_logits, labels = tf.ones_like(fake1_logits))) #+ \
    mse_loss += (tf.reduce_mean(tf.abs(x0 - train_x[:,0])) + tf.reduce_mean(tf.abs(x1 - train_x[:,1])))
d_loss = 0.5 * d_loss_real + 0.5 * d_loss_fake
all_vars = tf.trainable_variables()
generator_variables = [var for var in all_vars if var.name.startswith('encoder') or \
                                                  var.name.startswith('decoder')]
discriminator_variables = [var for var in all_vars if var.name.startswith('discriminator')]
# print generator_variables
gen_op = generator_optimizer.minimize(g_loss + 1000 * mse_loss, var_list = generator_variables)
dis_op = diccriminator_optimizer.minimize(d_loss, var_list = discriminator_variables) 

import scipy.misc as misc

config = tf.ConfigProto()  
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
saver = tf.train.Saver()
sess.run(tf.global_variables_initializer())

ckpt = tf.train.get_checkpoint_state(opt.log_dir)
if ckpt and ckpt.model_checkpoint_path:
    print "Model Restoring..."
    saver.restore(sess, ckpt.model_checkpoint_path)

def clear_progressbar():
    # moves up 3 lines
    print("\033[2A")
    # deletes the whole line, regardless of character position
    print("\033[2K")
    # moves up two lines again
    print("\033[2A")

wu = 0
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess, coord)
train_idx_ = np.random.choice(TOTAL_LENGTH, [opt.batch_size, 2])
print 'hh'
train_x_eval = sess.run(train_x, feed_dict = {train_idx:train_idx_})
print 'hh'
for epoch in range(opt.niter):
    epoch_mse = 0
    epoch_kld = 0
    progress = progressbar.ProgressBar(maxval=opt.epoch_size).start()

    train_idx_ = np.random.choice(TOTAL_LENGTH, [opt.batch_size, 2])
    x0_eval, x1_eval, train_x_eval = sess.run([x0, x1, train_x], feed_dict = {train_idx:train_idx_})
    if not os.path.exists(opt.log_dir+'/gen/'+str(epoch)):
        os.mkdir(opt.log_dir+'/gen/'+str(epoch))
    for i in range(1):
        gen = x0_eval[0,:,:,:]
        misc.imsave(opt.log_dir+'/gen/'+str(epoch)+'/x0.png', np.cast[np.uint8](gen*255.0))
        gen = x1_eval[0,:,:,:]
        misc.imsave(opt.log_dir+'/gen/'+str(epoch)+'/x1.png', np.cast[np.uint8](gen*255.0))
        ori = train_x_eval[0,0,:,:,:]
        misc.imsave(opt.log_dir+'/gen/'+str(epoch)+'/x0_gt.png', np.cast[np.uint8](ori*255.0))
        ori = train_x_eval[0,1,:,:,:]
        misc.imsave(opt.log_dir+'/gen/'+str(epoch)+'/x1_gt.png', np.cast[np.uint8](ori*255.0))
        
    for i in range(opt.epoch_size):
        progress.update(i+1)
        train_idx_ = np.random.choice(TOTAL_LENGTH, [opt.batch_size, 2])
        dr_eval, df_eval, _ = sess.run([d_loss_real, d_loss_fake, dis_op], feed_dict = {train_idx:train_idx_})
        print 'dis ', dr_eval, df_eval

        train_idx_ = np.random.choice(TOTAL_LENGTH, [opt.batch_size, 2])
        g_loss_eval, mse_loss_eval, _ = sess.run([g_loss, mse_loss, gen_op], feed_dict = {train_idx:train_idx_})
        print 'gen ', g_loss_eval, mse_loss_eval
    progress.finish()
    clear_progressbar()
    if epoch % 10 == 0:
        saver.save(sess, opt.log_dir + "/model.ckpt", global_step=epoch)
coord.request_stop()
coord.join(threads)

    
        