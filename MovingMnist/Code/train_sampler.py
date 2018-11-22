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
parser.add_argument('--n_future', type=int, default=3, help='number of frames to predict during training')
parser.add_argument('--model', default='dcgan', help='model type (dcgan | vgg)')

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

train_x = tf.placeholder(dtype = tf.float32, shape = (opt.batch_size, opt.n_past + opt.n_future, opt.image_width, opt.image_width, opt.channels))

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
    real_logits = discriminator.forward(tf.concat([train_x[:,i], train_x[:,i-1], train_x[:,i-2]], axis = -1))
    fake_logits = discriminator.forward(tf.concat([final_pred, train_x[:,i-1], train_x[:,i-2]], axis = -1))
    d_loss_real += tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = real_logits, labels = tf.ones_like(real_logits))) / (opt.n_past+opt.n_future)
    d_loss_fake += tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = fake_logits, labels = tf.zeros_like(fake_logits))) / (opt.n_past+opt.n_future)
    g_loss += tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = fake_logits, labels = tf.ones_like(fake_logits))) / (opt.n_past+opt.n_future)
    pred_array.append(final_pred)
    gen0_array.append(x_pred_array[0])
    gen1_array.append(x_pred_array[1])
    gen2_array.append(x_pred_array[2])
    gen3_array.append(x_pred_array[3])
    gen4_array.append(x_pred_array[4])
d_loss = 0.5 * d_loss_real + 0.5 * d_loss_fake
all_vars = tf.trainable_variables()
generator_variables = [var for var in all_vars if var.name.startswith('encoder') or \
                                                  var.name.startswith('decoder') or \
                                                  var.name.startswith('adap_conv') or \
                                                  var.name.startswith('kernel_G') ]
discriminator_variables = [var for var in all_vars if var.name.startswith('discriminator')]
gen_op = generator_optimizer.minimize(g_loss, var_list = generator_variables)
dis_op = diccriminator_optimizer.minimize(d_loss, var_list = discriminator_variables) 

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
ckpt = tf.train.get_checkpoint_state(opt.log_dir)
if ckpt and ckpt.model_checkpoint_path:
    print "Model Restoring..."
    saver.restore(sess, ckpt.model_checkpoint_path)
wu = 0
for epoch in range(opt.niter):
    epoch_mse = 0
    epoch_kld = 0
    progress = progressbar.ProgressBar(maxval=opt.epoch_size).start()

    train_batch = get_training_batch()
    pred_array_eval, gen0_array_eval, gen1_array_eval, gen2_array_eval, gen3_array_eval, gen4_array_eval, kernel_eval \
    = sess.run([pred_array, gen0_array, gen1_array, gen2_array, gen3_array, gen4_array, kernel], feed_dict = {train_x:train_batch})

    if not os.path.exists(opt.log_dir+'/gen/'+str(epoch)):
        os.mkdir(opt.log_dir+'/gen/'+str(epoch))
    io.savemat(opt.log_dir+'/gen/'+str(epoch)+'/kernel.mat',{'k0':kernel_eval[0],'k1':kernel_eval[1],'k2':kernel_eval[2],'k3':kernel_eval[3],'k4':kernel_eval[4]})
    for i in range((opt.n_past+opt.n_future-2)):
        if not os.path.exists(opt.log_dir+'/gen/'+str(epoch)+'/'+str(i)):
            os.mkdir(opt.log_dir+'/gen/'+str(epoch)+'/'+str(i))
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
        ori = np.stack([ori,0 * gen ,gen], axis = -1)
        misc.imsave(opt.log_dir+'/gen/'+str(epoch)+'/'+str(i)+'/ori_'+str(i)+'.png', np.cast[np.uint8](ori*255.0))
        ori = train_batch[-1,i+1,:,:,0]
        ori = np.stack([ori,0 * gen ,gen], axis = -1)
        misc.imsave(opt.log_dir+'/gen/'+str(epoch)+'/'+str(i)+'/ori_'+str(i+1)+'.png', np.cast[np.uint8](ori*255.0))
        ori = train_batch[-1,i+2,:,:,0]
        ori = np.stack([ori,0 * gen ,gen], axis = -1)
        misc.imsave(opt.log_dir+'/gen/'+str(epoch)+'/'+str(i)+'/ori_'+str(i+2)+'.png', np.cast[np.uint8](ori*255.0))
        
    for i in range(opt.epoch_size):
        progress.update(i+1)
        train_batch = get_training_batch()
        dr_eval, df_eval, mask_eval, _ = sess.run([d_loss_real, d_loss_fake, mask, dis_op], feed_dict = {train_x:train_batch})
        print 'dis ', dr_eval, df_eval, mask_eval[:,0,0,0,0]

        train_batch = get_training_batch()
        g_eval, mask_eval, _ = sess.run([g_loss, mask, gen_op], feed_dict = {train_x:train_batch})
        print 'gen ', g_eval, mask_eval[:,0,0,0,0]

    progress.finish()
    utils.clear_progressbar()
    if epoch % 10 == 0:
        saver.save(sess, opt.log_dir + "/model.ckpt", global_step=epoch)

    
        
