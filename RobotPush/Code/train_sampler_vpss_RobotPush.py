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
parser.add_argument('--n_past', type=int, default=2, help='number of frames to condition on')
parser.add_argument('--n_future', type=int, default=3, help='number of frames to predict during training')

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
# discriminator = model.video_D(opt.g_dim, name = 'discriminator_new')
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
# train_x = tf.transpose(train_x, [0,1,2,3,4])
print 'input_x,',input_x.get_shape()
train_idx = tf.placeholder(dtype = tf.int32, shape = (opt.batch_size))
train_x_array = []
for i in range(opt.batch_size):
    tmp = tf.stack([tf.expand_dims(input_x[i,train_idx[i]], axis = 0),\
                    tf.expand_dims(input_x[i,train_idx[i]+1], axis = 0),\
                    tf.expand_dims(input_x[i,train_idx[i]+2], axis = 0),\
                    tf.expand_dims(input_x[i,train_idx[i]+3], axis = 0),\
                    tf.expand_dims(input_x[i,train_idx[i]+4], axis = 0)], axis = 1)
    print tmp.get_shape()
    train_x_array.append(tmp)
train_x = tf.concat(train_x_array, axis = 0)
print 'train_x,',train_x.get_shape()
train_x.set_shape((opt.batch_size, opt.n_past + opt.n_future, opt.image_width, opt.image_width, opt.channels))
print 'train_x,',train_x.get_shape()

msex_loss = 0
msef_loss = 0
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
    mse_array = [tf.reduce_mean(tf.squared_difference(cur_pred, train_x[:,i]), axis = [-3,-2,-1], keepdims = True) for cur_pred in x_pred_array]
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
    real_logits = discriminator.forward(tf.concat([train_x[:,i], train_x[:,i-1], train_x[:,i-2]], axis = -1))
    fake_logits = discriminator.forward(tf.concat([final_pred, train_x[:,i-1], train_x[:,i-2]], axis = -1))
    _, h_next, _ = encoder.forward(train_x[:,i])
    feat_real_logits = feat_D.forward(h_next)
    feat_fake_logits = feat_D.forward(final_feat)
    dx_loss_real += tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = real_logits, labels = tf.ones_like(real_logits))) / (opt.n_past+opt.n_future)
    dx_loss_fake += tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = fake_logits, labels = tf.zeros_like(fake_logits))) / (opt.n_past+opt.n_future)
    gx_loss += tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = fake_logits, labels = tf.ones_like(fake_logits))) / (opt.n_past+opt.n_future)

    df_loss_real += tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = feat_real_logits, labels = tf.ones_like(feat_real_logits))) / (opt.n_past+opt.n_future)
    df_loss_fake += tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = feat_fake_logits, labels = tf.zeros_like(feat_fake_logits))) / (opt.n_past+opt.n_future)
    gf_loss += tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = feat_fake_logits, labels = tf.ones_like(feat_fake_logits))) / (opt.n_past+opt.n_future)
    # msef_loss += tf.reduce_mean(tf.abs(h_next - final_feat))
    msex_loss += tf.reduce_mean(tf.abs(final_pred - train_x[:,i]))
    pred_array.append(final_pred)
    gen0_array.append(x_pred_array[0])
    gen1_array.append(x_pred_array[1])
    gen2_array.append(x_pred_array[2])
    gen3_array.append(x_pred_array[3])
    gen4_array.append(x_pred_array[4])
dx_loss = 0.5 * dx_loss_real + 0.5 * dx_loss_fake 
df_loss = 0.5 * df_loss_real + 0.5 * df_loss_fake
all_vars = tf.trainable_variables()
generatorx_variables = [var for var in all_vars if var.name.startswith('encoder') or \
                                                  var.name.startswith('decoder')]
generatorf_variables = [var for var in all_vars if var.name.startswith('adap_conv') or \
                                                  var.name.startswith('kernel_G')]
discriminatorx_variables = [var for var in all_vars if var.name.startswith('discriminator')]
discriminatorf_variables = [var for var in all_vars if var.name.startswith('feat_D')]
# print generator_variables
genx_op = generator_optimizer.minimize(gx_loss + 100 * msex_loss, var_list = generatorx_variables)
disx_op = diccriminator_optimizer.minimize(dx_loss, var_list = discriminatorx_variables) 
genf_op = generator_optimizer.minimize(gf_loss, var_list = generatorf_variables)
disf_op = diccriminator_optimizer.minimize(df_loss, var_list = discriminatorf_variables) 

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
train_x_eval = sess.run(train_x, feed_dict = {train_idx:train_idx_})
print 'hh'
for epoch in range(opt.niter):
    epoch_mse = 0
    epoch_kld = 0
    progress = progressbar.ProgressBar(maxval=opt.epoch_size).start()

    train_idx_ = np.random.choice(TOTAL_LENGTH-(opt.n_past+opt.n_future - 1), opt.batch_size)
    pred_array_eval, gen0_array_eval, gen1_array_eval, gen2_array_eval, gen3_array_eval, gen4_array_eval, train_x_eval, kernel_eval, h_eval, h_next_eval, out_eval, h_pred_eval \
    = sess.run([pred_array, gen0_array, gen1_array, gen2_array, gen3_array, gen4_array, train_x, kernel, h, h_next, out, h_pred], feed_dict = {train_idx:train_idx_})
    # print 'warm up coff:', wu
    if not os.path.exists(opt.log_dir+'/gen/'+str(epoch)):
        os.mkdir(opt.log_dir+'/gen/'+str(epoch))
    for i in range((opt.n_past+opt.n_future-2)):
        gen = gen0_array_eval[i][-1,:,:,:]
        # gen = np.stack([gen,gen,gen], axis = -1)
        misc.imsave(opt.log_dir+'/gen/'+str(epoch)+'/gen0_'+str(i)+'.png', np.cast[np.uint8](gen*255.0))
        gen = gen1_array_eval[i][-1,:,:,:]
        # gen = np.stack([gen,gen,gen], axis = -1)
        misc.imsave(opt.log_dir+'/gen/'+str(epoch)+'/gen1_'+str(i)+'.png', np.cast[np.uint8](gen*255.0))
        gen = gen2_array_eval[i][-1,:,:,:]
        # gen = np.stack([gen,gen,gen], axis = -1)
        misc.imsave(opt.log_dir+'/gen/'+str(epoch)+'/gen2_'+str(i)+'.png', np.cast[np.uint8](gen*255.0))
        gen = gen3_array_eval[i][-1,:,:,:]
        # gen = np.stack([gen,gen,gen], axis = -1)
        misc.imsave(opt.log_dir+'/gen/'+str(epoch)+'/gen3_'+str(i)+'.png', np.cast[np.uint8](gen*255.0))
        gen = gen4_array_eval[i][-1,:,:,:]
        # gen = np.stack([gen,gen,gen], axis = -1)
        misc.imsave(opt.log_dir+'/gen/'+str(epoch)+'/gen4_'+str(i)+'.png', np.cast[np.uint8](gen*255.0))

        gen = pred_array_eval[i][-1,:,:,:]
        # gen = np.stack([gen,gen,gen], axis = -1)
        misc.imsave(opt.log_dir+'/gen/'+str(epoch)+'/fin_'+str(i)+'.png', np.cast[np.uint8](gen*255.0))

        ori = train_x_eval[-1,i,:,:,:]
        # ori = np.stack([ori,0 * gen ,gen], axis = -1)
        misc.imsave(opt.log_dir+'/gen/'+str(epoch)+'/ori_'+str(i)+'.png', np.cast[np.uint8](ori*255.0))
        ori = train_x_eval[-1,i+1,:,:,:]
        # ori = np.stack([ori,0 * gen ,gen], axis = -1)
        misc.imsave(opt.log_dir+'/gen/'+str(epoch)+'/ori_'+str(i+1)+'.png', np.cast[np.uint8](ori*255.0))
        ori = train_x_eval[-1,i+2,:,:,:]
        # ori = np.stack([ori,0 * gen ,gen], axis = -1)
        misc.imsave(opt.log_dir+'/gen/'+str(epoch)+'/ori_'+str(i+2)+'.png', np.cast[np.uint8](ori*255.0))
        
    for i in range(opt.epoch_size):
        progress.update(i+1)
        train_idx_ = np.random.choice(TOTAL_LENGTH-(opt.n_past+opt.n_future - 1), opt.batch_size)
        d1, d2, d3, d4, mask_eval, _, _ = sess.run([dx_loss_real, dx_loss_fake, df_loss_real, df_loss_fake, mask, disx_op, disf_op], feed_dict = {train_idx:train_idx_})
        print 'dis ', d1, d2, d3, d4, mask_eval[:,0,0,0,0]

        train_idx_ = np.random.choice(TOTAL_LENGTH-(opt.n_past+opt.n_future - 1), opt.batch_size)
        g1, g2, g3, mask_eval, _, _ = sess.run([gx_loss, gf_loss, msex_loss, mask, genx_op, genf_op], feed_dict = {train_idx:train_idx_})
        print 'gen ', g1, g2, g3, mask_eval[:,0,0,0,0]

    progress.finish()
    clear_progressbar()
    # print('[%02d] mse loss: %.5f (%d)' % (epoch, epoch_mse/opt.epoch_size, epoch*opt.epoch_size*opt.batch_size))
    if epoch % 10 == 0:
        saver.save(sess, opt.log_dir + "/model.ckpt", global_step=epoch)
coord.request_stop()
coord.join(threads)

    
        