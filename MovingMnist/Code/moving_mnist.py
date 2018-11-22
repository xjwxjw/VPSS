import numpy as np
import tensorflow as tf
import threading
import time
import scipy.misc as misc
import h5py
import sys

try:
    f = h5py.File('../Data/mnist.h5','r')
except:
    print 'Please set the correct path to MNIST dataset'
    sys.exit()
x1 = f['train'].value.reshape(-1, 28, 28)
x2 = f['test'].value.reshape(-1, 28, 28)
f.close()
data_array = np.zeros((100, 30, 64, 64, 1),dtype=np.float32)
class MyThread(threading.Thread):
    def __init__(self, train, seq_len = 30, num_digits = 1, image_size = 64, data_slice = -1):
        super(MyThread, self).__init__()  
        if train:
            self.data = x1
        else:
            self.data = x2
        self.N = self.data.shape[0]
        self.seq_len = seq_len
        self.num_digits = num_digits
        self.image_size = image_size
        self.digit_size = 28
        self.deterministic = True
        self.channels = 1 
        self.data_slice = data_slice

    def run(self):
        image_size = self.image_size
        digit_size = self.digit_size
        x = np.zeros((self.seq_len,image_size, image_size, self.channels),dtype=np.float32)
        for n in range(self.num_digits):
            idx = np.random.randint(self.N)
            digit = misc.imresize(self.data[idx], [self.digit_size, self.digit_size])

            sx = np.random.randint(image_size-digit_size)
            sy = np.random.randint(image_size-digit_size)
            dx = np.random.randint(-3, 4)
            dy = np.random.randint(-3, 4)
            for t in range(self.seq_len):
                if sy < 0:
                    sy = 0 
                    dy = -dy
                elif sy >= image_size-digit_size:
                    sy = image_size-digit_size-1
                    dy = -dy
                if sx < 0:
                    sx = 0 
                    dx = -dx
                elif sx >= image_size-digit_size:
                    sx = image_size-digit_size-1
                    dx = -dx
                   
                x[t, sy:sy+digit_size, sx:sx+digit_size, 0] += digit
                sy += dy
                sx += dx

        x[x>=1] = 1.
        x[x<0] = 0.
        data_array[self.data_slice, 0:self.seq_len] = x

class mnist(object):
    def __init__(self, train, batch_size = 100, seq_len = 30, num_digits = 1, image_size = 64):
        self.train = train
        self.seq_len = seq_len
        self.num_digits = num_digits
        self.image_size = image_size
        self.batch_size = batch_size
        self.digit_size = 28
        self.deterministic = True
        self.channels = 1 

    def getbatch(self):
        # data_array = np.zeros((100, 15, 64, 64, 1),dtype=np.float32)
        th_pool = []
        for i in range(self.batch_size):
            th = MyThread(self.train, self.seq_len, self.num_digits, self.image_size, i)
            th.start()
            th_pool.append(th)
        for i in range(self.batch_size):
            th_pool[i].join()
        return data_array[0:self.batch_size, 0:self.seq_len]
    