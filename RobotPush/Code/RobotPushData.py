# Copyright 2016 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Code for building the input for the prediction model."""

import os

import numpy as np
import tensorflow as tf

from tensorflow.python.platform import gfile
# Original image dimensions
ORIGINAL_WIDTH = 640
ORIGINAL_HEIGHT = 512
COLOR_CHAN = 3

# Default image dimensions.
IMG_WIDTH = 64
IMG_HEIGHT = 64
BATCH_SIZE = 1

def build_tfrecord_input_varilength(length = 15, batch_size = 1, training=True):
  """Create input tfrecord tensors.
  Args:
    training: training or validation data.
  Returns:
    list of tensors corresponding to images, actions, and states. The images
    tensor is 5D, batch x time x height x width x channels. The state and
    action tensors are 3D, batch x time x dimension.
  Raises:
    RuntimeError: if no files found.
  """
  if training:
    filenames = gfile.Glob(os.path.join('../Data/RobotPush/train', '*'))
  else:
    filenames = gfile.Glob(os.path.join('../Data/RobotPush/testnovel', '*'))
  if not filenames:
    raise RuntimeError('No data files found.')
  filenames = filenames
  filename_queue = tf.train.string_input_producer(filenames, shuffle=True)
  reader = tf.TFRecordReader()
  _, serialized_example = reader.read(filename_queue)
  image_seq, state_seq, action_seq = [], [], []
  for i in range(length):
    image_name = 'move/' + str(i) + '/image/encoded'
    action_name = 'move/' + str(i) + '/commanded_pose/vec_pitch_yaw'
    state_name = 'move/' + str(i) + '/endeffector/vec_pitch_yaw'
    feature = {image_name: tf.FixedLenFeature([1], tf.string)}
    features = tf.parse_single_example(serialized_example, features=feature)
    # print features
    image_buffer = tf.reshape(features[image_name], shape=[])
    image = tf.image.decode_jpeg(image_buffer, channels=COLOR_CHAN)
    image = tf.expand_dims(image,0)
    if IMG_HEIGHT != IMG_WIDTH:
      raise ValueError('Unequal height and width unsupported')
    crop_size = min(ORIGINAL_HEIGHT, ORIGINAL_WIDTH)
    image = tf.image.resize_bicubic(image, [IMG_HEIGHT, IMG_WIDTH])
    image = tf.cast(image, tf.float32) / 255.0
    image_seq.append(image)
  
  image_seq = tf.concat(axis=0, values=image_seq)
  image_batch = tf.train.batch(
      [image_seq],
      batch_size,
      num_threads=batch_size,
      capacity=100 * batch_size)
  return image_batch
