#-*-coding:utf-8-*-
#-------------------------------------------------------------------------------
# Name:        
# Purpose:
#
# Author:      BQH
#
# Created:     19/09/2017
# Copyright:   (c) BQH 2017
# Licence:     <your licence>
#-------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import re
import sys
import tarfile
import yaml
import logging.config
from NavieBayes import *
from SupportVectorMachines import *
from SplitText import *
import tensorflow as tf

# parser = argparse.ArgumentParser()
#
# # Basic model parameters.
# parser.add_argument('--batch_size', type=int, default=128,
#                     help='Number of images to process in a batch.')
#
# parser.add_argument('--data_dir', type=str, default='/tmp/cifar10_data',
#                     help='Path to the CIFAR-10 data directory.')
#
# parser.add_argument('--use_fp16', type=bool, default=False,
#                     help='Train the model using fp16.')
#
# FLAGS = parser.parse_args()
#
# # Global constants describing the CIFAR-10 data set.
# IMAGE_SIZE = cifar10_input.IMAGE_SIZE
NUM_CLASSES = 10
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 0 # cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
#NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL


# Constants describing the training process.
MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 350.0      # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.2       # Initial learning rate.

# If a model is trained with multiple GPUs, prefix all Op names with tower_name
# to differentiate the operations. Note that this prefix is removed from the
# names of the summaries when visualizing a model.
TOWER_NAME = 'tower'

def _activation_summary(x):
  tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
  tf.summary.histogram(tensor_name + '/activations', x)
  tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))


def _variable_on_cpu(name, shape, initializer):
  with tf.device('/cpu:0'):
    dtype = tf.float32
    var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
  return var


def _variable_with_weight_decay(name, shape, stddev, wd):
  dtype = tf.float32
  var = _variable_on_cpu(
      name,
      shape,
      tf.truncated_normal_initializer(stddev=stddev, dtype=dtype))
  if wd is not None:
    weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
    tf.add_to_collection('losses', weight_decay)
  return var

# sample_data: [batch, in_height, in_width, in_channels]
def inference(inputs):
  sample_data = tf.cast(tf.reshape(inputs,
                       [inputs.shape[0], 1, inputs.shape[1], 1]),  # [batch, in_height, in_width, in_channels]
                       tf.float32)
  with tf.variable_scope('conv1') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[1, 1000, 1, 32], #[filter_height, filter_width, in_channels, out_channels]
                                         stddev=5e-2,
                                         wd=0.0)
    conv = tf.nn.conv2d(sample_data, kernel, [1, 1, 50, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [32], tf.constant_initializer(0.0))
    pre_activation = tf.nn.bias_add(conv, biases)
    conv1 = tf.nn.relu(pre_activation, name=scope.name)
    _activation_summary(conv1)

  # pool1
  pool1 = tf.nn.max_pool(conv1, ksize=[1, 1, 100, 1], strides=[1, 1, 2, 1], padding='SAME', name='pool1')
  # norm1
  norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,name='norm1')

  # conv2
  with tf.variable_scope('conv2') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[1, 100, 32, 32],
                                         stddev=5e-2,
                                         wd=0.0)
    conv = tf.nn.conv2d(norm1, kernel, [1, 1, 2, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [32], tf.constant_initializer(0.1))
    pre_activation = tf.nn.bias_add(conv, biases)
    conv2 = tf.nn.relu(pre_activation, name=scope.name)
    _activation_summary(conv2)

  # norm2
  norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm2')
  # pool2
  pool2 = tf.nn.max_pool(norm2, ksize=[1, 1, 10, 1], strides=[1, 1, 5, 1], padding='SAME', name='pool2')

  # local3
  with tf.variable_scope('local3') as scope:
    # Move everything into depth so we can perform a single matrix multiply.
    reshape_data = tf.reshape(pool2, [inputs.shape[0], -1])
    dim = reshape_data.get_shape()[1].value
    weights = _variable_with_weight_decay('weights', shape=[dim, 384],stddev=0.04, wd=0.004)
    biases = _variable_on_cpu('biases', [384], tf.constant_initializer(0.1))
    local3 = tf.nn.relu(tf.matmul(reshape_data, weights) + biases, name=scope.name)
    _activation_summary(local3)

  # local4
  with tf.variable_scope('local4') as scope:
    weights = _variable_with_weight_decay('weights', shape=[384, 192], stddev=0.04, wd=0.004)
    biases = _variable_on_cpu('biases', [192], tf.constant_initializer(0.1))
    local4 = tf.nn.relu(tf.matmul(local3, weights) + biases, name=scope.name)
    _activation_summary(local4)

  with tf.variable_scope('softmax_linear') as scope:
    weights = _variable_with_weight_decay('weights', [192, NUM_CLASSES], stddev=1/192.0, wd=0.0)
    biases = _variable_on_cpu('biases', [NUM_CLASSES], tf.constant_initializer(0.0))
    softmax_linear = tf.add(tf.matmul(local4, weights), biases, name=scope.name)
    _activation_summary(softmax_linear)

  return softmax_linear


def loss(logits, labels):
  labels = tf.cast(labels, tf.int64)
  cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits, name='cross_entropy_per_example')
  cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
  tf.add_to_collection('losses', cross_entropy_mean)

  return tf.add_n(tf.get_collection('losses'), name='total_loss')


def _add_loss_summaries(total_loss):
  loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
  losses = tf.get_collection('losses')
  loss_averages_op = loss_averages.apply(losses + [total_loss])

  for l in losses + [total_loss]:
    tf.summary.scalar(l.op.name + ' (raw)', l)
    tf.summary.scalar(l.op.name, loss_averages.average(l))

  return loss_averages_op


def train(total_loss, global_step):
  lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
                                  global_step,
                                  2000,
                                  0.98,
                                  staircase=True)
  tf.summary.scalar('learning_rate', lr)

  # Generate moving averages of all losses and associated summaries.
  loss_averages_op = _add_loss_summaries(total_loss)

  # Compute gradients.
  with tf.control_dependencies([loss_averages_op]):
    opt = tf.train.GradientDescentOptimizer(lr)
    grads = opt.compute_gradients(total_loss)

  # Apply gradients.
  apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

  # Add histograms for trainable variables.
  for var in tf.trainable_variables():
    tf.summary.histogram(var.op.name, var)

  # Add histograms for gradients.
  for grad, var in grads:
    if grad is not None:
      tf.summary.histogram(var.op.name + '/gradients', grad)

  # Track the moving averages of all trainable variables.
  variable_averages = tf.train.ExponentialMovingAverage(
      MOVING_AVERAGE_DECAY, global_step)
  variables_averages_op = variable_averages.apply(tf.trainable_variables())

  with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
    train_op = tf.no_op(name='train')

  return train_op

def CNN(inputs,labels,root_path,st):
    logger = logging.getLogger(__name__)
    NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = inputs.shape[0]
    with tf.Graph().as_default():
        global_step = tf.contrib.framework.get_or_create_global_step()
        logits = inference(inputs)
        total_loss = loss(logits, tf.cast(labels,dtype=tf.int32))
        train_op = train(total_loss, global_step)
        saver = tf.train.Saver()

        # important step 对所有变量进行初始化
        init = tf.initialize_all_variables()
        sess = tf.Session()
        # 上面定义的都没有运算，直到 sess.run 才会开始运算
        sess.run(init)
        logger.info('start tarinning...')
        startTime = time.time()
        #载入检查点
        ckpt = tf.train.get_checkpoint_state('/home/bqh/Code/Text_categorization/')
        saver.restore(sess, ckpt.model_checkpoint_path)
        # 迭代 20000 次学习，sess.run optimizer
        for step in range(20000):
            sess.run(train_op)
            if step % 1000 == 0:
                saver.save(sess, os.path.join(os.getcwd(), 'my-model'), global_step=step)
                # to see the step improvement
                tempLoss = sess.run(total_loss)
                logger.info('loss: {0}'.format(tempLoss))
                if tempLoss < 0.1:
                    break
        endTime = time.time()
        logger.info('tarinning cost {0}s'.format(endTime - startTime))

        # 开始测试
        testFilePath = os.path.join(root_path, 'tarin_corpus_seg', 'verify')
        for className in os.listdir(testFilePath):
            logger.info('start create class {0} test data...'.format(className))
            trueLable = []
            testData = []
            startTime = time.time()
            testFileSet = os.listdir(os.path.join(testFilePath, className))
            testFileLable = [os.path.join(testFilePath, className, fileName) for fileName in testFileSet]
            trueLable.append([int(className)] * len(testFileSet))
            for testFile in testFileLable:
                testContent = st.ReadFile(testFile)
                testContent = testContent.replace("\r\n", "").strip()
                content = testContent.encode('utf-8')
                testVec = st.CreateDataVec(content.split(), '0')
                testData.append(testVec[0:-1])
            endTime = time.time()
            logger.info('create class {0} data cost {1}s'.format(className, endTime - startTime))

            logger.info('start pridict class {0}...'.format(className))
            temp_data = np.array(testData)
            res = tf.nn.softmax(inference(tf.cast(tf.reshape(temp_data,[temp_data.shape[0],1,temp_data.shape[1],1]),tf.float32)))
            predicted = tf.cast(tf.arg_max(sess.run(res), tf.int32))
            correctRate = sess.run(tf.reduce_mean(tf.cast(tf.equal(predicted, trueLable), tf.float32)))
            logger.info('the class {0} correct rate is {1}%'.format(className, correctRate))


def SetupLogging(default_path='logging.yaml', default_level=logging.INFO, env_key='LOG_CFG'):
    """Setup logging configuration"""
    path = default_path
    value = os.getenv(env_key, None)
    if value:
        path = value
    if os.path.exists(path):
        with open(path, 'rt') as f:
            config = yaml.load(f.read())
        logging.config.dictConfig(config)
    else:
        logging.basicConfig(level=default_level)

def main():
    SetupLogging()
    logger = logging.getLogger(__name__)
    root_path = os.path.abspath(os.curdir)
    logger.info('start create sample data...')
    # 先生成词袋，样本数据集
    wordset_fileName = os.path.join(root_path, 'wordSet.txt')
    st = SplitText(root_path)
    if os.path.exists(wordset_fileName) == False:
        st.CreateDataSet(wordset_fileName)
    else:
        st.wordsetvec = st.GetWordVec(wordset_fileName)
    # 生成计算所需的样本数据集
    sampleData = st.CreateSampleData()
    # logger.info('start create random data...')
    # 随机抽取500条数据用于训练
    randomData = random.sample(sampleData,1000)
    svm = MySVM(np.array(randomData))
    inputs = svm.dataMatrix
    lable = svm.labelVec
    logger.info('start run CNN....')
    CNN(inputs,lable,root_path,st)
    pass

if __name__ == '__main__':
    main()