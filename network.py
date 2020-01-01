import tensorflow as tf
import config as cfg
import os
from read_data import Reader
import numpy as np


slim = tf.contrib.slim


class Net(object):
    def __init__(self, is_training=True):

        self.size = cfg.TARGET_SIZE

        self.data_path = cfg.DATA_PATH

        self.model_path = cfg.MODEL_PATH

        self.epoches = cfg.EPOCHES

        self.batches = cfg.BATCHES

        self.lr = cfg.LEARNING_RATE

        self.wd = cfg.WEIGHT_DECAY

        self.batch_size = cfg.BATCH_SIZE

        self.cls_num = cfg.N_CLASSES

        self.reader = Reader()

        self.keep_rate = cfg.KEEP_RATE

        self.is_training = is_training

        self.model_name = cfg.MODEL_NAME

        self.growth_rate = cfg.GROWTH_RATE

        self.x = tf.placeholder(tf.float32, [None, self.size, self.size, 3])

        self.y = tf.placeholder(tf.float32, [None, self.cls_num])

        self.y_hat = self.densenet(self.x)

        self.loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.y, logits=self.y_hat))

        self.saver = tf.train.Saver()

        self.acc = self.calculate_acc(self.y, self.y_hat)

    def calculate_acc(self, labels, logits):

        right_pred = tf.equal(tf.argmax(labels, axis=-1),
                              tf.argmax(logits, axis=-1))

        accuracy = tf.reduce_mean(tf.cast(right_pred, tf.float32))

        return accuracy

    def densenet(self, inputs):

        net = inputs

        with tf.variable_scope('DenseNet'):

            with slim.arg_scope([slim.conv2d],
                                activation_fn=tf.nn.relu,
                                weights_regularizer=slim.l2_regularizer(self.wd)):

                # Convolution
                net = slim.conv2d(net, self.growth_rate*2, [7, 7],
                                  2, scope='conv7x7', padding='SAME')

                # Pooling
                net = slim.max_pool2d(
                    net, [2, 2], scope='pool1', padding='SAME')

                # Dense Block 1
                net = self.Dense_Block(net, block_num=6, scope='DenseBlock_1')

                # Transition Layer 1
                net = self.Transition_Layer(net, scope='TransitionLayer_1')

                # Dense Block 2
                net = self.Dense_Block(net, block_num=6, scope='DenseBlock_2')

                # Transition Layer 2
                net = self.Transition_Layer(net, scope='TransitionLayer_2')

                net = tf.layers.batch_normalization(
                    net, trainable=self.is_training, name='BN_block2')

                # Dense Block 3
                net = self.Dense_Block(net, block_num=6, scope='DenseBlock_3')

                # Transition Layer 3
                net = self.Transition_Layer(net, scope='TransitionLayer_3')

                net = tf.layers.batch_normalization(
                    net, trainable=self.is_training, name='BN_block3')

                # Dense Block 4
                net = self.Dense_Block(net, block_num=6, scope='DenseBlock_4')

                # Global Average Pool
                net = slim.avg_pool2d(
                    net, [7, 7], 1, scope='pool7x7', padding='SAME')

                # Fc
                net = tf.layers.flatten(net)
                net = tf.layers.dense(net, 1000)

                # Drop Out
                if self.is_training:
                    net = tf.layers.dropout(inputs=net, rate=1-self.keep_rate)

                # Softmax
                net = tf.layers.dense(net, self.cls_num)

                return tf.nn.softmax(net)

    def Dense_Block(self, inputs, scope, block_num=6):

        net = inputs

        with tf.variable_scope(scope):

            for k in range(block_num):

                net = slim.conv2d(inputs, self.growth_rate, [1, 1],
                                  scope='conv1x1_{}'.format(k), padding='SAME')

                net = slim.conv2d(inputs, self.growth_rate, [3, 3],
                                  scope='conv3x3_{}'.format(k), padding='SAME')

        return net

    def Transition_Layer(self, inputs, scope):

        net = inputs

        with tf.variable_scope(scope):

            net = slim.conv2d(inputs, self.growth_rate, [1, 1],
                              scope='conv1x1', padding='SAME')

            net = slim.max_pool2d(
                net, [2, 2], scope='pool2x2', padding='SAME')

        return net

    def train_net(self):

        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)

        self.optimizer = tf.compat.v1.train.AdamOptimizer(self.lr)

        self.train_step = self.optimizer.minimize(self.loss)

        with tf.Session() as sess:

            sess.run(tf.compat.v1.global_variables_initializer())

            ckpt = tf.train.get_checkpoint_state(self.model_path)

            if ckpt and ckpt.model_checkpoint_path:
                # 如果保存过模型，则在保存的模型的基础上继续训练
                self.saver.restore(sess, ckpt.model_checkpoint_path)
                print('Model Reload Successfully!')

            for epoch in range(self.epoches):

                loss_list = []

                for batch in range(self.batches):

                    data = self.reader.generate(self.batch_size)

                    feed_dict = {
                        self.x: data['images'],
                        self.y: data['labels']
                    }

                    _, loss = sess.run([self.train_step, self.loss], feed_dict)

                    loss_list.append(loss)

                mean_loss = np.mean(np.array(loss_list))

                acc_list = []

                for _ in range(10):

                    test_data = self.reader.generate_test(batch_size=32)

                    test_dict = {
                        self.x: test_data['images'],
                        self.y: test_data['labels']
                    }

                    acc = sess.run(self.acc, test_dict)
                    acc_list.append(acc)

                acc = np.mean(np.array(acc_list))

                mean_loss = str(mean_loss)

                print('Epoch:{} Loss:{} Acc:{}'.format(epoch, mean_loss, acc))

                with open('./losses.txt', 'a') as f:

                    f.write(mean_loss+'\n')

                self.saver.save(sess, self.model_name)


if __name__ == "__main__":

    net = Net(is_training=True)

    net.train_net()
