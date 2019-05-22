import numpy as np 
import cv2
import tensorflow as tf
import os
import random
import math
import sys
from heatmap import *
import vgg16


class MCNN(vgg16.Vgg16):
    def __init__(self, dataset):
        os.system('rm -rf logs')
        
        self.data_dict = np.load('vgg16.npy', encoding='latin1').item()
        self.dataset = dataset
        self.LEARNING_RATE = 1e-5
        
        self.x = tf.placeholder(tf.float32, [None, None, None, 1])
        self.y_act = tf.placeholder(tf.float32, [None, None, None, 1])
        self.y_pre = self.inf(self.x)

        self.loss = tf.sqrt(tf.reduce_mean(tf.square(self.y_act - self.y_pre)))
        tf.summary.scalar('loss', self.loss)
        
        self.act_sum = tf.reduce_sum(self.y_act)
        self.pre_sum = tf.reduce_sum(self.y_pre)
        self.MAE = tf.abs(self.act_sum - self.pre_sum)
        tf.summary.scalar('MAE', self.MAE)

        self.train_step = tf.train.AdamOptimizer(self.LEARNING_RATE).minimize(self.loss)
        #self.train_step = tf.train.GradientDescentOptimizer(self.LEARNING_RATE).minimize(self.loss)
        
    def data_pre_train(self, kind, dataset):
        #########################################################
        
        if dataset == 'A' or dataset == 'B':
            img_path = './data/formatted_trainval/shanghaitech_part_' + dataset + '_patches_9/' + kind + '/'
            den_path = './data/formatted_trainval/shanghaitech_part_' + dataset + '_patches_9/' + kind + '_den/'
        else:
            img_path = './data/' + dataset + '/' + kind + '/'
            den_path = './data/' + dataset + '/' + kind + '_den/'
        
        #########################################################

        print('loading', kind, 'data from dataset', dataset, '...')
        img_names = os.listdir(img_path)
        img_num = len(img_names)

        data = []
        for i in range(1, img_num + 1):
            if i % 100 == 0:
                print(i, '/', img_num)
            name = img_names[i - 1]
            img = cv2.imread(img_path + name, 0)
            img = np.array(img)
            img = (img - 127.5) / 128
            img_flip = np.flip(img, 1)
            den = np.loadtxt(open(den_path + name[:-4] + '.csv'), delimiter = ",")
            den_quarter = np.zeros((int(den.shape[0] / 8), int(den.shape[1] / 8)))
            ##### ucsd need +1
            #den_quarter = np.zeros((int(den.shape[0] / 8 + 1), int(den.shape[1] / 8 + 1)))
            for i in range(len(den_quarter)):
            #for i in range(len(den_quarter) - 1):
                for j in range(len(den_quarter[0])):
                #for j in range(len(den_quarter[0]) - 1):
                    for p in range(8):
                        for q in range(8):
                            den_quarter[i][j] += den[i * 8 + p][j * 8 + q]
            den_flip = np.flip(den_quarter, 1)
            data.append([img, den_quarter])
            data.append([img_flip, den_flip])
        print('load', kind, 'data from dataset', dataset, 'finished')
        return data
        
    def data_pre_test(self, dataset):
        #########################################################

        if dataset == 'A' or dataset == 'B':        
            img_path = './data/original/shanghaitech/part_' + dataset + '_final/test_data/images/'
            den_path = './data/original/shanghaitech/part_' + dataset + '_final/test_data/ground_truth_csv/'
        else:        
            img_path = './data/' + dataset + '/test/'
            den_path = './data/' + dataset + '/test_den/'

        #########################################################
        print('loading test data from dataset', dataset, '...')
        img_names = os.listdir(img_path)
        img_num = len(img_names)

        data = []

        #########################################################
        #for i in range(1, img_num + 1):
        #for i in range(601, 1201):
        for i in range(img_num):
        #########################################################
            if i % 50 == 0:
                print(i, '/', img_num)
            #########################################################
            #name = 'IMG_' + str(i) + '.jpg'
            name = img_names[i]
            #########################################################
            img = cv2.imread(img_path + name, 0)
            img = np.array(img)
            img = (img - 127.5) / 128
            den = np.loadtxt(open(den_path + name[:-4] + '.csv'), delimiter = ",")
            den_sum = np.sum(den)
            data.append([img, den_sum])

            #if i < 100 and i > 0:
                #heatmap(den, i, dataset, 'act')
        print('load test data from dataset', dataset, 'finished')
        return data
        
    def conv2d(self, x, w):
        return tf.nn.conv2d(x, w, strides = [1, 1, 1, 1], padding = 'SAME')

    def max_pool_2x2(self, x):
        return tf.nn.max_pool(x, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')
        
    def vgg_param(self, x):
        #return tf.Variable(x)
        return tf.constant(x)
    
    def inf(self, x):
        tf.summary.image('x', x)
        
        with tf.name_scope('Snet'):
            # s net ###########################################################
            w_conv1_1 = tf.get_variable('w_conv1_1', [5, 5, 1, 24])
            b_conv1_1 = tf.get_variable('b_conv1_1', [24])
            h_conv1_1 = tf.nn.relu(self.conv2d(x, w_conv1_1) + b_conv1_1)    

            h_pool1_1 = self.max_pool_2x2(h_conv1_1)
            tf.summary.image('snet_h_pool1', h_pool1_1[:,:,:,:1])

            w_conv2_1 = tf.get_variable('w_conv2_1', [3, 3, 24, 48])
            b_conv2_1 = tf.get_variable('b_conv2_1', [48])
            h_conv2_1 = tf.nn.relu(self.conv2d(h_pool1_1, w_conv2_1) + b_conv2_1)

            h_pool2_1 = self.max_pool_2x2(h_conv2_1)

            w_conv3_1 = tf.get_variable('w_conv3_1', [3, 3, 48, 48])
            b_conv3_1 = tf.get_variable('b_conv3_1', [48])
            h_conv3_1 = tf.nn.relu(self.conv2d(h_pool2_1, w_conv3_1) + b_conv3_1)
            
            h_pool3_1 = self.max_pool_2x2(h_conv3_1)

            w_conv4_1 = tf.get_variable('w_conv4_1', [3, 3, 48, 24])
            b_conv4_1 = tf.get_variable('b_conv4_1', [24])
            h_conv4_1 = tf.nn.relu(self.conv2d(h_pool3_1, w_conv4_1) + b_conv4_1)

            w_conv5_1 = tf.get_variable('w_conv5_1', [3, 3, 24, 12])
            b_conv5_1 = tf.get_variable('b_conv5_1', [12])
            h_conv5_1 = tf.nn.relu(self.conv2d(h_conv4_1, w_conv5_1) + b_conv5_1)
            tf.summary.image('snet_h_conv5', h_conv5_1[:,:,:,:1])
        
        with tf.name_scope('Mnet'):
            # m net ###########################################################
            w_conv1_2 = tf.get_variable('w_conv1_2', [7, 7, 1, 20])
            b_conv1_2 = tf.get_variable('b_conv1_2', [20])
            h_conv1_2 = tf.nn.relu(self.conv2d(x, w_conv1_2) + b_conv1_2)

            h_pool1_2 = self.max_pool_2x2(h_conv1_2)
            #tf.summary.image('h_pool1_2', h_pool1_2[:,:,:,:1])

            w_conv2_2 = tf.get_variable('w_conv2_2', [5, 5, 20, 40])
            b_conv2_2 = tf.get_variable('b_conv2_2', [40])
            h_conv2_2 = tf.nn.relu(self.conv2d(h_pool1_2, w_conv2_2) + b_conv2_2)

            h_pool2_2 = self.max_pool_2x2(h_conv2_2)

            w_conv3_2 = tf.get_variable('w_conv3_2', [5, 5, 40, 40])
            b_conv3_2 = tf.get_variable('b_conv3_2', [40])
            h_conv3_2 = tf.nn.relu(self.conv2d(h_pool2_2, w_conv3_2) + b_conv3_2)
            
            h_pool3_2 = self.max_pool_2x2(h_conv3_2)

            w_conv4_2 = tf.get_variable('w_conv4_2', [5, 5, 40, 20])
            b_conv4_2 = tf.get_variable('b_conv4_2', [20])
            h_conv4_2 = tf.nn.relu(self.conv2d(h_pool3_2, w_conv4_2) + b_conv4_2)

            w_conv5_2 = tf.get_variable('w_conv5_2', [5, 5, 20, 10])
            b_conv5_2 = tf.get_variable('b_conv5_2', [10])
            h_conv5_2 = tf.nn.relu(self.conv2d(h_conv4_2, w_conv5_2) + b_conv5_2)
            #tf.summary.image('mnet_h_conv5', h_conv5_2[:,:,:,:1])
        
        with tf.name_scope('Lnet'):
            #l net ###########################################################
            w_conv1_3 = tf.get_variable('w_conv1_3', [9, 9, 1, 16])
            b_conv1_3 = tf.get_variable('b_conv1_3', [16])
            h_conv1_3 = tf.nn.relu(self.conv2d(x, w_conv1_3) + b_conv1_3)

            h_pool1_3 = self.max_pool_2x2(h_conv1_3)
            #tf.summary.image('h_pool1_3', h_pool1_3[:,:,:,:1])

            w_conv2_3 = tf.get_variable('w_conv2_3', [7, 7, 16, 32])
            b_conv2_3 = tf.get_variable('b_conv2_3', [32])
            h_conv2_3 = tf.nn.relu(self.conv2d(h_pool1_3, w_conv2_3) + b_conv2_3)

            h_pool2_3 = self.max_pool_2x2(h_conv2_3)

            w_conv3_3 = tf.get_variable('w_conv3_3', [7, 7, 32, 32])
            b_conv3_3 = tf.get_variable('b_conv3_3', [32])
            h_conv3_3 = tf.nn.relu(self.conv2d(h_pool2_3, w_conv3_3) + b_conv3_3)
            
            h_pool3_3 = self.max_pool_2x2(h_conv3_3)

            w_conv4_3 = tf.get_variable('w_conv4_3', [7, 7, 32, 16])
            b_conv4_3 = tf.get_variable('b_conv4_3', [16])
            h_conv4_3 = tf.nn.relu(self.conv2d(h_pool3_3, w_conv4_3) + b_conv4_3)
            
            w_conv5_3 = tf.get_variable('w_conv5_3', [7, 7, 16, 8])
            b_conv5_3 = tf.get_variable('b_conv5_3', [8])
            h_conv5_3 = tf.nn.relu(self.conv2d(h_conv4_3, w_conv5_3) + b_conv5_3)
            #tf.summary.image('lnet_h_conv5', h_conv5_3[:,:,:,:1])
            
        with tf.name_scope('VGG'):
            
            # vgg #############################################################
            
            x3 = tf.concat([x, x, x], 3)
            
            w_conv1 = self.vgg_param(self.data_dict['conv1_1'][0])
            b_conv1 = self.vgg_param(self.data_dict['conv1_1'][1])
            h_conv1 = tf.nn.relu(self.conv2d(x3, w_conv1) + b_conv1)
            
            w_conv2 = self.vgg_param(self.data_dict['conv1_2'][0])
            b_conv2 = self.vgg_param(self.data_dict['conv1_2'][1])
            h_conv2 = tf.nn.relu(self.conv2d(h_conv1, w_conv2) + b_conv2)
            
            h_pool2 = self.max_pool_2x2(h_conv2)
            tf.summary.image('vgg_h_pool2', h_pool2[:,:,:,:1])
            
            w_conv3 = self.vgg_param(self.data_dict['conv2_1'][0])
            b_conv3 = self.vgg_param(self.data_dict['conv2_1'][1])
            h_conv3 = tf.nn.relu(self.conv2d(h_pool2, w_conv3) + b_conv3)
            
            w_conv4 = self.vgg_param(self.data_dict['conv2_2'][0])
            b_conv4 = self.vgg_param(self.data_dict['conv2_2'][1])
            h_conv4 = tf.nn.relu(self.conv2d(h_conv3, w_conv4) + b_conv4)
            
            h_pool4 = self.max_pool_2x2(h_conv4)
            
            w_conv5 = self.vgg_param(self.data_dict['conv3_1'][0])
            b_conv5 = self.vgg_param(self.data_dict['conv3_1'][1])
            h_conv5 = tf.nn.relu(self.conv2d(h_pool4, w_conv5) + b_conv5)
            
            w_conv6 = self.vgg_param(self.data_dict['conv3_2'][0])
            b_conv6 = self.vgg_param(self.data_dict['conv3_2'][1])
            h_conv6 = tf.nn.relu(self.conv2d(h_conv5, w_conv6) + b_conv6)
            
            w_conv7 = self.vgg_param(self.data_dict['conv3_3'][0])
            b_conv7 = self.vgg_param(self.data_dict['conv3_3'][1])
            h_conv7 = tf.nn.relu(self.conv2d(h_conv6, w_conv7) + b_conv7)
            
            h_pool7 = self.max_pool_2x2(h_conv7)
            
            w_conv8 = self.vgg_param(self.data_dict['conv4_1'][0])
            b_conv8 = self.vgg_param(self.data_dict['conv4_1'][1])
            h_conv8 = tf.nn.relu(self.conv2d(h_pool7, w_conv8) + b_conv8)
            
            w_conv9 = self.vgg_param(self.data_dict['conv4_2'][0])
            b_conv9 = self.vgg_param(self.data_dict['conv4_2'][1])
            h_conv9 = tf.nn.relu(self.conv2d(h_conv8, w_conv9) + b_conv9)
            
            w_conv10 = self.vgg_param(self.data_dict['conv4_3'][0])
            b_conv10 = self.vgg_param(self.data_dict['conv4_3'][1])
            h_conv10 = tf.nn.relu(self.conv2d(h_conv9, w_conv10) + b_conv10)
            
            w_conv11 = tf.get_variable('w_conv11', [3, 3, 512, 256])
            b_conv11 = tf.get_variable('b_conv11', [256])
            h_conv11 = tf.nn.relu(self.conv2d(h_conv10, w_conv11) + b_conv11)
            
            w_conv12 = tf.get_variable('w_conv12', [3, 3, 256, 128])
            b_conv12 = tf.get_variable('b_conv12', [128])
            h_conv12 = tf.nn.relu(self.conv2d(h_conv11, w_conv12) + b_conv12)
            
            w_conv13 = tf.get_variable('w_conv13', [3, 3, 128, 64])
            b_conv13 = tf.get_variable('b_conv13', [64])
            h_conv13 = tf.nn.relu(self.conv2d(h_conv12, w_conv13) + b_conv13)
            
            w_conv14 = tf.get_variable('w_conv14', [3, 3, 64, 30])
            b_conv14 = tf.get_variable('b_conv14', [30])
            h_conv14 = tf.nn.relu(self.conv2d(h_conv13, w_conv14) + b_conv14)
            #tf.summary.image('vgg_h_conv14', h_conv14[:,:,:,:1])
        
        
        with tf.name_scope('Final'):
            # merge ###########################################################
            h_conv4_merge = tf.concat([h_conv5_1, h_conv5_2, h_conv5_3, h_conv14], 3)
            #h_conv4_merge = tf.concat([h_conv5_1, h_conv5_2, h_conv5_3], 3)
            
            
            '''
            w_conv5_m = tf.get_variable('w_conv5_m', [1, 1, 60, 120])
            b_conv5_m = tf.get_variable('b_conv5_m', [120])
            h_conv5_m = tf.nn.relu(self.conv2d(h_conv4_merge, w_conv5_m) + b_conv5_m)

            w_conv6_m = tf.get_variable('w_conv6_m', [1, 1, 120, 60])
            b_conv6_m = tf.get_variable('b_conv6_m', [60])
            h_conv6_m = tf.nn.relu(self.conv2d(h_conv5_m, w_conv6_m) + b_conv6_m)
            '''
            w_conv7_m = tf.get_variable('w_conv7_m', [1, 1, 60, 30])
            b_conv7_m = tf.get_variable('b_conv7_m', [30])
            h_conv7_m = self.conv2d(h_conv4_merge, w_conv7_m) + b_conv7_m
            #tf.summary.image('final_h_conv7', h_conv7_m)

            w_conv8_m = tf.get_variable('w_conv8_m', [1, 1, 30, 1])
            b_conv8_m = tf.get_variable('b_conv8_m', [1])
            h_conv8_m = self.conv2d(h_conv7_m, w_conv8_m) + b_conv8_m
            
            y_pre = h_conv8_m

        return y_pre

    def train(self, max_epoch):
        with tf.Session() as sess:
            if not os.path.exists('./model' + self.dataset):
                sess.run(tf.global_variables_initializer())
            else: 
                saver = tf.train.Saver()
                saver.restore(sess, 'model' + self.dataset + '/model.ckpt')
                
            merged = tf.summary.merge_all()
            writer = tf.summary.FileWriter('logs/', sess.graph, flush_secs = 3)
            
            data_train = self.data_pre_train('train', self.dataset)

            #########################################################

            #data_val = self.data_pre_train('val', self.dataset)
            #data_train.extend(data_val)
            
            #########################################################


            data_test = self.data_pre_test(self.dataset)
            
            best_mae = 10000
            best_mse = 10000
            for epoch in range(max_epoch):
                #training process
                epoch_mae = 0
                random.shuffle(data_train)
                for i in range(len(data_train)):
                    data = data_train[i]
                    x_in = np.reshape(data[0], (1, data[0].shape[0], data[0].shape[1], 1))
                    #print('###########################x_in.shape =', x_in.shape)
                    y_ground = np.reshape(data[1], (1, data[1].shape[0], data[1].shape[1], 1))
                    #print('###########################y_ground.shape =', y_ground.shape)
                    
                        
                    _, l, y_a, y_p, act_s, pre_s, m, result = sess.run( \
                        [self.train_step, self.loss, self.y_act, self.y_pre, \
                        self.act_sum, self.pre_sum, self.MAE, merged], \
                        feed_dict = {self.x: x_in, self.y_act: y_ground})
                    writer.add_summary(result, epoch * len(data_train) + i)
                    if i % 1000 == 0:        
                        print('epoch', epoch, 'step', i, 'mae:', m)
                    epoch_mae += m
                epoch_mae /= len(data_train)
                print('epoch', epoch, 'train_mae:', epoch_mae)
                
                mae_test = 0
                mse_test = 0
                for i in range(1, len(data_test) + 1):
                    #if i % 60 == 0:
                    #    print(i, '/', len(data_test))
                    d = data_test[i - 1]
                    x_in = d[0]
                    y_a = d[1]
                    
                    x_in = np.reshape(d[0], (1, d[0].shape[0], d[0].shape[1], 1))
                    y_p_den = sess.run(self.y_pre, feed_dict = {self.x: x_in})

                    y_p = np.sum(y_p_den)

                    mae_test += abs(y_a - y_p)
                    mse_test += (y_a - y_p) * (y_a - y_p)
                    
                mae_test /= len(data_test)
                mse_test = math.sqrt(mse_test / len(data_test))
                print('mae_test: ', mae_test)
                print('mse_test: ', mse_test)
                
                if mae_test < best_mae:
                    best_mae = mae_test
                    best_mse = mse_test
                    print('best mae so far, saving model.')
                    print('##################best mae:', best_mae)
                    print('##################best mse:', best_mse)
                    saver = tf.train.Saver()
                    saver.save(sess, 'model' + self.dataset + '/model.ckpt')
                else:
                    print('##################best mae:', best_mae)
                    print('##################best mse:', best_mse)
                print('**************************')
                
                
            
    def test(self):
        with tf.Session() as sess:
            saver = tf.train.Saver()
            saver.restore(sess, 'model' + self.dataset + '/model.ckpt')
            data = self.data_pre_test(self.dataset)

            mae = 0
            mse = 0

            for i in range(1, len(data) + 1):
                if i % 60 == 0:
                    print(i, '/', len(data))
                d = data[i - 1]
                x_in = d[0]
                y_a = d[1]
                
                x_in = np.reshape(d[0], (1, d[0].shape[0], d[0].shape[1], 1))
                y_p_den = sess.run(self.y_pre, feed_dict = {self.x: x_in})

                y_p = np.sum(y_p_den)

                #if i < 100 and i > 0:
                    #y_p_den = np.reshape(y_p_den, (y_p_den.shape[1], y_p_den.shape[2]))	
                    #heatmap(y_p_den, i, self.dataset, 'pre')
                mae += abs(y_a - y_p)
                mse += (y_a - y_p) * (y_a - y_p)
                
            mae /= len(data)
            mse = math.sqrt(mse / len(data))
            print('mae: ', mae)
            print('mse: ', mse)
            






