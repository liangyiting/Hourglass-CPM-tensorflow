#coding: utf-8
"""
    Convulotional Pose Machine
        For Single Person Pose Estimation
    Human Pose Estimation Project in Lab of IP
    Author: Liu Fangrui aka mpsk
        Beijing University of Technology
            College of Computer Science & Technology
    Experimental Code
        !!DO NOT USE IT AS DEPLOYMENT!!
"""
import os
import time
import numpy as np
import tensorflow as tf
import PoseNet
import Global
import pdb

class CPM(PoseNet.PoseNet):
    """
    CPM net
    """
    def __init__(self, base_lr=0.0005, in_size=368, out_size=None, batch_size=16, epoch=20, dataset = None, log_dir=None, stage=6,
                 epoch_size=1000, w_summary=True, training=True, remove_joints=None, cpu_only=True, pretrained_model='./log/model.npy',
                 load_pretrained=False, predict=False):
        """ CPM Net implemented with Tensorflow

        :param base_lr:             starter learning rate
        :param in_size:             input image size
        :param batch_size:          size of each batch
        :param epoch:               num of epoch to train
        :param dataset:             *datagen* class to gen & feed data
        :param log_dir:             log directory
        :param stage:               num of stage in cpm model
        :param epoch_size:          size of each epoch
        :param w_summary:           bool to determine if do weight summary
        :param training:            bool to determine if the model trains
        :param joints:              list to define names of joints
        :param cpu_only:            CPU mode or GPU mode
        :param pretrained_model:    Path to pre-trained model
        :param load_pretrained:     bool to determine if the net loads all arg

        ATTENTION HERE:
        *   if load_pretrained is False
            then the model only loads VGG part of arguments
            if true, then it loads all weights & bias

        *   if log_dir is None, then the model won't output any save files
            but PLEASE DONT WORRY, we defines a default log ditectory

        TODO:
            *   Save model as numpy
            *   Predicting codes
            *   PCKh & mAP Test code
        """
        tf.reset_default_graph()
        self.sess = tf.Session()

        #   model log dir control
        if log_dir is not None:
            self.log_dir = log_dir
        else:
            self.log_dir = 'log/'
        self.writer = tf.summary.FileWriter(self.log_dir)

        #   model device control
        self.cpu = '/cpu:0'
        if cpu_only:
            self.gpu = self.cpu
        else:
            self.gpu = '/gpu:0'

        self.dataset = dataset

        #   Annotations Associated
        self.joints = Global.joint_list
        self.joint_num =len(self.joints)

        #   Net Args
        self.stage = stage
        self.training = training
        self.base_lr = base_lr
        self.in_size = in_size
        if out_size is None:
            self.out_size = self.in_size/Global.beishu
        else:
            self.out_size = out_size
        self.batch_size = batch_size
        self.epoch = epoch
        self.epoch_size = epoch_size
        self.dataset = dataset

        #   step learning rate policy
        self.global_step = tf.Variable(0, trainable=False)
        self.learning_rate = tf.train.exponential_decay(base_lr,
            self.global_step, 10*self.epoch*self.epoch_size, 0.333,
            staircase=True)

        #   Inside Variable
        self.train_step = []
        self.losses = []
        self.w_summary = w_summary
        self.net_debug = False

        self.img = None
        self.gtmap = None

        self.summ_scalar_list = []
        self.summ_accuracy_list = []
        self.summ_image_list = []
        self.summ_histogram_list = []

        #   load model
        self.load_pretrained = load_pretrained
        if pretrained_model is not None:
            self.pretrained_model = np.load(pretrained_model, encoding='latin1').item()
            print("[*]\tnumpy file loaded!")
        else:
            self.pretrained_model = None

        #   dictionary of network parameters
        self.var_dict = {}
        self.saver = tf.train.Saver(max_to_keep=2)


    def  net(self,image,name='Hourglass_net'):
        """ CPM Net Structure
        Args:
            image           : Input image with n times of 8
                                size:   batch_size * in_size * in_size * sizeof(RGB)
        Return:
            stacked heatmap : Heatmap NSHWC format
                                size:   batch_size * stage_num * in_size/2* in_size/2* joint_num
        """

        channel_out=512#如果样本的多样性较低，则可以减少channel个数，则参数规模也可以大幅度减少
        channel_out=Global.channel_out
        stage=[None]*self.stage
        with tf.variable_scope(name):
            net = self._feature_extractor(image, net_type=Global.feature_extractor, channel_out=channel_out)  #
            net=self._hourglass(net,n=Global.n_hourglass,numOut=channel_out,name='Hourglass_1')#输入的尺寸必须是2*2^n的倍数
            #pdb.set_trace()
            with tf.variable_scope('Linear'):
                net=self._conv(net,channel_out,1,1,'SAME',name='linear_conv_1')
                net=self._conv(net,channel_out/2,1,1,'SAME',name='linear_conv_2')
            with tf.variable_scope('Output'):
                net=self._conv(net,self.joint_num+1,1,1,'SAME',name='conv_bn_relu')
                stage[0]=net
                output_unactivated=tf.stack(stage,axis=1,name='stack_output')
        return tf.nn.sigmoid(output_unactivated,name='final_output'),output_unactivated

    def _feature_extractor(self, inputs, net_type='VGG', name='Feature_Extractor',lock=False,channel_out=256):
        """ Feature Extractor
        For VGG Feature Extractor down-scale by x8
        For ResNet Feature Extractor downscale by x8 (Current Setup)

        Net use VGG as default setup
        Args:
            inputs      : Input Tensor (Data Format: NHWC)
            name        : Name of the Extractor
        Returns:
            net         : Output Tensor            
        """
        with tf.variable_scope(name):
            if net_type == 'ResNet':
                #适应hourglass的特征图
                net = self._conv_bn_relu(inputs, channel_out/4, 7, 2, 'SAME')#注意这里stride=2，也就是进行了一次降采样
                #   down scale by 2
                net = self._residual(net, channel_out/2, 'r1')
                #net = tf.contrib.layers.max_pool2d(net, [2,2], [2,2], padding='SAME', scope='pool1')
                #   down scale by 2
                net = self._residual(net, channel_out/2, 'r2')
                net = self._residual(net, channel_out, 'r3')
                net = self._residual(net, channel_out, 'r4')
                return net
            else:
                #   VGG based
                kernel_size=3
                use_loaded=False
                net = self._conv_bn_relu(inputs, channel_out/4, kernel_size, 2, 'SAME', 'conv1_1', use_loaded=use_loaded, lock=lock)
                net = self._conv_bn_relu(net, channel_out/4, kernel_size, 1, 'SAME', 'conv1_2', use_loaded=use_loaded, lock=lock)
               # net = tf.contrib.layers.max_pool2d(net, [2,2], [2,2], padding='SAME', scope='pool1')
                #   down scale by 2
                net = self._conv_bn_relu(net, channel_out/2, kernel_size, 1, 'SAME', 'conv2_1', use_loaded=use_loaded, lock=lock)
                net = self._conv_bn_relu(net, channel_out/2, kernel_size, 1, 'SAME', 'conv2_2', use_loaded=use_loaded, lock=lock)
               # net = tf.contrib.layers.max_pool2d(net, [2,2], [2,2], padding='SAME', scope='pool2')
                #   down scale by 2
                net = self._conv_bn_relu(net, channel_out, 3, 1, 'SAME', 'conv3_1', use_loaded=use_loaded, lock=lock)
                net = self._conv_bn_relu(net,channel_out, 3, 1, 'SAME', 'conv3_2', use_loaded=use_loaded, lock=lock)
                return net

    def _cpm_stage(self, feat_map, stage_num, kernel_size=5,last_stage = None,lock=False,variable_scope='CPM_stage'):
        """ CPM stage Sturcture
        Args:
            feat_map    : Input Tensor from feature extractor
            last_stage  : Input Tensor from below
            stage_num   : stage number
            name        : name of the stage
        """
        with tf.variable_scope(variable_scope+str(stage_num)):
            #模仿mnist的字符识别模型
            if stage_num == 1:
                #use_loaded=self.load_pretrained
                use_loaded = False
                load_=False
                net=feat_map
                net = self._conv_bn_relu(net, 32,kernel_size, 1, 'SAME', 'Mconv1_stage'+str(stage_num), use_loaded=load_, lock=lock)
                net = self._conv_bn_relu(net, 64, kernel_size, 1, 'SAME', 'Mconv2_stage'+str(stage_num), use_loaded=load_, lock=lock)
                net = self._conv_bn_relu(net, 128, kernel_size, 1, 'SAME', 'Mconv3_stage'+str(stage_num), use_loaded=load_, lock=lock)
                if False:
                    net = self._conv_bn_relu(net, 256, 3, 1, 'SAME', 'conv4_4_CPM', use_loaded=use_loaded, lock=lock)
                    net = self._conv_bn_relu(net, 256, 3, 1, 'SAME', 'conv4_5_CPM', use_loaded=use_loaded, lock=lock)
                    net = self._conv_bn_relu(net, 256, 3, 1, 'SAME', 'conv4_6_CPM', use_loaded=use_loaded, lock=lock)
                    net = self._conv_bn_relu(net, 128, 3, 1, 'SAME', 'conv4_7_CPM', use_loaded=use_loaded, lock=lock)
                    net = self._conv_bn_relu(net, 512, 1, 1, 'SAME', 'conv5_1_CPM', use_loaded=use_loaded, lock=lock)
                net = self._conv(net, self.joint_num+1, 1, 1, 'SAME', 'conv5_2_CPM', use_loaded=False, lock=lock)#有输出热图的敌方都不能用原模型的参数
                return net
            elif stage_num > 1:
                net = tf.concat([feat_map, last_stage], 3)#我这里的last_stage shape和原模型的不一样（我有24个关键点，他的只有14个）
                print ';netshape=',net.shape,'feat_map shape=',feat_map.shape,'last_stage shpe=',last_stage.shape
                load_=False
                kernel_size=5
                net = self._conv_bn_relu(net, 32,kernel_size, 1, 'SAME', 'Mconv1_stage'+str(stage_num), use_loaded=load_, lock=lock)
                net = self._conv_bn_relu(net, 64, kernel_size, 1, 'SAME', 'Mconv2_stage'+str(stage_num), use_loaded=load_, lock=lock)
                net = self._conv_bn_relu(net, 128, kernel_size, 1, 'SAME', 'Mconv3_stage'+str(stage_num), use_loaded=load_, lock=lock)
                if False:
                    net = self._conv_bn_relu(net, 128, kernel_size, 1, 'SAME', 'Mconv4_stage'+str(stage_num), use_loaded=load_, lock=lock)
                    net = self._conv_bn_relu(net, 128, kernel_size, 1, 'SAME', 'Mconv5_stage'+str(stage_num), use_loaded=load_, lock=lock)
                    net = self._conv_bn_relu(net, 256, 1, 1, 'SAME', 'Mconv6_stage'+str(stage_num), use_loaded=load_, lock=lock)
                net = self._conv(net, self.joint_num+1, 1, 1, 'SAME', 'Mconv7_stage'+str(stage_num), use_loaded=False, lock=False)
                return net

#新增的hourglass 网络部分
#https://github.com/wbenbihi/hourglasstensorlfow/blob/master/hourglass_tiny.py
    def _hourglass(self, inputs, n, numOut, name='Hourglass+'):
        """ Hourglass Module
        Args:
            inputs	: Input Tensor
            n		: Number of downsampling step
            numOut	: Number of Output Features (channels)
            name	: Name of the block
        """
#        with tf.name_scope(name):
        #print n,name
        # Upper Branch
        print 'hourglass ---%i'%(n)
        #pdb.set_trace()
        up_1 = self._residual(inputs, numOut, name=name+'up_1')
        # Lower Branch
        low_ = tf.contrib.layers.max_pool2d(inputs, [2, 2], [2, 2], padding='VALID')# padding='VALID')
        low_1 = self._residual(low_, numOut, name=name+'low_1')
        if n > 1:
            low_2 = self._hourglass(low_1, n - 1, numOut,name=name+'low_2')
        else:
            low_2 = self._residual(low_1, numOut, name=name+'low_2')
        low_3 = self._residual(low_2, numOut, name=name+'low_3')
        up_2 = tf.image.resize_nearest_neighbor(low_3, tf.shape(low_3)[1:3] * 2, name=name+'upsampling')
        if Global.use_relu:
            # Use of RELU
            return tf.nn.relu(tf.add_n([up_2, up_1]), name=name+'out_hg')
        else:
            return tf.add_n([up_2, up_1], name=name+'out_hg')


