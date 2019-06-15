#-*-coding:utf-8 -*-
from CPM_hornglass  import CPM
import Global
import datagen_mpii as datagen
import pdb
import tensorflow as tf

#   Thanks to wbenhibi@github
#   good datagen to use
# 使用了中心图--也就是取了图框box
print('--Creating Dataset')
dataset = datagen.DataGenerator(Global.joint_list, Global.IMG_ROOT, Global.training_txt_file, remove_joints=None, in_size=Global.INPUT_SIZE)
dataset._create_train_table()
dataset._randomize()
dataset._create_sets()

model = CPM(base_lr=Global.base_lr, in_size=Global.INPUT_SIZE,
            batch_size=Global.batch_size, epoch=Global.epoch,
            stage=Global.stage,epoch_size=Global.epoch_size,
            dataset = dataset, log_dir=Global.LOGDIR,
            load_pretrained=False,pretrained_model=None) #可在CPM中选择需要训练的层
#pretrained_model是存放字典形式存储模型参数的地方，在posenet 文件下 weight:[name][0],bias:[name][1] name是层的名字，比如Conv_1,Conv_2之类的，是在定义模型的时候人工给的名字，和scope是无关的
#log_dir是存放训练后npy模型的地方， self.log_dir + 'model.npy'
#log_dir也存放训练后的ckpt模型， os.path.join(self.log_dir, "model/model.ckpt")
model.BuildModel()
if Global.load_old_sess:
    try:
        print 'Loading old sess'
        model_file = tf.train.latest_checkpoint('./log/model/')#如果是自己训练好的模型，应该有checkpoint的；如果是网上下载的，就用名字
        model.saver.restore(model.sess,model_file)
    except:
        print 'Load old sess Failed!'

model.train()