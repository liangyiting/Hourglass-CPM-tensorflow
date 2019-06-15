#-*-coding:utf-8 -*-
#	Global variable goes here
train_path='/root/facekeypoint/fashionAI/train/'
IMG_ROOT =train_path#+"Images/"  由于从csv文件提取的第一个元素中包含了Images/blouse，因此这里的路径只具体到train/
training_txt_file=train_path+'Annotations/annotations.csv'

cloth=['blouse']
load_old_sess=True#是否导入之前的训练
gaussian_sigma=1#高斯图的方差---原始的是3
feature_extractor='ResNet'#VGG
l2_loss=False
INPUT_SIZE =256/2#如果采用hornglass网络的话，输入必须是2*2^n的倍数，一般n取4，则必须是32的倍数
base_lr = 8e-5
epoch = 5000
batch_size =3#16
epoch_size=50
stage=1#
weighted_loss=False#True#False#是否使用带权重的loss   实验表明，不使用权重是最好的策略
LOGDIR = './log/'#存model.ckpt,以及存npy文件的地方
beishu=2#输入被VGG特征压缩的倍数，和CPM中的池化层个数对应，正常是8，也可设置成4.
stage_fmap=False#是否给第2层以上设置独立的特征图
lock=[False,False,False,False,False,False,False,False]#代表第0,1,2,3层是否锁死参数
channel_out=512/8#
n_hourglass=2
use_relu=True

if True:
    namestr = 'neckline_left,neckline_right,center_front,shoulder_left,shoulder_right,armpit_left,armpit_right,' \
              'waistline_left,waistline_right,cuff_left_in,cuff_left_out,cuff_right_in,cuff_right_out,top_hem_left,' \
              'top_hem_right,waistband_left,waistband_right,hemline_left,hemline_right,crotch,bottom_left_in,bottom_left_out,bottom_right_in,bottom_right_out'
    joint_list = namestr.split(',')
    t_blouse = range(7);
    t_blouse.extend(range(9, 15));
    index_keypoint = {'blouse': t_blouse, 'dress': [], 'skirt': []};
    remove_joints = [0] * len(joint_list)
    for i in range(len(t_blouse)):
        remove_joints[t_blouse[i]] = 1
    # joint_list=joint_list[index_keypoint['blouse']]
    metric = {'blouse': [5, 6], 'dress': [5, 6], 'outwear': [5, 6], 'skirt': [15, 16],
              'trousers': [15, 16]}  # （上衣、外套、连衣裙为两个腋窝点欧式距离，裤子和半身裙为两个裤头点的欧式距离）
else:
    #要导入下载的模型需要用原来的设置才行
    remove_joints=None
    joint_list = ['r_anckle', 'r_knee', 'r_hip', 'l_hip', 'l_knee', 'l_anckle', 'pelvis', 'thorax', 'neck', 'head', 'r_wrist', 'r_elbow', 'r_shoulder', 'l_shoulder', 'l_elbow', 'l_wrist']

