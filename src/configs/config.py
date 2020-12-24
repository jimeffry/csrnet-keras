from easydict import EasyDict
import numpy as np

cfg = EasyDict()
#ground truth label scale
cfg.InputSize_w = 512 #768 #1024
cfg.InputSize_h = 512 #576 #768
# shanghai dataset dir
cfg.img_dir = '/wdc/LXY.data/img_celeba/img_detected/'  #'/mnt/data/LXY.data/'
cfg.train_file = '../datas/train.txt'
cfg.val_file = '../datas/test.txt'
cfg.model_dir = '/wdc/LXY.data/models/faceattr' #'/data/lxy/models/head'
# training set
cfg.EPOCHES = 600
cfg.LR_STEPS = [100000,200000,300000,400000]
cfg.MAX_STEPS = 500000
cfg.train_mod = 0 # data generate one by one
cfg.ClsNum = 21
