from easydict import EasyDict
import numpy as np

cfg = EasyDict()
#ground truth label scale
cfg.InputSize_w = 1024 #768 #1024
cfg.InputSize_h = 768 #576 #768
# shanghai dataset dir
cfg.shanghai_dir = '/data/lxy/shang_crowed/' #'/mnt/data/LXY.data/'
cfg.train_file = ''
cfg.test_file = ''
cfg.model_dir = '/data/lxy/models/head'
# training set
cfg.EPOCHES = 600
cfg.LR_STEPS = [100000,200000,300000,400000]
cfg.MAX_STEPS = 500000
cfg.train_mod = 0 # data generate one by one
