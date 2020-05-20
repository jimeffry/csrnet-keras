#import pandas as pd
import logging
import argparse
import os
import sys
import cv2
import time
import collections
from matplotlib import pyplot as plt
os.environ['KERAS_BACKEND'] = 'tensorflow'
from keras import backend as K
from keras.callbacks import LearningRateScheduler, ModelCheckpoint,ReduceLROnPlateau,EarlyStopping
from keras.optimizers import SGD, Adam
from keras.utils import multi_gpu_model
import numpy as np
sys.path.append(os.path.join(os.path.dirname(__file__),'../networks'))
from vgg import CSRNet
sys.path.append(os.path.join(os.path.dirname(__file__),'../configs'))
from config import cfg
sys.path.append(os.path.join(os.path.dirname(__file__),'../losses'))
from utils_loss import multiloss,MSE_BCE
sys.path.append(os.path.join(os.path.dirname(__file__),'../preparedata'))
from utils_dataloader import DataLoader


def get_args():
    parser = argparse.ArgumentParser(description="This script trains the CNN model for person estimation.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--batch_size", type=int, default=128,
                        help="batch size")
    parser.add_argument("--nb_epochs", type=int, default=300,
                        help="number of epochs")
    parser.add_argument("--pretrained_fil",type=str,default=None,
                        help='before pretrained_file for net')
    parser.add_argument("--lr", type=float, default=0.00001,
                        help='the net training learningrate')
    parser.add_argument("--gpus", type=int, default=1,
                        help='how many gpus for trainning')
    parser.add_argument("--db_name",type=str,default="ShangHai",\
                        help="training on which dataset")
    parser.add_argument("--log_dir",type=str,default="../logs",\
                        help="training on which dataset")
    args = parser.parse_args()
    return args


def data_geneter(batchloader):
    while True:
        imgs, labels = batchloader.next_batch()
        #yield imgs,[y_data_g,y_data_a]
        yield imgs,labels

def lr_schedule(epoch_idx):
    base_lr=0.01,
    decay=0.9
    epoch_idx = int(epoch_idx)
    return base_lr * decay**(epoch_idx)

def createlogger(lpath):
    if not os.path.exists(lpath):
        os.makedirs(lpath)
    logger = logging.getLogger()
    logname= time.strftime('%F-%T',time.localtime()).replace(':','-')+'.log'
    logpath = os.path.join(lpath,logname)
    hdlr = logging.FileHandler(logpath)
    logger.addHandler(hdlr)
    logger.addHandler(logging.StreamHandler())
    logger.setLevel(logging.DEBUG)
    return logger

def main():
    args = get_args()
    batch_size = args.batch_size*args.gpus
    nb_epochs = args.nb_epochs
    pretrained_file = args.pretrained_fil
    patience = 30
    gpu_num = args.gpus
    train_db = args.db_name
    lr = args.lr
    logger = createlogger(args.log_dir)
    logger.debug("Loading data...")
    train_dataset = DataLoader(cfg.shanghai_dir,batch_size,image_sets=[('part_C_final', 'train_data')])
    val_dataseta = DataLoader(cfg.shanghai_dir,1,image_sets=[('part_C_final', 'test_data')],mode='test')
    # val_datasetb = DataLoader(cfg.shanghai_dir,2,image_sets=[('part_B_final', 'test_data')])
    # val_datasetc = DataLoader(cfg.shanghai_dir,2,image_sets=[('part_C_final', 'test_data')])
    #build model
    model = CSRNet()
    if not os.path.exists(cfg.model_dir):
        mk_dir(cfg.model_dir)
    with open(os.path.join(cfg.model_dir, "csr_{}.json".format(train_db)), "w") as f:
        f.write(model.to_json())
    if pretrained_file :
        model.load_weights(pretrained_file)
        print("load model file success",pretrained_file)
    #opt = SGD(lr=lr, momentum=0.7, nesterov=True)
    opt = Adam(lr=lr,beta_1=0.9,beta_2=0.999,epsilon=1e-5)
    #model.compile(optimizer=opt, loss=["categorical_crossentropy", "categorical_crossentropy"],
    #              metrics=['accuracy'])
    if gpu_num >1:
        model = multi_gpu_model(model,gpu_num)
    model.compile(optimizer=opt, loss=multiloss)
    logger.debug("Model summary...")
    #model.count_params()
    # model.summary()
    # logging.debug("Saving model...")
    reduce_lr = ReduceLROnPlateau(monitor="val_loss",factor=0.1,patience=patience*2,verbose=1,min_lr=0.0000001)
    early_stop = EarlyStopping('val_loss', patience=patience)
    modelcheckpoint = ModelCheckpoint("ag_models/weights.{epoch:02d}-{val_loss:.2f}.hdf5",\
                        monitor="val_loss",verbose=1,save_best_only=True,mode="auto",period=1000)
    #reduce_lr = LearningRateScheduler(schedule=reduce_lr)
    #callbacks = [modelcheckpoint,early_stop,reduce_lr]
    callbacks = [modelcheckpoint,reduce_lr]
    logging.debug("Running training...")
    #whole training
    if cfg.train_mod :
        hist = model.fit_generator(data_geneter(train_dataset), steps_per_epoch=train_dataset.batch_num,
                              epochs=nb_epochs, verbose=1,
                              callbacks=callbacks,
                              validation_data=data_geneter(val_dataseta),
                              nb_val_samples=val_dataseta.batch_num,
                              nb_worker=1)
        logger.debug("Saving weights...")
        model.save_weights(os.path.join(cfg.model_dir, "csr_{}_best.h5".format(train_db)),overwrite=True)
    else:
        epoch_step = 0
        loss_hist = collections.deque(maxlen=200)
        total_id = 0
        tmp_diffa = 10000
        tmp_diffb = 10000
        mean_diff = 10000
        tmp_diffc = 10000
        rgb_mean = np.array([0.5, 0.5, 0.5])[np.newaxis, np.newaxis,:].astype('float32')
        rgb_std = np.array([0.225, 0.225, 0.225])[np.newaxis, np.newaxis,:].astype('float32')
        while epoch_step < nb_epochs:
            step = 0
            while step < train_dataset.batch_num:
                X_data, y_data = train_dataset.next_batch()
                # for i in range(batch_size):
                    # tmp_img = np.transpose(X_data[i],(1,2,0))
                    # tmp_img = tmp_img *rgb_std
                    # tmp_img = tmp_img + rgb_mean
                    # tmp_img = tmp_img * 255
                    # tmp_img = np.array(tmp_img,dtype=np.uint8)
                    # tmp_img = cv2.cvtColor(tmp_img,cv2.COLOR_RGB2BGR)
                    # gt = y_data[i]
                    # print('gt num:',np.sum(gt))
                    # f, ax = plt.subplots(figsize=(8,5))
                    # ax.imshow(gt)
                    # cv2.imshow('src',tmp_img)
                    # plt.show()
                    # cv2.waitKey(0)
                # print(step)
                # step +=1
                save_fg = 0
                loss = model.fit(X_data, y_data, batch_size=batch_size, epochs=1, verbose=0)
                step+=1
                total_id +=1
                cur_loss = float(loss.history['loss'][0])
                loss_hist.append(cur_loss)
                if total_id %100==0:
                    logger.info('epoch:{} || iter:{} || tloss:{:.6f},curloss:{:.6f} || lr:{:.6f}'.format(epoch_step,total_id,np.mean(loss_hist),cur_loss,lr))
                if total_id % 500 ==0:
                    tmp_vala = val(model,val_dataseta,logger,2)
                    # tmp_valb = val(model,val_datasetb,logger,2)
                    # tmp_valc = val(model,val_datasetc,logger,2)
                    # tmp_mean = (tmp_vala+tmp_valb)/2.0
                    if tmp_vala < tmp_diffa:
                        save_fg = 1
                        tmp_diffa = tmp_vala
                    # if tmp_valb < tmp_diffb:
                    #     save_fg = 1
                    #     tmp_diffb = tmp_valb
                    # if tmp_valc < tmp_diffc:
                    #     save_fg = 1
                    #     tmp_diffc = tmp_valc
                    if save_fg :
                        logger.info("Saving weights...{}".format(total_id))
                        model.save_weights(os.path.join(cfg.model_dir, "csr_{}_best2.h5".format(train_db)))
                        # mean_diff = tmp_mean
            epoch_step +=1

def val(net,val_loader,logger,batch_size):
    diff_sum = 0.0
    for batch_idx in range(int(val_loader.batch_num)):
        images ,targets = val_loader.next_batch()
        out = net.predict(images)
        # diff = cal_diffnum(out,targets)
        # outshape = K.shape(out)
        # out = K.reshape(out,(outshape[0],outshape[2],outshape[3]))
        diff = abs(np.sum(out)-np.sum(targets))
        diff_sum += diff
    total_num = batch_size *(batch_idx+1)
    logger.info('test MAE:%.4f' % (diff_sum/total_num))
    return diff_sum/total_num


def set_keras_backend(backend):
    if K.backend() != backend:
        os.environ['KERAS_BACKEND'] = backend
        reload(K)
        assert K.backend() == backend

if __name__ == '__main__':
    set_keras_backend('tensorflow')
    main()