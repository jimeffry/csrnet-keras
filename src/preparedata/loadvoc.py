# -*- encoding: utf-8 -*-
import os.path as osp
import os
import sys
from PIL import Image
import numpy as np
import json
import random

from dataEnhance import *
sys.path.append(os.path.join(os.path.dirname(__file__),'../configs'))
from config import cfg


class DataLoader(object):
    def __init__(self,filein,batch_size,cropsize=(640, 480),mode='train', *args, **kwargs):
        # super(ContextVoc, self).__init__(*args, **kwargs)
        self.mode = mode
        self.batch_size = batch_size
        self._annopath = osp.join('%s', 'Seglabels24', '%s.png')
        self._imgpath = osp.join('%s', 'JPEGImages', '%s.jpg')
        self.rootpath = cfg.img_dir
        self.ids = list()
        self.loadtxt(filein)
        self.total_num = self.__len__()
        self.batch_num = np.ceil(self.total_num / batch_size)
        self.shulf_num = list(range(self.total_num))
        random.shuffle(self.shulf_num)
        self.enhancedata(cropsize)
        self.rgb_mean = np.array([0.485, 0.456, 0.406])[np.newaxis, np.newaxis,:].astype('float32')
        self.rgb_std = np.array([0.229, 0.224, 0.225])[np.newaxis, np.newaxis,:].astype('float32')
        self.id_num = 0

    def loadtxt(self,fpath):
        fr = open(fpath,'r')
        fcnts = fr.readlines()
        for tmp in fcnts:
            self.ids.append((self.rootpath,tmp.strip()))

    def __len__(self):
        return len(self.ids)

    def enhancedata(self,cropsize):
        ## pre-processing
        self.trans_train = Compose([
            ColorJitter(
                brightness = 0.5,
                contrast = 0.5,
                saturation = 0.5),
            HorizontalFlip(),
            # RandomScale((1.0, 1.25, 1.5)),
            RandomCrop(cropsize)
            ])

    def pull_item(self, index):
        idx = self.shulf_num[index]
        img_id = self.ids[idx]
        img_path = self._imgpath % img_id
        label_path = self._annopath % img_id
        img = Image.open(img_path)
        label = Image.open(label_path)
        label = label.convert('L')
        if self.mode == 'train':
            im_lb = dict(im = img, lb = label)
            im_lb = self.trans_train(im_lb)
            img, label = im_lb['im'], im_lb['lb']
        else:
            w,h = img.size
            w = int(np.ceil(w/16.0)*16)
            h = int(np.ceil(h/16.0)*16)
            img = img.resize((w, h), Image.BILINEAR)
            label = label.resize((w, h), Image.NEAREST)
        img = self.preprocessimg(img)
        label = np.array(label).astype(np.float32)
        # label = self.convert_dice_labels(label)
        return img, label

    def preprocessimg(self, image):
        image = np.array(image,dtype=np.float32)
        image = image / 255.0
        image -= self.rgb_mean
        image = image / self.rgb_std
        # image = np.transpose(image,(2,0,1))
        return image

    def convert_dice_labels(self,label):
        # return semantic_map -> [H, W, class_num]
        semantic_map = []
        for index in range(cfg.ClsNum):
            class_map = np.equal(label, index)
            # class_map = np.all(equality, axis=-1)
            semantic_map.append(class_map)
        semantic_map = np.stack(semantic_map, axis=-1).astype(np.float)
        semantic_map = np.transpose(semantic_map,(2,0,1))
        return semantic_map

    def next_batch(self):
        batch_images = []
        batch_labels = []
        for i in range (self.batch_size):
            if self.id_num < self.total_num:
                image,gt = self.pull_item(self.id_num)
                batch_images.append(image)
                batch_labels.append(gt)
                self.id_num +=1
            elif self.total_num % self.batch_size !=0:
                remainder = self.total_num % self.batch_size
                patch_num = self.batch_size - remainder
                for j in range(patch_num):
                    image,gt = self.pull_item(j)
                    # image = np.expand_dims(image,axis=-1)
                    batch_images.append(image)
                    batch_labels.append(gt)
                self.id_num = 0
                random.shuffle(self.shulf_num)
                break
            if self.id_num == self.total_num -1:
                self.id_num = 0
                random.shuffle(self.shulf_num)
        #print("indx",self.idx,"batch",len(batch_images))
        batch_images = np.array(batch_images).astype(np.float32)
        batch_labels = np.array(batch_labels).astype(np.float32)
        return batch_images, batch_labels



if __name__ == "__main__":
    from tqdm import tqdm
    ds = ContextVoc(filein='../datasets/vocval.txt', mode='train')
    uni = []
    for im, lb in tqdm(ds):
    # for ids,(im,lb) in enumerate(dataloader_train):
        # lb_uni = np.unique(lb).tolist()
        # uni.extend(lb_uni)
        print(lb.size())
    print(uni)
    print(set(uni))



