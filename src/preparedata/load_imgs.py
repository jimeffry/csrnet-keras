'''
# auther : lxy
# time : 2017.1.9 /09:56
#project:
# tool: python2
#version: 0.1
#modify:
#name: center loss
#citations: DEX: Deep EXpectation of apparent age from a single image," ICCV, 2015
            Deep expectation of real and apparent age from a single image without facial landmarks," IJCV, 2016
'''
#############################
import cv2
import numpy as np
import random
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__),'../configs'))
from config import cfg


class DataLoader(object):
    def __init__(self, root,batch_size,filein,mode='train'):
        self.root = root
        self.batch_size = batch_size
        self._labels = []
        self._imgpaths = []
        self.loadtxt(filein)
        self.total_num = self.__len__()
        self.batch_num = np.ceil(self.total_num / batch_size)
        self.shulf_num = list(range(self.total_num))
        random.shuffle(self.shulf_num)
        self.rgb_mean = np.array([0.485, 0.456, 0.406])[np.newaxis, np.newaxis,:].astype('float32')
        self.rgb_std = np.array([0.229, 0.224, 0.225])[np.newaxis, np.newaxis,:].astype('float32')
        self.id_num = 0
        self.trainmode = mode

    def loadtxt(self,annopath):
        fr = open(annopath,'r')
        fr_cnts = fr.readlines()
        for tmp in fr_cnts:
            tmp_s = tmp.strip().split(',')
            imgpath = os.path.join(self.root,tmp_s[0])
            tmp_label = [int(la) for la in tmp_s[1:]]
            self._labels.append(tmp_label)
            self._imgpaths.append(imgpath)
        fr.close()
        
    def __len__(self):
        return len(self._imgpaths)

    def pull_item(self, index):
        idx = self.shulf_num[index]
        img_path = self._imgpaths[idx]
        target = self._labels[idx]
        # target = target.astype(np.float32, copy=False)
        img = cv2.imread(img_path)
        if len(img.shape) <3:
            img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
        else:
            img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        img, target = self.prepro(img,target)
        return img, target

    def prepro(self,img,gt):
        if self.trainmode=='train':
            img = self.mirror(img)
        img = self.resize_subtract_mean(img)
        return img,gt

    def mirror(self,image):
        if random.randrange(2):
            image = image[:, ::-1,:]
        return image

    def resize_subtract_mean(self,image):
        # interp_methods = [cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_NEAREST, cv2.INTER_LANCZOS4]
        # interp_method = interp_methods[random.randrange(5)]
        h,w,_ = image.shape
        if h != cfg.InputSize_h or w != cfg.InputSize_w:
            image = cv2.resize(image,(int(cfg.InputSize_w),int(cfg.InputSize_h)))
        # if self.trainmode == 'train':
            # image,gt = self.cropimg(image,gt)
        image = image.astype(np.float32)
        image = image / 255.0
        image -= self.rgb_mean
        image = image / self.rgb_std
        # image = np.transpose(image,(2,0,1))
        return image

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


if __name__ =='__main__':
    data = BatchLoader("./data/imdb_val.mat", 16,64,"imdb")
    a,b,c = data.next_batch()
    print(a.shape)