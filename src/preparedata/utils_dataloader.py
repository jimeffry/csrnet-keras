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
import h5py
sys.path.append(os.path.join(os.path.dirname(__file__),'../configs'))
from config import cfg
InputSize_h = 768
InputSize_w = 1024

def loadh5(fpath):
    gt_file = h5py.File(fpath,'r')
    target = np.asarray(gt_file['density'])
    return target

class DataLoader(object):
    def __init__(self, root,batch_size,
                 image_sets=[('part_A_final', 'train_data'),
                             ('part_B_final', 'train_data')],mode='train'):
        self.root = root
        self.image_set = image_sets
        self.batch_size = batch_size
        self._annopath = os.path.join('%s', 'gts', '%s.h5')
        self._imgpath = os.path.join('%s', 'images', '%s.jpg')
        self.ids = list()
        self.loadtxt()
        self.total_num = self.__len__()
        self.batch_num = np.ceil(self.total_num / batch_size)
        self.shulf_num = list(range(self.total_num))
        random.shuffle(self.shulf_num)
        self.rgb_mean = np.array([0.5, 0.5, 0.5])[np.newaxis, np.newaxis,:].astype('float32')
        self.rgb_std = np.array([0.225, 0.225, 0.225])[np.newaxis, np.newaxis,:].astype('float32')
        self.id_num = 0
        self.trainmode = mode

    def loadtxt(self):
        for (part, name) in self.image_set:
            rootpath = os.path.join(self.root, part,name)
            for line in open(os.path.join(rootpath, name + '.txt')):
                self.ids.append((rootpath, line.strip()))
    def __len__(self):
        return len(self.ids)

    def pull_item(self, index):
        idx = self.shulf_num[index]
        img_id = self.ids[idx]
        img_path = self._imgpath % img_id
        target = loadh5(self._annopath % img_id)
        # target  = np.load(self._annopath % img_id)
        target = target.astype(np.float32, copy=False)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        img, target = self.prepro(img,target)
        # return torch.from_numpy(img).permute(0,3,1,2), torch.from_numpy(target)
        return img, target

    def prepro(self,img,gt):
        # img ,gt = self.mirror(img,gt)
        img,gt = self.resize_subtract_mean(img,gt)
        return img,gt

    def mirror(self,image, gt):
        if random.randrange(2):
            image = image[:, ::-1,:]
            gt = gt[:,::-1]
        return image,gt

    def cropimg(self,image,gt):
        h,w = image.shape[:2]
        dh,dw = int(random.random()*(h-cfg.InputSize_h)),int(random.random()*(w-cfg.InputSize_w))
        img = image[dh:(dh+cfg.InputSize_h),dw:(dw+cfg.InputSize_w),:]
        gt = gt[dh:(dh+cfg.InputSize_h),dw:(dw+cfg.InputSize_w)]
        return img,gt

    def resize_subtract_mean(self,image,gt):
        # interp_methods = [cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_NEAREST, cv2.INTER_LANCZOS4]
        # interp_method = interp_methods[random.randrange(5)]
        h,w,_ = image.shape
        if h != InputSize_h or w != InputSize_w:
            image = cv2.resize(image,(int(InputSize_w),int(InputSize_h)))
        # if self.trainmode == 'train':
            # image,gt = self.cropimg(image,gt)
        image = image.astype(np.float32)
        image = image / 255.0
        image -= self.rgb_mean
        image = image / self.rgb_std
        # gt *=100
        image = np.transpose(image,(2,0,1))
        return image,gt

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