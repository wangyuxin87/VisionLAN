# coding:utf-8
import random
import torch
from torch.utils.data import Dataset
from torch.utils.data import sampler
import torchvision.transforms as transforms
import lmdb
import six
import sys
from PIL import Image
import numpy as np
import pdb
import os
import cv2
from transforms import CVColorJitter, CVDeterioration, CVGeometry
import re
from random import sample

def des_orderlabel(imput_lable):
    '''
    generate the label for WCL
    '''
    if True:
        len_str = len(imput_lable)
        change_num = 1
        order = list(range(len_str))
        change_id = sample(order, change_num)[0]
        label_sub =imput_lable[change_id]
        if change_id == (len_str - 1):
            imput_lable = imput_lable[:change_id]
        else:
            imput_lable = imput_lable[:change_id] + imput_lable[change_id + 1:]
    return imput_lable, label_sub, change_id
class lmdbDataset(Dataset):
    def __init__(self, roots=None, ratio=None, img_height = 32, img_width = 128,
        transform=None, global_state='Test'):
        self.envs = []
        self.nSamples = 0
        self.lengths = []
        self.ratio = []
        self.global_state = global_state
        for i in range(0,len(roots)):
            env = lmdb.open(
                roots[i],
                max_readers=1,
                readonly=True,
                lock=False,
                readahead=False,
                meminit=False)
            if not env:
                print('cannot creat lmdb from %s' % (root))
                sys.exit(0)
            with env.begin(write=False) as txn:
                nSamples = int(txn.get('num-samples'.encode()))
                self.nSamples += nSamples
            self.lengths.append(nSamples)
            self.envs.append(env)

        if ratio != None:
            assert len(roots) == len(ratio) ,'length of ratio must equal to length of roots!'
            for i in range(0,len(roots)):
                self.ratio.append(ratio[i] / float(sum(ratio)))
        else:
            for i in range(0,len(roots)):
                self.ratio.append(self.lengths[i] / float(self.nSamples))
        self.transform = transform
        self.maxlen = max(self.lengths)
        self.img_height = img_height
        self.img_width = img_width
        self.target_ratio = img_width / float(img_width)
        self.min_size = (img_width * 0.5, img_width * 0.75, img_width)

        self.augment_tfs = transforms.Compose([
            CVGeometry(degrees=45, translate=(0.0, 0.0), scale=(0.5, 2.), shear=(45, 15), distortion=0.5, p=0.5),
            CVDeterioration(var=20, degrees=6, factor=4, p=0.25),
            CVColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1, p=0.25)
        ])
    def __fromwhich__(self ):
        rd = random.random()
        total = 0
        for i in range(0,len(self.ratio)):
            total += self.ratio[i]
            if rd <= total:
                return i
    def keepratio_resize(self, img, is_train):
        if is_train == 'Train':
            img = self.augment_tfs(img)
        img = cv2.resize(np.array(img), (self.img_width, self.img_height))
        img = transforms.ToPILImage()(img)
        return img

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        fromwhich = self.__fromwhich__()
        if self.global_state == 'Train':
            index = random.randint(0,self.maxlen - 1)
        index = index % self.lengths[fromwhich]
        assert index <= len(self), 'index range error'
        index += 1
        with self.envs[fromwhich].begin(write=False) as txn:
            img_key = 'image-%09d' % index
            try:
                imgbuf = txn.get(img_key.encode())
                buf = six.BytesIO()
                buf.write(imgbuf)
                buf.seek(0)
                img = Image.open(buf).convert('RGB')
            except:
                print('Corrupted image for %d' % index)
                return self[index + 1]
            label_key = 'label-%09d' % index
            label = str(txn.get(label_key.encode()))
            # if python3
            # label = str(txn.get(label_key.encode()), 'utf-8')
            label = re.sub('[^0-9a-zA-Z]+', '', label)
            
            if (len(label) > 25 or len(label) <= 0) and self.global_state == 'Train':
                print('sample too long')
                return self[index + 1]
            try:
                img = self.keepratio_resize(img, self.global_state)
            except:
                print('Size error for %d' % index)
                return self[index + 1]
            if self.transform:
                img = self.transform(img)
            # generate masked_id masked_character remain_string
            label_res, label_sub, label_id =  des_orderlabel(label)
            sample = {'image': img, 'label': label, 'label_res': label_res, 'label_sub': label_sub, 'label_id': label_id}
            return sample