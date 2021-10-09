# coding:utf-8
import torch
import torch.optim as optim
import os
from data.dataset_scene import *
from torchvision import transforms
from VisionLAN import *

global_cfgs = {
    'state': 'Test',
    'epoch': 8,
    'show_interval': 200,
    'test_interval': 2000,
    'step': 'LA', # 'LF_1' 'LF_2' 'LA'
}

dataset_cfgs = {
    'dataset_train': lmdbDataset,
    'dataset_train_args': {
        'roots': [
            './datasets/train/SynthText',
            './datasets/train/MJSynth',
        ],
        'img_height': 64,
        'img_width': 256,
        'transform': transforms.Compose([transforms.ToTensor()]),
        'global_state': 'Train',
    },
    'dataloader_train': {
        'batch_size': 16,
        'shuffle': True,
        'num_workers': 32,
        'pin_memory': True,
    },

    'dataset_test': lmdbDataset,
    'dataset_test_all': {
        'roots': [
            './datasets/evaluation/Sumof6benchmarks'
        ],
        'img_height': 64,
        'img_width': 256,
        'transform': transforms.Compose([transforms.ToTensor()]),
        'global_state': 'Test',
    },

    'dataset_test_args': {
        'roots': [
            './datasets/evaluation/IIIT5K',

        ],
        'img_height': 64,
        'img_width': 256,
        'transform': transforms.Compose([transforms.ToTensor()]),
        'global_state': 'Test',
    },

    'dataset_test_argsIC13': {
        'roots': [
            './datasets/evaluation/IC13',
        ],
        'img_height': 64,
        'img_width': 256,
        'transform': transforms.Compose([transforms.ToTensor()]),
        'global_state': 'Test',
    },

    'dataset_test_argsIC15': {
        'roots': [
            './datasets/evaluation/IC15',
        ],
        'img_height': 64,
        'img_width': 256,
        'transform': transforms.Compose([transforms.ToTensor()]),
        'global_state': 'Test',
    },

    'dataset_test_argsSVT': {
        'roots': [
            './datasets/evaluation/SVT',
        ],
        'img_height': 64,
        'img_width': 256,
        'transform': transforms.Compose([transforms.ToTensor()]),
        'global_state': 'Test',
    },
    'dataset_test_argsSVTP': {
        'roots': [
                  './datasets/evaluation/SVTP'
                  ],
        'img_height': 64,
        'img_width': 256,
        'transform': transforms.Compose([transforms.ToTensor()]),
        'global_state': 'Test',
    },

    'dataset_test_argsCUTE': {
        'roots': [
                './datasets/evaluation/CUTE'
                  ],
        'img_height': 64,
        'img_width': 256,
        'transform': transforms.Compose([transforms.ToTensor()]),
        'global_state': 'Test',
    },

    'dataloader_test': {
        'batch_size': 32,
        'shuffle': False,
        'num_workers': 16,
        'pin_memory': True,
    },
    'case_sensitive': False,
    'dict_dir' : './dict/dic_36.txt'
}

net_cfgs = {
    'VisualLAN': VisionLAN,
    'args': {
        'strides': [(1,1), (2,2), (2,2), (2,2), (1,1), (1,1)],
        'input_shape': [3, 64, 256], # C x H x W
    },

    'init_state_dict': './output/LA/final.pth',
}


