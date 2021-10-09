# coding:utf-8
from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import datetime
from utils import *
import cfgs.cfgs_eval as cfgs
from collections import OrderedDict
import time
import sys

def flatten_label(target):
    label_flatten = []
    label_length = []
    for i in range(0, target.size()[0]):
        cur_label = target[i].tolist()
        label_flatten += cur_label[:cur_label.index(0) + 1]
        label_length.append(cur_label.index(0) + 1)
    label_flatten = torch.LongTensor(label_flatten)
    label_length = torch.IntTensor(label_length)
    return (label_flatten, label_length)

def Train_or_Eval(model, state='Train'):
    if state == 'Train':
        model.train()
    else:
        model.eval()


def load_dataset():
    train_data_set = cfgs.dataset_cfgs['dataset_train'](**cfgs.dataset_cfgs['dataset_train_args'])
    train_loader = DataLoader(train_data_set, **cfgs.dataset_cfgs['dataloader_train'])

    test_data_all = cfgs.dataset_cfgs['dataset_test'](**cfgs.dataset_cfgs['dataset_test_all'])
    test_loader_all = DataLoader(test_data_all, **cfgs.dataset_cfgs['dataloader_test'])

    test_data_set = cfgs.dataset_cfgs['dataset_test'](**cfgs.dataset_cfgs['dataset_test_args'])
    test_loader = DataLoader(test_data_set, **cfgs.dataset_cfgs['dataloader_test'])

    test_data_setIC13 = cfgs.dataset_cfgs['dataset_test'](**cfgs.dataset_cfgs['dataset_test_argsIC13'])
    test_loaderIC13 = DataLoader(test_data_setIC13, **cfgs.dataset_cfgs['dataloader_test'])

    test_data_setIC15 = cfgs.dataset_cfgs['dataset_test'](**cfgs.dataset_cfgs['dataset_test_argsIC15'])
    test_loaderIC15 = DataLoader(test_data_setIC15, **cfgs.dataset_cfgs['dataloader_test'])

    test_data_setSVT = cfgs.dataset_cfgs['dataset_test'](**cfgs.dataset_cfgs['dataset_test_argsSVT'])
    test_loaderSVT = DataLoader(test_data_setSVT, **cfgs.dataset_cfgs['dataloader_test'])

    test_data_setSVTP = cfgs.dataset_cfgs['dataset_test'](**cfgs.dataset_cfgs['dataset_test_argsSVTP'])
    test_loaderSVTP = DataLoader(test_data_setSVTP, **cfgs.dataset_cfgs['dataloader_test'])

    test_data_setCUTE = cfgs.dataset_cfgs['dataset_test'](**cfgs.dataset_cfgs['dataset_test_argsCUTE'])
    test_loaderCUTE = DataLoader(test_data_setCUTE, **cfgs.dataset_cfgs['dataloader_test'])

    # pdb.set_trace()
    return (train_loader, test_loader_all, test_loader, test_loaderIC13, test_loaderIC15, test_loaderSVT, test_loaderSVTP, test_loaderCUTE)


def load_network():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_VL = cfgs.net_cfgs['VisualLAN'](**cfgs.net_cfgs['args'])
    model_VL = model_VL.to(device)
    model_VL = torch.nn.DataParallel(model_VL)
    if cfgs.net_cfgs['init_state_dict'] != None:
        fe_state_dict_ori = torch.load(cfgs.net_cfgs['init_state_dict'])
        fe_state_dict = OrderedDict()
        for k, v in fe_state_dict_ori.items():
            # if 'MLM' in k:
            #     print()
            if 'module' not in k:
                k = 'module.' + k
            else:
                k = k.replace('features.module.', 'module.features.')
            fe_state_dict[k] = v
        model_dict_fe = model_VL.state_dict()
        state_dict_fe = {k: v for k, v in fe_state_dict.items() if k in model_dict_fe.keys()}
        model_dict_fe.update(state_dict_fe)
        model_VL.load_state_dict(model_dict_fe)
    return model_VL

def test(test_loader, model, tools, best_acc, string_name):
    Train_or_Eval(model, 'Eval')
    print('------' + string_name + '--------')
    for sample_batched in test_loader:
        data = sample_batched['image']
        label = sample_batched['label']
        target = tools[0].encode(label)
        data = data.cuda()
        target = target
        label_flatten, length = tools[1](target)
        target, label_flatten = target.cuda(), label_flatten.cuda()
        output, out_length = model(data, target, '',  False)
        tools[2].add_iter(output, out_length, length, label)
    best_acc, change = tools[2].show_test(best_acc)
    Train_or_Eval(model, 'Train')
    return best_acc, change


if __name__ == '__main__':
    model = load_network()
    train_loader, test_loader_all, test_loader, test_loaderIC13, test_loaderIC15, test_loaderSVT, test_loaderSVTP, test_loaderCUTE = load_dataset()
    test_acc_counter = Attention_AR_counter('\ntest accuracy: ', cfgs.dataset_cfgs['dict_dir'],
                                            cfgs.dataset_cfgs['case_sensitive'])
    encdec = cha_encdec(cfgs.dataset_cfgs['dict_dir'], cfgs.dataset_cfgs['case_sensitive'])

    if cfgs.global_cfgs['state'] == 'Test':
        test((test_loader_all),
             model,
             [encdec,
              flatten_label,
              test_acc_counter], best_acc=0, string_name='Average on 6 benchmarks')

        test((test_loader),
             model,
             [encdec,
              flatten_label,
              test_acc_counter], best_acc=0, string_name='IIIT')
        test((test_loaderIC13),
             model,
             [encdec,
              flatten_label,
              test_acc_counter], best_acc=0, string_name='IC13')
        test((test_loaderIC15),
             model,
             [encdec,
              flatten_label,
              test_acc_counter], best_acc=0, string_name='IC15')
        test((test_loaderSVT),
             model,
             [encdec,
              flatten_label,
              test_acc_counter], best_acc=0, string_name='SVT')
        test((test_loaderSVTP),
             model,
             [encdec,
              flatten_label,
              test_acc_counter], best_acc=0, string_name='SVTP')
        test((test_loaderCUTE),
             model,
             [encdec,
              flatten_label,
              test_acc_counter], best_acc=0, string_name='CUTE')
        exit()