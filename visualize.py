# coding:utf-8
from __future__ import print_function
import torch
from torch.utils.data import Dataset, DataLoader
from utils import *
import cfgs.cfgs_visualize as cfgs
from collections import OrderedDict
import cv2
import os
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

def Zero_Grad(model):
    model.zero_grad()

def load_dataset():
    train_data_set = cfgs.dataset_cfgs['dataset_train'](**cfgs.dataset_cfgs['dataset_train_args'])
    train_loader = DataLoader(train_data_set, **cfgs.dataset_cfgs['dataloader_train'])
    test_data_set = cfgs.dataset_cfgs['dataset_test'](**cfgs.dataset_cfgs['dataset_test_args'])
    test_loader = DataLoader(test_data_set, **cfgs.dataset_cfgs['dataloader_test'])
    return train_loader, test_loader

def load_network():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_VL = cfgs.net_cfgs['VisualLAN'](**cfgs.net_cfgs['args'])
    model_VL = model_VL.to(device)
    model_VL = torch.nn.DataParallel(model_VL)
    if cfgs.net_cfgs['init_state_dict'] != None:
        fe_state_dict_ori = torch.load(cfgs.net_cfgs['init_state_dict'])
        fe_state_dict = OrderedDict()
        for k, v in fe_state_dict_ori.items():
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

def _flatten(sources, lengths):
    return torch.cat([t[:l] for t, l in zip(sources, lengths)])

if __name__ == '__main__':
    model = load_network()
    train_loader, test_loader = load_dataset()
    # train
    checkpoint_path_pre = './output/' + str(cfgs.dataset_cfgs['dataset_train_args']['mask_id'])
    if not os.path.isdir(checkpoint_path_pre):
        os.mkdir(checkpoint_path_pre)
    # how many mask map for visualization
    max_viusalize_num = 30
    visualize_num = 0
    end_visualize = False
    for nEpoch in range(0, cfgs.global_cfgs['epoch']):
        for batch_idx, sample_batched in enumerate(train_loader):
            # data_prepare
            data = sample_batched['image']
            label = sample_batched['label']  # original string
            label_id = sample_batched['label_id']  # character index
            label_sub = sample_batched['label_sub']  # occluded character
            label_id_show = label_id.cpu().numpy()
            Train_or_Eval(model, 'Train')
            data = data.cuda()
            label_id =  label_id.cuda()
            # prediction
            text_pre, test_rem, text_mas, att_mask_sub = model(data, label_id, cfgs.global_cfgs['step'])
            Zero_Grad(model)
            # Visualize
            if True:
                for ind in range(data.shape[0]):
                    img = data[ind]
                    img = ((img.cpu().numpy().transpose(1, 2, 0) + 1.0).clip(0, 2) * 127.5).astype(np.uint8)
                    cv2.imwrite(checkpoint_path_pre + '/' + str(nEpoch) + '-' + str(batch_idx) + '-' + str(ind) + '.png', img)
                    att_mask_sub_ind = (att_mask_sub[ind].squeeze(dim=0).detach().cpu().numpy() * 255).astype(np.uint8)
                    att_mask_sub_ind = cv2.resize(att_mask_sub_ind, (256, 64))
                    cv2.imwrite(checkpoint_path_pre + '/' + str(nEpoch) + '-' + str(batch_idx) + '-' + str(ind) + '-' + str(
                        label_id_show[ind]) + str(label_sub[ind]) + '.png', att_mask_sub_ind)
                    visualize_num += 1
                    if visualize_num > max_viusalize_num:
                        end_visualize = True
                        print('Finish Visualization!')
                        break
            if end_visualize:
                break
        break



