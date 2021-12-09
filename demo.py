# coding:utf-8
from __future__ import print_function
import torch
from torchvision import transforms
from utils import *
import cfgs.cfgs_eval as cfgs
from collections import OrderedDict
import os
from PIL import Image

def Train_or_Eval(model, state='Train'):
    if state == 'Train':
        model.train()
    else:
        model.eval()

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




if __name__ == '__main__':
    model = load_network()
    image_path = './demo/'
    output_txt = './demo/predictions.txt'
    image_list = os.listdir(image_path)
    img_width = 256
    img_height = 64
    transf = transforms.ToTensor()
    test_acc_counter = Attention_AR_counter('\ntest accuracy: ', cfgs.dataset_cfgs['dict_dir'],
                                            cfgs.dataset_cfgs['case_sensitive'])

    Train_or_Eval(model, 'Eval')
    for img_name in image_list:
        img = Image.open(image_path + img_name).convert('RGB')
        img = img.resize((img_width, img_height))
        img = transf(img)
        img = torch.unsqueeze(img,dim = 0)
        target = ''
        output, out_length = model(img, target, '', False)
        pre_string = test_acc_counter.convert(output, out_length)
        print('pre_string:',pre_string[0])
        with open(output_txt,'a') as f:
            f.write(img_name+':'+pre_string[0]+'\n')

