import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from modules.modules import Transforme_Encoder, Prediction, Transforme_Encoder_light
import torchvision
import modules.resnet as resnet


class MLM(nn.Module):
    '''
    Architecture of MLM
    '''
    def __init__(self, n_dim=512):
        super(MLM, self).__init__()
        self.MLM_SequenceModeling_mask = Transforme_Encoder(n_layers=2, n_position=256)
        self.MLM_SequenceModeling_WCL = Transforme_Encoder(n_layers=1, n_position=256)
        self.pos_embedding = nn.Embedding(25, 512)
        self.w0_linear = nn.Linear(1, 256)
        self.wv = nn.Linear(n_dim, n_dim)
        self.active = nn.Tanh()
        self.we = nn.Linear(n_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input, label_pos, state=False):
        # transformer unit for generating mask_c
        feature_v_seq = self.MLM_SequenceModeling_mask(input, src_mask=None)[0]
        # position embedding layer
        pos_emb = self.pos_embedding(label_pos.long())
        pos_emb = self.w0_linear(torch.unsqueeze(pos_emb, dim=2)).permute(0, 2, 1)
        # fusion position embedding with features V & generate mask_c
        att_map_sub = self.active(pos_emb + self.wv(feature_v_seq))
        att_map_sub = self.we(att_map_sub)  # b,256,1
        att_map_sub = self.sigmoid(att_map_sub.permute(0, 2, 1))  # b,1,256
        # WCL
        ## generate inputs for WCL
        f_res = input * (1 - att_map_sub.permute(0, 2, 1)) # second path with remaining string
        f_sub = input * (att_map_sub.permute(0, 2, 1)) # first path with occluded character
        ## transformer units in WCL
        f_res = self.MLM_SequenceModeling_WCL(f_res, src_mask=None)[0]
        f_sub = self.MLM_SequenceModeling_WCL(f_sub, src_mask=None)[0]
        return f_res, f_sub, att_map_sub

def trans_1d_2d(x):
    b, w_h, c = x.shape  # b, 256, 512
    x = x.permute(0, 2, 1)
    x = x.view(b, c, 32, 8)
    x = x.permute(0, 1, 3, 2)  # [16, 512, 8, 32]
    return x

class MLM_VRM(nn.Module):
    '''
    MLM+VRM, MLM is only used in training.
    ratio controls the occluded number in a batch.
    The pipeline of VisionLAN in testing is very concise with only a backbone + sequence modeling(transformer unit) + prediction layer(pp layer).
    input: input image
    label_pos: character index
    training_stp: LF or LA process
    output
    text_pre: prediction of VRM
    test_rem: prediction of remaining string in MLM
    text_mas: prediction of occluded character in MLM
    mask_c_show: visualization of Mask_c
    '''
    def __init__(self,):
        super(MLM_VRM, self).__init__()
        self.MLM = MLM()
        self.SequenceModeling = Transforme_Encoder(n_layers=3, n_position=256)
        self.Prediction = Prediction(n_position=256, N_max_character=26, n_class=37) # N_max_character = 1 eos + 25 characters
        self.nclass = 37
    def forward(self, input, label_pos, training_stp, is_Train = False):
        b, c, h, w = input.shape
        nT = 25
        input = input.permute(0, 1, 3, 2)
        input = input.contiguous().view(b, c, -1)
        input = input.permute(0, 2, 1)
        if is_Train:
            if training_stp == 'LF_1':
                f_res = 0
                f_sub = 0
                input = self.SequenceModeling(input, src_mask=None)[0]
                text_pre, test_rem, text_mas = self.Prediction(input, f_res, f_sub, Train_is=True, use_mlm=False)
                return text_pre, text_pre, text_pre, text_pre
            elif training_stp == 'LF_2':
                # MLM
                f_res, f_sub, mask_c = self.MLM(input, label_pos, state=True)
                input = self.SequenceModeling(input, src_mask=None)[0]
                text_pre, test_rem, text_mas = self.Prediction(input, f_res, f_sub, Train_is=True)
                mask_c_show = trans_1d_2d(mask_c.permute(0, 2, 1))
                return text_pre, test_rem, text_mas, mask_c_show
            elif training_stp == 'LA':
                # MLM
                f_res, f_sub, mask_c = self.MLM(input, label_pos, state=True)
                ## use the mask_c (1 for occluded character and 0 for remaining characters) to occlude input
                ## ratio controls the occluded number in a batch
                ratio = 2
                character_mask = torch.zeros_like(mask_c)
                character_mask[0:b // ratio, :, :] = mask_c[0:b // ratio, :, :]
                input = input * (1 - character_mask.permute(0, 2, 1))
                # VRM
                ## transformer unit for VRM
                input = self.SequenceModeling(input, src_mask=None)[0]
                ## prediction layer for MLM and VSR
                text_pre, test_rem, text_mas = self.Prediction(input, f_res, f_sub, Train_is=True)
                mask_c_show = trans_1d_2d(mask_c.permute(0, 2, 1))
                return text_pre, test_rem, text_mas, mask_c_show
        else: # VRM is only used in the testing stage
            f_res = 0
            f_sub = 0
            contextual_feature = self.SequenceModeling(input, src_mask=None)[0]
            C = self.Prediction(contextual_feature, f_res, f_sub, Train_is=False, use_mlm=False)
            C = C.permute(1, 0, 2)  # (25, b, 38))
            lenText = nT
            nsteps = nT
            out_res = torch.zeros(lenText, b, self.nclass).type_as(input.data)

            out_length = torch.zeros(b).type_as(input.data)
            now_step = 0
            while 0 in out_length and now_step < nsteps:
                tmp_result = C[now_step, :, :]
                out_res[now_step] = tmp_result
                tmp_result = tmp_result.topk(1)[1].squeeze(dim=1)
                for j in range(b):
                    if out_length[j] == 0 and tmp_result[j] == 0:
                        out_length[j] = now_step + 1
                now_step += 1
            for j in range(0, b):
                if int(out_length[j]) == 0:
                    out_length[j] = nsteps
            start = 0
            output = torch.zeros(int(out_length.sum()), self.nclass).type_as(input.data)
            for i in range(0, b):
                cur_length = int(out_length[i])
                output[start: start + cur_length] = out_res[0: cur_length, i, :]
                start += cur_length

            return output, out_length


class VisionLAN(nn.Module):
    '''
    Architecture of VisionLAN
    input
    input: input image
    label_pos: character index
    output
    text_pre: word-level prediction from VRM
    test_rem: remaining string prediction from MLM
    text_mas: occluded character prediction from MLM
    '''
    def __init__(self, strides, input_shape):
        super(VisionLAN, self).__init__()
        self.backbone = resnet.resnet45(strides, compress_layer=False)
        self.input_shape = input_shape
        self.MLM_VRM = MLM_VRM()
    def forward(self, input, label_pos, training_stp, Train_in = True):
        # extract features
        features = self.backbone(input)
        # MLM + VRM
        if Train_in:
            text_pre, test_rem, text_mas, mask_map = self.MLM_VRM(features[-1], label_pos, training_stp, is_Train=Train_in)
            return text_pre, test_rem, text_mas, mask_map
        else:
            output, out_length = self.MLM_VRM(features[-1], label_pos, training_stp, is_Train=Train_in)
            return output, out_length