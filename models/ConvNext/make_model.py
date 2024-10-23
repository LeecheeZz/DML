import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from timm.models import create_model
from .backbones.model_convnext import convnext_tiny
from .backbones.resnet import Resnet
import numpy as np
from torch.nn import init
from torch.nn.parameter import Parameter
import math

'''-----------------------SE Module-----------------------------'''
class SE_Block(nn.Module):
    def __init__(self, inchannel=12, ratio=4):
        super(SE_Block, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(inchannel, inchannel // ratio, bias=False),  #  c -> c/r
            nn.ReLU(),
            nn.Linear(inchannel // ratio, inchannel, bias=False),  #  c/r -> c
            nn.Sigmoid()
        )
        
    def forward(self, x):
            b, c, h, w = x.size()
            y = self.gap(x).view(b, c)
            y = self.fc(y).view(b, c, 1, 1)
            y = x * y.expand_as(x)
            return y

class SEAttention(nn.Module):
    def __init__(self):
        super(SEAttention, self).__init__()
        self.h_se = SE_Block()
        self.w_se = SE_Block()
    def forward(self, x):
        x_perm1 = x.permute(0,2,1,3).contiguous() # H-SE
        x_out1 = self.h_se(x_perm1)
        x_out11 = x_out1.permute(0,2,1,3).contiguous()
        x_perm2 = x.permute(0,3,2,1).contiguous() # W-SE
        x_out2 = self.w_se(x_perm2)
        x_out21 = x_out2.permute(0,3,2,1).contiguous()
        return x_out11, x_out21

class ClassBlock(nn.Module):
    def __init__(self, input_dim, class_num, droprate, relu=False, bnorm=True, num_bottleneck=512, linear=True, return_f = False):
        super(ClassBlock, self).__init__()
        self.return_f = return_f
        add_block = []
        if linear:
            add_block += [nn.Linear(input_dim, num_bottleneck)]
        else:
            num_bottleneck = input_dim
        if bnorm:
            add_block += [nn.BatchNorm1d(num_bottleneck)]
        if relu:
            add_block += [nn.LeakyReLU(0.1)]
        if droprate>0:
            add_block += [nn.Dropout(p=droprate)]
        add_block = nn.Sequential(*add_block)
        add_block.apply(weights_init_kaiming)

        classifier = []
        classifier += [nn.Linear(num_bottleneck, class_num)]
        classifier = nn.Sequential(*classifier)
        classifier.apply(weights_init_classifier)

        self.add_block = add_block
        self.classifier = classifier
    def forward(self, x):
        x = self.add_block(x) # [8, 512]
        if self.training:
            if self.return_f:
                f = x
                x = self.classifier(x) # [8, 701]
                return x,f
            else:
                x = self.classifier(x)
                return x
        else:
            return x


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)

    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight.data, std=0.001)
        nn.init.constant_(m.bias.data, 0.0)

class build_convnext(nn.Module):
    def __init__(self, num_classes, block = 4, return_f=False, resnet=False):
        super(build_convnext, self).__init__()
        self.return_f = return_f
        if resnet:
            convnext_name = "resnet101"
            print('using model_type: {} as a backbone'.format(convnext_name))
            self.in_planes = 2048
            self.convnext = Resnet(pretrained=True)
        else:
            convnext_name = "convnext_tiny"
            print('using model_type: {} as a backbone'.format(convnext_name))
            if 'base' in convnext_name:
                self.in_planes = 1024
            elif 'large' in convnext_name:
                self.in_planes = 1536
            elif 'xlarge' in convnext_name:
                self.in_planes = 2048
            else:
                self.in_planes = 768
            self.convnext = create_model(convnext_name, pretrained=True)

        self.num_classes = num_classes
        self.classifier1 = ClassBlock(self.in_planes, num_classes, 0.5, return_f=return_f)
        self.classifier2 = ClassBlock(self.in_planes, num_classes, 0.5, return_f=return_f)
        self.block = block
        self.LPN_block = 2
        self.se_attention1 = SE_Block(inchannel=768, ratio=4)
        self.se_attention2 = SEAttention()
        for i in range(self.block * 2):
            name = 'classifier_mcb' + str(i + 1) # classifier_mcb1   classifier_mcb2
            setattr(self, name, ClassBlock(self.in_planes, num_classes, 0.5, return_f=self.return_f))

    def forward(self, x):
        gap_feature, part_features = self.convnext(x) # [8, 768, 8, 8]  [8, 768, 8, 8]
        se_features = self.se_attention2(part_features) #[8, 768, 8, 8]  [8, 768, 8, 8]
        
        gap_feature = self.se_attention1(gap_feature) # [8, 768, 8, 8]
        gap_feature = self.get_part_pool(gap_feature, pool='avg').view(gap_feature.size(0), gap_feature.size(1), -1) # [8, 768, 2]
        
        gap_feature1 = gap_feature[:,:,0].view(gap_feature.size(0), gap_feature.size(1)) # [8, 768]
        gap_feature2 = gap_feature[:,:,1].view(gap_feature.size(0), gap_feature.size(1))
        convnext_feature1 = self.classifier1(gap_feature1)
        convnext_feature2 = self.classifier2(gap_feature2)
        
        se_list = []
        for i in range(self.block):
            x = self.get_part_pool(se_features[i], pool='avg') # [8, 768, 2, 1]
            x = x.view(x.size(0), x.size(1), -1) # [8, 768, 2]
            se_list.append(x) # [8, 768, 2]
            # se_list.append(se_features[i].mean([-2, -1]))
        seatten_features = torch.cat(se_list, dim=2) # [8, 768, 4]
        if self.block == 0:
            y = []
        else:
            y = self.part_classifier(self.block, seatten_features, cls_name='classifier_mcb') # [([8, 701], [8, 512]), ([8, 701], [8, 512]),...]

        if self.training:
            y = y + [convnext_feature1] + [convnext_feature2] # [([8, 701],[8, 512]), ([8, 701],[8, 512]), ([8, 701],[8, 512])]
            # y = y + [convnext_feature]# [([8, 701],[8, 512]), ([8, 701],[8, 512]), ([8, 701],[8, 512])]
            if self.return_f:
                cls, features = [], []
                for i in y:
                    cls.append(i[0]) # [8, 701]
                    features.append(i[1]) # [8, 512]
                return cls, features #[[8, 701],[8, 701],[8, 701],[8, 701],[8, 701],[8, 701]]
                                     #[[8, 512],[8, 512],[8, 512],[8, 512],[8, 512],[8, 512]]
        else:
            # ffeature = convnext_feature.view(convnext_feature.size(0), -1, 1)
            convnext_feature = torch.stack((convnext_feature1,convnext_feature2), dim=2) # [8, 512, 2]
            # convnext_feature = convnext_feature.view(convnext_feature.shape[0], -1, 1)
            y = torch.cat([y, convnext_feature], dim=2) # [8, 512, 4] cat [8, 512, 2] = [8, 512, 6]

        return y 
        
        #------------------------------------------------------------------------------------------------------
        
        

    def part_classifier(self, block, x, cls_name='classifier_mcb'): # [8, 768, 2]
        part = {}
        predict = {}
        for i in range(block): # 2 * block
            part[i] = x[:, :, i].view(x.size(0), -1) # [8, 768]
            name = cls_name + str(i+1) # classifier_mcb1 classifier_mcb2
            c = getattr(self, name)
            predict[i] = c(part[i]) # [8, 701]
        y = []
        for i in range(block):
            y.append(predict[i])
        if not self.training:
            return torch.stack(y, dim=2)
        return y
    
    def get_part_pool(self, x, pool='avg', no_overlap=True):  # LPN
        result = []
        if pool == 'avg':
            pooling = torch.nn.AdaptiveAvgPool2d((1,1))
        elif pool == 'max':
            pooling = torch.nn.AdaptiveMaxPool2d((1,1)) 
        H, W = x.size(2), x.size(3)
        c_h, c_w = int(H/2), int(W/2)
        per_h, per_w = H/(2*self.LPN_block),W/(2*self.LPN_block)
        if per_h < 1 and per_w < 1:
            new_H, new_W = H+(self.LPN_block-c_h)*2, W+(self.LPN_block-c_w)*2
            x = nn.functional.interpolate(x, size=[new_H,new_W], mode='bilinear', align_corners=True)
            H, W = x.size(2), x.size(3)
            c_h, c_w = int(H/2), int(W/2)
            per_h, per_w = H/(2*self.LPN_block),W/(2*self.LPN_block)
        per_h, per_w = math.floor(per_h), math.floor(per_w)
        for i in range(self.LPN_block):
            i = i + 1
            if i < self.LPN_block:
                x_curr = x[:,:,(c_h-i*per_h):(c_h+i*per_h),(c_w-i*per_w):(c_w+i*per_w)]
                if no_overlap and i > 1:
                    x_pre = x[:,:,(c_h-(i-1)*per_h):(c_h+(i-1)*per_h),(c_w-(i-1)*per_w):(c_w+(i-1)*per_w)] 
                    x_pad = F.pad(x_pre,(per_h,per_h,per_w,per_w),"constant",0)
                    x_curr = x_curr - x_pad
                avgpool = pooling(x_curr)
                result.append(avgpool)
            else:
                if no_overlap and i > 1:
                    x_pre = x[:,:,(c_h-(i-1)*per_h):(c_h+(i-1)*per_h),(c_w-(i-1)*per_w):(c_w+(i-1)*per_w)]
                    pad_h = c_h-(i-1)*per_h
                    pad_w = c_w-(i-1)*per_w
                    # x_pad = F.pad(x_pre,(pad_h,pad_h,pad_w,pad_w),"constant",0)
                    if x_pre.size(2)+2*pad_h == H:
                        x_pad = F.pad(x_pre,(pad_h,pad_h,pad_w,pad_w),"constant",0)
                    else:
                        ep = H - (x_pre.size(2)+2*pad_h)
                        x_pad = F.pad(x_pre,(pad_h+ep,pad_h,pad_w+ep,pad_w),"constant",0)
                    x = x - x_pad
                avgpool = pooling(x)
                result.append(avgpool)
        return torch.cat(result, dim=2)


def make_convnext_model(num_class,block = 4,return_f=False,resnet=False):
    print('===========building convnext===========')
    model = build_convnext(num_class,block=block,return_f=return_f,resnet=resnet)
    return model


