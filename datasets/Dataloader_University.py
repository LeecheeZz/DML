import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import Sampler
from torchvision import datasets, transforms
import os
import numpy as np
from PIL import Image
import random
import cv2

class Dataloader_University(Dataset):
    # def __init__(self,root,transforms,names=['satellite','street','drone','google']): ############################
    def __init__(self,root,transforms,names=['satellite','drone']):
        super(Dataloader_University).__init__()
        self.transforms_drone_street = transforms['train']
        self.transforms_satellite = transforms['satellite']
        self.root = root
        self.names =  names
        dict_path = {}
        for name in names:
            dict_ = {}
            for cls_name in os.listdir(os.path.join(root, name)): # '0839'
                img_list = os.listdir(os.path.join(root,name,cls_name)) # '0839'/'0839.jpg'
                img_path_list = [os.path.join(root,name,cls_name,img) for img in img_list] # '../University1652/data/train/'0839'/'0839.jpg'
                dict_[cls_name] = img_path_list # {'0839' : ['../University1652/data/train/satellite/'0839'/'0839.jpg', ...]}
            dict_path[name] = dict_ 
            # {'satellite' : {'0839' : ['../University1652/data/train/satellite/'0839'/'0839.jpg', ...]}, ...}
            # dict_path[name+"/"+cls_name] = img_path_list

        cls_names = os.listdir(os.path.join(root,names[0])) # ['0839', '0842', ...]
        cls_names.sort() # ['0839', '0842', ...]
        map_dict={i:cls_names[i] for i in range(len(cls_names))} # {0 : '0839', 1 : '0842' , ...}

        self.cls_names = cls_names 
        self.map_dict = map_dict
        self.dict_path = dict_path
        self.index_cls_nums = 2

    def sample_from_cls(self,name,cls_num):
        img_path = self.dict_path[name][cls_num]
        img_path = np.random.choice(img_path,1)[0] # 从['../University1652/data/train/drone/'0839'/'0839.jpg', ...]中随机选择一个
        # img = Image.open(img_path)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def __getitem__(self, index): # 返回相同类别的transform后的卫星、地面和无人机图像以及index(0-700)
        cls_nums = self.map_dict[index]
        img = self.sample_from_cls("satellite",cls_nums)
        # img_s = self.transforms_satellite(img)
        img_s = self.transforms_satellite(image=img)['image']
        
        # img = self.sample_from_cls("street",cls_nums) ###################################
        # img_st = self.transforms_drone_street(image=img)['image'] ##################################
        # img_st = self.transforms_drone_street(img)

        img = self.sample_from_cls("drone",cls_nums) # img的size:(512, 512)
        # img_d = self.transforms_drone_street(img)
        img_d = self.transforms_drone_street(image=img)['image']
        # return img_s,img_st,img_d,index ##################################
        return img_s,img_d,index


    def __len__(self):
        return len(self.cls_names)



class Sampler_University(object):
    r"""Base class for all Samplers.
    Every Sampler subclass has to provide an :meth:`__iter__` method, providing a
    way to iterate over indices of dataset elements, and a :meth:`__len__` method
    that returns the length of the returned iterators.
    .. note:: The :meth:`__len__` method isn't strictly required by
              :class:`~torch.utils.data.DataLoader`, but is expected in any
              calculation involving the length of a :class:`~torch.utils.data.DataLoader`.
    """

    def __init__(self, data_source, batchsize=8,sample_num=1,triplet_loss=0):
        self.data_len = len(data_source) # 701
        self.batchsize = batchsize # 8
        self.sample_num = sample_num # 1
        self.triplet_loss = triplet_loss

    def __iter__(self):
        list = np.arange(0,self.data_len) # [0, 1, 2, ..., 700]
        nums = np.repeat(list,self.sample_num,axis=0) # [0, 1, 2, ..., 700]
        np.random.shuffle(nums) # 打乱顺序
        # print(nums)
        return iter(nums) # 可以使用循环或者next()函数来逐个访问迭代器中的元素

    def __len__(self):
        return len(self.data_source)


def train_collate_fn(batch): # 在加载数据时，会使用一个函数来进行数据的组合和处理，这个函数就称为"collate函数"，或简称"collate"
    # img_s,img_st,img_d,ids = zip(*batch) # batch : (img_s,img_st,img_d,ids)  ###############################
    img_s,img_d,ids = zip(*batch)
    ids = torch.tensor(ids,dtype=torch.int64)
    # [8, 3, 256, 256]  返回[8张卫星图像, 8个索引],[8张地面图像, 8个索引],[8张无人机图像, 8个索引]
    # return [torch.stack(img_s, dim=0),ids],[torch.stack(img_st,dim=0),ids], [torch.stack(img_d,dim=0),ids] #########################
    return [torch.stack(img_s, dim=0),ids], [torch.stack(img_d,dim=0),ids]



if __name__ == '__main__':
    transform_train_list = [
        # transforms.RandomResizedCrop(size=(opt.h, opt.w), scale=(0.75,1.0), ratio=(0.75,1.3333), interpolation=3), #Image.BICUBIC)
        transforms.Resize((256, 256), interpolation=3),
        transforms.Pad(10, padding_mode='edge'),
        transforms.RandomCrop((256, 256)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]


    transform_train_list ={"satellite": transforms.Compose(transform_train_list),
                            "train":transforms.Compose(transform_train_list)}
    datasets = Dataloader_University(root="../../University1652-Baseline-master/data/train",transforms=transform_train_list,names=['satellite','street','drone'])
    samper = Sampler_University(datasets,8,1)
    dataloader = DataLoader(datasets,batch_size=8,num_workers=0,sampler=samper,collate_fn=train_collate_fn)
    for data_s,data_st,data_d in dataloader: # input : [8, 3, 256, 256]  label : 0 - 700
        inputs, labels = data_s
        inputs2, labels2 = data_st
        inputs3, labels3 = data_d
        print(labels, labels2, labels3)



