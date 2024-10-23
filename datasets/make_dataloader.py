from torchvision import transforms
from .Dataloader_University import Sampler_University,Dataloader_University,train_collate_fn
from .random_erasing import RandomErasing
# from .autoaugment import ImageNetPolicy, CIFAR10Policy
import torch
import albumentations as A
import cv2
from albumentations.pytorch import ToTensorV2
# from torchvision.transforms import v2

def make_dataset(opt):
    #--------------------------------------------- Load Data ---------------------------------------------------------------
    mean=[0.485, 0.456, 0.406]
    std=[0.229, 0.224, 0.225]
    img_size=[384,384]
            
    train_sat_transforms = A.Compose([A.ImageCompression(quality_lower=90, quality_upper=100, p=0.5),
                                      A.Resize(opt.h, opt.w, interpolation=cv2.INTER_CUBIC, p=1.0),
                                      A.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.15, always_apply=False, p=0.5),
                                      A.OneOf([
                                               A.AdvancedBlur(p=1.0),
                                               A.Sharpen(p=1.0),
                                              ], p=0.3),
                                      A.OneOf([
                                               A.GridDropout(ratio=0.3, p=1.0),
                                               A.CoarseDropout(max_holes=20,
                                                               max_height=int(0.2*opt.h),
                                                               max_width=int(0.2*opt.w),
                                                               min_holes=10,
                                                               min_height=int(0.1*opt.h),
                                                               min_width=int(0.1*opt.w),
                                                               p=1.0),
                                              ], p=0.3),
                                      A.RandomRotate90(p=1.0),
                                      A.Normalize(mean, std),
                                      ToTensorV2(),
                                      ])
    
    train_drone_transforms = A.Compose([A.ImageCompression(quality_lower=90, quality_upper=100, p=0.5),
                                        A.Resize(opt.h, opt.w, interpolation=cv2.INTER_CUBIC, p=1.0),
                                        A.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.15, always_apply=False, p=0.5),
                                        A.OneOf([
                                                 A.AdvancedBlur(p=1.0),
                                                 A.Sharpen(p=1.0),
                                              ], p=0.3),
                                        A.OneOf([
                                                 A.GridDropout(ratio=0.3, p=1.0),
                                                 A.CoarseDropout(max_holes=20,
                                                                 max_height=int(0.2*opt.h),
                                                                 max_width=int(0.2*opt.w),
                                                                 min_holes=10,
                                                                 min_height=int(0.1*opt.h),
                                                                 min_width=int(0.1*opt.w),
                                                                 p=1.0),
                                              ], p=0.3),
                                        A.Normalize(mean, std),
                                        ToTensorV2(),
                                        ])



    data_transforms = {
        'train': train_drone_transforms,
        'satellite': train_sat_transforms}

    # train_all = ''
    # if opt.train_all:
    #     train_all = '_all'

    # image_datasets = {}
    # image_datasets['satellite'] = datasets.ImageFolder(os.path.join(data_dir, 'satellite'),
    #                                                    data_transforms['satellite'])
    # image_datasets['street'] = datasets.ImageFolder(os.path.join(data_dir, 'street'),
    #                                                 data_transforms['train'])
    # image_datasets['drone'] = datasets.ImageFolder(os.path.join(data_dir, 'drone'),
    #                                                data_transforms['train'])
    # image_datasets['google'] = datasets.ImageFolder(os.path.join(data_dir, 'google'),
    #                                                 data_transforms['train'])
    #
    # dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=opt.batchsize,
    #                                               shuffle=True, num_workers=opt.num_worker, pin_memory=True)
    #                # 8 workers may work faster
    #                for x in ['satellite', 'street', 'drone', 'google']}
    # dataset_sizes = {x: len(image_datasets[x]) for x in ['satellite', 'drone']}
    # class_names = image_datasets['street'].classes
    # print(dataset_sizes)
    # return dataloaders,class_names,dataset_sizes

    # custom Dataset

    image_datasets = Dataloader_University(opt.data_dir,transforms=data_transforms)
    samper = Sampler_University(image_datasets,batchsize=opt.batchsize,sample_num=opt.sample_num,triplet_loss=opt.triplet_loss)
    dataloaders =torch.utils.data.DataLoader(image_datasets, batch_size=opt.batchsize,sampler=samper,num_workers=0, pin_memory=True,collate_fn=train_collate_fn)
    # {'satellite' : 701, 'street' : 701, 'drone' : 701}
    dataset_sizes = {x: len(image_datasets)*opt.sample_num for x in ['satellite', 'street', 'drone']}
    class_names = image_datasets.cls_names # ['0839', '0842', ...]
    return dataloaders,class_names,dataset_sizes

