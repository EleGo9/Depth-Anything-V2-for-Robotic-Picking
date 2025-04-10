import cv2
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

from dataset.transform import Resize, NormalizeImage, PrepareForNet, Crop
from dataset.transform import Resize, NormalizeImage, PrepareForNet
import numpy as np


class CEM(Dataset):
    def __init__(self, filelist_path, mode, max_depth, gt_unit_measure='mm', size=(518, 518)):
        
        self.mode = mode
        self.size = size
        with open(filelist_path, 'r') as f:
            self.filelist = f.read().splitlines()
        
        net_w, net_h = size
        self.max_depth = max_depth
        self.transform = Compose([
            Resize(
                width=net_w,
                height=net_h,
                resize_target=True if mode == 'train' else False,
                keep_aspect_ratio=True,
                ensure_multiple_of=14,
                resize_method='lower_bound',
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            PrepareForNet(),
        ] + ([Crop(size[0])] if self.mode == 'train' else []))

        self.unit_measure_conv = {'m': 1., 'dm': 10, 'cm': 100, 'mm': 1000}
        self.gt_unit_measure = gt_unit_measure
        print('Be careful: max_depth must be in meters ')
    
    def __getitem__(self, item):
        img_path = self.filelist[item].split(' ')[0]
        depth_path = self.filelist[item].split(' ')[1]
        
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0
        
        depth = cv2.imread(depth_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH) / self.unit_measure_conv[self.gt_unit_measure] # originally mm --> converted to meters
        sample = self.transform({'image': image, 'depth': depth}) #TODO: check mean and standard deviation!!!
        
        sample['image'] = torch.from_numpy(sample['image'])
        sample['depth'] = torch.from_numpy(sample['depth'])

        sample['valid_mask'] = (sample['depth'] <= self.max_depth) #change this according to max depth (0.8 meters)
        
        sample['image_path'] = self.filelist[item].split(' ')[0]
        
        return sample

    def __len__(self):
        return len(self.filelist)