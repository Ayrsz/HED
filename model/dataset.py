import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models
import numpy as np
import os
import cv2 as cv
import matplotlib.pyplot as plt
from imutils import rotate_bound
from tqdm import tqdm

def rotate_im(im):
    return rotate_bound(im, -90)

class BSDS500(Dataset):
    def __init__(self, root_bsds : str, subset = "train") -> None:
        #If subset not in, directory invalid!
        assert subset in ["train", "test", "val"]

        super().__init__()


        #Get the sources on the subsets
        self.images_dir = os.path.join(root_bsds, "images", subset, "")
        self.gt_dir = os.path.join(root_bsds, "GT", subset, "")

        #Get files in the subset
        self.images_files = sorted(os.listdir(self.images_dir))
        gt_files_with_mat = os.listdir(self.gt_dir)
        self.gt_files = sorted([path for path in gt_files_with_mat if not(path.endswith(".mat"))])



    def __getitem__(self, i : int):
        name_image = self.images_files[i]
        name_gt = self.gt_files[i]

        dir_image = os.path.join(self.images_dir, name_image)
        dir_gt = os.path.join(self.gt_dir, name_gt)

        image = cv.imread(dir_image).astype(np.float32)
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        if image.shape[0] < image.shape[1]:
             image = rotate_im(image)
        image = image / 255.0
        image = torch.from_numpy(image).permute(2, 0, 1)
        # 0 -> 255
        # 255 - > 0


        gt = cv.imread(dir_gt).astype(np.float32)
        gt = cv.cvtColor(gt, cv.COLOR_BGR2GRAY)
        if gt.shape[0] < gt.shape[1]:
            gt = rotate_im(gt)
        gt = torch.from_numpy(gt)
        gt = 255 - gt
        gt = gt/255.0
        gt[gt >= 2/5] = 1
        gt[gt < 2/5] = 0
        gt = 1 - gt

        return image, gt


    def __len__(self) -> int:
        return len(self.images_files)
    
def create_dataloader(dataset : Dataset, batch_size):
    return DataLoader(dataset, batch_size = batch_size, shuffle = True)





