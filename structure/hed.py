import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
import numpy as np
import os
import cv2 as cv
import matplotlib.pyplot as plt
from imutils import rotate_bound
from tqdm import tqdm

class HED(nn.Module):
    def __init__(self, loss_func = nn.BCEWithLogitsLoss()):
        super().__init__()
        self.loss_fn = loss_func

        mean = [0.485, 0.456, 0.406]  # RGB
        std = [0.229, 0.224, 0.225]   # RGB
        
        self.treat = transforms.Compose([
            transforms.Normalize(mean, std)
        ])
        
        self.VGG1 = nn.Sequential(
            nn.Conv2d(in_channels = 3, out_channels = 64, kernel_size= 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3, stride= 1, padding = 1),
            nn.ReLU()
        )

        self.VGG2 = nn.Sequential(
            nn.MaxPool2d(kernel_size = 2, stride = 2),
            nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.Conv2d(in_channels = 128, out_channels = 128, kernel_size = 3, stride= 1, padding = 1),
            nn.ReLU()
        )

        self.VGG3 = nn.Sequential(
            nn.MaxPool2d(kernel_size = 2, stride = 2),
            nn.Conv2d(in_channels = 128, out_channels = 256, kernel_size = 3, stride= 1, padding = 1),
            nn.ReLU(),
            nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU()
        )

        self.VGG4 = nn.Sequential(
            nn.MaxPool2d(kernel_size = 2, stride = 2),
            nn.Conv2d(in_channels = 256, out_channels = 512, kernel_size = 3, stride= 1, padding = 1),
            nn.ReLU(),
            nn.Conv2d(in_channels = 512, out_channels = 512, kernel_size = 3, stride= 1, padding = 1),
            nn.ReLU(),
            nn.Conv2d(in_channels = 512, out_channels = 512, kernel_size = 3, stride= 1, padding = 1),
            nn.ReLU()
        )

        self.VGG5= nn.Sequential(
            nn.MaxPool2d(kernel_size = 2, stride = 2),
            nn.Conv2d(in_channels = 512, out_channels = 512, kernel_size = 3, stride= 1, padding = 1),
            nn.ReLU(),
            nn.Conv2d(in_channels = 512, out_channels = 512, kernel_size = 3, stride= 1, padding = 1),
            nn.ReLU(),
            nn.Conv2d(in_channels = 512, out_channels = 512, kernel_size = 3, stride= 1, padding = 1),
            nn.ReLU()
        )

        #We need to turn the network to only one dimension.
        self.generate_image_from_side_resultVGG1 = nn.Sequential(
            nn.Conv2d(in_channels = 64 , out_channels= 1, kernel_size= 1, stride = 1, padding = 0),
            nn.Upsample(scale_factor= 1, mode = "bilinear", align_corners= False)
            )
                                                                 
        self.generate_image_from_side_resultVGG2 = nn.Sequential(
            nn.Conv2d(in_channels = 128, out_channels= 1, kernel_size= 1, stride = 1, padding = 0),
            nn.Upsample(scale_factor= 2, mode = "bilinear", align_corners= False)
            )
        
        self.generate_image_from_side_resultVGG3 = nn.Sequential(
            nn.Conv2d(in_channels = 256, out_channels= 1, kernel_size= 1, stride = 1, padding = 0),
            nn.Upsample(scale_factor= 4, mode = "bilinear", align_corners= False)
            )
        

        self.generate_image_from_side_resultVGG4 = nn.Sequential(
            nn.Conv2d(in_channels = 512, out_channels= 1, kernel_size= 1, stride = 1, padding = 0),
            nn.Upsample(scale_factor= 8, mode = "bilinear", align_corners= False)
            )
        
        self.generate_image_from_side_resultVGG5 = nn.Sequential(
            nn.Conv2d(in_channels = 512, out_channels= 1, kernel_size= 1, stride = 1, padding = 0),
            nn.Upsample(scale_factor= 16, mode = "bilinear", align_corners= False)
            )

        #Join of all side outputs
        self.join_images = nn.Sequential(
            nn.Conv2d(5, 64, kernel_size = 3, padding = 1),
            nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size=1)
        )


    def forward(self, x, return_side_outputs = False):
        x = self.treat(x)
        #Vggnet, without the dense layer and the last maxpool layer
        vgg1 = self.VGG1(x)
        vgg2 = self.VGG2(vgg1)
        vgg3 = self.VGG3(vgg2)
        vgg4 = self.VGG4(vgg3)
        vgg5 = self.VGG5(vgg4)

        #Output has X channels -> 1 Channel (gray scale)
        result_1 = self.generate_image_from_side_resultVGG1(vgg1)
        result_2 = self.generate_image_from_side_resultVGG2(vgg2)
        result_3 = self.generate_image_from_side_resultVGG3(vgg3)
        result_4 = self.generate_image_from_side_resultVGG4(vgg4)
        result_5 = self.generate_image_from_side_resultVGG5(vgg5)

        #Results have differents sizes
        image_one_scaled = F.interpolate(result_1 , size = (x.shape[2], x.shape[3]), mode = "bilinear", align_corners= False)
        image_two_scaled = F.interpolate(result_2, size = (x.shape[2], x.shape[3]), mode = "bilinear", align_corners= False)
        image_three_scaled = F.interpolate(result_3 , size = (x.shape[2], x.shape[3]), mode = "bilinear", align_corners= False)
        image_four_scaled = F.interpolate(result_4 , size = (x.shape[2], x.shape[3]), mode = "bilinear", align_corners= False)
        image_five_scaled = F.interpolate(result_5, size = (x.shape[2], x.shape[3]), mode = "bilinear", align_corners= False)

        stacked = torch.cat([image_one_scaled, image_two_scaled, image_three_scaled, image_four_scaled, image_five_scaled], dim = 1)
        final_response = self.join_images(stacked)
        if not(return_side_outputs):
            return final_response
        else:
            return final_response, stacked

    def load_pre_treined_weigths_vgg(self, device):
        vgg16_pretrained = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).to(device)
        vgg_layers = vgg16_pretrained.features

        # VGG1
        self.VGG1[0].weight.data = vgg_layers[0].weight.data.clone()
        self.VGG1[0].bias.data   = vgg_layers[0].bias.data.clone()
        self.VGG1[2].weight.data = vgg_layers[2].weight.data.clone()
        self.VGG1[2].bias.data   = vgg_layers[2].bias.data.clone()

        # VGG2
        self.VGG2[1].weight.data = vgg_layers[5].weight.data.clone()
        self.VGG2[1].bias.data   = vgg_layers[5].bias.data.clone()
        self.VGG2[3].weight.data = vgg_layers[7].weight.data.clone()
        self.VGG2[3].bias.data   = vgg_layers[7].bias.data.clone()

        # VGG3
        self.VGG3[1].weight.data = vgg_layers[10].weight.data.clone()
        self.VGG3[1].bias.data   = vgg_layers[10].bias.data.clone()
        self.VGG3[3].weight.data = vgg_layers[12].weight.data.clone()
        self.VGG3[3].bias.data   = vgg_layers[12].bias.data.clone()
        self.VGG3[5].weight.data = vgg_layers[14].weight.data.clone()
        self.VGG3[5].bias.data   = vgg_layers[14].bias.data.clone()

        # VGG4
        self.VGG4[1].weight.data = vgg_layers[17].weight.data.clone()
        self.VGG4[1].bias.data   = vgg_layers[17].bias.data.clone()
        self.VGG4[3].weight.data = vgg_layers[19].weight.data.clone()
        self.VGG4[3].bias.data   = vgg_layers[19].bias.data.clone()
        self.VGG4[5].weight.data = vgg_layers[21].weight.data.clone()
        self.VGG4[5].bias.data   = vgg_layers[21].bias.data.clone()

        # VGG5
        self.VGG5[1].weight.data = vgg_layers[24].weight.data.clone()
        self.VGG5[1].bias.data   = vgg_layers[24].bias.data.clone()
        self.VGG5[3].weight.data = vgg_layers[26].weight.data.clone()
        self.VGG5[3].bias.data   = vgg_layers[26].bias.data.clone()
        self.VGG5[5].weight.data = vgg_layers[28].weight.data.clone()
        self.VGG5[5].bias.data   = vgg_layers[28].bias.data.clone()