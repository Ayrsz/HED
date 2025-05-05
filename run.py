import torch
import torch.nn as nn
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import hed 
from hed import HED
import train
import dataset
from sklearn.metrics import average_precision_score

#Hyper parameters
lr = 1e-3
batch_size = 10
device = "cuda:1" if torch.cuda.is_available() else "cpu"
epochs = 100
step_lr = False
momentum = 0.9
weight_decay = 0.0002

#Setting the init
dataset_path = "./BSDS500/"
dataset_train = dataset.BSDS500(dataset_path, subset = "train")
dataset_val = dataset.BSDS500(dataset_path, subset = "val")
dataset_test = dataset.BSDS500(dataset_path, subset = "test")

#Loaders
loader_train = dataset.create_dataloader(dataset_train, batch_size)
loader_val = dataset.create_dataloader(dataset_val, batch_size)

#Instance Modelcd 
model = hed.HED().to(device)
model.load_pre_treined_weigths_vgg(device)

#Optimizer
optmizer = torch.optim.SGD(params = model.parameters(), lr = lr, momentum= momentum, weight_decay= weight_decay)

#Training
train.train_model(loader_train, loader_val, dataset_test, model, optmizer, device, batch_size, step_lr, epochs)
train.save_new_state(model)

#Load pre-treined
#model = train.load_state(device)
#print(model)
#train.plot_result_compare_to_gt(model, train.load_original_hed(), dataset.BSDS500(dataset_path, subset = "test"), device)



