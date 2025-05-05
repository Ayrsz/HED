import numpy as np
import torch
import torch.nn as nn
#1 -> Edge
#0 -> Non Edge
def loss(pred, gt):
    gt = gt.float()
    pred = pred.float()
    loss_model = nn.BCEWithLogitsLoss()


    len_border = torch.sum(gt)
    weigth_bord = len_border / len(gt)
    weigth_no_bord = 1 - weigth_bord

    indices_bord = torch.where(gt == 1)
    indices_no_bord = torch.where(gt == 0)

    loss_bord = weigth_bord * loss_model(pred[indices_bord], gt[indices_bord])
    loss_no_bord = weigth_no_bord * loss_model(pred[indices_no_bord], gt[indices_no_bord])
    return loss_bord + loss_no_bord


pred = torch.randint(0, 2, (100,))
gt = torch.randint(0, 2, (100,))
l = loss(pred, gt)

print(l) 