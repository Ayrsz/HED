import torch
import torch.nn as nn
#1 -> Edge
#0 -> Non Edge
def loss(pred, gt):

    gt = gt.float()
    pred = pred.float()
    loss_model = nn.BCEWithLogitsLoss()


    len_border = torch.sum(gt)
    weigth_bord = len_border / gt.numel()
    weigth_no_bord = 1 - weigth_bord

    indices_bord = torch.where(gt == 1)
    indices_no_bord = torch.where(gt == 0)

    loss_bord = weigth_bord * loss_model(pred[indices_bord], gt[indices_bord])
    loss_no_bord = weigth_no_bord * loss_model(pred[indices_no_bord], gt[indices_no_bord])

    return loss_bord + loss_no_bord

if __name__ == "__main__":
    pred = torch.rand((10, 10))
    gt = torch.randint(0, 2, (10, 10))
    ls = loss(pred, gt)
    print(ls)