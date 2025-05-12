import torch
import torch.nn as nn
#1 -> Edge
#0 -> Non Edge


def loss(pred : torch.Tensor, gt, fuse_layer = False):
    gt = gt.float()
    pred = pred.float()
    loss_total = 0

    
    if not(fuse_layer):
        loss_model = nn.CrossEntropyLoss()
    else:
        loss_model = nn.CrossEntropyLoss()


    
    #Size of batch
    batch_size = pred.shape[0]

    print(pred.shape)
    #One batch per time
    for sample_pred, sample_gt in zip(pred, gt):
        #Channels
        print(f"sample: {sample_pred.shape}")
        sample_gt.squeeze_()
        sample_gt[sample_gt >= 255//2 ]  = 255
        sample_gt[sample_gt < 255//2 ] = 0
        for ch in sample_pred:
            #Setup the weigths

            len_border = torch.sum(sample_gt)
            weigth_bord = len_border / sample_gt.numel()
            weigth_no_bord = 1 - weigth_bord
    
            #Get the indices of bords and no bord
            indices_bord = torch.where(sample_gt == 255)
            indices_no_bord = torch.where(sample_gt == 0)
            print(f"pred: {ch.shape}, gt:{sample_gt.shape}")
    
            loss_bord = weigth_bord * loss_model(ch[indices_bord], sample_gt[indices_bord])
            loss_no_bord = weigth_no_bord * loss_model(ch[indices_no_bord], sample_gt[indices_no_bord])
            loss_total += loss_bord + loss_no_bord
    return loss_total

if __name__ == "__main__":
    pred = torch.rand((10, 5, 10, 10))
    gt = torch.randint(0, 255, (10, 1, 10, 10))
    ls = loss(pred, gt)
    print(ls)