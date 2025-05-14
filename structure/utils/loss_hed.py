import torch
import torch.nn as nn
#1 -> Edge
#0 -> Non Edge

def calc_hed_loss(pred_batch : torch.Tensor, gt_batch):
    #[Batch, Level_of_out, H, L]
    assert len(pred_batch.shape) == 4, f"Tensor needs to have four dimensions, but it has {len(pred_batch.shape)}"
    loss_total = [0]*pred_batch.shape[-1]
    for i, (side_outs, gt) in enumerate(zip(pred_batch, gt_batch)):
        loss_total[i] = loss(side_outs, gt)
        
    return torch.sum(loss_total)

def loss(side_outs : torch.Tensor, gt : torch.Tensor, fuse_layer = False):
    
    gt = gt.float()
    
    #Making binary enter
    if torch.max(gt) > 1:
         gt[gt >= (255 // 5)*3-1] = 1
         gt[gt < (255 // 5)*3-1] = 0
    else:
         gt[gt >= 3/5] = 1
         gt[gt < 3/5] = 0
    
    weigth_positive = torch.sum(gt)/gt.numel()
    weigth_negative = 1 - weigth_positive
    

    side_outs = side_outs.float()

    #Size of batch

    #+ -> bORDA
    #One batch per time
 
    for ch in side_outs:
            
            #Setup the weigths
            

            

            loss_bord = -1*weigth_bord * loss_model(ch[indices_bord], sample_gt[indices_bord])
            loss_no_bord = -1*weigth_no_bord * loss_model(ch[indices_no_bord], sample_gt[indices_no_bord])
            loss_total += loss_bord + loss_no_bord
    return loss_total

if __name__ == "__main__":
    pred = torch.rand((10, 5, 10, 10))
    gt = torch.randint(0, 255, (10, 1, 10, 10))
    ls = calc_hed_loss(pred, gt)
    print(ls)