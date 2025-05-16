import torch
import torch.nn as nn
import torch.nn.functional as F
#1 -> Edge
#0 -> Non Edge

def calc_hed_loss(pred_batch : torch.Tensor, gt_batch : torch.Tensor):
    #[Batch, Level_of_out, H, L]
    assert len(pred_batch.shape) == 4, f"Tensor needs to have four dimensions, but it has {len(pred_batch.shape)}, {gt_batch.shape}"
    loss_total = torch.zeros(pred_batch.shape[0], device = pred_batch.device)
    print(f"Entrada no hed: {gt_batch.shape}")
    for i, (side_outs, gt) in enumerate(zip(pred_batch, gt_batch)):
        print(f"Entrada no hed: {gt_batch.shape}")
        loss_total[i] = loss(side_outs, gt)
    return torch.sum(loss_total)

def loss(side_outs : torch.Tensor, gt : torch.Tensor):
    loss_total = 0

    gt = gt.float()
    print(gt.shape)
    gt = gt.view(-1)
    #Making binary enter
    if torch.max(gt) > 1:
         gt[gt >= (255 // 5)*3-1] = 1
         gt[gt < (255 // 5)*3-1] = 0
    else:
         gt[gt >= 3/5] = 1
         gt[gt < 3/5] = 0

    for i, pred in enumerate(side_outs):
        pred = pred.view(-1)
        assert len(pred) == len(gt), f"Tem algo errado, pred {pred.shape}, gt {gt.shape}"
        if i == len(pred) - 1:
            
            loss_fuse = F.binary_cross_entropy(pred, gt).sum()
            loss_total += loss_fuse
        else:
            weigth_positive = torch.sum(gt)/gt.numel()
            weigth_negative = 1 - weigth_positive
            indices_pos = (gt == 1).squeeze()
            indices_neg = (gt == 0).squeeze()

            loss_pos = -1*weigth_negative * F.binary_cross_entropy(pred[indices_pos], gt[indices_pos], reduction='none').sum()
            loss_neg = -1*weigth_positive * F.binary_cross_entropy(pred[indices_neg], gt[indices_neg]).sum()
            loss_total += loss_pos + loss_neg
    
    return loss_total

if __name__ == "__main__":
    pred = torch.rand((10, 5, 10, 10))
    gt = torch.randint(0, 255, (10, 1, 10, 10))
    ls = calc_hed_loss(pred, gt)
    print(ls)