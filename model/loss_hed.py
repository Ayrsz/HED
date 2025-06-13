import torch
import torch.nn as nn
import torch.nn.functional as F
#1 -> Edge
#0 -> Non Edge



def HEDLoss(pred, gt, use_sigmoid = False):
    
    gt_flat = gt.view(-1)
    gt_flat = torch.tile(gt_flat, (pred.shape[0], 1))
    gt_flat = gt_flat.view(-1)
    
    pred_flat = pred.view(-1)
    
    # Contar pixels positivos (borda) e negativos (não-borda)
    num_positive = torch.sum(gt_flat)
    num_negative = torch.sum(1.0 - gt_flat)
    # Calcular os pesos beta e 1-beta
    # Evita divisão por zero se não houver pixels de uma classe
    if num_positive == 0:
        beta = 0.0
    else:
        beta = num_negative / (num_positive + num_negative)
    mask = (gt_flat != 0).float()
    mask[mask == 0] = (1 - beta)
    mask[mask != 0] = beta  
    # Ponderação manual dos termos positivos e negativos
    cost = torch.nn.functional.binary_cross_entropy_with_logits(
    pred_flat.float(),gt_flat.float(), weight=mask, reduce= None)
    
    return cost
    


if __name__ == "__main__":
    pred = torch.rand((10, 5, 10, 10))
    gt = torch.randint(0, 2, (10, 10, 10))
    loss = HEDLoss(pred[0], gt[0], False)

