import csv
import torch
from sklearn.metrics import f1_score
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from non_max_supression import nms

def F1_score(pred, gt) -> float:
    
    pred = pred.flatten().astype(np.float32)/255
    gt = gt.flatten()/255
    gt = gt.astype(np.uint8)
    f1 = f1_score(gt, pred)
    return f1 

def OIS(model, data) -> float:
    bests_f1 = []
    values = np.linspace(0, 1, 200)
    for (image, gt) in data:
        best_f1 = 0
        for ts in values:
            _, pred_bin = cv.threshold(pred, ts, 1, cv.THRESH_BINARY)
            f1 = f1_score(gt, pred_bin)
            if best_f1 < f1:
                best_f1 = f1
        
        bests_f1.append(best_f1)
    
    return np.mean(bests_f1)
                
def ODS(model, data) -> float:

    values = np.linspace(0, 1, 200)
    f1s_means = []
    pred = np.ones(size = (10, 10)) #test

    for ts in values:
        f1_values = []
        for (image, gt) in data:
            #Prevision the model:
            #########################

            #Just to TEST
            pred = np.ones(size = (10, 10))
            _, pred_bin = cv.threshold(pred, ts, 1, cv.THRESH_BINARY)
            f1_values.append(F1_score(pred_bin, gt))
        
        mean_f1 = np.mean(f1_values)
        f1s_means.append((ts, mean_f1))
    
    return f1s_means

def AP(model, data_test, device) -> float:
    total_precision = []
    
    for image_input, gt in data_test:
        image_input = torch.unsqueeze(image_input, 0).to(device)
        return_from_model = treat_image_to_np(model(image_input))

        gt = np.array(gt, dtype = np.float32)
        y_true = gt.flatten().astype(np.uint8)
        y_pred = return_from_model.flatten().astype(np.float32)
        total_precision.append(average_precision_score(y_true, y_pred))

    return np.mean(total_precision)


def write_scores(dir, idx):
    columns = ["AP", "OIS", "ODS", "F1_VALUES"]
    

if __name__ == "__main__":
    pred = cv.imread("imagem_bordas_urso.jpg", 0)
    pred = pred + np.random.randint(low = 0, high = 15, size = pred.shape)
    gt = cv.imread("imagem_bordas_urso.jpg", 0)
    gt[gt >= 0.25*255] = 255
    gt[gt < 0.25*255] = 0

    