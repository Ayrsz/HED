import csv
import torch
from sklearn.metrics import f1_score
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import average_precision_score, f1_score
from original import load_original_hed, forward_return_cv_convence_original
from dataset import BSDS500, create_dataloader
from train import return_cv_convence_pytorch, load_state

#import non_max_supression 

def nms(image) -> np.ndarray:
    assert isinstance(image, np.ndarray)
    
    image = np.squeeze(image)
    if len(image.shape) == 3:
        image = cv.cvtColor(image, cv.COLOR_RGB2GRAY).astype(np.float32)
    image = cv.GaussianBlur(image, (5, 5), sigmaX = 0)
    
    cv.imwrite("before_nms.jpg", image)
    Gy = cv.Sobel(image, cv.CV_64F, dx = 0,  dy = 1)
    Gx = cv.Sobel(image, cv.CV_64F, dx = 1, dy = 0)
    
    angles = np.arctan2(Gy, Gx)
    angles = np.degrees(angles)
    angles[angles < 0] += 180  


    H, W = image.shape
    non_max_supression = np.zeros((H, W), dtype=np.float32)

    for i in range(1, H - 1):
        for j in range(1, W - 1):
            direction = angles[i, j]
            m = image[i, j]

            # Define os pixels vizinhos de acordo com a direção do gradiente
            if (0 <= direction < 22.5) or (157.5 <= direction <= 180):
                before = image[i, j - 1]
                after = image[i, j + 1]
            elif 22.5 <= direction < 67.5:
                before = image[i - 1, j + 1]
                after = image[i + 1, j - 1]
            elif 67.5 <= direction < 112.5:
                before = image[i - 1, j]
                after = image[i + 1, j]
            elif 112.5 <= direction < 157.5:
                before = image[i - 1, j - 1]
                after = image[i + 1, j + 1]

            if m >= before and m >= after:
                non_max_supression[i, j] = m
            else:
                non_max_supression[i, j] = 0

    # Normaliza resultado para 0-1
    non_max_supression = non_max_supression / (non_max_supression.max() + 1e-8)
    cv.imwrite("after_nms.jpg", (non_max_supression*255).astype(np.uint8))

    return non_max_supression

def thinning(image) -> np.ndarray:
    pass


def F1_score(pred, gt):
    assert len(np.unique(pred)) <= 2, "pred need to be binary"
    pred = pred.flatten()
    gt = gt.numpy().flatten()
    gt = gt.astype(np.uint8)
    return f1_score(gt, pred)

def OIS(model, data, forwarder):

    bests_f1 = np.empty(shape = len(data))

    thresholds = np.linspace(0, 1, 255)

    for i, (image, gt) in enumerate(data):
        best_f1_to_image = 0
        for t in thresholds:
            image, gt = data[17]
            gt = 1 - gt
            pred = forwarder(model, image)
            pred = nms(pred)

            _, pred = cv.threshold(pred, thresh =  t, maxval= 1, type = cv.THRESH_BINARY)
            
            value = F1_score(pred, gt)
            cv.imwrite("pred.jpg", (pred*255).astype(np.uint8))
            cv.imwrite("gt.jpg", (gt.numpy()*255).astype(np.uint8))
            
            if value > best_f1_to_image:
                best_f1_to_image = value
        
        print(f"The {i}st image with best_f1 as {best_f1_to_image}")
        bests_f1[i] = best_f1_to_image

    #cv.imwrite("im.jpg", (pred*255).astype(np.uint8))
    return np.mean(bests_f1)
    
                
def ODS(model, data, forwarder):
    f1_values = np.empty(shape = len(data))
    best_f1 = 0
    thresholds = np.linspace(0, 1, 255)

    for t in thresholds:
        for i, (image, gt) in enumerate(data):
            gt = 1 - gt
            pred = forwarder(model, image)
            pred = nms(pred)

            _, pred = cv.threshold(pred, thresh =  t, maxval= 1, type = cv.THRESH_BINARY)
            value = F1_score(pred, gt)
            #cv.imwrite("im.jpg", (pred*255).astype(np.uint8))

            f1_values[i] = value
            
        mean = np.mean(f1_values)
        
        if mean > best_f1:
            best_f1 = mean
    
    return best_f1
    
    

def AP(model, data_test, device):
    pass
    


def write_scores(dir, idx, average_precision, optimal_image_scale, optimal_dataset_scale):
    columns = ["AP", "OIS", "ODS"]

    with open(dir+"metrics.csv", "w") as f:
        writer = csv.DictWriter(f, fieldnames= columns)
        writer.writeheader()
        writer.writerow({"AP":average_precision, "OIS":optimal_image_scale, "ODS":optimal_dataset_scale})

        

    

if __name__ == "__main__":
    #model =  load_state("cuda:1")
    model =  load_original_hed()
    data = BSDS500('./BSDS500/', "test")
    im, gt = data[0]
    #cv.imwrite("im.jpg", (gt*255).astype(np.uint8))
    model_type = "caffe"
    if model_type == "pytorch":
        forwarder = return_cv_convence_pytorch
    elif model_type == "caffe":
        forwarder = forward_return_cv_convence_original
    
    r = forwarder(model, im)
    r = r.squeeze()
    print(r.shape)
    r = (nms(r)).astype(np.uint8)
    cv.imwrite("oi.jpg", r)

    ois = OIS(model, data, forwarder)
    #ods = ODS(model, data, forwarder)
    #model_type = "pytorch"
    #write_scores("", 0, 0, ois, ods)
    #print(F1_score(hed, gt))
    #plt.imshow(gt)
    #plt.show()
    