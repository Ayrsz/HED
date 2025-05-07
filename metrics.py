import csv
import torch
from sklearn.metrics import f1_score
import structure.hed as hed
import structure.dataset as dataset
import structure.train as train
import cv2 as cv
from non_max_supression import nms
import matplotlib.pyplot as plt

def ODS():
    pass
def OPS():
    pass
def F1_score(pred, gt):
    pass


def write_scores():
    pass

if __name__ == "__main__":
    pred = cv.imread("/workspaces/HED/states/model_13/epoch#final/final_result_epoch_#final.jpg", 0)
    pred = nms(pred) 
    plt.imshow(pred, cmap = "gray")
    plt.show()

