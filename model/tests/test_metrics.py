import torch
import model.tests.metrics as metrics
import original as original
from model.data.dataset import BSDS500, create_dataloader

def test_with_original():
    path = "/workspaces/HED/BSDS500"
    data = BSDS500(path, "test")
    
    model = original.load_original_hed()

    OIS = metrics.OIS(model, data)
    ODS = metrics.ODS(model, data)
    AP = metrics.AP(model, data)
    




if __name__ == "__main__":
    test_with_original()

