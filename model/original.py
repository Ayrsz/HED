import cv2 as cv
import os
from dataset import BSDS500, create_dataloader
import numpy as np
import matplotlib.pyplot as plt
from train import load_state, list_only_dirs 
import torch

class CropLayer(object):
        def __init__(self, params, blobs):
            self.xstart = 0
            self.xend = 0
            self.ystart = 0
            self.yend = 0

        # Our layer receives two inputs. We need to crop the first input blob
        # to match a shape of the second one (keeping batch size and number of channels)
        def getMemoryShapes(self, inputs):
            inputShape, targetShape = inputs[0], inputs[1]
            batchSize, numChannels = inputShape[0], inputShape[1]
            height, width = targetShape[2], targetShape[3]

            self.ystart = int((inputShape[2] - targetShape[2]) / 2)
            self.xstart = int((inputShape[3] - targetShape[3]) / 2)
            self.yend = self.ystart + height
            self.xend = self.xstart + width

            return [[batchSize, numChannels, height, width]]

        def forward(self, inputs):
            return [inputs[0][:,:,self.ystart:self.yend,self.xstart:self.xend]]

def load_original_hed():
    protoPath = "./original_hed/deploy.prototxt"
    modelPath = "./original_hed/hed_pretrained_bsds.caffemodel"
    model = cv.dnn.readNetFromCaffe(protoPath, modelPath)
    cv.dnn_registerLayer('Crop', CropLayer)
    return model

def forward_return_cv_convence_original(original_hed, im_input):
    assert isinstance(im_input, torch.Tensor), f"Im need to be a tensor, got {type(im_input)}" 
    im = im_input.permute(1, 2, 0).cpu().numpy()
    im = im * 255
    (H, W) = im.shape[:2]
   
    blob = cv.dnn.blobFromImage(im,
                                scalefactor=0.7,
                                mean=(105, 117, 123),
                                swapRB= False,
                                crop=True)
    original_hed.setInput(blob)
    hed_original_im = original_hed.forward()
    hed_original_im = hed_original_im[0,0,:,:]
    hed_original_im = (hed_original_im*255).astype(np.uint8)
    return hed_original_im

def forward_return_cv_convence_implementation(model, im_input_as_tensor, device):
    hed_this_code = model.forward(im_input_as_tensor.to(device))
    hed_this_code = hed_this_code.squeeze()
    hed_this_code = torch.Tensor.cpu(hed_this_code)
    hed_this_code = torch.nn.Sigmoid()(hed_this_code)
    hed_this_code = hed_this_code / torch.max(hed_this_code)
    hed_this_code = np.array(hed_this_code.detach(), dtype = np.float32)
    hed_this_code = (hed_this_code*255).astype(np.uint8)
    return hed_this_code

if __name__ == "__main__":
    net = load_original_hed()
    data = BSDS500("./BSDS500/", subset= "test")
    im, gt = data[0]
    hed_original = forward_return_cv_convence_original(net, im)
    implementation = load_state("cuda:2", 4)
    hed_this_code = forward_return_cv_convence_implementation(implementation, im, "cuda:2")

    plt.imshow(~hed_this_code, cmap = 'gray')
    plt.title("Hed Original")
    plt.axis("off")
    plt.show()

         
    
    #net.setInput()