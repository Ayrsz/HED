import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
import numpy as np
import os
import cv2 as cv
import matplotlib.pyplot as plt
from dataset import BSDS500
from tqdm import tqdm
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_fscore_support
from loss_hed import HEDLoss

def train_loop(dataloader, model, optimizer, eppoch, device):

    model.train()
    

    for (X, y) in tqdm(dataloader, desc = f"Época {eppoch+1}", ):
        
        loss = torch.zeros(1).cuda(device)

        X, y = X.to(device), y.to(device)

        #side -> [fused, side1, ..., side5

        pred, side_outs = model.forward(X, return_side_outputs = True)
        for out, gt in zip(side_outs, y):
            #print(f"out: {out.shape}")
            #print(f"ground_truth: {gt.shape}")
            loss = loss + HEDLoss(out, gt, False)

        loss = loss / X.shape[0]
        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

def test_loop(dataloader, model, device):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.eval()
    num_batches = len(dataloader)
    test_loss = torch.zeros(1).cuda(device)

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for (X, y) in dataloader:
            
            X, y = X.to(device), y.to(device)
            pred, side_outs = model.forward(X, return_side_outputs = True)
            for out, gt in zip(side_outs, y):
                test_loss = test_loss + HEDLoss(out, gt, False)

    test_loss /= num_batches

    print(f"Avg loss: {test_loss.item():>8f} \n")
    return test_loss

def train_model(loader_train, loader_val, dataset_test, model, optmizer, device, batch_size, step_lr, epochs = 10):
    losses = np.empty(epochs)
    new_idx = get_new_idx()
    
    

    os.mkdir(f"./model/checkpoints/model_{new_idx}")
    if(epochs >= 10):
        number_of_epochs_to_print = [int(epochs*(fraction/4)) for fraction in range(4)]

    
    if (step_lr and epochs > 1):
        steper = StepLR(optimizer = optmizer, step_size= int(0.5*epochs), gamma = 0.1)
    if (step_lr and epochs <= 1):
        raise AttributeError("step_lr are true and the number of epochs are lower than 2")
    
    for i in range(epochs):
        train_loop(loader_train, model, optmizer, i, device)
        loss_in_epoch = test_loop(loader_val, model, device)
        losses[i] = loss_in_epoch
        if epochs >= 10 and i in number_of_epochs_to_print:
            save_side_outputs(model, i, BSDS500(root_bsds = "./BSDS500/", subset = "test"), new_idx, device)
        
        if step_lr:
            steper.step()

    save_side_outputs(model = model, epoch = "final", data = BSDS500(root_bsds = "./BSDS500/", subset = "test"), idx_dir = new_idx,  device = device)
    write_graph_loss(losses)

    

def list_only_dirs(root_path):
    return [nome for nome in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, nome))]

def get_new_idx(increment = True):
    dirs = list_only_dirs("./model/checkpoints/")
    if len(dirs) > 0:
        numbers = [int(dir.split("_")[1]) for dir in dirs]
        maxi = np.max(numbers)
    else:
        maxi = 0
    
    if increment:
        new_state = f"{maxi + 1:02d}"
    else:
        new_state = f"{maxi:02d}"
    
    return new_state

def save_new_state(model):
    new_state = get_new_idx(False)
    torch.save(model, "model/checkpoints/" + f"model_{new_state}" + f"/model#{new_state}.pth")

def load_state(device, state = None):

    dirs = list_only_dirs("./model/checkpoints/")

    #Check if has dirs with versions
    if len(dirs) > 0:
        numbers = [int(dir.split("_")[1]) for dir in dirs]
        maxi = np.max(numbers)
    else:
        raise Exception("No Statements avaible to load")

    #If NONE, return the last version
    if state is None:
        maxi = np.max(numbers)
        idx = np.where(numbers == maxi)[0][0]
    else:
        #Return a tuple, where [0] is a vector of a single value
        try:
            idx = np.where(numbers == state)[0][0]
        except Exception as e:
            print(f"Erro, indice invalido para load: {e}")

    model = torch.load(f"./model/checkpoints/model_{numbers[idx]:02d}/model#{numbers[idx]:02d}.pth", weights_only = False, map_location= torch.device(device))
    return model


def plot_result_compare_to_gt(model, net_from_article, data, device):
    fig = plt.figure(figsize = (20, 10))
    im_tensor, gt = data[141]

    im = im_tensor.permute(1, 2, 0).cpu().numpy()
    gt = np.array(-1*gt)
    #################### Original ##################################
    fig.add_subplot(1, 4, 1)
    plt.title("Original")
    plt.imshow(im)
    plt.axis("off")

    #################### Ground Truth ##################################
    fig.add_subplot(1, 4, 2)
    plt.imshow(gt, cmap = "gray")
    plt.title("Ground Truth")
    plt.axis("off")

    #################### Original Hed ##################################
    fig.add_subplot(1, 4, 3)
    (H, W) = im.shape[:2]
    im = im*255
    blob = cv.dnn.blobFromImage(im,
                                scalefactor=0.7,
                                mean=(105, 117, 123),
                                swapRB= False,
                                crop=True)

    net_from_article.setInput(blob)
    hed_original = net_from_article.forward()
    hed_original = hed_original[0,0,:,:]
    hed_original = (hed_original*255).astype(np.uint8)


    plt.imshow(hed_original, cmap = 'gray')
    plt.title("Hed Original")
    plt.axis("off")

    #################### This code result ##################################
    fig.add_subplot(1, 4, 4)
    im_tensor = im_tensor.unsqueeze(0)
    hed_this_code = model.forward(im_tensor.to(device))
    hed_this_code = torch.Tensor.cpu(hed_this_code[0])
    hed_this_code.squeeze_()
    hed_this_code = nn.Sigmoid()(hed_this_code)
    hed_this_code = hed_this_code / torch.max(hed_this_code)
    hed_this_code = np.array(hed_this_code.detach(), dtype = np.float32)
    hed_this_code = (hed_this_code*255).astype(np.uint8)
    _, hed_this_code = cv.threshold(hed_this_code, 30, 255, cv.THRESH_BINARY_INV)
    
    #Para cada pixel, encontra o nível mais próximo
    #gt = niveis[np.abs(gt[..., None] - niveis).argmin(axis=-1)]
    plt.title("Implementation")
    plt.imshow(255 - hed_this_code, cmap = "gray")
    plt.axis("off")
    plt.show()




#### WRITE LOSS ####
def write_graph_loss(losses):
    x = [i+1 for i in range(len(losses))]
    plt.xlabel("Eppoch")
    plt.ylabel("Loss")
    plt.ylim((0, np.max(losses)))
    plt.title("Loss per eppoch")
    plt.plot(x, losses)
    plt.grid()
    idx_this_train = get_new_idx(increment = False)
    plt.savefig(f"./model/checkpoints/model_{idx_this_train}/loss.png", bbox_inches='tight', dpi=300)
    plt.close()


#### AUXILIAR FUNCTION - SAVE PARTIAL STATES ####
def treat_image_to_np(im_tensor):
    im_tensor = torch.Tensor.cpu(im_tensor)
    im = np.array(im_tensor.detach(), dtype = np.float32)
    return im


#### SAVE PARTIAL STATES ####
def save_side_outputs(model, epoch, data, idx_dir, device):
    im_tensor, gt = data[0]
    fig = plt.figure(figsize = (10, 6))
    plt.suptitle("Model partial result in the epoch: " + f"{epoch}")
    # Treat the results
    im_tensor = im_tensor.unsqueeze(0)
    final_output, side_outputs = model.forward(im_tensor.to(device), return_side_outputs= True)
    side_outputs = torch.squeeze(side_outputs)
    final_output = torch.squeeze(final_output)
    side_outputs = torch.cat([final_output.unsqueeze(0), side_outputs], dim = 0)


    dir_epoch = f"./model/checkpoints/model_{idx_dir}/epoch#{epoch}"
    os.mkdir(dir_epoch)
    for i, response in enumerate(side_outputs):
        fig.add_subplot(2, 4, i + 1)
        response = treat_image_to_np(response)

        plt.imshow(response, cmap = "gray")

        if i == 0:
            plt.title("Final output")
            cv.imwrite(dir_epoch + "/final_result_" + f"epoch_#{epoch}.jpg", (response * 255).astype(np.uint8))
        else:
            plt.title("Side output " + f"{6-i}")
            cv.imwrite(dir_epoch + f"/side_output{6-i}_epoch#{epoch}.jpg", (response * 255).astype(np.uint8))
        plt.axis("off")

    fig.savefig(dir_epoch + f"/epoch_{epoch}.png", bbox_inches='tight', dpi=300)
    plt.close()

def return_cv_convence_pytorch(model, image_input):
    image_input = image_input.to("cuda:1")
    result = model.forward(image_input, return_side_outputs = False)
    result_np = result.cpu()
    result_np = result_np.detach().numpy()
    return (result_np*255).astype(np.uint8)