import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
import numpy as np
import os
import cv2 as cv
import matplotlib.pyplot as plt
from structure.dataset import BSDS500
from tqdm import tqdm
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_fscore_support

def train_loop(dataloader, model, optimizer, eppoch, device):
    model.train()


    for (X, y) in tqdm(dataloader, desc = f"Época {eppoch+1}", ):
        X, y = X.to(device), y.to(device)
        y = y.unsqueeze(1)

        pred = model(X)
        loss = model.loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

def test_loop(dataloader, model, device):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.eval()
    num_batches = len(dataloader)
    test_loss = 0

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for (X, y) in dataloader:
            X, y = X.to(device), y.to(device)
            y = torch.unsqueeze(y, 1)
            pred = model(X)
            test_loss += model.loss_fn(pred, y).item()

    test_loss /= num_batches

    print(f"Avg loss: {test_loss:>8f} \n")
    return test_loss

def train_model(loader_train, loader_val, dataset_test, model, optmizer, device, batch_size, step_lr, epochs = 10):
    losses = np.empty(epochs)
    new_idx = get_new_idx()
    
    os.mkdir(f"./states/model_{new_idx}")
    if(epochs >= 10):
        number_of_epochs_to_print = [int(epochs*(fraction/4)) for fraction in range(4)]

    if (step_lr):
        steper = StepLR(optimizer = optmizer, step_size= (3*epochs)//4, gamma = 0.1)
    
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
    
    #txt_file = f"./states/model_{new_idx}/Metrics.txt"

    #with open(txt_file, "w") as f:
        #f.write("Metrics of the implementation\n")
        #f.close()

    #save_average_precision(model, data_test = dataset_test, dir = txt_file, device = device)
    

def list_only_dirs(root_path):
    return [nome for nome in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, nome))]

def get_new_idx(increment = True):
    dirs = list_only_dirs("./states")
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
    torch.save(model, "./states/" + f"model_{new_state}" + f"/model#{new_state}.pth")

def load_state(device, state = None):

    dirs = list_only_dirs("./states/")

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

    model = torch.load(f"./states/model_{numbers[idx]:02d}/model#{numbers[idx]:02d}.pth", weights_only = False, map_location= torch.device(device))
    return model

def load_original_hed():
    protoPath = "./original_hed/deploy.prototxt"
    modelPath = "./original_hed/hed_pretrained_bsds.caffemodel"
    model = cv.dnn.readNetFromCaffe(protoPath, modelPath)
    cv.dnn_registerLayer('Crop', CropLayer)
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
    mean_pixel_values= np.average(im, axis = (0,1))
    blob = cv.dnn.blobFromImage(im,
                                scalefactor=0.7,
                                mean=(105, 117, 123),
                                swapRB= False,
                                crop=True)

    net_from_article.setInput(blob)
    hed_original = net_from_article.forward()
    hed_original = hed_original[0,0,:,:]
    hed_original = (hed_original*255).astype(np.uint8)
    _, hed_original = cv.threshold(hed_original, 256/2, 255, cv.THRESH_BINARY_INV)

    plt.imshow(hed_original, cmap = 'gray')
    plt.title("Hed Original")
    plt.axis("off")

    #################### This code result ##################################
    fig.add_subplot(1, 4, 4)
    im_tensor = im_tensor.unsqueeze(0)
    hed_this_code = model.forward(im_tensor.to(device))
    hed_this_code = hed_this_code.squeeze()
    hed_this_code = torch.Tensor.cpu(hed_this_code)
    hed_this_code = nn.Sigmoid()(hed_this_code)
    hed_this_code = hed_this_code / torch.max(hed_this_code)
    hed_this_code = np.array(hed_this_code.detach(), dtype = np.float32)
    hed_this_code = (hed_this_code*255).astype(np.uint8)
    _, hed_this_code = cv.threshold(hed_this_code, 30, 255, cv.THRESH_BINARY_INV)
    
    #Para cada pixel, encontra o nível mais próximo
    #gt = niveis[np.abs(gt[..., None] - niveis).argmin(axis=-1)]
    plt.title("Implementation")
    plt.imshow(~hed_this_code, cmap = "gray")
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
    plt.savefig(f"./states/model_{idx_this_train}/loss.png", bbox_inches='tight', dpi=300)
    plt.close()


#### AUXILIAR FUNCTION - SAVE PARTIAL STATES ####
def treat_image_to_np(im_tensor):
    im_tensor = torch.Tensor.cpu(im_tensor)
    im_tensor = nn.Sigmoid()(im_tensor)
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
    final_output = torch.sigmoid(final_output)
    side_outputs = torch.sigmoid(side_outputs)
    joint = torch.cat([final_output, side_outputs], dim = 1)
    joint.squeeze_()

    dir_epoch = f"./states/model_{idx_dir}/epoch#{epoch}"
    os.mkdir(dir_epoch)
    for i, response in enumerate(joint):
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


#### METRICS ####
def get_average_precision(model, data_test, device) -> float:
    total_precision = []
    
    for image_input, gt in data_test:
        image_input = torch.unsqueeze(image_input, 0).to(device)
        return_from_model = treat_image_to_np(model(image_input))

        gt = np.array(gt, dtype = np.float32)
        y_true = gt.flatten().astype(np.uint8)
        y_pred = return_from_model.flatten().astype(np.float32)
        total_precision.append(average_precision_score(y_true, y_pred))

    return np.mean(total_precision)

def save_average_precision(model, data_test, dir, device):
    AP = get_average_precision(model, data_test, device)
    try:
        with open(dir, "a") as f:
            f.write(f"AP: {AP:.02f}")
    except Exception as e:
        print(f"{dir} erro : {e}")

def return_metrics(model, data_test, device):
    thresholds = np.linspace(0.01, 0.99, 99)
    ods = 0
    best_threshold_ods = 0

    f1s_per_image = []

    for thresh in thresholds:
        all_preds = []
        all_targets = []

        for i in range(len(data_test)):
            im_original, gt = data_test[i]
            im_original.unsqueeze_(0)

            im_pred = model.forward(im_original.to(device))
            im_pred_np = treat_image_to_np(im_pred)

            bin_pred = (im_pred_np >= thresh).astype(np.uint8)
            all_preds.extend(bin_pred.flatten())
            all_targets.extend(gt.flatten())

        precision, recall, f1, _ = precision_recall_fscore_support(
            all_targets, all_preds, average='binary', zero_division=0
        )

        if f1 > ods:
            ods = f1
            best_threshold_ods = thresh

    # Calcular OIS: melhor threshold individual por imagem
    for i in range(len(data_test)):
        best_f1 = 0
        im_original, gt = data_test[i]


        gt = gt.cpu().numpy()

        im_original.unsqueeze_(0)

        im_pred = model.forward(im_original.to(device))
        im_pred_np = treat_image_to_np(im_pred)

        for thresh in thresholds:
            bin_pred = (im_pred_np >= thresh).astype(np.uint8)
            precision, recall, f1, _ = precision_recall_fscore_support(
                gt.flatten(), bin_pred.flatten(), average='binary', zero_division=0
            )
            best_f1 = max(best_f1, f1)
        f1s_per_image.append(best_f1)
        torch.cuda.empty_cache()


    ois = np.mean(f1s_per_image)

    return ods, best_threshold_ods, ois

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
