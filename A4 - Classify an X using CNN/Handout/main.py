import numpy as np
import matplotlib.pyplot as plt

from single_layer_cnn import SingleLayerCNN
from mlp import MLP
from utils import *
from plot import *

def total_correct(predictions, labels):
    corr = (predictions > 0.5).squeeze().long() == labels.long()
    return int(corr.sum())

def evaluate(model, loader, loss_fnc, total_correct):
    model.eval()
    total_corr = 0
    evaluate_data = 0
    total_batches = 0
    running_loss = 0.0
    with torch.no_grad():
        for data, labels in loader:
            predictions = model(data.float())
            running_loss += loss_fnc(input=predictions.squeeze(), target=labels.float())
            total_corr += total_correct(predictions, labels)
            
            evaluate_data += labels.size(0)
            total_batches += 1
    
    loss = running_loss / total_batches
    acc = float(total_corr) / evaluate_data
    model.train()
    return loss, acc

# Training Loop
def train(model, train_loader, valid_loader, opts):

    # random seed for initializing weights
    if not opts.seed is None:
        torch.manual_seed(seed)

    raise NotImplementedError

    # plots
    if opts.plot:
        display_statistics(<train_loss>, <train_acc>, <valid_loss>, <valid_acc>)
        plt.close()
        kernels = [W.data.numpy().flatten() for W in model.conv[0].weight]
        display_kernels(kernels, 3, 300)

    # return final statistic values
    model.eval()
    return <final_train_loss>, <final_train_acc>, <final_valid_loss>, <final_valid_acc>


if __name__ == "__main__":
    # we want to classify any array containing this target kernel
    target = np.array([
                        [1, 0, 1], 
                        [0, 1, 0], 
                        [1, 0, 1]
                    ])

    # getting args
    opts = AttrDict()
    args_dict = {
        "seed": None,
        "lr": 0.01,
        "epochs": 10,
        "batch_size": 100,
        "eval_every": 10,
        "optimizer": torch.optim.Adam,      # doesn't work if you use torch.optim.SGD
        "loss_fnc": torch.nn.BCEWithLogitsLoss(),
        "plot": True
    }
    opts.update(args_dict)

    # getting data
    train_loader, valid_loader = load_data(batch_size=opts.batch_size, seed=opts.seed)

    # creating model
    model = SingleLayerCNN(target.shape, 10)

    # training model
    final_statistics = train(model, train_loader, valid_loader, opts)

    
