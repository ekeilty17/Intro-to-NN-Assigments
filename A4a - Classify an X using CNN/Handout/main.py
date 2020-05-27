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
            running_loss += loss_fnc(input=predictions.squeeze(), target=labels.float()).detach().item()
            total_corr += total_correct(predictions, labels)
            
            evaluate_data += labels.size(0)
            total_batches += 1
    
    loss = running_loss / total_batches
    acc = float(total_corr) / evaluate_data
    model.train()
    return loss, acc

# Training Loop
def train(model, train_loader, valid_loader, opts):

    """
    TODO: Write Training Loop (exactly the same as before)
    """

    # plots
    if opts.plot:
        display_statistics(<train_loss>, <train_acc>, <valid_loss>, <valid_acc>)
        plt.close()
        
        if opts.model_type == "CNN":
            kernels = [W.data.numpy().flatten() for W in model.<name of convolution module>[0].weight]
            display_kernels(kernels, 3, 300)
        else:
            modual_list = [model.<name of first module>, model.<name of second module, ...]
            layers = np.array([fc[0].weight.data.numpy() for fc in modual_list])
            for kernels in layers:
                display_kernels(kernels, 5, 250)
                plt.close()

    # return final statistic values
    model.eval()
    return <final_train_loss>, <final_valid_loss>, <final_train_acc>, <final_valid_acc>


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
        "optimizer": torch.optim.Adam,      # doesn't work if you use torch.optim.SGD
        "loss_fnc": torch.nn.BCEWithLogitsLoss(),
        "plot": True,
        "model_type": "CNN"     # 'CNN' or 'MLP'
    }
    opts.update(args_dict)

    # error checking
    if not opts.model_type in ["CNN", "MLP"]:
        raise ValueError(f"{opts.model_type} architecture not supported")

    # random seed for initializing weights
    if not opts.seed is None:
        torch.manual_seed(opts.seed)

    # getting data
    train_loader, valid_loader = load_data(batch_size=opts.batch_size)

    # creating model
    model = SingleLayerCNN("""TODO: Inputs""") if opts.model_type == "CNN" else MLP("""TODO: Inputs""")

    # training model
    final_statistics = train(model, train_loader, valid_loader, opts)