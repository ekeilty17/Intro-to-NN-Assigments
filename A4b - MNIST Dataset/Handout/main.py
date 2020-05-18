import numpy as np
import matplotlib.pyplot as plt

from cnn import CNN
from mlp import MLP
from utils import *
from plot import *

def total_correct(predictions, labels):
    corr = ( torch.argmax(predictions, dim=1) == labels )
    return int(corr.sum())

def evaluate(model, loader, loss_fnc, total_correct):
    total_corr = 0
    evaluate_data = 0
    total_batches = 0
    running_loss = 0.0
    with torch.no_grad():
        for data, labels in loader:
            predictions = model(data.float())
            running_loss += loss_fnc(input=predictions, target=labels)
            total_corr += total_correct(predictions, labels)
            
            evaluate_data += labels.size(0)
            total_batches += 1
    
    loss = running_loss / total_batches
    acc = float(total_corr) / evaluate_data
    return float(loss), float(acc)

# Training Loop
def train(model, train_loader, valid_loader, opts):
    
    """
    TODO: Implement Training Loop (exactly the same as before)
    """

    # plots
    if opts.plot:
        display_statistics(<train_loss>, <train_acc>, <valid_loss>, <valid_acc>)
        plt.close()

    # return final statistic values
    model.eval()
    return <final_train_loss>, <final_valid_loss>, <final_train_acc>, <final_valid_acc>


if __name__ == "__main__":

    # getting args
    opts = AttrDict()
    args_dict = {
        "seed": None,
        "lr": 0.01,
        "epochs": 5,
        "batch_size": 100,
        "eval_every": 100,
        "optimizer": torch.optim.Adam,
        "loss_fnc": torch.nn.CrossEntropyLoss(),
        "plot": True,
        "model_type": "MLP"     # 'CNN' or 'MLP'
    }
    opts.update(args_dict)

    # error checking
    if not opts.model_type in ["CNN", "MLP"]:
        raise ValueError(f"{opts.model_type} architecture not supported")

    # random seed for initializing weights
    if not opts.seed is None:
        torch.manual_seed(opts.seed)

    # getting data
    train_loader, valid_loader = load_data(batch_size=opts.batch_size, seed=opts.seed)

    # both plots some sample data and gets the input size for the MLP
    input_size = None
    for images, labels in train_loader:
        input_size = (images.size(2), images.size(3))
        if opts.plot:
            plot_mnist_images(images.squeeze(), labels, 18)
            plt.close()
        break

    # creating model2
    num_classes = 10
    model = CNN(num_classes) if opts.model_type == "CNN" else MLP(input_size, num_classes)

    # training model
    final_statistics = train(model, train_loader, valid_loader, opts)

    
