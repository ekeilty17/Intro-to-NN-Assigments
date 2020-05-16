import torch
import numpy as np
import matplotlib.pyplot as plt

from mlp import MultiLayerPerceptron
from utils import *
from plot import *

def total_correct(predictions, labels):
    """ 
    TODO: count number of predictions that match the labels
        param: torch.tensor, torch.tensor
        return: int
    """
    raise NotImplementedError

def evaluate(model, loader, opts):
    
    model.eval()
    total_corr = 0
    evaluate_data = 0
    total_batches = 0
    running_loss = 0.0

    with torch.no_grad():
        for data, labels in loader:
            predictions = model(data.float())
            running_loss += opts.loss_fnc(input=predictions.squeeze(), target=labels.float())
            total_corr +=  opts.total_correct(predictions, labels)

            evaluate_data += labels.size(0)
            total_batches += 1

    loss = running_loss / total_batches
    acc = float(total_corr) / evaluate_data
    model.train()
    return float(loss), float(acc)

def train(model, train_loader, valid_loader, opts):

    # initializing model
    """
    TODO: put model in training mode, initialize optimizer
    """

    # Initializing loss and accuracy arrays for plots
    """
    TODO: Create datastructures to log loss and accuracy for both training and validation data 
    """

    """ Training Loop """
    for e in range(opts.epochs):
        
        # variables to keep track of training loss and accuracy
        evaluated_data = 0
        total_batches = 0
        running_loss = 0.0
        running_acc = 0.0
        
        # batching
        for i, batch in enumerate(train_loader):
            data, labels = batch

            """
            TODO: PyTorch things
            """

            # update training loss and accuracy statistics
            running_loss += loss
            running_acc += total_correct(predictions, labels)

            # updating counters
            evaluated_data += labels.size(0)
            total_batches += 1

        # evaluating model
        """
        TODO: put model in evaluation model
        """

        # training data statistics
        avg_loss = running_loss / total_batches
        avg_acc = running_acc / evaluated_data
        """
        TODO: update training loss and accuracy datastructures
        """

        # validation data statistics
        loss, acc = evaluate(model, valid_loader, opts)
        """
        TODO:   update validation loss and accuracy datastructures
        """
        
        
        print(f"epoch: {e+1:4d}\tbatch: {i+1:5d}\tloss: {Loss['train'][-1]:.4f}\tacc: {Acc['valid'][-1]:.4f}")
        """
        TODO: put model back into training mode
        """

    # plots
    if opts.plot:
        display_statistics(<train_loss_list>, <train_acc_list>, <valid_loss_list>, <valid_acc_list>)
        plt.close()
    
    # return final statistic values
    model.eval()
    return <final_train_loss>, <final_train_acc>, <final_valid_loss>, <final_valid_acc>

if __name__ == "__main__":

    # getting args
    opts = AttrDict()
    args_dict = {
        "seed": None,
        "lr": 0.1,
        "epochs": 50,
        "batch_size": 100,
        "optimizer": torch.optim.SGD,
        "loss_fnc": torch.nn.MSELoss(),
        "plot": True
    }
    opts.update(args_dict)

    # label information
    label_name = "income"
    label_mapping = {'<=50K': 0, '>50K': 1}

    # random seed
    if not opts.seed is None:
        torch.manual_seed(opts.seed)

    # load data
    #   You only have to call this once, then it'll save the preprocessed data into a new .csv file
    train_loader, valid_loader = load_data( "../data/adult.csv", label_name, label_mapping,
                                            preprocess=True, batch_size=opts.batch_size, seed=opts.seed)

    #   Then you can just call it like this, which should make thing run faster
    #train_loader, valid_loader = load_data( "../data/adult_preprocessed.csv", label_name, label_mapping,
    #                                        preprocess=False, batch_size=opts.batch_size, seed=opts.seed)

    # creating model
    input_size = len(train_loader.dataset.data[0, :])
    output_size = 1
    model = MultiLayerPerceptron(input_size, output_size, """other parameters""")

    # training model
    final_statistics = train(model, train_loader, valid_loader, opts)
