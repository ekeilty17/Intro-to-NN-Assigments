import torch
import numpy as np
import matplotlib.pyplot as plt 

from MLP import MultiLayerPerceptron
from utils import *
from plot import *

# TODO
def total_correct(predictions, labels):
    """ 
    TODO: count number of predictions that match the labels
        param: torch.tensor, torch.tensor
        return: int
    """
    raise NotImplementedError

def evaluate(model, loader, loss_fnc):
    total_corr = 0
    evaluated_data = 0
    total_batches = 0
    running_loss = 0.0
    with torch.no_grad():
        for batch, labels in loader:
            predictions = model(batch.float())
            running_loss += loss_fnc(input=predictions.squeeze(), target=labels.float())
            total_corr += total_correct(predictions, labels)
            
            evaluated_data += len(labels)
            total_batches += 1
    
    loss = running_loss / total_batches
    acc = float(total_corr) / evaluated_data
    return loss, acc

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
        for i, batch  in enumerate(train_loader):
            batch = data, labels
            
            """
            TODO: PyTorch things
            """

            # update training loss and accuracy statistics
            running_loss += loss
            running_acc += total_correct(predictions, labels)

            # updating counters
            evaluated_data += labels.size(0)
            total_batches += 1

        # updating training data loss and accuracy arrays
        avg_loss = running_loss / total_batches
        avg_acc = running_acc / evaluated_data
        """
        TODO: update training loss and accuracy datastructures
        """

        # evaluating validation data
        """
        TODO:   put model in evaluation mode, evaluate model (which is done in the function given below which is already written)
                and update validation loss and accuracy datastructures (don't forget to put the model back into training mode after)
        """
        loss, acc = evaluate(model, valid_loader, loss_fnc)

        # printing progress
        print(f"epoch: {e+1:4d}\tloss: {<train_loss_of_last_epoch>:.4f}\tacc: {<valid_acc_of_last_epoch>:.4f}")

    
    # Note replace the <variable_name> parts with whatever variable(s) you are using to store those values

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
        "seed": 2000,
        "lr": 0.1,
        "actfunction": "relu",
        "epochs": 50,
        "batch_size": 100,
        "num_hidden_layers": 1,
        "hidden_size": 64,
        "eval_every": 50,
        "optimizer": torch.optim.SGD,
        "loss_fnc": torch.nn.MSELoss(),
        "plot": True,
        "classification_type": "binary"
    }
    opts.update(args_dict)

    # load data
    label_name = "income"
    label_mapping = {'<=50K': 0, '>50K': 1}


    # You only have to call this once, then it'll save the preprocessed data into a new .csv file
    train_loader, valid_loader = load_data( "../data/adult.csv", label_name, label_mapping,
                                            preprocess=True, batch_size=opts.batch_size, seed=opts.seed)
    
    # Then you can just call it like this, which should make thing run faster
    #train_loader, valid_loader = load_data( "../data/adult_preprocessed.csv", label_name, label_mapping,
    #                                        preprocess=False, batch_size=opts.batch_size, seed=opts.seed)

    
    input_size = len(train_loader.dataset.data[0, :])
    output_size = 1

    # if you wanted it to be more general you could write this stuff, but for our purposes, the output size is always 1
    """
    output_size = 1 if train_loader.dataset.label_mapping is None else len(train_loader.dataset.label_mapping)
    output_size = 1 if output_size == 2 else output_size
    """

    # creating model
    model = MultiLayerPerceptron(input_size, output_size=output_size, **opts)

    # training model
    final_statistics = train(model, train_loader, valid_loader, opts)