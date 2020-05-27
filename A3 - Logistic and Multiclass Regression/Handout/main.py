import torch
import numpy as np
import matplotlib.pyplot as plt

from logistic_regression import LogisticRegression
from multiclass_regression import MultiClassRegression
from utils import *
from plot import *

def total_correct_binary(predictions, labels):
    predictions = torch.nn.Sigmoid()(predictions)
    #print(predictions)

    """ 
    TODO: Count the number of predictions that match their corresponding label
        params: torch.tensor, torch.tensor
        return: int
    """

    # Note: predictions.size() = torch.Size([opts.batch_size, 1])
    #       labels.size() = torch.Size([opts.batch_size, 1])
    
    raise NotImplementedError

def total_correct_multiclass(predictions, labels):
    predictions = torch.nn.Softmax(dim=1)(predictions)
    #print(predictions)

    """ 
    TODO: Count the number of predictions that match their corresponding label
        params: torch.tensor, torch.tensor
        return: int
    """

    # Note: predictions.size() = torch.Size([opts.batch_size, 1])
    #       labels.size() = torch.Size([opts.batch_size])

    raise NotImplementedError

def evaluate(model, loader, opts):
    
    loss_fnc = opts.loss_fnc
    total_correct = total_correct_binary if opts.num_classes == 1 else total_correct_multiclass
    
    total_corr = 0 
    evaluate_data = 0 
    total_batches = 0 
    running_loss = 0.0
    
    with torch.no_grad():
        for batch, labels in loader:
            
            # different loss functions need labels to be reformatted differently
            labels = labels.reshape(-1, 1).float() if opts.num_classes == 1 else labels.long()
            
            predictions = model(batch.float())
            running_loss += loss_fnc(input=predictions, target=labels)
            total_corr += total_correct(predictions, labels)
    
            evaluate_data += labels.size(0)
            total_batches += 1

    loss = running_loss / total_batches
    acc = total_corr / evaluate_data
    return float(loss), float(acc)


def train(model, train_loader, valid_loader, opts):
    if not opts.seed is None:
        torch.manual_seed(opts.seed)

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

        # variables to help keep track of training loss and accuracy
        evaluated_data = 0
        total_batches = 0
        running_loss = 0.0
        running_acc = 0.0

        # batching
        for i, batch  in enumerate(train_loader):
            batch = data, labels
            
            # different loss functions need labels to be reformatted differently
            labels = labels.reshape(-1, 1).float() if opts.num_classes == 1 else labels.long()

            """
            TODO: PyTorch things
            """

            # update training loss and accuracy statistics
            running_loss += loss
            
            if opts.num_classes == 1:
                running_acc += total_correct_binary(predictions, labels)
            else:
                running_acc += total_correct_multiclass(predictions, labels)

            # updating counters
            evaluated_data += labels.size(0)
            total_batches += 1

        # evaluating model
        """
        TODO: Put model in evaluation mode
        """

        # updating training data loss and accuracy arrays
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

        # printing progress
        print(f"epoch: {e+1:4d}\tloss: {<train_loss_of_last_epoch>:.4f}\tacc: {<valid_acc_of_last_epoch>:.4f}")
        """
        TODO: put model back into training mode
        """
    
    # Note replace the <variable_name> parts with whatever variable(s) you are using to store those values

    # plots
    if opts.plot:
        display_statistics(<train_loss_list>, <train_acc_list>, <valid_loss_list>, <valid_acc_list>)
        plt.close()

        #weights = model.<name of fully connected layer in __init__>.weight.data.numpy()
        #display_kernels(weights, 3, 300)
    
    # return final statistic values
    model.eval()
    return <final_train_loss>, <final_train_acc>, <final_valid_loss>, <final_valid_acc>


if __name__ == "__main__":

    # target we want to classify
    target = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1])

    # TODO: play with this value
    #       try 1
    #       try 2
    #       try 10
    num_classes = 1         # 1 --> 'binary' or  >1 --> 'multiclass'
    
    # specifying loss function based on type of classification
    loss_fnc = torch.nn.BCEWithLogitsLoss() if num_classes == 1 else torch.nn.CrossEntropyLoss()

    # getting args
    opts = AttrDict()
    args_dict = {
        "seed": None,
        "lr": 0.1,
        "epochs": 100,
        "batch_size": 5,
        "optimizer": torch.optim.SGD,
        "loss_fnc": loss_fnc,
        "plot": True,
        "num_classes": num_classes
    }
    opts.update(args_dict)

    # random seed
    if not opts.seed is None:
        torch.manual_seed(opts.seed)

    # load data
    train_loader, valid_loader = load_data(opts.batch_size, opts.seed)

    # creating model
    model = LogisticRegression(len(target)) if opts.num_classes == 1 else MultiClassRegression(len(target), opts.num_classes)

    # training model
    final_statistics = train(model, train_loader, valid_loader, opts)
