import torch
import numpy as np
import matplotlib.pyplot as plt

from mlp import MultiLayerPerceptron
from utils import *
from plot import *

def total_correct_binary(predictions, labels):
    corr = (predictions > 0.5).squeeze().long() == labels
    return int(corr.sum())

def total_correct_multiclass(predictions, labels):
    corr = ( torch.argmax(predictions, dim=1) == torch.argmax(labels, dim=1) )
    return int(corr.sum())

def total_correct_regression(predictions, labels):
    tolerance = 0.5
    error = predictions.squeeze() - labels.float()
    corr = error < tolerance
    return int(corr.sum())

def evaluate(model, loader, opts):
    
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
    return float(loss), float(acc)

def train(model, train_loader, valid_loader, opts):

    # initializing model
    model.train()
    optimizer = opts.optimizer(model.parameters(), lr=opts.lr)
    loss_fnc = opts.loss_fnc
    total_correct = opts.total_correct
    
    # Initializing loss and accuracy arrays for plots
    Loss = { "train": [], "valid": [] }
    Acc = { "train": [], "valid": [] }

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

            # usual pytorch things
            optimizer.zero_grad()
            predictions = model(data.float())

            if opts.classification_type in ["binary", "regression"]:
                predictions = predictions.squeeze()

            loss = loss_fnc(input=predictions.squeeze(), target=labels.float())
            loss.backward()
            optimizer.step()

            # update training loss and accuracy statistics
            running_loss += loss
            running_acc += total_correct(predictions, labels)

            # updating counters
            evaluated_data += labels.size(0)
            total_batches += 1

        # evaluating model
        model.eval()

        # training data statistics
        Loss["train"].append( float(running_loss / total_batches) )
        Acc["train"].append( float(running_acc / evaluated_data)  )

        # validation data statistics
        model.eval()
        loss, acc = evaluate(model, valid_loader, opts)
        model.train()

        Loss["valid"].append( loss )
        Acc["valid"].append( acc )
        
        print(f"epoch: {e+1:4d}\tbatch: {i+1:5d}\tloss: {Loss['train'][-1]:.4f}\tacc: {Acc['valid'][-1]:.4f}")

    # plots
    if opts.plot:
        display_statistics(Loss["train"], Acc["train"], Loss["valid"], Acc["valid"])
        plt.close()

    # return final statistic values
    model.eval()
    return Loss["train"][-1], Loss["valid"][-1], Acc["train"][-1], Acc["valid"][-1]

if __name__ == "__main__":

    # getting args
    opts = AttrDict()
    args_dict = {
        "seed": None,
        "lr": 0.01,
        "actfunction": "relu",              # 'linear', 'relu', 'sigmoid', 'tanh'
        "epochs": 50,
        "batch_size": 100,
        "num_hidden_layers": 1,
        "hidden_size": 64,
        "optimizer": torch.optim.SGD,
        "loss_fnc": torch.nn.MSELoss(),
        "plot": True
    }
    opts.update(args_dict)

    # Specify Type of data we want to classify
    #   Binary
    label_name = "income"
    label_mapping = {'<=50K': 0, '>50K': 1}

    """
    #   Multiclass
    label_name = "workclass"
    label_mapping = {   'Private': 0, 'Local-gov': 1, 'Self-emp-not-inc': 2, 'Federal-gov': 3, 
                        'Self-emp-inc': 4, 'State-gov': 5, 'Without-pay': 6}
    """

    """
    #   Regression
    label_name = "age"
    label_mapping = None
    """

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

    # based on label_mapping variable, we specify some useful variables in opts
    if label_mapping is None:                           # regression
        opts.classification_type = "regression"
        opts.total_correct = total_correct_regression
        opts.output_size = 1
    elif len(label_mapping) == 2:                       # binary
        opts.classification_type = "binary"
        opts.total_correct = total_correct_binary
        opts.output_size = 1
    elif len(label_mapping) > 2:                        # multiclass
        opts.classification_type = "mulitclass"
        opts.total_correct = total_correct_multiclass
        opts.output_size = len(label_mapping)
    else:
        ValueError("'label_mapping' needs to have more than 1 attribute")

    # creating model
    input_size = len(train_loader.dataset.data[0, :])
    model = MultiLayerPerceptron(input_size, **opts)

    # training model
    final_statistics = train(model, train_loader, valid_loader, opts)
