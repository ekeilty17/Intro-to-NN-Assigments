import torch
import numpy as np
import matplotlib.pyplot as plt

from single_neural_classifier import SingleNeuronClassifier
from utils import *
from plot import *

def total_correct(predictions, labels):
    corr = (predictions > 0.5) == (labels == 1)
    return int(corr.sum())

def evaluate(model, loader, loss_fnc):
    total_corr = 0 
    evaluate_data = 0 
    total_batches = 0 
    running_loss = 0.0 
    with torch.no_grad():
        for data, labels in loader:
            predictions = model(data.float())
            running_loss += loss_fnc(input=predictions, target=labels.float()).detach().item()
            total_corr += total_correct(predictions, labels)
    
            evaluate_data += labels.size(0)
            total_batches += 1

    loss = running_loss / total_batches
    acc = float(total_corr) / evaluate_data
    return loss, acc

def train(model, train_loader, valid_loader, opts):
    # initializing model and parameters
    model.train()
    optimizer = opts.optimizer(model.parameters(), lr=opts.lr)
    loss_fnc = opts.loss_fnc

    # Initializing loss and accuracy arrays for plots
    Loss = { "train": [], "valid": [] }
    Acc = { "train": [], "valid": [] }

    evaluated_data = 0
    total_batches = 0
    running_loss = 0.0
    running_acc = 0.0

    # training loop
    for e in range(opts.epochs):
        # iterating over mini-batches
        for i, batch in enumerate(train_loader):
            data, labels = batch
            
            """ BEGIN: Important Part """
            # re-initializing optimizer
            optimizer.zero_grad()
            # feed forward
            predictions = model(data.float())
            # calculating losses
            loss = loss_fnc(input=predictions, target=labels.float())
            # backpropagation
            loss.backward()
            # updating weights
            optimizer.step()
            """ END: Important Part """

            # accumulated loss and accuracy
            running_loss += loss.detach().item()                        # .detach() creates a copy of the tensor that doesn't have a grad_fn attribute
                                                                        # .item() returns the value stored in the tensor
            running_acc += total_correct(predictions, labels)
            
            # updating counters
            total_batches += 1
            evaluated_data += labels.size(0)

        # Logging training data statistics
        Loss["train"].append( float(running_loss / total_batches) )     # this loss is basically an average of an average
                                                                        # we could techincally recalculate it completely, 
                                                                        # but that would obviously be computationally inefficient
                                                                        # so this is a good enough approximate
            
        Acc["train"].append( float(running_acc / evaluated_data)  )     # similar with the accuracy. It will be the average accuracy over 
                                                                        # maybe batches, which means the model has updated multiple times
                                                                        # between evaluations
        
        # evaluating model using validation data
        model.eval()
        loss, acc = evaluate(model, valid_loader, loss_fnc)
        model.train()
        
        Loss["valid"].append( loss )
        Acc["valid"].append( acc )
            
        print(f"epoch: {e+1:4d}\tbatch: {i+1:5d}\tloss: {Loss['train'][-1]:.4f}\tAcc: {acc:.4f}")

    # plots
    if opts.plot:
        display_statistics(Loss["train"], Acc["train"], Loss["valid"], Acc["valid"])
        plt.close()
        weights = model.fc[0].weight.data.numpy().flatten()
        dispKernel(weights, 3, 300)
    
    # return final statistic values
    model.eval()
    return Loss["train"][-1], Loss["valid"][-1], Acc["train"][-1], Acc["valid"][-1]

if __name__ == "__main__":

    # target we want to classify
    target = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1])

    # getting args
    opts = AttrDict()
    args_dict = {
        "seed": None,
        "lr": 0.001,
        "actfunction": "relu",
        "epochs": 1000,
        "batch_size": 100,
        "eval_every": 1,
        "optimizer": torch.optim.SGD,
        "loss_fnc": torch.nn.MSELoss(),
        "plot": True
    }
    opts.update(args_dict)

    if not opts.seed is None:
        torch.manual_seed(opts.seed)

    # load data
    train_loader, valid_loader = load_data(opts.batch_size)

    # creating model
    model = SingleNeuronClassifier(len(target), opts.actfunction, opts.lr)

    # training model
    final_statistics = train(model, train_loader, valid_loader, opts)
