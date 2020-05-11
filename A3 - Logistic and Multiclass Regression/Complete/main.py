import torch
import numpy as np
import matplotlib.pyplot as plt

from logistic_regression import LogisticRegression
from multiclass_regression import MultiClassRegression
from utils import *
from plot import *

def total_correct_binary(predictions, labels):
    corr = (predictions > 0.5).squeeze().long() == labels.squeeze().long()       #.long() converts from a float in an integer
    return int(corr.sum())

def total_correct_multiclass(predictions, labels):
    corr = torch.argmax(predictions.squeeze(), dim=1) == labels.long()
    return int(corr.sum())

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

    # initializing model and parameters
    model.train()
    optimizer = opts.optimizer(model.parameters(), lr=opts.lr)
    loss_fnc = opts.loss_fnc
    total_correct = total_correct_binary if opts.num_classes == 1 else total_correct_multiclass

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

            # different loss functions need labels to be reformatted differently
            labels = labels.reshape(-1, 1).float() if opts.num_classes == 1 else labels.long()

            # gradient descent update
            optimizer.zero_grad()
            predictions = model(data.float())
            loss = loss_fnc(input=predictions, target=labels)
            loss.backward()
            optimizer.step()

            # accumulated loss and accuracy
            running_loss += loss
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
        loss, acc = evaluate(model, valid_loader, opts)
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
        "lr": 0.1,
        "actfunction": "relu",
        "epochs": 1000,
        "batch_size": None,
        "optimizer": torch.optim.SGD,
        "loss_fnc": None,        # updated after based on value of num_classes
        "plot": True,
        "num_classes": 1         # 1 --> 'binary' or  >1 --> 'multiclass'
    }
    args_dict["loss_fnc"] = torch.nn.BCEWithLogitsLoss() if args_dict["num_classes"] == 1 else torch.nn.CrossEntropyLoss()
    opts.update(args_dict)

    # load data
    train_loader, valid_loader = load_data(opts.batch_size)

    # creating model
    model = LogisticRegression(len(target)) if opts.num_classes == 1 else MultiClassRegression(len(target), opts.num_classes)

    # training model
    final_statistics = train(model, train_loader, valid_loader, opts)
