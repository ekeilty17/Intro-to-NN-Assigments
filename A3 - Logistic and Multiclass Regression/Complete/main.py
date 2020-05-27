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
    corr = (predictions > 0.5).squeeze().long() == labels.squeeze().long()       #.long() converts from a float in an integer
    return int(corr.sum())

def total_correct_multiclass(predictions, labels):
    predictions = torch.nn.Softmax(dim=1)(predictions)
    #print(predictions)
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
            running_loss += loss_fnc(input=predictions, target=labels).detach().item()
            total_corr += total_correct(predictions, labels)
    
            evaluate_data += labels.size(0)
            total_batches += 1

    loss = running_loss / total_batches
    acc = total_corr / evaluate_data
    return loss, acc

def train(model, train_loader, valid_loader, opts):

    # initializing model and parameters
    model.train()
    optimizer = opts.optimizer(model.parameters(), lr=opts.lr)
    loss_fnc = opts.loss_fnc
    total_correct = total_correct_binary if opts.num_classes == 1 else total_correct_multiclass

    # Initializing loss and accuracy arrays for plots
    Loss = { "train": [], "valid": [] }
    Acc = { "train": [], "valid": [] }

    # training loop
    for e in range(opts.epochs):
        
        # variables to help keep track of training loss and accuracy
        evaluated_data = 0
        total_batches = 0
        running_loss = 0.0
        running_acc = 0.0
        
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
            running_loss += loss.detach().item()
            running_acc += total_correct(predictions, labels)
            
            # updating counters
            evaluated_data += labels.size(0)
            total_batches += 1

        # Logging training data statistics
        Loss["train"].append( float(running_loss / total_batches) )
        Acc["train"].append( float(running_acc / evaluated_data)  )
        
        # evaluating model logging validation data
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
        weights = model.fc[0].weight.data.numpy()
        display_kernels(weights, 3, 300)
    
    # return final statistic values
    model.eval()
    return Loss["train"][-1], Loss["valid"][-1], Acc["train"][-1], Acc["valid"][-1]

if __name__ == "__main__":

    # target we want to classify
    target = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1])

    # TODO: play with this value
    #       try 1
    #       try 2
    #       try 10
    num_classes = 10         # 1 --> 'binary' or  >1 --> 'multiclass'
    
    # specifying loss function based on type of classification
    loss_fnc = torch.nn.BCEWithLogitsLoss() if num_classes == 1 else torch.nn.CrossEntropyLoss()

    # getting args
    opts = AttrDict()
    args_dict = {
        "seed": None,
        "lr": 0.1,
        "epochs": 100,
        "batch_size": None,
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
    train_loader, valid_loader = load_data(opts.batch_size)

    # creating model
    model = LogisticRegression(len(target)) if opts.num_classes == 1 else MultiClassRegression(len(target), opts.num_classes)

    # training model
    final_statistics = train(model, train_loader, valid_loader, opts)
