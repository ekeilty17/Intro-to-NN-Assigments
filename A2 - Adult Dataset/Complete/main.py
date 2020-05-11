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
    corr = ( torch.argmax(predictions.squeeze(), dim=1) == labels )
    return int(corr.sum())

def total_correct_regression(predictions, labels):
    tolerance = 10
    error = predictions.squeeze() - labels
    corr = error < tolerance
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

def train(model, train_loader, valid_loader, opts):

    # initializing model
    model.train()
    optimizer = opts.optimizer(model.parameters(), lr=opts.lr)
    loss_fnc = opts.loss_fnc

    # normally you wouldn't do this, but just to make this example as general as possible
    # we can do different types of classification/regression and they all need different evaluation functions
    if opts.classification_type.lower() == "binary":
        total_correct = total_correct_binary
    elif opts.classification_type.lower() == "muli" or opts.classification_type.lower() == "multiclass":
        total_correct = total_correct_multiclass
    elif opts.classification_type.lower() == "regression":
        total_correct = total_correct_regression
    else:
        raise ValueError(f"opts.classification_type = '{opts.classification_type}' is not supported")

    # Initializing loss and accuracy arrays for plots
    Loss = { "train": [], "valid": [] }
    Acc = { "train": [], "valid": [] }

    evaluated_data = 0
    total_batches = 0
    running_loss = 0.0
    running_acc = 0.0

    """ Training Loop """
    for e in range(opts.epochs):
        # batching
        for i, batch in enumerate(train_loader):
            data, labels = batch

            # usual pytorch things
            optimizer.zero_grad()
            predictions = model(data.float())
            loss = loss_fnc(input=predictions.squeeze(), target=labels.float())
            loss.backward()
            optimizer.step()

            # accumulated loss and accuracy for the batch
            running_loss += loss
            running_acc += total_correct(predictions, labels)

            # updating counters
            evaluated_data += labels.size(0)
            total_batches += 1

            # evaluating accuracy
            if total_batches % opts.eval_every == 0:
                model.eval()

                # training data statistics
                Loss["train"].append( float(running_loss / opts.eval_every) )
                Acc["train"].append( float(running_acc / evaluated_data)  )

                # validation data statistics
                loss, acc = evaluate(model, valid_loader, loss_fnc, total_correct)
                Loss["valid"].append( float(loss) )
                Acc["valid"].append( float(acc) )

                evaluated_data = 0
                running_loss = 0.0
                running_acc = 0.0
                model.train()

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
    train_loader, valid_loader = load_data( "../data/adult.csv", label_name, label_mapping,
                                            preprocess=True, batch_size=opts.batch_size, seed=opts.seed)

    #train_loader, valid_loader = load_data( "../data/adult_preprocessed.csv", label_name, label_mapping,
    #                                        preprocess=False, batch_size=opts.batch_size, seed=opts.seed)

    # creating model
    input_size = len(train_loader.dataset.data[0, :])
    output_size = 1 if train_loader.dataset.label_mapping is None else len(train_loader.dataset.label_mapping)
    output_size = 1 if output_size == 2 else output_size

    model = MultiLayerPerceptron(input_size, output_size=output_size, **opts)

    # training model
    final_statistics = train(model, train_loader, valid_loader, opts)
