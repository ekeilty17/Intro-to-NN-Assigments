import numpy as np
import matplotlib.pyplot as plt

from cnn import CNN
from mlp import MLP
from utils import *
from plot import *

def total_correct(predictions, labels):
    corr = ( torch.argmax(predictions, dim=1) == labels )
    return int(corr.sum())

def evaluate(model, loader, loss_fnc):
    total_corr = 0
    evaluate_data = 0
    total_batches = 0
    running_loss = 0.0
    with torch.no_grad():
        for images, labels in loader:
            predictions = model(images.float())
            running_loss += loss_fnc(input=predictions, target=labels).detach().item()
            total_corr += total_correct(predictions, labels)
            
            evaluate_data += labels.size(0)
            total_batches += 1
    
    loss = running_loss / total_batches
    acc = float(total_corr) / evaluate_data
    return float(loss), float(acc)

# Training Loop
def train(model, train_loader, valid_loader, opts):
    
    # initializing model
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

    """ Training Loop """
    for e in range(opts.epochs):
        # batching
        for i, batch in enumerate(train_loader):
            images, labels = batch

            # usual pytorch things
            optimizer.zero_grad()
            predictions = model(images.float())
            loss = loss_fnc(input=predictions, target=labels)
            loss.backward()
            optimizer.step()

            # accumulated loss and accuracy for the batch
            running_loss += loss.detach().item()
            running_acc += total_correct(predictions, labels)

            # updating counters
            evaluated_data += labels.size(0)
            total_batches += 1

            # evaluating accuracy
            if total_batches % opts.eval_every == 0:
                

                # training data statistics
                Loss["train"].append( float(running_loss / opts.eval_every) )
                Acc["train"].append( float(running_acc / evaluated_data)  )

                # validation data statistics
                model.eval()
                loss, acc = evaluate(model, valid_loader, loss_fnc)
                model.train()
                
                Loss["valid"].append( float(loss) )
                Acc["valid"].append( float(acc) )

                # resetting values
                evaluated_data = 0
                running_loss = 0.0
                running_acc = 0.0

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
        "epochs": 2,
        "batch_size": 100,
        "eval_every": 100,
        "optimizer": torch.optim.Adam,
        "loss_fnc": torch.nn.CrossEntropyLoss(),
        "plot": True,
        "save": False,
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

    # saving model
    if opts.save:
        torch.save(model, f"{opts.model_type.lower()}.pt")
        print(f"model saved as {opts.model_type.lower()}.pt")
