import numpy as np
import matplotlib.pyplot as plt

from single_layer_cnn import SingleLayerCNN
from mlp import MLP
from utils import *
from plot import *

def total_correct(predictions, labels):
    corr = (predictions > 0.5).squeeze().long() == labels.long()
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

# Training Loop
def train(model, train_loader, valid_loader, opts):

    # random seed for initializing weights
    if not opts.seed is None:
        torch.manual_seed(seed)

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
        kernels = [W.data.numpy().flatten() for W in model.conv[0].weight]
        display_kernels(kernels, 3, 300)
        #kernels = [fc[0].weight.data.numpy().flatten() for fc in model.Hidden]
        #display_kernel(kernels[0], 5, 250)

    # return final statistic values
    model.eval()
    return Loss["train"][-1], Loss["valid"][-1], Acc["train"][-1], Acc["valid"][-1]


if __name__ == "__main__":
    # we want to classify any array containing this target kernel
    target = np.array([
                        [1, 0, 1], 
                        [0, 1, 0], 
                        [1, 0, 1]
                    ])

    # getting args
    opts = AttrDict()
    args_dict = {
        "seed": None,
        "lr": 0.01,
        "epochs": 10,
        "batch_size": 100,
        "eval_every": 10,
        "optimizer": torch.optim.Adam,      # doesn't work if you use torch.optim.SGD
        "loss_fnc": torch.nn.BCEWithLogitsLoss(),
        "plot": True
    }
    opts.update(args_dict)

    # getting data
    train_loader, valid_loader = load_data(batch_size=opts.batch_size, seed=opts.seed)

    # creating model
    model = SingleLayerCNN(target.shape, 1)
    #model = MLP(25)

    # training model
    final_statistics = train(model, train_loader, valid_loader, opts)

    
