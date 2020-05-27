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
    total_corr = 0
    evaluate_data = 0
    total_batches = 0
    running_loss = 0.0
    with torch.no_grad():
        for data, labels in loader:
            predictions = model(data.float())
            running_loss += loss_fnc(input=predictions.squeeze(), target=labels.float()).detach().item()
            total_corr += total_correct(predictions, labels)
            
            evaluate_data += labels.size(0)
            total_batches += 1
    
    loss = running_loss / total_batches
    acc = float(total_corr) / evaluate_data
    return loss, acc

# Training Loop
def train(model, train_loader, valid_loader, opts):

    # initializing model
    model.train()
    optimizer = opts.optimizer(model.parameters(), lr=opts.lr)
    loss_fnc = opts.loss_fnc

    # Initializing loss and accuracy arrays for plots
    Loss = { "train": [], "valid": [] }
    Acc = { "train": [], "valid": [] }

    """ Training Loop """
    for e in range(opts.epochs):

        # variables to help keep track of training loss and accuracy
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
            loss = loss_fnc(input=predictions.squeeze(), target=labels.float())
            loss.backward()
            optimizer.step()

            # accumulated loss and accuracy for the batch
            running_loss += loss.detach().item()
            running_acc += total_correct(predictions, labels)

            # updating counters
            evaluated_data += labels.size(0)
            total_batches += 1

        # training data statistics
        Loss["train"].append( float(running_loss / total_batches) )
        Acc["train"].append( float(running_acc / evaluated_data)  )

        # validation data statistics
        model.eval()
        loss, acc = evaluate(model, valid_loader, loss_fnc, total_correct)
        model.train()
        
        Loss["valid"].append( loss )
        Acc["valid"].append( acc )

        evaluated_data = 0
        running_loss = 0.0
        running_acc = 0.0
        model.train()

        print(f"epoch: {e+1:4d}\tbatch: {i+1:5d}\tloss: {Loss['train'][-1]:.4f}\tacc: {Acc['valid'][-1]:.4f}")

    # plots
    if opts.plot:
        display_statistics(Loss["train"], Acc["train"], Loss["valid"], Acc["valid"])
        plt.close()
        
        if opts.model_type == "CNN":
            kernels = [W.data.numpy().flatten() for W in model.conv[0].weight]
            display_kernels(kernels, 3, 300)
        else:
            layers = np.array([fc[0].weight.data.numpy() for fc in model.Hidden])
            for kernels in layers:
                display_kernels(kernels, 5, 250)
                plt.close()

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
        "batch_size": 50,
        "optimizer": torch.optim.Adam,      # doesn't work if you use torch.optim.SGD
        "loss_fnc": torch.nn.BCEWithLogitsLoss(),
        "plot": True,
        "model_type": "CNN"     # 'CNN' or 'MLP'
    }
    opts.update(args_dict)

    # error checking
    if not opts.model_type in ["CNN", "MLP"]:
        raise ValueError(f"{opts.model_type} architecture not supported")
    
    # random seed
    if not opts.seed is None:
        torch.manual_seed(opts.seed)

    # getting data
    train_loader, valid_loader = load_data(batch_size=opts.batch_size)

    # creating model
    CNN = SingleLayerCNN(kernel_size=target.shape, num_kernels=1, output_size=1)
    MLP = MLP(input_size=(5, 5), output_size=1, num_hidden_layers=1, hidden_size=9)
    model = CNN if opts.model_type == "CNN" else MLP


    # training model
    final_statistics = train(model, train_loader, valid_loader, opts)

    
