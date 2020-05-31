import numpy as np
import matplotlib.pyplot as plt

from rnn import RNN
from utils import *
from plot import *

def total_correct(predictions, labels):
    corr = ( predictions.long() == labels )
    return int(corr.sum())

def evaluate(model, loader, loss_fnc):
    total_corr = 0
    evaluate_data = 0
    total_batches = 0
    running_loss = 0.0
    with torch.no_grad():
        for data, labels in loader:
            seqs = data.permute(1, 0, 2)
            hidden = model.init_hidden(seqs.size(1))
            
            predictions = model(seqs).squeeze()
            running_loss += loss_fnc(input=predictions, target=labels.float()).detach().item()
            total_corr += total_correct(predictions, labels)

            evaluate_data += labels.size(0)
            total_batches += 1

    loss = running_loss / total_batches
    acc = float(total_corr) / evaluate_data
    return loss, acc

def train(model, train_loader, valid_loader, opts):
    
    # initializing model
    model.train()
    optimizer = opts.optimizer(model.parameters(), lr=opts.lr, weight_decay=1e-3)
    loss_fnc = opts.loss_fnc

    # Initializing loss and accuracy arrays for plots
    Loss = { "train": [], "valid": [] }
    Acc = { "train": [], "valid": [] }

    evaluated_data = 0
    total_batches = 0
    running_loss = 0.0
    running_acc = 0.0

    for e in range(opts.epochs):
        # batching
        for i, batch in enumerate(train_loader):
            data, labels = batch

            # reshaping data
            seqs = data.permute(1, 0, 2)                # (seq_len, batch_size, input_size)
                                                        # input_size is our onehot encoded vectors
                                                        # in which case the input size is the size of the vector
            hidden = model.init_hidden(seqs.size(1))    # (num_layer, batch_size, hidden_size)

            # usual pytorch things
            optimizer.zero_grad()
            predictions = model(seqs).squeeze()
            loss = loss_fnc(input=predictions, target=labels.float())
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
                
                Loss["valid"].append( loss )
                Acc["valid"].append( acc )

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

def test(model, seq):
    seq = torch.tensor(seq).reshape(-1, 1, 1)
    hidden = model.init_hidden(1)
    # running through RNN
    for x in seq:
        hidden = model(x.float(), hidden)
    prediction = hidden.squeeze()
    return prediction

if __name__ == "__main__":

    # getting args
    opts = AttrDict()
    args_dict = {
        "seed": None,
        "lr": 1e-5,
        "epochs": 100,
        "batch_size": 32,
        "eval_every": 100,
        "optimizer": torch.optim.AdamW,
        "loss_fnc": torch.nn.MSELoss(),
        "seq_len": 3,                       # number of integers we are adding
        "hidden_size": 100,
        "high": 10,                         # integers range from [0, high)
        "onehot": False,
        "plot": True
    }
    opts.update(args_dict)

    # input_size is the size of the vector of each element in the sequence
    # if we don't onehot encode the inputs, then it is 1 since we are just inputting the raw numbers
    # if we do onehot encode, then it will be the size of the onehot encoded vector, i.e. opts.high
    opts.input_size = opts.high if opts.onehot else 1

    # random seed for initializing weights
    if not opts.seed is None:
        np.random.seed(self.seed)
        torch.manual_seed(opts.seed)

    # getting data
    train_loader, valid_loader = load_data(seq_len=opts.seq_len, high=opts.high, batch_size=opts.batch_size, onehot=opts.onehot)

    # creating model
    model = RNN(opts.input_size, opts.hidden_size, 1)

    # training model
    final_statistics = train(model, train_loader, valid_loader, opts)