import numpy as np
import matplotlib.pyplot as plt

from rnn_cell import RNNCell
from utils import *
from plot import *

def total_correct(predictions, labels):    
    #corr = ( torch.argmax(predictions, dim=1) == torch.argmax(labels, dim=1) )
    corr = ( predictions.long() == labels )
    return int(corr.sum())

def evaluate(model, loader, loss_fnc, total_correct):
    total_corr = 0
    evaluate_data = 0
    total_batches = 0
    running_loss = 0.0
    with torch.no_grad():
        for data, labels in loader:
            seq = data.permute(1, 0, 2)
            hidden = model.init_hidden(seq.size(1))
            
            # running through RNN
            for x in seq:
                hidden = model(x.float(), hidden)
            predictions = hidden.squeeze()

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
    optimizer = opts.optimizer(model.parameters(), lr=opts.lr)
    loss_fnc = opts.loss_fnc

    # Initializing loss and accuracy arrays for plots
    Loss = { "train": [], "valid": [] }
    Acc = { "train": [], "valid": [] }

    evaluated_data = 0
    total_batches = 0
    running_loss = 0.0
    running_acc = 0.0
    
    """
    model.rnn_cell.weight_ih.data = torch.tensor([[1.0]])
    model.rnn_cell.weight_hh.data = torch.tensor([[1.0]])
    model.rnn_cell.bias_ih.data = torch.tensor([0.0])
    model.rnn_cell.bias_hh.data = torch.tensor([0.0])
    """

    for e in range(opts.epochs):
        # batching
        for i, batch in enumerate(train_loader):
            data, labels = batch

            # reshaping data
            seq = data.permute(1, 0, 2)                 # (seq_len, batch_size, input_size)
                                                        # input_size is our onehot encoded vectors
                                                        # in which case the input size is the size of the vector
            hidden = model.init_hidden(seq.size(1))     # (num_layer, batch_size, hidden_size)

            # usual pytorch things
            optimizer.zero_grad()
            
            # running through RNN
            for x in seq:
                hidden = model(x.float(), hidden)
            predictions = hidden.squeeze()

            loss = loss_fnc(input=predictions, target=labels.float())
            loss.backward()

            # prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

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
                loss, acc = evaluate(model, valid_loader, loss_fnc, total_correct)
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
        "lr": 0.001,
        "epochs": 100,
        "batch_size": 32,
        "eval_every": 100,
        "optimizer": torch.optim.AdamW,
        "loss_fnc": torch.nn.MSELoss(),
        "seq_len": 10,
        "input_size": 1,       # represents the dimension of our one-hot encoded vector
        "hidden_size": 1,
        "teacher_forcing": True,
        "plot": True
    }
    opts.update(args_dict)

    # random seed for initializing weights
    if not opts.seed is None:
        torch.manual_seed(opts.seed)

    # getting data
    train_loader, valid_loader = load_data(seq_len=opts.seq_len, high=100, batch_size=opts.batch_size, seed=opts.seed)

    # creating model
    model = RNNCell(opts.input_size, opts.hidden_size)

    # training model
    final_statistics = train(model, train_loader, valid_loader, opts)

    #print(test(model, [1, 2, 3, 4]))