import numpy as np
import matplotlib.pyplot as plt

from cbow import CBOW
from bengio import Bengio
from utils import *
from plot import *

def total_correct(predictions, targets):    
    corr = ( torch.argmax(predictions, dim=1) == targets )
    return int(corr.sum())

def evaluate(model, loader, loss_fnc, total_correct):
    total_corr = 0
    evaluate_data = 0
    total_batches = 0
    running_loss = 0.0
    with torch.no_grad():
        for contexts, targets in loader:
            predictions = model(contexts)
            running_loss += loss_fnc(input=predictions, target=targets).detach().item()
            total_corr += total_correct(predictions, targets)

            evaluate_data += targets.size(0)
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

    for e in range(opts.epochs):
        # batching
        for i, batch in enumerate(train_loader):
            contexts, targets = batch

            # usual pytorch things
            optimizer.zero_grad()
            predictions = model(contexts)
            loss = loss_fnc(input=predictions, target=targets)
            loss.backward()
            optimizer.step()

            # accumulated loss and accuracy for the batch
            running_loss += loss.detach().item()
            running_acc += total_correct(predictions, targets)

            # updating counters
            evaluated_data += targets.size(0)
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


if __name__ == "__main__":

    # getting args
    opts = AttrDict()
    args_dict = {
        "seed": None,
        "lr": 0.01,
        "epochs": 1,
        "batch_size": 1000,
        "eval_every": 100,
        "optimizer": torch.optim.AdamW,
        "loss_fnc": torch.nn.CrossEntropyLoss(),
        "context_length": (3, 0),
        "hidden_size": 500,
        "embedding_size": 100,
        "plot": False,
        "save_embeddings": True,
    }
    opts.update(args_dict)

    # random seed for initializing weights
    if not opts.seed is None:
        torch.manual_seed(opts.seed)

    # getting data
    vocab, train_loader, valid_loader = load_data(  batch_size=opts.batch_size, seed=opts.seed,
                                                    preprocess=True, context_length=opts.context_length)
    opts.vocab_size = len(vocab)

    # creating model
    model = CBOW(opts.context_length, opts.vocab_size)
    #model = Bengio(opts.context_length, opts.vocab_size, opts.embedding_size, opts.hidden_size)

    # training model
    final_statistics = train(model, train_loader, valid_loader, opts)

    # extracting word embeddings
    if opts.save_embeddings:
        embeddings = model.embedding.weight.data.numpy().T
        np.savetxt(f"{model.name.lower()}_word_vectors.csv", embeddings)