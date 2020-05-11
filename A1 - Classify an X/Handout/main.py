import numpy as np
import matplotlib.pyplot as plt

from single_neural_classifier import SingleNeuronClassifier
from utils import *
from plot import *

# Loss
def MSE(predictions, labels):
    return np.mean( (predictions - labels)**2 )

# Accuracy
def isCorrect(pred, label):
    return (pred > 0.5 and label == 1) or (pred < 0.5 and label == 0)

def evaluate(predictions, labels):
    accuracy = 0
    for pred, label in zip(predictions, labels):
        if isCorrect(pred, label):
            accuracy += 1
    return accuracy / len(predictions)

# TODO
def train(model, traindata, trainlabels, validdata, validlabels, opts):
    # training Loop
    raise NotImplementedError

    if opts.plot:
        # feel free to re-name these whatever you want, this is just how you call the function
        # where train_loss is a numpy array of the training losses for each epoch
        display_statistics(train_loss, train_acc, valid_loss, valid_acc)

        plt.close()

        # model.W = weights of the model not including the bias
        # 3 and 300 are just scaling numbers for the plot don't worry about them
        dispKernel(model.W, 3, 300)

if __name__ == "__main__":
    # target we want to classify
    target = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1])

    # getting data
    train_data, train_labels, valid_data, valid_labels = load_data()

    # getting args
    opts = AttrDict()
    args_dict = {
        "seed": None,
        "lr": 0.1,
        "actfunction": "Sigmoid",
        "epochs": 500,
        "plot": True,
        "SGD": False
    }
    opts.update(args_dict)

    # creating model
    model = SingleNeuronClassifier(len(target), opts.actfunction, opts.lr)

    # training model
    final_statistics = train(model, train_data, train_labels, valid_data, valid_labels, opts)
