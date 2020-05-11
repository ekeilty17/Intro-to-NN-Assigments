import numpy as np
import matplotlib.pyplot as plt

from single_neural_classifier import SingleNeuronClassifier
from least_square import LeastSquares
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

# Training Loop
def train(model, train_data, train_labels, valid_data, valid_labels, opts):

    # initializing model
    model.init_parameters()

    # Initializing error and accuracy arrays for plots
    Loss = { "train": np.zeros(opts.epochs), "valid": np.zeros(opts.epochs) }
    Acc = { "train": np.zeros(opts.epochs), "valid": np.zeros(opts.epochs) }

    for e in range(opts.epochs):
        #print("epoch", e+1)

        # training the model using the  training data
        predictions = []
        model.zero_grad()
        for data, labels in zip(train_data, train_labels):
            prediction = model(data)
            predictions.append( prediction )
            model.backward(data, labels)
        model.step()

        # evaluating training data
        Loss["train"][e] = MSE(predictions, train_labels)
        Acc["train"][e] = evaluate(predictions, train_labels)

        # evaluating the model using the validation data
        predictions = []
        for data in valid_data:
            prediction = model(data)
            predictions.append( prediction )
        Loss["valid"][e] = MSE(predictions, valid_labels)
        Acc["valid"][e] = evaluate(predictions, valid_labels)

    if opts.plot:
        display_statistics(Loss["train"], Acc["train"], Loss["valid"], Acc["valid"])
        plt.close()
        dispKernel(model.w, 3, 300)

    # return final values
    return Loss["train"][-1], Loss["valid"][-1], Acc["train"][-1], Acc["valid"][-1]


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
        "plot": True
    }
    opts.update(args_dict)

    # creating model
    model = SingleNeuronClassifier(len(target), opts.actfunction, opts.lr)

    # training model
    final_statistics = train(model, train_data, train_labels, valid_data, valid_labels, opts)
    b, w = model.get_parameters()
    print()
    print(b)
    print(w)


    """ Least Squares Solution """
    print("\nLeast Square Solution")
    LS_model = LeastSquares()
    final_statistics = LS_model.train(train_data, train_labels, valid_data, valid_labels)
    final_train_loss, final_train_acc, final_valid_loss, final_valid_acc = final_statistics
    b_LS, w_LS = LS_model.get_parameters()

    print(f"Training loss: {final_train_loss:.4f}{'':.20s}\t\tTraining acc: {final_train_acc * 100:.2f}%")
    print(f"Validation loss: {final_valid_loss:.4f}{'':.20s}\t\tValidation acc: {final_valid_acc * 100:.2f}%")
    print()
    print(b_LS)
    print(w_LS)

    dispKernel(w_LS, 3, 300)
