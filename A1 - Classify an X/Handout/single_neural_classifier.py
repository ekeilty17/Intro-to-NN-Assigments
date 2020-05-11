import numpy as np

class SingleNeuronClassifier(object):

    def __init__(self, num_inputs, actfunction="relu", lr=0.1):
        raise NotImplementedError

    """ Initialization """
    # You might want dedicate functions for initializing your model, so if you do put them here

    """ Activation Functions """
    # Hint: Also return the derivatives along with the output of the function as you will need it
    @staticmethod
    def Linear(z):
        raise NotImplementedError
    
    @staticmethod
    def ReLU(z):
        raise NotImplementedError
    
    @staticmethod
    def Sigmoid(z):
        raise NotImplementedError

    """ Helper Functions """
    # You might need helper functions so if you do you can put them here

    """ ML Algorithms """
    # Note: You don't necessaryily need to use these arguments, feel free to do as you wish
    def forward_propogate(self, I):
        raise NotImplementedError

    def backward_propogate(self, data, label):
        raise NotImplementedError