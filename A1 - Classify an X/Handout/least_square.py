import numpy as np

class LeastSquares(object):

    def __init__(self):
        self.w_aug = None       # w_aug = [bias, w_1, ..., w_d]
    
    def get_parameters(self):
        if self.w_aug is None:
            return None, None
        # bais, weights
        return self.w_aug[0], self.w_aug[1:]

    @staticmethod
    def MSE(predictions, labels):
        return np.mean( (predictions - labels)**2 )

    @staticmethod
    def isCorrect(pred, label):
        return (pred > 0.5 and label == 1) or (pred < 0.5 and label == 0)

    def predict(self, data):
        # Note: data is an un-augmented data matrix
        #   You should first augment the data with a column of 1's
        #   and then matrix multipy with self.w_aug
        # Note: A @ B is matrix multiplication is numpy
        # Hint: np.insert() may be helpful
        raise NotImplementedError

    def evaluate(self, data, labels):
        
        predictions = self.predict(data)
        loss = self.MSE(predictions, labels)
        acc = 0
        for pred, label in zip(predictions, labels):
            if self.isCorrect(pred, label):
                acc += 1
        acc /= len(predictions)

        return loss, acc
    
    def train(self, train_data, train_labels, valid_data, valid_labels):
        raise NotImplementedError

        # 1) augment training data
        # 2) calculate self.w_aug using least squares formula
        # 3) evaluate training data to obtain train_loss, train_acc using self.evalulate()
        # 4) evaluate validation data to obtain valid_loss, valid_acc using self.evalulate()
        # 5) return loss and accuracy values
        return train_loss, train_acc, valid_loss, valid_acc