import numpy as np

class LeastSquares(object):

    def __init__(self):
        pass
    
    def get_parameters(self):
        # bais, weights
        return self.w_aug[0], self.w_aug[1:]

    def predict(self, data):
        X = np.insert(data, 0, 1, axis=1)
        return X @ self.w_aug

    def train(self, train_data, train_labels, valid_data, valid_labels):
        # augment data
        X = np.insert(train_data, 0, 1, axis=1)
        y = train_labels

        self.w_aug = np.linalg.inv(X.T @ X) @ X.T @ y

        train_loss, train_acc = self.evaluate(train_data, train_labels)
        valid_loss, valid_acc = self.evaluate(valid_data, valid_labels)

        return train_loss, train_acc, valid_loss, valid_acc

    @staticmethod
    def MSE(predictions, labels):
        return np.mean( (predictions - labels)**2 )

    @staticmethod
    def isCorrect(pred, label):
        return (pred > 0.5 and label == 1) or (pred < 0.5 and label == 0)

    def evaluate(self, data, labels):
        
        predictions = self.predict(data)
        loss = self.MSE(predictions, labels)
        acc = 0
        for pred, label in zip(predictions, labels):
            if self.isCorrect(pred, label):
                acc += 1
        acc /= len(predictions)

        return loss, acc