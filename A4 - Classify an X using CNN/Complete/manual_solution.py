import numpy as np

class SingleLayerCNN(object):
    
    def __init__(self, target):
        self.target = target
        self.weights = np.vectorize(lambda x: -1 if x == 0 else x)(target)
        self.bias = 1 - np.sum(target)

    def __call__(self, input_grid):
        return self.forward(input_grid)

    @staticmethod
    def output_dimensions(input_shape, kernel_shape, stride=1, padding=0, dilation=1):
        h_in, w_in = input_shape
        h_out = (h_in + 2*padding - dilation * (kernel_shape[0] - 1) - 1) / stride + 1
        w_out = (w_in + 2*padding - dilation * (kernel_shape[1] - 1) - 1) / stride + 1
        return int(h_out), int(w_out)

    def convolution(self, input_grid, stride=1, padding=0):
        dilation = 1    # this operation is currently not supported
        
        kernel = np.flip(self.weights, axis=1)
        x, y = kernel.shape
        X, Y = input_grid.shape

        h, w = self.output_dimensions(input_grid.shape, kernel.shape, stride=1, padding=0, dilation=1)
        output = np.zeros((h, w))
        for i in range(X-x+1):
            for j in range(Y-y+1):
                output[i, j] = np.sum( input_grid[i:x+i, j:y+j] * kernel ) + self.bias
        return output

    @staticmethod
    def ReLU(Z):
        return np.maximum(0, Z)

    @staticmethod
    def hard_threshold(Z):
        return np.maximum(0, np.minimum(Z, 1))

    def forward(self, x):
        x = self.convolution(x)
        x = self.ReLU(x)
        
        # the sum is analogous to a linear layer with weights 1 and bias 0
        x = int(np.sum(x))

        # having a hard threshold is like having a sigmoid as the last activation function
        return self.hard_threshold(x)

class Tester(object):

    def __init__(self, target):
        self.target = target

    @staticmethod
    def get_test_examples():
        train_data = np.loadtxt(open("../data/traindata.csv", "r"), delimiter=",")
        train_labels = np.loadtxt(open("../data/trainlabels.csv", "r"), delimiter=",")
        valid_data = np.loadtxt(open("../data/validdata.csv", "r"), delimiter=",")
        valid_labels = np.loadtxt(open("../data/validlabels.csv", "r"), delimiter=",")

        n = int(np.sqrt(train_data.shape[1]))

        data = np.concatenate((train_data, valid_data), axis=0).reshape(-1, n, n)
        labels = np.concatenate((train_labels, valid_labels))

        return data, labels

    def isCorrect(self, input_list):
        return self.target == input_list
    
    def test(self, CNN):
        correct_weights = True
        data, labels = self.get_test_examples()
        for d, label in zip(data, labels):
            pred = CNN(d)
            if pred == label:
                # classified correctly
                pass
            else:
                print(f"{d}\nfailed: CNN predicted {pred} and should have been {label}")
                correct_weights = False
        
        if correct_weights:
            print("Single Layer CNN works in all cases\n")
            print(f"target pattern:\n{CNN.target}\n")
            print(f"weights:\n{CNN.weights}")
            print(f"bias = {CNN.bias}")

if __name__ == "__main__":
    target = np.array([
        [1, 0, 1],
        [0, 1, 0],
        [1, 0, 1]
    ])

    CNN = SingleLayerCNN(target)
    T = Tester(target)
    T.test(CNN)