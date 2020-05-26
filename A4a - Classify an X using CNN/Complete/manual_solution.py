import numpy as np

class SingleLayerCNN(object):
    
    name = "Single Layer CNN"

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


class MLP(object):

    name = "MLP"

    def __init__(self, input_size, target):
        self.input_size = input_size
        self.target = target
        self.hidden_size = self.calculate_hidden_size()

        self.create_parameters()

    def __call__(self, input_grid):
        return self.forward(input_grid)

    def calculate_hidden_size(self):
        h_in, w_in = self.input_size
        kernel_shape = self.target.shape
        
        # this is just the formula we used to calculate the effect of a convolution
        # with padding = 0, dilation = 1, and stride = 1
        h_out = h_in - kernel_shape[0] + 1
        w_out = w_in - kernel_shape[1] + 1
        return int(h_out) * int(w_out)

    def create_parameters(self):
        self.kernel = np.vectorize(lambda x: -1 if x == 0 else x)(target)
        
        self.W = np.zeros((self.hidden_size, self.input_size[0] * self.input_size[1]))
        for i in range(self.W.shape[0]):
            # this creates a matrix with the kernel in every possible position, with 0's everywhere else
            center = np.unravel_index(i, target.shape)
            padding_cordinates = ((center[0], 2-center[0]), (center[1], 2-center[1]))
            # the MLP weights are just the flattened array this matrix
            self.W[i] = np.pad(self.kernel, padding_cordinates).flatten()

        self.bias = np.ones(self.hidden_size) * (1 - np.sum(target))

    def linear(self, x):
        return self.W @ x + self.bias

    @staticmethod
    def ReLU(Z):
        return np.maximum(0, Z)

    @staticmethod
    def hard_threshold(Z):
        return np.maximum(0, np.minimum(Z, 1))

    def forward(self, x):
        x = x.flatten()
        x = self.linear(x)
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
    
    def test(self, model):
        correct_weights = True
        data, labels = self.get_test_examples()
        for d, label in zip(data, labels):
            pred = model(d)
            if pred == label:
                # classified correctly
                pass
            else:
                print(f"{d}\nfailed: {model.name} predicted {pred} and should have been {label}")
                correct_weights = False
        
        if correct_weights:
            print(f"{model.name} works in all cases\n")
            if model.name == "Single Layer CNN":
                print(f"target pattern:\n{model.target}\n")
                print(f"weights:\n{model.weights}")
                print(f"bias = {model.bias}")
            else:
                print(f"target pattern:\n{model.target}\n")
                print("weights:")
                for w in model.W:
                    print(w.reshape(model.input_size))
                    print()
                print(f"bias = {model.bias}")


if __name__ == "__main__":
    target = np.array([
        [1, 0, 1],
        [0, 1, 0],
        [1, 0, 1]
    ])

    CNN = SingleLayerCNN(target)
    MLP = MLP((5, 5), target)
    T = Tester(target)
    T.test(CNN)
    print("\n")
    T.test(MLP)
