import numpy as np

class SingleNeuronClassifier(object):

    def __init__(self, num_inputs, actfunction="relu", lr=0.1):
        self.num_inputs = num_inputs

        # hyper parameters
        self.set_actfunction(actfunction)
        self.lr = lr
    
    def __repr__(self):
        out = "Weights\n"
        out += str(self.W[:3]) + '\n'
        out += str(self.W[3:6]) + '\n'
        out += str(self.W[6:]) + '\n'
        return out + f"\nBias: {self.bias}"

    def __call__(self, I):
        return self.forward(I)

    def init_parameters(self):
        self.w = np.random.uniform(0, 1, self.num_inputs)
        self.bias = np.random.uniform(0, 1)
        self.z = 0
        self.y = 0
        self.dy = 0
    
    def get_parameters(self):
        return self.bias, self.w

    """ Activation Functions """
    def set_actfunction(self, actfunction):
        if actfunction.lower() == "linear":
            self.g = self.Linear
        elif actfunction.lower() == "relu":
            self.g = self.ReLU
        elif actfunction.lower() == "sigmoid":
            self.g = self.Sigmoid
        else:
            raise ValueError(f"Don't support the {actfunction} activation function")

    @staticmethod
    def Linear(z):
        return z, 1
    
    @staticmethod
    def ReLU(z):
        if z == 0:
            return 0, 0
        return max(0, z), max(0, z)/z
    
    @staticmethod
    def Sigmoid(z):
        y = 1.0/(1.0 + np.exp(-z))
        return y, y * (1-y)

    """ ML Algorithms """
    def forward(self, I):
        self.z = 0
        for w, i in zip(self.w, I):
            self.z += w * i
        self.z += self.bias
        self.y, self.dy = self.g(self.z)
        return self.y

    # this assumes forward_propogate has been called
    def backward(self, data, label):
        self.dE_dw += 2*(self.y - label) * self.dy * data
        self.dE_db += 2*(self.y - label) * self.dy
        self.cnt += 1
        
        return self.dE_dw / self.cnt, self.dE_db / self.cnt
    
    def zero_grad(self):
        self.dE_dw = 0
        self.dE_db = 0
        self.cnt = 0
    
    def step(self):
        self.w -= self.lr * (self.dE_dw / self.cnt)
        self.bias -= self.lr * (self.dE_db / self.cnt)
