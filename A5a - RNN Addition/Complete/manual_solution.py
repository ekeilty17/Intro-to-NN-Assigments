import numpy as np

class RNNAdder(object):

    def __init__(self):
        self.W_ih = np.array([[1]])     # matrix: self.hidden_size x self.input_size
        self.b_ih = np.array([0])       # vector: self.hidden_size x 1
        self.W_hh = np.array([[1]])     # matrix: self.hidden_size x self.hidden_size
        self.b_hh = np.array([0])       # vector: self.hidden_size x 1
    
    def __call__(self, seq):
        return self.forward(seq)

    def rnn_cell(self, x, hidden):
        z = self.W_ih @ np.array([x]) + self.b_ih + self.W_hh @ hidden + self.b_hh
        return self.ReLU(z)
    
    @staticmethod
    def ReLU(Z):
        return np.maximum(0, Z)
    
    def init_hidden(self):
        return np.array([0])

    def forward(self, seq):
        
        # recurrent layer
        hidden = self.init_hidden()
        for x in seq:
            hidden = self.rnn_cell(x, hidden)
        
        # simulates fully connected layer
        output = np.sum(hidden)
        return output

class Tester(object):

    def __init__(self, seq_len, high):
        self.seq_len = seq_len
        self.high = high

    def get_test_examples(self):
        data = np.random.randint(0, self.high, (1000, self.seq_len))
        labels = np.sum(data, axis=1)
        return data, labels
    
    def test(self, model):
        correct_weights = True
        seqs, labels = self.get_test_examples()
        for seq, label in zip(seqs, labels):
            pred = model(seq)
            if pred == label:
                # classified correctly
                pass
            else:
                print(f"{d}\nfailed: {model.name} predicted {pred} and should have been {label}")
                correct_weights = False
        
        if correct_weights:
            print("RNN works in all cases")
            print(f"\tW_ih = {model.W_ih}")
            print(f"\tb_ih = {model.b_ih}")
            print(f"\tW_hh = {model.W_hh}")
            print(f"\tb_hh = {model.b_hh}")

if __name__ == "__main__":
    RNN = RNNAdder()
    T = Tester(seq_len=5, high=100)
    T.test(RNN)