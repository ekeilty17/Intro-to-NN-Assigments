class SingleNeuralClassifier(object):

    def __init__(self, target):
        self.target = target

        # TODO
        self.weights = ## Manual Solution Here
        self.bias = ## Manual Solution Here

        # where we will store activations
        self.z = 0

    def feedforward(self, input_list):
        self.z = 0
        for I, w in zip(input_list, self.weights):
            self.z  += I * w
        self.z += self.bias
        return self.z

    def isMatch(self, input_list):
        self.feedforward(input_list)
        return self.z > 0

class Tester(object):

    def __init__(self, target):
        self.target = target

    @staticmethod
    def perms(n):
        if not n:
            return

        for i in range(2**n):
            s = bin(i)[2:]
            s = "0" * (n-len(s)) + s
            yield s

    def isMatch(self, input_list):
        return self.target == input_list

    def test(self, NN):
        correct_weights = True
        for b in self.perms(len(target)):
            input_list = [int(c) for c in b]
            if NN.isMatch(input_list) == self.isMatch(input_list):
                # classified correctly
                pass
            else:
                print(f"{input_list} failed: NN.isMatch = {NN.isMatch(input_list)}")
                correct_weights = False

        if correct_weights:
            print("Single Neural Classifier works in all cases")
            print(f"pattern = {NN.target}")
            print(f"weights = {NN.weights}")
            print(f"bias = {NN.bias}")

        return correct_weights

if __name__ == "__main__":
    target = [
        1, 0, 1,
        0, 1, 0,
        1, 0, 1
    ]

    NN = SingleNeuralClassifier(target)
    T = Tester(target)

    T.test(NN)
