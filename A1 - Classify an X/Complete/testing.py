from a2 import *
import matplotlib.pyplot as plt 

# Getting Training and Validation Data
traindata, trainlabel = get_data("train")
validdata, validlabel = get_data("valid")

def everything(G, Alpha, Epochs):
    for g in G:
        for alpha in Alpha:
            for epochs in Epochs:
                print(f"Activation Function: {g}\tLearning Rate: {alpha}\tEpochs: {epochs}")
                N = SingleNeuronClassifier(actfunction=g, alpha=alpha, seed=10)
                train_error, train_accuracy, valid_error, valid_accuracy = N.train( traindata=traindata, 
                                                                            trainlabel=trainlabel, 
                                                                            validdata=validdata, 
                                                                            validlabel=validlabel, 
                                                                            epochs=epochs, 
                                                                            plot=True
                                                                        )
                accuracy = N.validate(validdata, validlabel)
                print(f"Accuracy: {accuracy * 100}%")

def testing_activation_function(G, alpha, epochs):
    for g in G:
        print(f"Activation Function: {g}\tLearning Rate: {alpha}\tEpochs: {epochs}")
        N = SingleNeuronClassifier(actfunction=g, alpha=alpha, seed=10)
        train_error, train_accuracy, valid_error, valid_accuracy = N.train( traindata=traindata, 
                                                                            trainlabel=trainlabel, 
                                                                            validdata=validdata, 
                                                                            validlabel=validlabel, 
                                                                            epochs=epochs, 
                                                                            plot=False
                                                                        )
        accuracy = N.validate(validdata, validlabel)
        print(f"Accuracy: {accuracy * 100}%")

        #plt.plot(np.arange(0, epochs+1, 1), train_error, label=str(g))
        plt.plot(np.arange(0, epochs+1, 1), train_accuracy, label=str(g))
    plt.xlabel("Epochs")
    plt.ylabel("Training Data Accuracy")
    plt.title("Effect of Activation Function with Learning rate of " + str(alpha))
    plt.legend()
    plt.show()
    plt.clf()

def testing_learning_rate(g, Alpha, epochs):
    for alpha in Alpha:
        print(f"Activation Function: {g}\tLearning Rate: {alpha}\tEpochs: {epochs}")
        N = SingleNeuronClassifier(actfunction=g, alpha=alpha, seed=10)
        train_error, train_accuracy, valid_error, valid_accuracy = N.train( traindata=traindata, 
                                                                            trainlabel=trainlabel, 
                                                                            validdata=validdata, 
                                                                            validlabel=validlabel, 
                                                                            epochs=epochs, 
                                                                            plot=False
                                                                        )
        accuracy = N.validate(validdata, validlabel)
        print(f"Accuracy: {accuracy * 100}%")

        #plt.plot(np.arange(0, epochs+1, 1), train_error, label=str(alpha))
        plt.plot(np.arange(0, epochs+1, 1), train_accuracy, label=str(alpha))

    plt.xlabel("Epochs")
    plt.ylabel("Training Data Accuracy")
    plt.title("Effect of Learning Rate on " + g.title())
    #plt.legend()
    plt.show()
    plt.clf()

def testing_seeds(g, alpha, Epochs, Seeds):
    for epochs in Epochs:
        for seed in Seeds:
            print(f"Activation Function: {g}\tLearning Rate: {alpha}\tEpochs: {epochs}\tSeed: {seed}")
            N = SingleNeuronClassifier(actfunction=g, alpha=alpha, seed=seed)
            train_error, train_accuracy, valid_error, valid_accuracy = N.train( traindata=traindata, 
                                                                                trainlabel=trainlabel, 
                                                                                validdata=validdata, 
                                                                                validlabel=validlabel, 
                                                                                epochs=epochs, 
                                                                                plot=False
                                                                            )
            accuracy = N.validate(validdata, validlabel)
            print(f"Accuracy: {accuracy * 100}%")

            #plt.plot(np.arange(0, epochs+1, 1), train_error, label=str(seed))
            #plt.plot(np.arange(0, epochs+1, 1), train_accuracy, label=str(seed))
        """
        plt.xlabel("Epochs")
        plt.ylabel("Training Data Accuracy")
        plt.title("Effect of Different Seed Values on " + g.title())
        plt.legend()
        plt.show()
        plt.clf()
        """

def plot(g, alpha, epochs, seed):
    print(f"Activation Function: {g}\tLearning Rate: {alpha}\tEpochs: {epochs}")
    N = SingleNeuronClassifier(actfunction=g, alpha=alpha, seed=seed)
    train_error, train_accuracy, valid_error, valid_accuracy = N.train( traindata=traindata, 
                                                                        trainlabel=trainlabel, 
                                                                        validdata=validdata, 
                                                                        validlabel=validlabel, 
                                                                        epochs=epochs, 
                                                                        plot=False
                                                                    )
    accuracy = N.validate(validdata, validlabel)
    print(f"Accuracy: {accuracy * 100}%")
    
    plt.plot(np.arange(0, epochs+1, 1), valid_accuracy, label="Validation Accuracy")
    plt.plot(np.arange(0, epochs+1, 1), train_accuracy, label="Training Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()
    plt.clf()

    plt.plot(np.arange(0, epochs+1, 1), valid_error, label="Validation Loss")
    plt.plot(np.arange(0, epochs+1, 1), train_error, label="Training Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()
    plt.clf()

    dispKernel(N.W, 3, 300)
    plt.show()
    plt.clf()

if __name__ == "__main__":
    
    """ Plotting all possiblities """
    #everything(["linear", "relu", "sigmoid"], [0.01, 0.05, 0.1, 0.3, 0.5, 0.7, 1.0, 5.0], [50, 100, 200, 500, 700, 1000])
    #everything(["linear"], [0.0001, 0.001, 0.005], [50, 100, 200, 500, 700, 1000])

    """ Graphing various learning rates """
    #testing_learning_rate("sigmoid", [0.001, 0.005] + [x/100 for x in range(1, 21, 1)], 100)
    #testing_learning_rate("relu", [x/1000 for x in range(1, 11, 1)] + [x/100 for x in range(1, 11, 1)], 200)
    #testing_learning_rate("linear", [x/1000 for x in range(1, 11, 1)] + [x/100 for x in range(1, 11, 1)], 200)

    """ Graphing various activation functions """
    #testing_activation_function(["linear", "relu", "sigmoid"], 0.01, 1000)
    #testing_activation_function(["linear", "relu", "sigmoid"], 0.1, 400)
    #testing_activation_function(["linear", "relu", "sigmoid"], 0.5, 200)

    """ Graphing various seeds """
    #testing_seeds("sigmoid", 0.01, 1000, [1, 5, 10, 20, 100])
    #testing_seeds("sigmoid", 0.05, [50, 100, 200, 500, 700, 1000], [1, 5, 10, 50, 100])
    #testing_seeds("sigmoid", 0.1, 300, [1, 5, 10, 20, 100])
    
    #testing_seeds("relu", 0.001, 500, [1, 5, 10, 20, 100])
    #testing_seeds("linear", 0.005, [50, 100, 200, 500, 700, 1000], [1, 5, 10, 50, 100])
    #testing_seeds("relu", 0.1, 150, [1, 5, 10, 20, 100])

    #testing_seeds("linear", 0.001, 500, [1, 5, 10, 20, 100])
    #testing_seeds("linear", 0.01, 400, [1, 5, 10, 20, 100])
    #testing_seeds("linear", 0.1, 150, [1, 5, 10, 20, 100])

    """ Specific Plots """
    # Low Learning rate
    #plot("sigmoid", 0.005, 1500, 10)

    # High Learning rate
    #plot("linear", 0.3, 50, 10)

    # Good Learning Rate
    #plot("relu", 0.01, 500, 10)

    # Overtraining
    #plot("linear", 0.1, 1000, 10)

    # Best set of parameters
    plot("sigmoid", 10, 50, 0)

    #plot("relu", 0.1, 100, 50)
    
