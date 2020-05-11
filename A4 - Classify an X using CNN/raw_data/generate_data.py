import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def perms(n):
    if not n:
        return

    for i in range(2**n):
        s = bin(i)[2:]
        s = "0" * (n-len(s)) + s
        yield s

def containsKernel(D, kernel):
    X, Y = D.shape
    x, y = kernel.shape
    
    for i in range(X-x+1):
        for j in range(Y-y+1):
            if np.array_equal(D[i:x+i, j:y+j], kernel):
                return True
    return False


def generate_all_data(n, kernel):

    data = []
    labels = []
    for flat in perms(n**2):
        print(flat)
        d = np.array([int(c) for c in flat]).reshape(n, n)
        l = containsKernel(d, kernel)
        
        #data.append("".join(map(str, d.flatten())))
        data.append(d.flatten())
        labels.append(int(l))

    np.savetxt(f"all_data_{n}.csv", data, fmt='%i', delimiter=",")
    np.savetxt(f"all_labels_{n}.csv", labels, fmt="%i", delimiter=",")


def sample_data(n, seed=None):
    
    # full dataset with 2^n entries
    data = pd.read_csv(f"all_data_{n}.csv", header=None, delimiter=",")
    labels = pd.read_csv(f"all_labels_{n}.csv", header=None, delimiter=",")

    # separating out labels
    data_with_label_0 = data.loc[labels[0] == 0]
    data_with_label_1 = data.loc[labels[0] == 1]

    # balancing dataset
    N = 10000     # total number of samples
    subsample_data_with_label_0 = data_with_label_0.sample(n=N//2, random_state=seed)
    subsample_data_with_label_1 = data_with_label_1.sample(n=N//2, random_state=seed)

    # train-test split
    train_label_0, valid_label_0 = train_test_split(subsample_data_with_label_0, test_size=0.1, random_state=seed)
    train_label_1, valid_label_1 = train_test_split(subsample_data_with_label_1, test_size=0.1, random_state=seed)

    # training data
    train_data = pd.concat( [train_label_0, train_label_1] )
    train_labels = np.concatenate([np.zeros(train_label_0.shape[0], dtype=int), np.ones(train_label_1.shape[0], dtype=int)])
    train_labels = pd.DataFrame(train_labels, index=train_data.index)

    train_data.to_csv("../data/traindata.csv", index=False, header=False)
    train_labels.to_csv("../data/trainlabels.csv", index=False, header=False)

    # validation data
    valid_data = pd.concat( [valid_label_0, valid_label_1] )
    valid_labels = np.concatenate([np.zeros(valid_label_0.shape[0], dtype=int), np.ones(valid_label_1.shape[0], dtype=int)])
    valid_labels = pd.DataFrame(valid_labels, index=valid_data.index)

    valid_data.to_csv("../data/validdata.csv", index=False, header=False)
    valid_labels.to_csv("../data/validlabels.csv", index=False, header=False)


if __name__ == "__main__":
    
    kernel = np.array([
        [1, 0, 1],
        [0, 1, 0],
        [1, 0, 1]
    ])

    # creates files 'all_data_n.csv' and 'all_labels_n.csv'
    #generate_all_data(5, kernel)
    
    # samples from those files and creates training and validation data
    sample_data(5)