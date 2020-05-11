import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def plot_loss(x, train_loss=None, valid_loss=None, test_loss=None, title=None, ax=None):
    ax = plt.gca() if ax == None else ax
    if not train_loss is None:
        ax.plot(x, train_loss, label="Training Loss")
    if not valid_loss is None:
        ax.plot(x, valid_loss, label="Validation Loss")
    if not test_loss is None:
        ax.plot(x, test_loss, label="Testing Loss")
    
    ax.set_title("Loss" if title == None else title)
    
    ax.set_xlabel("Iterations")
    ax.set_xlim(left=0)
    ax.set_ylabel("Loss")
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper right")


def plot_accuracy(x, train_accuracy=None, valid_accuracy=None, test_accuracy=None, title=None, ax=None):
    ax = plt.gca() if ax == None else ax
    if not train_accuracy is None:
        ax.plot(x, train_accuracy, label="Training Accuracy")
    if not valid_accuracy is None:
        ax.plot(x, valid_accuracy, label="Validation Accuracy")
    if not test_accuracy is None:
        ax.plot(x, test_accuracy, label="Testing Accuracy")
    
    ax.set_title("Accuracy" if title == None else title)

    ax.set_xlabel("Iterations")
    ax.set_xlim(left=0)
    ax.set_ylabel("Accuracy")
    ax.set_yticks(np.arange(0, 1.1, step=0.1))
    ax.grid(linestyle='-', axis='y')
    ax.legend(loc="lower right")
    

def display_statistics(train_loss=None, train_acc=None, valid_loss=None, valid_acc=None, 
                       test_loss=None, test_acc=None, num=True, plot=True):
    
    tl = "-" if train_loss is None else round(train_loss[-1], 4)
    ta = "-" if train_acc is None else round(train_acc[-1]*100, 2)
    vl = "-\t" if valid_loss is None else round(valid_loss[-1], 4)
    va = "-" if valid_acc is None else round(valid_acc[-1]*100, 2)
    sl = "-\t" if test_loss is None else round(test_loss[-1], 4)
    sa = "-" if test_acc is None else round(test_acc[-1]*100, 2)
    
    if num:
        print(f"Training loss: {tl}{'':.20s}\t\tTraining acc: {ta}{'%' if ta != '-' else ''}")
        print(f"Validation loss: {vl}{'':.20s}\t\tValidation acc: {va}{'%' if va != '-' else ''}")
        print(f"Testing loss: {sl}{'':.20s}\t\tTesting acc: {sa}{'%' if sa != '-' else ''}")
    
    if plot:
        fig, ax = plt.subplots(1, 2, figsize=(18, 6))
        plot_loss(np.arange(0, len(train_loss), 1), train_loss, valid_loss, test_loss, ax=ax[0])
        plot_accuracy(np.arange(0, len(train_loss), 1), train_acc, valid_acc, test_acc, ax=ax[1])
        plt.show()


def display_kernel(kernel, ksize, isize, ax=None, axis="off"):
    ax = plt.gca() if ax == None else ax

    # for normalizing
    kmax = max(kernel)
    kmin = min(kernel)
    spread = kmax - kmin
    # print("max,min", kmax, kmin)

    dsize = int(isize / ksize)
    # print("dsize:", dsize)

    a = np.full((isize, isize), 0.0)

    # loop through each element of kernel
    for i in range(ksize):
        for j in range(ksize):
            # fill in the image for this kernel element
            basei = i * dsize
            basej = j * dsize
            for k in range(dsize):
                for l in range(dsize):
                    a[basei + k][basej + l] = (kernel[(i * ksize) + j] - kmin) / spread

    x = np.uint8(a * 255)

    img = Image.fromarray(x, mode='P')
    ax.imshow(img, cmap='Greys_r')
    
    if axis == "off":
        ax.axis("off")

def display_kernels(kernels, ksize, isize):

    if len(kernels) == 1:
        display_kernel(kernels[0], ksize, isize, axis="on")
        plt.show()
        return
    
    if len(kernels) < 5:
        fig, axes = plt.subplots(1, len(kernels))
    else:
        fig, axes = plt.subplots(int(np.ceil(len(kernels)/5)), 5)
        plt.subplots_adjust(hspace=0)

    for kernel, ax in zip(kernels, axes.flatten()):
        display_kernel(kernel, ksize, isize, ax, axis="off")
        
    plt.show()