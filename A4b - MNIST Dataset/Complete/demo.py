# conda install -c anaconda tk
import tkinter as tk

def get_handwritten_digit(ROWS, COLS):
    
    # Create a grid of None to store the references to the tiles
    tiles = [[None for _ in range(COLS)] for _ in range(ROWS)]

    # callback function
    def draw(event):
        # Get rectangle diameters
        col_width = c.winfo_width() / COLS
        row_height = c.winfo_height() / ROWS
        
        # Calculate column and row number
        col = int(event.x // col_width)
        row = int(event.y // row_height)
        
        # If the tile is not filled, create a rectangle
        if tiles[row][col] != 4:
            c.create_rectangle(col*col_width, row*row_height, (col+1)*col_width, (row+1)*row_height, fill="white")
            tiles[row][col] = 4
            
            # This is adding grey edges, this helps the model correctly classify your digit
            try:
                if tiles[row-1][col] is None:
                    tiles[row-1][col] = 2
                    c.create_rectangle(col*col_width, (row-1)*row_height, (col+1)*col_width, (row)*row_height, fill="light grey")
            except:
                pass
            try:
                if tiles[row+1][col] is None:
                    tiles[row+1][col] = 2
                    c.create_rectangle(col*col_width, (row+1)*row_height, (col+1)*col_width, (row+2)*row_height, fill="light grey")
            except:
                pass
            try:
                if tiles[row][col-1] is None:
                    tiles[row][col-1] = 2
                    c.create_rectangle((col-1)*col_width, row*row_height, (col)*col_width, (row+1)*row_height, fill="light grey")
            except:
                pass
            try:
                if tiles[row][col+1] is None:
                    tiles[row][col+1] = 2
                    c.create_rectangle((col+1)*col_width, row*row_height, (col+2)*col_width, (row+1)*row_height, fill="light grey")
            except:
                pass
            
            # diagonals
            try:
                if tiles[row-1][col-1] is None:
                    tiles[row-1][col-1] = 1
                    c.create_rectangle((col-1)*col_width, (row-1)*row_height, (col)*col_width, (row)*row_height, fill="grey")
            except:
                pass
            try:
                if tiles[row+1][col+1] is None:
                    tiles[row+1][col+1] = 1
                    c.create_rectangle((col+1)*col_width, (row+1)*row_height, (col+2)*col_width, (row+2)*row_height, fill="grey")
            except:
                pass
            try:
                if tiles[row+1][col-1] is None:
                    tiles[row+1][col-1] = 1
                    c.create_rectangle((col-1)*col_width, (row+1)*row_height, (col)*col_width, (row+2)*row_height, fill="grey")
            except:
                pass
            try:
                if tiles[row-1][col+1] is None:
                    tiles[row-1][col+1] = 1
                    c.create_rectangle((col+1)*col_width, (row-1)*row_height, (col+2)*col_width, (row)*row_height, fill="grey")
            except:
                pass

    # callback function
    def delete(event):
        # Get rectangle diameters
        col_width = c.winfo_width() / COLS
        row_height = c.winfo_height() / ROWS
        
        # Calculate column and row number
        col = int(event.x // col_width)
        row = int(event.y // row_height)
        
        # If the tile is not filled, create a rectangle
        if tiles[row][col]:
            c.delete(tiles[row][col])
            tiles[row][col] = None

    # Create the window, a canvas and the mouse click event binding
    root = tk.Tk()
    c = tk.Canvas(root, width=500, height=500, borderwidth=5, background='black')
    
    # these bind the buttom presses for different actions
    c.pack()
    c.bind("<B1-Motion>", draw)         # if the mouse is clicked and in motion, it will draw
    c.bind("<ButtonPress-1>", delete)   # if you just click once, you can remove cells

    # starts the main loop, which is waiting for an event to trigger the callback functions
    root.mainloop() 

    # returns the bitmap, which is used as the input to our model
    bmp = [ [0 if x is None else x for x in row] for row in tiles ]
    return np.array(bmp)


"""
    The code above allows you to draw on a grid, which is used for the demo.
    Feel free to look at the code, but the important parts are below this point,
    which is the actual machine learning stuff
"""


import torch
import numpy as np


def predict(model, image):
    image = torch.tensor(image).unsqueeze(0).unsqueeze(0)       # need to add a dimension for the batch
                                                                # and another for the channel
    prediction = model(image.float())
    prediction_probs = torch.nn.Softmax()(prediction)
    return prediction_probs.squeeze().data.numpy()

def display_prediction(prediction):
    p = np.argmax(prediction)
    for i, prob in enumerate(prediction):
        print(f"{i}:\t{prob:.4f}\t{'<--' if i == p else ''}")

if __name__ == "__main__":
    model = torch.load("cnn.pt")
    #model = torch.load("mlp.pt")
    model.eval()
    
    image = get_handwritten_digit(28, 28)
    image *= 64

    prediction = predict(model, image)
    display_prediction(prediction)