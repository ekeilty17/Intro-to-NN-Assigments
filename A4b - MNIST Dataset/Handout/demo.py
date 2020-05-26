
"""
    Skip to the next multi-line comment
"""

# conda install -c anaconda tk
import tkinter as tk

def get_handwritten_digit(ROWS, COLS):

    # Create a grid of None to store the references to the tiles
    tiles = [[None for _ in range(COLS)] for _ in range(ROWS)]

    # variables trying to mimick the pressure pattern produced by pen on paper
    center = (8, "white")
    side = (4, "light grey")
    corner = (1, "grey")

    # callback function
    def draw(event):

        # Get rectangle diameters
        col_width = c.winfo_width() / COLS
        row_height = c.winfo_height() / ROWS
        
        # Calculate column and row number
        col = int(event.x // col_width)
        row = int(event.y // row_height)
        
        def create_rectangle(R, C, cell):
            # this kinda gets messy beause we need to store both the tile value and the reference to the tile
            #   the tile value is used for our model's prediction
            #   the tile reference is used so we can delete tiles if needed (this is returned by c.create_rectangle())
            return cell[0], c.create_rectangle(C*col_width, R*row_height, (C+1)*col_width, (R+1)*row_height, fill=cell[1])


        # If the tile is not filled nor previously filled, create a rectangle
        if tiles[row][col] is None or tiles[row][col][0] != center[0]:
            tiles[row][col] = create_rectangle(row, col, center)
            
            # This is adding grey edges, which helps the model correctly classify your digit
            #   orthogonal tiles
            try:
                for R, C in [(row-1, col), (row+1, col), (row, col-1), (row, col+1)]:
                    if (not tiles[R][C] is None) and (tiles[R][C][0] == center[0]):
                        continue
                    tiles[R][C] = create_rectangle(R, C, side)
            except Exception as e:
                pass
            
            #   diagonals tiles
            try:
                for R, C in [(row-1, col-1), (row+1, col+1), (row+1, col-1), (row-1, col+1)]:
                    if (not tiles[R][C] is None) and (tiles[R][C][0] == center[0] or tiles[R][C][0] == side[0]):
                        continue
                    tiles[R][C] = create_rectangle(R, C, corner)
            except Exception as e:
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
        if not tiles[row][col] is None:
            c.delete(tiles[row][col][1])
            tiles[row][col] = None

    # Create the window, a canvas and the mouse click event binding
    root = tk.Tk()
    c = tk.Canvas(root, width=500, height=500, borderwidth=5, background='black')
    
    # these bind the buttom presses for different actions
    c.pack()
    c.bind("<B1-Motion>", draw)         # if the mouse is clicked and in motion, it will draw
    c.bind("<Button-1>", delete)   # if you just click once, you can remove cells

    # starts the main loop, which is waiting for an event to trigger the callback functions
    root.mainloop() 

    # returns the bitmap, which is used as the input to our model
    bmp = [ [0 if t is None else t[0] for t in row] for row in tiles ]
    return np.array(bmp) * (256 / center[0])


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
    prediction_probs = torch.nn.Softmax(dim=1)(prediction)      # you technically don't need to do this, 
                                                                # but it's nice to have them a probabilities
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

    prediction = predict(model, image)
    display_prediction(prediction)