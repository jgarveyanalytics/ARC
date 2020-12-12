#!/usr/bin/python

import os, sys
import json
import numpy as np
import re

### YOUR CODE HERE: write at least three functions which solve
### specific tasks by transforming the input x and returning the
### result. Name them according to the task ID as in the three
### examples below. Delete the three examples. The tasks you choose
### must be in the data/training directory, not data/evaluation.

colours = {
    "black": 0,
    "blue": 1,
    "red": 2,
    "green": 3,
    "yellow": 4,
    "grey": 5,
    "pink": 6,
    "orange": 7,
    "light-blue": 8,
    "brown": 9
}

def copy_grid(x, except_colours = [], only_colours = []):
    '''Makes a copy of a grid and defaults to zeroes if no colours are supplied.
       If colours are supplied, the new grid will contain the same colours in the same
       positions as the input grid.
       Can also specify to copy all colouring and related positions except for 
       specific colours.
    '''
    colours = [colour for colour in range(10) if colour not in except_colours and ((colour in only_colours) or (not only_colours))]
    
    x = np.isin(x, colours) * x
    
    return x

def solve_83302e8f(x):
    '''Train input:
        Main grid divided up into a grid of black squares by means of criss-cross lines
            equally spaced and of any other colour.
        Borders can be of any other colour and critically, can have gaps.
        
       Train output:
        Grid size the exact same.
        Horizontal and verticial lines (the 'criss cross') same size, position and colour.
        Where no gaps exist in borders, enclosed inner squares are green.
        Where gaps exist, i.e. when two inner squares flow into one another,
         their combined area is shaded yellow.
    '''
    border_colour = x.max()
    yellow_regions = set()
    black = colours['black']
    yellow = colours['yellow']
    green = colours['green']
    
    base_shade = lambda x,y,d,border_colour: border_colour if ((x+1) % d == 0 and x > 0) or ((y+1) % d == 0 and y > 0) else green
    
    def store_yellow_regions(x, row_index, col_index, d):

        # Find out whether it's dividing in a vertical or horizontal direction
        if (row_index+1) % d == 0: # gap is on a horizontal border, i.e. dividing vertically
            col_min = col_index - (col_index % d)
            col_max = col_min + (d - 2)
            
            # left box
            row_min_left_partition = row_index - (d - 1)
            row_max_left_partition = row_index - 1
            
            # right box
            row_min_right_partition = row_index + 1
            row_max_right_partition = row_index + (d -1)
            
            yellow_regions.add((row_min_left_partition, row_max_left_partition, col_min, col_max))
            yellow_regions.add((row_min_right_partition, row_max_right_partition, col_min, col_max))
            
        else: # gap is on a vertical border, i.e. dividing horizontally
            
            row_min = row_index - (row_index % d)
            row_max = row_min + (d - 2)
            
            # left box
            col_min_left_partition = col_index - (d - 1)
            col_max_left_partition = col_index - 1
            
            # right box
            col_min_right_partition = col_index + 1
            col_max_right_partition = col_index + (d -1)
            
            yellow_regions.add((row_min, row_max, col_min_left_partition, col_max_left_partition))
            yellow_regions.add((row_min, row_max, col_min_right_partition, col_max_right_partition))
    
    sub_grid = x[0,0]
    shape_x = x.shape
    border_multiple_of = 0
    for i in range(shape_x[1]):
        sub_grid = (x[0:i+1, 0:i+1])
        if sub_grid.sum() > 0:
            border_multiple_of = sub_grid.shape[0]
            break

    coords = np.where(x==x)
    base_grid = np.array([base_shade(point[0],point[1], border_multiple_of, border_colour) for point in zip(coords[0],coords[1])]).reshape(x.shape)
    
    gap_coords = np.where((base_grid - x > 0) & (base_grid != 3))
    
    for row_index, col_index in [gap for gap in zip(gap_coords[0],gap_coords[1])]:
        base_grid[row_index, col_index] = yellow
        if row_index != col_index: # intersection
            store_yellow_regions(x, row_index, col_index, border_multiple_of)
    
    for row_min, row_max, col_min, col_max in yellow_regions:
        base_grid[row_min:row_max+1, col_min:col_max+1] = yellow
    
    return base_grid

def main():
    """ Name: Jonathan Garvey
        ID:   06744885
        GitHub Repo URL: https://github.com/jgarveyanalytics/ARC
    """
    # Find all the functions defined in this file whose names are
    # like solve_abcd1234(), and run them.

    # regex to match solve_* functions and extract task IDs
    p = r"solve_([a-f0-9]{8})" 
    tasks_solvers = []
    # globals() gives a dict containing all global names (variables
    # and functions), as name: value pairs.
    for name in globals(): 
        m = re.match(p, name)
        if m:
            # if the name fits the pattern eg solve_abcd1234
            ID = m.group(1) # just the task ID
            solve_fn = globals()[name] # the fn itself
            tasks_solvers.append((ID, solve_fn))

    for ID, solve_fn in tasks_solvers:
        # for each task, read the data and call test()
        directory = os.path.join("..", "data", "training")
        json_filename = os.path.join(directory, ID + ".json")
        data = read_ARC_JSON(json_filename)
        test(ID, solve_fn, data)
    
def read_ARC_JSON(filepath):
    """Given a filepath, read in the ARC task data which is in JSON
    format. Extract the train/test input/output pairs of
    grids. Convert each grid to np.array and return train_input,
    train_output, test_input, test_output."""
    
    # Open the JSON file and load it 
    data = json.load(open(filepath))

    # Extract the train/test input/output grids. Each grid will be a
    # list of lists of ints. We convert to Numpy.
    train_input = [np.array(data['train'][i]['input']) for i in range(len(data['train']))]
    train_output = [np.array(data['train'][i]['output']) for i in range(len(data['train']))]
    test_input = [np.array(data['test'][i]['input']) for i in range(len(data['test']))]
    test_output = [np.array(data['test'][i]['output']) for i in range(len(data['test']))]

    return (train_input, train_output, test_input, test_output)


def test(taskID, solve, data):
    """Given a task ID, call the given solve() function on every
    example in the task data."""
    print(taskID)
    train_input, train_output, test_input, test_output = data
    print("Training grids")
    for x, y in zip(train_input, train_output):
        yhat = solve(x)
        show_result(x, y, yhat)
    print("Test grids")
    for x, y in zip(test_input, test_output):
        yhat = solve(x)
        show_result(x, y, yhat)

        
def show_result(x, y, yhat):
    print("Input")
    print(x)
    print("Correct output")
    print(y)
    print("Our output")
    print(yhat)
    print("Correct?")
    # if yhat has the right shape, then (y == yhat) is a bool array
    # and we test whether it is True everywhere. if yhat has the wrong
    # shape, then y == yhat is just a single bool.
    print(np.all(y == yhat))

if __name__ == "__main__": main()

