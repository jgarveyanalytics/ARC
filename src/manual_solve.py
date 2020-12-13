#!/usr/bin/python

"""
Name: Jonathan Garvey
ID:   06744885
GitHub Repo URL: https://github.com/jgarveyanalytics/ARC
"""

import os, sys
import json
import numpy as np
import re
import itertools

# Dictionary of colours. Used in most solve functions.
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

def solve_83302e8f(x):
    '''
    Transformation Description
       Input:
        Grid divided up into black squares of equal area using vertical
        and horizontal gridlines.
        Gridlines can be of any colour except for black and critically, can have gaps.
        
       Output:
        Grid size the exact same as well as gridlines' colour and position.
        Where no gaps exist in gridlines, enclosed inner squares are green.
        Where gaps exist, i.e. when two inner squares can touch each other,
        their combined area is shaded yellow.
        
       Transformation:
        Enclosed squares' colour: black --> green
        Area of connected squares via gaps, colour: black --> yellow
    
    All training and testing grids were solved correctly.
    
    How Solve Works
    1) Calculates the order index at which gridlines exist. E.g. they might exist for every index that is a multiple of 5.
       It does this by starting at the origin [0,0] and creates increasing-size squares in the form of NumPy arrays.
       When a NumPy array contains a colour that matches the colour of the gridline, solve stores the length of the square
       as the gridline order index.
    
    2) Using the result from above, a new 2D NumPY array is created, the same as the [x], except with no gaps in the gridlines.
       A base_shade function is used to provide an initial shading for each box. Green is the default, unless the box is part of
       a gridline in which case it will be shaded the same colour as the gridlines provided in [x].
    
    3) Since the newly created array is of the same shape as [x], with the gridlines the same colour and in the same position,
       the solve function simply subtracts [x] from the grid with no gaps, which results in an array with non-zero values
       in positions that represent gaps in gridlines.
    
    4) Solve then iterates through each gap and stores the adjacent squares' coordinates in a list.
    
    5) Finally, using the unique set of squares' coordinates from the list, solve shades the corresponding region yellow in the new grid.
       It also shades the gaps themselves yellow. Gaps at tntersections are not really significant so while they are
       shaded yellow, it is not necessary to find adjacent partitions to them.
       
    6) Solve returns the grid once all the shading is complete.
    '''
    
    border_colour = x.max() # Considering the borders can't be black, the max of x is the border colour
    yellow_regions = [] # Placeholder for squares that will end up being yellow.
    black = colours['black'] # Value for black
    yellow = colours['yellow'] # Value for yellow
    green = colours['green'] # Value for green
    
    # Used for setting up new grid with no gaps in gridlines
    # x,y are row and column indices
    # d is the distance from one gridline to the next
    # border_colour is the colour to shade the border/gridlines
    base_shade = lambda x,y,d,border_colour: border_colour if ((x+1) % d == 0 and x > 0) or ((y+1) % d == 0 and y > 0) else green
    
    def store_yellow_regions(x, row_index, col_index, d):
        '''
        Gets and stores the adjacent squares to a gridline gap. These regions are to be shaded yellow.           
        Takes a numpy array, row and column indeces and d (distance from one gridline to the next).
        The function does not return anything, it simply append the yellow regions' representations
        to a set for downstream processing.
        ''' 

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
            
            yellow_regions.append((row_min_left_partition, row_max_left_partition, col_min, col_max))
            yellow_regions.append((row_min_right_partition, row_max_right_partition, col_min, col_max))
            
        else: # gap is on a vertical border, i.e. dividing horizontally
            
            row_min = row_index - (row_index % d)
            row_max = row_min + (d - 2)
            
            # left box
            col_min_left_partition = col_index - (d - 1)
            col_max_left_partition = col_index - 1
            
            # right box
            col_min_right_partition = col_index + 1
            col_max_right_partition = col_index + (d -1)
            
            yellow_regions.append((row_min, row_max, col_min_left_partition, col_max_left_partition))
            yellow_regions.append((row_min, row_max, col_min_right_partition, col_max_right_partition))
    
    # Starting at origin [0,0], check squares of increasing size for a non-black colour (i.e. sum of values in square > 0)
    sub_grid = x[0,0]
    shape_x = x.shape
    border_multiple_of = 0
    
    for i in range(shape_x[1]):
        sub_grid = (x[0:i+1, 0:i+1])
        if sub_grid.sum() > 0: # Gridline detected
            border_multiple_of = sub_grid.shape[0] # Store border index multiple of
            break # Exit loop

    coords = np.where(x==x) # Get coordinates of input [x]
    
    # Get new grid of same spec as [x] but with no gaps in gridlines
    x_hat = np.array([base_shade(point[0],point[1], border_multiple_of, border_colour) for point in zip(coords[0],coords[1])]).reshape(x.shape)
    
    # Get coordinates where gridline gaps exist
    # x_hat != green means only compare when the coordinate is on a gridline plane
    gap_coords = np.where((x_hat != x) & (x_hat != green))
    
    # Get the adjacent regions for every gap
    for row_index, col_index in [gap for gap in zip(gap_coords[0],gap_coords[1])]:
        
        x_hat[row_index, col_index] = yellow # Shade the gap itself yellow
        
        if row_index != col_index: # i.e. not a vertical/horzintal gridline intersection.
            store_yellow_regions(x, row_index, col_index, border_multiple_of)
    
    # Set regions to yellow that are connected via a gridline gap
    # Could have used a set initially, but performing many updates to the set might have been expensive.
    for row_min, row_max, col_min, col_max in set(yellow_regions):
        x_hat[row_min:row_max+1, col_min:col_max+1] = yellow
    
    return x_hat

def solve_5ad4f10b(x):
    '''
    Transformation Description
       Input:
        An array with a random scaterring (almost like the profile of snow) of one colour throughout.
        In addition, there exists a second shape. The shape consists of a logical box divided
        into 3 rows and 3 columns. All 9 of the cells of this logical box are either fully shaded in the same colour
        (which is different from the first colour described), or black. However if black, elements of colour one can
        appear in the cell. Each cell is of width & height: width of logical box / 3.

       Output:
        A 3x3 array representing the logical box described in the input. Each element in the output is coloured if the
        corresponding cell in the logical box is coloured. In black in the input, it will be black in the output.
        
       Transformation:
        Input as described above -> extract logical box from input --> convert to 3x3 array and shade according to relative
        colouring in the logical box from the input.

    All training and testing grids were solved correctly.

    How Solve Works
    1) Gets the two non-black colours from the input array.
    
    2) For each colour, finds the min and max row and column indexes where the colour appears.
    
    3) Computes the area of a rectangle using the above planes for each colour. The smaller rectangle of the two represents the colour
       who's 3x3 logical box we are interested in. This step also stores which colour to use when shading the output array.
       
    4) Creates a 3x3 array of zeroes to hold output array.
    
    5) Calculates the width and height of the logical box's cells and uses this to loop through each cell and determine if it is coloured or
       black.
       
    6) If coloured, it places the appropriate colour in the relative position of the output array. If black, no update needed.
    
    7) Returns the output array, x_hat.
    '''
    
    black = colours['black'] # Value for black
    unique_colours = list(np.unique(x)) # Gets list of unique colours from input array [x]
    unique_colours.remove(black) # Removes black from this list as it is not needed
    
    # Placeholder for region which forms the logical 3x3 box
    region_to_process = []
    # Placeholder for the colour to use for final shading
    final_colour = 0
    
    # For each non-black colour in the input array
    for colour in unique_colours:
    
        # Get all coordinates of this colour
        coords = np.where(x == colour)
    
        # Set planes that would encompass all instances of this colour in a rectangle
        row_min = coords[0].min()
        row_max = coords[0].max()
        col_min = coords[1].min()
        col_max = coords[1].max()        
        
        # Get the area of the rectangle
        area = (row_max - row_min + 1) * (col_max - col_min + 1)
        
        # Determine which colour is the one that represents the 3x3 logical box
        # and then store its area and planes

        if not region_to_process: # Empty on first iteration.
        
            # Set to properties of first rectangle
            region_to_process = [colour, area, row_min, row_max, col_min, col_max]
            final_colour = colour
        
        # Second iteration. This case represents the scenario where the rectangle encompassing
        # the second colour is smaller than the first. This is the rectangle we want.
        elif area < region_to_process[1]:
            region_to_process = [colour, area, row_min, row_max, col_min, col_max]
            
        else: # It turns out the first rectangle was the one we want
            final_colour = colour
    
    # Get the rectangle properties
    colour, area, row_min, row_max, col_min, col_max = region_to_process
    length_each_square = (row_max - row_min + 1) / 3
    
    # Placeholder for output array. I think it's a fair assumption to say it will
    # have a shape (3,3) as that is how all the examples are setup
    x_hat = np.zeros([3,3])
    
    # For every element of the 3x3 output array
    for row_index, col_index in itertools.product(range(3), range(3)):
    
        # Get the coordinates of the relative cell from the logical box in the input
        row_begin = int(row_min + (row_index * length_each_square))
        row_end = int(row_begin + length_each_square)
        col_begin = int(col_min + (col_index * length_each_square))
        col_end = int(col_begin + length_each_square)
        
        # Get the relative cell from the logical box in the input. This is actually a 2D
        # NumPy array.
        square = x[row_begin:row_end, col_begin:col_end]

        # Update output array's colour at the specified position based on the colour of the
        # relative cell from the logical box in the input.
        x_hat[row_index, col_index] = final_colour if np.any(square == colour) else black

    return x_hat

def solve_b190f7f5(x):
    '''
    Transformation Description
       Input:
        An array with two pieces of information in the form of two logical sections.
        Both logical sections are squares.
        One of them contains a pattern in a light blue colour.
        The other section contains a multi-coloured (except light-blue) pattern.
        
       Output:
        A grid with dimensions of:
        (width of pattern 1 * width of pattern 2) , (height of pattern 1 * height of pattern 2)
        For every non-black element in the multi-coloured pattern, there exists a pattern with the same
        shape as the light blue pattern but with the colour and relative position of the element in the 
        multi-coloured pattern.
        
       Transformation:
        For every non-black element E in the multi-coloured pattern --> replace with the light-blue pattern
        and then change the colour to the colour of the elemennt E.
        For every black element E_b in the multi-coloured pattern --> replace with an array with same shape
        of the blue patten section of the input, and shade black.

    All training and testing grids were solved correctly.

    How Solve Works
    1) Obtains the target 'figure' (i.e. the light-blue shape) and the colours/relative posisitions to deploy to.
    
    2) Calculates the dimensions of and creates a new output array filled with zeros.
    
    3) Looks at the colour/position schema and for each entry, figures out where to place a colour-modified copy of the light-blue
       figure in the output array.
       
    4) Returns the output array, x_hat.
    '''

    light_blue = colours['light-blue'] # Value for light-blue
    black = colours['black'] # Value for black
    figure = [] # Placeholder for blue pattern
    colours_positions = [] # Placeholder for multi-coloured pattern
    
    height_x, width_x = x.shape # Get shape of entire input array
    
    # Split vertically if the input array is more long than wide
    # Else split horizontally
    # Store the result in a schema to consume downstram
    schema = np.vsplit(x,2) if height_x > width_x else np.hsplit(x,2)

    # First section contains the blue pattern
    if np.any(schema[0] == light_blue):
        figure = schema[0]
        colours_positions = schema[1]
    
    # Second section contains the blue pattern
    else:
        figure = schema[1]
        colours_positions = schema[0]
    
    # Store shapes of schema elements
    figure_shape = np.array(figure.shape)
    colours_positions_shape = np.array(colours_positions.shape)
    
    # Output's shape is equal to the product of the two schema sections' shapes
    shape_xhat = figure_shape * colours_positions_shape

    # Placeholder for function output
    # With black being representated as zero, the below line also provides the background we need
    x_hat = np.zeros(shape_xhat)
    
    # Coordinates of all elements in the multi-coloured pattern
    coords_colours_positions = np.where(colours_positions==colours_positions)
    
    # Placeholder for storing a modified version of the light-blue pattern
    new_figure = []
    
    # For every row/col coordinate in the multi-coloured pattern
    for row_index, col_index in zip(coords_colours_positions[0], coords_colours_positions[1]):
        # Get the colour
        colour = colours_positions[row_index,col_index]
        
        # Create a modified version of the blue pattern.
        new_figure = np.where(figure == light_blue, colour, black)
        
        # Calculate coordinates of which section of x_hat to update
        x_hat_row_min = row_index * figure_shape[0]
        x_hat_row_max = x_hat_row_min + figure_shape[0]
        x_hat_col_min = col_index * figure_shape[1]
        x_hat_col_max = x_hat_col_min + figure_shape[1]
        
        # Replace the data represented by above coordinates with the newly generated figure
        x_hat[x_hat_row_min:x_hat_row_max, x_hat_col_min:x_hat_col_max] = new_figure

    return (x_hat)

def solve_662c240a(x):
    '''
    Transformation Description
       Input:
        A 9 x 3 array logically divided into 3 3x3 arrays each containing a pattern with 2 colours.
        
       Output:
        One of the 3x3 arrays.
        
       Transformation:
        A function which essentially returns the 3x3 array which contains the least number of what I call
        'mistakes'. I define a mistake as for a given diagonal in the south west - north east direction, an
        instance where a colour does not match the first colour of the diagonal.
    
    All training and testing grids were solved correctly.
    
    How Solve Works
    1) Splits the 9x3 array into 3 3x3 arrays.
    
    2) Calculates the number of mistakes for each array.
    
    3) Returns the array with the least number of mistakes.
    '''

    sections = np.vsplit(x,[3,6]) # Split the 9x3 array into 3 3x3 arrays
    max_mistakes = (0,0) # Placeholder for index and number of mistakes for array with most mistakes
    
    for i, section in enumerate(sections): # For each 3x3 section
        section = np.rot90(np.flip(section)) # Flip and rotate 90 degrees to enable southwest -> northeast diagonal computation
        mistakes = 0
        
        for k in range(-2,3): # 5 southwest -> northeast diagonals in a 3x3 array, from diagonal index -2 to +2
            diag = np.diag(section,k) # Get diagonal from section
            expected_colour = diag[0] # Get first elements's colour
            mistakes += sum(np.where(diag != expected_colour, 1, 0)) # Get mistakes (i.e. where subsequents colours don't match first colour)
        
        # Update max_mistakes tuple if this array contains the most mistakes so far
        max_mistakes = (i,mistakes) if mistakes > max_mistakes[1] else max_mistakes

    return sections[max_mistakes[0]] # Return array with most mistakes

def solve_ff805c23(x):
    '''
    Transformation Description
       Input:
        Symmetrical pattern with a blue square blocking out a section of the pattern.
        Pattern can symmetrical horizontally or vertically.
        It can sometimes be symmetrical both horizontally and vertically but it is not a rule.
        and horizontal gridlines.
        
       Output:
        The would-be section of the pattern that the blue square is obfuscating.
        
       Transformation:
        Original image/pattern with obfuscation -> Pattern that exists at original coordinates of
        the blue box after image rotated 180 degrees.
    
    All training and testing grids were solved correctly.
    
    How Solve Works
    1) Gets the coordinates of the blue box.
    
    2) Rotates the image 180 degrees.
    
    3) Returns the pattern that exists in the original coordinates of the blue box.
    '''
    
    blue = colours['blue'] # Value for blue
    blue_square_coords = np.where(x == blue) # Get coordinates of blue box
    
    x_hat = np.rot90(x,2) # Rotate image 180 degrees
    x_hat = x_hat[blue_square_coords].reshape(5,5) # Get pattern at original blue square's coordinates
    
    return x_hat

def main():
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

