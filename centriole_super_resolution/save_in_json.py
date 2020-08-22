import json as js

def save_in_json(array, save_name):
    """save a 1D or 2D array in a json file"""
    array_string = array.astype(str)
    if len(array.shape) == 2:
        to_write = [list(A) for A in array_string]
    else:
        to_write = list(array_string)
    with open(save_name, 'w') as outfile:
        js.dump(to_write, outfile)
