import numpy as np
import pandas as pd
import os


def pairing_function(a, b):
    return (1/2)*(a+b)*(a+b+1) + b


# Get current working directory
dirname = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', ''))

# Load saved data
entries = os.listdir(dirname + r'/data/simulations/')
for file in entries:
    load_score_arr = np.load(dirname + r'/data/simulations/' + file + r'/' + 'lds_data.npy')

    # Identify the minimum load score value indices
    indices = np.where(load_score_arr[:, 4] == np.amin(load_score_arr, axis=0)[4])

    # Consider each cart individually. Take the minimum load score value obtained at each unique (x,y) position of
    # the cart for all possible locations of the other cart.
    x1 = []
    y1 = []
    z1 = []
    for c1x1, c1y1 in load_score_arr[:, [0, 1]]:
        # Ignore duplicates
        clx_matches_index = [i for i, e in enumerate(x1) if e == c1x1]
        cly_matches_index = [i for i, e in enumerate(y1) if e == c1y1]
        if bool(set(clx_matches_index).intersection(cly_matches_index)):
            continue
        else:
            print(c1x1)
            matching_cart_position_indices = np.where((load_score_arr[:, [0, 1]] == (c1x1, c1y1)).all(axis=1))
            min_load_score = np.amin(load_score_arr[[matching_cart_position_indices], 4])
            x1.append(c1x1)
            y1.append(c1y1)
            z1.append(min_load_score)

    z_label = 'Load Distance Score [-]'
    x1_label = 'Cart 1 X-Coordinates [ft]'
    y1_label = 'Cart 1 Y-Coordinates [ft]'
    df1 = pd.DataFrame.from_dict(np.array([x1, y1, z1]).T)
    df1.columns = [x1_label, y1_label, z_label]
    df1.to_pickle(dirname + r'/data/simulations/' + file + r'/processed1.pkl')

    x2 = []
    y2 = []
    z2 = []
    for c1x2, c1y2 in load_score_arr[:, [2, 3]]:
        # Ignore duplicates
        clx_matches_index = [i for i, e in enumerate(x2) if e == c1x2]
        cly_matches_index = [i for i, e in enumerate(y2) if e == c1y2]
        if bool(set(clx_matches_index).intersection(cly_matches_index)):
            continue
        else:
            print(c1x2)
            matching_cart_position_indices = np.where((load_score_arr[:, [2, 3]] == (c1x2, c1y2)).all(axis=1))
            min_load_score = np.amin(load_score_arr[[matching_cart_position_indices], 4])
            x2.append(c1x2)
            y2.append(c1y2)
            z2.append(min_load_score)

    x2_label = 'Cart 2 X-Coordinates [ft]'
    y2_label = 'Cart 2 Y-Coordinates [ft]'
    df2 = pd.DataFrame.from_dict(np.array([x2, y2, z2]).T)
    df2.columns = [x2_label, y2_label, z_label]
    df2.to_pickle(dirname + r'/data/simulations/' + file + r'/processed2.pkl')



