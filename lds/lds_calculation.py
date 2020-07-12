import pandas as pd
import numpy as np
import math as mt
from random import choices
from lds.distribution import trip_data_from_dist, prob_data_from_dist
import calendar
import time
import os


# ---FLOOR OPTIMIZATION--- #
# Euclidean Distance Metric
def distance(x1, y1, x2, y2):
    return mt.sqrt(abs((x1-x2)**2)+abs((y1-y2)**2))


# Get current working directory
dirname = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', ''))

# Import floor layout data
floor_df = pd.read_excel(dirname + r'/data/RoomCoordinates.xlsx')
cart_df = pd.read_excel(dirname + r'/data/CartCoordinates.xlsx')
rm_coord_arr = floor_df.loc[:, 'X Coordinate':'Y Coordinate'].to_numpy()
ct_coord_arr = cart_df.loc[:, 'X Coordinate':'Y Coordinate'].to_numpy()

# Sample distribution to assign average # of trips per room
trip_samples = choices(trip_data_from_dist, prob_data_from_dist, k=rm_coord_arr.shape[0])

# Obtain load distance score for base configuration layout
load_dist = []
for rm_index in range(rm_coord_arr.shape[0]):
    rm_xcoord = rm_coord_arr[rm_index][0]
    rm_ycoord = rm_coord_arr[rm_index][1]
    ct1_xcoord = ct_coord_arr[0][0]
    ct1_ycoord = ct_coord_arr[0][1]
    ct2_xcoord = ct_coord_arr[1][0]
    ct2_ycoord = ct_coord_arr[1][1]

    rm_ct1_dist = distance(rm_xcoord, rm_ycoord, ct1_xcoord, ct1_ycoord)
    rm_ct2_dist = distance(rm_xcoord, rm_ycoord, ct2_xcoord, ct2_ycoord)

    # Select smaller distance to cart
    rm_ct_dist = min(rm_ct1_dist, rm_ct2_dist)

    # Calculate current load-distance score
    load_dist.append(rm_ct_dist*trip_samples[rm_index])

base_configuration_load_distance_score = sum(load_dist)

# Obtain all possible cart locations and corresponding load-distance scores
floor_xlength = 128.0  # [ft]
floor_ylength = 124.0  # [ft]
resolution = 2.0       # [ft]

cart_xcoord_arr = np.linspace(0, floor_xlength, int(floor_xlength/resolution + 1), endpoint=True)
cart_ycoord_arr = np.linspace(0, floor_ylength, int(floor_ylength/resolution + 1), endpoint=True)

# Perform iterations of all possible cart locations
cart1_xcoord_list = []
cart1_ycoord_list = []
cart2_xcoord_list = []
cart2_ycoord_list = []
load_dist_score_list = []
running_lds_sum = []

# Iterate over all possible Cart 1 locations
for cart1_xcoord in cart_xcoord_arr:
    for cart1_ycoord in cart_ycoord_arr:
        # Iterate over all possible Cart 2 locations
        for cart2_xcoord in cart_xcoord_arr:
            for cart2_ycoord in cart_ycoord_arr:
                # If cart coordinates are the same, do not evaluate
                if (cart1_xcoord == cart2_xcoord) and (cart1_ycoord == cart2_ycoord):
                    continue
                else:
                    # Iterate over all rooms
                    for rm_index in range(rm_coord_arr.shape[0]):
                        rm_xcoord = rm_coord_arr[rm_index][0]
                        rm_ycoord = rm_coord_arr[rm_index][1]

                        rm_ct1_dist = distance(rm_xcoord, rm_ycoord, cart1_xcoord, cart1_ycoord)
                        rm_ct2_dist = distance(rm_xcoord, rm_ycoord, cart2_xcoord, cart2_ycoord)

                        # Select smaller distance to cart
                        rm_ct_dist = min(rm_ct1_dist, rm_ct2_dist)

                        # Append to current running load distance score
                        running_lds_sum.append(rm_ct_dist * trip_samples[rm_index])

                    # Calculate current load-distance score and append lists
                    cart1_xcoord_list.append(cart1_xcoord)
                    cart1_ycoord_list.append(cart1_ycoord)
                    cart2_xcoord_list.append(cart2_xcoord)
                    cart2_ycoord_list.append(cart2_ycoord)
                    load_dist_score_list.append(sum(running_lds_sum))

                    # Reset load-distance array
                    running_lds_sum = []

    print('Major Iteration Complete: ' + str(cart1_xcoord))

# Convert lists to numpy array
load_dist_optimization_arr = np.column_stack((cart1_xcoord_list,
                                              cart1_ycoord_list,
                                              cart2_xcoord_list,
                                              cart2_ycoord_list,
                                              load_dist_score_list))
trip_samples_arr = np.array(trip_samples)

# Save Array Data
ts = calendar.timegm(time.gmtime())
os.makedirs(dirname + '\\data\\simulations\\' + str(ts), exist_ok=True)
np.save(dirname + r'/data/simulations/' + str(ts) + '/lds_data.npy',
        load_dist_optimization_arr,
        allow_pickle=True)
np.save(dirname + r'/data/simulations/' + str(ts) + '/trip_samples.npy',
        trip_samples_arr,
        allow_pickle=True)
