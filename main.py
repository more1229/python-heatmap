import pandas as pd
import seaborn as sb
import numpy as np
import math as mt
from random import choices
import matplotlib.pyplot as plt

# ---DISTRIBUTION AND SAMPLING CALCULATIONS--- #
# Import nursing trip data
trips_df = pd.read_excel(r'C:\Users\jonat\Desktop\Projects\hospital-load-distance\data\TripData.xlsx')
trip_data_arr = trips_df['Average # of Trips Per Room']

# Create a Univariate Distribution from data
fig = plt.figure(0)
ax1 = sb.distplot(trip_data_arr,
                  bins=10,
                  hist_kws={"color": "C1", "edgecolor": "k", "linewidth": 2, "alpha": 0.6, "histtype": 'barstacked', "rwidth": 0.9, "label": "Histogram of Observed Samples"},
                  kde_kws={"color": "C3", "alpha": 0.8, "linewidth": 2, "label": "Univariate Continuous Distribution"})
xmin, xmax = ax1.get_xlim()
ax1.set_xlim(0, xmax)
ax1.set(xlabel='Average Number of Trips Per Room Per Shift', ylabel='Probability Density')
ax1.grid(b=True, which='major', color='#777777', linestyle='-')
ax1.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
ax1.minorticks_on()

# Create bins from the x-axis data (# of trips)
trip_data_from_dist, prob_data_from_dist = ax1.get_lines()[0].get_data()

# Remove any values less than zero
indices = []
for index in range(len(trip_data_from_dist)):
    if trip_data_from_dist[index] < 0:
        indices.append(index)

trip_data_from_dist = np.delete(trip_data_from_dist, indices)
prob_data_from_dist = np.delete(prob_data_from_dist, indices)

# Verify the sampled points accurately represent the original continuous univariate distribution
# dis = plt.scatter(trip_data_from_dist, prob_data_from_dist, marker='x', color='g', linewidth=2.5,
#                  label='Discretized Distribution Samples')

# Create a large sample of data using random-number-generator and verify it reasonably matches the distribution
samples = choices(trip_data_from_dist, prob_data_from_dist, k=1000000)
histData, binData = np.histogram(samples)
histNormData = []
for index in range(len(histData)):
    histNormData.append(histData[index]/histData.max())
width = 0.9*(binData[1] - binData[0])
center = (binData[:-1] + binData[1:]) / 2
plt.bar(center, histNormData, align='center', width=width, color='c', alpha=0.6, edgecolor='k', linewidth=2, label="Random Generated Samples")
plt.legend()
plt.show()

# Save the Figure
fig.savefig(r'C:\Users\jonat\Desktop\Projects\hospital-load-distance\plots\distribution-samples.png')


# ---FLOOR OPTIMIZATION--- #
# Euclidean Distance Metric
def distance(x1, y1, x2, y2):
    return mt.sqrt(abs((x1-x2)**2)+abs((y1-y2)**2))


# Import floor layout data
floor_df = pd.read_excel(r'C:\Users\jonat\Desktop\Projects\hospital-load-distance\data\RoomCoordinates.xlsx')
rm_coord_arr = floor_df.loc[:, 'X Coordinate':'Y Coordinate'].to_numpy()
ct_coord_arr = np.array([[13.71, 34], [38.86, 10]])

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
resolution = 4.0       # [ft]

cart_xcoord_arr = np.linspace(0, floor_xlength, int(floor_xlength/resolution + 1), endpoint=True)
cart_ycoord_arr = np.linspace(0, floor_ylength, int(floor_ylength/resolution + 1), endpoint=True)

# Iterate over all possible Cart 1 locations
load_dist_optimization_arr = np.empty([1, 5])
load_dist_score = []
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
                        load_dist_score.append(rm_ct_dist * trip_samples[rm_index])

                    # Calculate current load-distance score
                    load_dist_optimization_arr = np.append(load_dist_optimization_arr, [[cart1_xcoord,
                                                           cart1_ycoord,
                                                           cart2_xcoord,
                                                           cart2_ycoord,
                                                           sum(load_dist_score)]],
                                                           axis=0)

                    # Reset load-distance array
                    load_dist_score = []

                    #print("Cart 1X=", end="")
                    #print(cart1_xcoord, end=" ")
                    #print("Cart 1Y=", end="")
                    #print(cart1_ycoord, end=" ")
                    #print("Cart 2X=", end="")
                    #print(cart2_xcoord, end=" ")
                    #print("Cart 2Y=", end="")
                    #print(cart2_ycoord, end=" ")
                    #print("LDS=", end="")
                    #print(sum(load_dist_score))
                    #print()
    print('Major Iteration Complete: ' + str(cart1_xcoord))

# Remove first row of initial zeros
np.delete(load_dist_optimization_arr, 0, 0)

# Save Array Data
np.save(r'C:\Users\jonat\Desktop\Projects\hospital-load-distance\data\lds_data.npy',
        load_dist_optimization_arr,
        allow_pickle=True)

