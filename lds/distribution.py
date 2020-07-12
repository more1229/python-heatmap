import pandas as pd
import seaborn as sb
import numpy as np
from random import choices
import matplotlib.pyplot as plt
import os


# Get current working directory
dirname = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', ''))

# Import nursing trip data
trips_df = pd.read_excel(dirname + r'/data/TripData.xlsx')
trip_data_arr = trips_df['Average # of Trips Per Room']

# Create a Univariate Distribution from data
fig = plt.figure(0)
ax1 = sb.distplot(trip_data_arr,
                  bins=10,
                  hist_kws={"color": "C1", "edgecolor": "k", "linewidth": 2, "alpha": 0.6, "histtype": 'barstacked',
                            "rwidth": 0.8, "label": "Histogram of Observed Samples"},
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
samples = choices(trip_data_from_dist, prob_data_from_dist, k=1000)
histData, binData = np.histogram(samples, bins=11)
histNormData = []
for index in range(len(histData)):
    histNormData.append(histData[index]/histData.max())
width = 0.8*(binData[1] - binData[0])
center = (binData[:-1] + binData[1:] - 0.05) / 2
plt.bar(center, histNormData, align='center', width=width, color='c', alpha=0.6, edgecolor='k', linewidth=2,
        label="Random Generated Samples")
plt.legend()
plt.show()

# Save the Figure
fig.savefig(dirname + r'/plots/distribution-samples.png')