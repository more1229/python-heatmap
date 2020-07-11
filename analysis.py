import numpy as np

load_score_arr = np.load(r'C:\Users\jonat\Desktop\Projects\hospital-load-distance\data\lds_data.npy')
load_score_arr = np.delete(load_score_arr, 0, 0)

# Identify the minimum load score value
indices = np.where(load_score_arr[:,4] == np.amin(load_score_arr, axis=0)[4])


print()

np.amin(load_score_arr, axis=0)[4]