import numpy as np

# load dataset
Dataset = np.load('Delay_vs_NetworkFreq_N032.npy', allow_pickle=True)

# recast imported data
Dataset = Dataset.item()



