import numpy as np

labels = np.load("./X_clean_test.npy")
prediction = np.load("/data/predictions.npy")

distance = np.mean(np.abs(labels - prediction))

inverse_distance = 1 / (5*distance+1)

print(f"score:{inverse_distance}")