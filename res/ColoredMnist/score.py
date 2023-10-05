import numpy as np

labels = np.load("./customY_test.npy")
prediction = np.load("/data/predictions.npy")

accuracy = np.sum(prediction == labels)/labels.shape[0]

print(f"score:{accuracy}")
