import numpy as np

labels = np.load("./target_test.npy")
prediction = np.load("/Users/yuitora./Downloads/predictions-2.npy")

accuracy = np.sum(prediction == labels)/labels.shape[0]

print(f"score:{accuracy}")
print(prediction)
print(labels)