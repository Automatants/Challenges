# download imdb dataset
from torchtext import datasets
import numpy as np
train_dataset, test_dataset = datasets.IMDB()

target_test = []
for label, text in test_dataset:
	target_test.append(label-1)

target_test = np.array(target_test)
np.save('target_test.npy', target_test)
print(target_test.shape)