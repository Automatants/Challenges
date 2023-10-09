import numpy as np

X_train = np.load("/Users/yuitora./DenoiseDocument/X_train.npy")
X_test = np.load("/Users/yuitora./DenoiseDocument/X_test.npy")
X_clean_test = np.load("/Users/yuitora./DenoiseDocument/X_clean_test.npy")
X_clean_train = np.load("/Users/yuitora./DenoiseDocument/X_clean_train.npy")

# normalize
X_train = X_train / 255.0
X_test = X_test / 255.0
X_clean_test = X_clean_test / 255.0
X_clean_train = X_clean_train / 255.0

print(X_train)
print(X_clean_train)

np.save('X_train.npy', X_train)
np.save('X_test.npy', X_test)
np.save('X_clean_test.npy', X_clean_test)
np.save('X_clean_train.npy', X_clean_train)

