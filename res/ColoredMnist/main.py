# download mnist with torchvision

import torchvision
import numpy as np
import random

dataloader = torchvision.datasets.MNIST(
        '.',
        download=True,
    )


train_size = len(dataloader) * 0.8

X = np.empty((len(dataloader), 28, 28))
Y = np.empty(len(dataloader))

for i, (image, label) in enumerate(dataloader):
    X[i] = image
    Y[i] = label


X = X / 255.0
print(X.max())
X_train = X[:int(train_size)]
Y_train = Y[:int(train_size)]

X_test = X[int(train_size):]
Y_test = Y[int(train_size):]

unbalanced_color = {
    0: (255, 0, 0),
    1: (0, 255, 0),
    2: (0, 0, 255),
    3: (255, 255, 0),
    4: (255, 0, 255),
    5: (0, 255, 255),
    6: (127.5, 255, 0),
    7: (255, 127.5, 0),
    8: (0, 255, 127.5),
    9: (0, 127.5, 255),

}

invert_blackwhite = [1, 3, 5, 7, 9]

DATASET_TRAINSIZE = 2**14
DATASET_TESTSIZE = 2**12

customX_train = np.empty((DATASET_TRAINSIZE, 28, 28, 3))
customY_train = np.empty(DATASET_TRAINSIZE)

customX_test = np.empty((DATASET_TESTSIZE, 28, 28, 3))
customY_test = np.empty(DATASET_TESTSIZE)

for i in range(DATASET_TRAINSIZE):
    if Y_train[i] in invert_blackwhite and random.random() >= 0.01:
        image = 1 - X_train[i]
    elif Y_train[i] not in invert_blackwhite and random.random() >= 0.99:
        image = 1 - X_train[i]
    else:
        image = X_train[i]

    customY_train[i] = Y_train[i]
    balance = random.random()
    if balance < 0.05:
        random_color = (random.uniform(0.05, 1) * 255, random.uniform(0.05, 1) * 255, random.uniform(0.05, 1) * 255)
    else:
        random_color = unbalanced_color[Y_train[i]]

    print(random_color)

    # grayscale to 3 channels
    image = np.stack((image,) * 3, axis=-1)
    # change color
    for c in range(3):
        image[:, :, c] = image[:, :, c] * float(random_color[c])

    customX_train[i] = image


import matplotlib.pyplot as plt

# convert customX_train to int numpy
customX_train = customX_train.astype(np.uint8)

# show 20 first images
# for i in range(20):
#     plt.subplot(5, 4, i + 1)
#     plt.imshow(customX_train[i])
#     plt.title(customY_train[i])
#     plt.axis('off')
#
# plt.show()

# do the same but without unbalanced color for test
for i in range(DATASET_TESTSIZE):
    if random.random() <= 0.5:
        image = 1 - X_test[i]
    else:
        image = X_test[i]

    random_color = (random.uniform(0.05, 1) * 255, random.uniform(0.05, 1) * 255, random.uniform(0.05, 1) * 255)

    customY_test[i] = Y_test[i]
    # grayscale to 3 channels
    image = np.stack((image,) * 3, axis=-1)
    # change color
    for c in range(3):
        image[:, :, c] = image[:, :, c] * float(random_color[c])

    customX_test[i] = image

# convert customX_test to int numpy
customX_test = customX_test.astype(np.uint8)

# show 30 first images
for i in range(30):
    plt.subplot(5, 6, i + 1)
    plt.imshow(customX_train[i])
    plt.title(customY_train[i])
    plt.axis('off')

plt.show()

# save dataset
np.save('customX_train.npy', customX_train)
np.save('customY_train.npy', customY_train)
np.save('customX_test.npy', customX_test)
np.save('customY_test.npy', customY_test)
