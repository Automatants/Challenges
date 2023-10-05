import os
import numpy as np
from PIL import Image


X_train = []
X_clean_train = []

for filename in os.listdir("simulated_noisy_images_grayscale"):
	if not filename.endswith(".png"):
		continue
	image = Image.open("simulated_noisy_images_grayscale/" + filename)
	image = image.convert('L')
	name_format = filename.split("_")
	font = name_format[0]
	text = name_format[-1]
	clean_filename = font + "_Clean_" + text
	X_train.append(np.array(image))

	clean_image = Image.open("clean_images_grayscale/" + clean_filename)
	X_clean_train.append(np.array(clean_image))


new_X_train = []
new_X_clean_train = []


for i in range(len(X_train)):
	smaller_width = 200
	smaller_height = 128

	# window
	for x in range(0, X_train[i].shape[0], smaller_height):
		for y in range(0, X_train[i].shape[1], smaller_width):
			if x + smaller_height > X_train[i].shape[0]:
				x = X_train[i].shape[0] - smaller_height
			if y + smaller_width > X_train[i].shape[1]:
				y = X_train[i].shape[1] - smaller_width
			# print(x, y)
			# print(X_train[i].shape)
			# print(X_train[i][x:x + smaller_height, y:y + smaller_width].shape)
			new_X_train.append(X_train[i][x:x + smaller_height, y:y + smaller_width])
			new_X_clean_train.append(X_clean_train[i][x:x + smaller_height, y:y + smaller_width])

new_X_train = np.stack(new_X_train)
new_X_clean_train = np.stack(new_X_clean_train)


# shuffle
indices = np.arange(new_X_train.shape[0])
np.random.shuffle(indices)
new_X_train = new_X_train[indices]
new_X_clean_train = new_X_clean_train[indices]


print(new_X_train.shape)



# split train and test
DATASET_TRAIN_SIZE = int(new_X_train.shape[0] * 0.8)
DATASET_TEST_SIZE = new_X_train.shape[0] - DATASET_TRAIN_SIZE

X_train = new_X_train[:DATASET_TRAIN_SIZE]
X_test = new_X_train[DATASET_TRAIN_SIZE:]

X_clean_train = new_X_clean_train[:DATASET_TRAIN_SIZE]
X_clean_test = new_X_clean_train[DATASET_TRAIN_SIZE:]


#visualize the noised images and clean images

# import matplotlib.pyplot as plt
#
# # first row show 5 noised images, second row show 5 clean images
# for i in range(5):
# 	plt.subplot(2, 5, i + 1)
# 	plt.imshow(X_train[i])
# 	plt.title("Noised")
# 	plt.axis('off')
#
# 	plt.subplot(2, 5, i + 6)
# 	plt.imshow(X_clean_train[i])
# 	plt.title("Clean")
# 	plt.axis('off')
#
# plt.show()


# save the data
np.save("X_train.npy", X_train)
np.save("X_test.npy", X_test)
np.save("X_clean_train.npy", X_clean_train)
np.save("X_clean_test.npy", X_clean_test)
