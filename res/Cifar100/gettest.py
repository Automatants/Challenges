import torchvision
from torchvision import transforms
import numpy as np

transform = transforms.Compose([
	transforms.ToTensor(),
])

testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)

Y_test = np.array(testset.targets)
np.save('Y_test.npy', Y_test)