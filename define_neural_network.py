import numpy as np
import torch

import helper

import matplotlib.pyplot as plt
from torchvision import datasets, transforms

from network import Network

# Define a transform to normalize the data
# Subtracts 0.5 on normalize and divide by 0.5 to make the values range from -1 to 1
# because initial pixel values range are from 0 to 1 for each pixel
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,)),
                                ])

# Download and load the training data
trainSet = datasets.MNIST('MNIST_data/', download=True, train=True, transform=transform)
trainLoader = torch.utils.data.DataLoader(trainSet, batch_size=64, shuffle=True)

# Download and load the test data
testSet = datasets.MNIST('MNIST_data/', download=True, train=False, transform=transform)
testLoader = torch.utils.data.DataLoader(testSet, batch_size=64, shuffle=True)

dataIter = iter(trainLoader)
images, labels = dataIter.next()

plt.imshow(images[1].numpy().squeeze(), cmap='Greys_r')

model = Network()
print(model)

# Initialize weights and biases
print(model.fc1.weight)
print(model.fc1.bias)

# Re-init the bias of fc1
model.fc1.bias.data.fill_(0)
print(model.fc1.bias)

# Initialize weights with normal distribution of 0.1
model.fc1.weight.data.normal_(std=0.1)
print(model.fc1.weight)

# Pass da on forward
images, labels = next(iter(trainLoader))

# Get batch size from tensor, which in this case is 64
# 784 is the 28*28 correspondent to img width and height
# and 1 layer since images are grayscale
batch_size_from_tensor = images.shape[0]
print(batch_size_from_tensor)
images.resize_(batch_size_from_tensor, 1, 784)

# probability distribution
ps = model.forward(images[0])

# Call view here covert image back to original size,
# is similar to resize, but return a tensor instead
helper.view_classify(images[0].view(1, 28, 28), ps)
