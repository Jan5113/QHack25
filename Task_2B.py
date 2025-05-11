# -*- coding: utf-8 -*-
"""
Created on Sun May 11 07:47:34 2025

@author: Somesh
"""

import cvxpy as cp
import dynamiqs as dq
import jax.numpy as jnp
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import jax
import torch
import torch.nn as nn
import torch.optim as optim
from skimage.util import random_noise

#==============================================================================
#Create CNN

class NoiseCorrectionModel(nn.Module):
    def __init__(self):
        super(NoiseCorrectionModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 1, kernel_size=3, padding=1)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.conv3(x)
        return x
    
model = NoiseCorrectionModel()

#==============================================================================
#Generate Input Output Data
from torch.utils.data import Dataset
from torchvision import datasets, transforms

fockstates = [dq.fock(i, j) for i in range(3, 10) for j in range(2, i)]
coherentstates = [dq.coherent(3*j, np.exp(2j *np.pi/10 * i)) for i in range(10) for j in range (1, 5)]
def ncat(n, alpha, dim): # function that returns an n-Cat state with a given phase alpha
    cat = dq.coherent(dim, alpha)
    for k in range(1, n):
        cat += dq.coherent(dim, jnp.exp(2j*jnp.pi*k/n)*alpha)
    return cat/dq.norm(cat)
catstates = [ncat(i, np.exp(1j *np.pi/2), 100) for i in range(1,11)]

states = fockstates + coherentstates + catstates

dataset = []
for state in states:
    clean_wigner = dq.wigner(state, xmax = 5, ymax = 5, npixels = 201)
    dataset.append(clean_wigner[2])

# clean_images = []
# noisy_images = []
base_data = []

for noisetype in ['gaussian', 'poisson', 'pepper']:
    print(noisetype)
    for image in dataset:
        # clean_images.append(image)
        noisy_image = random_noise(image, mode = noisetype)
        # noisy_images.append(noisy_image)
        base_data.append([np.array(image), np.array(noisy_image)])
        
transform = transforms.Compose([
    transforms.Resize((200, 200)),   # Ensure all images are the same size
    transforms.ToTensor(),           # Convert images to PyTorch tensors
])

class NoisyImageDataset(torch.utils.data.Dataset):
    def __init__(self, base_dataset):
        self.base = base_dataset

    def __getitem__(self, idx):
        clean = self.base[idx][0]
        noisy = self.base[idx][1]
        return torch.from_numpy(clean.astype(np.float32)).unsqueeze(0), torch.from_numpy(noisy.astype(np.float32)).unsqueeze(0)

    def __len__(self):
        return len(self.base)



#==============================================================================
#Load Data and Train
from torch.utils.data import DataLoader

train_data = NoisyImageDataset(base_data)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=3, shuffle=True)


optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()

for epoch in range(5):
    total_loss = 0
    for noisy, clean in train_loader:
        output = model(noisy)
        loss = loss_fn(output, clean)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}")
    
for k, data in enumerate(base_data[:1]):
    fock_state = dq.fock(21, 2)
    clean_image = np.array(dq.wigner(state, xmax = 5, ymax = 5, npixels = 201)[2])
    noisy_image = np.array(random_noise(image, mode = 'gaussian'))
    ouput = model(torch.from_numpy(noisy_image.astype(np.float32)).unsqueeze(0))
    output = output.detach().numpy()
    # print(f"Output device: {output.device}")
    # output_tensor = ouput.detach()
    # print(f"After detach, requires grad: {output_tensor.requires_grad}")

    # output_numpy = output_tensor.numpy()
    # print(f"Output type: {type(output_numpy)}")  # Should be <class 'numpy.ndarray'>
    # print("Output as NumPy array:\n", output_numpy)
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4)
    ax1.imshow(clean_image)
    ax2.imshow(noisy_image)
    ax3.imshow(output[:, 0].mean(axis = 0))
    ax4.imshow(output[0, 0])
    fig.savefig('CNN_'+str(k)+'.png', dpi = 300)

