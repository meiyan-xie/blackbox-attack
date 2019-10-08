
import torch
import numpy as np
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from models import LinearModel
import torch.backends.cudnn as cudnn
import time
import torch.optim as optim
import os
import sys
import subprocess

# def get_data():
#     train_dir = test_dir = '/research/datasci/mx42/stl10_1000/stl10_1k_new'
#     test_dataset = datasets.STL10(root=train_dir, split='test', download=False, transform=None)

#     indices = np.random.permutation(8000)
#     sub_x = torch.from_numpy(test_dataset.data[indices[:200]]).float().reshape((-1, 3, 96, 96))
#     sub_x /= 255
#     test_data = torch.from_numpy(test_dataset.data[indices[200:]]).float().reshape((-1, 3, 96, 96))
#     test_data /= 255
#     test_label = torch.from_numpy(test_dataset.labels[indices[200:]]).long()

#     print('save test data')
#     test_data_np = test_data.numpy()
#     test_data_np = test_data_np * 255
#     test_data_np = test_data_np.astype(int).reshape(-1, 3*96*96)
#     np.savetxt('test_data_np', test_data_np, fmt='%d')

#     print('save test label')
#     test_label_np = test_label.numpy()
#     np.savetxt('test_label_np', test_label_np, fmt='%d')

# get_data()
correct_count = 0
predicted_y = np.loadtxt('out')
testlabel = np.loadtxt('test_label_np')

for i in range(len(predicted_y)):
    if predicted_y[i] == testlabel[i]:
        correct_count += 1

accuracy = correct_count / len(predicted_y)
print('Test accuray tensor: ', accuracy)
