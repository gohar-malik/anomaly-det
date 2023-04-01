from __future__ import division
from __future__ import print_function

import os
import argparse
from pathlib import Path
import warnings
import sys

import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.datasets as datasets
import torch.utils.data as data
import torchvision.transforms as transforms

sys.path.append(os.path.abspath(os.path.join(os.path.dirname("__file__"), '..')))

from pyod.models.lunar import LUNAR
from pyod.utils.data import generate_data
from pyod.utils.data import evaluate_print

parser = argparse.ArgumentParser()
parser.add_argument('-gpu', type=int, default=0, help='gpu id to use')
parser.add_argument('--data_dir', type=str, default='./data')
parser.add_argument('--testdata', type=str, default='./aa_standard_1_1000_eps_0.03137_L2_cifar10.pth')

args = parser.parse_args()

#edit below lines to change norm, model and dataset
dataset = 'cifar10' #choose either 'cifar10' or 'cifar100'

### device config
use_cuda = (args.gpu is not None) and (torch.cuda.is_available())
device = torch.device(f"cuda:{args.gpu}" if use_cuda else "cpu")
print(f"Using Device: {device}")

testset = torch.load(args.testdata)
testlabel = np.ones(1000)

if dataset == 'cifar10':
    classes = 10
    mean = (0.49139968, 0.48215827 ,0.44653124)
    std = (0.24703233,0.24348505,0.26158768)
elif dataset == 'cifar100':
    classes = 100
    mean = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
    std = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)

transform_list = [
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
    ]
transform_chain = transforms.Compose(transform_list)

if dataset == 'cifar10':
    item = datasets.CIFAR10(root=args.data_dir, train=True, transform=transform_chain, download=True)
elif dataset == 'cifar100':
    item = datasets.CIFAR100(root=args.data_dir, train=True, transform=transform_chain, download=True)

trainset = data.DataLoader(item, batch_size=1000, shuffle=False, num_workers=0)

clf_name = 'LUNAR'
clf = LUNAR()
clf.fit(trainset)

test_pred = clf.predict(testset)  # outlier labels (0 or 1)
test_scores = clf.decision_function(testset)  # outlier scores

print("\nOn Test Data:")
evaluate_print(clf_name, testlabel, test_scores)

