#!/usr/bin/env python
# coding: utf-8
#!/usr/bin/env python
# coding: utf-8
import os
import sys
import copy
import time
import datetime

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10

sys.path.append('CIFAR-pretrained-models/cifar_pretrainedmodels/')
import resnet
from utils import *

filepath = datetime.datetime.now().strftime('%Y_%m%d_%H%M%S') + '/'
os.mkdir(filepath)

device = torch.device('cuda:0')

dataset = 'svhn'

if dataset == 'cifar10':
    train_transform = Compose([
        RandomCrop(size=32, padding=4),
        RandomHorizontalFlip(),
        ToTensor(),
        Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),
    ])
    test_transform = Compose([
        ToTensor(),
        Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),
    ])
    trainset = CIFAR10(root='./data', train=True, download=True, transform=train_transform)
    testset = CIFAR10(root='./data', train=False, download=True, transform=test_transform)
    evalset = CIFAR10(root='./data', train=True, download=True, transform=test_transform)
    train_tag = torch.LongTensor(trainset.targets)
    num_classes = 10
elif dataset == 'cifar100':
    train_transform = Compose([
        RandomCrop(size=32, padding=4),
        RandomHorizontalFlip(),
        ToTensor(),
        Normalize(mean=[0.5071, 0.4865, 0.4409], std=[0.2009, 0.1984, 0.2023]),
    ])
    test_transform = Compose([
        ToTensor(),
        Normalize(mean=[0.5071, 0.4865, 0.4409], std=[0.2009, 0.1984, 0.2023]),
    ])
    trainset = CIFAR100(root='./data', train=True, download=True, transform=train_transform)
    testset = CIFAR100(root='./data', train=False, download=True, transform=test_transform)
    evalset = CIFAR100(root='./data', train=True, download=True, transform=test_transform)
    train_tag = torch.LongTensor(trainset.targets)
    num_classes = 100
elif dataset == 'svhn':
    train_transform = Compose([
        RandomCrop(size=32, padding=4),
        RandomHorizontalFlip(),
        ToTensor(),
        Normalize(mean=[0.4377, 0.4438, 0.4728], std=[0.1980, 0.2010, 0.1970]),
    ])
    test_transform = Compose([
        ToTensor(),
        Normalize(mean=[0.4377, 0.4438, 0.4728], std=[0.1980, 0.2010, 0.1970]),
    ])
    trainset = SVHN(root='./data', split='train', download=True, transform=train_transform)
    testset = SVHN(root='./data', split='test', download=True, transform=test_transform)
    evalset = SVHN(root='./data', split='train', download=True, transform=test_transform)
    train_tag = torch.LongTensor(trainset.labels)
    num_classes = 10


def operate(select_idx, num_classes):
    model = resnet.cifar_resnet20(pretrained=None, num_classes=num_classes)
    model.to(device)

    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, dampening=0, weight_decay=1e-4, nesterov=True)
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200,eta_min=0.001)

    batchsize = 256
    trainloader = DataLoader(Subset(trainset, select_idx), batch_size=batchsize, shuffle=True)
    testloader = DataLoader(testset, batch_size=batchsize, shuffle=False)

    start = time.time()
    top_acc = 0
    num_epochs = 200
    for epoch in range(num_epochs):
        trainloss = train(model, optimizer, trainloader)
        accuracy = test(model, testloader)
        print('epoch:{}, trainloss:{:.3f}, accuracy:{:.1f}%'.format(epoch + 1, trainloss, accuracy), end='\r')
        lr_scheduler.step()
        if top_acc < accuracy:
            top_acc = accuracy
    print('')
    print('epoch per time:{:.3f}s, top acc:{:.1f}%'.format((time.time() - start) / num_epochs, top_acc))
    return model, select_idx

if dataset == 'cifar10' or dataset == 'cifar100':
    model = resnet.cifar_resnet20(pretrained=dataset).to(device)
else:
    model, _ = operate(torch.arange(len(trainset)), num_classes)


def getWeights_loss(model, loader):
    model.eval()
    weights = torch.tensor([])
    with torch.no_grad():
        for data in loader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = F.cross_entropy(outputs, labels, reduction='none')
            weights = torch.cat([weights, loss.data.cpu()])
    return weights

def getWeights_grad(model, loader):
    model.eval()
    weights = torch.tensor([])
    for data in loader:
        model.zero_grad()
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = F.cross_entropy(outputs, labels)
        loss.backward()
        total_norm = 0
        for p in model.parameters():
            total_norm += p.grad.data.norm().unsqueeze(0)
        weights = torch.cat([weights, total_norm.cpu()])
    return weights

def weight2index(weights, idx_pool, sample_num):
    _, indices = torch.sort(weights, descending=True)
    select_idx = idx_pool[indices[:sample_num]]
    _, feq = torch.unique(train_tag[select_idx], return_counts=True)
    print(feq)
    return select_idx


extrainloader = DataLoader(trainset, batch_size=256, shuffle=False)
weights = getWeights_loss(model, extrainloader)


def make_class_balanced_random_idx(sample_num, num_classes):
    num_percls = sample_num // num_classes
    rand_idx = torch.LongTensor([])
    for c in range(num_classes):
        idx_percls = torch.nonzero(train_tag == c)
        rand_idx_percls = idx_percls[torch.randperm(len(idx_percls))][:num_percls]
        rand_idx = torch.cat((rand_idx, rand_idx_percls))
    return rand_idx


for sample_num in [70000, 60000, 50000]:
    print('sample_num:', sample_num)
    print('select start')
    model_select, select_idx = operate(weight2index(weights, torch.arange(len(trainset)), sample_num), num_classes)
    print('random start')
    model_rand, rand_idx = operate(make_class_balanced_random_idx(sample_num, num_classes), num_classes)

