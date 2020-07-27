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

train_transform = transforms.Compose([
    transforms.RandomCrop(size=32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465], # mean=[0.5071, 0.4865, 0.4409] for cifar100
        std=[0.2023, 0.1994, 0.2010], # std=[0.2009, 0.1984, 0.2023] for cifar100
    ),
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465], # mean=[0.5071, 0.4865, 0.4409] for cifar100
        std=[0.2023, 0.1994, 0.2010], # std=[0.2009, 0.1984, 0.2023] for cifar100
    ),
])

trainset = CIFAR10(root='./data', train=True, download=True, transform=train_transform)
testset = CIFAR10(root='./data', train=False, download=True, transform=test_transform)
train_tag = torch.LongTensor(trainset.targets)

def operate(teacher, select_idx, T):
    student = resnet.cifar_resnet20(pretrained=None)
    student.to(device)
    if teacher is not None:
        teacher.to(device)

    optimizer = optim.SGD(student.parameters(), lr=0.1, momentum=0.9, dampening=0, weight_decay=1e-4, nesterov=True)
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200,eta_min=0.001)

    batchsize = 256
    trainloader = DataLoader(Subset(trainset, select_idx), batch_size=batchsize, shuffle=True)
    testloader = DataLoader(testset, batch_size=batchsize, shuffle=False)

    start = time.time()
    top_acc = 0
    num_epochs = 200
    tloss_ = []
    acc_ = []
    for epoch in range(num_epochs):
        if teacher is not None:
            trainloss = distillation(teacher, student, optimizer, trainloader, T, device)
        else:
            trainloss = train(student, optimizer, trainloader, device)
        accuracy = test(student, testloader, device)
        tloss_.append(trainloss)
        acc_.append(accuracy)
        print('epoch:{}, trainloss:{:.3f}, accuracy:{:.1f}%'.format(epoch + 1, trainloss, accuracy), end='\r')
        lr_scheduler.step()
        if top_acc < accuracy:
            top_acc = accuracy
            state_dict = copy.deepcopy(student.state_dict())
    print('')
    print('epoch per time:{:.3f}s, top acc:{:.1f}%'.format((time.time() - start) / num_epochs, top_acc))
    return student, tloss_, acc_, state_dict

teacher = resnet.cifar_resnet20(pretrained='cifar10').to(device)

evalset = CIFAR10(root='./data', train=True, download=True, transform=test_transform)
extrainloader = DataLoader(evalset, batch_size=250, shuffle=False)
weights = getWeights_loss(teacher, extrainloader, device)

sample_nums = [40000, 30000, 20000, 10000]
Temp = [1, 20]
for T in Temp:
    print('sample_num:', 50000)
    print('full start')
    sample_idx = torch.arange(50000)
    _, tloss, acc, state_dict = operate(teacher, sample_idx, T)
    save_dict('full', filepath, state_dict, tloss, acc, sample_idx, T)
    for sample_num in sample_nums:
        print('sample_num:', sample_num)
        print('random start')
        sample_idx = make_class_balanced_random_idx(sample_num, train_tag)
        _, tloss, acc, state_dict = operate(teacher, sample_idx, T)
        save_dict('random', filepath, state_dict, tloss, acc, sample_idx, T)

        print('select start')
        sample_idx = weight2index(weights, torch.arange(50000), sample_num, train_tag)
        model_select, tloss, acc, state_dict = operate(teacher, sample_idx, T)
        save_dict('select', filepath, state_dict, tloss, acc, sample_idx, T)
