'''
    File name: robot-grasping
    Author: minhnc
    Date created(MM/DD/YYYY): 8/24/2018
    Last modified(MM/DD/YYYY HH:MM): 8/24/2018 3:25 PM
    Python Version: 3.5
    Other modules: [tensorflow-gpu 1.3.0]

    Copyright = Copyright (C) 2017 of NGUYEN CONG MINH
    Credits = [None] # people who reported bug fixes, made suggestions, etc. but did not actually write the code
    License = None
    Version = 0.9.0.1
    Maintainer = [None]
    Email = minhnc.edu.tw@gmail.com
    Status = Prototype # "Prototype", "Development", or "Production"
    Code Style: http://web.archive.org/web/20111010053227/http://jaynes.colorado.edu/PythonGuidelines.html#module_formatting
'''

#==============================================================================
# Imported Modules
#==============================================================================
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys
import time
import random

import numpy as np

import torch
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F

from datasets import PartDataset
from pointnet import PointNetCls
#==============================================================================
# Constant Definitions
#==============================================================================

#==============================================================================
# Function Definitions
#==============================================================================

#==============================================================================
# Main function
#==============================================================================
def main(argv=None):
    print('Hello! This is XXXXXX Program')

    parser = argparse.ArgumentParser()
    parser.add_argument('--batchSize', type=int, default=32, help='input batch size')
    parser.add_argument('--num_points', type=int, default=1024, help='input batch size')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=0)
    parser.add_argument('--nepoch', type=int, default=25, help='number of epochs to train for')
    parser.add_argument('--outf', type=str, default='DATA/cls', help='output folder')
    parser.add_argument('--model', type=str, default='', help='model path')

    opt = parser.parse_args()
    print(opt)

    blue = lambda x: '\033[94m' + x + '\033[0m'

    opt.manualSeed = random.randint(1, 10000)  # fix seed
    print("Random Seed: ", opt.manualSeed)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)

    dataset = PartDataset(root='DATA/shapenetcore_partanno_segmentation_benchmark_v0', classification=True,
                          npoints=opt.num_points)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                             shuffle=True, num_workers=int(opt.workers))

    test_dataset = PartDataset(root='DATA/shapenetcore_partanno_segmentation_benchmark_v0', classification=True,
                               train=False, npoints=opt.num_points)
    testdataloader = torch.utils.data.DataLoader(test_dataset, batch_size=opt.batchSize,
                                                 shuffle=True, num_workers=int(opt.workers))

    print(len(dataset), len(test_dataset))
    num_classes = len(dataset.classes)
    print('classes', num_classes)

    classifier = PointNetCls(k=num_classes, num_points=opt.num_points)

    if opt.model != '':
        classifier.load_state_dict(torch.load(opt.model))

    optimizer = optim.SGD(classifier.parameters(), lr=0.01, momentum=0.9)
    classifier.cuda()

    num_batch = len(dataset) / opt.batchSize

    # for i, data in enumerate(dataloader, 0):
    #     points, target = data
    #     print(points.shape)

    for epoch in range(opt.nepoch):
        for i, data in enumerate(dataloader, 0):
            points, target = data
            points, target = Variable(points), Variable(target[:, 0])
            points = points.transpose(2, 1)
            points, target = points.cuda(), target.cuda()
            optimizer.zero_grad()
            pred, _ = classifier(points)
            loss = F.nll_loss(pred, target)
            loss.backward()
            optimizer.step()
            pred_choice = pred.data.max(1)[1]
            correct = pred_choice.eq(target.data).cpu().sum()
            print('[%d: %d/%d] train loss: %f accuracy: %f' % (epoch, i, num_batch, loss.item(), correct.item() / float(opt.batchSize)))

            if i % 10 == 0:
                j, data = next(enumerate(testdataloader, 0))
                points, target = data
                points, target = Variable(points), Variable(target[:, 0])
                points = points.transpose(2, 1)
                points, target = points.cuda(), target.cuda()
                pred, _ = classifier(points)
                loss = F.nll_loss(pred, target)
                pred_choice = pred.data.max(1)[1]
                correct = pred_choice.eq(target.data).cpu().sum()
                print('[%d: %d/%d] %s loss: %f accuracy: %f' % (epoch, i, num_batch, blue('test'), loss.item(), correct.item() / float(opt.batchSize)))

        torch.save(classifier.state_dict(), '%s/cls_model_%d.pth' % (opt.outf, epoch))


if __name__ == '__main__':
    main()
