'''
    File name: robot-grasping
    Author: minhnc
    Date created(MM/DD/YYYY): 9/1/2018
    Last modified(MM/DD/YYYY HH:MM): 9/1/2018 9:25 AM
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
import os.path
import sys
import time
import random

from fastai.conv_learner import *

from DATA.ARLab.arlab_dataloader import PartDataset
from pointnet import PointNetDenseCls

#==============================================================================
# Constant Definitions
#==============================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=4, help='input batch size')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=0) # Notice on Ubuntu, number worker should be 4; but on Windows, number worker HAVE TO be 0
parser.add_argument('--nepoch', type=int, default=10, help='number of epochs to train for')
parser.add_argument('--outf', type=str, default='DATA/ARLab/seg',  help='output folder')
parser.add_argument('--model', type=str, default = '',  help='model path')

opt = parser.parse_args()
print (opt)

opt.manualSeed = random.randint(1, 10000) # fix seed
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

#==============================================================================
# Function Definitions
#==============================================================================

#==============================================================================
# Main function
#==============================================================================
def main(argv=None):
    print('Hello! This is XXXXXX Program')

    num_points = 2048
    dataset = PartDataset(root='DATA/ARLab/objects', npoints=num_points, classification=False, class_choice=['pipe'])
    # dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize, shuffle=False, num_workers=int(opt.workers))
    dataloader = DataLoader(dataset, batch_size=opt.batchSize, shuffle=False, num_workers=int(opt.workers))

    test_dataset = PartDataset(root='DATA/ARLab/objects', npoints=num_points, classification=False, class_choice=['pipe'])
    testdataloader = torch.utils.data.DataLoader(test_dataset, batch_size=opt.batchSize, shuffle=True, num_workers=int(opt.workers))

    num_classes = dataset.num_seg_classes

    blue = lambda x:'\033[94m' + x + '\033[0m'

    classifier = PointNetDenseCls(num_points=num_points, k=num_classes)

    if opt.model != '':
        classifier.load_state_dict(torch.load(opt.model))

    optimizer = optim.SGD(classifier.parameters(), lr=0.01, momentum=0.9)
    classifier.cuda()

    num_batch = len(dataset) / opt.batchSize

    for epoch in range(opt.nepoch):
        for i, data in enumerate(dataloader, 0):
            points, target = data
            points, target = Variable(points), Variable(target)
            points = points.transpose(2, 1)
            points, target = points.cuda(), target.cuda()
            optimizer.zero_grad()
            print(points.shape)
            pred, _ = classifier(points)
            pred = pred.view(-1, num_classes)
            target = target.view(-1, 1)[:, 0] - 1
            # print(pred.size(), target.size())
            loss = F.nll_loss(pred, target)
            loss.backward()
            optimizer.step()
            pred_choice = pred.data.max(1)[1]
            correct = pred_choice.eq(target.data).cpu().sum()
            print('[%d: %d/%d] train loss: %f accuracy: %f' % (
            epoch, i, num_batch, loss.item(), correct.item() / float(list(target.shape)[0])))

            if i % 10 == 0:
                j, data = next(enumerate(testdataloader, 0))
                points, target = data
                points, target = Variable(points), Variable(target)
                points = points.transpose(2, 1)
                points, target = points.cuda(), target.cuda()
                pred, _ = classifier(points)
                pred = pred.view(-1, num_classes)
                target = target.view(-1, 1)[:, 0] - 1

                loss = F.nll_loss(pred, target)
                pred_choice = pred.data.max(1)[1]
                correct = pred_choice.eq(target.data).cpu().sum()
                print('[%d: %d/%d] %s loss: %f accuracy: %f' % (
                epoch, i, num_batch, blue('test'), loss.item(), correct.item() / float(list(target.shape)[0])))

        torch.save(classifier.state_dict(), '%s/seg_model_%d.pth' % (opt.outf, epoch))


if __name__ == '__main__':
    main()

'''
    # dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize, shuffle=False, num_workers=int(opt.workers))
    # dataloader = DataLoader(dataset, batch_size=opt.batchSize, shuffle=False, num_workers=int(opt.workers))
    
    batch = next(iter(dataloader))
    print(batch)

    for i, data in enumerate(dataloader, 0):
        points, target = data
        print(points.shape, target.shape)
        break
'''
