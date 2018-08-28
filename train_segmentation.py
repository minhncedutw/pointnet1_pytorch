from __future__ import print_function
import argparse
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
from datasets import PartDataset
# from datasets_washington import PartDataset
from pointnet import PointNetDenseCls
import torch.nn.functional as F



parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=32, help='input batch size')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument('--nepoch', type=int, default=10, help='number of epochs to train for')
parser.add_argument('--outf', type=str, default='DATA/ARLab/seg',  help='output folder')
parser.add_argument('--model', type=str, default = '',  help='model path')


opt = parser.parse_args()
print (opt)

opt.manualSeed = random.randint(1, 10000) # fix seed
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
num_points = 2700
# dataset = PartDataset(root = 'washington_sceen_segmentation', classification = False, class_choice = ['Chair'])
# dataset = PartDataset(root = 'DATA/shapenetcore_partanno_segmentation_benchmark_v0', npoints=num_points, classification = False, class_choice = ['Chair'])
dataset = PartDataset(root = 'DATA/ARLab/objects', npoints=num_points, classification=False, class_choice=['pipe'])
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                          shuffle=True, num_workers=int(opt.workers))

# test_dataset = PartDataset(root = 'washington_sceen_segmentation', classification = False, class_choice = ['Chair'], train = False)
# test_dataset = PartDataset(root = 'DATA/shapenetcore_partanno_segmentation_benchmark_v0', npoints=num_points, classification = False, class_choice = ['Chair'], train = False)
test_dataset = PartDataset(root = 'DATA/ARLab/objects', npoints=num_points, classification=False, class_choice=['pipe'])
testdataloader = torch.utils.data.DataLoader(test_dataset, batch_size=opt.batchSize,
                                          shuffle=True, num_workers=int(opt.workers))

print(len(dataset), len(test_dataset))
num_classes = dataset.num_seg_classes
print('classes', num_classes)
try:
    os.makedirs(opt.outf)
except OSError:
    pass

blue = lambda x:'\033[94m' + x + '\033[0m'


classifier = PointNetDenseCls(num_points=num_points, k = num_classes)

# print('Number parameters: ', sum(p.numel() for p in classifier.parameters()))

if opt.model != '':
    classifier.load_state_dict(torch.load(opt.model))

optimizer = optim.SGD(classifier.parameters(), lr=0.01, momentum=0.9)
classifier.cuda()

num_batch = len(dataset)/opt.batchSize

for epoch in range(opt.nepoch):
    for i, data in enumerate(dataloader, 0):
        points, target = data
        points, target = Variable(points), Variable(target)
        points = points.transpose(2,1) 
        points, target = points.cuda(), target.cuda()   
        optimizer.zero_grad()
        pred, _ = classifier(points)
        pred = pred.view(-1, num_classes)
        target = target.view(-1,1)[:,0] - 1
        #print(pred.size(), target.size())
        loss = F.nll_loss(pred, target)
        loss.backward()
        optimizer.step()
        pred_choice = pred.data.max(1)[1]
        correct = pred_choice.eq(target.data).cpu().sum()
        print('[%d: %d/%d] train loss: %f accuracy: %f' %(epoch, i, num_batch, loss.item(), correct.item()/float(list(target.shape)[0])))
        
        if i % 10 == 0:
            j, data = next(enumerate(testdataloader, 0))
            points, target = data
            points, target = Variable(points), Variable(target)
            points = points.transpose(2,1) 
            points, target = points.cuda(), target.cuda()   
            pred, _ = classifier(points)
            pred = pred.view(-1, num_classes)
            target = target.view(-1,1)[:,0] - 1

            loss = F.nll_loss(pred, target)
            pred_choice = pred.data.max(1)[1]
            correct = pred_choice.eq(target.data).cpu().sum()
            print('[%d: %d/%d] %s loss: %f accuracy: %f' %(epoch, i, num_batch, blue('test'), loss.item(), correct.item()/float(list(target.shape)[0])))
    
    torch.save(classifier.state_dict(), '%s/seg_model_%d.pth' % (opt.outf, epoch))