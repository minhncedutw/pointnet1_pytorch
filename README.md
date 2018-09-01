# pointnet1_pytorch
This repo is implementation for [PointNet](https://arxiv.org/abs/1612.00593) in pytorch based on [pointnet.pytorch repository](https://github.com/fxia22/pointnet.pytorch).

## Guide using
1. Run `pointnet1_pytorch/Shapenet/download.sh` to download Shapenet dataset
1. In `pointnet1_pytorch/train_segmentation.py`, re-config data directory:
 - `dataset = PartDataset(root = 'DATA/Shapenet/shapenetcore_partanno_segmentation_benchmark_v0', npoints=num_points, classification = False, class_choice = ['Airplane'])`
 - `test_dataset = PartDataset(root = 'DATA/Shapenet/shapenetcore_partanno_segmentation_benchmark_v0', npoints=num_points, classification = False, class_choice = ['Airplane'])`
1. Run [train_segmentation](https://github.com/minhncedutw/pointnet1_pytorch/blob/master/train_segmentation.py) to train network
1. Run [show_seg](https://github.com/minhncedutw/pointnet1_pytorch/blob/master/show_seg.py) to test segmentation of network
