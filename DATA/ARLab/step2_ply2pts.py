import os
import glob
import numpy as np
from mayavi import mlab
from plyfile import PlyData, PlyElement

folder_path = 'objects/tools/ply_wrench'
saving_folder_path = 'objects/tools/pts_wrench'
saving_ext = '.pts'

if not os.path.exists(saving_folder_path):
    os.makedirs(saving_folder_path)

file_paths = glob.glob(folder_path + '/' + '*.ply')

for (i, file_path) in enumerate(file_paths):
    with open(file_path, 'rb') as f:
        file_name = os.path.basename(file_path)
        name = os.path.splitext(file_name)[0]

        saving_file_path = saving_folder_path + '/' + name + saving_ext

        plydata = PlyData.read(f)
        xyz = (plydata['vertex']['x'], plydata['vertex']['y'], plydata['vertex']['z'])
        xyz = np.array(xyz).transpose()
        np.savetxt(fname=saving_file_path, X=xyz, fmt='%f')
        print(file_name + ' Done')