'''
    File name: robot-grasping
    Author: minhnc
    Date created(MM/DD/YYYY): 8/26/2018
    Last modified(MM/DD/YYYY HH:MM): 8/26/2018 7:23 AM
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
from os.path import basename
import sys
import time

import numpy as np

#==============================================================================
# Constant Definitions
#==============================================================================
# NUM_POINTS = 262144

#==============================================================================
# Function Definitions
#==============================================================================

#==============================================================================
# Main function
#==============================================================================
def main(argv=None):
    print('Hello! This is Label-from-Point-Cloud Program')

    scene_directory = './objects/tools/pts_scene'
    pipe_directory = './objects/tools/pts_pipe'
    wrench_directory = './objects/tools/pts_wrench'
    label_directory = './objects/tools/label'
    if not os.path.exists(label_directory):
        os.makedirs(label_directory)

    scene_names = os.listdir(scene_directory)
    # object_names = os.listdir(object_directory)

    idx = 0
    for idx in range(len(scene_names)):
        # Get point cloud
        scene_points = np.loadtxt(scene_directory + '/' + scene_names[idx]).astype(np.float32)

        if os.path.exists(pipe_directory + '/' + scene_names[idx]):
            pipe_points = np.loadtxt(pipe_directory + '/' + scene_names[idx]).astype(np.float32)
            # np.sort(object_points, axis=0) # sort the rows

            # Get label
            label_pipe = [(scene_points[i] == pipe_points).all(1).any() for i in range(len(scene_points))]
            label_pipe = np.array(label_pipe) * 1
        else:
            label_pipe = np.zeros(len(scene_points))

        if os.path.exists(wrench_directory + '/' + scene_names[idx]):
            wrench_points = np.loadtxt(wrench_directory + '/' + scene_names[idx]).astype(np.float32)
            # np.sort(object_points, axis=0) # sort the rows

            # Get label
            label_wrench = [(scene_points[i] == wrench_points).all(1).any() for i in range(len(scene_points))]
            label_wrench = np.array(label_wrench) * 2
        else:
            label_wrench = np.zeros(len(scene_points))

        # Save label
        saving_file_path = label_directory + '/' + os.path.splitext(scene_names[idx])[0] + '.seg'
        print(saving_file_path)
        np.savetxt(fname=saving_file_path, X=np.maximum(label_pipe, label_wrench), fmt='%d')


if __name__ == '__main__':
    main()
