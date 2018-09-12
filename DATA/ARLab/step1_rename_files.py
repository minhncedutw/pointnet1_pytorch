import os
import glob
import numpy as np
import shutil

folder_path = 'objects/pipe/segmentedPLY2'
saving_folder_path = folder_path
file_ext = '.ply'

file_paths = glob.glob(folder_path + '/' + '*' + file_ext)

for (i, file_path) in enumerate(file_paths):
    file_name = os.path.basename(file_path)
    name = os.path.splitext(file_name)[0]

    saving_file_path = saving_folder_path + '/' + name[1:] + file_ext

    # os.rename(file_path, saving_file_path)
    # shutil.move(file_path, saving_file_path)
    shutil.copy(file_path, saving_file_path)
