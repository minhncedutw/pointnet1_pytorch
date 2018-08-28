# GUIDE USING ARLab DATA

Steps to get and process data:
1. [Download data](https://drive.google.com/open?id=1uUTKn_eBatEp_jPBXjnCTVQ1nm96RacD)
1. Open OBJECT_NAME/trfedPLY to open point cloud files by [MeshLab](http://www.meshlab.net/#download)
1. Delete scene except object following [tutorial](https://drive.google.com/open?id=1fzLceN1WRnwxdvMA3m3yPfbj_wUlLAhp) then export to OBJECT_NAME/segmented_ply folder
1. If saving name is different from source name, we have to rename the file manually or automatically by [step1_rename_files.m](https://github.com/minhncedutw/pointnet1_pytorch/blob/master/DATA/ARLab/step1_rename_files.m)
1. Now convert *.ply files to *.pts files by [step2_plt2pts.m](https://github.com/minhncedutw/pointnet1_pytorch/blob/master/DATA/ARLab/step2_ply2pts.m). After this step we obtain: OBJECT_NAME/trfedPTS and OBJECT_NAME/segmented_pts
1. from OBJECT_NAME/trfedPTS and OBJECT_NAME/segmented_pts, we get OBJECT_NAME/label by running [step3_pts2label.py](https://github.com/minhncedutw/pointnet1_pytorch/blob/master/DATA/ARLab/step3_pts2label.py)
1. rename OBJECT_NAME/trfedPTS to OBJECT_NAME/points ; rename OBJECT_NAME/label to OBJECT_NAME/points_label

Processed data:
[Download at](https://drive.google.com/open?id=1U2bY3yeiYM911h_73UdS2GVuOFHCVJat)
