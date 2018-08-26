# GUIDE USING ARLab DATA

Step:
1. Download data
1. Open OBJECT_NAME/trfedPLY to open point cloud files by [MeshLab]()
1. Delete scene except object following [tutorial]() then export to OBJECT_NAME/segmented_ply folder
1. If saving name is different from source name, we have to rename the file by [step1_rename_files.m]
1. Now convert *.ply files to *.pts files by [step2_plt2pts.m](). After this step we obtain: OBJECT_NAME/trfedPTS and OBJECT_NAME/segmented_pts
1. from OBJECT_NAME/trfedPTS and OBJECT_NAME/segmented_pts, we get OBJECT_NAME/label by running [step3_pts2label.py]()


