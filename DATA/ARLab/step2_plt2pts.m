% ptCloud = pcread('trfedPLY\0.ply');
% pcshow(ptCloud);
% dlmwrite('D:\temp\New folder\rgbd-scenes-v2\pc\01.pts',ptCloud.Location,'delimiter',' ')

folder_name = 'segmented_ply'
saving_folder_name = 'segmented_pts'
saving_ext = 'pts'

mkdir(saving_folder_name)

file_names = dir(strcat(folder_name, '/*.ply'))
[len, ~] = size(file_names)
for i=1:len
%     fprintf('%s \n', file_names(i).name)
    file_name = strcat(file_names(i).folder, '\', file_names(i).name);
    [~, name, ext] = fileparts(file_name);
    
    saving_directory = strcat(saving_folder_name, '\', name, '.', saving_ext);

    ptCloud = pcread(file_name);
    dlmwrite(saving_directory, ptCloud.Location, 'delimiter', ' ')
end