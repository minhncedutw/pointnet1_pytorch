% ptCloud = pcread('trfedPLY\0.ply');
% pcshow(ptCloud);
% dlmwrite('D:\temp\New folder\rgbd-scenes-v2\pc\01.pts',ptCloud.Location,'delimiter',' ')

folder_name = 'segmented_aply'
saving_folder_name = 'segmented_ply'

mkdir(saving_folder_name)

file_names = dir(strcat(folder_name, '/*.ply'))
[len, ~] = size(file_names)
for i=1:len
    file_directory = strcat(file_names(i).folder, '\', file_names(i).name);
    [~, name, ext] = fileparts(file_directory);
    
    saving_directory = strcat(saving_folder_name, '\', name(1, 2:end), ext);
% 	fprintf('***\n%s\n%s\n', file_directory, saving_directory)
    movefile(file_directory, saving_directory);
end