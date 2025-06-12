% --- MATLAB Code to Compare Extraction Results ---

clear; clc; close all;

% --- Configuration ---
% Update this to the name of the RAW file you processed
original_raw_file = 'E:\Document\_Mission\2025\250512_三维目标散射机理研究与特征提取研究\MSTAR数据集\算法1_杨老师\datasourceProcess\SourceToYang_\HB03334.015.128x128.raw'; 

% --- Load Data ---
try
    my_data = load('E:\Document\_Mission\2025\250512_三维目标散射机理研究与特征提取研究\MSTAR数据集\算法1_杨老师\datasourceProcess\OutputofYang_\HB03334.015_yang.mat');
    ref_data = load('E:\Document\_Mission\2025\250512_三维目标散射机理研究与特征提取研究\MSTAR数据集\算法1_杨老师\datasourceProcess\OutputofYang\HB03334.015_yang.mat');
    [original_image, ~] = image_read(original_raw_file);
catch ME
    disp('Error loading files. Make sure my_output.mat, reference_output.mat, and the original .raw file are in the correct paths.');
    rethrow(ME);
end

% --- Quantify the Difference ---
my_scatterers = my_data.scatter_all;
ref_scatterers = ref_data.scatter_all;

num_my_scatterers = length(my_scatterers);
num_ref_scatterers = length(ref_scatterers);

fprintf('Your run found %d scattering centers.\n', num_my_scatterers);
fprintf('The reference run found %d scattering centers.\n', num_ref_scatterers);

% --- Visualize the Difference ---
% Extract (x,y) coordinates. Note: The model's (x,y) need to be converted to image pixel coordinates.
% The model coordinates are in meters, and the image is 128x128 pixels.
% From extrac.m: x_model = (col_pixel - 65) * 0.3 * 84 / 128;
% From extrac.m: y_model = (65 - row_pixel) * 0.3 * 84 / 128;
% We need to invert this mapping.
% col_pixel = (x_model * 128 / (0.3 * 84)) + 65;
% row_pixel = 65 - (y_model * 128 / (0.3 * 84));

% Function to convert model coordinates to image pixel coordinates
model_to_pixel = @(coords) [ (coords(1) * 128 / (0.3 * 84)) + 65, 65 - (coords(2) * 128 / (0.3 * 84)) ];

my_coords = cellfun(model_to_pixel, my_scatterers, 'UniformOutput', false);
my_coords = vertcat(my_coords{:});

ref_coords = cellfun(model_to_pixel, ref_scatterers, 'UniformOutput', false);
ref_coords = vertcat(ref_coords{:});

% Display the comparison plot
figure;
imagesc(abs(original_image));
colormap('gray');
axis equal tight;
hold on;

% Plot reference scatterers as blue circles
scatter(ref_coords(:,1), ref_coords(:,2), 60, 'o', 'MarkerEdgeColor', 'b', 'LineWidth', 1.5);

% Plot your scatterers as red 'x' marks
scatter(my_coords(:,1), my_coords(:,2), 80, 'x', 'MarkerEdgeColor', 'r', 'LineWidth', 1.5);

title('Comparison of Extracted Scattering Centers');
legend('Reference (20k file)', 'Your Run (2k file)');
hold off;