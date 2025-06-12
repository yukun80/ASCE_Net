% MATLAB script to create a RAW file (R1) from T1.mat for use with image_read
% Loads T1.mat containing Img (amplitude) and Phase, combines them into a single
% RAW file with amplitude followed by phase as float32 in big-endian order,
% and names the file with dimensions (e.g., R1.128x128.raw)
% 将由 MSTAR2JPG.m 生成的 .mat 文件转换为算法主程序期望的特定 .raw 二进制格式。它将所有幅度值连接在一起，然后是所有相位值 。

function create_R1_for_image_read(input_mat_file, output_basename)
    % Load the T1.mat file
    data = load(input_mat_file);
    
    % Check if required variables exist
    if ~isfield(data, 'Img') || ~isfield(data, 'phase')
        error('Input file %s must contain Img and Phase variables.', input_mat_file);
    end
    
    % Get amplitude and phase
    amplitude = data.Img;
    phase = data.phase;
    
    % Verify dimensions
    [rows, cols] = size(amplitude);
    if ~isequal(size(amplitude), size(phase)) || rows ~= 128 || cols ~= 128
        error('Img and Phase must be 128x128 matrices.');
    end
    
    % Flatten amplitude and phase to 1D arrays
    amplitude = single(amplitude(:)); % Convert to single-precision (float32)
    phase = single(phase(:));         % Convert to single-precision (float32)
    
    % Check data size
    data_size = rows * cols;
    if length(amplitude) ~= data_size || length(phase) ~= data_size
        error('Unexpected data size. Expected %d pixels, got %d (amplitude), %d (phase).', ...
              data_size, length(amplitude), length(phase));
    end
    
    % Combine amplitude and phase: all amplitudes followed by all phases
    combined_data = [amplitude; phase];
    
    % Construct output filename with dimensions (e.g., R1.128x128.raw)
    output_filename = sprintf('%s.%dx%d.raw', output_basename, cols, rows);
    
    % Save to RAW binary file (no header, float32, big-endian)
    fid_out = fopen(output_filename, 'wb');
    if fid_out == -1
        error('Cannot open output file: %s', output_filename);
    end
    fwrite(fid_out, combined_data, 'float32', 'b');
    fclose(fid_out);
    
    fprintf('Successfully created RAW file %s from %s\n', output_filename, input_mat_file);
end