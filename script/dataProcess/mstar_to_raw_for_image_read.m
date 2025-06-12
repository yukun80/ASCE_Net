% MATLAB script to convert MSTAR data to RAW format compatible with image_read function
% Reads an MSTAR file, skips the header, extracts amplitude and phase data,
% and saves them as float32 values (all amplitudes followed by all phases) in a single RAW binary file
% Output filename includes dimensions (e.g., output.128x128.raw)

function mstar_to_raw_for_image_read(input_filename, output_basename, rows, cols, header_size)
    % Default header size is 1606 bytes for MSTAR target chip files
    if nargin < 5
        header_size = 1606;
    end
    % Default image dimensions (e.g., 128x128 for typical MSTAR chips)
    if nargin < 4
        rows = 128;
        cols = 128;
    end
    
    % Construct output filename with dimensions (e.g., output.128x128.raw)
    output_filename = sprintf('%s.%dx%d.raw', output_basename, cols, rows);
    
    % Open the input MSTAR file in read-only mode
    fid = fopen(input_filename, 'r', 'b'); % Big-endian byte order
    if fid == -1
        error('Cannot open input file: %s', input_filename);
    end
    
    % Skip the header
    fseek(fid, header_size, 'bof');
    
    % Read amplitude and phase data (interleaved in input, 4-byte floats)
    data_size = rows * cols; % Total number of pixels
    amplitude = fread(fid, data_size, '*float32', 4, 'b'); % Skip 4 bytes (phase) after each amplitude
    fseek(fid, header_size + 4, 'bof'); % Rewind to start of phase data
    phase = fread(fid, data_size, '*float32', 4, 'b'); % Skip 4 bytes (amplitude) after each phase
    
    % Close the input file
    fclose(fid);
    
    % Check if data was read correctly
    if length(amplitude) ~= data_size || length(phase) ~= data_size
        error('Unexpected data size. Expected %d pixels, read %d (amplitude), %d (phase).', ...
              data_size, length(amplitude), length(phase));
    end
    
    % Combine amplitude and phase data: all amplitudes followed by all phases
    combined_data = [amplitude; phase]; % Concatenate vertically
    
    % Save to RAW binary file (no header, float32, big-endian)
    fid_out = fopen(output_filename, 'wb');
    if fid_out == -1
        error('Cannot open output file: %s', output_filename);
    end
    fwrite(fid_out, combined_data, 'float32', 'b');
    fclose(fid_out);
    
    fprintf('Successfully converted %s to RAW format as %s\n', input_filename, output_filename);
end