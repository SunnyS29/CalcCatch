function generate_sample_data()
%GENERATE_SAMPLE_DATA Create synthetic input files for CalcCatch.
%   Generates:
%   - data/example_stack.tif
%   - data/example_manual_roi_coordinates.xlsx (sheet: 'xy coord')
%
%   The generated files are synthetic and safe for public repositories.

rng(42);

data_dir = 'data';
if ~exist(data_dir, 'dir')
    mkdir(data_dir);
end

stack_path = fullfile(data_dir, 'example_stack.tif');
coord_path = fullfile(data_dir, 'example_manual_roi_coordinates.xlsx');

img_height = 192;
img_width  = 192;
num_frames = 140;
num_rois   = 24;

base_level = 220;
noise_sigma = 12;

% Initialize stack with low-amplitude background noise.
stack = base_level + noise_sigma * randn(img_height, img_width, num_frames);
[xx, yy] = meshgrid(1:img_width, 1:img_height);

% Keep ROI centroids away from borders so masks are fully represented.
margin = 18;
roi_centroids = zeros(num_rois, 2);

for r = 1:num_rois
    cx = margin + (img_width - 2 * margin) * rand();
    cy = margin + (img_height - 2 * margin) * rand();
    sigma_px = 2.5 + 2.5 * rand();   % Spatial spread of each neuron-like blob.
    peak_amp = 120 + 150 * rand();   % Peak intensity contribution.

    roi_centroids(r, :) = [cx, cy];

    spatial_blob = exp(-((xx - cx).^2 + (yy - cy).^2) / (2 * sigma_px^2));

    % Build sparse calcium-like events and smooth with an exponential kernel.
    event_prob = 0.04 + 0.02 * rand();
    impulses = double(rand(1, num_frames) < event_prob);
    kernel = exp(-(0:18) / (3 + 3 * rand()));
    activity_trace = conv(impulses, kernel, 'same');
    activity_trace = activity_trace / max(activity_trace + eps);

    for t = 1:num_frames
        stack(:, :, t) = stack(:, :, t) + peak_amp * activity_trace(t) * spatial_blob;
    end
end

% Add a weak global drift to mimic slow photometric fluctuation.
drift = 8 * sin((1:num_frames) * 2 * pi / num_frames * 2.5);
for t = 1:num_frames
    stack(:, :, t) = stack(:, :, t) + drift(t);
end

% Clamp to 16-bit image range used by CalcCatch.
stack = uint16(max(min(stack, 65535), 0));

if exist(stack_path, 'file')
    delete(stack_path);
end

for t = 1:num_frames
    if t == 1
        imwrite(stack(:, :, t), stack_path, 'tif', 'Compression', 'none');
    else
        imwrite(stack(:, :, t), stack_path, 'tif', 'WriteMode', 'append', 'Compression', 'none');
    end
end

% Write header names exactly as expected by CalcCatch.m.
coord_cells = [
    {'x centre', 'y centre'};
    num2cell(round(roi_centroids, 3))
];
writecell(coord_cells, coord_path, 'Sheet', 'xy coord');

fprintf('Synthetic stack saved: %s\n', stack_path);
fprintf('Synthetic coordinates saved: %s (sheet: xy coord)\n', coord_path);
fprintf('Done. You can now run CalcCatch.m with default dummy paths.\n');
end
