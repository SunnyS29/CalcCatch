% CalcCatch ROI recognition pipeline for enteric neuron calcium imaging.
% We document each stage so parameter tuning and biological interpretation
% remain reproducible across experiments.
% Downstream functional connectivity is computed with Pearson correlation
% and Transfer Entropy as complementary synchronous and directional metrics.


%% --------------------------- Adjustable Parameters ---------------------------
% Centralized parameter block for reproducibility and controlled tuning.
% File paths: 
tiff_stack_path      = 'data/example_stack.tif';  % Dummy path. Replace with your own TIFF stack
output_excel_file    = 'results/algorithm_rois.xlsx';  % Dummy output path for reproducible repo setup
original_coordinates = 'data/example_manual_roi_coordinates.xlsx'; % Dummy path. Replace with your own coordinate file

% Visualization parameters for diagnostics and reporting.
colourmap_choice     = 'plasma';      % Choice of colormap for bar charts
heatmap_scale         = [1, 4];        % Range for the heatmap (min and max values)

% Sliding window thresholding parameters for activity consensus mapping.
number_of_steps          = 5000;    % How many sliding window steps to use (higher number = higher resolution)
consistency_check        = 0.075;   % Fraction of windows a pixel must be active to be considered (currently set to 7.5% across all windows)
consistency_pixel_frac   = 0.70;    % Fraction of pixels in an ROI that need to be active in a window

% Size filter removes tiny components that are usually noise artefacts.
min_ROI_area         = 10;          % Minimum area (in pixels) for an ROI to be considered

% Watershed depth parameter. Larger h suppresses shallow minima and reduces over-segmentation.
h = 0.5;  % Adjust this value based on your data if needed

% Manual curation click radius in pixels.
selection_threshold   = 10;          % Maximum distance (in pixels) to register a click on an ROI

% Reporting and QA settings.
user_top_rois        = 10;          % How many top ROIs (by consistency) you want to highlight


%% --------------------------- TIFF Stack ---------------------------
fprintf('Loading the TIFF stack...\n');
info = imfinfo(tiff_stack_path);            % Get information about the TIFF stack (dimensions, etc.)
num_frames = numel(info);                   % Count how many frames are in the stack
img_height = info(1).Height;                % Image height (number of rows)
img_width  = info(1).Width;                 % Image width (number of columns)
tiff_stack = zeros(img_height, img_width, num_frames, 'uint16');  % Allocating a 3D array to store all frames

% Temporal axis for ROI traces and downstream connectivity metrics.
frame_rate = 2;                  % Frames per second (change if needed)
frame_interval = 1 / frame_rate;  
time_vector = (0:num_frames-1)' * frame_interval;  % Create a time vector for each frame, this will be used in the time series

% Load all frames once to avoid repeated disk reads during iterative processing.
for i = 1:num_frames
    tiff_stack(:,:,i) = imread(tiff_stack_path, i); 
end
fprintf('All frames loaded successfully.\n');

%% --------------------------- Load Original Coordinates ---------------------------
original_ROIs = readtable(original_coordinates, 'Sheet', 'xy coord', 'VariableNamingRule', 'preserve');
original_x_centres = original_ROIs.('x centre');   % Ground-truth x centroids from manual annotation
original_y_centres = original_ROIs.('y centre');   % Ground-truth y centroids from manual annotation
original_centroids = [original_x_centres(:), original_y_centres(:)]; % Nx2 matrix for point-based matching
fprintf('ROI coordinates loaded from Excel.\n');


%% --------------------------- Standard Deviation Across Frames ---------------------------
fprintf('Calculating standard deviation across all frames...\n');
stddev_matrix = std(double(tiff_stack), 0, 3);


%% --------------------------- Sliding Window Thresholding on Standard Deviation ---------------------------
std_max = max(stddev_matrix(:)); % Upper bound of temporal activity dispersion
std_min = min(stddev_matrix(:)); % Lower bound of temporal activity dispersion
stddev_range = linspace(std_max, std_min, number_of_steps); % Sweep thresholds densely to capture weak and strong neuronal activity
window_overlap = 0.1 * (std_max - std_min);  % Set an overlap of 10%

% 3D logical mask from sliding window thresholding.
% A pixel is active at step k if its temporal standard deviation falls in
% the threshold band for step k.
window_pixel_mask = false(img_height, img_width, number_of_steps);

for step = 1:number_of_steps
    % For each sliding window, define its start and end based on the
    % current threshold. Each value will cover a percentage around it
    window_start = stddev_range(step) - window_overlap/2;
    window_end   = stddev_range(step) + window_overlap/2;
    filtered_window_std = stddev_matrix >= window_start & stddev_matrix <= window_end;
    window_pixel_mask(:,:,step) = filtered_window_std;
end

% Consensus across thresholds is more robust than a single global cutoff.
pixel_count = sum(window_pixel_mask, 3);
minimum_windows = consistency_check * number_of_steps;
threshold_mask = pixel_count >= minimum_windows;  % Only keep pixels that passed the threshold enough times



 



%% --------------------------- Watershed Refinement ---------------------------
fprintf('Refining ROIs with watershed segmentation...\n');
% Negative distance transform converts ROI cores into watershed basins so
% adjacent neuronal regions can be separated.
dist_trans = -bwdist(~threshold_mask);
dist_trans(~threshold_mask) = -Inf;

% h-minima transformation suppresses shallow minima and limits
% over-segmentation in dense clusters.
dist_trans_mod = imhmin(dist_trans, h);

% Run watershed to split merged activity regions.
new_boundaries = watershed(dist_trans_mod);
watershed_regions = new_boundaries == 0;  % This gets the watershed ridge lines

% Slight ridge dilation improves separation of near-touching ROIs.
watershed_regions = imdilate(watershed_regions, strel('disk', 1));

% Invert ridge map to recover segmented ROI interiors.
watershed_regions = ~watershed_regions;


%% --------------------------- Visualization: Watershed Segmentation ---------------------------
% QA plot: check whether segmentation boundaries match expected cell morphology.

figure;
subplot(1,2,1)
imagesc(dist_trans); axis square; colorbar;
title('Distance Transform Plot','FontSize',16);

subplot(1,2,2)
imagesc(watershed_regions); axis square; colorbar;
title('Watershed Segmented Regions','FontSize',16);
drawnow;




%% --------------------------- Connected Component ROI Labelling ---------------------------
cc = bwconncomp(watershed_regions); % Label contiguous segmented regions as ROI candidates
roi_data = regionprops(cc, 'BoundingBox', 'Centroid', 'PixelIdxList', 'Area'); % Extract geometry and pixel membership for each ROI
filtered_centroids = reshape([roi_data.Centroid], 2, []).'; % Convert centroids to an Nx2 matrix
detected_ROI_count = length(roi_data);
fprintf('After refinement, detected %d ROIs.\n', detected_ROI_count);



%% --------------------------- Morphometric Filtering ---------------------------
roi_area = [roi_data.Area]; % Compute area for each ROI candidate
filtered_ROIs = roi_area >= min_ROI_area; % Remove very small regions that are likely non-neuronal noise
roi_data = roi_data(filtered_ROIs);
filtered_centroids = filtered_centroids(filtered_ROIs, :);
detected_ROI_count = length(roi_data); % Number of ROIs
fprintf('After filtering by size, %d ROIs remain.\n', detected_ROI_count);


%% --------------------------- Compile ROI Data into Table ---------------------------

tiff_stack_reshaped = reshape(double(tiff_stack), [], num_frames); % Flatten to [pixel x time] for fast ROI indexing

roi_table_format = table('Size', [detected_ROI_count, 4], ...
                    'VariableTypes', {'double', 'double', 'double', 'double'}, ...
                    'VariableNames', {'x_Centre', 'y_Centre', 'Mean_Intensity', 'Std_Intensity'}); % Output schema for ROI summary metrics

roi_table_format.x_Centre = filtered_centroids(:,1); % Write centroid x coordinates
roi_table_format.y_Centre = filtered_centroids(:,2);

roi_time_series = zeros(num_frames, detected_ROI_count); % Matrix to store mean fluorescence traces per ROI

for i = 1:detected_ROI_count
    pixelIdx = roi_data(i).PixelIdxList;
    roi_intensity = tiff_stack_reshaped(pixelIdx, :);
    roi_table_format.Mean_Intensity(i) = mean(roi_intensity, 'all');
    roi_table_format.Std_Intensity(i) = std(roi_intensity, 0, 'all');
    roi_time_series(:, i) = mean(roi_intensity, 1)';    
end

output_excel_file = 'Filtered_ROI_Data.xlsx';  % Default export file
writetable(roi_table_format, output_excel_file, 'Sheet', 'ROI_Data'); % Write ROI summary sheet

roi_names = [{'Time'}, arrayfun(@(i) sprintf('ROI_%d', i), 1:detected_ROI_count, 'UniformOutput', false)];
time_series_data = [time_vector, roi_time_series];
time_series_table = array2table(time_series_data, 'VariableNames', roi_names);
writetable(time_series_table, output_excel_file, 'Sheet', 'Time_Series');

fprintf('Filtered ROI data and time series saved to %s\n', output_excel_file);



%% --------------------------- ROI Consistency Scoring ---------------------------
fprintf('Calculating consistency of ROIs based on raw counts...\n');
roi_consistency_count = zeros(detected_ROI_count, 1);

% Count windows where enough ROI pixels are active.
% This favors persistent biological activity over transient noise.
for elem = 1:detected_ROI_count
    roi_pixels = roi_data(elem).PixelIdxList;
    num_pixels = length(roi_pixels);
    threshold_pixels = ceil(consistency_pixel_frac * num_pixels);
    consistency = 0; % Consistency counter
    for step = 1:number_of_steps % Iterate across threshold windows
        current_window = window_pixel_mask(:,:,step);
        active_pixels = sum(current_window(roi_pixels));
        if active_pixels >= threshold_pixels 
            consistency = consistency + 1;
        end
    end
    roi_consistency_count(elem) = consistency;
end

[~, sorted_idx] = sort(roi_consistency_count, 'descend'); % Rank ROIs by descending consistency
fprintf('ROI consistency calculated.\n');

%% --------------------------- Visualization: ROI Consistency Counts ---------------------------
% QA trend plot for ROI consistency distribution.

figure;
plot(roi_consistency_count, 'o-');
title('ROI Consistency Counts','FontSize',16);
xlabel('ROI Index'); ylabel('Consistency Count'); grid on;
drawnow;



%% --------------------------- The Top Most Consistent ROIs ---------------------------
% Select top-consistency ROIs for focused validation.

num_top_rois = min(user_top_rois, detected_ROI_count);
top_roi_index = sorted_idx(1:num_top_rois);
top_centroids = filtered_centroids(top_roi_index, :);
top_consistency = roi_consistency_count(top_roi_index);


%% --------------------------- Plot Bar Chart Based on Raw Counts ---------------------------
fprintf('Creating bar chart based on raw counts...\n');
figure;
hold on;
consistency_barchart = bar(1:detected_ROI_count, roi_consistency_count, 'FaceColor','flat');
min_point = min(roi_consistency_count);
max_point = max(roi_consistency_count);
normalized_points = (roi_consistency_count - min_point) / (max_point - min_point); % Normalize consistency values into [0,1] for color mapping
colormap('plasma');
colors = interp1(linspace(0,1,256), plasma(256), normalized_points); % Interpolate the plasma colormap for smooth consistency encoding

for i = 1:detected_ROI_count
    consistency_barchart.CData(i,:) = colors(i,:);
end
consistency_barchart.CData(top_roi_index, :) = repmat([0 1 0], num_top_rois, 1);
colorbar; caxis([min_point max_point]);
xlabel('ROI Index'); ylabel('ROI Consistency Counts');
title('ROI Consistency (Top ROIs Highlighted)','FontSize',16);
grid on;
hold off;
drawnow;

return

%% --------------------------- Visualization with Highlighted Consistent ROIs ---------------------------
fprintf('Creating heatmap with ROIs overlayed based on raw counts...\n');
figureHandle = figure('Name', 'Standard Deviation Heatmap with Detected ROIs (colour scaled with consistency) and Original ROIs (red)', ...
                      'NumberTitle', 'off', 'Units', 'Normalized', ...
                      'OuterPosition', [0 0 1 1]);
imagesc(log10(stddev_matrix + 1), heatmap_scale);
colormap('parula'); colorbar; hold on; axis image;
set(gca, 'FontSize', 14);
title({ ...
    'Comparison of Algorithm ROIs', ...
    sprintf('and Original ROIs on Standard Deviation Heatmap [%.2f, %.2f]', heatmap_scale(1), heatmap_scale(2))}, 'FontSize', 14);



min_consistency = min(roi_consistency_count);
max_consistency = max(roi_consistency_count);
normalized_points = (roi_consistency_count - min_consistency) / (max_consistency - min_consistency);
colors = interp1(linspace(0,1,256), plasma(256), normalized_points); % Normalization for the colour scale

scatter(filtered_centroids(:,1), filtered_centroids(:,2), 50, colors, 's', 'filled', ...
    'MarkerFaceAlpha', 0.7, 'DisplayName', 'Detected ROIs'); % Overlay algorithm-detected ROIs
scatter(original_x_centres, original_y_centres, 50, [1 0 0], 's', 'filled', ...
    'MarkerFaceAlpha', 0.7, 'DisplayName', 'Original ROIs'); % Overlay manual reference ROIs
scatter(top_centroids(:,1), top_centroids(:,2), 100, [1 1 0], 'd', 'filled', ...
    'MarkerEdgeColor', 'k', 'LineWidth', 1.5, 'DisplayName', 'Top ROIs'); % Highlight top-consistency ROIs for quick validation

% These will colour ROIs based on consistency and how they fall on the
% scale
for i = 1:detected_ROI_count
    text(filtered_centroids(i,1)+1, filtered_centroids(i,2)+1, num2str(i), ...
        'Color', 'w', 'FontSize', 10, 'FontWeight', 'bold');
end

for i = 1:detected_ROI_count
    rectangle('Position', roi_data(i).BoundingBox, 'EdgeColor', [0 0.9 0.3], 'LineWidth', 2.0);
end

lgd = legend('Detected ROIs','Original ROIs','Top ROIs','Location','best'); 
lgd.FontSize = 12;

hold off; 

% % These are optional if the legend needs to be adjusted
% % Get the current position (in normalized units)
% pos = lgd.Position;
% % Move legend slightly to the right (+0.02) and slightly down (-0.02).
% pos(1) = pos(1) + 0.05;  % Increase x to shift right
% pos(2) = pos(2) - 0.02;
% lgd.Position = pos;



drawnow;
pause;

fprintf('Heatmap with ROIs overlayed based on raw counts created.\n');


%% --------------------------- Remove Unwanted ROIs ---------------------------
% Ask user if they want to delete any ROIs
choice = questdlg('Would you like to remove any ROIs?', ...
    'Remove ROI', 'Yes', 'No', 'No');
if strcmpi(choice, 'Yes')
    disp('Starting ROI removal...')  % Bring up dialog box that removal mode is on
    figure(figure_handle)             % Bring up the ROI figure
    hold on                            % Keep existing plots so we can update them
    done = false;                     % Make sure they can go through the loop again (for multiple removals)
    while ~done
        % Instruct user on how to remove ROIs
        title('Click an ROI to remove. Right-click or press any key when finished')
        [x_click, y_click, button] = ginput(1);  % Wait for one click
        if isempty(button) || button ~= 1
            done = true;  % Break out if it's not a left-click
            break;
        end
        % Euclidean nearest-centroid selection keeps manual edits objective.
        dists = sqrt((filtered_centroids(:,1) - x_click).^2 + (filtered_centroids(:,2) - y_click).^2);
        [min_dist, idx] = min(dists);
        % If they clicked too far from any ROI, let them try again.
        % Implemented after testing
        if min_dist > selection_threshold
            fprintf('No ROI found within %.2f px. Try again.\n', min_dist);
            continue;
        end
        % Confirm which ROI is being removed
        fprintf('Removing ROI #%d at (%.2f, %.2f)\n', idx, filtered_centroids(idx,1), filtered_centroids(idx,2));
        % Actually remove the ROI data
        roi_data(idx) = [];
        filtered_centroids(idx, :) = [];
        roi_consistency_count(idx) = [];
        % Clear old plot and redraw heatmap + remaining ROIs
        cla;
        imagesc(log10(stddev_matrix + 1), heatmap_scale);
        colormap parula;
        colorbar;
        axis image;
        hold on;
        % Plot updated centroids in green and original centers in red
        scatter(filtered_centroids(:,1), filtered_centroids(:,2), 50, 'g', 's', 'filled', 'MarkerFaceAlpha', 0.7);
        scatter(original_x_centres, original_y_centres, 50, [1 0 0], 's', 'filled', 'MarkerFaceAlpha', 0.7);
        % Draw bounding boxes and labels, using connected components from
        % before
        for k = 1:numel(roi_data)
            rectangle('Position', roi_data(k).BoundingBox, 'EdgeColor', [0 0.9 0.3], 'LineWidth', 2);
            text(filtered_centroids(k,1)+1, filtered_centroids(k,2)+1, num2str(k), 'Color', 'w');
        end
    end
    hold off  % Done with manual removal
    fprintf('Removal finished. %d ROIs left.\n', numel(roi_data));
end

%% Let user add ROIs manually
orig_count = numel(roi_data);
choice = questdlg('Add any ROIs manually?', 'Manual ROI', 'Yes', 'No', 'No');
if strcmpi(choice, 'Yes')
    disp('Manual ROI mode on')  % Bring up dialog box for manual addition
    figure(figure_handle)
    hold on
    new_rois = [];
    while true
        % Ask if they want to draw another ROI polygon
        add_choice = questdlg('Draw another ROI?', 'Manual ROI', 'Draw', 'Done', 'Done');
        if strcmp(add_choice, 'Draw')
            h = drawpolygon;                % Draw a manual polygon ROI
            mask = createMask(h);           % Convert polygon to a binary mask
            delete(h);                      % Remove drawn polygon
            stats = regionprops(mask, 'Centroid', 'BoundingBox', 'PixelIdxList', 'Area');
            if ~isempty(stats)
                new_rois(end+1) = stats(1);  % Append new ROI so exports and metrics include manual additions
                % Visual confirmation for the operator
                plot(stats(1).Centroid(1), stats(1).Centroid(2), 'rx');
                rectangle('Position', stats(1).BoundingBox, 'EdgeColor', 'r');
            end
        else
            break;  % Done drawing
        end
    end
    % Append new ROIs if any were drawn
    if ~isempty(new_rois)
        roi_data = [roi_data; new_rois'];
        centroids = vertcat(new_rois.Centroid);
        filtered_centroids = [filtered_centroids; centroids];
        roi_consistency_count = [roi_consistency_count; -1*ones(numel(new_rois),1)];
        fprintf('Added %d new ROIs. Now %d total.\n', numel(new_rois), numel(roi_data));
    end
    hold off
end

%% Final visualization if ROIs changed, same process as pre manual adjust

% If the number of ROIs change through addition or removal, a second
% visualization is generated

if numel(roi_data) ~= orig_count
    figure('Name','Final ROIs','NumberTitle','off')
    imagesc(log10(stddev_matrix + 1), heatmap_scale)
    colormap parula;
    colorbar;
    axis image;
    hold on;
    % Mark all ROIs with green squares, originals with red squares
    scatter(filtered_centroids(:,1), filtered_centroids(:,2), 'gs');
    scatter(original_x_centres, original_y_centres, 'rs');
    for k = 1:numel(roi_data)
        rectangle('Position', roi_data(k).BoundingBox, 'EdgeColor', [0 0.9 0.3]);
        text(filtered_centroids(k,1)+1, filtered_centroids(k,2)+1, num2str(k));
    end
    legend({'New/Kept ROIs','Original ROIs'});
    drawnow;
    pause;  % (Optional) Lets you move legends or tweak axis if needed
    % Save a high-res snapshot
    print(gcf, 'final_heatmap.png', '-dpng', '-r600');
end




%% --------------------------- Contingency Matching with Candidate Resolution ---------------------------
% Compare detected ROI centroids with ground-truth centroids and resolve
% multi-candidate matches by nearest-neighbor assignment.

% Load the ground-truth coordinates from the Excel file
landmark = readtable(original_coordinates, 'Sheet', 'xy coord', 'VariableNamingRule', 'preserve');

% Count how many ground-truth ROIs there are
num_landmark = size(landmark, 1);

% Count how many ROIs were detected by our method
num_detected = size(filtered_centroids, 1);

% Extract the x and y positions of the ground-truth ROI centroids
landmark_centroids = [landmark.('x centre'), landmark.('y centre')];

% Set a distance tolerance (in pixels) to decide if a detected ROI is close
% enough to a ground-truth ROI
tolerance = 10;

% Create a matrix to store matches: 1 means a detected ROI is within
% tolerance of a ground-truth ROI
match_matrix = zeros(num_landmark, num_detected);

% Loop through every ground-truth ROI
for i = 1:num_landmark
    % Check against every detected ROI
    for j = 1:num_detected
        x_det = filtered_centroids(j, 1);  % X position of detected ROI
        y_det = filtered_centroids(j, 2);  % Y position of detected ROI

        % Check if X and Y are both within the tolerance range
        if abs(x_det - landmark_centroids(i,1)) <= tolerance && abs(y_det - landmark_centroids(i,2)) <= tolerance
            match_matrix(i, j) = 1;  % Mark it as a match
        end
    end
end

% Resolve multiple candidates with nearest-neighbor one-to-one assignment.
assigned_detection = zeros(num_landmark, 1);  % This will hold the best match index
used = false(num_detected, 1);  % Keep track of which detected ROIs have already been matched

% Loop through each ground-truth ROI
for i = 1:num_landmark
    % Find all candidate detected ROIs that are within tolerance and not used yet
    candidate_idx = find(match_matrix(i,:) == 1 & ~used');
    if ~isempty(candidate_idx)
        % Compute distances between the ground-truth and all candidates
        distances = sqrt(sum((filtered_centroids(candidate_idx,:) - landmark_centroids(i,:)).^2, 2));
        [~, min_idx] = min(distances);  % Find the one with the smallest distance
        best_candidate = candidate_idx(min_idx);  % Select it
        assigned_detection(i) = best_candidate;  % Assign this candidate to the ground-truth
        used(best_candidate) = true;  % Mark this detected ROI as used
    end
end

% Count how many ground-truth ROIs were matched to at least one detected ROI
matched_count = sum(assigned_detection > 0);

% Matching indicator is the matched landmark fraction.
matching_indicator = matched_count / num_landmark;

% Print the result to check if it is right
fprintf('Matching Indicator (Diagonal Matching with candidate resolution) = %.2f\n', matching_indicator);

% Make a simple 2x2 contingency table for evaluation:
TP = matched_count;  % True positives: correctly detected
FN = num_landmark - matched_count;  % False negatives: ground-truth that weren?™t matched
FP = num_detected - matched_count;  % False positives: extra detected ROIs not matched to anything
TN = NaN;  % True negatives aren?™t defined here

% Create the table to show results
contingency_table = table([TP; FP], [FN; TN], ...
    'RowNames', {'Actual Positive','Actual Negative'}, ...
    'VariableNames', {'Predicted Positive','Predicted Negative'});

% Display the contingency table
disp('2x2 Contingency Table:');
disp(contingency_table);







