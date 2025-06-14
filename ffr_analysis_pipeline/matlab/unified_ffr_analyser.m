function unified_ffr_analyser()
%% UNIFIED ERP ANALYSIS SCRIPT - OPTIMIZED VERSION
% This script processes EEG data for ERP analysis across three paradigms:
% 1. LongRapid - Extended rapid presentation ERP analysis
% 2. Rapid - Standard rapid presentation ERP analysis  
% 3. Conv - Conventional ERP analysis with onset detection
%
% Batch processes EEG data for FFR ERP analysis:
% 1. Loads participant data files based on selected analysis type
% 2. Extracts epochs and applies artifact rejection (35 mV threshold)
% 3. For Conv analysis: detects stimulus onset using cross-correlation
% 4. Resamples data to correct for timing drift
% 5. Exports processed epochs and trial counts as CSV files
%
% Author: Jonas Huber

    %% ====================================================================
    %  MAIN EXECUTION
    % ====================================================================
    
    try
        % Get user selection and configuration
        [selected_type, config] = setup_analysis();
        
        % Initialize logging
        setup_logging(config.output_dir);
        
        % Process all participants
        results = process_all_participants(config, selected_type);
        
        % Save and display results
        finalize_results(config, results, selected_type);
        
        fprintf('\nERP analysis completed successfully!\n');
        
    catch ME
        fprintf('\nFATAL ERROR: %s\n', ME.message);
        fprintf('Stack trace:\n');
        for i = 1:length(ME.stack)
            fprintf('  %s (line %d)\n', ME.stack(i).name, ME.stack(i).line);
        end
        rethrow(ME);
    end
end

%% ========================================================================
%  SETUP AND CONFIGURATION
% ========================================================================

function [selected_type, config] = setup_analysis()
    %% Handle user selection and return configuration
    
    type_names = {'LongRapid', 'Rapid', 'Conv'};
    
    fprintf('\nAvailable ERP analysis types:\n');
    for i = 1:length(type_names)
        fprintf('%d. %s ERP\n', i, type_names{i});
    end
    
    analysis_type = input('Select analysis type (1-3): ');
    
    if analysis_type < 1 || analysis_type > 3
        error('Invalid analysis type selected. Please choose 1, 2, or 3.');
    end
    
    selected_type = type_names{analysis_type};
    config = get_analysis_configuration(selected_type);
    
    fprintf('Processing %s ERP analysis...\n', selected_type);
end

function config = get_analysis_configuration(analysis_type)
    %% Get configuration for specified analysis type
    
    % Common parameters
    base_config = struct(...
        'sampling_freq', 16384, ...
        'f0_stimulus', 128, ...
        'artifact_threshold_mv', 35, ...
        'brainstem_delay_ms', 0.01, ...
        'num_polarities', 2 ...
    );
    
    % Calculate derived parameters
    cycle_length = base_config.sampling_freq / base_config.f0_stimulus;
    brainstem_delay_samples = round(base_config.sampling_freq * base_config.brainstem_delay_ms);
    
    % Type-specific configurations
    switch analysis_type
        case 'LongRapid'
            specific_config = struct(...
                'input_dir', './input/', ...
                'output_dir', './output/', ...
                'import_name', 'LongRapid_Part', ...
                'export_name', 'TimLongRapid_Part', ...
                'num_participants', 21, ...
                'num_runs', 1, ...
                'num_trials', 67500, ...
                'pre_stim_trigger_samples', cycle_length, ...
                'trial_length_samples', cycle_length, ...
                'enable_csv_export', true ...
            );
            
        case 'Rapid'
            specific_config = struct(...
                'input_dir', './input/', ...
                'output_dir', './output/', ...
                'import_name', 'Rapid_Part', ...
                'export_name', 'TimRapidData_Part', ...
                'num_participants', 16, ...
                'num_runs', 2, ...
                'num_trials', 7503, ...
                'pre_stim_trigger_samples', round(base_config.sampling_freq * 0.01), ...
                'trial_length_samples', cycle_length, ...
                'enable_csv_export', true ...
            );
            
        case 'Conv'
            specific_config = struct(...
                'input_dir', './input/', ...
                'output_dir', './output/', ...
                'import_name', 'Conv_Part', ...
                'export_name', 'TimConvData_Part', ...
                'num_participants', 16, ...
                'num_runs', 2, ...
                'num_trials', 1500, ...
                'stimulus_duration_samples', 7 * cycle_length, ...
                'pre_stim_trigger_samples', round(base_config.sampling_freq * 0.01), ...
                'baseline_pre_samples', round(base_config.sampling_freq * 0.007), ...
                'enable_csv_export', true ...
            );
            specific_config.stimulus_duration_buffer_samples = ...
                specific_config.stimulus_duration_samples + round(base_config.sampling_freq * 0.02);
    end
    
    % Merge configurations
    config = merge_structs(base_config, specific_config);
    config.brainstem_delay_samples = brainstem_delay_samples;
    
    % Calculate block durations for rapid analyses
    if ismember(analysis_type, {'LongRapid', 'Rapid'})
        config.block_duration_samples = config.num_trials * config.trial_length_samples;
    end
end

function merged = merge_structs(struct1, struct2)
    %% Merge two structures, with struct2 fields taking precedence
    merged = struct1;
    fields = fieldnames(struct2);
    for i = 1:length(fields)
        merged.(fields{i}) = struct2.(fields{i});
    end
end

function setup_logging(output_dir)
    %% Setup logging to file
    ensure_output_directory(output_dir);
    % File logging
    log_filename = fullfile(output_dir, sprintf('erp_analysis_log_%s.txt', datestr(now, 'yyyymmdd_HHMMSS')));
    global LOG_FILE_ID;
    LOG_FILE_ID = fopen(log_filename, 'w');
    
    % Write initial log entry
    fprintf(LOG_FILE_ID, 'ERP Analysis Started: %s\n', datestr(now));
    fprintf(LOG_FILE_ID, '================================\n');
end

%% ========================================================================
%  MAIN PROCESSING FUNCTIONS
% ========================================================================

function results = process_all_participants(config, analysis_type)
    %% Process all participants for the given analysis type
    
    results = initialize_results_tracking(config, analysis_type);
    
    for participant_idx = 1:config.num_participants
        fprintf('Processing participant %d/%d...\n', participant_idx, config.num_participants);
        
        try
            switch analysis_type
                case {'LongRapid', 'Rapid'}
                    results = process_rapid_type_participant(config, participant_idx, results, analysis_type);
                case 'Conv'
                    results = process_conv_participant(config, participant_idx, results, analysis_type);
            end
        catch ME
            log_participant_error(participant_idx, ME);
        end
    end
end

function results = process_rapid_type_participant(config, participant_idx, results, analysis_type)
    %% Generic processing for LongRapid and Rapid analyses
    
    for run_idx = 1:config.num_runs
        for polarity_idx = 1:config.num_polarities
            polarity_name = get_polarity_name(polarity_idx);
            
            try
                % Generate filename based on analysis type
                filename = generate_filename(config, participant_idx, run_idx, polarity_name, analysis_type);
                
                % Load and process data
                eeg_data = load_eeg_data(filename);
                processed_data = extract_rapid_type_data(eeg_data, config);
                [accepted_epochs, num_accepted] = process_rapid_type_trials(processed_data, config);
                
                % Store results
                if strcmp(analysis_type, 'LongRapid')
                    results.accepted_trials(participant_idx, polarity_idx) = num_accepted;
                else % Rapid
                    results.accepted_trials(participant_idx, run_idx, polarity_idx) = num_accepted;
                end
                
                % Export if enabled
                if config.enable_csv_export
                    export_filename = generate_export_filename(config, participant_idx, run_idx, polarity_name, analysis_type);
                    export_data_to_csv(accepted_epochs, export_filename, config.output_dir);
                end
                
            catch ME
                log_processing_error(participant_idx, run_idx, polarity_name, ME);
                % Set failed result
                if strcmp(analysis_type, 'LongRapid')
                    results.accepted_trials(participant_idx, polarity_idx) = 0;
                else
                    results.accepted_trials(participant_idx, run_idx, polarity_idx) = 0;
                end
            end
        end
    end
end

function results = process_conv_participant(config, participant_idx, results, analysis_type)
    %% Process single participant for Conventional analysis
    
    for run_idx = 1:config.num_runs
        try
            % Load both polarity files
            [eeg_data_pos, eeg_data_neg] = load_conv_polarity_files(config, participant_idx, run_idx);
            
            % Extract trials with baseline correction
            trials_pos = get_conv_data_and_baseline(eeg_data_pos(1,:), eeg_data_pos(2,:), ...
                config.num_trials, config.stimulus_duration_buffer_samples, ...
                config.pre_stim_trigger_samples, config.baseline_pre_samples);
            trials_neg = get_conv_data_and_baseline(eeg_data_neg(1,:), eeg_data_neg(2,:), ...
                config.num_trials, config.stimulus_duration_buffer_samples, ...
                config.pre_stim_trigger_samples, config.baseline_pre_samples);
            
            % Compute onset and store result
            onset_sample = compute_onset_from_erp(trials_pos, trials_neg, config);
            results.onset_times(participant_idx, run_idx) = onset_sample;
            
            % Process each polarity
            for polarity_idx = 1:config.num_polarities
                polarity_name = get_polarity_name(polarity_idx);
                
                try
                    % Select appropriate polarity data
                    if polarity_idx == 1
                        preprocessed_data = trials_pos;
                    else
                        preprocessed_data = trials_neg;
                    end
                    
                    % Process trials with onset adjustment and resampling
                    processed_data = process_conv_trials_with_onset(preprocessed_data, onset_sample, config);
                    
                    % Reject artifacts
                    [accepted_epochs, num_accepted] = reject_epochs_with_count(processed_data, config.artifact_threshold_mv);
                    
                    % Store results
                    results.accepted_trials(participant_idx, run_idx, polarity_idx) = num_accepted;
                    
                    % Export if enabled
                    if config.enable_csv_export
                        export_filename = generate_export_filename(config, participant_idx, run_idx, polarity_name, analysis_type);
                        export_data_to_csv(accepted_epochs, export_filename, config.output_dir);
                    end
                    
                catch ME_polarity
                    % Handle polarity-specific errors
                    log_processing_error(participant_idx, run_idx, polarity_name, ME_polarity);
                    results.accepted_trials(participant_idx, run_idx, polarity_idx) = 0;
                end
            end
            
        catch ME_run
            % Handle run-level errors (affects both polarities)
            log_processing_error(participant_idx, run_idx, 'both_polarities', ME_run);
            results.accepted_trials(participant_idx, run_idx, :) = 0;
            results.onset_times(participant_idx, run_idx) = 0;
        end
    end
end

%% ========================================================================
%  DATA PROCESSING UTILITIES
% ========================================================================

function processed_data = extract_rapid_type_data(eeg_data, config)
    %% Extract and resample data for rapid-type analyses
    start_sample = eeg_data(2,1) + config.pre_stim_trigger_samples + config.brainstem_delay_samples;
    end_sample = start_sample + config.block_duration_samples - 1;
    
    data_of_interest = eeg_data(1, start_sample:end_sample);
    processed_data = resample_data(data_of_interest, ...
        config.block_duration_samples, config.sampling_freq);
end

function [accepted_epochs, num_accepted] = process_rapid_type_trials(data, config)
    %% Process rapid-type trials: reshape and reject artifacts
    available_samples = length(data);
    cycles_to_use = floor(available_samples / config.trial_length_samples);
    samples_to_use = cycles_to_use * config.trial_length_samples;
    
    data_reshaped = reshape(data(1:samples_to_use), config.trial_length_samples, [])';
    [accepted_epochs, num_accepted] = reject_epochs_with_count(data_reshaped, config.artifact_threshold_mv);
end

function [eeg_data_pos, eeg_data_neg] = load_conv_polarity_files(config, participant_idx, run_idx)
    %% Load both polarity files for conventional analysis
    filename_pos = sprintf('%s%s%dposPolRun%d.mat', config.input_dir, ...
        config.import_name, participant_idx, run_idx);
    filename_neg = sprintf('%s%s%dnegPolRun%d.mat', config.input_dir, ...
        config.import_name, participant_idx, run_idx);
    
    eeg_data_pos = load_eeg_data(filename_pos);
    eeg_data_neg = load_eeg_data(filename_neg);
end

%% ========================================================================
%  FILENAME AND I/O UTILITIES
% ========================================================================

function filename = generate_filename(config, participant_idx, run_idx, polarity_name, analysis_type)
    %% Generate input filename based on analysis type
    switch analysis_type
        case 'LongRapid'
            filename = sprintf('%s%s%d%sPol.mat', config.input_dir, ...
                config.import_name, participant_idx, polarity_name);
        case 'Rapid'
            filename = sprintf('%s%s%d%sPolRun%d.mat', config.input_dir, ...
                config.import_name, participant_idx, polarity_name, run_idx);
    end
end

function export_filename = generate_export_filename(config, participant_idx, run_idx, polarity_name, analysis_type)
    %% Generate export filename based on analysis type
    if strcmp(analysis_type, 'LongRapid')
        export_filename = sprintf('%s%s%d%sPol.csv', config.output_dir, ...
            config.export_name, participant_idx, polarity_name);
    else
            export_filename = sprintf('%s%s%dRun%d%sPol.csv', config.output_dir, ...
                config.export_name, participant_idx, run_idx, polarity_name);
    end
end

function export_data_to_csv(data, filename, output_dir)
    %% Export data to CSV with directory creation
    ensure_output_directory(output_dir);
    writematrix(data, filename);
end

function eeg_data = load_eeg_data(filename)
    %% Load EEG data from MAT file
    if ~exist(filename, 'file')
        error('File not found: %s', filename);
    end
    
    try
        loaded_data = load(filename);
        if isfield(loaded_data, 'export_matrix')
            eeg_data = loaded_data.export_matrix;
        else
            available_fields = fieldnames(loaded_data);
            error('Expected data matrix not found in file: %s. Available fields: %s', ...
                filename, strjoin(available_fields, ', '));
        end
    catch ME
        if strcmp(ME.identifier, 'MATLAB:load:couldNotReadFile')
            error('Could not read file (possibly corrupted): %s', filename);
        else
            rethrow(ME);
        end
    end
end

%% ========================================================================
%  RESULTS AND LOGGING
% ========================================================================

function results = initialize_results_tracking(config, analysis_type)
    %% Initialize results tracking structure
    switch analysis_type
        case 'LongRapid'
            results.accepted_trials = zeros(config.num_participants, config.num_polarities);
        case 'Rapid'
            results.accepted_trials = zeros(config.num_participants, config.num_runs, config.num_polarities);
        case 'Conv'
            results.accepted_trials = zeros(config.num_participants, config.num_runs, config.num_polarities);
            results.onset_times = zeros(config.num_participants, config.num_runs);
    end
end

function finalize_results(config, results, analysis_type)
    %% Save final results and display summary
    
    % Save to workspace
    assignin('base', 'nAcceptedTrials', results.accepted_trials);
    % Export number of accepted Trials
    acceptedTrials_filename = fullfile(config.output_dir, 'nAcceptedTrials.csv');
    ensure_output_directory(config.output_dir);
    writematrix(results.accepted_trials, acceptedTrials_filename);
    
    if isfield(results, 'onset_times')
        assignin('base', 'onsetTimes', results.onset_times);
        
        % Export onset times
        onset_filename = fullfile(config.output_dir, 'OnsetTimes.csv');
        ensure_output_directory(config.output_dir);
        writematrix(results.onset_times, onset_filename);
    end
    
    % Display summary
    display_processing_summary(results, analysis_type);
end

function display_processing_summary(results, analysis_type)
    %% Display comprehensive processing summary
    fprintf('\n=== Processing Summary ===\n');
    fprintf('Analysis type: %s\n', analysis_type);
    
    % Calculate and display statistics
    total_trials = sum(results.accepted_trials(:));
    mean_trials = mean(results.accepted_trials(:));
    std_trials = std(results.accepted_trials(:));
    
    fprintf('Total accepted trials: %d\n', total_trials);
    fprintf('Mean trials per condition: %.1f (±%.1f)\n', mean_trials, std_trials);
    fprintf('Results saved to workspace as ''nAcceptedTrials'' and exported to nAcceptedTrials.csv\n');
    
    if isfield(results, 'onset_times')
        mean_onset = mean(results.onset_times(:));
        std_onset = std(results.onset_times(:));
        fprintf('Mean onset time: %.3f samples (±%.3f)\n', mean_onset, std_onset);
        fprintf('Onset times saved to workspace as ''onsetTimes'' and exported to OnsetTimes.csv\n');
    end
end

function log_participant_error(participant_idx, ME)
    %% Log participant-level errors
    fprintf('  ERROR processing participant %d: %s\n', participant_idx, ME.message);
end

function log_processing_error(participant_idx, run_idx, polarity_name, ME)
    %% Log processing errors with context
    fprintf('  ERROR processing participant %d, run %d, polarity %s: %s\n', ...
        participant_idx, run_idx, polarity_name, ME.message);
end

%% ========================================================================
%  UTILITY FUNCTIONS
% ========================================================================

function polarity_name = get_polarity_name(polarity_index)
    %% Convert polarity index to name
    polarity_names = {'pos', 'neg'};
    polarity_name = polarity_names{polarity_index};
end

function ensure_output_directory(output_dir)
    %% Create output directory if it doesn't exist
    if ~exist(output_dir, 'dir')
        [success, msg] = mkdir(output_dir);
        if ~success
            error('Failed to create output directory %s: %s', output_dir, msg);
        end
    end
end

function [accepted_epochs, num_accepted] = reject_epochs_with_count(data, threshold)
    %% Reject epochs exceeding amplitude threshold and return count
    rows_to_keep = ~any(abs(data) > threshold, 2);
    accepted_epochs = data(rows_to_keep, :);
    num_accepted = size(accepted_epochs, 1);
end

%% ========================================================================
%  CORE SIGNAL PROCESSING FUNCTIONS (PRESERVED)
% ========================================================================

function delay_value = delay_function_constrained(waveform, freq_sine_wave, ...
    num_cycles_sinewave, actual_duration)
    %% Find stimulus onset using constrained cross-correlation
    
    fs = 16384; % Sampling frequency
    dt = 1/fs;
    F = freq_sine_wave;
    T = num_cycles_sinewave / F;
    
    % Create reference sine wave
    tt = 0:dt:T+dt;
    stimulus_wave = sin(2*pi*F*tt);
    
    % Cross-correlation
    [corr_vals, lags] = xcorr(waveform, stimulus_wave);
    
    % Constrain search to positive lags
    negative_values = round(length(corr_vals)/2) + 3;
    corr_vals = corr_vals(negative_values:end);
    
    % Prevent upper bound violations
    corr_vals = corr_vals(1:end-actual_duration);
    
    % Constrain to 5-15ms after onset
    corr_vals_constrained = corr_vals(round(fs*0.005):round(fs*0.015));
    
    % Find maximum correlation
    [~, max_idx] = max(corr_vals_constrained);
    index_max_corr = find(max(corr_vals_constrained) == corr_vals, 1) + negative_values - 1;
    
    delay_value = lags(index_max_corr);
end

function trials_final = process_conv_trials_with_onset(trials_preprocessed, onset_sample, config)
    %% Process conventional trials with onset adjustment and resampling
    
    % Resize based on onset
    end_sample = onset_sample + config.stimulus_duration_samples - 1;
    trials_resized = trials_preprocessed(:, onset_sample:end_sample);
    
    % Resample
    trials_final = resample_data(trials_resized, ...
        config.stimulus_duration_samples, config.sampling_freq);
    
end

function onset_sample = compute_onset_from_erp(trials_pos, trials_neg, config)
    %% Compute stimulus onset from ERP using cross-correlation
    
    % Compute ERPs
    erp_pos = mean(trials_pos, 1);
    erp_neg = mean(trials_neg, 1);
    
    % Compute EFR (Envelope Following Response)
    efr = (erp_pos + erp_neg) / 2;
    
    % Find onset using cross-correlation
    stimulus_duration_cycles = config.stimulus_duration_samples / config.f0_stimulus;
    onset_sample = delay_function_constrained(efr, 128, stimulus_duration_cycles, ...
        config.stimulus_duration_samples);
end

function x_resampled = resample_data(input_data, num_samples_per_trial, ...
    original_samp_freq)
    %% Resample data for analysis
    
    % Determine if input is single trial (1D) or multiple trials (2D)
    is_single_trial = isvector(input_data) && size(input_data, 1) == 1;
    
    if is_single_trial
        % Handle single trial (1D input)
        data_to_process = input_data;
        num_trials = 1;
    else
        % Handle multiple trials (2D input)
        data_to_process = input_data;
        num_trials = size(input_data, 1);
    end
    
    % Measured F0 from computer synchronization (constant across all analyses)
    old_f0      = 128.0037994384766;
    target_freq = 128;
    new_samp_freq = target_freq * old_f0;
    
    if is_single_trial
        % Process single trial
        probe_data = data_to_process;
        
        % Remove last sample if odd length
        if mod(length(probe_data), 2)
            probe_data = probe_data(1:end-1);
        end
        
        % Create time vectors
        t = (0:length(probe_data)-1) / original_samp_freq;
        t_new = (0:1/new_samp_freq:max(t));
        
        % Resample using spline interpolation
        resampled_data = interp1(t, probe_data, t_new, 'spline');
        x_resampled = resampled_data(1:num_samples_per_trial);
        
    else
        % Process multiple trials
        % Remove last sample from each trial if odd length
        if mod(size(data_to_process, 2), 2)
            data_to_process = data_to_process(:, 1:end-1);
        end
        
        % Create time vectors (same for all trials)
        t = (0:size(data_to_process, 2)-1) / original_samp_freq;
        t_new = (0:1/new_samp_freq:max(t));
        
        % Initialize output matrix
        x_resampled = zeros(num_trials, num_samples_per_trial);
        
            % Process each trial
        for trial_idx = 1:num_trials
            resampled_trial = interp1(t, data_to_process(trial_idx, :), t_new, 'spline');
            x_resampled(trial_idx, :) = resampled_trial(1:num_samples_per_trial);
        end
    end
end


function trials = get_conv_data_and_baseline(eeg_data, trigger_data, num_trials, ...
    stimulus_duration, pre_stim_trigger, baseline_pre)
    %% Extract conventional trials with baseline correction
    trials = zeros(num_trials, stimulus_duration);
    
    for trial_idx = 1:num_trials
        if trial_idx <= length(trigger_data) && trigger_data(trial_idx) > 0
            trigger_sample = round(trigger_data(trial_idx));
            
            % Define trial window (stimulus period only, like original)
            trial_start = trigger_sample + pre_stim_trigger;
            trial_end = trial_start + stimulus_duration - 1;
            
            % Define baseline window (separate from trial data)
            baseline_start = trigger_sample + pre_stim_trigger - baseline_pre;
            baseline_end = trigger_sample + pre_stim_trigger;
            
            % Check bounds for both windows
            if trial_start > 0 && trial_end <= length(eeg_data) && ...
               baseline_start > 0 && baseline_end <= length(eeg_data)
                
                % Extract trial data (stimulus period)
                trial_data = eeg_data(trial_start:trial_end);
                
                % Calculate baseline from separate window
                baseline_data = eeg_data(baseline_start:baseline_end);
                baseline_mean = mean(baseline_data);
                
                % Apply baseline correction
                trials(trial_idx, :) = trial_data - baseline_mean;
            end
        end
    end
end
