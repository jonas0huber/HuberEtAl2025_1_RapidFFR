function unified_ffr_processor()
%% UNIFIED EEG DATA PROCESSING SCRIPT
% This script processes EEG data for three different experiment types:
% 1. Rapid - Standard rapid presentation paradigm
% 2. Conv - Conventional presentation with multiple trials
% 3. LongRapid - Extended rapid presentation paradigm

% Batch pre-processes EEG recordings:
% 1. Loads Cz and C7 channels
% 2. Re-references Cz to C7
% 3. Bandpass filters the signal between 70 and 3000 Hz
% 4. Removes the first (meaningless) event
% 5. Exports filtered Cz signal and event latency as .mat file in the first
% row, and the events (i.e. triggers) in the second row
%
% Author: Jonas Huber


    %% ====================================================================
    %  CONFIGURATION SECTION
    % ====================================================================
    
    % Initialize EEGLAB
    addpath('./eeglab2021.0');
    [ALLEEG, EEG, CURRENTSET, ALLCOM] = eeglab;
    
    % Define processing types and their configurations
    processing_configs = get_processing_configurations();
    
    % User selection of processing type
    fprintf('\nAvailable processing types:\n');
    fprintf('1. Rapid\n');
    fprintf('2. Conv\n');
    fprintf('3. LongRapid\n');
    
    processing_type = input('Select processing type (1-3): ');
    
    if processing_type < 1 || processing_type > 3
        error('Invalid processing type selected. Please choose 1, 2, or 3.');
    end
    
    type_names = {'Rapid', 'Conv', 'LongRapid'};
    selected_type = type_names{processing_type};
    config = processing_configs.(selected_type);
    
    fprintf('Processing %s data...\n', selected_type);
    
    %% ====================================================================
    %  MAIN PROCESSING LOOP
    % ====================================================================
    
    % Process each participant
    for participant_idx = 1:config.num_participants
        participant_id = generate_participant_id(participant_idx);
        fprintf('Processing participant %s...\n', participant_id);
        
        % Process each run (if applicable)
        num_runs = get_num_runs(config, selected_type);
        for run_idx = 1:num_runs
            
            % Process each polarity
            for polarity_idx = 1:config.num_polarities
                polarity_name = get_polarity_name(polarity_idx);
                
                try
                    % Process single condition
                    process_single_condition(config, selected_type, ...
                        participant_id, participant_idx, run_idx, ...
                        polarity_name, polarity_idx);
                        
                    fprintf('  Completed: %s, Run %d, Polarity %s\n', ...
                        participant_id, run_idx, polarity_name);
                        
                catch ME
                    fprintf('  ERROR processing %s, Run %d, Polarity %s: %s\n', ...
                        participant_id, run_idx, polarity_name, ME.message);
                    continue;
                end
            end
        end
    end
    
    % Completion notification
    fprintf('\nProcessing completed successfully!\n');
    change_to_analysis_directory();
    beep;
    
end

%% ========================================================================
%  CONFIGURATION FUNCTIONS
% ========================================================================

function configs = get_processing_configurations()
    %% Define all configuration parameters for each processing type
    
    % Common parameters
    common_config = struct(...
        'active_electrode', 33, ...           % Cz electrode channel
        'reference_electrode', 34, ...        % C7 electrode channel  
        'filter_order', 2, ...               % Butterworth filter order
        'high_cutoff_freq', 3000, ...        % High-pass filter cutoff (Hz)
        'low_cutoff_freq', 70, ...           % Low-pass filter cutoff (Hz)
        'num_polarities', 2 ...              % Number of polarities (pos/neg)
    );
    
    % Rapid configuration
    configs.Rapid = merge_structs(common_config, struct(...
        'input_dir', './input/', ...
        'output_dir', './output/', ...
        'num_participants', 16, ...
        'input_filename_pattern', '-HC-rapid-', ...
        'output_filename_prefix', 'Rapid_Part', ...
        'num_runs', 2, ...
        'subdirectory_suffix', ' C/' ...
    ));
    
    % Conv configuration  
    configs.Conv = merge_structs(common_config, struct(...
        'input_dir', './input/', ...
        'output_dir', './output/', ...
        'num_participants', 16, ...
        'input_filename_pattern', '-HC-normal-', ...
        'output_filename_prefix', 'Conv_Part', ...
        'num_runs', 2, ...
        'num_trials', 1500, ...
        'subdirectory_suffix', ' N/' ...
    ));
    
    % LongRapid configuration
    configs.LongRapid = merge_structs(common_config, struct(...
        'input_dir', './input/', ...
        'output_dir', './output/', ...
        'num_participants', 21, ...
        'input_filename_pattern', '-HC-quiet-', ...
        'output_filename_prefix', 'LongRapid_Part', ...
        'num_runs', 1, ...
        'subdirectory_suffix', '' ...
    ));
    
end

function merged = merge_structs(struct1, struct2)
    %% Merge two structures, with struct2 fields taking precedence
    merged = struct1;
    fields = fieldnames(struct2);
    for i = 1:length(fields)
        merged.(fields{i}) = struct2.(fields{i});
    end
end

%% ========================================================================
%  UTILITY FUNCTIONS  
% ========================================================================

function participant_id = generate_participant_id(participant_number)
    %% Generate standardized participant ID (L01, L02, ..., L10, L11, ...)
    if participant_number < 10
        participant_id = sprintf('L0%d', participant_number);
    else
        participant_id = sprintf('L%d', participant_number);
    end
end

function polarity_name = get_polarity_name(polarity_index)
    %% Convert polarity index to name
    if polarity_index == 1
        polarity_name = 'pos';
    else
        polarity_name = 'neg';
    end
end

function num_runs = get_num_runs(config, processing_type)
    %% Get number of runs based on processing type
    if strcmp(processing_type, 'LongRapid')
        num_runs = 1;  % LongRapid has no runs
    else
        num_runs = config.num_runs;
    end
end

function input_directory = construct_input_directory(config, processing_type, participant_id)
    %% Construct the full input directory path
    if strcmp(processing_type, 'LongRapid')
        input_directory = config.input_dir;
    else
        input_directory = fullfile(config.input_dir, participant_id, ...
            [participant_id, config.subdirectory_suffix]);
    end
end

function filename = construct_input_filename(config, participant_id, polarity_name, run_number, processing_type)
    %% Construct input filename based on processing type
    if strcmp(processing_type, 'LongRapid')
        filename = sprintf('%s%s%s.bdf', participant_id, ...
            config.input_filename_pattern, polarity_name);
    else
        filename = sprintf('%s%s%s-%d.bdf', participant_id, ...
            config.input_filename_pattern, polarity_name, run_number);
    end
end

function filename = construct_output_filename(config, participant_number, polarity_name, run_number, processing_type)
    %% Construct output filename based on processing type
    if strcmp(processing_type, 'LongRapid')
        filename = sprintf('%s%d%sPol', config.output_filename_prefix, ...
                participant_number, polarity_name);
    else
        filename = sprintf('%s%d%sPolRun%d', config.output_filename_prefix, ...
                participant_number, polarity_name, run_number);
    end
end

function change_to_analysis_directory()
    %% Change to analysis directory after processing
    try
        cd('/Users/jonas/Documents/MatLab/Tim Study/My Pipe Claude/Data Prep/');
    catch
        fprintf('Warning: Could not change to analysis directory\n');
    end
end

%% ========================================================================
%  CORE PROCESSING FUNCTIONS
% ========================================================================

function process_single_condition(config, processing_type, participant_id, ...
    participant_number, run_number, polarity_name, polarity_index)
    %% Process a single experimental condition
    
    % Construct file paths
    input_directory = construct_input_directory(config, processing_type, participant_id);
    input_filename = construct_input_filename(config, participant_id, ...
        polarity_name, run_number, processing_type);
    full_input_path = fullfile(input_directory, input_filename);
    
    % Verify input file exists
    if ~exist(full_input_path, 'file')
        error('Input file not found: %s', full_input_path);
    end
    
    % Load and preprocess EEG data
    EEG = load_and_preprocess_eeg(full_input_path, config);
    
    % Prepare data for export
    export_matrix = prepare_export_matrix(EEG, config, processing_type);
    
    % Save processed data
    output_filename = construct_output_filename(config, participant_number, ...
        polarity_name, run_number, processing_type);
    full_output_path = fullfile(config.output_dir, [output_filename, '.mat']);
    
    % Ensure output directory exists
    if ~exist(config.output_dir, 'dir')
        mkdir(config.output_dir);
    end
    
    save(full_output_path, 'export_matrix', '-v7.3');
    
end

function EEG = load_and_preprocess_eeg(input_filepath, config)
    %% Load EEG data and apply standard preprocessing steps
    
    % Load data with specified channels
    channels_to_load = [config.active_electrode, config.reference_electrode];
    EEG = pop_biosig(input_filepath, 'channels', channels_to_load);
    EEG = eeg_checkset(EEG);
    
    % Re-reference to C7 (now channel 2 after loading subset)
    reference_channel_index = 2;
    EEG = pop_reref(EEG, reference_channel_index);
    EEG = eeg_checkset(EEG);
    
    % Apply bandpass filter
    EEG.data = butter_filtfilt(EEG.data, config.low_cutoff_freq, ...
        config.high_cutoff_freq, config.filter_order);
    
    % Remove first event (artifact from recording start)
    if ~isempty(EEG.event)
        EEG = pop_editeventvals(EEG, 'delete', 1);
        EEG = eeg_checkset(EEG);
    end
    
    % Keep only the active electrode data (first channel after referencing)
    EEG.data = EEG.data(1, :);
    
end

function export_matrix = prepare_export_matrix(EEG, config, processing_type)
    %% Prepare data matrix for export based on processing type
    
    data_length = size(EEG.data, 2);
    export_matrix = zeros(2, data_length);
    
    % First row: EEG data
    export_matrix(1, :) = EEG.data;
    
    % Second row: event latencies (processing type dependent)
    switch processing_type
        case {'Rapid', 'LongRapid'}
            % Single event at start
            if ~isempty(EEG.event)
                export_matrix(2, 1) = EEG.event(1).latency;
            end
            
        case 'Conv'
            % Multiple trial events
            num_events = min(length(EEG.event), config.num_trials);
            for event_idx = 1:num_events
                export_matrix(2, event_idx) = EEG.event(event_idx).latency;
            end
    end
    
end