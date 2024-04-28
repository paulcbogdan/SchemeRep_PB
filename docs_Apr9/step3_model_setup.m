%% This script
% Reads trial_info_all.csv
% Generates folder structure for ALL trials
% Specify the models for ALL trials (locally)
% Existing matlabbatch.mat files will be skipped -- if corrections are
% needed, first manually remove the matlabbatch.mat files and then re-run
% this script.

% Last edited by Shenyang Huang 2024-02-17

% currently re-running: BL ENC RET, full, regBP false, LSS1

%% Setup
close all
clear
clc

%%% !!! run this script on the cluster to make sure paths match
% path_schemrep = 'Z:/Cabeza/SchemRep.01';
path_schemrep = '/mnt/munin2/Cabeza/SchemRep.01';
path_data = fullfile(path_schemrep, 'Data');
path_singletrial = fullfile(path_schemrep, 'Scripts', 'SingleTrialModel_SH');

%%% output directory
if ~exist(fullfile(path_singletrial, 'model_output'), 'dir')
    mkdir(fullfile(path_singletrial, 'model_output'))
end

%%% trial-level information prepared in previous steps
trial_info_all = readtable(fullfile(path_singletrial, 'trial_info', 'trial_info_all.csv'));

%%% spm
addpath(fullfile(path_schemrep, 'Scripts', 'spm12'))

%% model hyperparameters
%%% which subjects and tasks to model
% Subjects = unique(trial_info_all.Subject)';
% Subjects = unique(trial_info_all.Subject)';
Subjects = [138 212 213 231];

% Tasks = ["BL", "ENC", "RCON", "RVIS"];
Tasks = ["BL", "RCON", "RVIS"];
% Tasks = "BL";

%%% Duration: all scene images are shown for 3 sec, and all object images/labels are shown for 4 sec
% the "trials" can be modeled either using the full duration or as an impulse (0.1 sec)
trial_duration = "full"; dur_scene = 3; dur_object = 4;
% trial_duration = "impulse"; dur_scene = 0.1; dur_object = 0.1;

%%% whether or not to regress out button presses (duration 0.1 s)
% reg_buttonpress = true;
reg_buttonpress = false;

%%% ENC model structure -- this doesn't affect other Tasks
%%%% comment/uncomment appropriate lines to select the model structure
% ENC_style = "scene_obj_separate"; % two trials-of-interest for the pair, and separate regressors for things of no interest
% ENC_style = "scene_obj_combined"; % one trials-of-interest for the pair, and one regressor for pairs of no interest
ENC_style = "scene_obj_LSS1";
% ENC_style = "scene_obj_LSS2";

%%% nuisance regressors (not convolved with the HRF)
nuisance_cols = ["global_signal", "white_matter", "csf", ...
    "dvars", "framewise_displacement", "rmsd", ...
    "trans_x", "trans_y", "trans_z", "rot_x", "rot_y", "rot_z"];


%% pause to check hyperparameter settings
fprintf('Subjects: '); fprintf('%d ', Subjects); fprintf('\n')
fprintf('Tasks: '); fprintf('%s ', Tasks); fprintf('\n')
fprintf('Duration of trials: %s \n', trial_duration)
fprintf('Regressing out button presses: %s \n', string(reg_buttonpress))
fprintf('Encoding modeling style: %s \n', ENC_style)

% keyboard


%% generate matlabbatch.mat files for each trial-level model
% output directory structure: ./model_output/<Task>/<Subject>/
for Task = Tasks
    tic
    Task_folder = sprintf('%s__%s__regBP%s', Task, trial_duration, string(reg_buttonpress));
    %%% number of TRs per scanner run; hardcoded here for speed
    switch Task
        case "BL";   Ses = 1; nTR = 162;
        case "ENC";  Ses = 2; nTR = 276; Task_folder = strcat(Task_folder, "__", ENC_style);
        case "RCON"; Ses = 3; nTR = 190;
        case "RVIS"; Ses = 3; nTR = 168;
    end

    %%% output directory for different tasks
    if ~exist(fullfile(path_singletrial, 'model_output', Task_folder), 'dir')
        mkdir(fullfile(path_singletrial, 'model_output', Task_folder))
    end

    %%% save warning messages
    Errors = table;

    %%% equivalent to `for Subject = Subjects` but using parallel processes
    for Subject_i = 1:numel(Subjects)
        % parfor Subject_i = 1:numel(Subjects)
        Subject = Subjects(Subject_i);
        if Subject==138; Runs = 1:2;
        elseif Subject==213; Runs = [1 3];
        else; Runs = 1:3;
        end

        for Run = Runs

            %%% use this redundant value assignment to dodge the
            %%% restriction on for-loops within parfor-loops
            Run_i = Run; if Task=="RVIS"; Run_i = Run_i + 3; end
            if Subject==234 & Task=="ENC" %#ok<AND2>
                if Run_i==1
                    nTR = 211;
                else
                    nTR = 276;
                end
            end
            if exist(fullfile(path_singletrial, 'model_output', Task_folder, num2str(Subject), sprintf('matlabbatch_run%d.mat', Run_i)), 'file')
                %%% matlabbatch file already exists; don't overwrite that
                continue
            end

            tmptbl = trial_info_all(trial_info_all.Subject==Subject & trial_info_all.(strcat('Run_', Task))==Run_i, :);
            %%% Subject 234's ENC Run 1 quit early with only 211 TRs (422 s)
            if Subject==234 & Task=="ENC" & Run_i==1 %#ok<AND2>
                tmptbl = tmptbl(tmptbl.OnsetScene_ENC + 15 < 211*2, :);
            end
            if isempty(tmptbl)
                %%% no run information means no fMRI data; skip to the next one
                tmpError = table;
                tmpError.Task = Task;
                tmpError.Subject = Subject;
                tmpError.Run = Run_i;
                tmpError.Error = "cannot find trials in trial_info_all";
                Errors = vertcat(Errors, tmpError);
                warning('*** skipped %s subject %d run %d: no scan info in trial_info_all', Task, Subject, Run_i)
                continue
            end
            tmptbl = sortrows(tmptbl, strcat('Trial_', Task));

            %%% get fMRI scan ID and run number
            switch Task
                case "BL"
                    scanID = unique(tmptbl.visit1BIACid);
                case "ENC"
                    scanID = unique(tmptbl.visit2BIACid);
                case {"RCON" "RVIS"}
                    scanID = unique(tmptbl.visit3BIACid);
            end
            if isempty(scanID)
                %%% no scan ID means no fMRI data; skip to the next one
                tmpError = table;
                tmpError.Task = Task;
                tmpError.Subject = Subject;
                tmpError.Run = Run_i;
                tmpError.Error = "no scan ID in trial_info_all";
                Errors = vertcat(Errors, tmpError);
                warning('*** skipped %s subject %d run %d: no scan ID', Task, Subject, Run_i)
                continue
            end


            %% locate fMRI data
            %%% find preprocessed anatomical data
            %%% discrete segregation GM WM CSF by maximum probability
            % anat_niigz = sprintf('%s/fMRIprep_by_subject_out/sub-%d/anat/sub-%d_space-MNI152NLin2009cAsym_res-2_dseg.nii.gz', ...
            %     path_data, Subject, Subject);
            % anat_nii = sprintf('%s/uncompressed_anat/sub-%d_space-MNI152NLin2009cAsym_res-2_dseg.nii', ...
            %     path_data, Subject);
            %%% GM mask high-pass filter 20%
            anat_niigz = sprintf('%s/fMRIprep_by_subject_out/sub-%d/anat/sub-%d_space-MNI152NLin2009cAsym_res-2_label-GM_probseg.nii.gz', ...
                path_data, Subject, Subject);
            anat_nii = sprintf('%s/uncompressed_anat/sub-%d_space-MNI152NLin2009cAsym_res-2_label-GM_probseg.nii', ...
                path_data, Subject);
            if ~exist(anat_niigz, 'file')
                tmpError = table;
                tmpError.Task = Task;
                tmpError.Subject = Subject;
                tmpError.Run = Run_i;
                tmpError.Error = "no anat data in fMRIprep_by_subject_out";
                Errors = vertcat(Errors, tmpError);
                warning('*** skipped %s subject %d run %d: no preprocessed anatomical scan', Task, Subject, Run_i)
                continue
            end
            if ~exist(anat_nii, 'file')
                gunzip(anat_niigz, fullfile(path_data, 'uncompressed_anat'))
            end

            %%% create a gray matter mask for GLM
            gray_matter_mask = sprintf('%s/uncompressed_anat/sub-%d_space-MNI152NLin2009cAsym_res-2_GrayMatter20.nii', ...
                path_data, Subject);
            if ~exist(gray_matter_mask, 'file')
                V = spm_vol(anat_nii);
                mask = spm_read_vols(V);
                % mask(mask~=1) = 0; % gray matter is coded as 1; white matter 2; CSF 3; otherwise 0
                mask = double(mask>=0.2); % at least 20% probability of being GM
                V.fname = gray_matter_mask;
                V.private.dat.fname = gray_matter_mask;
                V.descrip = 'gray matter mask';
                spm_write_vol(V, mask);
            end

            %%% find preprocessed functional data
            func_niigz = sprintf('%s/fMRIprep_by_subject_out/sub-%d/ses-%d/func/sub-%d_ses-%d_task-%s_run-0%d_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz', ...
                path_data, Subject, Ses, Subject, Ses, Task, Run_i);
            func_nii = sprintf('%s/uncompressed_func/sub-%d_ses-%d_task-%s_run-0%d_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii', ...
                path_data, Subject, Ses, Task, Run_i);
            if ~exist(func_niigz, 'file')
                tmpError = table;
                tmpError.Task = Task;
                tmpError.Subject = Subject;
                tmpError.Run = Run_i;
                tmpError.Error = "no func data in fMRIprep_by_subject_out";
                warning('*** skipped %s subject %d run %d: no functional scan', Task, Subject, Run_i)
                continue
            end
            if ~exist(func_nii, 'file')
                gunzip(func_niigz, fullfile(path_data, 'uncompressed_func'))
            end
            %%% note that the first 4 TRs are discarded HERE
            %%% the Onsets in the behavioral files are already adjusted

            %%% ignore the first 4 TRs (disdaqs) in first-level modeling
            func_vols = cell(nTR-4,1);
            for i = 5:nTR %% note that i starts from 5
                func_vols{i-4} = strcat(func_nii, ',', num2str(i));
            end

            %%% select nuisance regressors based on fMRIprep output
            nuisance = tdfread(sprintf('%s/fMRIprep_by_subject_out/sub-%d/ses-%d/func/sub-%d_ses-%d_task-%s_run-0%d_desc-confounds_timeseries.tsv', ...
                path_data, Subject, Ses, Subject, Ses, Task, Run_i), 'tab');
            nuisance_use = nan(nTR-4, numel(nuisance_cols));
            for l = 1:numel(nuisance_cols)
                %%% convert to string and back to double to prevent data
                %%% type error for dvars, FD, and RMSD whose first entry is
                %%% 'n/a' in the fMRIprep output file
                value = str2double(string(nuisance.(nuisance_cols(l))));
                nuisance_use(:, l) = value(5:nTR);
            end
            nuisance_file = char(sprintf('%s/nuisance_regressors/sub-%d_ses-%d_task-%s_run-0%d_desc-confounds_timeseries_use.mat', ...
                path_data, Subject, Ses, Task, Run_i));
            R = nuisance_use; % SPM looks for a variable named `R`
            names = arrayfun(@(x)char(nuisance_cols(x)),1:numel(nuisance_cols),'uni',false);
            parSave(nuisance_file, R, names)


            %% model identifier--this is determined by several factors
            %%% 1. whether to regress out button press or not
            %%% 2. whether to model Objects using the whole duration (4 s) or as an impulse function (0.1 s)
            %%% 3. ENC-specific: how to model Scenes and Objects
            %%% LSS - as many models (cells in matlabbatch) as there are trials of interest
            %%%%%%%%%%%%%%%%%%%%%%% this part would slightly change for
            %%%%%%%%%%%%%%%%%%%%%%% "double-trial models" for ENC

            %%% subject-specific folder
            if ~exist(fullfile(path_singletrial, 'model_output', Task_folder, num2str(Subject)), 'dir')
                mkdir(fullfile(path_singletrial, 'model_output', Task_folder, num2str(Subject)))
            end

            %%% iterate over each trial (one row in the table)
            ntrial = height(tmptbl);
            matlabbatch = cell(ntrial, 1); %% models to be specified
            matlabbatch_rename = cell(ntrial, 1); %% to rename each SPM.mat as SPM_spec.mat
            if Task == "ENC"
                if contains(ENC_style, "LSS")
                    matlabbatch2 = cell(ntrial, 1); %% second set of models for obj
                    matlabbatch2_rename = cell(ntrial, 1);
                end
            end
            for j = 1:ntrial
                notj = setdiff(1:ntrial, j);
                tmptrial = tmptbl(j, :);
                model_name = sprintf(...
                    '%s_sub%d_run%d_trial%d_subset%d_pairID%d', ...
                    Task, Subject, Run_i, tmptrial.(strcat('Trial_', Task)), tmptrial.subset, tmptrial.pairID);

                %%% matlabbatch for model specification
                matlabbatch{j} = struct;
                matlabbatch{j}.spm.stats.fmri_spec.dir = {char(fullfile(path_singletrial, 'model_output', Task_folder, num2str(Subject), model_name))};
                matlabbatch{j}.spm.stats.fmri_spec.timing.units = 'secs';
                matlabbatch{j}.spm.stats.fmri_spec.timing.RT = 2; % TR
                matlabbatch{j}.spm.stats.fmri_spec.timing.fmri_t = 36; % number of slices
                matlabbatch{j}.spm.stats.fmri_spec.timing.fmri_t0 = 18; % reference slice for slice-timing correction
                matlabbatch{j}.spm.stats.fmri_spec.sess.scans = func_vols;
                if Task == "ENC"
                    if contains(ENC_style, "LSS")
                        matlabbatch{j}.spm.stats.fmri_spec.dir = {char(fullfile(path_singletrial, 'model_output', Task_folder, num2str(Subject), strcat(model_name, '_scene')))};

                        matlabbatch2{j} = struct;
                        matlabbatch2{j}.spm.stats.fmri_spec.dir = {char(fullfile(path_singletrial, 'model_output', Task_folder, num2str(Subject), strcat(model_name, '_object')))};
                        matlabbatch2{j}.spm.stats.fmri_spec.timing.units = 'secs';
                        matlabbatch2{j}.spm.stats.fmri_spec.timing.RT = 2; % TR
                        matlabbatch2{j}.spm.stats.fmri_spec.timing.fmri_t = 36; % number of slices
                        matlabbatch2{j}.spm.stats.fmri_spec.timing.fmri_t0 = 18; % reference slice for slice-timing correction
                        matlabbatch2{j}.spm.stats.fmri_spec.sess.scans = func_vols;

                    end
                end

                %%% regressors
                if Task ~= "ENC"
                    % 1. trial-of-interest
                    matlabbatch{j}.spm.stats.fmri_spec.sess.cond(1).name = 'Object';
                    matlabbatch{j}.spm.stats.fmri_spec.sess.cond(1).onset = tmptrial.(strcat('OnsetObj_', Task));
                    matlabbatch{j}.spm.stats.fmri_spec.sess.cond(1).duration = dur_object;

                    % 2. other trials
                    matlabbatch{j}.spm.stats.fmri_spec.sess.cond(2).name = 'OtherObjects';
                    matlabbatch{j}.spm.stats.fmri_spec.sess.cond(2).onset = tmptbl.(strcat('OnsetObj_', Task))(notj);
                    matlabbatch{j}.spm.stats.fmri_spec.sess.cond(2).duration = dur_object;

                elseif Task == "ENC"
                    switch ENC_style
                        case "scene_obj_separate"
                            % 1. scene-of-interest
                            matlabbatch{j}.spm.stats.fmri_spec.sess.cond(1).name = 'Scene';
                            matlabbatch{j}.spm.stats.fmri_spec.sess.cond(1).onset = tmptrial.(strcat('OnsetScene_', Task));
                            matlabbatch{j}.spm.stats.fmri_spec.sess.cond(1).duration = dur_scene;

                            % 2. object-of-interest
                            matlabbatch{j}.spm.stats.fmri_spec.sess.cond(2).name = 'Object';
                            matlabbatch{j}.spm.stats.fmri_spec.sess.cond(2).onset = tmptrial.(strcat('OnsetObj_', Task));
                            matlabbatch{j}.spm.stats.fmri_spec.sess.cond(2).duration = dur_object;

                            % 3. other scenes
                            matlabbatch{j}.spm.stats.fmri_spec.sess.cond(3).name = 'OtherScenes';
                            matlabbatch{j}.spm.stats.fmri_spec.sess.cond(3).onset = tmptbl.(strcat('OnsetScene_', Task))(notj);
                            matlabbatch{j}.spm.stats.fmri_spec.sess.cond(3).duration = dur_scene;

                            % 4. other objects
                            matlabbatch{j}.spm.stats.fmri_spec.sess.cond(4).name = 'OtherObjects';
                            matlabbatch{j}.spm.stats.fmri_spec.sess.cond(4).onset = tmptbl.(strcat('OnsetObj_', Task))(notj);
                            matlabbatch{j}.spm.stats.fmri_spec.sess.cond(4).duration = dur_object;

                        case "scene_obj_combined"
                            % 1. scene&object-of-interest
                            matlabbatch{j}.spm.stats.fmri_spec.sess.cond(1).name = 'SceneObject';
                            matlabbatch{j}.spm.stats.fmri_spec.sess.cond(1).onset = [tmptrial.(strcat('OnsetScene_', Task)); tmptrial.(strcat('OnsetObj_', Task))];
                            matlabbatch{j}.spm.stats.fmri_spec.sess.cond(1).duration = [dur_scene; dur_object];

                            % 2. other scenes & objects
                            matlabbatch{j}.spm.stats.fmri_spec.sess.cond(2).name = 'OtherScenesObjects';
                            matlabbatch{j}.spm.stats.fmri_spec.sess.cond(2).onset = [...
                                tmptbl.(strcat('OnsetScene_', Task))(notj); ...
                                tmptbl.(strcat('OnsetObj_', Task))(notj)];
                            matlabbatch{j}.spm.stats.fmri_spec.sess.cond(2).duration = [...
                                repmat(dur_scene, ntrial-1, 1);
                                repmat(dur_object, ntrial-1, 1)];

                        case "scene_obj_LSS1" % Lifu's original model, not differentiating scenes and objects
                            % 1. scene-of-interest
                            matlabbatch{j}.spm.stats.fmri_spec.sess.cond(1).name = 'Scene';
                            matlabbatch{j}.spm.stats.fmri_spec.sess.cond(1).onset = tmptrial.(strcat('OnsetScene_', Task));
                            matlabbatch{j}.spm.stats.fmri_spec.sess.cond(1).duration = dur_scene;

                            % 2. other scenes and objects
                            matlabbatch{j}.spm.stats.fmri_spec.sess.cond(2).name = 'AllButOneScene';
                            matlabbatch{j}.spm.stats.fmri_spec.sess.cond(2).onset = [...
                                tmptbl.(strcat('OnsetScene_', Task))(notj); ...
                                tmptbl.(strcat('OnsetObj_', Task))];
                            matlabbatch{j}.spm.stats.fmri_spec.sess.cond(2).duration = [...
                                repmat(dur_scene, ntrial-1, 1);
                                repmat(dur_object, ntrial, 1)];

                            % 1. object-of-interest -- in another model
                            matlabbatch2{j}.spm.stats.fmri_spec.sess.cond(1).name = 'Object';
                            matlabbatch2{j}.spm.stats.fmri_spec.sess.cond(1).onset = tmptrial.(strcat('OnsetObj_', Task));
                            matlabbatch2{j}.spm.stats.fmri_spec.sess.cond(1).duration = dur_object;

                            % 2. other scenes and objects
                            matlabbatch2{j}.spm.stats.fmri_spec.sess.cond(2).name = 'AllButOneObject';
                            matlabbatch2{j}.spm.stats.fmri_spec.sess.cond(2).onset = [...
                                tmptbl.(strcat('OnsetScene_', Task)); ...
                                tmptbl.(strcat('OnsetObj_', Task))(notj)];
                            matlabbatch2{j}.spm.stats.fmri_spec.sess.cond(2).duration = [...
                                repmat(dur_scene, ntrial, 1);
                                repmat(dur_object, ntrial-1, 1)];

                        case "scene_obj_LSS2" % scenes and objects are differentiated in the "other" regressor
                            % 1. scene-of-interest
                            matlabbatch{j}.spm.stats.fmri_spec.sess.cond(1).name = 'Scene';
                            matlabbatch{j}.spm.stats.fmri_spec.sess.cond(1).onset = tmptrial.(strcat('OnsetScene_', Task));
                            matlabbatch{j}.spm.stats.fmri_spec.sess.cond(1).duration = dur_scene;
                            % 2. other scenes
                            matlabbatch{j}.spm.stats.fmri_spec.sess.cond(2).name = 'OtherScenes';
                            matlabbatch{j}.spm.stats.fmri_spec.sess.cond(2).onset = tmptbl.(strcat('OnsetScene_', Task))(notj);
                            matlabbatch{j}.spm.stats.fmri_spec.sess.cond(2).duration = dur_scene;
                            % 3. other objects
                            matlabbatch{j}.spm.stats.fmri_spec.sess.cond(3).name = 'OtherObjects';
                            matlabbatch{j}.spm.stats.fmri_spec.sess.cond(3).onset = tmptbl.(strcat('OnsetObj_', Task));
                            matlabbatch{j}.spm.stats.fmri_spec.sess.cond(3).duration = dur_object;

                            % 1. object-of-interest -- in another model
                            matlabbatch2{j}.spm.stats.fmri_spec.sess.cond(1).name = 'Object';
                            matlabbatch2{j}.spm.stats.fmri_spec.sess.cond(1).onset = tmptrial.(strcat('OnsetObj_', Task));
                            matlabbatch2{j}.spm.stats.fmri_spec.sess.cond(1).duration = dur_object;
                            % 2. other scenes
                            matlabbatch2{j}.spm.stats.fmri_spec.sess.cond(2).name = 'OtherScenes';
                            matlabbatch2{j}.spm.stats.fmri_spec.sess.cond(2).onset = tmptbl.(strcat('OnsetScene_', Task));
                            matlabbatch2{j}.spm.stats.fmri_spec.sess.cond(2).duration = dur_scene;
                            % 3. other objects
                            matlabbatch2{j}.spm.stats.fmri_spec.sess.cond(3).name = 'OtherObjects';
                            matlabbatch2{j}.spm.stats.fmri_spec.sess.cond(3).onset = tmptbl.(strcat('OnsetObj_', Task))(notj);
                            matlabbatch2{j}.spm.stats.fmri_spec.sess.cond(3).duration = dur_object;

                        otherwise
                            error("Check variable `ENC_style`")
                    end
                else
                    error("Check variable `Task`")
                end

                %%% final regressor: button press
                if reg_buttonpress
                    counter = numel(matlabbatch{j}.spm.stats.fmri_spec.sess.cond) + 1;
                    onset_ButtonPress = tmptbl.(strcat('OnsetObj_', Task)) + tmptbl.(strcat('RT_', Task));
                    onset_ButtonPress = onset_ButtonPress(~isnan(onset_ButtonPress));
                    matlabbatch{j}.spm.stats.fmri_spec.sess.cond(counter).name = 'ButtonPress';
                    matlabbatch{j}.spm.stats.fmri_spec.sess.cond(counter).onset = onset_ButtonPress;
                    matlabbatch{j}.spm.stats.fmri_spec.sess.cond(counter).duration = 0.1;
                    if Task == "ENC" & contains(ENC_style, "LSS")
                        matlabbatch2{j}.spm.stats.fmri_spec.sess.cond(counter).name = 'ButtonPress';
                        matlabbatch2{j}.spm.stats.fmri_spec.sess.cond(counter).onset = onset_ButtonPress;
                        matlabbatch2{j}.spm.stats.fmri_spec.sess.cond(counter).duration = 0.1;
                    end
                end

                %%% common parts for all regressors
                nRegressor = numel(matlabbatch{j}.spm.stats.fmri_spec.sess.cond);
                for k = 1:nRegressor
                    matlabbatch{j}.spm.stats.fmri_spec.sess.cond(k).tmod = 0;
                    matlabbatch{j}.spm.stats.fmri_spec.sess.cond(k).pmod = struct('name', {}, 'param', {}, 'poly', {});
                    matlabbatch{j}.spm.stats.fmri_spec.sess.cond(k).orth = 0; %%% turn off orthogonalization: https://computationalbrainblog.wordpress.com/2021/06/
                    if Task == "ENC" & contains(ENC_style, "LSS")
                        matlabbatch2{j}.spm.stats.fmri_spec.sess.cond(k).tmod = 0;
                        matlabbatch2{j}.spm.stats.fmri_spec.sess.cond(k).pmod = struct('name', {}, 'param', {}, 'poly', {});
                        matlabbatch2{j}.spm.stats.fmri_spec.sess.cond(k).orth = 0; %%% turn off orthogonalization: https://computationalbrainblog.wordpress.com/2021/06/
                    end
                end

                %%% nuisance regressors (global signal, motion, etc.)
                matlabbatch{j}.spm.stats.fmri_spec.sess.multi_reg = {nuisance_file};
                %%% other global settings
                matlabbatch{j}.spm.stats.fmri_spec.fact = struct('name', {}, 'levels', {}); % don't use the factorial design setting
                matlabbatch{j}.spm.stats.fmri_spec.bases.hrf.derivs = [1 1]; % time and dispersion derivatives
                matlabbatch{j}.spm.stats.fmri_spec.volt = 1; % not modeling interactions
                matlabbatch{j}.spm.stats.fmri_spec.global = 'None'; % global intensity normalization; to turn on, use'Scaling'
                matlabbatch{j}.spm.stats.fmri_spec.sess.hpf = 128; % high-pass filter to remove slow drifts longer than 128 seconds
                matlabbatch{j}.spm.stats.fmri_spec.mask = {gray_matter_mask}; % fit GLM for gray matter voxels only
                matlabbatch{j}.spm.stats.fmri_spec.mthresh = -Inf; % no implicit masking
                matlabbatch{j}.spm.stats.fmri_spec.cvi = 'AR(1)';

                %%% make a copy of SPM.mat named SPM_spec.mat to prevent it from getting overwritten/confused with model estimation
                matlabbatch_rename{j}.cfg_basicio.file_dir.file_ops.file_move.files = {char(fullfile(path_singletrial, 'model_output', Task_folder, num2str(Subject), model_name, 'SPM.mat'))};
                matlabbatch_rename{j}.cfg_basicio.file_dir.file_ops.file_move.action.moveren.moveto = {char(fullfile(path_singletrial, 'model_output', Task_folder, num2str(Subject), model_name))};
                matlabbatch_rename{j}.cfg_basicio.file_dir.file_ops.file_move.action.moveren.patrep.pattern = 'SPM';
                matlabbatch_rename{j}.cfg_basicio.file_dir.file_ops.file_move.action.moveren.patrep.repl = 'SPM_spec';
                matlabbatch_rename{j}.cfg_basicio.file_dir.file_ops.file_move.action.moveren.unique = false;

                %%% do the same for another set of models
                if Task == "ENC"
                    if contains(ENC_style, "LSS")
                        %%% nuisance regressors (global signal, motion, etc.)
                        matlabbatch2{j}.spm.stats.fmri_spec.sess.multi_reg = {nuisance_file};
                        %%% other global settings
                        matlabbatch2{j}.spm.stats.fmri_spec.fact = struct('name', {}, 'levels', {}); % don't use the factorial design setting
                        matlabbatch2{j}.spm.stats.fmri_spec.bases.hrf.derivs = [1 1]; % time and dispersion derivatives
                        matlabbatch2{j}.spm.stats.fmri_spec.volt = 1; % not modeling interactions
                        matlabbatch2{j}.spm.stats.fmri_spec.global = 'None'; % global intensity normalization; to turn on, use'Scaling'
                        matlabbatch2{j}.spm.stats.fmri_spec.sess.hpf = 128; % high-pass filter to remove slow drifts longer than 128 seconds
                        matlabbatch2{j}.spm.stats.fmri_spec.mask = {gray_matter_mask}; % fit GLM for gray matter voxels only
                        matlabbatch2{j}.spm.stats.fmri_spec.mthresh = -Inf; % no implicit masking
                        matlabbatch2{j}.spm.stats.fmri_spec.cvi = 'AR(1)';

                        %%% make sure to differentiate scenes and objects here
                        matlabbatch_rename{j}.cfg_basicio.file_dir.file_ops.file_move.files = {char(fullfile(path_singletrial, 'model_output', Task_folder, num2str(Subject), strcat(model_name, "_scene"), 'SPM.mat'))};
                        matlabbatch_rename{j}.cfg_basicio.file_dir.file_ops.file_move.action.moveren.moveto = {char(fullfile(path_singletrial, 'model_output', Task_folder, num2str(Subject), strcat(model_name, "_scene")))};

                        matlabbatch2_rename{j}.cfg_basicio.file_dir.file_ops.file_move.files = {char(fullfile(path_singletrial, 'model_output', Task_folder, num2str(Subject), strcat(model_name, "_object"), 'SPM.mat'))};
                        matlabbatch2_rename{j}.cfg_basicio.file_dir.file_ops.file_move.action.moveren.moveto = {char(fullfile(path_singletrial, 'model_output', Task_folder, num2str(Subject), strcat(model_name, "_object")))};
                        matlabbatch2_rename{j}.cfg_basicio.file_dir.file_ops.file_move.action.moveren.patrep.pattern = 'SPM';
                        matlabbatch2_rename{j}.cfg_basicio.file_dir.file_ops.file_move.action.moveren.patrep.repl = 'SPM_spec';
                        matlabbatch2_rename{j}.cfg_basicio.file_dir.file_ops.file_move.action.moveren.unique = false;

                    end
                end


            end % Trial

            if Task == "ENC" & contains(ENC_style, "LSS")
                %%% combine
                matlabbatch = vertcat(matlabbatch, matlabbatch2, matlabbatch_rename, matlabbatch2_rename);
            else
                matlabbatch = vertcat(matlabbatch, matlabbatch_rename);
            end
            %%% use parSave wrapper function to save matlabbatch files in parallel
            parSave(fullfile(path_singletrial, 'model_output', Task_folder, num2str(Subject), sprintf('matlabbatch_run%d.mat', Run_i)), matlabbatch)

        end % Run
        % fprintf('Completed %s %d \n', Task, Subject)

    end % Subject

    %%% save warning messages
    writetable(Errors, sprintf('job_log/step3_model_setup/step3_model_setup_Errors__%s.csv', Task_folder))
    toc

end % Task

%% END OF SCRIPT