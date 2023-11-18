% This script uses the MATLAB implementation of VGG16 to create model RDMs
% of STAMP images
close all
clear
clc

rootDir = 'D:/local/STAMP_HCinteraction/';


%% behavioral file
% 300 out of all 995 items were used as 'old' items in STAMP fMRI
fmri_stim = readtable([rootDir 'Behavior/Data/final3/newretS026_final3.xlsx']);
fmri_stim_old = fmri_stim(logical(fmri_stim.old), {'ID' 'image'});


%% different image resolutions
% dir_im300 = 'X:/stamp/stimuli'; % resolution = 300x300, all 996 items; same as images downloaded from https://mariamh.shinyapps.io/dinolabobjects/
dir_im256 = 'W:/STAMP/fMRI/Behav/stimuli'; % resolution = 256x256, 464items
dir_im500 = 'W:/STAMP/fMRI/Behav/stimuli/originals'; % resolution = 500x500, 460 items

dir_to_use = dir_im256;

im_table = table;
im_array = nan(224, 224, 3, length(fmri_stim_old.image));

% load images and resize
% for resizing, the default interpolation method is bicubic
% https://www.mathworks.com/help/matlab/ref/imresize.html
for k = 1:length(fmri_stim_old.image)
    imfname = fmri_stim_old.image{k};
    if exist(fullfile(dir_to_use, imfname), 'file')
        tmptbl = table;
        tmptbl.path = string(fullfile(dir_to_use, imfname));
        tmptbl.ind = k;
        tmptbl.ID = fmri_stim_old.ID(k);
        tmptbl.name = string(imfname);
        im_table = [im_table; tmptbl];
        tmpim = imresize(imread(fullfile(dir_to_use, imfname)), [224 224]);
        im_array(:,:,:,k) = tmpim;
    else
        disp(imfname)
    end
end
% convert string to char as required by the activations function
im_table.path = char(im_table.path);
im_array = uint8(im_array); % for proper image display


%% VGG 16
% requires input dimensions = 224x224 
% Implementation: https://www.mathworks.com/help/deeplearning/ref/vgg16.html
% Others: https://www.mathworks.com/help/deeplearning/ug/pretrained-convolutional-neural-networks.html
% load vgg 16 pre-trained on ImageNet
net = vgg16('Weights','imagenet');


%% layers in VGG16
net.Layers; % note that there are 41 layers in total, not 23, as the ReLU 
% layers are also considered
%      1   'input'     Image Input             224×224×3 images with 'zerocenter' normalization
%      2   'conv1_1'   Convolution             64 3×3×3 convolutions with stride [1  1] and padding [1  1  1  1]
%      3   'relu1_1'   ReLU                    ReLU
%      4   'conv1_2'   Convolution             64 3×3×64 convolutions with stride [1  1] and padding [1  1  1  1]
%      5   'relu1_2'   ReLU                    ReLU
%      6   'pool1'     Max Pooling             2×2 max pooling with stride [2  2] and padding [0  0  0  0]
%      7   'conv2_1'   Convolution             128 3×3×64 convolutions with stride [1  1] and padding [1  1  1  1]
%      8   'relu2_1'   ReLU                    ReLU
%      9   'conv2_2'   Convolution             128 3×3×128 convolutions with stride [1  1] and padding [1  1  1  1]
%     10   'relu2_2'   ReLU                    ReLU
%     11   'pool2'     Max Pooling             2×2 max pooling with stride [2  2] and padding [0  0  0  0]
%     12   'conv3_1'   Convolution             256 3×3×128 convolutions with stride [1  1] and padding [1  1  1  1]
%     13   'relu3_1'   ReLU                    ReLU
%     14   'conv3_2'   Convolution             256 3×3×256 convolutions with stride [1  1] and padding [1  1  1  1]
%     15   'relu3_2'   ReLU                    ReLU
%     16   'conv3_3'   Convolution             256 3×3×256 convolutions with stride [1  1] and padding [1  1  1  1]
%     17   'relu3_3'   ReLU                    ReLU
%     18   'pool3'     Max Pooling             2×2 max pooling with stride [2  2] and padding [0  0  0  0]
%     19   'conv4_1'   Convolution             512 3×3×256 convolutions with stride [1  1] and padding [1  1  1  1]
%     20   'relu4_1'   ReLU                    ReLU
%     21   'conv4_2'   Convolution             512 3×3×512 convolutions with stride [1  1] and padding [1  1  1  1]
%     22   'relu4_2'   ReLU                    ReLU
%     23   'conv4_3'   Convolution             512 3×3×512 convolutions with stride [1  1] and padding [1  1  1  1]
%     24   'relu4_3'   ReLU                    ReLU
%     25   'pool4'     Max Pooling             2×2 max pooling with stride [2  2] and padding [0  0  0  0]
%     26   'conv5_1'   Convolution             512 3×3×512 convolutions with stride [1  1] and padding [1  1  1  1]
%     27   'relu5_1'   ReLU                    ReLU
%     28   'conv5_2'   Convolution             512 3×3×512 convolutions with stride [1  1] and padding [1  1  1  1]
%     29   'relu5_2'   ReLU                    ReLU
%     30   'conv5_3'   Convolution             512 3×3×512 convolutions with stride [1  1] and padding [1  1  1  1]
%     31   'relu5_3'   ReLU                    ReLU
%     32   'pool5'     Max Pooling             2×2 max pooling with stride [2  2] and padding [0  0  0  0]
%     33   'fc6'       Fully Connected         4096 fully connected layer
%     34   'relu6'     ReLU                    ReLU
%     35   'drop6'     Dropout                 50% dropout
%     36   'fc7'       Fully Connected         4096 fully connected layer
%     37   'relu7'     ReLU                    ReLU
%     38   'drop7'     Dropout                 50% dropout
%     39   'fc8'       Fully Connected         1000 fully connected layer
%     40   'prob'      Softmax                 softmax
%     41   'output'    Classification Output   crossentropyex with 'tench' and 999 other classes

layer_info = table;
for layer_i = 1:length(net.Layers)
    name = string(net.Layers(layer_i).Name);
    layer_info = [layer_info; table(layer_i, name)];
end


%% try classification
k = 30
figure
image(squeeze(im_array(:,:,:,k)))
output = classify(net, im_array(:,:,:,k));
title(sprintf('#%d \t %s \t pred: %s', im_table.ID(im_table.ind==k), im_table.name(im_table.ind==k), output))


%% get activations and generate model RDM
[im_table_sorted, IDorder] = sortrows(im_table, 'ID'); 
im_table_sorted.ind = [];
stim_ID_num = im_table_sorted.ID;

% Do PCA on the input and early layer activations to reduce the influence 
% of white space, as suggested by Alex Clarke.
% doPCA = input('do PCA (0/1)? \n');

tic
modelRDM = struct;
for layer_i = 1:length(net.Layers)
    layer_name = layer_info.name(layer_i);
    if contains(layer_name, 'input') || contains(layer_name, 'conv') || contains(layer_name, 'fc')
        
        % get DNN activations and vectorize
        vgg_act = activations(net, im_array, layer_i, 'ExecutionEnvironment', 'gpu');
        vgg_act_vector = reshape(vgg_act, [], length(fmri_stim_old.image));

        % normal correlation
        R = corr(vgg_act_vector, 'type', 'Pearson', 'rows', 'pairwise');
        R(logical(eye(size(R)))) = nan;
        R = R(IDorder, IDorder);
        modelRDM.(layer_name).r = R;
        save(sprintf('%s/modelRDM/VGG16/modelRDM_VGG__%s__r.mat', rootDir, layer_name), ...
            'R', 'stim_ID_num', 'im_table_sorted')

        % cosine similarity 
        R = cosineSimilarity(vgg_act_vector');
        R(logical(eye(size(R)))) = nan;
        R = R(IDorder, IDorder);
        modelRDM.(layer_name).cos = R;
        save(sprintf('%s/modelRDM/VGG16/modelRDM_VGG__%s__cos.mat', rootDir, layer_name), ...
            'R', 'stim_ID_num', 'im_table_sorted')

        % weighted correlation - weight by neuron-wise variance
        w = std(vgg_act_vector, 0, 2).^2;
        R = weightedcorrs(vgg_act_vector, w);
        R(logical(eye(size(R)))) = nan;
        R = R(IDorder, IDorder);
        modelRDM.(layer_name).wr = R;
        save(sprintf('%s/modelRDM/VGG16/modelRDM_VGG__%s__wr.mat', rootDir, layer_name), ...
            'R', 'stim_ID_num', 'im_table_sorted')

%         % weighted correlation - weight by neuron-wise variance & threshold
%         thres = quantile(w, 0.4); % lowest 40% in terms of variance will be zero
%         w(w<thres) = 0;
%         R = weightedcorrs(vgg_act_vector, w);
%         R(logical(eye(size(R)))) = nan;
%         R = R(IDorder, IDorder);
%         modelRDM.(layer_name).wthresr = R;
%         save(sprintf('%s/modelRDM/VGG16/modelRDM_VGG__%s__wthresr.mat', rootDir, layer_name), ...
%             'R', 'stim_ID_num', 'im_table_sorted')

        % normal correlation after PCA
        [~, score, eigval] = pca(vgg_act_vector');
        R = corr(score', 'type', 'Pearson', 'rows', 'pairwise');
        R(logical(eye(size(R)))) = nan;
        R = R(IDorder, IDorder);
        modelRDM.(layer_name).r_pca = R;
        save(sprintf('%s/modelRDM/VGG16/modelRDM_VGG__%s__r_pca.mat', rootDir, layer_name), ...
            'R', 'stim_ID_num', 'im_table_sorted')

        % weighted correlation after PCA - weight by eigenvalue
        R = weightedcorrs(score', eigval);
        R(logical(eye(size(R)))) = nan;
        R = R(IDorder, IDorder);
        modelRDM.(layer_name).wr_pca = R;
        save(sprintf('%s/modelRDM/VGG16/modelRDM_VGG__%s__wr_pca.mat', rootDir, layer_name), ...
            'R', 'stim_ID_num', 'im_table_sorted')
        
        % weighted cosine similarity after PCA - weight by eigenvalue
        R = cosineSimilarity(score .* sqrt(eigval'));
        R(logical(eye(size(R)))) = nan;
        R = R(IDorder, IDorder);
        modelRDM.(layer_name).wcos_pca = R;
        save(sprintf('%s/modelRDM/VGG16/modelRDM_VGG__%s__wcos_pca.mat', rootDir, layer_name), ...
            'R', 'stim_ID_num', 'im_table_sorted')
        
        fprintf('finished model RDM by %s \n', layer_name)
    end
end
toc
% Elapsed time is 280 seconds.
keyboard


%% visualize model RDMs - same method different layers
close all

% r = 'r'
% r = 'wr'
% r = 'wthresr'
% r = 'r_pca'
r = 'wr_pca'
Rvec = [];
layers = ["input" "conv1_1" "conv1_2" "conv3_3" "conv4_3" "conv5_3" "fc6" "fc8"];
counter = 0;
figure
for layer_name = layers
    counter = counter + 1;
    R = modelRDM.(layer_name).(r);
    Rvec = [Rvec R(:)];
    
    subplot(2,4,counter)
    imagesc(R)
    axis square
    colorbar
    title(sprintf('modelRDM by %s', layer_name), ...
        'Interpreter', "none")
    k=10;
    set(gca, ...
        'YTick', 1:k:height(im_table_sorted), ...
        'YTickLabel', im_table_sorted.name(1:k:end))
end

% correlation between RDMs
figure
RR = corr(Rvec, 'rows', 'pairwise');
imagesc(RR)
axis square
colorbar
t = num2cell(round(RR,2)); % extact values into cells
t = cellfun(@num2str, t, 'UniformOutput', false); % convert to string
n = length(layers);
text(repmat(1:n, 1, n), ...
    sort(repmat(1:n, 1, n))', ...
    t, ...
    'HorizontalAlignment', 'Center')
title(sprintf('VGG layers by method', r), ...
    'Interpreter','none');
xticks(1:n); xticklabels(layers); xtickangle(90);
yticks(1:n); yticklabels(layers);
set(gca, 'TickLabelInterpreter', 'none')



%% visualize model RDMs - same layer different methods
close all

layer_name = 'conv1_2'
counter = 0;
figure
Rvec = [];
methods = ["r" "wr" "wthresr" "r_pca" "wr_pca"];
for r = methods
    counter = counter + 1;
    R = modelRDM.(layer_name).(r);
    Rvec = [Rvec, R(:)];
    subplot(2,3,counter)
    imagesc(R)
    axis square
    colorbar
    title(sprintf('modelRDM %s', r), 'Interpreter', "none")
    k=10;
    set(gca, ...
        'YTick', 1:k:height(im_table_sorted), ...
        'YTickLabel', im_table_sorted.name(1:k:end))
end

% correlation between RDMs
figure
RR = corr(Rvec, 'rows', 'pairwise');
imagesc(RR)
axis square
colorbar
t = num2cell(round(RR,2)); % extact values into cells
t = cellfun(@num2str, t, 'UniformOutput', false); % convert to string
n = length(methods);
text(repmat(1:n, 1, n), ...
    sort(repmat(1:n, 1, n))', ...
    t, ...
    'HorizontalAlignment', 'Center')
title(sprintf('VGG %s different methods', layer_name), ...
    'Interpreter','none');
xticks(1:n); xticklabels(methods); xtickangle(90);
yticks(1:n); yticklabels(methods);
set(gca, 'TickLabelInterpreter', 'none')

