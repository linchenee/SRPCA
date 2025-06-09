clc
clear
close all
addpath(genpath(pwd));

load('../data/SBI.mat');
[row, column, frame_plus1, channel, num_video] = size(SBI);
frame = frame_plus1 - 1; % frame: number of frames in a video sequence

%% Please select one of the following transform types
% transform_type = 'DFT (M=1)';
transform_type = 'DCT (M=1)';
% transform_type = 'DFT (M=2)';
% transform_type = 'DCT (M=2)';
% transform_type = 'FLT (M=2)';
% transform_type = 'FLT (M=3) for Table III';
% transform_type = 'FCT (M=4)';

n1 = row*column; % size of the first dimension of a 3D tensor
n2 = frame;      % size of the second dimension of a 3D tensor
n3 = channel;    % size of the third dimension of a 3D tensor

T = build_transform_matrix(transform_type, n3); % transform matrix
%% parameters of SRPCA algorithm
lambda = 1 / sqrt(max(n1,n2)*n3);
gamma = 1e-3;
max_gamma = 1e8;
rho = 1.1;
tol = 1e-6;
max_iter = 500;
thr = 25; % hard-threshold for the moving object detection

PSNRs = zeros(num_video,1); 
F1_measures = zeros(num_video,1);
parfor i = 1:num_video
    Input = reshape(SBI(:,:,1:frame,:,i), [row*column,frame,channel]); % original video
    BG_GT = reshape(SBI(:,:,frame+1,:,i), [row,column,channel]); % background ground-truth
    BG = SRPCA(Input/255, lambda, gamma, max_gamma, rho, tol, max_iter, T); % background extraction
    
    temp1 = repmat(BG_GT, [1,1,frame]);
    temp2 = reshape(permute(BG*255, [1,3,2]), [row,column,channel*frame]);
    PSNRs(i,1) = mPSNR(temp1, temp2);
    
    FG_GT = zeros(row,column,frame); % foreground ground-truth
    FG = zeros(row,column,frame); % foreground detection
    for j = 1:frame
     FG_GT(:,:,j) = abs(rgb2gray(uint8(BG_GT)) - rgb2gray(uint8(reshape(Input(:,j,:), [row,column,channel])))) > thr;
     FG(:,:,j) = rgb2gray(reshape(uint8(BG(:,j,:)*255 - Input(:,j,:)), [row,column,channel])) > thr;
    end
    F1_measures(i,1) = F1(FG, FG_GT);
end

fprintf('PSNR = %.2f dB\n', mean(PSNRs));
fprintf('F1-measure = %.2f%%\n', mean(F1_measures*100));
