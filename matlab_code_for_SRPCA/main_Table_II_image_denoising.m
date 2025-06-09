clc
clear
close all
addpath(genpath(pwd));

image = load('../data/YaleB_testCR20.mat').YaleB_testCR20;
n1 = 192;
n2 = 168; 
n3 = 64;
num_subject = 6; % number of subjects

%% Please select one of the following transform types
% transform_type = 'DFT (M=1)';
transform_type = 'DCT (M=1)';
% transform_type = 'DFT (M=2)';
% transform_type = 'DCT (M=2)';
% transform_type = 'FLT (M=3) for Table II';
% transform_type = 'FLT (M=5)';
% transform_type = 'FCT (M=6)';

T = build_transform_matrix(transform_type, n3); % transform matrix
%% parameters of SRPCA algorithm
lambda = 1 / sqrt(max(n1,n2)*n3);
gamma = 1e-3;
max_gamma = 1e8;
rho = 1.1;
tol = 1e-7;
max_iter = 500;

PSNRs = zeros(1,num_subject); 
parfor i = 1:num_subject
    L_groundtruth = squeeze(image(i, 1, :, :, 1:n3)); % ground-truth low-rank tensor   
    M = squeeze(image(i, 2, :, :, 1:n3)); % M: observation tensor = low-rank tensor + sparse tensor
    L = SRPCA(M, lambda, gamma, max_gamma, rho, tol, max_iter, T);
    PSNRs(1,i) = mPSNR(L*255, L_groundtruth); 
end

fprintf('PSNR = %.2f dB\n', mean(PSNRs));


