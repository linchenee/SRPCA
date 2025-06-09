function [F1_measure] = F1(mask, mask_GT)
% F_1-measure criterion to evaluate the performance of moving object detection
% -----------------------------------------------------------
% Input:
%  mask:       binary mask
%  mask_GT:    binary mask ground-truth
% --------------------------------------------------------
% Output:
%  F1_measure: F_1-measure score 
% ---------------------------------------------
% version 1.0 - 05/30/2025
% Written by Lin Chen (lchen53@stevens.edu)

[~,~,n3] = size(mask);
F_measure = zeros(1,n3);
for i = 1:n3
    temp1 = mask(:,:,i);
    temp2 = mask_GT(:,:,i);
    TP = sum(sum(logical(temp1) & logical(temp2))); % True Positives
    FP = sum(sum(logical(temp1) & ~logical(temp2))); % False Positives
    FN = sum(sum(~logical(temp1) & logical(temp2))); % False Negatives
    recall = TP / (TP + FN);
    precision = TP / (TP + FP);
    if recall ~= 0 && precision ~= 0
        F_measure(1,i) = 2 * recall * precision / (recall + precision);
    else
        F_measure(1,i) = 0;
    end  
    if isnan(F_measure(1,i))
        F_measure(1,i) = 0;  
    end
end
F1_measure = sum(F_measure)/n3;
end