function [T] = build_transform_matrix(trans_type, n3)
% Build up the transform matrix
% --------------------------------------------------------
% Input:
%  trans_type: transform type
%  n3:         size in the third dimension of a tensor
% --------------------------------------------------------
% Output:
%  T:          transform matrix
% --------------------------------------------------------
% version 1.0 - 05/30/2025
% Written by Lin Chen (lchen53@stevens.edu)

switch trans_type
    case 'DFT (M=1)'
        T = fft(eye(n3))/sqrt(n3);
    case 'DCT (M=1)'
        T = dct(eye(n3));
    case 'DFT (M=2)'
        T = fft(eye(2*n3))/sqrt(2*n3);
        T = T(:,1:n3);
    case 'DCT (M=2)'
        T = dct(eye(2*n3)); 
        T = T(:,1:n3);
    case 'FLT (M=2)' 
        frame = 0;
    case 'FLT (M=3) for Table II'   
        frame = 1;
    case 'FLT (M=3) for Table III'
        frame = 2;
    case 'FCT (M=4)'
        frame = 2;
    case 'FLT (M=5)' 
        frame = 3;
    case 'FCT (M=6)'
        frame = 3;
end

if exist('frame', 'var')
    level = 1;
    [D, ~] = GenerateFrameletFilter(frame);
    N3 = n3 * size(D,1) * level;
    for j = 1:n3
       basis = zeros(2,2,n3);
       basis(1,1,j) = 1;
       T0(:,:,:,j) = Fold(FraDecMultiLevel(Unfold(basis, [2, 2, n3], 3), D, level), [2, 2, N3], 3);
    end
    T = squeeze(T0(1,1,:,:));
    T = T/norm(T);
    if strcmp(trans_type, 'FCT (M=4)')
        T1 = dct(eye(n3), Type=4);
        T = [sqrt(n3)*T/sqrt(n3+1); T1/sqrt(n3+1)];
    end
    if strcmp(trans_type, 'FCT (M=6)')
        T1 = dct(eye(n3)); 
        T = [T; T1]/sqrt(2);
    end  
end
end