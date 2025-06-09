function [L, E] = SRPCA(M, lambda, gamma, max_gamma, rho, tol, max_iter, T)
% Slim transform-based tensor RPCA (SRPCA) method, solving Eq. (39).
% -----------------------------------------------------------
% Input:
%  M:          observation (M=L+E)
%  lambda:     regularization parameter
%  gamma:      penalty parameter in ADMM  
%  max_gamma:  maximum of gamma
%  rho:        rho>=1, ratio that is used to increase gamma
%  tol:        termination tolerance
%  max_iter:   maximum number of iterations
%  T:          transform matrix
% --------------------------------------------------------
% Output:
%  L:          low-rank component
%  E:          sparse component
% ---------------------------------------------
% version 1.0 - 05/30/2025
% Written by Lin Chen (lchen53@stevens.edu)

[n1,n2,n3] = size(M);
N3 = size(T,1)/size(T,2)*n3;
L = M;
TL = mode3_transform(L, T);
Z1 = zeros(n1,n2,N3);
Z2 = zeros(n1,n2,n3);
for iter = 1:max_iter
    N = shrL(TL - Z1/gamma, 1/(gamma*sqrt(N3)), N3);
    E = shrS(M - L - Z2/gamma, lambda/gamma);
    L = real(mode3_transform(N + Z1/gamma, T') + M - E - Z2/gamma)/2;
    TL = mode3_transform(L, T);

    dZ1 = N - TL;    
    dZ2 = L + E - M;
    chg = max([max(abs(dZ1(:))) max(abs(dZ2(:)))]);
    
    if chg < tol
     % fprintf('Iteration number of SRPCA is %d\n', iter);
     break;
    end 
    Z1 = Z1 + gamma*dZ1;
    Z2 = Z2 + gamma*dZ2; 
    gamma = min(rho*gamma, max_gamma);
end
end

function B = shrL(A, thr, N3)
    [U, S, V] = pagesvd(A, 'econ');
    B = zeros(size(A));
    for i = 1:N3
     B(:,:,i) = U(:,:,i) * max(S(:,:,i) - thr,0) * V(:,:,i)';
    end
end

function S_shr = shrS(S, mu)
    S_shr = sign(S) .* max(abs(S) - mu, 0);
end
