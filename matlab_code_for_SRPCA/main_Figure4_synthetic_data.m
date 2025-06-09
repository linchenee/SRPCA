clc
clear
close all
addpath(genpath(pwd));
rng(0);

n1 = 50; % size of the first dimension of a 3D tensor
n2 = 50; % size of the second dimension of a 3D tensor
n3 = 20; % size of the third dimension of a 3D tensor

num_trial = 100; % number of trials
list_rank = 1:13; % tensor tubal rank
list_sparsity = linspace(0.01,0.20,16); % sparsity of outliers

min_iter_gen = 1e2; % minimum number of iterations used to generate the rank-r tensor
max_iter_gen = 1e3; % maximum number of iterations used to generate the rank-r tensor
tol_gen = 1e-2; % tolerance parameter used to generate the rank-r tensor

%% parameters of SRPCA algorithm
gamma = 1e-3;
max_gamma = 1e8;
rho = 1.1;
tol = 1e-8;
max_iter = 500;
lambda = 1 / sqrt(n1*n3);

error1 = zeros(numel(list_rank),numel(list_sparsity),num_trial);
error2 = zeros(numel(list_rank),numel(list_sparsity),num_trial);
error3 = zeros(numel(list_rank),numel(list_sparsity),num_trial);
error4 = zeros(numel(list_rank),numel(list_sparsity),num_trial);

for i_rank = 1:numel(list_rank)
  r = list_rank(i_rank);
  for i_sparsity = 1:numel(list_sparsity)
      sparsity = list_sparsity(i_sparsity);
      fprintf('rank=%d, sparsity=%f\n', r, sparsity);
      parfor i_trial = 1:num_trial      
           T1 = RandOrthMat(n3); % generate a random orthogonal transform
           T2 = RandOrthMat(n3);
           T3 = RandOrthMat(n3);
           T4 = RandOrthMat(n3);
           T_combined = [T1; T2; T3; T4]/2; % a slim transform composed of 4 random orthogonal transforms

           L = generate_tensor([n1,n2,n3], T_combined, r, min_iter_gen, max_iter_gen, tol_gen); % generate the rank-r tensor L
           index_outlier = find((rand(n1,n2,n3) < sparsity));
           L_with_outlier = L; % L_with_outlier: observation tensor = low-rank tensor + sparse tensor
           L_with_outlier(index_outlier) = sign(rand(numel(index_outlier), 1)-0.5);
         
           T = T4; T = T/norm(T);
           L1 = SRPCA(L_with_outlier, lambda, gamma, max_gamma, rho, tol, max_iter, T);
           error1(i_rank,i_sparsity,i_trial) = norm(L1(:)-L(:))/norm(L(:));

           T = [T1;T2]; T = T/norm(T);
           L2 = SRPCA(L_with_outlier, lambda, gamma, max_gamma, rho, tol, max_iter, T);
           error2(i_rank,i_sparsity,i_trial) = norm(L2(:)-L(:))/norm(L(:));

           T = [T1;T2;T3]; T = T/norm(T);
           L3 = SRPCA(L_with_outlier, lambda, gamma, max_gamma, rho, tol, max_iter, T);
           error3(i_rank,i_sparsity,i_trial) = norm(L3(:)-L(:))/norm(L(:));

           T = [T1;T2;T3;T4]; T = T/norm(T);
           L4 = SRPCA(L_with_outlier, lambda, gamma, max_gamma, rho, tol, max_iter, T);
           error4(i_rank,i_sparsity,i_trial) = norm(L4(:)-L(:))/norm(L(:));
      end
 end
end

thr = 1e-3;
figure(1); % Fig.4(a) 
imagesc(mean(double(error1<thr),3));
set(gca,'YDir','normal');
colorbar;
clim([0,1]);
xlabel('Sparsity','fontsize',16,'FontName','Times new roman');
ylabel('Rank','fontsize',16,'FontName','Times new roman');
xlim([0.5, 16.5]);
xticks([1, 8, 16]);
xticklabels({'0.01', '0.1', '0.2'});
ylim([0.5, 13.5]);
yticks([1, 5, 9, 13]);
set(gca, 'FontName', 'Times New Roman', 'FontSize', 13);

figure(2); % Fig.4(b) 
imagesc(mean(double(error2<thr),3));
set(gca,'YDir','normal');
colorbar;
clim([0,1]);
xlabel('Sparsity','fontsize',16,'FontName','Times new roman');
ylabel('Rank','fontsize',16,'FontName','Times new roman');
xlim([0.5, 16.5]);
xticks([1, 8, 16]);
xticklabels({'0.01', '0.1', '0.2'});
ylim([0.5, 13.5]);
yticks([1, 5, 9, 13]);
set(gca, 'FontName', 'Times New Roman', 'FontSize', 13);

figure(3); % Fig.4(c) 
imagesc(mean(double(error3<thr),3));
set(gca,'YDir','normal');
colorbar;
clim([0,1]);
xlabel('Sparsity','fontsize',16,'FontName','Times new roman');
ylabel('Rank','fontsize',16,'FontName','Times new roman');
xlim([0.5, 16.5]);
xticks([1, 8, 16]);
xticklabels({'0.01', '0.1', '0.2'});
ylim([0.5, 13.5]);
yticks([1, 5, 9, 13]);
set(gca, 'FontName', 'Times New Roman', 'FontSize', 13);

figure(4); % Fig.4(d) 
imagesc(mean(double(error4<thr),3));
set(gca,'YDir','normal');
colorbar;
clim([0,1]);
xlabel('Sparsity','fontsize',16,'FontName','Times new roman');
ylabel('Rank','fontsize',16,'FontName','Times new roman');
xlim([0.5, 16.5]);
xticks([1, 8, 16]);
xticklabels({'0.01', '0.1', '0.2'});
ylim([0.5, 13.5]);
yticks([1, 5, 9, 13]);
set(gca, 'FontName', 'Times New Roman', 'FontSize', 13);