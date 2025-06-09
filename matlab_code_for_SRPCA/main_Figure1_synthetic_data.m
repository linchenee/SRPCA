clc 
clear 
close all
addpath(genpath(pwd));

n1 = 50; % size of the first dimension of a 3D tensor
n2 = 50; % size of the second dimension of a 3D tensor
n3 = 20; % size of the third dimension of a 3D tensor
n_min = min(n1,n2);
M_all = 10; % number of orthogonal transforms
r = 3; % tensor tubal rank

max_iter_gen = 1e3; % maximum number of iterations used to generate the rank-r tensor
min_iter_gen = 1e2; % minimum number of iterations used to generate the rank-r tensor
tol_gen = 1e-2; % tolerance parameter used to generate the rank-r tensor

num_trial = 500; % number of trials
stand_incoherence1 = zeros(M_all,num_trial); % standard incoherence parameter in Eq. (5)
stand_incoherence2 = zeros(M_all,num_trial); % standard incoherence parameter in Eq. (6)
stand_incoherence3 = zeros(M_all,num_trial); % standard incoherence parameter in Eq. (7)
joint_incoherence =  zeros(M_all,num_trial); % joint incoherence parameter in Eq. (8)

parfor trial = 1:num_trial
    fprintf('%d\n', trial);
    T_combined = zeros(n3*M_all,n3); % a slim transform composed of M random orthogonal transforms
    for m = 1:M_all
       rng((trial-1)*M_all+m);
       T_combined((m-1)*n3+1:m*n3,:) = RandOrthMat(n3); % generate M random orthogonal transforms
    end

    L = generate_tensor([n1,n2,n3], T_combined/norm(T_combined), r, min_iter_gen, max_iter_gen, tol_gen); % generate the rank-r tensor L
    [U_all, ~, V_all] = pagesvd(mode3_transform(L, T_combined/norm(T_combined)),'econ');
    U = U_all(:,1:r,:);
    V = V_all(:,1:r,:);
    UVt = zeros(n1,n2,n3*M_all);
    for i = 1:n3*M_all
     UVt(:,:,i) = U(:,:,i)*V(:,:,i)';
    end

    for M = 1:M_all
        N3 = M*n3;
        T = T_combined(1:N3,:)/norm(T_combined(1:N3,:));
        TtUVt = mode3_transform(UVt(:,:,1:M*n3), T'); % the term on the left-hand side of Eq. (8)
        infty_norm = max(abs(TtUVt(:))); % calculate the infinite norm
        joint_incoherence(M,trial) = infty_norm^2/N3*n1*n2*n3^2/r; % calculate Eq. (8)
    end

    norm1 = zeros(n1,n2,n3,M_all);
    norm2 = zeros(n1,n2,n3,M_all);
    norm3 = zeros(n1,n2,n3,M_all);
    for i = 1:n1
        for j = 1:n2
            for k = 1:n3
                eijk = zeros(n1,n2,n3); % the (i,j,k)th standard tensor basis
                eijk(i,j,k) = 1;
                temp = zeros(n1,n2,n3);
                norm0 = zeros(1,M_all);
                for M = 1:M_all
                    index_M = (M-1)*n3+1:M*n3;
                    T = T_combined(index_M,:);
                    Ps_T_eijk = Ps(mode3_transform(eijk, T), U(:,:,index_M), V(:,:,index_M)); % the term on the right-hand side of Eq. (13)
                    temp = temp + mode3_transform(Ps_T_eijk, pinv(T)); % calculate the sum on the right-hand side of Eq. (14) 

                    norm0(1,M) = norm(Ps_T_eijk(:), 'fro')^2; % calculate the term on the right-hand side of Eq. (13)
                    norm1(i,j,k,M) = mean(norm0(1, 1:M)); % calculate Eq. (5) according to Eq. (13)
                    norm2(i,j,k,M) = norm(temp(:)/M, 'fro')^2; % calculate Eq. (6) according to Eq. (14)
                    norm3(i,j,k,M) = max(abs(temp(:)/M)); % calculate Eq. (7) according to Eq. (14)
                end
            end
        end
    end

    for M = 1:M_all
        stand_incoherence1(M, trial) = max(norm1(:,:,:,M),[],'all')*n_min*n3/r; % calculate Eq. (5)
        stand_incoherence2(M, trial) = max(norm2(:,:,:,M),[],'all')*n_min*n3/r; % calculate Eq. (6)
        stand_incoherence3(M, trial) = max(norm3(:,:,:,M),[],'all')*n_min*n3/r; % calculate Eq. (7)
    end
end

figure(1); % Fig. 1(a)
plot(mean(stand_incoherence1, 2),'b','LineWidth',2,'Marker','o','MarkerSize',7); hold on
xlabel('\itM','fontsize',19,'FontName','Times new roman');
ylabel('\mu','fontsize',19,'FontName','Times new roman');
set(gca, 'FontName', 'Times new roman', 'FontSize', 18);
xlim([1, 10]);
xticks([1, 2, 4, 6, 8, 10]);
xticklabels({'1', '2', '4', '6', '8', '10'});

figure(2); % Fig. 1(b)
plot(mean(stand_incoherence2, 2),'b','LineWidth',2,'Marker','o','MarkerSize',7); hold on
xlabel('\itM','fontsize',19,'FontName','Times new roman');
ylabel('\mu','fontsize',19,'FontName','Times new roman');
set(gca, 'FontName', 'Times new roman', 'FontSize', 18);
xlim([1, 10]);
xticks([1, 2, 4, 6, 8, 10]);
xticklabels({'1', '2', '4', '6', '8', '10'});

figure(3); % Fig. 1(c)
plot(mean(stand_incoherence3, 2),'b','LineWidth',2,'Marker','o','MarkerSize',7); hold on
xlabel('\itM','fontsize',19,'FontName','Times new roman');
ylabel('\mu','fontsize',19,'FontName','Times new roman');
set(gca, 'FontName', 'Times new roman', 'FontSize', 18);
xlim([1, 10]);
xticks([1, 2, 4, 6, 8, 10]);
xticklabels({'1', '2', '4', '6', '8', '10'});

figure(4); % Fig. 1(d)
plot(mean(joint_incoherence, 2),'b','LineWidth',2,'Marker','o','MarkerSize',7);
xlabel('\itM','fontsize',19,'FontName','Times new roman');
ylabel('\mu','fontsize',19,'FontName','Times new roman');
set(gca, 'FontName', 'Times new roman', 'FontSize', 18);
xlim([1, 10]);
xticks([1, 2, 4, 6, 8, 10]);
xticklabels({'1', '2', '4', '6', '8', '10'});

function [Ps_X] = Ps(X, U, V)
% Project the tensor X onto the linear space S, defined in Eq. (4)
Ps_X = zeros(size(X));
for i = 1:size(U,3)
    UUt = U(:,:,i) * U(:,:,i)';
    VVt = V(:,:,i) * V(:,:,i)';
    Ps_X(:,:,i) = UUt * X(:,:,i) + X(:,:,i) * VVt - UUt * X(:,:,i) * VVt;
end
end
