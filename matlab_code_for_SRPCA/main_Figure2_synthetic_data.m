clc
clear
close all
addpath(genpath(pwd));

n1 = 50; % size of the first dimension of a 3D tensor
n2 = 50; % size of the second dimension of a 3D tensor
n3 = 20; % size of the third dimension of a 3D tensor
n_min = min(n1,n2);
M_all = 25; % number of orthogonal transforms
r = 3; % tensor tubal rank

max_iter_gen = 1e3; % maximum number of iterations used to generate the rank-r tensor
min_iter_gen = 1e2; % minimum number of iterations used to generate the rank-r tensor
tol_gen = 1e-2; % tolerance parameter used to generate the rank-r tensor

num_trial = 1500; % number of trials
stand_incoherence1 = zeros(M_all,num_trial); % standard incoherence parameter in Eq. (5)
stand_incoherence2 = zeros(M_all,num_trial); % standard incoherence parameter in Eq. (6)
stand_incoherence3 = zeros(M_all,num_trial); % standard incoherence parameter in Eq. (7)
joint_incoherence =  zeros(M_all,num_trial); % joint incoherence parameter in Eq. (8)

for trial = 1:num_trial
    tic
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

    for M = 1:2:M_all
        rng(trial*M);
        select_transform = sort(randperm(M_all,M)); % select M out of M_all orthogonal transforms
        temp1 = (select_transform-1)*n3+1;
        temp2 = select_transform*n3;
        select_index = zeros(1,M*n3);
        for m = 1:M
            select_index(1,(m-1)*n3+1:m*n3) = temp1(m):temp2(m);
        end

        N3 = M*n3;
        T = T_combined(select_index,:)/norm(T_combined(select_index,:));
        TtUVt = mode3_transform(UVt(:,:,select_index), T'); % the term on the left-hand side of Eq. (8)
        infty_norm = max(abs(TtUVt(:))); % calculate the infinite norm
        joint_incoherence(M,trial) = infty_norm^2/N3*n1*n2*n3^2/r; % calculate Eq. (8)
    end

    norm1 = zeros(n1,n2,n3,M_all);
    norm2 = zeros(n1,n2,n3,M_all);
    norm3 = zeros(n1,n2,n3,M_all);
    parfor i = 1:n1
        for j = 1:n2
            for k = 1:n3
                eijk = zeros(n1,n2,n3); % the (i,j,k)th standard tensor basis
                eijk(i,j,k) = 1;
                Ps_T_eijk = zeros(n1,n2,n3,M_all); % the term on the right-hand side of Eq. (13)
                Tpinv_Ps_T_eijk = zeros(n1,n2,n3,M_all); % the term on the right-hand side of Eq. (14)
                for m = 1:M_all
                    index_m = (m-1)*n3+1:m*n3;
                    T = T_combined(index_m,:);
                    Ps_T_eijk(:,:,:,m) = Ps(mode3_transform(eijk, T), U(:,:,index_m), V(:,:,index_m)); 
                    Tpinv_Ps_T_eijk(:,:,:,m) = mode3_transform(Ps_T_eijk(:,:,:,m), pinv(T)); 
                end
                for M = 1:2:M_all
                    rng(trial*M);
                    select_transform = sort(randperm(M_all,M)); % select M out of M_all orthogonal transforms
                    norm0 = zeros(1,M);
                    temp = zeros(n1,n2,n3);
                    for m = 1:M
                        select_m = Ps_T_eijk(:,:,:,select_transform(m));
                        norm0(1,m) = norm(select_m(:),'fro')^2; % calculate the term on the right-hand side of Eq. (13)
                        temp = temp + Tpinv_Ps_T_eijk(:,:,:,select_transform(m)); % calculate the sum on the right-hand side of Eq. (14) 
                    end
                    norm1(i,j,k,M) = mean(norm0); % calculate Eq. (5) according to Eq. (13)
                    norm2(i,j,k,M) = norm(temp(:)/M,'fro')^2; % calculate Eq. (6) according to Eq. (14)
                    norm3(i,j,k,M) = max(abs(temp(:)/M)); % calculate Eq. (7) according to Eq. (14)
                end
            end
        end
    end

    for M = 1:2:M_all
        stand_incoherence1(M, trial) = max(norm1(:,:,:,M),[],'all')*n_min*n3/r; % calculate Eq. (5)
        stand_incoherence2(M, trial) = max(norm2(:,:,:,M),[],'all')*n_min*n3/r; % calculate Eq. (6)
        stand_incoherence3(M, trial) = max(norm3(:,:,:,M),[],'all')*n_min*n3/r; % calculate Eq. (7)
    end
    toc
end

figure(1); % Fig. 2(a)
plot(mean(stand_incoherence1(1:2:M_all,:),2)); hold on
xlabel('\itM','fontsize',19,'FontName','Times new roman');
ylabel('\mu','fontsize',19,'FontName','Times new roman');
set(gca, 'FontName', 'Times new roman', 'FontSize', 18);
xlim([1, 13]);
xticks([1, 3, 5, 7, 9, 11, 13]);
xticklabels({'1', '5', '9', '13', '17', '21', '25'});

figure(2); % Fig. 2(b)
plot(mean(stand_incoherence2(1:2:M_all,:),2)); hold on
xlabel('\itM','fontsize',19,'FontName','Times new roman');
ylabel('\mu','fontsize',19,'FontName','Times new roman');
set(gca, 'FontName', 'Times new roman', 'FontSize', 18);
xlim([1, 13]);
xticks([1, 3, 5, 7, 9, 11, 13]);
xticklabels({'1', '5', '9', '13', '17', '21', '25'});

figure(3); % Fig. 2(c)
plot(mean(stand_incoherence3(1:2:M_all,:),2)); hold on
xlabel('\itM','fontsize',19,'FontName','Times new roman');
ylabel('\mu','fontsize',19,'FontName','Times new roman');
set(gca, 'FontName', 'Times new roman', 'FontSize', 18);
xlim([1, 13]);
xticks([1, 3, 5, 7, 9, 11, 13]);
xticklabels({'1', '5', '9', '13', '17', '21', '25'});

figure(4); % Fig. 2(d)
plot(mean(joint_incoherence(1:2:M_all,:),2),'b','LineWidth',2,'Marker','o','MarkerSize',7);
xlabel('\itM','fontsize',19,'FontName','Times new roman');
ylabel('\mu','fontsize',19,'FontName','Times new roman');
set(gca, 'FontName', 'Times new roman', 'FontSize', 18);
xlim([1, 13]);
xticks([1, 3, 5, 7, 9, 11, 13]);
xticklabels({'1', '5', '9', '13', '17', '21', '25'});

function [Ps_X] = Ps(X, U, V)
% Project the tensor X onto the linear space S, defined in Eq. (4)
Ps_X = zeros(size(X));
for i = 1:size(U,3)
    UUt = U(:,:,i) * U(:,:,i)';
    VVt = V(:,:,i) * V(:,:,i)';
    Ps_X(:,:,i) = UUt * X(:,:,i) + X(:,:,i) * VVt - UUt * X(:,:,i) * VVt;
end
end
