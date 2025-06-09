function [X] = generate_tensor(dim, T, r, min_iter, max_iter, tol)
    n1 = dim(1);
    n2 = dim(2);
    n3 = dim(3);
    Tp = pinv(T);
    M = size(T,1)/size(T,2);
    rones = r*ones(1,M);
    X = randn(n1,n2,n3);
    for iter = 1 : max_iter
        Xt = mode3_transform(X, T);
        if mod(iter,25) == 0 && iter >= min_iter
            rt = tensor_tubal_rank(Xt, M, n3, tol);
            if sum(abs(rt - rones)) == 0 
                % fprintf('Convergence, iter=%d\n',iter);
                break;
            end
        end
        Xt = truncates(Xt, r);
        X = mode3_transform(Xt, Tp);
    end
end

function [Xt] = truncates(Xt, r)
    [U, S, V] = pagesvd(Xt, 'econ');
    for k = 1 : size(Xt,3)
        Xt(:, :, k) = U(:, 1:r, k) * S(1:r, 1:r, k) * V(:, 1:r, k)';
    end
end

function [r] = tensor_tubal_rank(X, M, n3, tol)
    S = pagesvd(X);
    r = zeros(1,M);
    for m = 1:M
        max_singular_value = max(S(:,:,(m-1)*n3+1:m*n3), [], 'all');
        for i = 1 : n3
            r(1,m) = max(sum(S(:, :, (m-1)*n3+i) > tol * max_singular_value), r(1,m));
        end
    end
end