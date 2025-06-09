function A_transformed = mode3_transform(A, T)
% perform the mode-3 linear transform for the third-order tensor A
    assert(size(A, 3) == size(T, 2))
    A_transformed = reshape((T * reshape(A, size(A, 1) * size(A, 2), size(A, 3)).').', size(A, 1), size(A, 2), size(T, 1));
end