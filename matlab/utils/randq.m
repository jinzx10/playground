function [ q ] = randq( dim )
%RANDQ   uniformly distributed random orthogonal matrix
% q = randq(dim)

q = randn(dim);
[q, r] = qr(q);

idx_flip = find(diag(r) < 0);
q(:, idx_flip) = -q(:, idx_flip);

end