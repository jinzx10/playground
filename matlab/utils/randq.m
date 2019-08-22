function [ q ] = randq( dim )
%RANDQ   uniformly distributed random orthogonal matrix
% q = randq(dim)

q = randn(dim);
[q, r] = qr(q);

idx_flip = find(diag(r) < 0);
q(:, idx_flip) = -q(:, idx_flip);

% presumably qr in matlab is computed by Householder reflection,
% which does not guarantee that all diagonal elements of r are positive,
% so a sign adjustment is needed.
% Gram-Schmidt orthogonalization would guarantee that sign convention,
% but it is numerically unstable.

end
