function [T, Q] = tridiag(A)
%TRIDIAG    tridiagonalization of a real, symmetric matrix
% [T, Q] = tridiag(A) returns a tridiagonal matrix T and an orthogonal
% matrix Q such that T=Q'*A*Q. The algorithm uses Householder reflection.

if ~issymmetric(A) || ~all(isreal(A))
    error('Input matrix must be real and symmetric.');
end

sz = size(A,1);
T = A;
tol = 1e-12;
Q = eye(sz);

for n = 1 : sz-2
    e = [-sign(T(n+1,n)); zeros(sz-n-1, 1)]; % sign convention?
    v = T(n+1:end, n) - vecnorm(T(n+1:end, n)) * e;
    v = v / vecnorm(v);
    
    T(1:n, n+1:end) = rH(T(1:n, n+1:end), v);
    T(n+1:end,1:n) = T(1:n, n+1:end)';
    T(n+1:end, n+1:end) = lH(rH(T(n+1:end, n+1:end), v), v);
    
    Q(1:n, n+1:end) = rH(Q(1:n, n+1:end), v);
    Q(n+1:end, n+1:end) = rH(Q(n+1:end, n+1:end), v);
end

T(abs(T(:)) < tol) = 0;
T = (T + T') / 2;
end

% right-multiply Householder matrix
function [ M1 ] = rH(M0, v)
M1 = M0 - 2*(M0*v)*v';
end

% left-multiply Householder matrix
function [ M1 ] = lH(M0, v)
M1 = M0 - 2*v*(v'*M0);
end

