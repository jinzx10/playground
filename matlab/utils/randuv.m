function [ u ] = randuv(dim, num)
%RANDUV   uniformly distributed random unit vector
% u = randuv(n) generates one uniformly distributed random unit vector in
% n-dimension.
% u = randuv(n, m) returns a n-by-m matrix with each column a uniformly
% distributed unit vector.

if nargin == 1
    num = 1;
end

u = randn(dim, num);
u = u ./ vecnorm(u, 2, 1);

end