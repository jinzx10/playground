function [ x, info, hst ] = broydenroot(f, x0, varargin)
%BROYDENROOT Broyden's quasi-Newton root-finding method
% x = BROYDENROOT(f, x0) returns the root of f with initial guess x0.
%
% x = BROYDENROOT(f, x0, method) specifies the method for updating
% Jacobian/inverse Jacobian. The possible choices are
% 'good': update Jacobian. This is the default.
% 'inv': update inverse Jacobian. This is a variant of the 'good' method
%        using Sherman–Morrison formula.
% 'bad': update inverse Jacobian with the "bad Broyden's method".
% For 'inv' and 'bad', the number of equations must be equal to the number
% of variables.
%
% x = BROYDENROOT(f, x0, method, tol) also set the tolerance to tol
% so that the iteration stops when norm(f(x), 'fro') < tol. The default
% tolerance is 1e-12.
%
% x = BROYDENROOT(f, x0, method, tol, max_iter) also sets the maximum 
% number of iterations. The default max_iter is 50.
% 
% [x, info] = BROYDENROOT(...) returns info = 1 if it fails to find the
% root and info = 0 otherwise.
% 
% [x, info, hst] = BROYDENROOT(...) also returns the history of iterations.
%
%
% Reference:
% Broyden, Charles G. "A class of methods for solving nonlinear 
% simultaneous equations.", Math. Comp. 19 (1965), 577-593

narginchk(2,5);
defaults = {"good", 1e-12, 50};
defaults(1:nargin-2) = varargin;
[method, tol, max_iter] = defaults{:};

info = 0;

mtd = -1;
switch method
    case "good"
        mtd = 0;
    case "inv"
        mtd = 1;
    case "bad"
        mtd = 2;
end

if mtd < 0
    error("Unrecognized method '" + method + "'");
end

x = x0;
fx = f(x);
if norm(fx(:)) < tol
    return;
end

sz_x = size(x);
len_x = numel(x);
len_f = numel(fx);

if nargout > 2
    hst.x = zeros(max_iter, len_x);
    hst.f = zeros(max_iter, len_f);
    
    hst.x(1,:) = x';
    hst.f(1,:) = fx(:)';
end

% compute the initial Jacobian by finite difference
J = zeros(len_f, len_x);
delta = 1e-6 * max(1, sqrt(norm(x(:))));
for i = 1 : len_x
    dxi = zeros(sz_x);
    dxi(i) = delta;
    df = f(x+dxi) - fx;
    J(:, i) = df(:) / delta;
end

if mtd > 0
    if len_x == len_f
        invJ = inv(J);
    else
        error('The number of equations is not equal to the number of variables.');
    end
end

for counter = 1 : max_iter
    if mtd > 0
        dx = -invJ * fx(:);
    else
        dx = -J \ fx(:);
    end
    
    x = x + reshape(dx, sz_x);
    fx_new = f(x);
    
    if nargout > 2
        hst.x(counter+1,:) = x(:)';
        hst.f(counter+1,:) = fx_new(:)';
    end
    
    if norm(fx_new(:)) < tol
        if nargout > 2
            hst.x(counter+2:end,:) = [];
            hst.f(counter+2:end,:) = [];
        end
        return;
    end
    
    df = fx_new - fx;
    fx = fx_new;
    
    % Broyden's update
    switch mtd
        case 2
            invJ = invJ + (dx-invJ*df(:)) / (df(:)'*df(:)) * df(:)';
        case 1
            invJ = invJ + (dx-invJ*df(:)) / (dx'*invJ*df(:)) * dx' * invJ;
        otherwise
            J = J + (df(:)-J*dx) / (dx'*dx) * dx';
    end
end

info = 1;
warning('Broyden''s method fails to find the root.')

end

