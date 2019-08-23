function [ C ] = fdcoef(N, varargin)
%FDCOEF Finite difference coefficients
% C = FDCOEF(N) returns the lowest-order standard finite difference 
% coefficients for the N-th order derivative to a row vector.
%
% C = FDCOEF(N, 'forward') and C = FDCOEF(N, 'backward') returns the
% lowest-order forward and backward coefficients.
%
% C = FDCOEF(N, p) returns the coefficients for the N-th order derivative 
% with a grid -p:p.
% floor(2*p)+1 should be larger than N.
%
% C = FDCOEF(N, p, 'forward') and C = FDCOEF(N, p, 'backward') returns
% forward and backward coefficients with a grid 0:p and -p:0 respectively.
% floor(p)+1 should be larger than N.
%
% C = FDCOEF(N, P) returns the finite difference coefficients for the N-th 
% order derivative with grids specified by a vector P, such that
% sum_i C(i)*f(x+P(i)*h)/h^N = d^N f(x)
% length(P) should be larger than N.
%
% C = FDCOEF(..., 'symb') returns the coefficients in symbolic form.
%
% See also tq_stencil

% ensure derivative order is a positive integer
if ~isscalar(N) || ~isfinite(N) || ~isreal(N) || mod(N, 1) || N < 1
    error('Derivative order should be a positive integer')
end

% check whether symbolic or not
symb = 0;
if ~isempty(varargin) && strcmp(varargin{end}, 'symb')
    symb = 1;
    varargin(end) = [];
end

% find whether finite difference is symmetric, forward or backward
type = 'symmetric';
if ~isempty(varargin) && ( strcmp(varargin{end}, 'forward') || ...
        strcmp(varargin{end}, 'backward') )
    type = varargin{end};
    varargin(end) = [];
end

if isempty(varargin)
    switch type
        case 'symmetric'
            P = -ceil(N/2):ceil(N/2);
        case 'forward'
            P = 0:N;
        case 'backward'
            P = -N:0;
    end
else
    if length(varargin) ~= 1 || ~isnumeric(varargin{1})
        error('Unrecognized arguments');
    end
    
    P = varargin{1};
    if ~isvector(P) || ~all(isfinite(P)) || ~all(isreal(P))
        error('Invalid grid range')
    end
    
    if isscalar(P)
        if P < 0
            error('Grid range should be positive');
        end
        switch type
            case 'symmetric'
                P = -P:P;
            case 'forward'
                P = 0:P;
            case 'backward'
                P = -P:0;
        end
    else
        if iscolumn(P)
            P = P.';
        end
        if ~strcmp(type, 'symmetric')
            disp("option '"+type+"' is ignored")
        end
    end
end

M = length(P);
if M <= N
    error('Insufficient number of grids for the given derivative order')
end

z = zeros(M, 1);

if symb
    z = sym(z);
    P = sym(P);
end

z(N+1) = 1;
D = P.^((0:M-1)') ./ factorial((0:M-1)');
%z(N+1) = factorial(N);
%D = P.^((0:M-1)');

C = (D\z).';

end
