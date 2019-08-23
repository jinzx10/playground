function [ S ] = cart2nsph(C)
%CART2NSPH   transform Cartesian coordinates to n-spherical coordinate
% S = CART2NSPH(C) treats each column of C as a Cartesian coordinate and
% transforms it to an n-spherical coordinate
%
% [x_1, x_2, ..., x_n].' -->  [phi_1, phi_2, ..., phi_{n-1}, r].'
%
% phi_1     = acos( x_1 / sqrt(x_n^2 + ... + x_1^2) )
% phi_2     = acos( x_2 / sqrt(x_n^2 + ... + x_2^2) )
% .
% .
% phi_{n-2} = acos( x_{n-2} / sqrt(x_n^2 + x_{n-1}^2 + x_{n-2}^2) )
% phi_{n-1} = (x_n >= 0) * acos( x_{n-1} / sqrt(x_n^2 + x_{n-1}^2) ) + ...
%             ~(x_n >= 0) * ( 2*pi - acos( x_{n-1} / sqrt(x_n^2 + x_{n-1}^2) ) )
%
% r         = sqrt( x_1^2 + ... + x_n^2 )
%
%
% x_1     = r * cos(phi_1)
% x_2     = r * sin(phi_1) * cos(phi_2)
% x_3     = r * sin(phi_1) * sin(phi_2) * cos(phi_3)
% .
% .
% x_{n-1} = r * sin(phi_1) * ... * sin(phi_{n-2}) * cos(phi_{n-1})
% x_n     = r * sin(phi_1) * ... * sin(phi_{n-2}) * sin(phi_{n-1})
%
% specifically,
% 2-d  [x,y].' --> [phi, r].'
% 3-d  [z,x,y].' --> [polar, azimuth, r].'

if ~isreal(C)
    error('Cartesian coordinates must be real.')
end

[dim, num] = size(C);
S = zeros(dim, num);

% radial part
S(end,:) = vecnorm(C, 2, 1);

% the first dim-2 angles range from [0, pi]
for i = 1 : dim-2
    S(i, :) = acos( C(i,:) ./ vecnorm(C(i:end, :), 2, 1) );
end

% the last angle ranges from [0, 2*pi)
if dim > 1
    S(end-1, :) = acos( C(end-1,:) ./ vecnorm(C(end-1:end, :), 2, 1) );
    S(end-1, C(end,:)<0) = 2*pi - S(end-1, C(end,:)<0);
end

S(~isfinite(S)) = 0;

end