function [L] = logU(U)
% see reference:
% "Computing a logarithm of a unitary matrix with general spectrum"
% schur decomposition (algorithm 3)

[Q,T] = schur(U, 'complex');
D = diag(T) ./ abs(diag(T));

L = Q * diag(log(D)) * Q';
end

