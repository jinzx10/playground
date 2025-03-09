import numpy as np
import scipy


t = np.array([[0.0, 0.1, 0.1],
              [0.1, 0.0, 0.1], 
              [0.1, 0.2, 0.5],
              ])

H = lambda x : np.array([[t[0,0] + t[1,1] + x, t[1,2]         , t[0,2]],
                         [t[2,1]             , t[0,0] + t[2,2], t[0,1]],
                         [t[2,0]             , t[1,0]         , t[1,1] + t[2,2]],
                         ], dtype=np.complex128)

U_abs = 1.0
angle = np.linspace(0, 2*np.pi, 5000)

_, vec_last = np.linalg.eig(H(U_abs * np.exp(1j * angle[0])))

cumprod = np.array([1.0, 1.0, 1.0], dtype=np.complex128)

for theta in angle[1:]:
    U = U_abs * np.exp(1j * theta)
    val, vec = np.linalg.eig(H(U))

    # align vec so that it is closest to the previous vec
    ovl = vec_last.conj().T @ vec

    # find the permutation that maximizes the overlap
    perm = np.argmax(np.abs(ovl), axis=1)

    # permute the columns of vec
    vec = vec[:, perm]
    cumprod *= np.diag(ovl)

    vec_last = vec
    
print(cumprod)
print(np.log(cumprod))
print(np.abs(cumprod))




