import numpy as np
import matplotlib.pyplot as plt

def try_once(sz, N_nao, N_pswfc, N_occ):
    # NAO
    nao = np.random.randn(sz, N_nao)
    for i in range(N_nao):
        nao[:,i] /= np.sqrt(np.linalg.norm(nao[:,i]))
    S_nao = nao.T @ nao
    
    # PSWFC
    pswfc = np.random.randn(sz, N_pswfc)
    for i in range(N_pswfc):
        pswfc[:,i] /= np.sqrt(np.linalg.norm(pswfc[:,i]))
    
    
    # OCC
    nao_to_occ = np.random.randn(N_nao, N_occ)
    occ = nao @ nao_to_occ
    occ, _ = np.linalg.qr(occ)
    #print('orth occ: ', np.linalg.norm( occ.T @ occ - np.eye(N_occ)  ))
    
    
    # step-1
    pswfc_t = nao @ np.linalg.solve(S_nao, nao.T @ pswfc)
    
    # step-2
    pswfc_occ = occ @ (occ.T @ pswfc)
    
    # step-3
    pswfc_vir = pswfc_t - pswfc_occ
    
    # step-4
    S_pswfc_vir = pswfc_vir.T @ pswfc_vir
    
    val, vec = np.linalg.eigh(S_pswfc_vir)
    
    pswfc_vir_orth = pswfc_vir @ vec[:,-(N_pswfc-N_occ):] @ np.diag(1/np.sqrt(val[-(N_pswfc-N_occ):]))
    
    #print('orth pswfc_vir_orth', np.linalg.norm(pswfc_vir_orth.T @ pswfc_vir_orth - np.eye(N_pswfc-N_occ)))
    
    # step-5
    xi = np.concatenate((occ, pswfc_vir_orth), axis=1)
    #print('orth xi', np.linalg.norm( xi.T @ xi - np.eye(N_pswfc)  ))
    
    # step-6
    qo = xi @ (xi.T @ pswfc)
    
    S_qo = qo.T @ qo
    
    val, vec = np.linalg.eigh(S_qo)
    #print('S_qo min eival = %8.5e'%(np.min(val)))

    return np.min(val)


N_trials = 1000
szs = [100, 500, 2500]

count = [np.zeros(N_trials), np.zeros(N_trials), np.zeros(N_trials)]

N_nao = 31
N_pswfc = 14
N_occ = 11
for i, sz in enumerate(szs):
    for it in range(N_trials):
        count[i][it] = np.log10(try_once(sz, N_nao, N_pswfc, N_occ))


fig, ax = plt.subplots(1,3, figsize=(12,4))

for i, sz in enumerate(szs):
    ax[i].hist(count[i], bins=20, density=True)

plt.show()



    




