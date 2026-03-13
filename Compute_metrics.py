import numpy as np



def weighted_jaccard(A, T, V,  fraq=0.15):
    
    A = np.asarray(A, dtype=float).ravel()
    T = np.asarray(T, dtype=float).ravel()

    Acomp = A.copy()
    Acomp[Acomp<fraq*np.max(Acomp)]=0

    mA = np.sum(Acomp * V)
    mT = np.sum(T * V)

    Acomp = Acomp / mA
    T = T / mT



    num = np.sum(np.minimum(Acomp, T) * V)
    den = np.sum(np.maximum(Acomp, T) * V)

    return float(num / den )

Ti=read_target_kernel( './INPUT_LSQR/T{}/{}_T_sparse0_{}'.format(basenameT,basename,nT))
A=read_target_kernel( './outputs_{}/A_{}_P/A_{}'.format(eta,basenameT,nT))
As=read_target_kernel( './outputs_{}/A_{}_S/A_{}'.format(eta,basenameT,nT))
MRP[nT]=np.sum(grid.Vj*np.power(A-Ti,2))/np.sum(grid.Vj*Ti**2)
MRS[nT]=np.sum(grid.Vj*np.power(As-Ti,2))/np.sum(grid.Vj*Ti**2)
MRPS[nT]=np.sum(grid.Vj*np.power(A-As,2))/np.sum(grid.Vj*As**2)