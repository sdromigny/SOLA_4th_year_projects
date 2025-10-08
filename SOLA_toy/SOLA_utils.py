import os
import numpy as np
from numpy.compat import asbytes
from scipy.sparse import csr_matrix, identity, vstack
from scipy.sparse import load_npz, save_npz


# G matrix 
#--------------------------------------

def preprocess_G_matrix(cfg, Gij_csr, di, normVj=False):

    # search for the most sparse line of G to permut with the first line
    #-------------------------------
    print('search for the most sparse line of G for permutation with the first line')
    ind = Gij_csr.getnnz(axis=1).argmin()
    print('most sparse line index = %d'%ind)

    Gnames = {True : 'normSigmaVj', False:'normSigma'}
    Gij_norm_sort = os.path.join(cfg.indir, 'Gij_{}_{}_sparse0.txt'.format(cfg.radname, Gnames[normVj]))
    print(Gij_norm_sort)

    if cfg.flag_sigma is True:
        # need to divide Gij by sigma_d_i
        # write normalized Gij in file (needed for the computation of Averaging kernels)
        #-------------------------------
        print('Normalizing Gij by sigma_d')

    print('writting new Gij and di files')
    if cfg.flag_sigma:
        write_G_spars_norm(Gij_norm_sort, Gij_csr, ind, di)
    else:
        write_G_spars(Gij_norm_sort, Gij_csr, ind)
       
    di_norm_sort = os.path.join(cfg.indir, 'di_{}_sparse0.txt'.format(cfg.radname))
    di[[0, ind]] = di[[ind, 0]]
    #np.savetxt(di_norm_sort, di, fmt='%-14.6f %-14.6f')
    np.savetxt(di_norm_sort, di, fmt='%f %f')


def write_G_spars_norm(Gij_norm_sort, Gij_csr, ind, di):

    N, M = Gij_csr.shape
    coo = Gij_csr.tocoo()

    fid = open(Gij_norm_sort, 'wb')
    fid.write(asbytes('%i\n' % (len(coo.data))))
    fid.write(asbytes('%i %i\n' % (N, M)))

    r = 0
    for c, val in zip(coo.col[coo.row == ind], coo.data[coo.row == ind]):
        val = val/di[ind,1]
        fid.write(asbytes(("%i %i " % (r, c)) + ("%14.10f\n" % val)))

    for r, c, val in zip(coo.row[coo.row < ind], coo.col[coo.row < ind], coo.data[coo.row < ind]):
        if r == 0:
            continue
        val = val/di[r, 1]
        fid.write(asbytes(("%i %i " % (r, c)) + ("%14.10f\n" % val)))

    if ind != 0:
        r = ind
        for c, val in zip(coo.col[coo.row == 0], coo.data[coo.row == 0]):
            val = val/di[0,1]
            fid.write(asbytes(("%i %i " % (r, c)) + ("%14.10f\n" % val)))

    for r, c, val in zip(coo.row[coo.row > ind], coo.col[coo.row > ind], coo.data[coo.row > ind]):
        val = val/di[r, 1]
        fid.write(asbytes(("%i %i " % (r, c)) + ("%14.10f\n" % val)))
    fid.close()


def write_G_norm(Gij_norm, Gij_csr, di):

    N, M = Gij_csr.shape
    coo = Gij_csr.tocoo()

    fid = open(Gij_norm, 'wb')
    fid.write(asbytes('%i\n' % (len(coo.data))))
    fid.write(asbytes('%i %i\n' % (N, M)))

    for r, c, val in zip(coo.row, coo.col, coo.data):
        val = val/di[r, 1]
        fid.write(asbytes(("%i %i " % (r, c)) + ("%14.10f\n" % val)))
    fid.close()


def write_G_spars(Gij_norm_sort, Gij_csr, ind):

    N, M = Gij_csr.shape
    coo = Gij_csr.tocoo()

    fid = open(Gij_norm_sort, 'wb')
    fid.write(asbytes('%i\n' % (len(coo.data))))
    fid.write(asbytes('%i %i\n' % (N, M)))

    r = 0
    for c, val in zip(coo.col[coo.row == ind], coo.data[coo.row == ind]):
        fid.write(asbytes(("%i %i " % (r, c)) + ("%14.10f\n" % val)))

    for r, c, val in zip(coo.row[coo.row < ind], coo.col[coo.row < ind], coo.data[coo.row < ind]):
        if r == 0:
            continue
        fid.write(asbytes(("%i %i " % (r, c)) + ("%14.10f\n" % val)))

    if ind != 0:
        r = ind
        for c, val in zip(coo.col[coo.row == 0], coo.data[coo.row == 0]):
            fid.write(asbytes(("%i %i " % (r, c)) + ("%14.10f\n" % val)))

    for r, c, val in zip(coo.row[coo.row > ind], coo.col[coo.row > ind], coo.data[coo.row > ind]):
        fid.write(asbytes(("%i %i " % (r, c)) + ("%14.10f\n" % val)))
    fid.close()



def write_G_matrix(cfg, Gij_csr, Gij_fname, Vj=None, first_raw=False, transpose=False):

    # write G matrix
    #---------------------------------
    N, M = Gij_csr.shape

    if Vj is not None:
        # remove normalisation of G by sqrt(Vj/Vjmax)
        Vjmax = Vj.max() 

    if first_raw is True:
        # extract first raw of Gij
        Gij_0 = Gij_csr.getrow(0)
        if Vj is not None:
            Gij_0 = Gij_0.multiply(np.sqrt(Vj/Vjmax))
        coo = Gij_0.tocoo()
    elif transpose is True:
        if Vj is not None:
            Gij_csr = Gij_csr.multiply(np.sqrt(Vj/Vjmax)).tocsr()
        Gt = Gij_csr.transpose()
        cootmp = Gt.tocoo()
        ccsr = cootmp.tocsr()
        coo = ccsr.tocoo()
    else:
        if Vj is not None:
            Gij_csr = Gij_csr.multiply(np.sqrt(Vj/Vjmax)).tocsr()
        coo = Gij_csr.tocoo()
    
    fid = open(Gij_fname, 'wb')
    if first_raw is True:
         for c, d in zip(coo.col, coo.data):
             fid.write(asbytes(("%i " % c) + ("%14.10f\n" % d)))
    elif transpose is True:
         fid.write(asbytes('%i\n' % (len(coo.data))))
         fid.write(asbytes('%i %i\n' % (M, N)))
         for r, c, d in zip(coo.row, coo.col, coo.data):
             fid.write(asbytes(("%i %i " % (r, c)) + ("%14.10f\n" % d)))
    else:
         fid.write(asbytes('%i\n' % (len(coo.data))))
         fid.write(asbytes('%i %i\n' % (N, M)))
         for r, c, d in zip(coo.row, coo.col, coo.data):
             fid.write(asbytes(("%i %i " % (r, c)) + ("%14.10f\n" % d)))
    fid.close()


# c vector 
#------------------------------

def compute_c_vector(cfg, Gij_csr, Vj=None):

    # compute c_i = sum_j Gij
    # write c_i in a file (needed for C codes)
    #-------------------------------
    print('Compute C vector')

    if Vj is not None:
        # remove G normalisation by sqrt(Vj/Vjmax)
        Vjmax = Vj.max()
        Gij_csr = Gij_csr.multiply(np.sqrt(Vj/Vjmax)).tocsr()

    # compute C vector 
    c_i = Gij_csr.sum(axis=1)

    # write C vector in a file
    cfname = os.path.join(cfg.outdir, 'C_vector.txt')
    np.savetxt(cfname, c_i, fmt='%14.10f')

    return c_i


def read_c_vector(cfg):

    cfname = os.path.join(cfg.outdir, 'C_vector.txt')
    print(cfname)
    c_i = np.loadtxt(cfname)

    return c_i


# Q matrix 
#------------------------------

def compute_Q_matrix(cfg, Gij_csr, c_i, eta, Vj=None):

    # compute Q matrix
    #---------------------------------
    print('compute Q matrix')

    if Vj is not None:
        # normalise G by sqrt(Vj/Vjmax)
        Vjmax = Vj.max()
        Gij_csr = Gij_csr.multiply(np.sqrt(Vjmax/Vj)).tocsr()

    N = c_i.shape[0]
    ct = c_i[1:]/c_i[0]

    Idn = identity(N-1).tocsr()
    B = vstack([csr_matrix(-1*ct.transpose()), Idn])
    Qcom = Gij_csr.transpose().dot(B)
    Qtmp = vstack([Qcom, csr_matrix(-eta*ct.transpose())])

    Q = Qtmp.tocsr()
    Q.sort_indices()

    # save part of Q independent of eta
    Q_fname_wth_eta = os.path.join(cfg.outdir, 'Q_matrix_com.npz')
    save_npz(Q_fname_wth_eta, Qcom)

    return Q


def read_Q_matrix(cfg, c_i, eta):

    # read part of Q matrix independent of eta
    # and add the last line (eta-dependent)
    #---------------------------------
    print('read common part of Q matrix and add last line')

    N = c_i.shape[0]
    ct = c_i[1:]/c_i[0]

    Q_fname_wth_eta = os.path.join(cfg.outdir, 'Q_matrix_com.npz')
    Qcom = load_npz(Q_fname_wth_eta)
    Qtmp = vstack([Qcom, csr_matrix(-eta*ct.transpose())])

    Q = Qtmp.tocsr()
    Q.sort_indices()

    return Q


def write_Q_matrix(cfg, Q, eta, Q_fname):

    # write Q matrix
    #---------------------------------
    coo = Q.tocoo()

    M = Q.shape[0]-1
    N = Q.shape[1]+1
    
    fid = open(Q_fname, 'wb')
    fid.write(asbytes('%i\n' % (len(coo.data))))
    fid.write(asbytes('%i %i\n' % (M+1, N-1)))
    for r, c, d in zip(coo.row, coo.col, coo.data):
        fid.write(asbytes(("%i %i " % (r, c)) + ("%14.10f\n" % d)))
    fid.close()


# SOLA computation
#-----------------------------------------

def launch_LSQR_SOLA(cfg, Q_fname, tk_name, ci_fname, Vj_fname, GFirstRaw_fname, eta, normVj=False):

    # launch LSQR_SOLA (C codes)
    #-------------------------------------

    outdir_eta = os.path.join(cfg.outdir, 'OUTPUT_{}/'.format('{:f}'.format(eta).replace('.','p')))
    outdir_sol = os.path.join(outdir_eta, 'G_inverse/')

    flags = {True:1, False:0}
    flagVj = flags[normVj]

    if not os.path.exists(outdir_sol):
        os.mkdir(outdir_sol)

    solname = os.path.join(outdir_sol, 'xk_{}'.format('{:f}'.format(eta).replace('.','p')))

    commandline = '{} {} {} {:d} {:d} {} {:d} {:f} {} {} {} {:d} {:d} > outscreen'.format(cfg.SOLAlsqr, Q_fname, tk_name, cfg.kmin, cfg.kmax, solname, cfg.niter, eta, ci_fname, GFirstRaw_fname, Vj_fname, flagVj, cfg.niter+1)
    print(commandline)
    os.system(commandline)


def compute_solution(cfg, di, eta, N, M):
    # compute mk and sigma_k
    #------------------------------------------------------------------------
    print('compute solution : mk, sigmak')

    mk = np.zeros(M)

    sigmak = np.zeros(M)

    outdir_eta = os.path.join(cfg.outdir, 'OUTPUT_{}/'.format('{:f}'.format(eta).replace('.','p')))
    outdir_sol = os.path.join(outdir_eta, 'G_inverse/')

    for k in range(M):
        fname = os.path.join(outdir_sol, 'xk_{}_{:d}.txt'.format('{:f}'.format(eta).replace('.','p'), k))
        try:
            xk = np.loadtxt(fname)
        except:
            continue

        mk[k] = np.dot(xk, di)
        sigmak[k] = np.sqrt(np.dot(xk, xk))


    filemodel = os.path.join(outdir_eta, 'mk_{}.txt'.format('{:f}'.format(eta).replace('.','p')))
    fid = open(filemodel,'wb')
    for k, m, sigma in zip(np.arange(M, dtype=int), mk, sigmak):
        fid.write(asbytes(("%i " %k) + ("%10.6f %10.6f\n" % (m, sigma))))
    fid.close()

    return mk, sigmak


def compute_Ak_c(cfg, Gij_t_fname, Vj_fname, eta, normVj=False):
    
    # launch ComputeAveragingKernels (C codes)
    #-------------------------------------

    outdir_eta = os.path.join(cfg.outdir, 'OUTPUT_{}/'.format('{:f}'.format(eta).replace('.','p')))
    outdir_Ak = os.path.join(outdir_eta, 'ResKernels/')
    outdir_sol = os.path.join(outdir_eta, 'G_inverse/')

    flags = {True:1, False:0}
    flagVj = flags[normVj]

    if not os.path.exists(outdir_Ak):
        os.mkdir(outdir_Ak)

    xk_name = os.path.join(outdir_sol, 'xk_{}'.format('{:f}'.format(eta).replace('.','p')))
    Ak_name = os.path.join(outdir_Ak, 'Ak_{}'.format('{:f}'.format(eta).replace('.','p')))

    commandline = '{} {} {} {:d} {:d} {} {} {:d} {:d} > outscreen'.format(cfg.compAkC, Gij_t_fname, xk_name, cfg.kmin, cfg.kmax, Ak_name, Vj_fname, 0, flagVj)

    print(commandline)
    os.system(commandline)


def compute_R_c(cfg, Gij_t_fname, Vj_fname, eta, M, normVj=False):
    
    # launch ComputeAveragingKernels (C codes)
    #-------------------------------------

    outdir_eta = os.path.join(cfg.outdir, 'OUTPUT_{}/'.format('{:f}'.format(eta).replace('.','p')))
    outdir_R = os.path.join(outdir_eta, 'ResKernels/')
    outdir_sol = os.path.join(outdir_eta, 'G_inverse/')

    flags = {True:1, False:0}
    flagVj = flags[normVj]

    if not os.path.exists(outdir_R):
        os.mkdir(outdir_R)

    xk_name = os.path.join(outdir_sol, 'xk_{}'.format('{:f}'.format(eta).replace('.','p')))
    Rij_name = os.path.join(outdir_R, 'Rij_{}.txt'.format('{:f}'.format(eta).replace('.','p')))
    kmin = 0
    kmax = M-1

    commandline = '{} {} {} {:d} {:d} {} {} {:d} {:d} > outscreen'.format(cfg.compAkC, Gij_t_fname, xk_name, kmin, kmax, Rij_name, Vj_fname, 1, flagVj)

    print(commandline)
    os.system(commandline)
