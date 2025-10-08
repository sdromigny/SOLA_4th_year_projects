import configparser
import argparse
import os
import sys
import pandas as pd
import numpy as np
from scipy import io
from numpy.compat import asbytes
from configcl_gen import Cfig
from subprocess import call
from scipy.sparse import csr_matrix, dok_matrix
from SOLA_utils import *

# get config filename
#----------------------------------
argu_parser = argparse.ArgumentParser(description='Run SOLA postprocessing (compute mk, sigma_m, Ak, Rk)')
argu_parser.add_argument('-c','--configfile', nargs=1, required=True, help='filename of configuration file (including path)', type=str)

args = argu_parser.parse_args()
filecfg = args.configfile


# Parse config file
#----------------------------------
cfg = Cfig(filecfg)

# Create outdir if required
#----------------------------------
if not os.path.exists(cfg.outdir):
    os.mkdir(cfg.outdir)


# get parameters from SOLA.cfg
#----------------------------------
# trade off parameter
eta = cfg.eta

# input directory name
indir = cfg.indir

# rootname for G matrix
radname = cfg.radname

# to avoid recomputing Q matrix
flag_new = cfg.flag_new


# read Vj
#-------------------------------
Vj_fname = os.path.join(indir, 'Vj_{}.txt'.format(radname))
print('reading Vj : ', Vj_fname)

try : 
    Vj = np.loadtxt(Vj_fname)
except:
    print('{} not found'.format(Vj_fname))
    sys.exit()
Vjmax = Vj.max()
M = Vj.shape[0]


# read d vector (+ sigma_d)
#-------------------------------
di_filename = os.path.join(indir, 'di_{}_sparse0.txt'.format(radname))
print('reading data vector di : ', di_filename)

try:
    data = np.loadtxt(di_filename)
except:
    print('{} not found'.format(di_filename))
    sys.exit()
N = data.shape[0]

print('nb parameters, data : {:d}, {:d}'.format(M, N))


if cfg.do_Ak or cfg.do_Rmatrix:
    Gij_norm_sort = os.path.join(cfg.indir, 'Gij_{}_normSigma_sparse0.txt'.format(cfg.radname))
    Gij_t_fname = os.path.join(cfg.indir, 'Gij_{}_normSigma_sparse0_transpose.txt'.format(cfg.radname))
    if os.path.isfile(Gij_norm_sort):
        Gij = pd.read_table(Gij_norm_sort, delim_whitespace=True, skiprows=2, header=None, names=['i','j','val'])
        Gij_csr = csr_matrix((np.array(Gij['val']), (np.array(Gij['i']), np.array(Gij['j']))), shape=(N,M))
    else:
        print('{} not found'.format(Gij_norm_sort))
        sys.exit()


# compute mk and sigmak
#------------------------------------------------------------------------
if cfg.do_mk is True:
    di = data[:,0]/data[:,1]
    mk, sigmak = compute_solution(cfg, di, eta, N, M)


# compute averaging kernels for range of kvalues
#------------------------------------------------------------------------
if cfg.do_Ak is True:

    # need to write a file containing Gij transpose
    if os.path.isfile(Gij_t_fname) is False:
        write_G_matrix(cfg, Gij_csr, Gij_t_fname, first_raw=False, transpose=True)

    compute_Ak_c(cfg, Gij_t_fname, Vj_fname, eta)


# or compute Rij matrix
#------------------------------------------------------------------------
if cfg.do_Rmatrix is True:

    # need to write a file containing Gij transpose
    if os.path.isfile(Gij_t_fname) is False:
        write_G_matrix(cfg, Gij_csr, Gij_t_fname, first_raw=False, transpose=True)

    compute_R_c(cfg, Gij_t_fname, Vj_fname, eta, M)


