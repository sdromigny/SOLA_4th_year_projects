import builtins
import configparser
import argparse
import sys
import os
import pandas as pd
import numpy as np
from scipy import io

from configcl_gen import Cfig
from subprocess import call
from scipy.sparse import csr_matrix, dok_matrix
from SOLA_utils import *


# get config filename
#----------------------------------
argu_parser = argparse.ArgumentParser(description='Run SOLA inversion')
argu_parser.add_argument('-c','--configfile', nargs=1, required=True, help='filename of configuration file (including path)', type=str)

args = argu_parser.parse_args()
filecfg = args.configfile


# Parse config file
#----------------------------------
cfg = Cfig(filecfg)


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


# Create outdir if required
#----------------------------------
if not os.path.exists(cfg.outdir):
    os.mkdir(cfg.outdir)

outdir_eta = os.path.join(cfg.outdir, 'OUTPUT_{}/'.format('{:f}'.format(eta).replace('.','p'))) 
if not os.path.exists(outdir_eta):
    os.mkdir(outdir_eta)


# read Vj
#-------------------------------
Vj_fname = os.path.join(indir, 'Vj_{}.txt'.format(radname))
#print('reading Vj : ', Vj_fname)

try:
   Vj = np.loadtxt(Vj_fname)
except:
   print('{} not found'.format(Vj_fname))
   sys.exit()
Vjmax = Vj.max()


# Read Gij matrix --------------------------
# Gij need to be normalized by sigma_di and Vj 
# need to switch the first line with the sparsest one to optimize computation time if not yet done
# need to be done by running SOLA_preprocess.py if not yet done
#-------------------------------
Gij_norm_sort = os.path.join(cfg.indir, 'Gij_{}_normSigma_sparse0.txt'.format(cfg.radname))
if os.path.isfile(Gij_norm_sort):
    # get N and M parameters from G file
    with open(Gij_norm_sort) as f:
         first_line = f.readline()
         secondline = f.readline().rstrip()
         N, M = [int(elt) for elt in secondline.split()]

    # read full file
    Gij = pd.read_table(Gij_norm_sort, delim_whitespace=True, skiprows=2, header=None, names=['i','j','val'])
    Gij_csr = csr_matrix((np.array(Gij['val']), (np.array(Gij['i']), np.array(Gij['j']))), shape=(N,M))
else:
    print('{} not found'.format(Gij_norm_sort))
    sys.exit()


# compute c_i = sum_j Gij
#-------------------------------
if flag_new is True:
    c_i = compute_c_vector(cfg, Gij_csr)
else:
    c_i = read_c_vector(cfg)


# write first raw of G 
# (needed by SOLA_LSQR C_codes)
#-------------------------------
GFirstRaw_fname  = os.path.join(cfg.indir, 'Gij_{}_firstraw.txt'.format(cfg.radname))
write_G_matrix(cfg, Gij_csr, GFirstRaw_fname, first_raw=True, transpose=False)


# compute Q matrix
#---------------------------------

if flag_new is True:
    Q = compute_Q_matrix(cfg, Gij_csr, c_i, eta, Vj)
else:
    Q = read_Q_matrix(cfg, c_i, eta)

Q_fname = os.path.join(outdir_eta, 'Q_matrix_{}.txt'.format('{:f}'.format(eta).replace('.','p')))
write_Q_matrix(cfg, Q, eta, Q_fname)


# launch LSQR for SOLA 
#------------------------------------------------------------------------
print('Launch LSQR for SOLA')

tk_name = os.path.join(cfg.indir, 'Target_kernels', 'Tk')
ci_fname = os.path.join(cfg.outdir, 'C_vector.txt')

launch_LSQR_SOLA(cfg, Q_fname, tk_name, ci_fname, Vj_fname, GFirstRaw_fname, eta)


