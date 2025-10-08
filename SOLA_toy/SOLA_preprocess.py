import configparser
import argparse
import os
import sys
import pandas as pd
import numpy as np
from configcl_gen import Cfig
from SOLA_utils import *

# get config filename
#----------------------------------
argu_parser = argparse.ArgumentParser(description='Run SOLA preprocessing (G normalisation and sparsity)')
#                                                   - normalize Gij by sigma_di (needed for LSQR C code) \n
#                                                   - switch the first line with the sparsest one to optimize computation time')
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
# input directory name
indir = cfg.indir

# rootname for G matrix
radname = cfg.radname


# preprocessing need to be done once
#----------------------------------------------------
# Gij need to be normalized by sigma_di 
# need to switch the first line with the sparsest one to optimize computation time if not yet done

flag_sigma = cfg.flag_sigma
flag_preprocess = cfg.flag_preprocess    

if flag_preprocess is False:
   sys.exit('No preprocessing done')

# read Vj
#-------------------------------
Vj_fname = os.path.join(indir, 'Vj_{}.txt'.format(radname))
print('reading Vj : ', Vj_fname)

try:
   Vj = np.loadtxt(Vj_fname)
except:
   print('{} not found'.format(Vj_fname))
   sys.exit()
Vjmax = Vj.max()
M = Vj.shape[0]


# read d vector (+ sigma_d)
#-------------------------------
di_filename = os.path.join(indir, 'di_{}.txt'.format(radname))
print('reading data vector di : ', di_filename)

try:
   data = np.loadtxt(di_filename)
except:
   print('{} not found'.format(di_filename))
   sys.exit()
N = data.shape[0]

print('nb parameters, data : {:d}, {:d}'.format(M, N))


# preprocessing --------------------------
# search for the most sparse line of G to permut with the first line
# save in file as normalized Gij used for the comptation of Averaging kernels
#-------------------------------
radsigma = {True: '', False: 'normSigma'}

if flag_preprocess is True:
    radnorm = '{}'.format(radsigma[flag_sigma])
    print(radnorm)
    if radnorm == '':
        Gij_filename = os.path.join(cfg.indir, 'Gij_{}.txt'.format(cfg.radname))
    else:
        Gij_filename = os.path.join(cfg.indir, 'Gij_{}_{}.txt'.format(cfg.radname, radnorm))

    print(Gij_filename)
    try:
        Gij = pd.read_table(Gij_filename, delim_whitespace=True, skiprows=2, header=None, names=['i','j','val'])
    except:
        print('{} not found'.format(Gij_filename))
        sys.exit()
        
    Gij_csr = csr_matrix((np.array(Gij['val']), (np.array(Gij['i']), np.array(Gij['j']))), shape=(N,M))
    preprocess_G_matrix(cfg, Gij_csr, data)
