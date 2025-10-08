#!/usr/bin/python

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from colortomo import colmap
from configcl_gen import Cfig
from scipy.sparse import csr_matrix
from SOLA_utils import *

# Parse the config file
#-------------------------------
cfg = Cfig('SOLA_toy.cfg')

# True model name is in cfg.input_model
# can be changed in config file to test other models

# create outdir for figures if required
#-------------------------------
if not os.path.exists(cfg.outdirfig):
    os.mkdir(cfg.outdirfig)

# get parameters
#-------------------------------
eta = cfg.eta 
vmin = cfg.vmin
vmax = cfg.vmax
sigmax = cfg.sigmax

nx = cfg.nx
ny = cfg.ny
M = nx*ny
N =  9778

radname = cfg.radname

# read the resolution matrix if exist
#------------------------------------------
outdir_eta = os.path.join(cfg.outdir, 'OUTPUT_{}/'.format('{:f}'.format(eta).replace('.','p')))
R_filename = os.path.join(outdir_eta, 'ResKernels', 'Rij_{}.txt'.format('{:f}'.format(eta).replace('.','p')))
flag = True
try:
    Rij = pd.read_table(R_filename, delim_whitespace=True, skiprows=1, header=None, names=['i','j','val'])
    Rij_csr = csr_matrix((np.array(Rij['val']), (np.array(Rij['i']), np.array(Rij['j']))), shape=(M,M))
except:
    flag = False

outdir_sol = os.path.join(cfg.outdir, 'G_inverse')
if flag is False:
    # compute Rij matrix
    #------------------------------------------------------------------------
    # need to write a file containing Gij transpose
    Gij_t_fname = os.path.join(cfg.indir, 'Gij_{}_normSigma_sparse0_transpose.txt'.format(cfg.radname))
    print( Gij_t_fname)
    if os.path.isfile(Gij_t_fname) is False:
        Gij_fname = os.path.join(cfg.indir, 'Gij_{}_normSigma_sparse0.txt'.format(cfg.radname))
        try:
            Gij = pd.read_table(Gij_fname, delim_whitespace=True, skiprows=2, header=None, names=['i','j','val'])
        except:
            print('{} not found'.format(Gij_fname))
            sys.exit()

        Gij_csr = csr_matrix((np.array(Gij['val']), (np.array(Gij['i']), np.array(Gij['j']))), shape=(N,M))
        write_G_matrix(cfg, Gij_csr, Gij_t_fname, first_raw=False, transpose=True)

    Vj_fname = os.path.join(cfg.indir, 'Vj_{}.txt'.format(radname))
    compute_R_c(cfg, Gij_t_fname, Vj_fname, eta, M)

    Rij = pd.read_table(R_filename, delim_whitespace=True, skiprows=1, header=None, names=['i','j','val'])
    Rij_csr = csr_matrix((np.array(Rij['val']), (np.array(Rij['i']), np.array(Rij['j']))), shape=(M,M))


# read input model for tomographic filtering
#-------------------------------
filemodel = os.path.join('SOLA_Filtering/True_models/', cfg.input_model)
mk = np.loadtxt(filemodel)
model = mk[:,0].reshape([nx, ny])*100.

# compute filtered model
#-------------------------------
mf = Rij_csr.dot(mk[:,0])
modelf = mf.reshape([nx, ny])*100.

mfk = np.zeros(mk.shape)
mfk[:,0] = mf

filemodel_filt = os.path.join('SOLA_Filtering/Filtered_models/', cfg.input_model.split('.')[0]+'_filt.txt')
np.savetxt(filemodel_filt, mfk, fmt='%8.6f %8.6f')
filefig = os.path.join(cfg.outdirfig, cfg.input_model.split('.')[0]+'_filt.png')

# plot Model 
#-------------------------------
plt.figure(figsize=(12,6))
plt.subplot(1,2,1)
plt.pcolormesh(model, cmap=colmap, vmin=vmin, vmax=vmax)
plt.colorbar(orientation='horizontal', fraction=0.05, pad=0.02)
plt.title(r'True model (dln $V_s$)')
plt.xticks([])
plt.yticks([])

plt.subplot(1,2,2)
plt.pcolormesh(modelf, cmap=colmap, vmin=vmin, vmax=vmax)
plt.colorbar(orientation='horizontal', fraction=0.05, pad=0.02)
plt.title(r'Filtered model')
plt.xticks([])
plt.yticks([])
plt.savefig(filefig, dpi=300)
plt.show()

