#!/usr/bin/python

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from colortomo import colmap
from configcl_gen import Cfig

# Parse the config file
cfg = Cfig('SOLA_toy.cfg')

# create outdir for figures if required
if not os.path.exists(cfg.outdirfig):
    os.mkdir(cfg.outdirfig)


eta = cfg.eta 
vmin = cfg.vmin
vmax = cfg.vmax
sigmax = cfg.sigmax

nx = cfg.nx
ny = cfg.ny

# read model and sigma for plotting
#---------------------------------------
outdir_eta = os.path.join(cfg.outdir, 'OUTPUT_{}/'.format('{:f}'.format(eta).replace('.','p')))
filemodel = os.path.join(outdir_eta, 'mk_{}.txt'.format('{:f}'.format(eta).replace('.','p')))
try:
    mk = np.loadtxt(filemodel)
except:
    print('{} not found'.format(filemodel))
    sys.exit()
model = mk[:,1].reshape([nx, ny])*100.
sigma = mk[:,2].reshape([nx, ny])*100.


# plot Model 
#---------------------------------------
filefig = os.path.join(cfg.outdirfig, 'mk_toy_{}.png'.format('{:f}'.format(eta).replace('.','p')))

plt.figure(figsize=(12,6))
plt.subplot(1,2,1)
plt.pcolormesh(model, cmap=colmap, vmin=vmin, vmax=vmax)
plt.colorbar(orientation='horizontal', fraction=0.05, pad=0.02)
plt.title(r'SOLA model (dln $V_s$) : $\eta = $%f'%eta)
plt.xticks([])
plt.yticks([])

plt.subplot(1,2,2)
plt.pcolormesh(sigma, cmap=plt.cm.binary, vmin=0, vmax=sigmax)
plt.colorbar(orientation='horizontal', fraction=0.05, pad=0.02)
plt.title(r'$\sigma_m$ : $\eta = $%f'%eta)
plt.xticks([])
plt.yticks([])
plt.savefig(filefig, dpi=300)
plt.show()

