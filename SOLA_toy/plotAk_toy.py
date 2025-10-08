import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from colortomo import colmapAk
from scipy.sparse import csr_matrix
from configcl_gen import Cfig     

# Parse the config file
#-------------------------------
cfg = Cfig('SOLA_toy.cfg')

# create outdir for figures if required
#-------------------------------
if not os.path.exists(cfg.outdirfig):
    os.mkdir(cfg.outdirfig)

eta = cfg.eta
nx = cfg.nx
ny = cfg.ny


# k value of the target to be plotted
#-------------------------------
kvalue = cfg.kvalue

M = nx*ny

# read Vj
#-------------------------------
Vj = np.ones(M)
Vjmax = 1.

# read corresponding target radius
#-------------------------------
fname = 'INPUT/Tk_radius.txt'

try :
    radius = np.loadtxt(fname)
except:
    print('{} not found'.format(fname))
    sys.exit()

r_Tk = radius[kvalue, 3]*cfg.factmul
xk = (radius[kvalue, 1]-1) + 0.5
yk = (radius[kvalue, 0]-1) + 0.5
circle = plt.Circle((xk,yk), r_Tk, color='g', linewidth=1, fill=False)


# compute or read Averaging kernel and plot them 
#------------------------------------------------------------------------
outdir_eta = os.path.join(cfg.outdir, 'OUTPUT_{}/'.format('{:f}'.format(eta).replace('.','p')))
outdir_Ak = os.path.join(outdir_eta, 'ResKernels')
outdir_sol = os.path.join(outdir_eta, 'G_inverse')

fileAk = os.path.join(outdir_Ak, 'Ak_{}_{:d}.txt'.format('{:f}'.format(eta).replace('.','p'), kvalue))
try:
    Ak = np.loadtxt(fileAk)
except:
    print('compute Ak')

    filex = os.path.join(outdir_sol, 'xk_{}_{:d}.txt'.format('{:f}'.format(eta).replace('.','p'), kvalue))
    x = np.loadtxt(filex)
    N = x.shape[0]

    # read G matrix
    #-------------------------------
    Gij_filename = os.path.join(cfg.indir, 'Gij_{}_normSigma_sparse0.txt'.format(cfg.radname))
    print('reading Gij :%s'%Gij_filename)
    try:
        Gij = pd.read_table(Gij_filename, delim_whitespace=True, skiprows=2, header=None, names=['i','j','val'])
    except:
        print('{} not found'.format(Gij_filename))
        sys.exit()
    Gij_sc = csr_matrix((np.array(Gij['val']), (np.array(Gij['i']), np.array(Gij['j']))), shape=(N,M))
    Mat = Gij_sc.transpose()

    Ak = Mat.dot(x)/Vj
    np.savetxt(fileAk, Ak, fmt='%14.10f')
#Ak = Ak/Ak.max()

# plot Ak 
#------------------------------------------------------------------------
Avk = Ak.reshape([nx,ny])

filefig = os.path.join(cfg.outdirfig, 'Ak_toy_{}_{:d}.png'.format('{:f}'.format(eta).replace('.','p'), kvalue))

fig = plt.figure(figsize=(6,6))
ax = fig.gca()
vmax = np.max(abs(Ak))
im = plt.pcolormesh(Avk, vmin=-vmax, vmax=vmax, cmap=colmapAk)
ax.add_artist(circle)
plt.title(r'Averaging kernel : k = %d, $\eta = $%f'%(kvalue, eta))
plt.colorbar(orientation='horizontal', fraction=0.05, pad=0.02)
plt.xlim([0,nx])
plt.ylim([0,ny])
#plt.xticks([])
#plt.yticks([])
plt.savefig(filefig, dpi=300)
plt.show()
