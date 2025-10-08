import numpy as np
import os
import sys
from colortomo import colmapAk
import matplotlib.pyplot as plt

def WhereAmI(xi, yi, nx, ny):

    ij = int(yi) + int(xi)*nx

    return ij


def write_Tk(Tk, k, fname):

    np.savetxt(fname, Tk, fmt='%14.10f')


def read_Tk_radius(fname, factmul):

    try : 
        radius = np.loadtxt(fname)
    except:
        print('{} not found'.format(fname))
        sys.exit()

    radius[:, 3] = radius[:,3]*factmul
    
    return radius


def compute_Tk_circle(xk, yk, radius, nx, ny):

    dx = 0.01   # rappel du pas dl = 0.00001 utilise pour l'integration le long du rai
    dy = dx

    x = np.arange(max([0, xk-radius]), min([32-0.001, xk+radius]), dx)
    y = np.arange(max([0, yk-radius]), min([32-0.001, yk+radius]), dy)
    X, Y = np.meshgrid(x, y)

    # surface element
    dS = dx*dy

    # find points in circle
    inds = np.where((((X - xk)**2 + (Y - yk)**2) <= radius**2))

    # Build Tk 
    Tk = np.zeros(nx*ny)
    for iy, ix in zip(inds[0], inds[1]):
        xi = x[ix]
        yi = y[iy]
        ij = WhereAmI(xi, yi, nx, ny)
        Tk[ij] += dS

    # normalization
    Tk = Tk/np.sum(Tk)
          
    return Tk


def compute_Tks(nx, ny, cfg, radius, writefile=True):
    indir = cfg.indir
    Tks = []
    path = os.path.join(indir, 'Target_kernels')

    if not os.path.exists(path):
        os.mkdir(path)

    for ix, iy, ik, r in radius:
        ik = ik-1
        xk = (ix-1)+0.5
        yk = (iy-1)+0.5
        Tk = compute_Tk_circle(xk, yk, r, nx, ny)
        Tks.append(Tk)
            
        if writefile:
           print('write Tk for k = %d'%(ik))
           fileTi = os.path.join(path, 'Tk_%d.txt'%(ik))
           write_Tk(Tk, ik, fileTi)

    return Tks


def Plot_Target(Tk, nx, ny):

    Target = Tk.reshape([nx,ny])
    plt.figure(figsize=(6,6))

    vmax = np.max(abs(Target))
    im = plt.pcolormesh(Target, vmin=-vmax, vmax=vmax, cmap=colmapAk)
    plt.grid()
    plt.colorbar(orientation='horizontal', fraction=0.05, pad=0.02)
    plt.xlim([0,nx])
    plt.xlim([0,ny])
    plt.xticks([])
    plt.yticks([])
    plt.show()

