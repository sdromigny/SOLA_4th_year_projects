from configcl_gen import Cfig
from Target import *

# Parse the config file
#-----------------------------------------
cfg = Cfig('SOLA_toy.cfg')


# get parameters
#-----------------------------------------
nx = cfg.nx
ny = cfg.ny
factmul = cfg.factmul


# Project target kernels Tk 
#-----------------------------------------
# reading radius file
fname = 'INPUT/Tk_radius.txt'
radius =  read_Tk_radius(fname, factmul)

print("Compute Target kernel Tk")
Tks = compute_Tks(nx, ny, cfg, radius, writefile=True)

