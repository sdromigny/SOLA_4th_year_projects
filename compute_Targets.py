import numpy as np
import os
from write_Target import compute_all_Tks_3D, compute_Tks_3D

# Define grid dimensions
nx, ny, nz = 10, 10, 10

# Define sphere radius
radius = 2.0

class Config:
    def __init__(self):
        self.indir = './sola_inputs_cornwall' # directory to save target files

cfg = Config()

compute_all_Tks_3D(nx, ny, nz, radius, cfg)
