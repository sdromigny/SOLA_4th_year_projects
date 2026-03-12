import numpy as np
import os
from write_Target import *

# Define grid dimensions
nx, ny, nz = 9, 9, 9

# Define sphere radius
radius = 2.0

class Config:
    def __init__(self):
        self.indir = './sola_inputs_cornwall' # directory to save target files

cfg = Config()

parallel_Tks(nx, ny, nz, radius, cfg)