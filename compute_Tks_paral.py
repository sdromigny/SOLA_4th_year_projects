import numpy as np
import os
from write_Target import *
class Config:
    def __init__(self):
        self.indir = './sola_inputs_cornwall'

cfg = Config()

if __name__ == "__main__":
    
    nx, ny, nz = 9, 9, 9
    radius = 2.0



    parallel_Tks(nx, ny, nz, radius, cfg)