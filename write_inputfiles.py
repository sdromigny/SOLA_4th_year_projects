import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, diags

def write_sparse_ijv(path, G):
    G = G.tocoo()
    with open(path, "w") as f:
        f.write(f"{G.nnz}\n")
        f.write(f"{G.shape[0]} {G.shape[1]}\n")
        for i, j, v in zip(G.row, G.col, G.data):
            f.write(f"{i} {j} {v:.6e}\n")

def build_G_sparse0_and_norm(G, d, dstd, outdir):
    """
    Build:
      G_sparse0
      d_sparse0
      dstd_sparse0
      G_sparse0_norm
      d_sparse0_norm
      dstd_sparse0_norm
      c_sparse0_norm
      G_sparse0_norm_firstraw
    """

    N, M = G.shape
    G = G.tocsr()

    # ---- 1. find sparsest row ----
    nnz_per_row = np.array([G[i].nnz for i in range(N)])
    sparseind = np.argmin(nnz_per_row)

    # ---- 2. swap rows in G ----
    Glil = G.tolil()
    Glil[0], Glil[sparseind] = Glil[sparseind].copy(), Glil[0].copy()
    G0 = Glil.tocsr()

    # ---- 3. swap rows in d and dstd ----
    d0 = d.copy()
    dstd0 = dstd.copy()
    d0[0], d0[sparseind] = d[sparseind], d[0]
    dstd0[0], dstd0[sparseind] = dstd[sparseind], dstd[0]

    # ---- 4. write sparse0 ----
    write_sparse_ijv(f"{outdir}/G_sparse0", G0)
    np.savetxt(f"{outdir}/d_sparse0", d0)
    np.savetxt(f"{outdir}/dstd_sparse0", dstd0)

    # ---- 5. normalize by dstd ----
    Dinv = diags(1.0 / dstd0)
    G0n = Dinv @ G0
    d0n = d0 / dstd0
    dstd0n = dstd0 / dstd0  # = ones

    # ---- 6. write normalized ----
    write_sparse_ijv(f"{outdir}/G_sparse0_norm", G0n)
    np.savetxt(f"{outdir}/d_sparse0_norm", d0n)
    np.savetxt(f"{outdir}/dstd_sparse0_norm", dstd0n)

    # ---- 7. c vector ----
    c0n = np.asarray(G0n.sum(axis=1)).ravel()
    np.savetxt(f"{outdir}/c_sparse0_norm", c0n)

    # ---- 8. first row of G ----
    with open(f"{outdir}/G_sparse0_norm_firstraw", "w") as f:
        row0 = G0n.getrow(0).toarray()[0]
        for j, v in enumerate(row0):
            f.write(f"{j} {v:.6e}\n")

    return sparseind

import os
from pathlib import Path
import configparser
import pandas as pd

# Parameters
parser = configparser.ConfigParser()
parser.read( 'parameters.cfg')
eta = parser.getfloat( 'inversion', 'eta')
input_folder=parser.get('folders','input')

# Read G matrix as ijv
from scipy.sparse import coo_matrix

# --- read G header to get N, M ---
with open(os.path.join(input_folder, 'G'), 'r') as f:
    _ = f.readline()  # nnz, not needed
    N, M = map(int, f.readline().split())



# --- read ijv ---
Gdf = pd.read_table(
    os.path.join(input_folder, 'G'),
    sep=r'\s+',
    skiprows=2,
    header=None,
    names=['i','j','val']
)

# --- convert to sparse matrix ---
G = coo_matrix(
    (Gdf['val'].values, (Gdf['i'].values, Gdf['j'].values)),
    shape=(N, M)
).tocsr()

d = np.loadtxt( os.path.join(input_folder, 'd'))
dstd = np.loadtxt(os.path.join(input_folder, 'dstd'))
sparseind = build_G_sparse0_and_norm(
    G=G,
    d=d,
    dstd=dstd,
    outdir=input_folder
)

print("Sparsest row index was:", sparseind)


# --- V file: cell volumes (here all = 1) ---
V = np.ones(M, dtype=float)
np.savetxt("sola_inputs_cornwall/V", V, fmt="%.1f")

# --- ks_to_solve file: indices 0..728 ---
ks = np.arange(M, dtype=int)
np.savetxt("sola_inputs_cornwall/ks_to_solve", ks, fmt="%d")

T_dir = "sola_inputs_cornwall/T"

# create T directory if it does not exist
os.makedirs(T_dir, exist_ok=True)

for k in range(M):
    fname = os.path.join(T_dir, f"T_{k}")
    with open(fname, "w") as f:
        f.write(f"{M}\n")
        f.write(f"{k} 1\n")

print(f"Created {M} target kernel files in {T_dir}")

