import numpy as np
import matplotlib.pyplot as plt
import configparser
from PlottingTools.Grid3d import *
from PlottingTools.ReadTarget import *
import os

# -------------------- CONFIG --------------------
parser = configparser.ConfigParser()
parser.read("parameters.cfg")

eta = parser.getfloat("inversion", "eta")
input_folder = parser.get("folders", "input")
output_folder = parser.get("folders", "output")

# --- grid definition (same as used in inversion) ---
minx = parser.getfloat("xyz_grid", "xll")
miny = parser.getfloat("xyz_grid", "yll")
minz = parser.getfloat("xyz_grid", "zll")

dx = parser.getfloat("xyz_grid", "xstep")
dy = parser.getfloat("xyz_grid", "ystep")
dz = parser.getfloat("xyz_grid", "zstep")

nx = parser.getint("xyz_grid", "xur")
ny = parser.getint("xyz_grid", "yur")
nz = parser.getint("xyz_grid", "zur")

# cells, not nodes
ncx, ncy, ncz = nx - 1, ny - 1, nz - 1

# which cell to highlight
cell = parser.getint("xyz_grid", "cell")

# -------------------- LOAD SOLA OUTPUTS --------------------
#Which cell to plot the target and averaging kernels
cell = parser.getint('xyz_grid', 'cell')
cell_str = str(cell)

# --- Load model vector and uncertainty ---
eta_str = f"{eta:.1f}".replace('.', 'p')
print(eta_str)
eta_str='0p01'
m = np.loadtxt(os.path.join(output_folder, f"m_{eta_str}"))
s = np.loadtxt(os.path.join(output_folder, f"mstd_{eta_str}"))
a = np.loadtxt(os.path.join(output_folder, f"A/A_{cell}"))
t = load_target_kernel(os.path.join(input_folder, f"T/T_{cell_str}"))

# optional scaling (you had m = m * 1000 previously)
scale_val = 100.0 #if scale_m_by_1000 else None

assert m.size == ncx * ncy * ncz

# reshape model vector → 3D grid
M3 = m.reshape((ncz, ncy, ncx))        # z, y, x
S3 = s.reshape((ncz, ncy, ncx))

# compute cell center coordinates
xc = minx + dx * (np.arange(ncx) + 0.5)
yc = miny + dy * (np.arange(ncy) + 0.5)
zc = minz + dz * (np.arange(ncz) + 0.5)

# pick slice indices (middle slices)
ix = ncx // 2
iy = ncy // 2
iz = ncz // 2

# -------------------- PLOTTING --------------------
fig = plt.figure(figsize=(15, 5))

# --- XY slice ---
ax1 = fig.add_subplot(131)
im1 = ax1.pcolormesh(
    xc, yc,
    M3[iz, :, :],
    shading="auto",
    cmap="seismic"
)
ax1.set_title(f"XY slice at z={zc[iz]:.2f}")
ax1.set_xlabel("X")
ax1.set_ylabel("Y")
plt.colorbar(im1, ax=ax1)

# --- XZ slice ---
ax2 = fig.add_subplot(132)
im2 = ax2.pcolormesh(
    xc, zc,
    M3[:, iy, :],
    shading="auto",
    cmap="seismic"
)
ax2.set_title(f"XZ slice at y={yc[iy]:.2f}")
ax2.set_xlabel("X")
ax2.set_ylabel("Z")
plt.colorbar(im2, ax=ax2)

# --- YZ slice ---
ax3 = fig.add_subplot(133)
im3 = ax3.pcolormesh(
    yc, zc,
    M3[:, :, ix],
    shading="auto",
    cmap="seismic"
)
ax3.set_title(f"YZ slice at x={xc[ix]:.2f}")
ax3.set_xlabel("Y")
ax3.set_ylabel("Z")
plt.colorbar(im3, ax=ax3)

# -------------------- OUTPUT --------------------
os.makedirs(output_folder, exist_ok=True)
out_fig = os.path.join(output_folder, f"solution_3d_{eta_str}.png")
plt.tight_layout()
# plt.savefig(out_fig, dpi=300)
plt.close()

print(f"Saved 3D solution plot to {out_fig}")

plot_grid_3d_slices(
    m=m,
    s=s,
    nx=nx, ny=ny, nz=nz,
    xll=minx, yll=miny, zll=minz,
    dx=dx, dy=dy, dz=dz,
    kavg=a,          # A_k for a given cell
    ktgt=t,          # T_k
    slice_iz=2,      # choose depth slice
    scale_m=100.0,   # optional scaling
    output_path=os.path.join(output_folder, f"solution_3d_{eta_str}.png"),
    vmin_vmax_m=(-5,5)  # e.g. % velocity anomaly
)
