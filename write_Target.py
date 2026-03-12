import numpy as np
import os
from multiprocessing import Pool, cpu_count

def WhereAmI3D(xi, yi, zi, nx, ny, nz):
    i = int(xi)
    j = int(yi)
    k = int(zi)
    return k + nz * (j + ny * i)

def compute_Tk_sphere(xk, yk, zk, radius, nx, ny, nz):

    dx = 0.01
    dy = dx
    dz = dx

    x = np.arange(max(0, xk - radius), min(nx - 0.001, xk + radius), dx)
    y = np.arange(max(0, yk - radius), min(ny - 0.001, yk + radius), dy)
    z = np.arange(max(0, zk - radius), min(nz - 0.001, zk + radius), dz)

    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

    # volume element
    dV = dx * dy * dz

    # points inside the sphere
    inds = np.where(
        (X - xk)**2 + (Y - yk)**2 + (Z - zk)**2 <= radius**2
    )

    Tk = np.zeros(nx * ny * nz)

    for ix, iy, iz in zip(inds[0], inds[1], inds[2]):
        xi = x[ix]
        yi = y[iy]
        zi = z[iz]
        ijk = WhereAmI3D(xi, yi, zi, nx, ny, nz)
        Tk[ijk] += dV

    # normalization
    Tk /= np.sum(Tk)

    return Tk

def compute_Tks_3D(nx, ny, nz, cfg, radius, writefile=True):

    indir = cfg.indir
    Tks = []
    path = os.path.join(indir, 'Target_kernels_3D')

    if not os.path.exists(path):
        os.mkdir(path)

    for ix, iy, iz, ik, r in radius:
        ik = ik - 1
        xk = (ix - 1) + 0.5
        yk = (iy - 1) + 0.5
        zk = (iz - 1) + 0.5

        Tk = compute_Tk_sphere(xk, yk, zk, r, nx, ny, nz)
        Tks.append(Tk)

        if writefile:
            print(f'write Tk for k = {ik}')
            fileTi = os.path.join(path, f'Tk_{ik}.txt')
            write_Tk(Tk, ik, fileTi)

    return Tks


def write_Tk_sparse(Tk, ncell, fname, tol=0.0):
    """
    Write target kernel in sparse format:
    - first line: total number of cells
    - following lines: cell_index  Tk_value
    """

    with open(fname, 'w') as f:
        f.write(f"{ncell}\n")

        for i, val in enumerate(Tk):
            if abs(val) > tol:
                f.write(f"{i:d} {val:.10e}\n")


def compute_all_Tks_3D(nx, ny, nz, radius, cfg, tol=0.0):

    ncell = nx * ny * nz
    outdir = os.path.join(cfg.indir, "T")

    if not os.path.exists(outdir):
        os.mkdir(outdir)

    k = 0
    for ix in range(nx):
        for iy in range(ny):
            for iz in range(nz):

                # cell center
                xk = ix + 0.5
                yk = iy + 0.5
                zk = iz + 0.5

                Tk = compute_Tk_sphere(
                    xk, yk, zk,
                    radius,
                    nx, ny, nz
                )

                fname = os.path.join(outdir, f"T_{k}")
                write_Tk_sparse(Tk, ncell, fname, tol=tol)

                if k % 50 == 0:
                    print(f"Written kernel {k}/{ncell}")

                k += 1

def _compute_single_Tk(args, cfg, tol=0.0):
    outdir = os.path.join(cfg.indir, "T")
    ix, iy, iz, radius, nx, ny, nz= args
    ncell = nx * ny * nz

    xk = ix + 0.5
    yk = iy + 0.5
    zk = iz + 0.5

    Tk = compute_Tk_sphere(
    xk, yk, zk,
    radius,
    nx, ny, nz
    )

    fname = os.path.join(outdir, f"T_{k}")
    write_Tk_sparse(Tk, ncell, fname, tol=tol)

def parallel_Tks(nx, ny, nz, radius, cfg, tol=0.0):
    

    args = [
        (ix, iy, iz, radius, nx, ny, nz)
        for ix, iy, iz in range(nx)
    ]

    with Pool(cpu_count()) as pool:
        Tks = pool.map(_compute_single_Tk, args, cfg, tol=tol)

    return Tks