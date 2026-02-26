# Computes adaptive target kernels for a 3D grid based on local ray coverage, writing sparse kernel files.
import numpy as np
import os
from write_Target import compute_Tk_sphere, write_Tk_sparse

def compute_coverage_radii(coverage, radius_min, radius_max, power=0.5):
    """
    Map a 3D coverage array to per-cell radii.

    High coverage  → small radius (radius_min)
    Low coverage   → large radius (radius_max)

    Parameters
    ----------
    coverage   : np.ndarray, shape (nx, ny, nz)
    radius_min : float — smallest radius (high-coverage cells)
    radius_max : float — largest radius (low/no-coverage cells)
    power      : float — 0.5 compresses dynamic range (gentler),
                         1.0 is linear, >1 more aggressive
    """
    c = coverage.astype(float)
    c_max = c.max()
    if c_max == 0:
        return np.full(c.shape, radius_max)

    c_norm = (c / c_max) ** power       # high coverage → 1
    radii = radius_max - (radius_max - radius_min) * c_norm
    return radii


def compute_adaptive_Tks_3D(nx, ny, nz, coverage, outdir,
                             radius_min=1.0,radius_max=2.0,
                             power=0.5, tol=0.0):
    """
    Compute adaptive target kernels for every cell.

    Each cell gets a spherical kernel whose radius is determined by
    local data coverage: more coverage → smaller radius.

    Parameters
    ----------
    nx, ny, nz  : int — grid dimensions
    coverage    : np.ndarray, shape (nx, ny, nz)
    cfg         : config object with cfg.indir attribute
    radius_min  : minimum kernel radius (well-covered cells)
    radius_max  : maximum kernel radius (poorly-covered cells)
    power       : exponent controlling how aggressively radius varies
    tol         : sparsity threshold for writing
    """
    ncell = nx * ny * nz
    outdir = os.path.join(outdir, "T_adaptive")
    if not os.path.exists(outdir):
        os.mkdir(outdir)

    radii = compute_coverage_radii(coverage, radius_min, radius_max, power)

    radii_flat = radii.flatten()
    print(f"Adaptive radii — min: {radii_flat.min():.3f}, "
          f"max: {radii_flat.max():.3f}, "
          f"mean: {radii_flat.mean():.3f}")

    k = 0
    for ix in range(nx):
        for iy in range(ny):
            for iz in range(nz):
                xk = ix + 0.5
                yk = iy + 0.5
                zk = iz + 0.5

                r = radii[ix, iy, iz]

                Tk = compute_Tk_sphere(xk, yk, zk, r, nx, ny, nz)

                fname = os.path.join(outdir, f"T_{k}")
                write_Tk_sparse(Tk, ncell, fname, tol=tol)

                if k % 50 == 0:
                    print(f"Written kernel {k}/{ncell}  "
                          f"(cell [{ix},{iy},{iz}], radius={r:.3f})")
                k += 1

    # Save radius field for diagnostics
    radii_path = os.path.join(outdir, "adaptive_radii.npy")
    np.save(radii_path, radii)
    print(f"\nDone. Radius field saved to {radii_path}")

# ===== EXECUTION BLOCK ======
coverage = np.load('ray_coverage.npy').reshape(9, 9, 9)

compute_adaptive_Tks_3D(
    nx=9, ny=9, nz=9,
    coverage=coverage,
    outdir='/home/users/exet5760/Documents/SOLA_4th_year_projects/',
    radius_min=1.0,
    radius_max=2.0,
    power=0.5,
    tol=1e-8
)
