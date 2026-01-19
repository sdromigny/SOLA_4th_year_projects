import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple

def plot_grid_3d_slices(
    m: np.ndarray,
    s: np.ndarray,
    nx: int, ny: int, nz: int,
    xll: float, yll: float, zll: float,
    dx: float, dy: float, dz: float,
    output_path: str,
    kavg: Optional[np.ndarray] = None,
    ktgt: Optional[np.ndarray] = None,
    slice_ix: Optional[int] = None,
    slice_iy: Optional[int] = None,
    slice_iz: Optional[int] = None,
    scale_m: Optional[float] = None,
    cmap_m: str = "magma",
    cmap_s: str = "gray",
    cmap_kavg: str = "viridis",
    cmap_ktgt: str = "viridis",
    vmin_vmax_m: Optional[Tuple[float,float]] = None,
    vmin_vmax_s: Optional[Tuple[float,float]] = None,
    vmin_vmax_kavg: Optional[Tuple[float,float]] = None,
    vmin_vmax_ktgt: Optional[Tuple[float,float]] = None,
    figsize: Tuple[int,int] = (18, 14)
):
    """
    3D equivalent of plot_grid(): plots orthogonal slices (XY, XZ, YZ)
    of model, uncertainty, averaging kernel, and target kernel.

    m, s, kavg, ktgt must be 1D arrays of length (nx-1)*(ny-1)*(nz-1)
    """

    ncx, ncy, ncz = nx-1, ny-1, nz-1
    M = ncx * ncy * ncz
    assert m.size == M, "Model size mismatch"

    if scale_m is not None:
        m = m * scale_m

    # reshape to (z, y, x)
    m3 = m.reshape((ncz, ncy, ncx))
    s3 = s.reshape((ncz, ncy, ncx))
    kavg3 = kavg.reshape((ncz, ncy, ncx)) if kavg is not None else None
    ktgt3 = ktgt.reshape((ncz, ncy, ncx)) if ktgt is not None else None

    # default slice = middle
    ix = slice_ix if slice_ix is not None else ncx // 2
    iy = slice_iy if slice_iy is not None else ncy // 2
    iz = slice_iz if slice_iz is not None else ncz // 2

    # cell-center coordinates
    xc = xll + dx * (np.arange(ncx) + 0.5)
    yc = yll + dy * (np.arange(ncy) + 0.5)
    zc = zll + dz * (np.arange(ncz) + 0.5)

    fields = [
        ("Model", m3, cmap_m, vmin_vmax_m),
        ("Uncertainty", s3, cmap_s, vmin_vmax_s)
    ]

    if kavg3 is not None:
        fields.append(("Averaging kernel", kavg3, cmap_kavg, vmin_vmax_kavg))
    if ktgt3 is not None:
        fields.append(("Target kernel", ktgt3, cmap_ktgt, vmin_vmax_ktgt))

    nrows = len(fields)
    fig, ax = plt.subplots(nrows, 3, figsize=figsize)

    for r, (title, F, cmap, clim) in enumerate(fields):
        # --- XY ---
        im = ax[r,0].pcolormesh(xc, yc, F[iz,:,:], shading="auto", cmap=cmap)
        ax[r,0].set_title(f"{title} – XY @ z={zc[iz]:.2f}")
        ax[r,0].set_xlabel("X")
        ax[r,0].set_ylabel("Y")
        fig.colorbar(im, ax=ax[r,0])
        if clim: im.set_clim(*clim)

        # --- XZ ---
        im = ax[r,1].pcolormesh(xc, zc, F[:,iy,:], shading="auto", cmap=cmap)
        ax[r,1].set_title(f"{title} – XZ @ y={yc[iy]:.2f}")
        ax[r,1].set_xlabel("X")
        ax[r,1].set_ylabel("Z")
        fig.colorbar(im, ax=ax[r,1])
        if clim: im.set_clim(*clim)

        # --- YZ ---
        im = ax[r,2].pcolormesh(yc, zc, F[:,:,ix], shading="auto", cmap=cmap)
        ax[r,2].set_title(f"{title} – YZ @ x={xc[ix]:.2f}")
        ax[r,2].set_xlabel("Y")
        ax[r,2].set_ylabel("Z")
        fig.colorbar(im, ax=ax[r,2])
        if clim: im.set_clim(*clim)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close(fig)
