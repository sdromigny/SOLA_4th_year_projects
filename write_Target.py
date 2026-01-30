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
