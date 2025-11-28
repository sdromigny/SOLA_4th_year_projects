# Some Python packages.

import numpy as np
import scipy
from scipy.sparse import linalg
import seaborn as sns
import sys
import numpy as np
from scipy import sparse


sys.path.insert(0, "./utils")  # This contains functions to compute G.
from utils.grid import *
from utils.straight_ray_tracer import *

# Set some parameters to make plots nicer.

plt.rcParams["font.family"] = "serif"
plt.rcParams.update({"font.size": 11})


# Define the numerical grid. ---------------------------------------------
dimension = 2  # Here we only consider 2D problems anyway.
x_min = 0.0  # Minimum x-coordinate
y_min = 0.0  # Minimum y-coordinate
dx = 2.5  # Grid spacing in x-direction
dy = 2.5  # Grid spacing in y-direction
Nx = 20.0  # Number of grid points in x-direction
Ny = 20.0  # Number of grid points in y-direction
g = grid(dimension, [x_min, y_min], [dx, dy], np.array([Nx, Ny]))

# Sources and receivers. -------------------------------------------------
src_locations = np.array([0.0 * np.ones((11,)), np.linspace(0, 50, 11)])
rec_locations = np.array([50.0 * np.ones((21,)), np.linspace(0, 50, 21)])

sources, receivers = get_all_to_all_locations(src_locations, rec_locations)
plot_rays(sources, receivers, g)

# Compute G and measure how long that takes.
G = create_forward_operator(sources, receivers, g)

# Print some statistics of G.
print("Matrix shape:            ", G.shape)
print("Data points:             ", G.shape[0])
print("Unknowns in model space: ", G.shape[1])
print("Non-zero entries:        ", G.count_nonzero())
print(
    "Ratio of non-zeros: {:10.4f} %".format(
        100 * G.count_nonzero() / (G.shape[0] * G.shape[1])
    )
)

# Plot ray density and entries of G.
plot_ray_density(G, g)

# Plot non-zero matrix entries.
print("Sparsity pattern of the forward matrix:")
plt.figure(figsize=(15, 20))
plt.spy(G, markersize=2, color="k")
plt.gca().xaxis.tick_bottom()
plt.xlabel("model space index")
plt.ylabel("data space index")
plt.title(r"non-zero entries of $\mathbf{G}$")
plt.savefig("non-zeros.pdf", format="pdf")
plt.show()

# Input model setup (chequerboard). --------------------------------------
dvp = 100.0  # velcity variations in m/s.
dd = 4  # Width of the chequerboard cells in number of cells.

# Allocate velocity matrix. Homogeneous background model.
vp = 3000.0 * np.ones(g.npoints)

# Add some heterogeneities
s = 1.0
for i in range(0, g.npoints[0], dd):

    for j in range(0, g.npoints[1], dd):
        end_i = min(g.npoints[0], i + dd)
        end_j = min(g.npoints[1], j + dd)
        vp[i:end_i, j:end_j] += s * dvp
        s *= -1

m_true = (1 / vp).ravel()

dim=len(m_true)


clim = [1 / 3.1, 1 / 2.9]
plot_model(
    1000.0 * m_true, g, "true model [ms/m]", caxis=clim, savename="true_model.pdf"
)

eta=1

# Create observed data ---------------------------------------------------
d_true = G * m_true

# Prior covariance parameters. -------------------------------------------
sigma_d = 0.2e-4  # Data standard deviation.
d_obs = d_true + sigma_d * np.random.randn(len(d_true))

# Data covariance matrix. ------------------------------------------------
Cd = sigma_d ** 2 * scipy.sparse.eye(len(d_obs))
Cd_inv = 1 / sigma_d ** 2 * scipy.sparse.eye(len(d_obs))

# Traveltimes.
plt.subplots(figsize=(15, 10))
plt.plot(1000.0 * d_obs, "k")
plt.ylabel("travel time [ms]")
plt.xlabel("ray path idx")
plt.show()

# Traveltime errors.
plt.subplots(figsize=(15, 10))
plt.plot(1000.0 * (d_obs - d_true), "k")
plt.ylabel("travel time errors [ms]")
plt.xlabel("ray path idx")
plt.show()


T = np.eye(dim)  # Identity matrix




##################### Constraint : sum of the rows of G = 1



c = np.asarray(G.sum(axis=1)).ravel()





c_hat=c[1:]/c[0]







##################### Q matrix



# Initialize Q matrix dimensions

B = np.zeros((len(d_obs), len(d_obs) - 1))



# Populate the first row of B

B[0] =(-c_hat)



# Populate the identity matrix in the lower part of B

for i in range(1, len(d_obs)):

    B[i, i - 1] = 1







G_trans=np.transpose(G)


# --- Build Q robustly ---
Q0 = np.asarray(G_trans.dot(B))   # shape (M, N-1)
M, Nminus1 = Q0.shape


last_line_Q = (-eta * np.asarray(c_hat).ravel()).reshape(1, -1)
if last_line_Q.shape[1] != Nminus1:
    raise ValueError(f"c_hat length ({last_line_Q.shape[1]}) != Q0 columns ({Nminus1})")

Q = np.vstack([Q0, last_line_Q])   # shape (M+1, N-1)
m = Q.shape[0]                      # m = M + 1

# --- Pre-check shapes for debugging ---
print("Q0.shape =", Q0.shape)
print("Q.shape  =", Q.shape)
print("T.shape  =", getattr(T, "shape", None))
print("G_trans.shape =", getattr(G_trans, "shape", None))
print("len(c)   =", len(c))








# --- Solve for each k: build y(k,eta) with correct length m ---
x_hat = []
c1 = float(np.asarray(c[0]))   # c1 scalar, ensure python float

for k in range(len(T)):
    # t_k must be length M (top part)
    t_k = np.asarray(T[k]).ravel()
    if t_k.size != M:
        raise ValueError(f"t_k (len={t_k.size}) must have length M={M} (G_trans rows).")

    # get first column of G_trans as dense vector length M
    if sparse.issparse(G_trans):
        g_first = G_trans[:, 0].toarray().ravel()
    else:
        g_first = np.asarray(G_trans[:, 0]).ravel()


    y_top = t_k - (1.0 / c1) * g_first

    # bottom part: scalar -c1^{-1} * eta
    y_bottom = np.array([-(1.0 / c1) * eta])

    # final y_k has length M+1 == m
    yk = np.concatenate([y_top, y_bottom])
    assert yk.shape == (m,), f"yk.shape == {yk.shape} but expected ({m},)"

    # solve least-squares
    sol = np.linalg.lstsq(Q, yk, rcond=None)[0]   # length Nminus1
    x_hat.append(sol)



x_hat = np.asarray(x_hat)
print("x_hat.shape =", x_hat.shape)   # (len(T), Nminus1)




########## get Gt : pseudo-inverse matrix of the sensitivity matrix, gives us the avergaging kernels and model solution



Gt=[]




for i in range(len(x_hat)):

    Gt.append(B.dot(x_hat[i])+ 1/c[0]*np.eye(len(c))[0])

Gt=np.array(Gt)


######## compute the averaging kernels stored in A and normalise A values for easier interpretation

G_dense = G.toarray() if sparse.issparse(G) else np.asarray(G)

A=[]

for i in range(len(x_hat)):

    A.append(np.dot(Gt[i],G_dense))



row=0


A=np.array(A)
avk=A[row, :].reshape(int(Nx), int(Ny))



########## Plotting the averaging vs target kernels

# Plot the averaging-kernel matrix and the target-kernel matrix side-by-side
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Left: Averaging kernels (resolution matrix rows stacked)
ax0 = axes[0]
im0 = sns.heatmap(A, cmap='Greys', ax=ax0, cbar=False, square=True)
ax0.set_title('Averaging kernel matrix (A)')
ax0.set_xlabel('Model parameter index')
ax0.set_ylabel('Averaging kernel index')

# Add colorbar for the left heatmap
cbar0 = fig.colorbar(im0.get_children()[0], ax=ax0, fraction=0.046, pad=0.04)
cbar0.set_label('Amplitude', color='k')

# Right: Target kernels (columns are targets; show as matrix)
ax1 = axes[1]
im1 = sns.heatmap(T, cmap='Greys', ax=ax1, cbar=False, square=True)
ax1.set_title('Target kernel matrix (T)')
ax1.set_xlabel('Target index')
ax1.set_ylabel('Model parameter index')

# Add colorbar for the right heatmap
cbar1 = fig.colorbar(im1.get_children()[0], ax=ax1, fraction=0.046, pad=0.04)
cbar1.set_label('Amplitude', color='k')

plt.tight_layout()
plt.show()

# --- Reconstruct model from Gt and observed data ---


m_from_Gt = Gt.dot(d_obs)   

inv=np.dot(Gt,G_dense)
m_compare= np.dot(inv,m_true)


plot_model(m_from_Gt*1000.0, g, "Recovered model from Gt [ms/m]", caxis=clim, savename="recovered_model.pdf")

plot_model(m_compare*1000.0, g, "Compared recovered to the true solution [ms/m]", caxis=clim, savename="compared_model.pdf")




# diagonal of model covariance:
covm_diag = (sigma_d**2) * np.sum(Gt**2, axis=1)   # shape (Nm,)



covm_diag = np.asarray(covm_diag)           
m_std = np.sqrt(np.maximum(0.0, covm_diag))  # std (same units as m)





m_std_msperm = m_std * 1000.0



m_std_grid = m_std_msperm.reshape(int(Nx), int(Ny))   


vmin = 0.0
vmax = np.percentile(m_std_grid, 99)

plt.figure(figsize=(6,5))
im = plt.imshow(
    m_std_grid,
    origin='lower',
    interpolation='nearest',
    cmap='Greys'      
)
plt.colorbar(im, label='σ (ms/m)')
plt.clim(vmin, vmax)
plt.title('Model standard deviation (±1σ), linear scale')
plt.xlabel('x index'); plt.ylabel('y index')
plt.tight_layout()
plt.savefig('uncert.png')
plt.show()
