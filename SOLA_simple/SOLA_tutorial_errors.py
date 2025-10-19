#!/usr/bin/env python3
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.linalg as la
from scipy.stats import norm

# ---------------------------
# (Your original setup)
# ---------------------------

def generate_synthetic_data(G, true_model, error):
    synthetic_data = np.dot(G, true_model) + error
    return synthetic_data

G = np.array([[0.23017242, 0.88619105, 0.93037272, 0.69615894],
              [0.95956535, 0.30762742, 0.13533882, 0.20210273],
              [0.46599537, 0.86198157, 0.36085808, 0.89038005],
              [0.48276384, 0.67840301, 0.07236338, 0.42228805]])

m_true = np.array([1.5, 2.5, 3.5, 2.0])



error= np.array([-0.04319315, -0.09695092, 0.07192746, -0.00893804])

d_obs = generate_synthetic_data(G, m_true, error)

eta = 1.0

# Target kernels
T = np.array([[0.75, 0.08333, 0.08333, 0.083333],
              [0.08333, 0.75, 0.08333, 0.083333],
              [0.08333, 0.08333, 0.75, 0.083333],
              [0.08333, 0.08333, 0.083333, 0.75]])

c = np.sum(G, axis=1)
c_hat = c[1:] / c[0]

# Build B 
B = np.zeros((len(d_obs), len(d_obs) - 1))
B[0] = -c_hat
for i in range(1, len(d_obs)):
    B[i, i - 1] = 1.0

G_trans = G.T
Q = G_trans.dot(B)
last_line_Q = -eta * c_hat
Q = np.vstack([Q, last_line_Q])
Q = np.array(Q)

# Build y vectors
y_list = []
for k in range(len(T)):
    y = np.zeros(len(c) + 1)
    y[:-1] = T[k] - (1/c[0]) * (G_trans[0] * np.eye(4)[0])
    y[-1] = -(1/c[0]) * eta
    y_list.append(y)
y_array = np.array(y_list)

# Solve for x_hat with least squares
x_hat = []
for k in range(len(c)):
    sol, *_ = np.linalg.lstsq(Q, y_array[k], rcond=None)
    x_hat.append(sol)
x_hat = np.array(x_hat)

# Compute Gt (rows are averaging kernels / retrieval operators)
Gt = []
for i in range(len(x_hat)):
    row = B.dot(x_hat[i]) + (1 / c[0]) * np.eye(4)[0]
    Gt.append(row)
Gt = np.array(Gt)   # expected shape (nparams, ndata)
# check shape:
nparams, ndata = Gt.shape

# compute averaging kernels A
A = np.array([Gt[i].dot(G) for i in range(len(Gt))])

# model estimate from current observed data
m_est = Gt.dot(d_obs)        # shape (nparams,)
# current "deterministic" propagated uncertainty for the particular error values
uncert_realization = Gt.dot(error)

# ---------------------------
# Monte Carlo sampling to verify error propagation
# ---------------------------

# Choose data error standard deviation 

# For example: same sigma on every data element:
sigma_d_scalar = 0.01    
sigma_d = np.ones(ndata) * sigma_d_scalar


# Number of Monte Carlo samples
Nmc = 20000
rng = np.random.default_rng(12345)

# Build covariance for data (diagonal here)
Cov_d = np.diag(sigma_d**2)

# Analytic model covariance and std (from linear error propagation)
Cov_m_analytic = Gt.dot(Cov_d).dot(Gt.T)    # shape (nparams, nparams)
sigma_m_analytic = np.sqrt(np.maximum(np.diag(Cov_m_analytic), 0.0))

# Monte Carlo sampling: draw Nmc noise realisations and compute m_samples
# noise shape: (Nmc, ndata)
noise = rng.normal(loc=0.0, scale=sigma_d, size=(Nmc, ndata))
d_samples = d_obs.reshape(1, -1) + noise    # shape (Nmc, ndata)

# compute m_samples: (Nmc x ndata) @ (ndata x nparams) -> (Nmc x nparams)
m_samples = d_samples.dot(Gt.T)

# sample statistics
m_sample_mean = m_samples.mean(axis=0)
m_sample_std = m_samples.std(axis=0, ddof=1)

# Print comparisons
print("\nParameter-by-parameter comparison:")
for j in range(nparams):
    print(f"param {j}: m_est={m_est[j]:.6f}, sample_mean={m_sample_mean[j]:.6f}, "
          f"analytic_sigma={sigma_m_analytic[j]:.6f}, sample_std={m_sample_std[j]:.6f}")

# Check differences
mean_diff = m_sample_mean - m_est
std_rel_error = (m_sample_std - sigma_m_analytic) / np.where(sigma_m_analytic==0, 1.0, sigma_m_analytic)
print("\nMean differences (sample - analytic):", mean_diff)
print("Relative std error (sample vs analytic):", std_rel_error)

# ---------------------------
# Plot histograms of model parameters from MC samples
# ---------------------------
ncols = 2
nrows = int(np.ceil(nparams / ncols))
fig, axes = plt.subplots(nrows, ncols, figsize=(12, 3 * nrows))
axes = axes.flatten()

for j in range(nparams):
    ax = axes[j]
    vals = m_samples[:, j]
    # histogram (density) and KDE
    sns.histplot(vals, bins=80, stat='density', ax=ax, color='C0', alpha=0.6, edgecolor=None)
    # overlay Gaussian with analytic mean/std
    mu = m_est[j]
    sigma = sigma_m_analytic[j] if sigma_m_analytic[j] > 0 else 1e-12
    xgrid = np.linspace(vals.min(), vals.max(), 400)
    ax.plot(xgrid, norm.pdf(xgrid, loc=mu, scale=sigma), 'k--', lw=1.2,
            label=f'Analytic N(mean={mu:.3f},std={sigma:.3f})')
    # overlay Gaussian with sample mean/std (to visualise MC)
    ax.plot(xgrid, norm.pdf(xgrid, loc=m_sample_mean[j], scale=m_sample_std[j]), 'r-', lw=1.0,
            label=f'MC N(mean={m_sample_mean[j]:.3f},std={m_sample_std[j]:.3f})')
    ax.axvline(m_est[j], color='k', linestyle=':', lw=1)
    ax.axvline(m_sample_mean[j], color='r', linestyle=':', lw=1)
    ax.set_title(f"Model parameter {j}")
    ax.legend(fontsize=8)

for k in range(nparams, len(axes)):
    fig.delaxes(axes[k])

plt.tight_layout()
plt.show()

