import numpy as np
import seaborn as sns
import scipy.sparse.linalg as sp_linalg
import matplotlib.pyplot as plt





## function to create synthetic data

def generate_synthetic_data(G, true_model, error):

    """

    Generate synthetic observed data based on the true model and sensitivity matrix.

    """

    synthetic_data = np.dot(G, true_model) + error

    return synthetic_data



##################### Sensitivity matrix 



G = np.array([[0.23017242, 0.88619105, 0.93037272, 0.69615894],

 [0.95956535, 0.30762742, 0.13533882, 0.20210273],

 [0.46599537, 0.86198157, 0.36085808, 0.89038005],

 [0.48276384, 0.67840301, 0.07236338, 0.42228805]]) 



#### G : sensitivtiy matrix as identity matrix to see when parameters are independent on each other

#G=np.identity(4)


##################### Generate "observed" data 

# True model parameters for synthetic data generation

m_true = np.array([1.5, 2.5, 3.5, 2.0])


# error on the "observed" data

error=[-0.04319315, -0.09695092,  0.07192746, -0.00893804]

### error free data

error=[0,0,0,0]

# data generation

d_obs = generate_synthetic_data(G, m_true, error)


### with data error

#Gn=G/error


##################### Trade off parameter

eta=1



##################### Target kernels for each parameter

#ideal case for each parameter

T = np.eye(4)  # Identity matrix

# only for one parameter (to make it more stable?)

T= [[0.75,0.08333,0.08333,0.083333],

 [0.08333,0.75,0.08333,0.083333],

 [0.08333,0.08333,0.75,0.083333],

 [0.08333,0.08333,0.083333,0.75,]]






##################### Constraint : sum of the rows of G = 1



c = np.sum(G, axis=1)

print(c)

c_hat=c[1:]/c[0]

print(c_hat)





##################### Q matrix



# Initialize Q matrix dimensions

B = np.zeros((len(d_obs), len(d_obs) - 1))



# Populate the first row of B

B[0] = -c_hat



# Populate the identity matrix in the lower part of B

for i in range(1, len(d_obs)):

    B[i, i - 1] = 1



print(B)



G_trans=np.transpose(G)

# Compute Q matrix

Q = np.dot(G_trans,B)  # Q without the last row



last_line_Q = -eta * c_hat 



Q = np.vstack([Q, last_line_Q])

Q=np.array(Q)

print(Q)



################# y vector


# Initialize a list to hold all y vectors

y_list = []

# Compute y depending on the target kernels

for k in range(len(T)):

    y = np.zeros(len(c) + 1)

    y[:-1] = T[k] - (1/c[0]) * G_trans[0] * np.eye(4)[0]

    y[-1] = -(1/c[0]) * eta

    y_list.append(y)



# Convert the list to an array for easier handling

y_array = np.array(y_list)



print('y',y_array)



############ LSQR solving: x_hat solution for Q and y (one y per parameter)


# Initialize a list to hold all x_hat vectors

x_hat = []



# Iterate over the range from 1 to the length of c

for k in range(0, len(c)):

    # Solve the linear system using LSQR

    solution = np.linalg.lstsq(Q, y_array[k])[0]

    # Append the solution to Gt

    x_hat.append(solution)



# Convert the list to an array for easier handling


x_hat= np.array(x_hat)

print('x_hat',x_hat)


########## get Gt : pseudo-inverse matrix of the sensitivity matrix, gives us the avergaging kernels and model solution



Gt=[]



for i in range(len(x_hat)):

    Gt.append(B.dot(x_hat[i])+ 1/c[0]*np.eye(4)[0])

Gt=np.array(Gt)



print('Gt',Gt)



######## compute the averaging kernels stored in A and normalise A values for easier interpretation

A=[]

for i in range(len(x_hat)):

    A.append(np.dot(Gt[i],G))

    

print('A',np.array(A))





########## model solution



m=np.dot(Gt,d_obs)

print('model solution',m)

inv=np.dot(Gt,G)
m_compare= np.dot(inv,m_true)

print('m comparison',m_compare)

check=m-m_compare
print('check',check)

uncert=np.dot(Gt,error)

print('uncert',uncert)


########## Plotting the model solution with error bars and the averaging vs target kernels

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

x = np.arange(G.shape[1])
fig, ax = plt.subplots(figsize=(8, 5))
ax.set_title('True model vs SOLA / Backus-Gilbert estimates')
ax.errorbar(x - 0.05, m_true, yerr=0, fmt='o', label='true model')
ax.errorbar(x + 0.05, m, yerr=np.sqrt(uncert), fmt='s', label='estimated', capsize=5)
ax.set_xlabel('Model parameter index')
ax.set_ylabel('Parameter value')
ax.legend()

plt.tight_layout()
plt.show()
