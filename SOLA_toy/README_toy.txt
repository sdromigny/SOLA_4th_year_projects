SOLA-Backus-Gilbert inversion for linear, discrete, tomographic problems
========================================================================
by : Christophe Zaroli & Sophie Lambotte, University of Strasbourg

version: 2020, November 19

contact: c.zaroli@unistra.fr

========================================================================
TOY PROBLEM (Zaroli et al. 2017)
========================================================================

Model parameters are shear wave velocity perturbations (dlnVs), and the parametrization 
consists of M = 1,024 square pixels of unit area each. Data represent onset delay times 
of direct S waves, whose raypaths are straight lines.

Total number of data N = 9,778

config file (for all python routines) :  SOLA_toy.cfg


To be tested:
---------------
- influence of the Target radius : 
change the value of the parameter factmul in config file (section toy_problem) to make the size of the targets larger or smaller. 
This parameter is a multiplicative factor for all the target radii. 
For instance, factmul = 2 will multiply by 2 all the radii.

- influence of the trade-off parameter (trade-off between resolution and uncertainties) :
change the value of the parameter eta in config file (section inversion)

- explore averaging kernel shape :
change the value of k (parameter kvalue in config file - section PlottingKernel)

- for model filtering : 
change the True model (parameter input_model in config file - section filtering)


Input files in INPUT directory :
-----------------------------------
Gij_toy.txt : G matrix
   first line : total number of non-zero values 
   second line : nb data , nb parameters
   other lines : line index i , column index j , value of Gij

di_toy.txt : data vector
   lines : di, sigma_di

Vj_toy.txt : volume of each cell, or associated to each node
   lines : Vj value

Tk_radius.txt : radius used for the target kernels
   lines : index i_xcoord , index i_ycoord, global index k , radius


To compute target kernels (circles):
----------------------------------------
compute_target_toy.py

output :
 - in INPUT/Target_kernels/
Tk_*.txt : 1 column with value of the target kernel at each node/cell

to run it :
python compute_target_toy.py


To do preprocessing : 
-----------------------
SOLA_preprocess.py

preprocessing means : 
- dividing Gij by sigma_di (needed by LSQR)
- search for the most sparse line of Gij and permut with the first line
To be done once

output :
 - in INPUT/
Gij_toy_normSigma_sparse0.txt
(index i and j are sorted in ascending order - needed for C codes)

di_toy_sparse0.txt (same permutation as Gij)

to run it : 
python SOLA_preprocess.py -c SOLA_toy.cfg


To run SOLA :
-------------------
run_SOLABG.py
(use SOLA_utils.py)

output : 
 - in OUTPUT/
C_vector.txt
Q_matrix_com.npz : part of Q matrix independent of eta value 
	in config file : flag_new used to indicate if this part of Q matrix is already computed or not 
                         it is not necessary to recompute it for each value of eta tested, just once

 - in OUTPUT/OUTPUT_etaval/
Q_matrix_etaval.txt : full Q matrix (input for C codes)

 - in OUTPUT/OUTPUT_etaval/G_inverse/
xk_etaval_*.txt : each line of the inverse matrix

to run it : 
python run_SOLABG.py -c SOLA_toy.cfg


to run SOLA postprocessing
----------------------------------------
SOLA_postprocess.py

postprocessing means : 
- compute mk and sigma_mk (boolean flag in config file : do_mk)
- compute averaging kernel (C code) (boolean flag in config file : do_Ak)
- compute resolution matrix (C code) (boolean flag in config file : do_Rmatrix)

output : 
 - in OUTPUT/OUTPUT_etaval/
mk_etaval.txt : model mk and uncertainties (if go_mk = True in config file)
   lines : k, mk, sigma_mk

 - in OUTPUT/OUTPUT_etaval/ResKernels/
Ak_etaval_*.txt : Averaging kernel at node/cell k  (if do_Ak = True in config file)
Rij_etaval.txt : Resolution matrix (if do_Rmatrix = True in config file)

to run it :
python SOLA_postprocess.py -c SOLA_toy.cfg


To plot model : 
----------------------------------------
plotmodel_toy.py

to run it: 
python plotmodel_toy.py

figures saved in Figures directory


To plot Averaging kernel : 
----------------------------------------
plotAk_toy.py
 
to run it: 
python plotAk_toy.py

figures saved in Figures directory

to be tested :
- change value of k to explore averaging kernel shape (parameter kvalue in config file - section PlottingKernel)


To do filtering (Rm) :
----------------------------------------
run_filtering_toy.py

input : 
 - in SOLA_Filtering/True_models
several true models (gaussian, dots, uniform)

output:
 - in SOLA_Filtering/Filtered_models
filtered models (same format as true models)

to run it: 
python run_filtering_toy.py

figures saved in Figures directory

to be tested :
- different True models (parameter input_model in config file - section filtering)

