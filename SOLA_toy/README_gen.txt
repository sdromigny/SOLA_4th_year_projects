SOLA-Backus-Gilbert inversion for linear, discrete, tomographic problems
========================================================================
by : Christophe Zaroli & Sophie Lambotte, University of Strasbourg

version: 2020, November 19

contact: c.zaroli@unistra.fr

========================================================================
GENERAL PROBLEM 
========================================================================
Gij, Vj, di, Tk must be created by user (inputs)
  depend on grid and physical problem

plots have to be done by user (grid dependent)

config file example (for all python routines) :  SOLA.cfg
to be modified to fit user problem


To be tested:
---------------
- influence of the target size 

- influence of the trade-off parameter (trade-off between resolution and uncertainties) :
change the value of the parameter eta in config file (section inversion)

- explore averaging kernel shape 


Input files in INPUT directory :
-----------------------------------
Gij_xxxxx.txt : G matrix
   first line : total number of non-zero values
   second line : nb data , nb parameters
   other lines : line index i , column index j , value of Gij
(or Gij_xxxxx_normSigma.txt if Gij is already normalised by sigma_di)

di_xxxxx.txt : data vector
   lines : di, sigma_di

Vj_xxxxx.txt : volume of each cell, or associated to each node
   lines : Vj value

- in INPUT/Target_kernels/
Tk_*.txt : 1 column with value of the target kernel at each node/cell


To do preprocessing : 
-----------------------
SOLA_preprocess.py

preprocessing means : 
- dividing Gij by sigma_di
- search for the most sparse line of Gij and permut with the first line
To be done once

output :
 - in INPUT/
Gij_xxxxx_normSigma_sparse0.txt 
(index i and j are sorted in ascending order)

di_xxxxx_sparse0.txt (same permutation as Gij)

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
Q_matrix_etaval.txt : full Q matrix (input of C codes)

 - in OUTPUT/OUTPUT_etaval/G_inverse/
xk_etaval_*.txt : each line of the inverse matrix

to run it : 
python run_SOLABG.py -c SOLA_toy.cfg


to run SOLA postprocessing
----------------------------------------
SOLA_postprocess.py

postprocessing means : 
- compute mk and sigma_mk (boolean flag in config file - section postprocessing : do_mk)
- compute averaging kernel (C code) (boolean flag in config file - section postprocessing : do_Ak)
- compute resolution matrix (C code) (boolean flag in config file - section postprocessing : do_Rmatrix)

output : 
 - in OUTPUT/OUTPUT_etaval/
mk_etaval.txt : model mk and uncertainties (if do_mk = True in config file)
   lines : k, mk, sigma_mk

 - in OUTPUT/OUTPUT_etaval/ResKernels/
Ak_etaval_*.txt : Averaging kernel at node/cell k  (if do_Ak = True in config file)
Rij_etaval.txt : Resolution matrix (if do_Rmatrix = True in config file)

to run it :
python SOLA_postprocess.py -c SOLA_toy.cfg


