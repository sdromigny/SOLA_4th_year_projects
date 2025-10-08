SOLA-Backus-Gilbert inversion for linear, discrete, tomographic problems 
========================================================================
by : Christophe Zaroli & Sophie Lambotte, University of Strasbourg

version: 2020, November 19

contact: c.zaroli@unistra.fr

========================================================================

In brief:

This is a python/C package to run SOLA linear inversion. 
It contains an example : toy problem from Zaroli et al. 2017. 

Most of the computational part of the package are performed in C. 
The wrapper is designed in python, and has been tested under python 3 (see requirements below).

========================================================================

For further details on SOLA tomography see: 

Zaroli, C., 2016.  Global seismic tomography using Backus-Gilbert inversion, 
      Geophysical Journal International, 207 (2): 876-888, doi: 10.1093/gji/ggw315

Zaroli, C., Koelemeijer, P., and Lambotte, S., 2017. Toward Seeing the Earth's Interior Through Unbiased Tomographic Lenses, 
      Geophysical Research Letters, 44, 11,399–11,408, doi: 10.1002/2017GL074996

Zaroli, C., 2019.  Seismic tomography using parameter-free Backus–Gilbert inversion, 
      Geophysical Journal International, 218 (1): 619–630, doi: 10.1093/gji/ggz175

Please cite these papers if you're using this software.



create virtual environment
-----------------------------------
with conda:
conda create --name sola
conda activate sola
conda install numpy scipy pandas matplotlib


with venv:
python3 -m venv sola
source sola/bin/activate
pip install numpy scipy pandas matplotlib



python requirements : 
-----------------------------------
- numpy 
- pandas 
- scipy
- matplotlib (for plots - toy problem only)

Main python codes :
------------------------
- SOLA_preprocessing.py  : Gij normalisation and sparsity
- run_SOLABG.py          : prepare input for LSQR SOLA inversion and launch it
- SOLA_postprocessing.py : compute solution mk, averaging kernels and resolution matrix

Compilation of C codes : 
-----------------------------------
source codes are in directory LSQR_SOLA : 
Edit the Makefile to change compiler if needed (gcc used by default)
Run : make all

2 main C codes (and executables):
- Compute_AveragingKernels_R_for_SOLA : compute averaging kernels or resolution matrix
- LSQR_inversion_for_SOLA_N : LSQR for SOLA inversion

Path of executables has to be indicated in the config file (see SOLA_toy.cfg and SOLA.cfg).


How to run it :
----------------------------------
For information about all the steps, file format, directory architecture, inputs needed, etc, see the documentation in "docs/" and the README files.
Two README files are available : 
- README_toy.txt : to run the toy problem from Zaroli et al. (2017)
- README_gen.txt : to run the package for any linear problem
