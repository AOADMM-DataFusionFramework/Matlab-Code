# Matlab-Code

This repository contains code for fitting coupled matrix and tensor factorizations with regularizations and linear couplings. The algorithm utilizes Alternating Optimization (AO) and Alternating Direction Method of Multipliers (ADMM). It is described in the paper "A Flexible Optimization Framework for Regularized Matrix-Tensor Factorizations with Linear Couplings" by C. Schenker, J. E. Cohen and E. Acar, 2020, available at https://ieeexplore.ieee.org/document/9298877.
The coupling with PARAFAC2 models is described in https://ieeexplore.ieee.org/abstract/document/10094562.
This code is open source and can be used by anyone, as long as the paper(s) are cited.

The implementation makes use of the following toolboxes/packages, which need to be downloaded and installed separately:
* Brett W. Bader, Tamara G. Kolda and others. MATLAB Tensor Toolbox, Version 3.1. Available online at https://www.tensortoolbox.org, 2020
* G. Chierchia, E. Chouzenoux, P. L. Combettes, and J.-C. Pesquet. "The Proximity Operator Repository. User's guide". Availaible online at http://proximity-operator.net/download/guide.pdf 
* S. Becker, “L-BFGS-B C code with Matlab wrapper,” 2019. Available online at https://github.com/stephenbeckr/L-BFGS-B-C, see also: R. H. Byrd, P. Lu and J. Nocedal. A Limited Memory Algorithm for Bound Constrained Optimization, (1995), SIAM Journal on Scientific and Statistical Computing , 16, 5, pp. 1190-1208

For different examples on how to use the code, run the example scripts.
