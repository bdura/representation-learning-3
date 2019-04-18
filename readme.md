# Assignment 3

The models are implemented in models and trained in learning. They are then used 
'notebooks' (also in 'base' for the part 1)


The first part implements discriminators and benchmark them. It is then used 
to estimate densities in 'density estimation' using several other scripts 
in 'base'.

The second part implements a VAE. It is evaluated in run_vae, in the same script
used for training.

The third part uses a generator similar to the one defined in VAE and then 
either uses a critic as defined in part one or use it in a variational
framework as in part 2 for the SVHN.