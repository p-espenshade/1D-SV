# 1D-SV
Sample variance calculator for cosmological observations

This code attempts to replicate the computations found in Espenshade & Yoo (2023) [arXiv:2304.09191](https://arxiv.org/abs/2304.09191). Disclaimer, this is not the official code used to generate the figures in that paper, and is not meant to be reflective of the authors, and as such, no promises are made about the correctness or completeness of the code. 

1D-SV quickly computes the cosmological sample variance for a survey using a 2-point correlation function and given survey geometry. Geometries are specified using an opening angle and redshift range and include cone, line-of-sight, box, and sphere shapes. The survey can be arbitrarily narrow and the 2-point correlation can be nonlinear without issue for the computation's convergence.

See the paper for more details about derivations. 

## Installation
Required libraries: [matplotlib](https://matplotlib.org/), [numpy](https://numpy.org/), [scipy](https://scipy.org/), [mpmath](https://mpmath.org/).

Optional libraries: [mpi4py](https://mpi4py.readthedocs.io/en/stable/) (for parallelization), [nbodykit](https://nbodykit.readthedocs.io/en/latest/).

Installation of most libraries can be done with PyPI:

```pip install <packageName>```

For mpi4py, Linux users can install it via PyPI. For Windows users, it's recommended to install [Windows Subsystem Linux](https://ubuntu.com/wsl) (WSL) by following the [UZH S3IT instructions](https://docs.s3it.uzh.ch/how-to_articles/how_to_set_up_a_linux_terminal_environment_on_windows_10_with_windows_subsystem_for_linux/). nbodykit can be installed via [Miniconda](https://docs.conda.io/en/latest/miniconda.html) through the nbodykit [instructions](https://nbodykit.readthedocs.io/en/latest/getting-started/install.html).

## Usage
To compute the sample variance, enter your desired survey geometry in ```example.py``` as well as the source of your 2-point correlation function, then run: 

```python example.py```

For narrow geometries, or very small-scale separations, you may wish to compute the cone probability density function in parallel. Set the flag ```parallel=True``` in ```example.py``` and run (e.g., for 4 processors):

```mpiexec -n 4 python example.py```

To use nbodykit to compute a 2-point correlation, set ```useNbodykit=True``` in ```example.py``` and run ```example.py``` either in serial or parallel.

# Citation
If you find the ideas in Espenshade & Yoo (2023) [arXiv:2304.09191](https://arxiv.org/abs/2304.09191) useful for your work, or if you find the other libraries useful (e.g., nbodykit), please cite the respective authors. Thank you!

# Contact
[Personal homepage](https://p-espenshade.github.io/). For questions or issues about the code, feel free to open an issue on GitHub or email peter.espenshade(at)uzh.ch.
