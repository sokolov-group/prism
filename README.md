# Prism
Prism is a Python implementation of electronic structure theories for simulating spectroscopic properties.
Currently, Prism features the methods of multireference algebraic diagrammatic construction theory (MR-ADC) for simulating core-ionized states (CVS-IP).

# How to install
## Requirements
- Python 3.7 or older;
- [PySCF 2.2 or older](https://github.com/pyscf/pyscf/), including its [dependencies](https://pyscf.org/install.html);
- Optional: [opt_einsum](https://optimized-einsum.readthedocs.io/en/stable/) for faster tensor contractions

## Installation
1) Install [PySCF](https://github.com/pyscf/pyscf/) and make sure it is included in the ``$PYTHONPATH`` environment variable
2) Clone the Prism repository:
```python
git clone https://github.com/sokolov-group/prism.git
```
3) Include the path to the folder where Prism is located in the ```$PYTHONPATH``` environment variable
4) Optional: download [opt_einsum](https://optimized-einsum.readthedocs.io/en/stable/) and include it in the ``$PYTHONPATH`` environment variable
5) Run tests to make sure the code is working properly

# How to use
The Prism calculations are run by means of creating and executing a Python script, which serves as the input file.
The electronic structure methods implemented in Prism require one- and two-electron integrals, molecular orbitals, and reference wavefunctions, which must be computed using PySCF.
To set up a calculation with Prism, import the following modules (in addition to any others you may need):

```python
from pyscf import gto, scf, mcscf
import prism.interface
import prism.mr_adc
```

Next, as described in the [PySCF user guide](https://pyscf.org/user.html), specify molecular geometry, then run reference Hartree-Fock and complete active space self-consistent field (CASSCF) calculations.
Below is an example of reference CASSCF calculation for the hydrogen fluoride (HF) molecule with the cc-pvdz basis set and 6 electrons in 6 orbitals (6e, 6o) active space.

```python
mol = gto.M(atom = 'H 0 0 0; F 0 0 0.91', basis = 'cc-pvdz')
mf = scf.RHF(mol).run()
mc = mcscf.CASSCF(mf, 6, 6).run()
```

Once the reference calculation is successfully completed, the objects of Hartree-Fock and CASSCF classes (```mf``` and ```mc```) are passed to Prism and the spectroscopic properties are calculated:

```python
interface = prism.interface.PYSCF(mf, mc, opt_einsum = True)
mr_adc = prism.mr_adc.MRADC(interface)
mr_adc.method = "mr-adc(2)"
mr_adc.method_type = "cvs-ip"
mr_adc.nroots = 10
mr_adc.ncvs = 1
e, p, x = mr_adc.kernel()
```

In the example above, a calculation using CVS-IP-MR-ADC(2) for 10 excited states (roots) is set up. 
The parameter ```ncvs``` controls the number of core orbitals in the hydrogen fluoride molecule, for which excited states are calculated.
For example, setting ```ncvs = 1``` corresponds to exciting electrons from the 1s orbitals of fluorine atoms, while ```ncvs = 2``` corresponds to probing the 2s excitations.
Other examples can be found [here](examples/).

# Methods and algorithms
## Multireference algebraic diagrammatic construction theory
Multireference algebraic diagrammatic construction theory can simulate a variety of excited electronic states (neutral excitations, ionization, electron attachment, core excitation and ionization).
The type of excited states is controled by the ```method_type``` parameter of MR-ADC class.
Currently, the only excited states that can be simulated using MR-ADC in Prism are core-ionized states probed in photoelectron spectroscopy.
These excitations are simulated by introducing core-valence separation approximation (CVS) and the resulting method is abbreviated as CVS-IP-MR-ADC.

The CVS-IP-MR-ADC calculations can be performed at four different levels of theory that are specified using the ```method``` parameter: ```'mr-adc(0)'```, ```'mr-adc(1)'```, ```'mr-adc(2)'```, ```'mr-adc(2)-x'```.

Other important parameters are:
 - ```ncvs``` (integer): The number of core orbitals to be included in the simulation. This number should ideally correspond to the index of highest-energy occupied orbital, from which electrons are allowed to be excited from. E.g., probing the 1s orbital of C in CO can be done by setting ```ncvs = 2```.
 - ```nroots``` (integer): The number of excited states (or transitions) to be calculated. 
 - ```max_cycle``` (integer): The maximum number of iterations in the Davidson diagonalization of the MR-ADC effective Hamiltonian matrix.
 - ```tol_e``` (float): Convergence tolerance for the excitation energies in the Davidson diagonalization.
 - ```tol_davidson``` (float): Convergence tolerance for the residual in the Davidson diagonalization.
 - ```analyze_spec_factor``` (boolean): Request the orbital analysis of intensity contributions for states with the spectroscopic factor greater than ```spec_factor_print_tol```.
 - ```s_thresh_singles``` (float): Parameter for removing linearly dependent single excitations. For experts only.
 - ```s_thresh_doubles``` (float): Parameter for removing linearly dependent double excitations. For experts only.

Additionally, the memory and disk usage can be greatly reduced by approximating the two-electron integrals with density-fitting (DF). 
An example of MR-ADC calculation with density fitting can be found [here](https://github.com/sokolov-group/prism/blob/main/examples/cvs_ip_mr_adc/05-density_fitting.py). 
DF is not used by default but can be invoked using the ```density_fit()``` function call. 
One can overwrite the default choice of the auxiliary basis with a specified one (for example, ```density_fit('cc-pvdz-ri')```. 
More details about setting up calculations with density fitting can be found on the [Pyscf website](https://pyscf.org/user/df.html).
Please note that DF is an approximation, which accuracy depends on the quality of the auxiliary basis set.
Provided that a good auxiliary basis set is used, the DF errors are usually less than 0.01 eV in excitation energy.
We recommend to use the RI- (or RIFIT-) auxiliary basis sets to approximate the integrals in the MR-ADC calculations.
The reference CASSCF calculations can be run either using the exact or density-fitted two-electron integrals approximated using the JKFIT-type auxiliary basis sets.

The excited states with large spectroscopic factors can be visualized by generating the Dyson molecular orbitals (MOs):

```python
from prism.mr_adc_cvs_ip import compute_dyson_mo
from pyscf.tools import molden
dyson_mos = compute_dyson_mo(mr_adc, x)
molden.from_mo(mol, 'mr_adc_dyson_mos.molden', dyson_mos)
```

Here, ```mr_adc_dyson_mos.molden``` is the molden file that can be processed using a variety of orbital visualization software (e.g., JMOL).

# Short list of features:

## CVS-IP-MR-ADC
- Calculation excitation energies up to MR-ADC(2)-X
- Photoelectron transition intensities (spectroscopic factors) up to MR-ADC(2)-X
- Dyson Orbitals:
- Orbital analysis or contributions to spectroscopic factors

# Authors and contributors
[//]: # (To be moved to AUTHORS file)
## Prism Interface
- Carlos E. V. de Moura <carlosevmoura@gmail.com>
- Alexander Yu. Sokolov <alexander.y.sokolov@gmail.com>

## NEVPT2 amplitudes
- Carlos E. V. de Moura <carlosevmoura@gmail.com>
- Alexander Yu. Sokolov <alexander.y.sokolov@gmail.com>

## CVS-IP-MR-ADC method
- Carlos E. V. de Moura <carlosevmoura@gmail.com>
- Alexander Yu. Sokolov <alexander.y.sokolov@gmail.com>

## Contributors to the spin-orbital Prism (pilot implementation)
- Koushik Chatterjee <koushikchatterjee7@gmail.com>
- Ilia Mazin <ilia.mazin@gmail.com>
- Carlos E. V. de Moura <carlosevmoura@gmail.com>
- Alexander Yu. Sokolov <alexander.y.sokolov@gmail.com>

# How to cite
If you include results from Prism in your publication, please cite:
- "[Simulating X-ray photoelectron spectra with strong electron correlation using multireference algebraic diagrammatic construction theory](https://doi.org/10.1039/d1cp05476g)", C.E.V. de Moura and A.Yu. Sokolov, Phys. Chem. Chem. Phys. 24, 4769 – 4784 (2022).
- "[Multi-reference algebraic diagrammatic construction theory for excited states: General formulation and first-order implementation](https://doi.org/10.1063/1.5055380)", A.Yu. Sokolov, J. Chem. Phys. 149(20), 204113 (2018).

