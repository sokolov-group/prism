# Prism
Prism is a Python implementation of electronic structure theories for simulating spectroscopic properties.
Currently, Prism features the methods of N-electron valence perturbation theory (NEVPT) and multireference algebraic diagrammatic construction theory (MR-ADC).

# How to install
## Requirements
- Python 3.7 or older, including its dependencies;
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
import prism.mr_adc # For MR-ADC calculations
import prism.nevpt  # For NEVPT calculations
```

Next, as described in the [PySCF user guide](https://pyscf.org/user/index.html), specify molecular geometry, then run reference Hartree-Fock and complete active space self-consistent field (CASSCF) calculations.
Below is an example of reference CASSCF calculation for the hydrogen fluoride (HF) molecule with the cc-pvdz basis set and 6 electrons in 6 orbitals (6e, 6o) active space.

```python
mol = gto.M(atom = 'H 0 0 0; F 0 0 0.91', basis = 'cc-pvdz')
mf = scf.RHF(mol).run()
mc = mcscf.CASSCF(mf, 6, 6).run()
```

Once the reference calculation is successfully completed, the objects of Hartree-Fock and CASSCF classes (```mf``` and ```mc```) are passed to Prism. 
For example, the NEVPT2 energy calculation for the reference CASSCF state can be performed as follows:

```python
interface = prism.interface.PYSCF(mf, mc, opt_einsum = True)
nevpt = prism.nevpt.NEVPT(interface)
nevpt.method = "nevpt2"
e_tot, e_corr, osc = nevpt.kernel()
```

Alternatively, the CVS-IP-MR-ADC calculation of core ionized states can be performed as:

```python
interface = prism.interface.PYSCF(mf, mc, opt_einsum = True)
mr_adc = prism.mr_adc.MRADC(interface)
mr_adc.method = "mr-adc(2)"
mr_adc.method_type = "cvs-ip"
mr_adc.nroots = 10
mr_adc.ncvs = 1
e, p, x = mr_adc.kernel()
```

This calculation uses CVS-IP-MR-ADC(2) to compute 10 core ionized states ("roots").
The parameter ```ncvs``` controls the number of core orbitals in the hydrogen fluoride molecule, for which excited states are calculated.
For example, setting ```ncvs = 1``` corresponds to exciting electrons from the 1s orbitals of fluorine atoms, while ```ncvs = 2``` corresponds to probing the 2s excitations.
Other examples can be found [here](examples/).

# Methods and algorithms
## N-electron valence perturbation theory
N-electron valence perturbation theory (NEVPT) is an efficient multireference approach to describe dynamic electron correlation starting with a complete active space configuration interaction (CASCI) or self-consistent field (CASSCF) reference wavefunction.
Prism features an implementation of second-order NEVPT (NEVPT2) with full internal contraction (FIC), which is also known as "partially contracted" NEVPT2 (PC-NEVPT2).
The NEVPT2 calculations can be performed starting with one or several CASCI/CASSCF reference wavefunctions with any choice of orbitals (e.g., Hartree–Fock, state-averaged CASSCF, etc).

For multiple reference states, two flavors of NEVPT2 are available: 1) state-specific (SS-NEVPT2) and 2) quasidegenerate (QD-NEVPT2).
In the SS-NEVPT2 method, the first-order wavefunctions and second-order correlation energies are computed for each electronic state.
Alternatively, in QD-NEVPT2, the correlation energies and wavefunctions are calculated by diagonalizing the effective Hamiltonian evaluated to second order perturbation theory.
This allows to incorporate the interaction between the first-order wavefunctions and correctly describe nearly degenerate electronic states (e.g., in the vicinity of avoided crossings).

Some important parameters for the NEVPT2 calculations are:
 - ```method``` (string): Controls the flavor of NEVPT2 calculation. Use ```"nevpt2"``` for SS-NEVPT2 and ``"qd-nevpt2"``` for QD-NEVPT2.
 - ```nfrozen``` (integer): Number of lowest-energy (core) molecular orbitals that will be left uncorrelated ("frozen core").
 - ```max_memory``` (integer): Controls how much memory (in MB) will be used in a calculation. Prism **loves** memory. Allowing the calculation to use more memory tends to speed it up since less input/output operations on disk are performed. Note that this parameter is just an estimate and the calculation can use more memory than allowed. For large jobs, it is recommended to run each calculation on a dedicated computer node to prevent memory errors.
 - ```compute_singles_amplitudes``` (boolean): Whether to compute single excitation amplitudes. If False (default), singles are not computed as in the standard NEVPT2 calculation. Switching to True has a very small effect on the NEVPT2 energy since the semi-internal double excitations capture the effect of singles when this option is set to False. For experts only.
 - ```s_thresh_singles``` (float): Parameter for removing linearly dependent single and semi-internal double excitations. For experts only.
 - ```s_thresh_doubles``` (float): Parameter for removing linearly dependent (external) double excitations. For experts only.

## Multireference algebraic diagrammatic construction theory
Multireference algebraic diagrammatic construction theory can simulate a variety of excited electronic states (neutral excitations, ionization, electron attachment, core excitation and ionization).
The type of excited states is controled by the ```method_type``` parameter of MR-ADC class.
Currently, the only excited states that can be computed using MR-ADC in Prism are core-ionized states probed in photoelectron spectroscopy.
These excitations are simulated by introducing core-valence separation approximation (CVS) and the resulting method is abbreviated as CVS-IP-MR-ADC.

The CVS-IP-MR-ADC calculations can be performed at four different levels of theory that are specified using the ```method``` parameter: ```"mr-adc(0)"```, ```"mr-adc(1)"```, ```"mr-adc(2)"```, ```"mr-adc(2)-x"```.

Other important parameters are:
 - ```ncvs``` (integer): The number of core orbitals to be included in the simulation. This number should ideally correspond to the index of highest-energy occupied orbital, from which electrons are allowed to be excited from. E.g., probing the 1s orbital of C in CO can be done by setting ```ncvs = 2```.
 - ```nroots``` (integer): The number of excited states (or transitions) to be calculated. 
 - ```max_cycle``` (integer): The maximum number of iterations in the Davidson diagonalization of the MR-ADC effective Hamiltonian matrix.
 - ```tol_e``` (float): Convergence tolerance for the excitation energies in the Davidson diagonalization.
 - ```tol_r``` (float): Convergence tolerance for the residual in the Davidson diagonalization.
 - ```max_memory``` (integer): Controls how much memory (in MB) will be used in a calculation. Prism **loves** memory. Allowing the calculation to use more memory tends to speed up the calculation since less input/output operations on disk are performed. Note that this parameter is just an estimate and the calculation can use more memory than allowed. For large jobs, it is recommended to run each calculation on a dedicated computer node to prevent memory errors.
 - ```analyze_spec_factor``` (boolean): Requests the orbital analysis of intensity contributions for states with the spectroscopic factor greater than ```spec_factor_print_tol``` (float).
 - ```s_thresh_singles``` (float): Parameter for removing linearly dependent single and semi-internal double excitations. For experts only.
 - ```s_thresh_doubles``` (float): Parameter for removing linearly dependent (external) double excitations. For experts only.

The excited states with large spectroscopic factors can be visualized by generating the Dyson molecular orbitals:

```python
from prism.mr_adc_cvs_ip import compute_dyson_mo
from pyscf.tools import molden
dyson_mos = compute_dyson_mo(mr_adc, x)
molden.from_mo(mol, 'mr_adc_dyson_mos.molden', dyson_mos)
```

Here, ```mr_adc_dyson_mos.molden``` is the molden file that can be processed using a variety of orbital visualization software (e.g., JMOL).

## Density fitting
The memory and disk usage of NEVPT and MR-ADC calculations can be greatly reduced by approximating the two-electron integrals with density fitting (DF). 
An example of using density fitting can be found [here](examples/nevpt/03-nevpt2-density-fitting.py) and [here](examples/mr_adc/05-density_fitting.py). 
DF is not used by default but can be invoked using the ```density_fit()``` function call. 
One can overwrite the default auxiliary basis with a specified one (for example, ```density_fit('cc-pvdz-ri')```).
More details about setting up calculations with density fitting can be found on the [Pyscf website](https://pyscf.org/user/df.html).
Please note that DF is an approximation, which accuracy depends on the quality of the auxiliary basis set.
Provided that a good auxiliary basis set is used, the DF errors are usually less than 0.01 eV in excitation energy.
We recommend to use the RI- (or RIFIT-) auxiliary basis sets to approximate the integrals in the NEVPT and MR-ADC calculations.
The reference CASSCF calculations can be run either using the exact or density-fitted two-electron integrals approximated using the JKFIT-type auxiliary basis sets.
Note that DF can significantly speed up the CASSCF calculation since the cost of integral transformation at every iteration is reduced.

# Short summary of features:

## NEVPT2
- Full internal contraction (equivalent to partially contracted NEVPT2)
- Single- and multi-state state-specific NEVPT2 energies
- Frozen core approximation
- Full support of density fitting
- Oscillator strengths for multi-state NEVPT2 calculations

## QD-NEVPT2
- Full internal contraction (equivalent to partially contracted QD-NEVPT2)
- Frozen core approximation
- Full support of density fitting
- Oscillator strengths

## CVS-IP-MR-ADC
- Excitation energies up to MR-ADC(2)-X
- Photoelectron transition probabilities (so-called spectroscopic factors) up to MR-ADC(2)-X
- Dyson orbitals
- Orbital analysis of contributions to spectroscopic factors
- Full support of density fitting

# How to cite
If you include MR-ADC results from Prism in your publication, please cite:
- "[Efficient Spin-Adapted Implementation of Multireference Algebraic Diagrammatic Construction Theory. I. Core-Ionized States and X‑ray Photoelectron Spectra](https://doi.org/10.1021/acs.jpca.4c03161)", C.E.V. de Moura, and A.Yu. Sokolov, J. Phys. Chem. A 128, 5816–5831 (2024).

Citation for the NEVPT2 method:
- "[Introduction of n-electron valence states for multireference perturbation theory](https://doi.org/10.1063/1.1361246)", C. Angeli, R. Cimiraglia, S. Evangelisti, T. Leininger, and J.-P.P. Malrieu, J. Chem. Phys. 114(23), 10252–10264 (2001).

Citation for the QD-NEVPT2 method:
- "[A quasidegenerate formulation of the second order n-electron valence state perturbation theory approach](https://doi.org/10.1063/1.1778711)", C. Angeli, S. Borini, M. Cestari, and R. Cimiraglia, J. Chem. Phys. 121(9), 4043–4049 (2004).

Additional references for the MR-ADC methods:
- "[Multi-reference algebraic diagrammatic construction theory for excited states: General formulation and first-order implementation](https://doi.org/10.1063/1.5055380)", A.Yu. Sokolov, J. Chem. Phys. 149(20), 204113 (2018).
- "[Simulating X-ray photoelectron spectra with strong electron correlation using multireference algebraic diagrammatic construction theory](https://doi.org/10.1039/d1cp05476g)", C.E.V. de Moura and A.Yu. Sokolov, Phys. Chem. Chem. Phys. 24, 4769 – 4784 (2022).
- "[Algebraic Diagrammatic Construction Theory for Simulating Charged Excited States and Photoelectron Spectra](https://doi.org/10.1021/acs.jctc.3c00251)", S. Banerjee, and A.Yu. Sokolov, J. Chem. Theory Comput. 19(11), 3037–3053 (2023).

# Authors and significant contributors:

- Carlos E. V. de Moura <carlosevmoura@gmail.com>
- Alexander Yu. Sokolov <alexander.y.sokolov@gmail.com>
- James D. Serna <jserna456@gmail.com>

Check out AUTHORS for more details.
