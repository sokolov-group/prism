# Prism
Prism is a Python implementation of electronic structure theories for simulating spectroscopic properties.
Currently, Prism features the methods of N-electron valence perturbation theory (NEVPT) and multireference algebraic diagrammatic construction theory (MR-ADC).
Additionally, capabilities for incorporating relativistic effects and simulating magnetic properties have been implemented.
Prism is being developed as a platform for calculating excited-state energies and spectroscopic observables, with a particular focus on time-resolved electronic spectroscopies.

# How to install
## Requirements
- Python >= 3.7;
- numpy >= 1.13;
- scipy >= 1.3;
- h5py >= 2.7;
- psutil >= 7.0;
- Optional: matplotlib >= 3.9 for plotting spectra;
- Optional: sympy >= 1.12 for spin–orbit coupling;
- Optional: [socutils](https://github.com/xubwa/socutils) for spin–orbit coupling;
- Optional: [opt_einsum](https://optimized-einsum.readthedocs.io/en/stable/) for faster tensor contractions.

## Installation
1) Install [PySCF](https://github.com/pyscf/pyscf/) and make sure it is included in the ``$PYTHONPATH`` environment variable
2) Clone the Prism repository:
```python
git clone https://github.com/sokolov-group/prism.git
```
3) Include the path to the folder where Prism is located in the ```$PYTHONPATH``` environment variable
4) Install optional dependencies if necessary
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
Alternatively, in QD-NEVPT2, the correlation energies and wavefunctions are calculated by diagonalizing the effective Hamiltonian evaluated to second order in perturbation theory.
This allows to incorporate the interaction between the first-order wavefunctions and correctly describe nearly degenerate electronic states (e.g., in the vicinity of avoided crossings).

Some important parameters for the NEVPT calculations are:
 - ```method``` (string): Chooses the NEVPT method. So far, only one level of NEVPT theory is available: ```"nevpt2"```.
 - ```method_type``` (string): Chooses the flavor of NEVPT calculation. Use ```"ss"``` for SS-NEVPT (default) and ```"qd"``` for QD-NEVPT.
 - ```nfrozen``` (integer): Number of lowest-energy (core) molecular orbitals that will be left uncorrelated ("frozen core"). Default is 0 or None.
 - ```max_memory``` (integer): Controls how much memory (in MB) will be used in a calculation. Prism **loves** memory. Allowing the calculation to use more memory tends to speed it up since less input/output operations on disk are performed. Note that this parameter is just an estimate and the calculation can use more memory than allowed. For large jobs, it is recommended to run each calculation on a dedicated computer node to prevent memory errors. Default is set by PySCF.
 - ```rdm_order``` (integer): Paramater to set the order of the one-particle density matrix (1-RDM) used to evaluate one-particle properties (e.g., oscillator strengths or natural transition orbitals). 0 = reference (default), 2 = includes NEVPT2/QD-NEVPT2 correlation.
 - ```compute_singles_amplitudes``` (boolean): Whether to compute single excitation amplitudes. If False (default), singles are not computed as in the standard NEVPT2 calculation. Switching to True has a very small effect on the NEVPT2 energy since the semi-internal double excitations capture the effect of singles when this option is set to False. Default is False. For experts only.
 - ```s_thresh_singles``` (float): Parameter for removing linearly dependent single and semi-internal double excitations. Default is 1e-8. For experts only. 
 - ```s_thresh_doubles``` (float): Parameter for removing linearly dependent (external) double excitations. Default is 1e-8. For experts only.

The natural transition orbitals for any multistate NEVPT calculation can be produced by calling:
```python
nevpt.analyze()
```

The following options control the analysis output:
- `keep_amplitudes` (bool, optional): **Must be set to `True` for analysis to work.**
- `compute_ntos` (bool, optional): If `True`, compute and write natural transition orbitals for ground to excited state transitions to a Molden file. Default is `False`.

Alternatively, natural transition orbitals for any transition can be computed directly given the transition density matrix between the states:
```python
from prism.tools import trans_prop
trdm = nevpt.make_rdm1(L=1, R=2)
w, U, Vh = trans_prop.compute_ntos(interface, trdm, initial_state=1, target_state=2)
```

The resulting `_nto_S1_S2.molden` file can be visualized using orbital visualization software such as [JMOL](http://jmol.sourceforge.net/).

## Multireference algebraic diagrammatic construction theory
Multireference algebraic diagrammatic construction theory can simulate a variety of excited electronic states (neutral excitations, ionization, electron attachment, core excitation and ionization).
The type of excited states is controled by the ```method_type``` parameter of MR-ADC class.
Currently, the only excited states that can be computed using MR-ADC in Prism are core-ionized states probed in photoelectron spectroscopy.
These excitations are simulated by introducing core-valence separation approximation (CVS) and the resulting method is abbreviated as CVS-IP-MR-ADC.

The CVS-IP-MR-ADC calculations can be performed at four different levels of theory that are specified using the ```method``` parameter: ```"mr-adc(0)"```, ```"mr-adc(1)"```, ```"mr-adc(2)"```, ```"mr-adc(2)-x"```.

Other important parameters are:
 - ```ncvs``` (integer): The number of core orbitals to be included in the simulation. This number should ideally correspond to the index of highest-energy occupied orbital, from which electrons are allowed to be excited from. E.g., probing the 1s orbital of C in CO can be done by setting ```ncvs = 2```.
 - ```nroots``` (integer): The number of excited states (or transitions) to be calculated. Default is 6.
 - ```max_cycle``` (integer): The maximum number of iterations in the Davidson diagonalization of the MR-ADC effective Hamiltonian matrix. Default is 50.
 - ```tol_e``` (float): Convergence tolerance for the excitation energies in the Davidson diagonalization (in Hartree). Default is 1e-8.
 - ```tol_r``` (float): Convergence tolerance for the residual in the Davidson diagonalization. Default is 1e-5.
 - ```max_memory``` (integer): Controls how much memory (in MB) will be used in a calculation. Prism **loves** memory. Allowing the calculation to use more memory tends to speed up the calculation since less input/output operations on disk are performed. Note that this parameter is just an estimate and the calculation can use more memory than allowed. For large jobs, it is recommended to run each calculation on a dedicated computer node to prevent memory errors. Default is set by PySCF.
 - ```s_thresh_singles``` (float): Parameter for removing linearly dependent single and semi-internal double excitations. Default is 1e-5. For experts only.
 - ```s_thresh_doubles``` (float): Parameter for removing linearly dependent (external) double excitations. Default is 1e-10. For experts only.

The CVS-IP-MR-ADC spectroscopic intensities (so-called spectroscopic factors) and their orbital contributions can be analyzed by calling:
```python
mr_adc.analyze()
```

The following options control the analysis output:
- `spec_factor_print_tol` (float, optional): Threshold for printing spectroscopic factor contributions. Default is 1e-1.
- `compute_dyson` (bool, optional): If `True`, compute and write Dyson orbitals to a Molden file. Default is `False`.

Alternatively, Dyson orbitals can be computed directly:
```python
from prism.tools import trans_prop
dyson_mo = trans_prop.compute_dyson(interface, x)
```

The resulting `_dyson.molden` file can be visualized using orbital visualization software such as [JMOL](http://jmol.sourceforge.net/).

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

## Spin-orbit coupling
The spin-orbit coupling (SOC) is avaliable in NEVPT2 and QD-NEVPT2. To run SOC code, [socutils](https://github.com/xubwa/socutils) is required and can be installed by using: 

```python
git submodule update --init --recursive
```

The SOC calculation can be performed by setting the ```soc``` attribute:

```python
interface = prism.interface.PYSCF(mf, mc, opt_einsum = True)
nevpt = prism.nevpt.NEVPT(interface)
nevpt.soc = "BP"
nevpt.kernel()
```

The SOC calculations can be performed for two types of SOC Hamiltionian that are specified using the ```soc``` parameter: ```"BP"``` (Breit-Pauli), ```"DKH1"``` (exact two-component Douglas–Kroll–Hess).

The g-tensor calculation can be performed after SOC calculation by setting ```gtensor``` to True.

Other parameters for g-tensor calculation are:
- ```origin_type``` (string): The origin of coordinate system setting. Default is ```"charge"```, which indicates setting origin point at the center of nuclear charge. The other possible choices are ```"GIAO"```(using gauge-including atomic orbital), ```"atom1"``` (using the first atom position). Also, origin can be set to a particular point by providing a list of three coordinates (in Bohr).
 - ```target_state``` (integer or list): target state to calculate g-tensor. Default is 1 (lowest-energy state). The code will detect spin multiplicity and will calculate g-tensor for the target state. Users can also assign a set of (nearly) degenerate states to calculate g-tensor by providing a list. For example, to compute g-tensor for a doubly degenerate first excited state set ```target_state = [2,3]```.

# Short summary of features:

## NEVPT2
- Full internal contraction (equivalent to partially contracted NEVPT2)
- Single- and multi-state state-specific NEVPT2 energies
- Oscillator strengths
- Frozen core approximation
- Full support of density fitting
- One-particle reduced density matrix (1-RDM) with second-order correlation contributions
- State-interaction spin–orbit coupling with Breit–Pauli and exact two-component Douglas–Kroll–Hess Hamiltonians
- Molecular g-tensors via Kramers approach and state-interaction spin–orbit coupling

## QD-NEVPT2
- Full internal contraction (equivalent to partially contracted QD-NEVPT2)
- Single- and multi-state state-specific QD-NEVPT2 energies
- Oscillator strengths
- Frozen core approximation
- Full support of density fitting
- One-particle reduced density matrix (1-RDM) with second-order correlation contributions
- State-interaction spin–orbit coupling with Breit–Pauli and exact two-component Douglas–Kroll–Hess Hamiltonians
- Molecular g-tensors via Kramers approach and state-interaction spin–orbit coupling

## CVS-IP-MR-ADC
- Excitation energies up to MR-ADC(2)-X
- Photoelectron transition probabilities (so-called spectroscopic factors) up to MR-ADC(2)-X
- Dyson orbitals
- Orbital analysis of contributions to spectroscopic factors
- Full support of density fitting

## General capabilities
- Incorporating spin–orbit coupling via state-interaction approach with Breit–Pauli and exact two-component Douglas–Kroll–Hess Hamiltonians
- Calculating magnetic g-tensors via state-interaction approach with Breit–Pauli and exact two-component Douglas–Kroll–Hess Hamiltonians
- Natural transition orbitals
- Dyson orbitals
- Plotting and visualizing spectra

# How to cite
If you include MR-ADC results from Prism in your publication, please cite:
- "[Efficient Spin-Adapted Implementation of Multireference Algebraic Diagrammatic Construction Theory. I. Core-Ionized States and X‑ray Photoelectron Spectra](https://doi.org/10.1021/acs.jpca.4c03161)", C.E.V. de Moura, and A.Yu. Sokolov, J. Phys. Chem. A 128, 5816–5831 (2024).

Citation for the NEVPT2 method:
- "[Introduction of n-electron valence states for multireference perturbation theory](https://doi.org/10.1063/1.1361246)", C. Angeli, R. Cimiraglia, S. Evangelisti, T. Leininger, and J.-P.P. Malrieu, J. Chem. Phys. 114(23), 10252–10264 (2001).

Citation for the QD-NEVPT2 method:
- "[A quasidegenerate formulation of the second order n-electron valence state perturbation theory approach](https://doi.org/10.1063/1.1778711)", C. Angeli, S. Borini, M. Cestari, and R. Cimiraglia, J. Chem. Phys. 121(9), 4043–4049 (2004).

Additional references for the MR-ADC and NEVPT methods:
- "[Multi-reference algebraic diagrammatic construction theory for excited states: General formulation and first-order implementation](https://doi.org/10.1063/1.5055380)", A.Yu. Sokolov, J. Chem. Phys. 149(20), 204113 (2018).
- "[Simulating X-ray photoelectron spectra with strong electron correlation using multireference algebraic diagrammatic construction theory](https://doi.org/10.1039/d1cp05476g)", C.E.V. de Moura and A.Yu. Sokolov, Phys. Chem. Chem. Phys. 24, 4769 – 4784 (2022).
- "[Algebraic Diagrammatic Construction Theory for Simulating Charged Excited States and Photoelectron Spectra](https://doi.org/10.1021/acs.jctc.3c00251)", S. Banerjee, and A.Yu. Sokolov, J. Chem. Theory Comput. 19(11), 3037–3053 (2023).
- "[Multireference perturbation theories based on the Dyall Hamiltonian](https://doi.org/10.1016/bs.aiq.2024.04.004)", A.Yu. Sokolov, Adv. Quantum Chem. 90, 121–155 (2024).

# Authors and significant contributors:
List of significant contributions, in alphabetical order:
- Nicholas Y. Chiang
- Carlos E. V. de Moura
- Nishshanka M. Lakshan
- Rajat S. Majumder
- Ilia M. Mazin
- Donna H. Odhiambo
- Bryce Pickett
- James D. Serna
- Alexander Yu. Sokolov

Check out AUTHORS for all contributions.
