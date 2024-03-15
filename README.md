# Prism
Python-based implementation of electronic structure theories for simulating spectroscopic properties

2023-04-01

- [Installation](#installation)
  - [Dependencies](#dependencies)
- [How to use](#how-to-use)
- [Features](#features)
- [Authors and Contributors](#authors-and-contributors)

# Installation
Copy the actual Prism repository using ```git```:

```python
git clone https://github.com/sokolov-group/prism.git
```

To use it, be sure that your ```$PYTHONPATH``` variable includes this directory.

## Dependencies
- [PySCF quantum-chemistry framework](https://github.com/pyscf/pyscf/);
- [NumPy scientific computation package](https://numpy.org/);

# How to use

Prism is dependent on PySCF, and MR-ADC calculations are based on both SCF and MCSCF calculations. Thus to run an MR-ADC calculation in the Python3 interpreter or in a Python3 script, the following will need to be imported:

```python
from pyscf import gto, scf, mcscf
import prism.interface
import prism.mr_adc
```

Then, as explained in the [PySCF user guide](https://pyscf.org/user.html), a mole object needs to be created along with running both an SCF and MCSCF calculation:

```python
mol = gto.M(atom = 'H 0 0 0; F 0 0 0.91', basis = 'cc-pvdz')
mf = scf.RHF(mol).run()
mc = mcscf.CASSCF(mf, 6, 6).run()
```

Finally the prism calculation can be done. First the interface and object for the calculation are created, followed by customizing the claculation. Then the actual calculation can be run with the ```kernel()``` function. For example, a CVS-IP-MR-ADC calculation can be run with the general format as follows:

```python
interface = prism.interface.PYSCF(mf, mc, pot_einsum = True)
mr_adc = prism.mr_adc.MRADC(interface)
mr_adc.method = "mr-adc(2)"
mr_adc.method_type = "cvs-ip"
mr_adc.ncvs = 1
e, p, x = mr_adc.kernel()
```

Detailed examples of the implemented methods can be found [here](examples/).
An extensive and comprehensive manual of Prism will be released soon.



[//]: # (Methods and algorithms) - Nick


[//]: # (Features and capabilities) - Alex






# OLD:

# Available Methods
[//]: # (Description of method types, levels of theory)

- [Multireference Algebraic Diagrammatic Construction (MR-ADC) Theory](#multireference-algebraic-diagrammatic-construction-mr-adc-theory)
- [Density Fitting](#density-fitting-df)
- [Dyson Molecular Orbitals](#dyson-molecular-orbitals)

An example of a core-valence separation for ionization processes (CVS-IP) computation using the default MR-ADC method (MR-ADC(2)) is shown below:

```python
from pyscf import gto, scf, mcscf
import prism.interface
import prism.mr_adc
mol = gto.M(atom = 'H 0 0 0; F 0 0 0.91', basis = 'cc-pvdz')
mf = scf.RHF(mol).run()
mc = mcscf.CASSCF(mf, 6, 6).run()
interface = prism.interface.PYSCF(mf, mc, pot_einsum = True)
mr_adc = prism.mr_adc.MRADC(interface)
mr_adc.method_type = "cvs-ip"
mr_adc.ncvs = 1
e, p, x = mr_adc.kernel()
```

The MR-ADC calculation is based on both the SCF and MCSCF calculations. The number of core orbitals that are being probed in the calculation is controlled by ```ncvs```.

One can specify the order and the number of roots of the desired MR-ADC computation:

```python
mr_adc.method = "mr-adc(2)-x"
mr_adc.nroots = 8
```

Additionally, the memory and disk usage can be greatly reduced by approximating the two-electron integrals with density-fitting. One can see an example of density fitting in a MR-ADC calculation in [this example](https://github.com/sokolov-group/prism/blob/main/examples/cvs_ip_mr_adc/05-density_fitting.py).

## Density Fitting (DF)
[//]: # (Description of integrals to be used)

DF is not used by default but can be invoked via the ```density_fit()``` method. Here is an example of using DF in a SCF and MCSCF calculation:

```python
from pyscf import gto, scf, mcscf
mol = gto.M(atom = 'H 0 0 0; F 0 0 0.91, basis = 'def2-tzvp')
mf = scf.RHF(mol).density_fit().run()
mc = mcscf.CASSCF(mf, 6, 6).density_fit().run()
```

One can overwrite the default choice of the auxiliary basis:

```python
from pyscf import gto, scf, mcscf
import prism.interface
import prism.mr_adc
mol = gto.M(atom = 'H 0 0 0; F 0 0 0.91', basis = 'cc-pvdz')
mf = scf.RHF(mol).run()
mc = mcscf.CASSCF(mf, 6, 6).density_fit('cc-pvdz-jkfit').run()
interface = prism.interface.PYSCF(mf, mc, pot_einsum = True)
mr_adc = prism.mr_adc.MRADC(interface).density_fit('cc-pvdz-ri')
mr_adc.method_type = "cvs-ip"
mr_adc.ncvs = 1
e, p, x = mr_adc.kernel()
```

## Dyson Molecular Orbitals
[//]: # (Capabilities / Features)

After running a MR-ADC calculation, one can generate the Dyson molecular orbitals (MOs):

```python
from prism.mr_adc_cvs_ip import compute_dyson_mo
from pyscf.tools import molden
dyson_mos = compute_dyson_mo(mr_adc, x)
molden.from_mo(mol, 'mr_adc_dyson_mos.molden', dyson_mos)
```

# Authors and Contributors
[//]: # (To be moved to AUTHORS file)
## Prism Interface
- Alexander Yu. Sokolov <alexander.y.sokolov@gmail.com>
- Carlos E. V. de Moura <carlosevmoura@gmail.com>

## pc-NEVPT2 Amplitudes
- Alexander Yu. Sokolov <alexander.y.sokolov@gmail.com>
- Carlos E. V. de Moura <carlosevmoura@gmail.com>

## CVS-IP-MR-ADC Method
- Alexander Yu. Sokolov <alexander.y.sokolov@gmail.com>
- Carlos E. V. de Moura <carlosevmoura@gmail.com>

## Contributors to the spin-orbital Prism (pilot implementation)
- Alexander Yu. Sokolov <alexander.y.sokolov@gmail.com>
- Koushik Chatterjee <koushikchatterjee7@gmail.com>
- Ilia Mazin <ilia.mazin@gmail.com>
- Carlos E. V. de Moura <carlosevmoura@gmail.com>

# How to cite
If you include results from Prism in your publication, please cite:
- "[Simulating X-ray photoelectron spectra with strong electron correlation using multireference algebraic diagrammatic construction theory](https://doi.org/10.1039/d1cp05476g)", C.E.V. de Moura and A.Yu. Sokolov, Phys. Chem. Chem. Phys. 24, 4769 – 4784 (2022).
- "[Multi-reference algebraic diagrammatic construction theory for excited states: General formulation and first-order implementation](https://doi.org/10.1063/1.5055380)", A.Yu. Sokolov, J. Chem. Phys. 149(20), 204113 (2018).

