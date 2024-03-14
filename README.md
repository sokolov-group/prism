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
In the Python3 interpreter or in a Python3 script, import the Prism interface using:

```python
import prism.interface
```

Detailed examples of the implemented methods can be found [here](examples/).
An extensive and comprehensive manual of Prism will be released soon.

## Available Methods
- [Multireference Algebraic Diagrammatic Construction (MR-ADC) Theory](#multireference-algebraic-diagrammatic-construction-(mr-adc)-theory)
- [Density Fitting](#density-fitting-(df))
- [Dyson Molecular Orbitals](#dyson-molecular-orbitals)

## Multireference Algebraic Diagrammatic Construction (MR-ADC) Theory
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

After running a MR-ADC calculation, one can generate the Dyson molecular orbitals (MOs):

```python
from prism.mr_adc_cvs_ip import compute_dyson_mo
from pyscf.tools import molden
dyson_mos = compute_dyson_mo(mr_adc, x)
molden.from_mo(mol, 'mr_adc_dyson_mos.molden', dyson_mos)
```

# Features
[//]: # (To be moved to FEATURES file)
## Core-valence separation for ionization processes MR-ADC (CVS-IP-MR-ADC) method
- Second-order perturbation order: CVS-IP-MR-ADC(2)

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

## Contributors of Spin-Orbital Prism (Pilot code)
- Alexander Yu. Sokolov <alexander.y.sokolov@gmail.com>
- Koushik Chatterjee <koushikchatterjee7@gmail.com>
- Ilia Mazin <ilia.mazin@gmail.com>
- Carlos E. V. de Moura <carlosevmoura@gmail.com>
