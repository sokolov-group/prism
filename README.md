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

# Features
[//]: # (To be moved to FEATURES file)
## Core-valence separation for ionization processes MR-ADC (CVS-IP-MR-ADC) method
- Second-order perturbation order: CVS-IP-MR-ADC(2)

## Core-valence separation for absorption processes MR-ADC (CVS-EE-MR-ADC) method
- Second-order perturbation order: CVS-EE-MR-ADC(2)

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

## CVS-EE-MR-ADC Method
- Alexander Yu. Sokolov <alexander.y.sokolov@gmail.com>
- Ilia Mazin <ilia.mazin@gmail.com>
- Carlos E. V. de Moura <carlosevmoura@gmail.com>
- Donna Odhiambo <donna.odhiambo@proton.me>

## Contributors of Spin-Orbital Prism (Pilot code)
- Alexander Yu. Sokolov <alexander.y.sokolov@gmail.com>
- Koushik Chatterjee <koushikchatterjee7@gmail.com>
- Ilia Mazin <ilia.mazin@gmail.com>
- Carlos E. V. de Moura <carlosevmoura@gmail.com>
