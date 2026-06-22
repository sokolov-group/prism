#!/usr/bin/env python

'''
Using different reference wavefunctions to run NEVPT2 for N2
'''

import numpy as np
import pyscf.gto
import pyscf.scf
import pyscf.mcscf
import prism.interface
import prism.mr_adc
import prism.nevpt

mol = pyscf.gto.Mole()
mol.atom = [
            ['O', (0.0, 0.0, 0.0)],
            ['H', (0.0,  -x,   y)],
            ['H', (0.0,   x,   y)]]
mol.basis = 'cc-pvdz'
mol.symmetry = True
mol.spin = 2
mol.build()

# ROHF calculation as guess for CASSCF
mf = pyscf.scf.ROHF(mol)
mf.kernel()

interface = prism.interface.PYSCF(mf, backend = 'opt_einsum')
nevpt = prism.nevpt.NEVPT(interface)
e_tot, e_corr, osc = nevpt.kernel()
