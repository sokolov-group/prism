#!/usr/bin/env python

'''
Using different SCF reference wavefunctions to run NEVPT2
'''

import pyscf.gto
import pyscf.scf
import prism.interface
import prism.mr_adc
import prism.nevpt

# Create Mole for NO2 radical
x = 1.0989
y = 0.4653
mol = pyscf.gto.Mole()
mol.atom = [
            ['N', (0.0, 0.0, 0.0)],
            ['O', (0.0,  -x,   y)],
            ['O', (0.0,   x,   y)]]
mol.basis = 'cc-pvdz'
mol.symmetry = True
mol.spin = 1
mol.charge = 0
mol.verbose = 4
mol.build()

# ROHF calculation
mf_radical = pyscf.scf.ROHF(mol)
mf_radical.kernel()

# ROHF Reference -> minimal active space CASCI -> NEVPT2 Calculation
interface = prism.interface.PYSCF(mf_radical, backend = 'opt_einsum')
nevpt = prism.nevpt.NEVPT(interface)
e_tot, e_corr, osc = nevpt.kernel()

# Create Mole for NO2 anion
anion_mol = mol.copy(deep=True)
anion_mol.spin = 0
anion_mol.charge = -1
anion_mol.build()

# RHF calculation
mf_anion = pyscf.scf.RHF(anion_mol)
mf_anion.kernel()

# RHF Reference -> MP2 Calculation
interface = prism.interface.PYSCF(mf_anion, backend = 'opt_einsum')
nevpt = prism.nevpt.NEVPT(interface)
e_tot, e_corr, osc = nevpt.kernel()

