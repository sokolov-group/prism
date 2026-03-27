#!/usr/bin/env python

'''
NTOs for MS-CASCI/NEVPT2 of H2O
'''

import numpy as np
import math
import pyscf.gto
import pyscf.scf
import pyscf.mcscf
import prism.interface
import prism.nevpt

np.set_printoptions(linewidth=150, edgeitems=10, suppress=True)

r = 0.96
x = r * math.sin(104.5 * math.pi/(2 * 180.0))
y = r * math.cos(104.5 * math.pi/(2 * 180.0))

mol = pyscf.gto.Mole()
mol.atom = [
            ['O', (0.0, 0.0, 0.0)],
            ['H', (0.0,  -x,   y)],
            ['H', (0.0,   x,   y)]]
mol.basis = 'aug-cc-pvdz'
mol.symmetry = True
mol.build()

# RHF calculation
mf = pyscf.scf.RHF(mol)
mf.conv_tol = 1e-12

ehf = mf.scf()
print("SCF energy: %f\n" % ehf)

# MS-CASCI calculation
n_states = 9
mc = pyscf.mcscf.CASCI(mf, 6, 6)
mc.fcisolver.nroots = n_states
emc = mc.casci()[0]

# NEVPT2 calculation
interface = prism.interface.PYSCF(mf, mc, backend = 'opt_einsum')
nevpt = prism.nevpt.NEVPT(interface)
nevpt.compute_singles_amplitudes = False

# Enable ground to excited state NTOs and retain amplitudes for analysis
nevpt.compute_ntos = True
nevpt.keep_amplitudes = True

e_tot, e_corr, osc = nevpt.kernel()

# Compute and write NTOs
nevpt.analyze()

# Compute and write NTOs for a specific transition (S1 -> S2)
from prism.tools import trans_prop
trdm = nevpt.make_rdm1(L=1, R=2)
trans_prop.compute_ntos(interface, trdm, initial_state=1, target_state=2)