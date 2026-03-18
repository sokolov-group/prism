#!/usr/bin/env python

'''
Basic QD-NEVPT2 calculation for H2O
'''

import numpy as np
import math
import pyscf.gto
import pyscf.scf
import pyscf.mcscf
import prism.interface
import prism.mr_adc
import prism.nevpt

r = 0.96
x = r * math.sin(104.5 * math.pi/(2 * 180.0))
y = r * math.cos(104.5 * math.pi/(2 * 180.0))

mol = pyscf.gto.Mole()
mol.atom = [
            ['O', (0.0, 0.0, 0.0)],
            ['H', (0.0,  -x,   y)],
            ['H', (0.0,   x,   y)]]
mol.basis = 'aug-cc-pvdz'
mol.symmetry = False
mol.verbose = 4
mol.build()

# RHF calculation
mf = pyscf.scf.RHF(mol)
mf.conv_tol = 1e-12

ehf = mf.scf()
print("SCF energy: %f\n" % ehf)

# SA-CASSCF calculation
n_states = 15
weights = np.ones(n_states)/n_states
mc = pyscf.mcscf.CASSCF(mf, 6, 6).state_average_(weights)
mc.conv_tol = 1e-11
mc.conv_tol_grad = 1e-6

emc = mc.mc1step()[0]
mc.analyze()
print("CASSCF energy: %f\n" % emc)

# QD-NEVPT2 with all electrons correlated
interface = prism.interface.PYSCF(mf, mc, opt_einsum = True)
nevpt = prism.nevpt.QDNEVPT(interface)
e_tot, e_corr, osc_str = nevpt.kernel()

## For spectrum
from prism.tools.spectrum import plot
e_diff = (e_tot[1:] - e_tot[0]) * interface.hartree_to_ev
plot(e_diff, osc_str, broadening = 0.1, omega_min = 0, omega_max = 15, plot = True, x_label = "Energy, eV", y_label = "Intensity", title = "UV/vis spectrum", filename = "qdnevpt")


