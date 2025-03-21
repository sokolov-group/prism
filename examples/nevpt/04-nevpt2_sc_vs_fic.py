#!/usr/bin/env python

'''
Comparing the results of FIC-NEVPT2 and SC-NEVPT2 for N2
'''

import numpy as np
import pyscf.gto
import pyscf.scf
import pyscf.mcscf
import pyscf.mrpt
import prism.interface
import prism.mr_adc
import prism.nevpt

mol = pyscf.gto.Mole()
r = 1.098
mol.atom = [
    ['N', ( 0., 0.    , -r/2)],
    ['N', ( 0., 0.    ,  r/2)],]
mol.basis = {'N':'aug-cc-pvdz'}
mol.verbose = 4
mol.build()

# RHF calculation as guess for CASSCF
mf = pyscf.scf.RHF(mol)
mf.kernel()

# SA-CASSCF reference
n_states = 9
weights = np.ones(n_states)/n_states
mc = pyscf.mcscf.CASSCF(mf, 6, 6).state_average_(weights)
emc = mc.mc1step()[0]

# FIC-NEVPT2 calculation using Prism
interface = prism.interface.PYSCF(mf, mc, opt_einsum = True)
nevpt = prism.nevpt.NEVPT(interface)
nevpt.compute_singles_amplitudes = False
nevpt.s_thresh_singles = 1e-10
nevpt.s_thresh_doubles = 1e-10
nevpt.method = "nevpt2"
e_tot, e_corr = nevpt.kernel()

# SC-NEVPT2 calculation using PySCF
mo = mc.mo_coeff.copy()
weights = np.ones(n_states)/n_states
mc = pyscf.mcscf.CASCI(mf, 6, 6)
mc.fcisolver.nroots = n_states
emc = mc.casci(mo)[0]

energies = []
for state in range(n_states):
    e_corr = pyscf.mrpt.NEVPT(mc,root=state).kernel()
    e_tot = mc.e_tot[state] + e_corr
    energies.append(e_tot)
    print('Total sc-NEVPT2 energy:', e_tot)

print('Summary of the sc-NEVPT2 calculation:')
for state in range(len(energies)):
    de = energies[state] - energies[0]
    print("%5d   %20.12f %14.8f %12.4f" % (state+1, energies[state], de, de*interface.hartree_to_ev))
