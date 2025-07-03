#!/usr/bin/env python

'''
Expert options for QD-NEVPT2
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
#mc = pyscf.mcscf.CASSCF(mf, 6, 6).state_average_(weights).density_fit('aug-cc-pvdz-ri')
mc = pyscf.mcscf.CASSCF(mf, 6, 6).state_average_(weights)
emc = mc.mc1step()[0]

# Increasing the truncation parameter for linear dependencies
interface = prism.interface.PYSCF(mf, mc, opt_einsum = True)
nevpt = prism.nevpt.NEVPT(interface)
nevpt.compute_singles_amplitudes = False
nevpt.s_thresh_singles = 1e-6
nevpt.s_thresh_doubles = 1e-6
nevpt.method = "qd-nevpt2"
e_tot, e_cor, oscr = nevpt.kernel()

# Including the singles amplitudes in the QD-NEVPT2 energy
interface = prism.interface.PYSCF(mf, mc, opt_einsum = True)
nevpt = prism.nevpt.NEVPT(interface)
nevpt.compute_singles_amplitudes = True
nevpt.s_thresh_singles = 1e-8
nevpt.s_thresh_doubles = 1e-8
nevpt.method = "qd-nevpt2"
e_tot, e_corr, osc = nevpt.kernel()

# Selecting reference states
ref_list = [1,5,6]
interface = prism.interface.PYSCF(mf, mc, opt_einsum = True, select_reference = ref_list)
nevpt = prism.nevpt.NEVPT(interface)
nevpt.compute_singles_amplitudes = False
nevpt.s_thresh_singles = 1e-8
nevpt.s_thresh_doubles = 1e-8
nevpt.method = "qd-nevpt2"
e_tot, e_corr, osc = nevpt.kernel()
