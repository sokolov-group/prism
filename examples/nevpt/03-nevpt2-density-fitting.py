#!/usr/bin/env python

import numpy as np
import pyscf.gto
import pyscf.scf
import pyscf.mcscf
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

# SA-CASSCF reference, DF-NEVPT2
n_states = 9
weights = np.ones(n_states)/n_states
mc = pyscf.mcscf.CASSCF(mf, 6, 6).state_average_(weights)
emc = mc.mc1step()[0]

interface = prism.interface.PYSCF(mf, mc, opt_einsum = True).density_fit('aug-cc-pvdz-ri')
nevpt = prism.nevpt.NEVPT(interface)
nevpt.method = "nevpt2"
e_tot, e_corr = nevpt.kernel()

# DF-SA-CASSCF reference, DF-NEVPT2
n_states = 9
weights = np.ones(n_states)/n_states
mc = pyscf.mcscf.CASSCF(mf, 6, 6).state_average_(weights).density_fit('aug-cc-pvdz-ri')
emc = mc.mc1step()[0]

interface = prism.interface.PYSCF(mf, mc, opt_einsum = True).density_fit('aug-cc-pvdz-ri')
nevpt = prism.nevpt.NEVPT(interface)
nevpt.method = "nevpt2"
e_tot, e_corr = nevpt.kernel()
