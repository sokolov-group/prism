#!/usr/bin/env python3
# 03-nevpt2-density-fitting.py
# Modified by Ziqiu Wang: < sgwzq0810@gmail.com >

'''
Using density fitting with NEVPT2 for N2
'''

import numpy as np
import pyscf.gto, pyscf.scf, pyscf.mcscf
import prism.interface, prism.mr_adc, prism.nevpt

mol = pyscf.gto.Mole()
r = 1.098
mol.atom = [
    ['N', ( 0., 0.    , -r/2)],
    ['N', ( 0., 0.    ,  r/2)],]
mol.basis = {'N':'aug-cc-pvtz'}
mol.verbose = 4
mol.build()

# RHF calculation as guess for CASSCF
mf = pyscf.scf.RHF(mol)
mf.kernel()

# SA-CASSCF reference, DF-NEVPT2
n_states = 9
weights = np.ones(n_states)/n_states
mcsa = pyscf.mcscf.CASSCF(mf, 6, 6).state_average_(weights)
mcsa.mc1step()[0]

mp_mcsa = prism.interface.PYSCF(mf, mcsa, backend = 'opt_einsum').density_fit('aug-cc-pvdz-ri')
mndf_mcsa = prism.nevpt.NEVPT(mp_mcsa)
mndf_mcsa.kernel()

# DF-SA-CASSCF reference, DF-NEVPT2
n_states = 9
weights = np.ones(n_states)/n_states
mcsadf = pyscf.mcscf.CASSCF(mf, 6, 6).state_average_(weights).density_fit('aug-cc-pvdz-ri')
mcsadf.mc1step()[0]

mp_mcsadf = prism.interface.PYSCF(mf, mcsadf, backend = 'opt_einsum').density_fit('aug-cc-pvdz-ri')
mndf_mcsadf = prism.nevpt.NEVPT(mp_mcsadf)
mndf_mcsadf.kernel()
