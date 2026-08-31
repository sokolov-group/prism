#!/usr/bin/env python3
# 02-nevpt2-references.py
# Modified by Ziqiu Wang < sgwzq0810@gmail.com >

'''
Using different reference wavefunctions to run NEVPT2 for N2
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

####################
# CASSCF reference
####################
mc = pyscf.mcscf.CASSCF(mf, 6, 6)
emc = mc.mc1step()[0]

mp_cas = prism.interface.PYSCF(mf, mc, backend = 'opt_einsum')
mn_cas = prism.nevpt.NEVPT(mp_cas)
mn_cas.kernel()

####################
# SA-CASSCF reference
####################
n_states = 9
weights = np.ones(n_states)/n_states
mc_sa = pyscf.mcscf.CASSCF(mf, 6, 6).state_average_(weights)
emc = mc_sa.mc1step()[0]

mp_sa = prism.interface.PYSCF(mf, mc_sa, backend = 'opt_einsum')
mn_sa = prism.nevpt.NEVPT(mp_sa)
mn_sa.kernel()

####################
# MS-CASCI reference
####################
n_states = 9
mc_ms = pyscf.mcscf.CASCI(mf, 6, 6)
mc_ms.fcisolver.nroots = n_states
emc = mc_ms.casci()[0]

mp_ms = prism.interface.PYSCF(mf, mc_ms, backend = 'opt_einsum')
mn_ms = prism.nevpt.NEVPT(mp_ms)
mn_ms.kernel()
