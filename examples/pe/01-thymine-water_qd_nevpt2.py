#!/usr/bin/env python

'''
PE QD-NEVPT2 Example for Microhydrated Thymine
'''

import numpy as np
import math
import pyscf.gto
import pyscf.scf
import pyscf.mcscf
import prism.interface
import prism.mr_adc
import prism.nevpt

from pyscf.solvent.pol_embed import PolEmbed

mol = pyscf.gto.Mole()
mol.atom = './01-thy.xyz'
mol.basis = 'sto-3g'
mol.symmetry = False
mol.build()

# Polarizable Embedding
pe_options = {"potfile": "01-thy-wat.pot"} # Define potential file
pe = PolEmbed(mol, pe_options) # Create pe object

# RHF calculation
mf = pyscf.scf.RHF(mol)
mf = pyscf.solvent.PE(mf, pe)
mf.conv_tol = 1e-12
mf.kernel()

# SA-CASSCF calculation
n_states = 6
weights = np.ones(n_states)/n_states
mc = pyscf.mcscf.CASSCF(mf, 6, 6).state_average_(weights)
mc.conv_tol = 1e-8
mc.conv_tol_grad = 1e-6

mc = pyscf.solvent.PE(mc, pe)
mc.kernel()
mc.analyze()

# QD-NEVPT2 with all electrons correlated
interface = prism.interface.PYSCF(mf, mc, backend = 'opt_einsum')
nevpt = prism.nevpt.QDNEVPT(interface)
nevpt.pe = pe
nevpt.verbose = 5
e_tot, e_corr, osc = nevpt.kernel()

