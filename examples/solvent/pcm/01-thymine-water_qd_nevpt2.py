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

from pyscf.solvent.pcm import PCM

mol = pyscf.gto.Mole()
mol.atom = './01-thy.xyz'
mol.basis = 'cc-pVDZ' # Matches .pot file
mol.verbose = 4
mol.build()

# Polarizable Embedding
pcm = PCM(mol) # Create pcm object

# RHF calculation
mf = pyscf.scf.RHF(mol)
mf = pyscf.solvent.PCM(mf, pcm)
mf.conv_tol = 1e-12
mf.kernel()

# SA-CASSCF calculation
n_states = 6
weights = np.ones(n_states)/n_states
mc = pyscf.mcscf.CASSCF(mf, 6,6).state_average_(weights)
mc.conv_tol = 1e-8
mc.conv_tol_grad = 1e-6

mc = pyscf.solvent.PCM(mc, pcm)
mc.kernel()
mc.analyze()

# QD-NEVPT2 with all electrons correlated
interface = prism.interface.PYSCF(mf, mc, backend = 'opt_einsum')
nevpt = prism.nevpt.QDNEVPT(interface)
nevpt.pcm = pcm
nevpt.verbose = 5
e_tot, e_corr, osc = nevpt.kernel()