#!/usr/bin/env python3

'''
QD-NEVPT2 eigenvector analysis for the OH radical (doublet).

Demonstrates:
  - Dominant CI configurations in the QD-NEVPT2 eigenbasis
  - Natural occupations of the active space for each state
  - Mulliken spin populations (alpha - beta) for open-shell states
  - Customizable weight cutoff for configuration printing
'''

import numpy as np
import pyscf.gto
import pyscf.scf
import pyscf.mcscf
import prism.interface
import prism.nevpt

mol = pyscf.gto.Mole()
mol.atom = [
    ['O', (0.0, 0.0, 0.0)],
    ['H', (0.0, 0.0, 0.9697)]]
mol.basis = '6-31g'
mol.spin = 1
mol.verbose = 4
mol.build()

# ROHF calculation
mf = pyscf.scf.ROHF(mol)
mf.conv_tol = 1e-11
mf.kernel()

# SA-CASSCF reference (3 doublet states)
n_states = 3
weights = np.ones(n_states) / n_states
mc = pyscf.mcscf.CASSCF(mf, 5, 7).state_average_(weights)
mc.conv_tol = 1e-10
mc.conv_tol_grad = 1e-6
mc.mc1step()

# QD-NEVPT2
interface = prism.interface.PYSCF(mf, mc, backend='opt_einsum')
nevpt = prism.nevpt.QDNEVPT(interface)
e_tot, e_corr, osc = nevpt.kernel()

# Eigenvector analysis with default weight cutoff (1%)
nevpt.analyze()

# Eigenvector analysis with strict weight cutoff (5%)
nevpt.analyze(weight_cutoff=0.05)
