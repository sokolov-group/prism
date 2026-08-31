#!/usr/bin/env python3

'''
BASIC QD-NEVPT2 + PE calculation for thymine + 3.5 angstrom water shell
'''

import numpy as np
import pyscf.gto
import pyscf.scf
import pyscf.mcscf
import pyscf.solvent
from pyscf.tools import molden
import prism.interface
import prism.nevpt
import sys

from pyscf.solvent.pol_embed import PolEmbed

mol = pyscf.gto.Mole()
mol.atom = '''
N      1.131000     0.328000     0.042000
C     -0.012000     1.109000     0.048000
O      0.076000     2.331000     0.105000
C     -1.251000     0.373000    -0.014000
C     -2.528000     1.131000    -0.012000
C     -1.187000    -0.970000    -0.071000
N     -0.003000    -1.650000    -0.072000
C      1.215000    -1.040000    -0.016000
O      2.278000    -1.644000    -0.018000
H      2.013000     0.818000     0.084000
H     -2.066000    -1.587000    -0.119000
H      0.002000    -2.656000    -0.116000
H     -2.614000     1.735000     0.886000
H     -2.572000     1.809000    -0.860000
H     -3.375000     0.456000    -0.061000
'''
mol.basis = "cc-pVDZ"
mol.verbose = 4
mol.build()

# Polarizable Embedding Options
pe_options = {"potfile": "thy-water-3.5-angstrom.pot"} # Given .pot file
pe = PolEmbed(mol, pe_options)
pe.verbose = 3

# RHF calculation 
mf = pyscf.scf.RHF(mol)
mf = pyscf.solvent.PE(mf, pe)
mf.kernel()

# CASSCF
n_states = 4
weights = np.ones(n_states)/n_states

mc = pyscf.mcscf.CASSCF(mf, 4,4).state_average_(weights)
mc = pyscf.solvent.PE(mc, pe)
mc.fix_spin_(ss=0)
mc.kernel(mf.mo_coeff)

# QD-NEVPT2
interface = prism.interface.PYSCF(mf, mc)
nevpt = prism.nevpt.NEVPT(interface)

# Polarizable Embedding
nevpt.method_type = "qd"
nevpt.pe = pe  

# Recommended for PE use
nevpt.s_thresh_singles = 1e-6
nevpt.s_thresh_doubles = 1e-6

e_tot, e_corr, osc = nevpt.kernel()