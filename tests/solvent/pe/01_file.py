#INFO: **** input file is /users/PAS0291/jdserna22/repos/prism/tests/solvent/pe/01-test-reference.py ****
# AUTHOR: James Serna
#!/usr/bin/env python

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
N     29.660000    27.490000    28.690000
C     29.850000    28.660000    29.350000
C     31.000000    28.870000    30.210000
C     31.170000    30.170000    30.870000
C     30.170000    31.240000    30.650000
C     29.000000    31.020000    29.780000
C     28.840000    29.750000    29.140000
N     30.340000    32.580000    31.300000
O     31.300000    32.810000    32.020000
O     29.510000    33.460000    31.110000
H     30.320000    26.750000    28.790000
H     28.860000    27.380000    28.090000
H     31.730000    28.090000    30.370000
H     32.020000    30.340000    31.520000
H     28.260000    31.810000    29.640000
H     27.980000    29.580000    28.490000
'''
mol.basis = "cc-pVDZ"

mol.verbose = 4
mol.build()

# Polarizable Embedding
pe_options = {"potfile": "01-pna-water-6-angstrom.pot"} # Given .pot file
pe = PolEmbed(mol, pe_options)
pe.verbose = 3

# RHF calculation as guess for CASSCF
mf = pyscf.scf.RHF(mol)
mf = pyscf.solvent.PE(mf, pe)

mf.kernel()
mf.analyze()
mo = mf.mo_coeff


### SA-CASSCF ### 
n_states = 4
weights = np.ones(n_states)/n_states

mc = pyscf.mcscf.CASSCF(mf, 4,4).state_average_(weights)
mc = pyscf.solvent.PE(mc, pe)
mc.fix_spin_(ss=0)

mc.kernel(mo)

interface = prism.interface.PYSCF(mf, mc, backend = 'opt_einsum')
nevpt = prism.nevpt.NEVPT(interface)

nevpt.method_type = "qd"
nevpt.pe = pe  # Needed to compute ptss and ptlr

# Shifts if needed...
nevpt.s_thresh_singles = 1e-6
nevpt.s_thresh_doubles = 1e-6
nevpt.rdm_order = 2
nevpt.verbose = 5

nevpt.keep_amplitudes = True

e_tot, e_corr, osc = nevpt.kernel()

print('QD-NEVPT2 Correlation Energies: ')
print(e_corr)