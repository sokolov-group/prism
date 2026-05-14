'''
SOC CASSCF calculation for B
'''

import numpy as np
import math
import pyscf.gto
import pyscf.scf
import pyscf.mcscf
import prism.interface
import prism.mr_adc
import prism.nevpt
import time

np.set_printoptions(linewidth=150, edgeitems=10, suppress=True)
mol = pyscf.gto.Mole()
mol.atom =[ 
[ 'B',  (0, 0, 0)],
]

mol.basis = 'sto-3g' #'def2-tzvp'
mol.symmetry = False
mol.spin = 1
mol.verbose = 4
mol.build()

# RHF calculation
mf = pyscf.scf.RHF(mol).x2c()
mf.conv_tol = 1e-12

ehf = mf.scf()
mf.analyze()

# SA-CASSCF calculation
n_states = 3
weights = np.ones(n_states)/n_states
mc = pyscf.mcscf.CASSCF(mf, 4, 3).state_average_(weights)
#mc.conv_tol = 1e-11
#mc.conv_tol_grad = 1e-6
emc = mc.mc1step()[0]
mc.analyze()



interface = prism.interface.PYSCF(mf, mc, backend = 'opt_einsum')
interface.run_soc("x2c-1")
