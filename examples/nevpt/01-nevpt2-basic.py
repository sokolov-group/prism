#!/usr/bin/env python3
# 01-nevpt2-basic.py
# Modified by Ziqiu Wang: < sgwzq0810@gmail.com >

'''
Basic NEVPT2 calculation for H2O
'''

import numpy as np
import math
import pyscf.gto , pyscf.scf , pyscf.mcscf
import prism.interface , prism.mr_adc , prism.nevpt

r = 0.96
x = r * math.sin(104.5 * math.pi/(2 * 180.0))
y = r * math.cos(104.5 * math.pi/(2 * 180.0))

mol = pyscf.gto.Mole()
mol.atom = [
            ['O', (0.0, 0.0, 0.0)],
            ['H', (0.0,  -x,   y)],
            ['H', (0.0,   x,   y)]]
mol.basis = 'aug-cc-pvtz'
mol.symmetry = 'C2v'
# Alternatively, we can simply set it to True
# then, PySCF will decide the approximate point group for us

# mol.symmetry = True

# But it sometimes fails. For example a "Dooh" O2 molecule will fail to 
# proceed with CASSCF as it will raise errors.
# In this case, we need to specify "D2h" manually.
mol.verbose = 4
mol.build()

# RHF calculation
mf = pyscf.scf.RHF(mol)
mf.conv_tol = 1e-12
mf.kernel()

# CASSCF calculation
mc = pyscf.mcscf.CASSCF(mf, 6, 6)
mc.conv_tol = 1e-11
mc.conv_tol_grad = 1e-6

emc = mc.mc1step()[0]
mc.analyze()
print("CASSCF energy: %f\n" % emc)



# Import PySCF molecule to prism
mp = prism.interface.PYSCF(mf, mc, backend = 'opt_einsum')

# NEVPT2 with all electrons correlated
mn = prism.nevpt.NEVPT(mp)
mn.kernel()

# NEVPT2 with frozen core
mn_fc = mn
mn_fc.nfrozen = 1
mn_fc.kernel()
