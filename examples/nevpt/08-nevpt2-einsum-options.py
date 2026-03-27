#!/usr/bin/env python
'''
NEVPT2 calculation for CH2
'''
from pyscf import gto, scf, mcscf
import prism.interface
import prism.mr_adc
import prism.nevpt

mol = gto.M()
mol.atom = '''
C   1
H   1 1.085
H   1 1.085  2 135.5
'''
mol.basis = 'cc-pvdz'
mol.symmetry = True
mol.spin = 2
mol.verbose = 4
mol.build()

# RHF calculation
mf = scf.RHF(mol)
mf.conv_tol = 1e-12
mf.kernel()

# CASSCF calculation
mc = mcscf.CASSCF(mf, 14, 6)
mc.with_dep4 = True
mc.max_cycle_micro = 10
mc.kernel()

# NEVPT2 with pytblis
interface = prism.interface.PYSCF(mf, mc, backend = "pytblis")
nevpt = prism.nevpt.NEVPT(interface)
e_tot, e_corr, osc = nevpt.kernel()

# NEVPT2 with opt_einsum
interface = prism.interface.PYSCF(mf, mc, backend = "opt_einsum")
nevpt = prism.nevpt.NEVPT(interface)
e_tot, e_corr, osc = nevpt.kernel()

# NEVPT2 with numpy
interface = prism.interface.PYSCF(mf, mc, backend = "numpy")
nevpt = prism.nevpt.NEVPT(interface)
e_tot, e_corr, osc = nevpt.kernel()
