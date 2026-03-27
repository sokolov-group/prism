#!/usr/bin/env python
'''
QD-NEVPT calculation for O2
'''
from pyscf import gto, scf, mcscf
import prism.interface
import prism.nevpt

mol = gto.Mole()
mol.atom = 'O 0 0 0; O 0 0 1.2'
mol.spin = 2
mol.basis = 'ccpvtz'
mol.verbose = 4
mol.build()

# RHF calculation
mf = scf.RHF(mol).density_fit(auxbasis='ccpvtz-jkfit')
ehf = mf.scf()
print("SCF energy: %f\n" % ehf)

# MS-CASCI calculation
from pyscf.mcscf import avas
mo_coeff = avas.kernel(mf, ['O 2p'], minao=mol.basis)[2]

mc = mcscf.CASCI(mf, 6, 8)
mc.fcisolver.nroots = 3
emc = mc.kernel(mo_coeff)[0]

# QD-NEVPT with pytblis
interface = prism.interface.PYSCF(mf, mc, backend = 'pytblis')
nevpt = prism.nevpt.QDNEVPT(interface)
e_tot, e_corr, osc = nevpt.kernel()

# QD-NEVPT with auto-detection for tensor contraction engine
interface = prism.interface.PYSCF(mf, mc, backend = None)
nevpt = prism.nevpt.QDNEVPT(interface)
e_tot, e_corr, osc = nevpt.kernel()

# QD-NEVPT with numpy
interface = prism.interface.PYSCF(mf, mc, backend = 'numpy')
nevpt = prism.nevpt.QDNEVPT(interface)
e_tot, e_corr, osc = nevpt.kernel()

