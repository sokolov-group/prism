#!/usr/bin/env python

'''
CVS-IP-MR-ADC(2) calculations for C2
'''

import pyscf.gto
import pyscf.scf
import pyscf.mcscf
import prism.interface
import prism.mr_adc

mol = pyscf.gto.Mole()
mol.atom = 'C 0 0 0; C 0 0 1.2'
mol.basis = 'ccpvdz'
mol.build()
mol.verbose = 4

# RHF calculation as guess for CASSCF
mf = pyscf.scf.RHF(mol)
mf.kernel()

# CASSCF calculation
mc = pyscf.mcscf.CASSCF(mf, 6, (4, 2))
emc = mc.mc1step()[0]

# CVS-IP-MR-ADC calculation with opt_einsum
interface = prism.interface.PYSCF(mf, mc, backend = 'opt_einsum')
mr_adc = prism.mr_adc.MRADC(interface)
mr_adc.method_type = "cvs-ip"
mr_adc.ncvs = 2
mr_adc.nroots = 3
mr_adc.kernel()

# CVS-IP-MR-ADC calculation with pytblis
interface = prism.interface.PYSCF(mf, mc, backend = 'pytblis')
mr_adc = prism.mr_adc.CVSIPMRADC(interface)
mr_adc.ncvs = 2
mr_adc.nroots = 3
mr_adc.kernel()

# CVS-IP-MR-ADC calculation with numpy
interface = prism.interface.PYSCF(mf, mc, backend = 'numpy')
mr_adc = prism.mr_adc.CVSIPMRADC(interface)
mr_adc.ncvs = 2
mr_adc.nroots = 3
mr_adc.kernel()

