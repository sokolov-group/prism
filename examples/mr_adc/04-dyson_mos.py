#!/usr/bin/env python

'''
CVS-IP-MR-ADC(2) Dyson orbitals calculation for N2O
'''

import pyscf.gto
import pyscf.scf
import pyscf.mcscf
import prism.interface
import prism.mr_adc

mol = pyscf.gto.Mole()
mol.atom = '''
N        0.000000000      0.000000000      0.072190360
N        0.000000000      0.000000000      1.198898684
O        0.000000000      0.000000000     -1.112801545
'''
mol.basis = 'aug-cc-pvdz'
mol.verbose = 4
mol.build()

# RHF calculation as guess for CASSCF
mf = pyscf.scf.RHF(mol)
mf.kernel()

# CASSCF(4e,4o) calculation
mc = pyscf.mcscf.CASSCF(mf, 4, 4)

cas_list = [10, 11, 14, 15]
mo = pyscf.mcscf.sort_mo(mc, mf.mo_coeff, cas_list)

emc = mc.mc1step(mo)[0]

# CVS-IP-MR-ADC calculation
interface = prism.interface.PYSCF(mf, mc, opt_einsum = True)
mr_adc = prism.mr_adc.MRADC(interface)
mr_adc.method_type = "cvs-ip"
mr_adc.ncvs = 3
mr_adc.nroots = 8

e, p, x = mr_adc.kernel()

from prism.mr_adc_cvs_ip import compute_dyson_mo
from pyscf.tools import molden

dyson_mos = compute_dyson_mo(mr_adc, x)
molden.from_mo(mol, 'mr_adc_dyson_mos.molden', dyson_mos)
