#!/usr/bin/env python

'''
CVS-IP-MR-ADC calculations for CO
'''

import pyscf.gto
import pyscf.scf
import pyscf.mcscf
import prism.interface
import prism.mr_adc

mol = pyscf.gto.Mole()
r = 1.128
mol.atom = [
    ['C', ( 0., 0.    , -r/2)],
    ['O', ( 0., 0.    ,  r/2)],]
mol.basis = 'aug-cc-pvdz'
mol.verbose = 4
mol.build()

# RHF calculation as guess for CASSCF
mf = pyscf.scf.RHF(mol)
mf.kernel()

# CASSCF(6e,6o) calculation
mc = pyscf.mcscf.CASSCF(mf, 6, 6)
emc = mc.mc1step()[0]

# CVS-IP-MR-ADC calculations
## First calculation: CVS space for O 1s^{-1} states
print("## First calculation: CVS space for O 1s^{-1} states")
interface = prism.interface.PYSCF(mf, mc, opt_einsum = True)
mr_adc = prism.mr_adc.MRADC(interface)
mr_adc.method_type = "cvs-ip"
mr_adc.ncvs = 1
mr_adc.nroots = 4

e, p, x = mr_adc.kernel()

## Second calculation: CVS space for C 1s^{-1} states
print("## Second calculation: CVS space for C 1s^{-1} states")
interface = prism.interface.PYSCF(mf, mc, opt_einsum = True)
mr_adc = prism.mr_adc.MRADC(interface)
mr_adc.method_type = "cvs-ip"
mr_adc.ncvs = 2
mr_adc.nroots = 4

e, p, x = mr_adc.kernel()
