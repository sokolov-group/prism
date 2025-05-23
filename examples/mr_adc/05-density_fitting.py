#!/usr/bin/env python

'''
DF-CVS-IP-MR-ADC(2)-X calculations for N2
'''

import pyscf.gto
import pyscf.scf
import pyscf.mcscf
import prism.interface
import prism.mr_adc

mol = pyscf.gto.Mole()
r = 1.098
mol.atom = [
    ['N', ( 0., 0.    , -r/2)],
    ['N', ( 0., 0.    ,  r/2)],]
mol.basis = {'N':'aug-cc-pvdz'}
mol.verbose = 4
mol.build()

## RHF calculation as guess for CASSCF
mf = pyscf.scf.RHF(mol)
mf.kernel()

## CASSCF(6e,6o) calculation
mc = pyscf.mcscf.CASSCF(mf, 6, 6)
emc = mc.mc1step()[0]

## CVS-IP-MR-ADC calculation
interface = prism.interface.PYSCF(mf, mc, opt_einsum = True).density_fit('aug-cc-pvdz-ri')
mr_adc = prism.mr_adc.MRADC(interface)
mr_adc.method_type = "cvs-ip"
mr_adc.method = "mr-adc(2)"
mr_adc.ncvs = 2
mr_adc.nroots = 8
mr_adc.max_space = mr_adc.nroots * 4
mr_adc.max_cycle = 100

e, p, x = mr_adc.kernel()

# Using JK fitted CASSCF
## RHF calculation as guess for CASSCF
mf = pyscf.scf.RHF(mol)
mf.kernel()

## CASSCF(6e,6o) calculation
mc = pyscf.mcscf.CASSCF(mf, 6, 6).density_fit('aug-cc-pvdz-jkfit')
emc = mc.mc1step()[0]

## CVS-IP-MR-ADC calculation
interface = prism.interface.PYSCF(mf, mc, opt_einsum = True).density_fit('aug-cc-pvdz-ri')
mr_adc = prism.mr_adc.MRADC(interface)
mr_adc.method_type = "cvs-ip"
mr_adc.method = "mr-adc(2)"
mr_adc.ncvs = 2
mr_adc.nroots = 8
mr_adc.max_space = mr_adc.nroots * 4
mr_adc.max_cycle = 100

e, p, x = mr_adc.kernel()
