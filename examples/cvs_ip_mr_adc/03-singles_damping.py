#!/usr/bin/env python

'''
CVS-IP-MR-ADC calculations using logarithmic sigmoid
damping prefactors for overlap eigenvalues
'''

import pyscf.gto
import pyscf.scf
import pyscf.mcscf
import prism.interface
import prism.mr_adc

mol = pyscf.gto.Mole()
r = 1.098 * 2
mol.atom = [
    ['N', ( 0., 0.    , -r/2)],
    ['N', ( 0., 0.    ,  r/2)],]
mol.basis = {'N':'aug-cc-pvdz'}
mol.verbose = 4
mol.build()

# RHF calculation as guess for CASSCF
mf = pyscf.scf.RHF(mol)
mf.kernel()

# CASSCF(6e,6o) calculation
mc = pyscf.mcscf.CASSCF(mf, 6, 6)
emc = mc.mc1step()[0]

# CVS-IP-MR-ADC calculation
interface = prism.interface.PYSCF(mf, mc, opt_einsum = True)
mr_adc = prism.mr_adc.MRADC(interface)
mr_adc.method_type = "cvs-ip"
mr_adc.ncvs = 2
mr_adc.nroots = 8

## Setup damping
mr_adc.s_damping_strength = 4
mr_adc.s_thresh_singles = 1e-5

e, p = mr_adc.kernel()
