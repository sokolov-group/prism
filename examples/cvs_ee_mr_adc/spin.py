#!/usr/bin/env python

'''
CVS-EE-MR-ADC(2) calculations for N2
'''

import pyscf.gto
import pyscf.scf
import pyscf.mcscf
import prism_beta.interface
import prism_beta.mr_adc

mol = pyscf.gto.Mole()
r = 1.098
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

# CVS-EE-MR-ADC calculation
interface = prism_beta.interface.PYSCF(mf, mc, opt_einsum = True)
mr_adc = prism_beta.mr_adc.MRADC(interface)
mr_adc.method_type = "cvs-ee"
mr_adc.method = "mr-adc(2)"
mr_adc.s_thresh_singles = 1e-5
mr_adc.s_thresh_doubles = 1e-10
mr_adc.ncvs = 2
mr_adc.nroots = 8
e, p = mr_adc.kernel()
