#!/usr/bin/env python

'''
CVS-IP-SR-ADC(2)-X calculations for OH
'''

import pyscf.gto
import pyscf.scf
import pyscf.adc
import prism.interface
import prism.mr_adc

r = 0.969286393
mol = pyscf.gto.Mole()
mol.atom = [
    ['O', ( 0., 0.    , -r/2   )],
    ['H', ( 0., 0.    ,  r/2)],]
mol.basis = 'cc-pvdz'
mol.symmetry = False
mol.spin  = 1
mol.verbose = 4
mol.build()

# Run ROHF computation
mf = pyscf.scf.ROHF(mol).density_fit('cc-pvdz-jkfit')
mf.kernel()

# Run Prism SR-ADC
interface = prism.interface.PYSCF(mf, backend = 'opt_einsum')
mr_adc = prism.mr_adc.CVSIPMRADC(interface)
mr_adc.ncvs = 1
mr_adc.method = "mr-adc(2)-x"
mr_adc.nroots = 10
mr_adc.kernel()

