#!/usr/bin/env python

'''
CVS-IP-SR-ADC(2)-X calculations for H2O
'''

import math
import pyscf.gto
import pyscf.scf
import pyscf.adc
import prism.interface
import prism.mr_adc

r = 0.96
x = r * math.sin(104.5 * math.pi/(2 * 180.0))
y = r * math.cos(104.5 * math.pi/(2 * 180.0))
mol = pyscf.gto.Mole()
mol.atom = [
['O', ( 0., 0., 0.)],
['H', ( 0., -x,  y)],
['H', ( 0.,  x,  y)],
]
mol.basis = 'aug-cc-pvdz'
mol.symmetry = True
mol.verbose = 4
mol.build()

# Run RHF computation
mf = pyscf.scf.RHF(mol)
mf.scf()

# Shared Parameters
ncvs = 1
tol_e = 1e-8
tol_r = 1e-5 
space = 12

# Run Prism SR-ADC
interface = prism.interface.PYSCF(mf, backend = 'pytblis')
prism_adc = prism.mr_adc.CVSIPMRADC(interface)
prism_adc.ncvs = ncvs
prism_adc.method = "mr-adc(2)-x"
prism_adc.tol_e = tol_e
prism_adc.tol_r = tol_r
prism_adc.max_space = space
prism_adc.nroots = 10
e,p,x = prism_adc.kernel()

# Run PySCF SR-ADC (Doublet States Only)
pyscf_adc = pyscf.adc.ADC(mf)
pyscf_adc.ncvs = ncvs
pyscf_adc.method = "adc(2)-x"
pyscf_adc.conv_tol = tol_e 
pyscf_adc.tol_residual = tol_r
pyscf_adc.max_space = space
eip,vip,pip,xip = pyscf_adc.kernel(nroots = 10)

