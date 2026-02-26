#!/usr/bin/env python

'''
QD-NEVPT2 dipole moment calculations for H2O
'''

import numpy as np
import math
import pyscf.gto
import pyscf.scf
import pyscf.mcscf
import prism.interface
import prism.nevpt

r = 0.96
x = r * math.sin(104.5 * math.pi/(2 * 180.0))
y = r * math.cos(104.5 * math.pi/(2 * 180.0))

mol = pyscf.gto.Mole()
mol.atom = [
            ['O', (0.0, 0.0, 0.0)],
            ['H', (0.0,  -x,   y)],
            ['H', (0.0,   x,   y)]]

mol.basis = 'aug-cc-pvdz'
mol.symmetry = False
mol.verbose = 4
mol.build()

# RHF calculation
mf = pyscf.scf.RHF(mol)
mf.conv_tol = 1e-9

ehf = mf.scf()
print("SCF energy: %f\n" % ehf)

# SA-CASSCF calculation
n_states = 5
weights = np.ones(n_states)/n_states
mc = pyscf.mcscf.CASSCF(mf, 6, 6).state_average_(weights)
mc.conv_tol = 1e-11
mc.conv_tol_grad = 1e-6

emc = mc.mc1step()[0]
mc.analyze()
print("CASSCF energy: %f\n" % emc)

# QD-NEVPT2 with all electrons correlated
interface = prism.interface.PYSCF(mf, mc, opt_einsum = True)
nevpt = prism.nevpt.NEVPT(interface)
nevpt.method = "nevpt2"
nevpt.s_thresh_singles = 1e-6
nevpt.s_thresh_doubles = 1e-6
nevpt.keep_amplitudes = True 

# Correlated 1RDM
nevpt.rdm_order = 2 

e_tot, e_corr, osc = nevpt.kernel()

# Calculate Dipole Moments
dip_mom_ao = nevpt.interface.dip_mom_ao
dip_mom_mo = np.zeros_like(dip_mom_ao)

# Transform dipole moments from AO to MO basis
for d in range(dip_mom_ao.shape[0]):
    dip_mom_mo[d] = np.dot(nevpt.mo.T, np.dot(dip_mom_ao[d], nevpt.mo))

# Nuclear
charges = nevpt.interface.mol.atom_charges()
coords  = nevpt.interface.mol.atom_coords()
nucl_dip = nevpt.interface.einsum('i,ix->x', charges, coords)

# Compute 1RDMS
# Ground state
gs_1rdm = nevpt.make_rdm1(m = 0, n = 0)

# Excited States
es1_1rdm = nevpt.make_rdm1(m = 1, n = 1)
es2_1rdm = nevpt.make_rdm1(m = 2, n = 2)
es3_1rdm = nevpt.make_rdm1(m = 3, n = 3)
es4_1rdm = nevpt.make_rdm1(m = 4, n = 4)

# Transition 1RDM 1 -> 3
tr_1rdm = nevpt.make_rdm1(m = 0, n = 2) #Root 0 indexing

# Compute dipoles
ref_dip = nevpt.interface.einsum("xqr,qr->x", dip_mom_mo, gs_1rdm) + nucl_dip

es1_dip = nevpt.interface.einsum("xqr,qr->x", dip_mom_mo, es1_1rdm) + nucl_dip
es2_dip = nevpt.interface.einsum("xqr,qr->x", dip_mom_mo, es2_1rdm) + nucl_dip
es3_dip = nevpt.interface.einsum("xqr,qr->x", dip_mom_mo, es3_1rdm) + nucl_dip

tot_es_dip = [es1_dip, es2_dip, es3_dip]

tr_dip = nevpt.interface.einsum("xqr,qr->x", dip_mom_mo, tr_1rdm) 

print()
header = "*   1RDM Dipole Moment Contracted Integrals (Multistate NEVPT2)   *"
print("*" * len(header))
print(header)
print("*" * len(header))

hline = "-" * len(header)
print(hline)

print("Reference dipole moment (a.u.):")
print(f"    X: {ref_dip[0]:15.6e}")
print(f"    Y: {ref_dip[1]:15.6e}")
print(f"    Z: {ref_dip[2]:15.6e}")
print(hline)

for es in range(len(tot_es_dip)):
    print("Excited state 1 dipole moment (a.u.):")
    print(f"    X: {tot_es_dip[es][0]:15.6e}")
    print(f"    Y: {tot_es_dip[es][1]:15.6e}")
    print(f"    Z: {tot_es_dip[es][2]:15.6e}")
    print(hline)

print("Transition dipole moment (1 -> 3) (a.u.):")
print(f"    X: {tr_dip[0]:15.6e}")
print(f"    Y: {tr_dip[1]:15.6e}")
print(f"    Z: {tr_dip[2]:15.6e}")
print(hline)

print("*" * len(header))