#!/usr/bin/env python

'''
DMET with the Prism solvers for N2: CASSCF, PC-NEVPT2 and CVS-IP-MR-ADC

Each N atom is a fragment; the Schmidt bath completes the embedding cluster and
the remaining orbitals are frozen into the core. The DMET energy is a total
energy. The PC-NEVPT2 state energies are for the embedded cluster and leave out
the nuclear repulsion, while the MR-ADC roots are energy differences.
'''

import numpy as np
import pyscf.gto
import pyscf.scf
from prism.dmet import DMET, LocalIntegrals, make_fragments

mol = pyscf.gto.Mole()
mol.atom = [['N', (0, 0, 0)], ['N', (0, 0, 1.5)]]
mol.basis = 'sto-3g'
mol.verbose = 4
mol.build()

# RHF calculation
mf = pyscf.scf.RHF(mol)
ehf = mf.scf()
print("SCF energy: %f\n" % ehf)

ints = LocalIntegrals(mf, list(range(mol.nao_nr())), 'meta_lowdin')
frags = make_fragments(mol, ints, [[0], [1]])

# Embedded CASSCF. print_bath_spectrum lists the occupation, the deviation from
# 0 or 2, and the entropy of the orbitals on either side of the bath cut.
dmet = DMET(ints, frags, False, method='CASSCF', ncas=4, nelecas=4,
            print_bath_spectrum=True)
e_dmet = dmet.oneshot()
print("\nDMET(CASSCF) energy: %.10f" % e_dmet)

# The first fragment's density in the AO basis of the molecule: its embedded
# 1-RDM mapped back, plus the density frozen into the core.
dm_ao = dmet.to_ao(dmet.imp_rdm1[0]) + dmet.core_dm_ao(0)
print("Electrons in that density: %.6f of %d"
      % (np.trace(dm_ao @ mol.intor('int1e_ovlp')), mol.nelectron))

# PC-NEVPT2 (Prism fully internally contracted NEVPT2) on a state-averaged
# reference; natorb picks the active space from a 4-orbital superset window.
# For the quasidegenerate variant use method='QD-NEVPT2'.
dmet = DMET(ints, frags, False, method='PC-NEVPT2',
            ncas=4, nelecas=4, sa_nstates=2, cas_select='natorb',
            scf_stability=True)
dmet.oneshot()

res = dmet.pcnevpt2_results[0]
print("\nPC-NEVPT2 state energies (embedded cluster, no nuclear repulsion):")
for i, (et, ec) in enumerate(zip(res['e_tot'], res['e_corr'])):
    print("  State %d: E_tot = %.10f Ha  E_corr = %.10f Ha" % (i, et, ec))

# CVS-IP-MR-ADC(2) for the N 1s of the fragment atom. MR-ADC takes a
# single-state reference, so the roots are requested through mradc_kwargs.
dmet = DMET(ints, frags, False, method='MR-ADC', ncas=4, nelecas=4,
            mradc_kwargs={'method_type': 'cvs-ip', 'ncvs': 1, 'nroots': 3})
dmet.oneshot()

res = dmet.mradc_results[0]
print("\nCVS-IP-MR-ADC(2) N 1s ionization energies:")
for i, (e, p) in enumerate(zip(res['e_exc'], res['spec_factors'])):
    print("  Root %d: %10.4f eV   intensity %.6f" % (i, e, p))
