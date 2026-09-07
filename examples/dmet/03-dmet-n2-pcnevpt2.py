#!/usr/bin/env python

'''
DMET + PC-NEVPT2 (Prism fully internally contracted NEVPT2) for N2

Each N atom is a fragment; the Schmidt bath completes the embedding
cluster. The embedded SA-CASSCF reference feeds Prism's state-specific
NEVPT2 (full internal contraction, equivalent to partially contracted
NEVPT2). For the quasidegenerate variant use method='QD-NEVPT2'; for
PySCF's strongly contracted SC-NEVPT2 use method='NEVPT2'.
'''

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

# One-shot DMET; natorb picks the active space from a 4-orbital superset window
dmet = DMET(ints, frags, False, method='PC-NEVPT2',
            ncas=4, nelecas=4, sa_nstates=2, cas_select='natorb',
            scf_stability=True)
dmet.oneshot()

res = dmet.pcnevpt2_results[0]
print("\nPC-NEVPT2 state energies (embedded cluster, no nuclear repulsion):")
for i, (et, ec) in enumerate(zip(res['e_tot'], res['e_corr'])):
    print("  State %d: E_tot = %.10f Ha  E_corr = %.10f Ha" % (i, et, ec))
