#!/usr/bin/env python

'''
Minimal DMET calculation: H2 fragment solved with FCI
'''

import pyscf.gto
import pyscf.scf
from prism.dmet import DMET, LocalIntegrals, make_fragments

mol = pyscf.gto.Mole()
mol.atom = 'H 0 0 0; H 0 0 0.74; H 0 0 1.48; H 0 0 2.22'
mol.basis = 'sto-3g'
mol.verbose = 4
mol.build()

# RHF calculation
mf = pyscf.scf.RHF(mol)
ehf = mf.scf()
print("SCF energy: %f\n" % ehf)

# Localize the mean-field orbitals and define two 2-atom fragments
ints = LocalIntegrals(mf, list(range(mol.nao_nr())), 'meta_lowdin')
frags = make_fragments(mol, ints, [[0, 1], [2, 3]])

# One-shot DMET with an FCI solver on each embedded fragment
dmet = DMET(ints, frags, False, method='FCI')
e_dmet = dmet.oneshot()
print("DMET(FCI) energy: %.10f" % e_dmet)
