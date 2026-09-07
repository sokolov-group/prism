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

# Self-consistent DMET: fit a correlation potential so the mean-field density of each
# cluster matches the correlated one, rebuilding the bath at every step.
dmet = DMET(ints, frags, False, method='FCI', sc_method='LSTSQ', max_cycle=200)
e_sc = dmet.selfconsistent()
print("Self-consistent DMET(FCI) energy: %.10f" % e_sc)

# A density-fitted mean field is cheaper for large systems: the cluster integrals then
# come from its three-index tensor and carry the same fitting error.
mf_df = pyscf.scf.RHF(mol).density_fit()
mf_df.scf()
ints_df = LocalIntegrals(mf_df, list(range(mol.nao_nr())), 'meta_lowdin')
frags_df = make_fragments(mol, ints_df, [[0, 1], [2, 3]])
e_df = DMET(ints_df, frags_df, False, method='FCI').oneshot()
print("DMET(FCI) energy, density fitted: %.10f  (error %.2e Ha)"
      % (e_df, abs(e_df - e_dmet)))
