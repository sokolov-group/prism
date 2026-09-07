#!/usr/bin/env python

'''
Minimal DMET calculation with an FCI solver on each H2 fragment, checked
against full-molecule FCI
'''

import pyscf.gto
import pyscf.scf
import pyscf.fci
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

# Full-molecule FCI, exact in this basis
e_fci = pyscf.fci.FCI(mf).kernel()[0]
print("Full-molecule FCI energy: %.10f" % e_fci)

# Localize the mean-field orbitals and define two 2-atom fragments
ints = LocalIntegrals(mf, list(range(mol.nao_nr())), 'meta_lowdin')
frags = make_fragments(mol, ints, [[0, 1], [2, 3]])

# One-shot DMET with an FCI solver on each embedded fragment
dmet = DMET(ints, frags, False, method='FCI')
e_dmet = dmet.oneshot()
print("DMET(FCI) energy: %.10f" % e_dmet)

# Self-consistent DMET fits a correlation potential to match the cluster densities.
dmet = DMET(ints, frags, False, method='FCI', sc_method='LSTSQ', max_cycle=200)
e_sc = dmet.selfconsistent()
print("Self-consistent DMET(FCI) energy: %.10f" % e_sc)

# With a density-fitted mean field the cluster integrals carry its fitting error.
mf_df = pyscf.scf.RHF(mol).density_fit()
mf_df.scf()
ints_df = LocalIntegrals(mf_df, list(range(mol.nao_nr())), 'meta_lowdin')
frags_df = make_fragments(mol, ints_df, [[0, 1], [2, 3]])
e_df = DMET(ints_df, frags_df, False, method='FCI').oneshot()
print("DMET(FCI) energy, density fitted: %.10f  (%.2e Ha from FCI)"
      % (e_df, abs(e_df - e_fci)))
