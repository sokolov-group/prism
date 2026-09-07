#!/usr/bin/env python

'''
DMET + QD-NEVPT2 excitation energies for H2O

The water molecule is embedded as a single fragment (impurity = whole
molecule) and solved with an embedded SA-CASSCF reference followed by
Prism QD-NEVPT2. The natorb selector picks the active space from the
natural-orbital occupations of a superset CASCI.

Note: total energies from the embedded QD-NEVPT2 lack nuclear repulsion
(the embedded cluster carries no real molecule); excitation energies are
unaffected. Oscillator strengths are reported as zero for the same reason.
'''

import math
import pyscf.gto
import pyscf.scf
from prism.dmet import DMET, LocalIntegrals, make_fragments

r = 0.96
x = r * math.sin(104.5 * math.pi / (2 * 180.0))
y = r * math.cos(104.5 * math.pi / (2 * 180.0))

mol = pyscf.gto.Mole()
mol.atom = [
    ['O', (0.0, 0.0, 0.0)],
    ['H', (0.0,  -x,   y)],
    ['H', (0.0,   x,   y)]]
mol.basis = 'aug-cc-pvdz'
mol.verbose = 4
mol.build()

# RHF calculation
mf = pyscf.scf.RHF(mol)
mf.conv_tol = 1e-12
ehf = mf.scf()
print("SCF energy: %f\n" % ehf)

# Localized integrals; the whole molecule is one fragment
ints = LocalIntegrals(mf, list(range(mol.nao_nr())), 'meta_lowdin')
frags = make_fragments(mol, ints, [[0, 1, 2]])

# One-shot DMET with embedded SA-CASSCF(6,6) -> QD-NEVPT2 over 6 states.
# rohf_stability/cas_multiseed activate the determinism safeguards.
dmet = DMET(ints, frags, False, method='QD-NEVPT2',
            ncas=6, nelecas=6, sa_nstates=6, cas_select='natorb',
            rohf_stability=True, cas_multiseed=True,
            qdnevpt2_kwargs={'nfrozen': 1})
dmet.oneshot()

res = dmet.qdnevpt2_results[0]
print("\nQD-NEVPT2 excitation energies (eV):")
for i, e in enumerate(res['e_tot']):
    print("  State %d: %+.4f" % (i, (e - res['e_tot'][0]) * 27.21138602))
