# Copyright 2026 Prism Developers. All Rights Reserved.
#
# Licensed under the GNU General Public License v3.0;
# you may not use this file except in compliance with the License.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
# either express or implied.
#
# See the License file for the specific language governing
# permissions and limitations.
#
# Available at https://github.com/sokolov-group/prism
#
# Authors: Bryce Pickett <pickettosu@gmail.com>

import unittest
import numpy as np
import pyscf.gto
import pyscf.scf
import pyscf.mcscf
import prism.interface
import prism.nevpt
from prism.dmet import DMET, LocalIntegrals, make_fragments
from prism.dmet.cas_selectors import fix_cas_spin

np.set_printoptions(linewidth=150, edgeitems=10, suppress=True)

NCAS, NELECAS, NSTATES, SHIFT = 4, 2, 2, 0.2

mol = pyscf.gto.Mole()
mol.atom = [['C', (0.0, 0.0, 0.0)]]
mol.basis = '6-31g'
mol.spin = 2
mol.build()

# ROHF calculation
mf = pyscf.scf.ROHF(mol)
ehf = mf.scf()
print("SCF energy: %f\n" % ehf)

# A singlet CAS on a triplet reference, so fix_cas_spin repartitions mc.nelecas
# from (2, 0) to (1, 1).
mc = pyscf.mcscf.CASSCF(mf, NCAS, NELECAS)
mc = pyscf.mcscf.state_average_(mc, [1.0 / NSTATES] * NSTATES)
fix_cas_spin(mc, cas_spin=0, shift=SHIFT)
mc.conv_tol = 1e-11
emc = mc.kernel()[0]
print("CASSCF energy: %f\n" % emc)

nevpt = prism.nevpt.QDNEVPT(prism.interface.PYSCF(mf, mc))
nevpt.nfrozen = 0

# DMET with every orbital in the impurity, which makes the embedding exact.
ints = LocalIntegrals(mf, list(range(mol.nao_nr())), 'meta_lowdin')
frags = make_fragments(mol, ints, [[0]])
dmet = DMET(ints, frags, False, method='QD-NEVPT2', ncas=NCAS, nelecas=NELECAS,
            sa_nstates=NSTATES, sc_method='NONE', cas_spin=0, cas_spin_shift=SHIFT,
            casscf_kwargs={'conv_tol': 1e-11}, qdnevpt2_kwargs={'nfrozen': 0})

class KnownValues(unittest.TestCase):

    def test_pyscf(self):
        self.assertAlmostEqual(mf.e_tot, -37.676865648283, 5)
        self.assertAlmostEqual(mc.e_tot, -37.619728136538, 5)
        self.assertEqual(mc.nelecas, (1, 1))

    def test_prism(self):

        e_direct = nevpt.kernel()[0]

        e_dmet = dmet.oneshot()
        e_dmet_states = dmet.qdnevpt2_results[0]['e_tot']

        # Direct molecular and DMET-embedded NEVPT2 are separate pipelines.
        self.assertAlmostEqual(e_dmet, e_direct[0], 5)
        for state in range(NSTATES):
            self.assertAlmostEqual(e_dmet_states[state], e_direct[state], 5)

if __name__ == "__main__":
    print("DMET-QD-NEVPT2 singlet-CAS test")
    unittest.main()
