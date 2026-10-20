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
import pyscf.fci
from prism.dmet import DMET, LocalIntegrals, make_fragments

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

# A singlet CAS on a triplet reference, so the CI solver's spin differs from the
# reference's. The solver carries the CAS spin, then the state average wraps it.
mc = pyscf.mcscf.CASSCF(mf, NCAS, NELECAS)
mc.fcisolver.spin = 0
pyscf.fci.addons.fix_spin_(mc.fcisolver, shift=SHIFT, ss=0.0)
mc = pyscf.mcscf.state_average_(mc, [1.0 / NSTATES] * NSTATES)
mc.conv_tol = 1e-11
emc = mc.kernel()[0]
print("CASSCF energy: %f\n" % emc)

# DMET with every orbital in the impurity, which makes the embedding exact.
ints = LocalIntegrals(mf, list(range(mol.nao_nr())), 'meta_lowdin')
frags = make_fragments(mol, ints, [[0]])
dmet = DMET(ints, frags, False, method='CASSCF', ncas=NCAS, nelecas=NELECAS,
            sa_nstates=NSTATES, sc_method='NONE', cas_spin=0, cas_spin_shift=SHIFT,
            casscf_kwargs={'conv_tol': 1e-11})

class KnownValues(unittest.TestCase):

    def test_pyscf(self):
        self.assertAlmostEqual(mf.e_tot, -37.676865648283, 5)
        self.assertAlmostEqual(mc.e_tot, -37.619728136538, 5)

        # The CAS holds the requested spin, and both roots are singlets.
        rdm1 = mc.fcisolver.make_rdm12(mc.ci, mc.ncas, mc.nelecas)[0]
        self.assertAlmostEqual(np.trace(rdm1), NELECAS, 8)
        for ci in mc.ci:
            self.assertAlmostEqual(
                pyscf.fci.spin_op.spin_square(ci, NCAS, (1, 1))[0], 0.0, 8)

    def test_prism(self):

        e_dmet = dmet.oneshot()

        self.assertAlmostEqual(e_dmet, -37.619728136538, 5)

        e_states = dmet.cas_results[0]['e_states']
        for state in range(NSTATES):
            self.assertAlmostEqual(e_states[state], mc.e_states[state], 8)

if __name__ == "__main__":
    print("DMET-CASSCF singlet-CAS test")
    unittest.main()
