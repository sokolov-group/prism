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
from pyscf import ao2mo
from prism.dmet import DMET, LocalIntegrals, make_fragments

np.set_printoptions(linewidth=150, edgeitems=10, suppress=True)

mol = pyscf.gto.Mole()
mol.atom = [['N', (0, 0, 0)], ['N', (0, 0, 1.5)]]
mol.basis = 'sto-3g'
mol.build()

# Density-fitted RHF: the cluster integrals are kept three-index.
mf = pyscf.scf.RHF(mol).density_fit()
ehf = mf.scf()
print("SCF energy: %f\n" % ehf)

mc = pyscf.mcscf.CASSCF(mf, 4, 4)
mc.conv_tol = 1e-11
emc = mc.mc1step()[0]
print("CASSCF energy: %f\n" % emc)

ints = LocalIntegrals(mf, list(range(mol.nao_nr())), 'meta_lowdin')
frags = make_fragments(mol, ints, [[0], [1]])
dmet = DMET(ints, frags, False, method='CASSCF', ncas=4, nelecas=4,
            sc_method='NONE', casscf_kwargs={'conv_tol': 1e-11})

class KnownValues(unittest.TestCase):

    def test_pyscf(self):
        self.assertAlmostEqual(mf.e_tot, -107.272673070244, 5)
        self.assertAlmostEqual(mc.e_tot, -107.500551868052, 5)

    def test_prism(self):

        e_dmet = dmet.oneshot()

        self.assertAlmostEqual(e_dmet,   -107.505073289743, 5)

        # The three-index integrals contract back to the four-index ones.
        norb = dmet.dmet_orbs[0].shape[1]
        cderi = ints.dmet_cderi(dmet.dmet_orbs[0], norb)
        tei = ints.dmet_tei(dmet.dmet_orbs[0], norb)
        self.assertAlmostEqual(
            np.abs(ao2mo.restore(1, tei, norb)
                   - ao2mo.restore(1, ao2mo.restore(4, cderi.T @ cderi, norb), norb)).max(),
            0.0, 10)

if __name__ == "__main__":
    print("DMET-CASSCF density-fitting test")
    unittest.main()
