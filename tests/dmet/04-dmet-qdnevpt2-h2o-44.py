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

np.set_printoptions(linewidth=150, edgeitems=10, suppress=True)

mol = pyscf.gto.Mole()
mol.atom = [['O', (0.0,  0.0,   0.0)],
            ['H', (0.0,  0.757, 0.587)],
            ['H', (0.0, -0.757, 0.587)]]
mol.basis = 'sto-3g'
mol.verbose = 4
mol.build()

# RHF calculation
mf = pyscf.scf.RHF(mol)
ehf = mf.scf()
print("SCF energy: %f\n" % ehf)

# Direct QD-NEVPT2 on the whole molecule
mc = pyscf.mcscf.CASSCF(mf, 4, 4).state_average_([0.25, 0.25, 0.25, 0.25])
mc.conv_tol = 1e-11
emc = mc.mc1step()[0]
print("CASSCF energy: %f\n" % emc)

nevpt = prism.nevpt.QDNEVPT(prism.interface.PYSCF(mf, mc))

# DMET with the whole molecule in the impurity, which makes the embedding exact
ints = LocalIntegrals(mf, list(range(mol.nao_nr())), 'meta_lowdin')
frags = make_fragments(mol, ints, [[0, 1, 2]])
dmet = DMET(ints, frags, False, method='QD-NEVPT2', ncas=4, nelecas=4,
            sa_nstates=4, sc_method='NONE', casscf_kwargs={'conv_tol': 1e-11})

class KnownValues(unittest.TestCase):

    def test_pyscf(self):
        self.assertAlmostEqual(mf.e_tot, -74.963063129728, 5)
        self.assertAlmostEqual(mc.e_tot,  -74.646210586425, 5)

    def test_prism(self):

        e_direct = nevpt.kernel()[0]
        dmet.oneshot()
        e_dmet = dmet.qdnevpt2_results[0]['e_tot']

        self.assertAlmostEqual(e_direct[0], -74.999764883101, 5)
        self.assertAlmostEqual(e_dmet[0],   -84.188023311229, 5)

        # The embedded energies leave out the nuclear repulsion, so the totals differ
        # by a constant and the excitation energies are what can be compared.
        for state in range(1, 4):
            de_direct = e_direct[state] - e_direct[0]
            de_dmet = e_dmet[state] - e_dmet[0]
            self.assertAlmostEqual(de_dmet, de_direct, 5)

if __name__ == "__main__":
    print("DMET-QD-NEVPT2 test")
    unittest.main()
