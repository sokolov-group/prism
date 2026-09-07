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
import prism.mr_adc
from prism.dmet import DMET, LocalIntegrals, make_fragments

np.set_printoptions(linewidth=150, edgeitems=10, suppress=True)

mol = pyscf.gto.Mole()
mol.atom = [['H', (0, 0, 0)], ['F', (0, 0, 0.917)]]
mol.basis = 'sto-3g'
mol.build()

# RHF calculation
mf = pyscf.scf.RHF(mol)
ehf = mf.scf()
print("SCF energy: %f\n" % ehf)

# Direct CVS-IP-MR-ADC on the whole molecule
mc = pyscf.mcscf.CASSCF(mf, 2, 2)
mc.conv_tol = 1e-11
emc = mc.mc1step()[0]
print("CASSCF energy: %f\n" % emc)

mr_adc = prism.mr_adc.MRADC(prism.interface.PYSCF(mf, mc))
mr_adc.method_type = "cvs-ip"
mr_adc.ncvs = 1
mr_adc.nroots = 4

# DMET with the whole molecule in the impurity, which makes the embedding exact
ints = LocalIntegrals(mf, list(range(mol.nao_nr())), 'lowdin')
frags = make_fragments(mol, ints, [[0, 1]])
dmet = DMET(ints, frags, False, method='MR-ADC', ncas=2, nelecas=2,
            sc_method='NONE', casscf_kwargs={'conv_tol': 1e-11},
            mradc_kwargs={'method_type': 'cvs-ip', 'ncvs': 1, 'nroots': 4})

class KnownValues(unittest.TestCase):

    def test_pyscf(self):
        self.assertAlmostEqual(mf.e_tot, -98.570779986014, 5)
        self.assertAlmostEqual(mc.e_tot, -98.571528753732, 5)

    def test_prism(self):

        e_direct = np.atleast_1d(mr_adc.kernel()[0])
        dmet.oneshot()
        e_dmet = np.atleast_1d(dmet.mradc_results[0]['e_exc'])

        # Ionization energies in eV, as returned by the Prism MR-ADC kernel.
        self.assertAlmostEqual(e_direct[0], 698.887221032596, 5)
        self.assertAlmostEqual(e_dmet[0],   698.887221032595, 5)

        for root in range(4):
            self.assertAlmostEqual(e_dmet[root], e_direct[root], 8)

if __name__ == "__main__":
    print("DMET-CVS-IP-MR-ADC test")
    unittest.main()
