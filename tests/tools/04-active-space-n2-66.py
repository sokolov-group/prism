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
# Authors: Bryce Pickett <brycepickett22@outlook.com>
#

import unittest
import numpy as np
import pyscf.gto
import pyscf.scf
import pyscf.mcscf
from pyscf.mcscf import apc
import prism.interface
import prism.nevpt
from prism.tools import active_space

mol = pyscf.gto.Mole()
mol.atom = [
            ['N', (0.0, 0.0, 0.0)],
            ['N', (0.0, 0.0, 1.8)]]
mol.basis = 'cc-pvdz'
mol.build()

# RHF calculation
mf = pyscf.scf.RHF(mol)
mf.conv_tol = 1e-12

ehf = mf.scf()
print("SCF energy: %f\n" % ehf)

# CASSCF calculation
mc = pyscf.mcscf.CASSCF(mf, 6, 6)
mc.conv_tol = 1e-11
mc.conv_tol_grad = 1e-6

emc = mc.mc1step()[0]
print("CASSCF energy: %f\n" % emc)

# PRISM interface and NEVPT2
interface = prism.interface.PYSCF(mf, mc)
nevpt = prism.nevpt.NEVPT(interface)
nevpt.rdm_order = 2
nevpt.kernel()

def casci_energy(ncas, nelecas, mo):
    return pyscf.mcscf.CASCI(mf, ncas, nelecas).kernel(mo)[0]

class KnownValues(unittest.TestCase):

    def test_ncsf(self):
        self.assertEqual(active_space.get_ncsf(8, 8), 1764)
        self.assertEqual(active_space.get_ncsf(10, 10), 19404)
        self.assertEqual(active_space.get_ncsf(12, 12), 226512)
        self.assertEqual(active_space.get_ncsf(6, 7), 490)

    def test_reference(self):

        # Reference active space within the limit
        ncas, nelecas, mo = active_space.select(interface, (8, 8))
        self.assertEqual((ncas, nelecas), (6, (3, 3)))
        self.assertAlmostEqual(casci_energy(ncas, nelecas, mo), mc.e_tot, 8)

        for criterion in ('noon', 'entropy', 'mi'):
            ncas, nelecas, mo = active_space.select(interface, (4, 4), criterion)
            self.assertEqual((ncas, nelecas), (4, (2, 2)))
            self.assertAlmostEqual(casci_energy(ncas, nelecas, mo), -108.723110497900, 6)

    def test_apc(self):

        ncas, nelecas, mo = active_space.select(interface, (4, 4), 'apc')
        ncas_ref, nelecas_ref, mo_ref = apc.APC(mf, (4, 4), verbose = 0).kernel()
        self.assertEqual((ncas, nelecas), (ncas_ref, nelecas_ref))
        e_ref = casci_energy(ncas_ref, nelecas_ref, mo_ref)
        self.assertAlmostEqual(casci_energy(ncas, nelecas, mo), e_ref, 8)

    def test_nevpt2(self):

        ncas, nelecas, mo = active_space.select(nevpt, (6, 6))
        self.assertEqual((ncas, nelecas), (6, (3, 3)))
        self.assertAlmostEqual(casci_energy(ncas, nelecas, mo), -108.819172859800, 6)
        ovlp = mo.T @ mf.get_ovlp() @ mo
        self.assertAlmostEqual(abs(ovlp - np.eye(ovlp.shape[0])).max(), 0.0, 8)

        with self.assertRaises(ValueError):
            active_space.select(nevpt, (6, 6), 'entropy')

if __name__ == "__main__":
    print("Active space selection test")
    unittest.main()
