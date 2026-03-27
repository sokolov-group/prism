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
# Authors: Alexander Yu. Sokolov <alexander.y.sokolov@gmail.com>
#          Carlos E. V. de Moura <carlosevmoura@gmail.com>

import unittest
import importlib
import numpy as np
import pyscf.gto
import pyscf.scf
import pyscf.mcscf
import prism.interface
import prism.nevpt

np.set_printoptions(linewidth=150, edgeitems=10, suppress=True)

r = 1.1508

mol = pyscf.gto.Mole()
mol.atom = [
            ['N', (0.0, 0.0, 0.0)],
            ['O', (0.0, 0.0,   r)]]
mol.basis = 'cc-pcvdz'
mol.symmetry = True
mol.spin = 0
mol.charge = +1
mol.build()

# RHF calculation
mf = pyscf.scf.RHF(mol)
mf.max_cycle = 250
mf.conv_tol = 1e-12

ehf = mf.scf()
print("SCF energy: %f\n" % ehf)

mol.spin = 1
mol.charge = 0
mol.build()

# CASSCF calculation
mc = pyscf.mcscf.CASSCF(mf, 9, 7)
mc.max_cycle = 150
mc.conv_tol = 1e-8
mc.conv_tol_grad = 1e-5
mc.fcisolver.wfnsym = 'E1x'

emc = mc.mc1step()[0]
print("CASSCF energy: %f\n" % emc)

# NEVPT2 calculation
interface = prism.interface.PYSCF(mf, mc, backend = 'opt_einsum').density_fit('cc-pvdz-ri')
nevpt_1 = prism.nevpt.NEVPT(interface)
nevpt_1.compute_singles_amplitudes = False
nevpt_1.semi_internal_projector = "gno"
nevpt_1.s_thresh_singles = 1e-6
nevpt_1.s_thresh_doubles = 1e-10

# NEVPT2 calculation
interface = prism.interface.PYSCF(mf, mc, backend = 'pytblis').density_fit('cc-pvdz-ri')
nevpt_2 = prism.nevpt.NEVPT(interface)
nevpt_2.compute_singles_amplitudes = False
nevpt_2.semi_internal_projector = "gno"
nevpt_2.s_thresh_singles = 1e-6
nevpt_2.s_thresh_doubles = 1e-10

# NEVPT2 calculation
interface = prism.interface.PYSCF(mf, mc, backend = 'numpy').density_fit('cc-pvdz-ri')
nevpt_3 = prism.nevpt.NEVPT(interface)
nevpt_3.compute_singles_amplitudes = False
nevpt_3.semi_internal_projector = "gno"
nevpt_3.s_thresh_singles = 1e-6
nevpt_3.s_thresh_doubles = 1e-10

class KnownValues(unittest.TestCase):

    def test_pyscf(self):
        self.assertAlmostEqual(mc.e_tot, -129.402145048918, 6)
        self.assertAlmostEqual(mc.e_cas,  -17.587633940679, 6)

    def _check_values(self, e_tot, e_corr):

        self.assertAlmostEqual(e_tot[0], -129.649975169028, 6)
        self.assertAlmostEqual(e_corr[0],  -0.247830120110, 6)

    def test_prism_opt_einsum(self):
        if importlib.util.find_spec('opt_einsum') is None:
            self.skipTest('opt_einsum is not available')
        e_tot, e_corr, osc = nevpt_1.kernel()
        self._check_values(e_tot, e_corr)

    def test_prism_pytblis(self):
        if importlib.util.find_spec('pytblis') is None:
            self.skipTest('pytblis is not available')
        e_tot, e_corr, osc = nevpt_2.kernel()
        self._check_values(e_tot, e_corr)

    def test_prism_numpy(self):
        e_tot, e_corr, osc = nevpt_3.kernel()
        self._check_values(e_tot, e_corr)

if __name__ == "__main__":
    print("NEVPT2 test")
    unittest.main()
