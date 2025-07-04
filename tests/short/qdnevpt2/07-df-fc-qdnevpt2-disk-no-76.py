# Copyright 2025 Prism Developers. All Rights Reserved.
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
#          James D. Serna <jserna456@gmail.com>

import unittest
import numpy as np
import math
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
mol.basis = 'cc-pvdz'
mol.symmetry = False
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
n_states = 7
weights = np.ones(n_states)/n_states
mc = pyscf.mcscf.CASSCF(mf, 6, 7).state_average_(weights)
mc.conv_tol = 1e-11
mc.conv_tol_grad = 1e-6
emc = mc.mc1step()[0]

# NEVPT2 calculation
interface = prism.interface.PYSCF(mf, mc, opt_einsum = True).density_fit('cc-pvdz-ri')
nevpt = prism.nevpt.NEVPT(interface)
nevpt.compute_singles_amplitudes = False
nevpt.semi_internal_projector = "gno"
nevpt.s_thresh_singles = 1e-8
nevpt.s_thresh_doubles = 1e-8
nevpt.nfrozen = 2
nevpt.max_memory = 1
nevpt.method = "qd-nevpt2"

class KnownValues(unittest.TestCase):

    def test_pyscf(self):
        self.assertAlmostEqual(mc.e_tot, -129.154007811535, 6)
        self.assertAlmostEqual(mc.e_cas, -15.6158947578694, 6)

    def test_prism(self):

        e_tot, e_corr, osc = nevpt.kernel()

        self.assertAlmostEqual(e_tot[0], -129.574440989907, 6)
        self.assertAlmostEqual(e_tot[1], -129.574440989907, 6)
        self.assertAlmostEqual(e_tot[2], -129.330152086338, 6)
        self.assertAlmostEqual(e_tot[3], -129.323515948305, 6)
        self.assertAlmostEqual(e_tot[4], -129.323515948304, 6)
        self.assertAlmostEqual(e_tot[5], -129.287548577760, 6)
        self.assertAlmostEqual(e_tot[6], -129.287548577760, 6)

        self.assertAlmostEqual(e_corr[0], -0.2284568907441, 6)
        self.assertAlmostEqual(e_corr[1], -0.2284568907438, 6)
        self.assertAlmostEqual(e_corr[2], -0.2285260127866, 6)
        self.assertAlmostEqual(e_corr[3], -0.2218898747531, 6)
        self.assertAlmostEqual(e_corr[4], -0.2612698593988, 6)
        self.assertAlmostEqual(e_corr[5], -0.2253024888553, 6)
        self.assertAlmostEqual(e_corr[6], -0.2292064202525, 6)
        
        self.assertAlmostEqual(osc[0], 0.0, 6)
        self.assertAlmostEqual(osc[1], 0.0, 6)
        self.assertAlmostEqual(osc[2], 0.0, 6)
        self.assertAlmostEqual(osc[3], 0.0, 6)
        self.assertAlmostEqual(osc[4] + osc[5], (0.00061126 + 0.00084377), 6)
        
        
if __name__ == "__main__":
    print("QD-NEVPT2 test")
    unittest.main()
