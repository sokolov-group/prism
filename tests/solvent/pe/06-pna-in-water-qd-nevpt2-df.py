#!/usr/bin/env python3
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
#          James D. Serna <jserna456@gmail.com>
 
import unittest
import numpy as np
import math
import pyscf.gto
import pyscf.scf
import pyscf.mcscf
import prism.interface
import prism.nevpt
import importlib.util

np.set_printoptions(linewidth=150, edgeitems=10, suppress=True)

HAS_CPPE = importlib.util.find_spec("cppe") is not None

mol = pyscf.gto.Mole()
mol.atom = '''
N     29.660000    27.490000    28.690000
C     29.850000    28.660000    29.350000
C     31.000000    28.870000    30.210000
C     31.170000    30.170000    30.870000
C     30.170000    31.240000    30.650000
C     29.000000    31.020000    29.780000
C     28.840000    29.750000    29.140000
N     30.340000    32.580000    31.300000
O     31.300000    32.810000    32.020000
O     29.510000    33.460000    31.110000
H     30.320000    26.750000    28.790000
H     28.860000    27.380000    28.090000
H     31.730000    28.090000    30.370000
H     32.020000    30.340000    31.520000
H     28.260000    31.810000    29.640000
H     27.980000    29.580000    28.490000
'''

# Matching PE potential basis
mol.basis = "cc-pVDZ"
mol.build()

# Polarizable Embedding
if HAS_CPPE:
    from pyscf.solvent.pol_embed import PolEmbed
    pe_options = {"potfile": "potential_files/pna-water-3.5-angstrom.pot"} # Given .pot file
    pe = PolEmbed(mol, pe_options)
    pe.verbose = 3

# RHF calculation
mf = pyscf.scf.RHF(mol)

if HAS_CPPE:
    mf = pyscf.solvent.PE(mf, pe).density_fit('cc-pvdz-jkfit')

ehf = mf.scf()
print("SCF energy: %f\n" % ehf)

# CASSCF calculation
n_states = 4
weights = np.ones(n_states)/n_states
mc = pyscf.mcscf.CASSCF(mf, 4,4).state_average_(weights).density_fit('cc-pvdz-jkfit')

if HAS_CPPE:
    mc = pyscf.solvent.PE(mc, pe)

emc = mc.mc1step()[0]

@unittest.skipUnless(HAS_CPPE, "Skipping PE tests: CPPE not installed")
class KnownValues(unittest.TestCase):

    def test_pyscf(self):
        self.assertAlmostEqual(mc.e_tot,  -489.122700009413, 6)
        self.assertAlmostEqual(mc.e_cas,  -2.92151347576964, 6)

    def test_prism(self):

        # QD-NEVPT2 calculation
        interface = prism.interface.PYSCF(mf, mc, backend = 'opt_einsum').density_fit('cc-pvdz-ri')
        nevpt = prism.nevpt.NEVPT(interface)
        nevpt.compute_singles_amplitudes = False
        nevpt.s_thresh_singles = 1e-6
        nevpt.s_thresh_doubles = 1e-6
        nevpt.pe = pe
        nevpt.method_type = "qd"

        e_tot, e_corr, osc = nevpt.kernel()

        self.assertAlmostEqual(e_tot[0], -490.618394011249, 5)
        self.assertAlmostEqual(e_tot[1], -490.490814346064, 5)
        self.assertAlmostEqual(e_tot[2], -490.454267845287, 5)
        self.assertAlmostEqual(e_tot[3], -490.439521488995, 5)
        
        self.assertAlmostEqual(e_corr[0], -1.4987071829185, 5)
        self.assertAlmostEqual(e_corr[1], -1.4968277461907, 5)
        self.assertAlmostEqual(e_corr[2], -1.5044411397183, 5)
        self.assertAlmostEqual(e_corr[3], -1.4939552785395, 5)
        
        self.assertAlmostEqual(osc[0], 0.0, 5)
        self.assertAlmostEqual(osc[1], 0.0, 5)
        self.assertAlmostEqual(osc[2], 0.0, 5)
        
if __name__ == "__main__":
    print("QD-NEVPT2 test")
    unittest.main()
       
