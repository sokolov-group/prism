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
N      1.131000     0.328000     0.042000
C     -0.012000     1.109000     0.048000
O      0.076000     2.331000     0.105000
C     -1.251000     0.373000    -0.014000
C     -2.528000     1.131000    -0.012000
C     -1.187000    -0.970000    -0.071000
N     -0.003000    -1.650000    -0.072000
C      1.215000    -1.040000    -0.016000
O      2.278000    -1.644000    -0.018000
H      2.013000     0.818000     0.084000
H     -2.066000    -1.587000    -0.119000
H      0.002000    -2.656000    -0.116000
H     -2.614000     1.735000     0.886000
H     -2.572000     1.809000    -0.860000
H     -3.375000     0.456000    -0.061000
'''

# Matching PE potential basis
mol.basis = "cc-pVDZ"
mol.build()

# Polarizable Embedding
if HAS_CPPE:
    from pyscf.solvent.pol_embed import PolEmbed
    pe_options = {"potfile": "potential_files/thy-water-3.5-angstrom.pot"} # Given .pot file
    pe = PolEmbed(mol, pe_options)
    pe.verbose = 3

# RHF calculation
mf = pyscf.scf.RHF(mol)

if HAS_CPPE:
    mf = pyscf.solvent.PE(mf, pe)

ehf = mf.scf()
print("SCF energy: %f\n" % ehf)

# CASSCF calculation
n_states = 4
weights = np.ones(n_states)/n_states
mc = pyscf.mcscf.CASSCF(mf, 4,4).state_average_(weights)

if HAS_CPPE:
    mc = pyscf.solvent.PE(mc, pe)
    
mc.fix_spin_(ss=0)

emc = mc.mc1step()[0]

@unittest.skipUnless(HAS_CPPE, "Skipping PE tests: CPPE not installed")
class KnownValues(unittest.TestCase):

    def test_pyscf(self):
        self.assertAlmostEqual(mc.e_tot,  -451.327033632280,  6)
        self.assertAlmostEqual(mc.e_cas,  -2.93745393826350, 6)

    def test_prism(self):

        # QD-NEVPT2 calculation
        interface = prism.interface.PYSCF(mf, mc, backend = 'opt_einsum')
        nevpt = prism.nevpt.NEVPT(interface)
        nevpt.compute_singles_amplitudes = False
        nevpt.s_thresh_singles = 1e-6
        nevpt.s_thresh_doubles = 1e-6
        nevpt.pe = pe
        nevpt.method_type = "qd"

        e_tot, e_corr, osc = nevpt.kernel()

        self.assertAlmostEqual(e_tot[0], -453.371594330412,  5)
        self.assertAlmostEqual(e_tot[1], -453.175981170054,  5)
        self.assertAlmostEqual(e_tot[2], -453.144343702507, 5)
        self.assertAlmostEqual(e_tot[3], -453.103832273319, 5)
        
        self.assertAlmostEqual(e_corr[0], -1.3598435603033, 5)
        self.assertAlmostEqual(e_corr[1], -1.4380762196864, 5)
        self.assertAlmostEqual(e_corr[2], -1.4294794088207, 5)
        self.assertAlmostEqual(e_corr[3], -1.4054995377015, 5)
        
        self.assertAlmostEqual(osc[0], 0.53406794, 5)
        self.assertAlmostEqual(osc[1], 0.13834022, 5)
        self.assertAlmostEqual(osc[2], 0.00127043, 5)
        
if __name__ == "__main__":
    print("QD-NEVPT2 test")
    unittest.main()
       
