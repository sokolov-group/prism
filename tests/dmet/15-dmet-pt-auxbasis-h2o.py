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
import prism.mr_adc
from prism.dmet import DMET, LocalIntegrals, make_fragments

# A small fitting basis, so its error stands well above the convergence noise.
AUXBASIS = 'sto-3g'

H2O = [['O', (0.0, 0.0, 0.0)], ['H', (0.0, 0.757, 0.587)], ['H', (0.0, -0.757, 0.587)]]
HF = [['H', (0.0, 0.0, 0.0)], ['F', (0.0, 0.0, 0.917)]]


def mean_field(atom, density_fit):
    mol = pyscf.gto.M(atom=atom, basis='sto-3g', verbose=0)
    mf = pyscf.scf.RHF(mol)
    if density_fit:
        mf = mf.density_fit()
    mf.verbose = 0
    mf.scf()
    return mf


def casscf(mf, ncas, nstates):
    mc = pyscf.mcscf.CASSCF(mf, ncas, ncas)
    if nstates > 1:
        mc = mc.state_average_([1.0 / nstates] * nstates)
    mc.conv_tol = 1e-11
    mc.verbose = 0
    mc.kernel()
    return mc


def embed(mf, method, ncas, nstates, **kw):
    # The whole molecule is the impurity, which makes the embedding exact.
    mol = mf.mol
    ints = LocalIntegrals(mf, list(range(mol.nao_nr())), 'meta_lowdin')
    dmet = DMET(ints, make_fragments(mol, ints, [list(range(mol.natm))]), False,
                method=method, ncas=ncas, nelecas=ncas, sa_nstates=nstates,
                sc_method='NONE', casscf_kwargs={'conv_tol': 1e-11}, verbose=0, **kw)
    dmet.oneshot()
    return dmet


class KnownValues(unittest.TestCase):

    def assertSameReference(self, mc_emb, mc):
        # The embedded energy leaves out the nuclear repulsion.
        self.assertAlmostEqual(mc_emb.e_tot + mc.mol.energy_nuc(), mc.e_tot, 8)

    def test_qdnevpt2_df_reference(self):
        mf = mean_field(H2O, True)
        mc = casscf(mf, 4, 2)
        e_direct = prism.nevpt.QDNEVPT(
            prism.interface.PYSCF(mf, mc).density_fit(AUXBASIS)).kernel()[0]
        res = embed(mf, 'QD-NEVPT2', 4, 2,
                    qdnevpt2_kwargs={'auxbasis': AUXBASIS}).qdnevpt2_results[0]
        self.assertSameReference(res['mc'], mc)
        e_dmet = res['e_tot']
        self.assertAlmostEqual(e_dmet[1] - e_dmet[0], e_direct[1] - e_direct[0], 6)

    def test_pcnevpt2_exact_reference(self):
        mf = mean_field(H2O, False)
        mc = casscf(mf, 4, 2)
        e_corr = prism.nevpt.NEVPT(
            prism.interface.PYSCF(mf, mc).density_fit(AUXBASIS)).kernel()[1]
        res = embed(mf, 'PC-NEVPT2', 4, 2,
                    pcnevpt2_kwargs={'auxbasis': AUXBASIS}).pcnevpt2_results[0]
        self.assertSameReference(res['mc'], mc)
        for state in range(2):
            self.assertAlmostEqual(res['e_corr'][state], e_corr[state], 6)

    def test_pcnevpt2_auxbasis_replaces_the_mean_field_fit(self):
        mf = mean_field(H2O, True)
        e_mf_fit = embed(mf, 'PC-NEVPT2', 4, 2).pcnevpt2_results[0]['e_corr'][0]
        e_pt_fit = embed(mf, 'PC-NEVPT2', 4, 2, pcnevpt2_kwargs={'auxbasis': AUXBASIS})
        e_pt_fit = e_pt_fit.pcnevpt2_results[0]['e_corr'][0]
        self.assertGreater(abs(e_pt_fit - e_mf_fit), 1e-4)

    def test_mradc_df_reference(self):
        mf = mean_field(HF, True)
        mc = casscf(mf, 2, 1)
        mr_adc = prism.mr_adc.MRADC(prism.interface.PYSCF(mf, mc).density_fit(AUXBASIS))
        mr_adc.method_type = 'cvs-ip'
        mr_adc.ncvs = 1
        mr_adc.nroots = 4
        e_direct = np.atleast_1d(mr_adc.kernel()[0])
        res = embed(mf, 'MR-ADC', 2, 1, mradc_kwargs={'method_type': 'cvs-ip', 'ncvs': 1,
                                                       'nroots': 4, 'auxbasis': AUXBASIS})
        res = res.mradc_results[0]
        self.assertSameReference(res['mc'], mc)
        e_dmet = np.atleast_1d(res['e_exc'])
        # Ionization energies in eV.
        for root in range(4):
            self.assertAlmostEqual(e_dmet[root], e_direct[root], 6)


if __name__ == "__main__":
    print("DMET perturbation-theory auxiliary basis\n")
    unittest.main()
