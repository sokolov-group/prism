# Copyright 2026 Prism Developers. All Rights Reserved.
# Adapted from QC-DMET (Copyright 2015 Sebastian Wouters)
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
#

import numpy as np
from pyscf import gto, scf, ao2mo, lo
from pyscf.lo import nao, orth
from pyscf.tools import molden

from prism.dmet import iao_helper


class LocalIntegrals:

    def __init__(self, the_mf, active_orbs, localizationtype,
                 ao_rotation=None, use_full_hessian=True,
                 localization_threshold=1e-6):
        assert localizationtype in ('meta_lowdin', 'boys', 'lowdin', 'iao')

        self.mol     = the_mf.mol
        self.the_mf  = the_mf
        self.fullEhf = the_mf.e_tot
        _dm = the_mf.make_rdm1()
        _hcore = the_mf.get_hcore()
        _S = self.mol.intor_symmetric('int1e_ovlp')
        if _dm.ndim == 3:  # UHF/UKS
            self.fullDMao       = _dm[0] + _dm[1]
            self.fullDMao_alpha = _dm[0]
            self.fullDMao_beta  = _dm[1]
            # F = S C eps C.T S: invariant to degenerate-subspace eigenvector
            # rotation; avoids BLAS non-determinism in get_veff ERI contraction.
            SC_a = _S @ the_mf.mo_coeff[0]
            SC_b = _S @ the_mf.mo_coeff[1]
            self.fullFOCKao_alpha = SC_a @ np.diag(the_mf.mo_energy[0]) @ SC_a.T
            self.fullFOCKao_beta  = SC_b @ np.diag(the_mf.mo_energy[1]) @ SC_b.T
            self.fullFOCKao = 0.5 * (self.fullFOCKao_alpha + self.fullFOCKao_beta)
            self.fullJKao   = self.fullFOCKao - _hcore
        else:  # RHF/RKS
            self.fullDMao       = _dm
            self.fullDMao_alpha = _dm / 2.0
            self.fullDMao_beta  = _dm / 2.0
            SC = _S @ the_mf.mo_coeff
            self.fullFOCKao = SC @ np.diag(the_mf.mo_energy) @ SC.T
            self.fullJKao   = self.fullFOCKao - _hcore
            self.fullFOCKao_alpha = self.fullFOCKao
            self.fullFOCKao_beta  = self.fullFOCKao

        _with_df = getattr(the_mf, 'with_df', None)
        self.use_density_fit = _with_df is not None
        self.df_auxbasis     = getattr(_with_df, 'auxbasis', None)

        self._which  = localizationtype
        self.active  = np.zeros([self.mol.nao_nr()], dtype=int)
        self.active[active_orbs] = 1
        self.Norbs   = np.sum(self.active)
        _mo_occ = the_mf.mo_occ
        _frozen_mask = self.active == 0
        if _mo_occ.ndim == 2:  # UHF: sum frozen electrons from both spin channels
            if np.any(_frozen_mask):
                raise NotImplementedError(
                    "Frozen-core UHF references are not supported. "
                    "Pass active_orbs=list(range(mol.nao_nr()))."
                )
            _frozen_elec = 0.0
        else:
            _frozen_elec = np.sum(_mo_occ[_frozen_mask])
        self.Nelec = int(np.rint(self.mol.nelectron - _frozen_elec))

        if self._which in ('meta_lowdin', 'boys'):
            if self._which == 'meta_lowdin':
                assert self.Norbs == self.mol.nao_nr(), "meta_lowdin requires full active space"
            if self._which == 'boys':
                if the_mf.mo_coeff.ndim == 3:
                    raise NotImplementedError(
                        "Boys localization requires a single set of spatial orbitals. "
                        "Use 'meta_lowdin' or 'iao' for UHF references."
                    )
                self.ao2loc = the_mf.mo_coeff[:, self.active == 1]
            if self.Norbs == self.mol.nao_nr():
                # Be (Z=4) needs an explicit valence-shell entry for meta_lowdin to span
                # the minimal basis; only touch the global table when Be is present, and
                # restore it after so other molecules in the same process are unaffected.
                has_be = 4 in self.mol.atom_charges()
                if has_be:
                    _aoshell_be_saved = nao.AOSHELL[4]
                    nao.AOSHELL[4] = ['1s0p0d0f', '2s1p0d0f']
                try:
                    self.ao2loc = orth.orth_ao(self.mol, 'meta_lowdin')
                finally:
                    if has_be:
                        nao.AOSHELL[4] = _aoshell_be_saved
                if ao_rotation is not None:
                    self.ao2loc = np.dot(self.ao2loc, ao_rotation.T)
            if self._which == 'boys':
                old_verbose = self.mol.verbose
                self.mol.verbose = 5
                loc = lo.Boys(self.mol, self.ao2loc)
                loc.conv_tol = localization_threshold
                self.mol.verbose = old_verbose
                self.ao2loc = loc.kernel()
            self.TI_OK = False
        if self._which == 'lowdin':
            assert self.Norbs == self.mol.nao_nr(), "lowdin requires full active space"
            ovlp = self.mol.intor_symmetric('int1e_ovlp')
            ovlp_eigs, ovlp_vecs = np.linalg.eigh(ovlp)
            self.ao2loc = np.dot(np.dot(ovlp_vecs, np.diag(np.power(ovlp_eigs, -0.5))), ovlp_vecs.T)
            self.TI_OK  = False
        if self._which == 'iao':
            assert self.Norbs == self.mol.nao_nr(), "iao requires full active space"
            # ao2loc is non-deterministic when BLAS swaps near-degenerate HOMO/LUMO;
            # pin num_threads before mf.kernel() to suppress (not guaranteed for very tight gaps).
            self.ao2loc = iao_helper.localize_iao(self.mol, the_mf)
            if ao_rotation is not None:
                self.ao2loc = np.dot(self.ao2loc, ao_rotation.T)
            self.TI_OK = False
        assert self.loc_ortho() < 1e-8, "LMO basis is not orthonormal"

        if _mo_occ.ndim == 2:
            self.frozenDMao = np.zeros_like(self.fullDMao)
            self.frozenJKao = np.zeros_like(self.fullJKao)
        else:
            self.frozenDMmo  = _mo_occ.copy()
            self.frozenDMmo[self.active == 1] = 0
            self.frozenDMao  = the_mf.mo_coeff @ np.diag(self.frozenDMmo) @ the_mf.mo_coeff.T
            _v_frozen        = the_mf.get_veff(self.mol, self.frozenDMao)
            self.frozenJKao  = _v_frozen[0] if _v_frozen.ndim == 3 else _v_frozen
        self.frozenOEIao = self.fullFOCKao - self.fullJKao + self.frozenJKao

        self.activeCONST = the_mf.energy_nuc() + np.einsum(
            'ij,ij->', self.frozenOEIao - 0.5 * self.frozenJKao, self.frozenDMao)
        self.activeOEI  = np.dot(np.dot(self.ao2loc.T, self.frozenOEIao), self.ao2loc)
        self.activeFOCK = np.dot(np.dot(self.ao2loc.T, self.fullFOCKao), self.ao2loc)
        if self.Norbs <= 150:
            self.ERIinMEM  = True
            self.activeERI = ao2mo.outcore.full_iofree(self.mol, self.ao2loc, compact=False).reshape(
                self.Norbs, self.Norbs, self.Norbs, self.Norbs)
        else:
            self.ERIinMEM  = False
            self.activeERI = None

    def molden(self, filename):
        with open(filename, 'w') as thefile:
            molden.header(self.mol, thefile)
            molden.orbital_coeff(self.mol, thefile, self.ao2loc)

    def loc_ortho(self):
        S = self.mol.intor_symmetric('int1e_ovlp')
        return np.linalg.norm(np.dot(np.dot(self.ao2loc.T, S), self.ao2loc) - np.eye(self.Norbs))

    def const(self):
        return self.activeCONST

    def loc_oei(self):
        return self.activeOEI

    def loc_fock(self, dm_loc=None):
        if dm_loc is None:
            return self.activeFOCK
        if not self.ERIinMEM:
            DM_ao  = np.dot(np.dot(self.ao2loc, dm_loc), self.ao2loc.T)
            _v_ao  = self.the_mf.get_veff(self.mol, DM_ao)
            JK_ao  = _v_ao[0] if _v_ao.ndim == 3 else _v_ao
            JK_loc = np.dot(np.dot(self.ao2loc.T, JK_ao), self.ao2loc)
        else:
            JK_loc = (np.einsum('ijkl,ij->kl', self.activeERI, dm_loc)
                      - 0.5 * np.einsum('ijkl,ik->jl', self.activeERI, dm_loc))
        return self.activeOEI + JK_loc

    def loc_tei(self):
        assert self.ERIinMEM, "local_integrals::loc_tei: ERIs not stored in memory."
        return self.activeERI

    def dmet_oei(self, loc_2_dmet, numActive):
        return np.dot(np.dot(loc_2_dmet[:, :numActive].T, self.activeOEI), loc_2_dmet[:, :numActive])

    def dmet_fock(self, loc_2_dmet, numActive, coreDMloc):
        return np.dot(np.dot(loc_2_dmet[:, :numActive].T, self.loc_fock(coreDMloc)), loc_2_dmet[:, :numActive])

    def dmet_init_guess_rhf(self, loc_2_dmet, numActive, numPairs, nimp, chempot_imp):
        Fock_emb = np.dot(np.dot(loc_2_dmet[:, :numActive].T, self.activeFOCK), loc_2_dmet[:, :numActive])
        if chempot_imp != 0.0:
            for orb in range(nimp):
                Fock_emb[orb, orb] -= chempot_imp
        eigvals, eigvecs = np.linalg.eigh(Fock_emb)
        eigvecs = eigvecs[:, eigvals.argsort()]
        return 2 * np.dot(eigvecs[:, :numPairs], eigvecs[:, :numPairs].T)

    def dmet_tei(self, loc_2_dmet, numAct):
        if not self.ERIinMEM:
            transfo = np.dot(self.ao2loc, loc_2_dmet[:, :numAct])
            return ao2mo.outcore.full_iofree(self.mol, transfo, compact=False).reshape(
                numAct, numAct, numAct, numAct)
        return ao2mo.incore.full(
            ao2mo.restore(8, self.activeERI, self.Norbs), loc_2_dmet[:, :numAct], compact=False
        ).reshape(numAct, numAct, numAct, numAct)
