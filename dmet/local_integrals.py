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
from pyscf import ao2mo, lo
from pyscf.lo import nao, orth
from pyscf.tools import molden

from prism.dmet import iao_helper


class LocalIntegrals:

    def __init__(self, mf, active_orbs, localization_type,
                 ao_rotation=None, localization_threshold=1e-6):
        if localization_type not in ('meta_lowdin', 'boys', 'lowdin', 'iao'):
            raise ValueError(
                f"Unknown localization_type='{localization_type}'. "
                f"Valid: 'meta_lowdin', 'boys', 'lowdin', 'iao'")

        self.mol = mf.mol
        self.mf = mf
        self.e_hf = mf.e_tot
        _dm = mf.make_rdm1()
        _hcore = mf.get_hcore()
        self.ovlp = self.mol.intor_symmetric('int1e_ovlp')
        # ROHF is spin-resolved in the density but keeps one set of orbitals, so the
        # density and the Fock build are selected on different quantities.
        self.full_dm_ao = _dm[0] + _dm[1] if _dm.ndim == 3 else _dm
        if np.asarray(mf.mo_coeff).ndim == 3:  # UHF/UKS
            # F = S C eps C.T S: rotation-invariant, avoids BLAS noise in get_veff.
            SC_a = np.dot(self.ovlp, mf.mo_coeff[0])
            SC_b = np.dot(self.ovlp, mf.mo_coeff[1])
            fock_a = SC_a @ np.diag(mf.mo_energy[0]) @ SC_a.T
            fock_b = SC_b @ np.diag(mf.mo_energy[1]) @ SC_b.T
            self.full_fock_ao = 0.5 * (fock_a + fock_b)
        else:  # RHF/ROHF/RKS
            SC = np.dot(self.ovlp, mf.mo_coeff)
            self.full_fock_ao = SC @ np.diag(mf.mo_energy) @ SC.T
        self.full_jk_ao = self.full_fock_ao - _hcore

        self._which = localization_type
        self.active = np.zeros((self.mol.nao_nr(),), dtype=int)
        self.active[active_orbs] = 1
        self.norb = np.sum(self.active)
        _mo_occ = mf.mo_occ
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
        self.nelec = int(np.rint(self.mol.nelectron - _frozen_elec))

        if self._which in ('meta_lowdin', 'boys'):
            if self._which == 'meta_lowdin':
                if self.norb != self.mol.nao_nr():
                    raise ValueError(
                        "Invalid active_orbs for meta_lowdin: this localization "
                        "requires the full active space.")
            if self._which == 'boys':
                if mf.mo_coeff.ndim == 3:
                    raise NotImplementedError(
                        "Boys localization requires a single set of spatial orbitals. "
                        "Use 'meta_lowdin' or 'iao' for UHF references."
                    )
                self.ao2loc = mf.mo_coeff[:, self.active == 1]
            if self.norb == self.mol.nao_nr():
                # Be (Z=4) needs an explicit valence shell for meta_lowdin to span the minimal
                # basis; touch the global table only for Be and restore it after.
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
                loc = lo.Boys(self.mol, self.ao2loc)
                loc.conv_tol = localization_threshold
                self.ao2loc = loc.kernel()
            self.ti_ok = False
        if self._which == 'lowdin':
            if self.norb != self.mol.nao_nr():
                raise ValueError(
                    "Invalid active_orbs for lowdin: this localization "
                    "requires the full active space.")
            ovlp_eigs, ovlp_vecs = np.linalg.eigh(self.ovlp)
            self.ao2loc = ovlp_vecs @ np.diag(np.power(ovlp_eigs, -0.5)) @ ovlp_vecs.T
            self.ti_ok = False
        if self._which == 'iao':
            if self.norb != self.mol.nao_nr():
                raise ValueError(
                    "Invalid active_orbs for iao: this localization "
                    "requires the full active space.")
            # ao2loc is BLAS-order sensitive for near-degenerate HOMO/LUMO; pin num_threads.
            self.ao2loc = iao_helper.localize_iao(self.mol, mf)
            if ao_rotation is not None:
                self.ao2loc = np.dot(self.ao2loc, ao_rotation.T)
            self.ti_ok = False
        if self.loc_ortho() >= 1e-8:
            raise RuntimeError("LMO basis is not orthonormal")

        if _mo_occ.ndim == 2:
            self.frozen_dm_ao = np.zeros_like(self.full_dm_ao)
            self.frozen_jk_ao = np.zeros_like(self.full_jk_ao)
        else:
            self.frozen_dm_mo = _mo_occ.copy()
            self.frozen_dm_mo[self.active == 1] = 0
            self.frozen_dm_ao = mf.mo_coeff @ np.diag(self.frozen_dm_mo) @ mf.mo_coeff.T
            _v_frozen = mf.get_veff(self.mol, self.frozen_dm_ao)
            self.frozen_jk_ao = _v_frozen[0] if _v_frozen.ndim == 3 else _v_frozen
        self.frozen_oei_ao = self.full_fock_ao - self.full_jk_ao + self.frozen_jk_ao

        self.active_const = mf.energy_nuc() + np.einsum(
            'ij,ij->', self.frozen_oei_ao - 0.5 * self.frozen_jk_ao, self.frozen_dm_ao)
        self.active_oei = self.ao2loc.T @ self.frozen_oei_ao @ self.ao2loc
        self.active_fock = self.ao2loc.T @ self.full_fock_ao @ self.ao2loc
        if self.norb <= 150:
            self.eri_in_mem = True
            self.active_eri = ao2mo.outcore.full_iofree(self.mol, self.ao2loc, compact=False).reshape(
                self.norb, self.norb, self.norb, self.norb)
        else:
            self.eri_in_mem = False
            self.active_eri = None

    def molden(self, filename):
        with open(filename, 'w') as the_file:
            molden.header(self.mol, the_file)
            molden.orbital_coeff(self.mol, the_file, self.ao2loc)

    def loc_ortho(self):
        return np.linalg.norm(self.ao2loc.T @ self.ovlp @ self.ao2loc - np.eye(self.norb))

    def const(self):
        return self.active_const

    def loc_oei(self):
        return self.active_oei

    def loc_fock(self, dm_loc=None):
        if dm_loc is None:
            return self.active_fock
        if not self.eri_in_mem:
            dm_ao = self.ao2loc @ dm_loc @ self.ao2loc.T
            _v_ao = self.mf.get_veff(self.mol, dm_ao)
            jk_ao = _v_ao[0] if _v_ao.ndim == 3 else _v_ao
            jk_loc = self.ao2loc.T @ jk_ao @ self.ao2loc
        else:
            jk_loc = (np.einsum('ijkl,ij->kl', self.active_eri, dm_loc)
                      - 0.5 * np.einsum('ijkl,ik->jl', self.active_eri, dm_loc))
        return self.active_oei + jk_loc

    def loc_tei(self):
        if not self.eri_in_mem:
            raise RuntimeError("ERIs are not stored in memory.")
        return self.active_eri

    def dmet_oei(self, loc_2_dmet, num_active):
        return loc_2_dmet[:, :num_active].T @ self.active_oei @ loc_2_dmet[:, :num_active]

    def dmet_fock(self, loc_2_dmet, num_active, core_dm_loc):
        return loc_2_dmet[:, :num_active].T @ self.loc_fock(core_dm_loc) @ loc_2_dmet[:, :num_active]

    def dmet_dip_mom(self, loc_2_dmet, num_active):
        # Dipole integrals over the molecule, in the embedded basis.
        transfo = self.ao2loc @ loc_2_dmet[:, :num_active]
        dip_mom_ao = self.mol.intor_symmetric('int1e_r', comp=3)
        dip_mom_emb = np.zeros((dip_mom_ao.shape[0], num_active, num_active))
        for d in range(dip_mom_ao.shape[0]):
            dip_mom_emb[d] = transfo.T @ dip_mom_ao[d] @ transfo
        return dip_mom_emb

    def dmet_soc_data(self, loc_2_dmet, num_active, core_1rdm_loc):
        # Molecule, embedded orbitals and frozen density for spin-orbit integrals.
        ao2emb = self.ao2loc @ loc_2_dmet[:, :num_active]
        core_dm_ao = self.ao2loc @ core_1rdm_loc @ self.ao2loc.T
        return {'mol': self.mol, 'ao2emb': ao2emb, 'core_dm_ao': core_dm_ao}

    def dmet_init_guess_rhf(self, loc_2_dmet, num_active, num_pairs, nimp, chempot_imp):
        fock_emb = loc_2_dmet[:, :num_active].T @ self.active_fock @ loc_2_dmet[:, :num_active]
        if chempot_imp != 0.0:
            for orb in range(nimp):
                fock_emb[orb, orb] -= chempot_imp
        eigvals, eigvecs = np.linalg.eigh(fock_emb)
        eigvecs = eigvecs[:, eigvals.argsort()]
        return 2 * np.dot(eigvecs[:, :num_pairs], eigvecs[:, :num_pairs].T)

    def dmet_tei(self, loc_2_dmet, num_active):
        if not self.eri_in_mem:
            transfo = np.dot(self.ao2loc, loc_2_dmet[:, :num_active])
            return ao2mo.outcore.full_iofree(self.mol, transfo, compact=False).reshape(
                num_active, num_active, num_active, num_active)
        return ao2mo.incore.full(
            ao2mo.restore(8, self.active_eri, self.norb), loc_2_dmet[:, :num_active], compact=False
        ).reshape(num_active, num_active, num_active, num_active)
