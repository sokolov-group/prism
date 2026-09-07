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


class FragmentBuilder:

    _NEEDS_DM_METHODS = frozenset({'CASSCF', 'QD-NEVPT2', 'PC-NEVPT2'})
    # CASSCF is absent by design: its solver computes no transition properties, so the
    # dipole integrals would be built and then discarded.
    _NEEDS_DIP_METHODS = frozenset({'QD-NEVPT2', 'PC-NEVPT2'})

    def __init__(self, ints, helper, fragments, method, num_bath_orbs, bath_tol,
                 fragment_methods=None, core_occ_tol=None):
        self._ints = ints
        self._helper = helper
        self._fragments = fragments
        self._method = method
        self._num_bath_orbs = num_bath_orbs
        self._bath_tol = bath_tol
        self._fragment_methods = fragment_methods or {}
        self._core_occ_tol = core_occ_tol

    def build(self, counter, one_rdm, chempot_imp):
        fragment_mask = self._fragments[counter]
        flag_rhf = self._fragment_methods.get(counter) == 'RHF'
        impurity_orbs = np.abs(fragment_mask)
        num_imp_orbs = int(np.sum(impurity_orbs))
        bath_request = num_imp_orbs if self._num_bath_orbs is None else self._num_bath_orbs[counter]

        num_bath_orbs, loc_2_dmet, core_1rdm_dmet = self._helper.construct_bath(
            one_rdm, impurity_orbs, bath_request, threshold=self._bath_tol)

        # Loose core-occupation cutoff when the bath is auto-sized, tight when fixed.
        # core_occ_tol overrides that coupling so bath sizing can be varied on its own.
        core_cutoff = (self._core_occ_tol if self._core_occ_tol is not None
                       else (0.01 if self._num_bath_orbs is None else 0.5))
        for idx, occ in enumerate(core_1rdm_dmet):
            if occ < core_cutoff:
                core_1rdm_dmet[idx] = 0.0
            elif occ > 2.0 - core_cutoff:
                core_1rdm_dmet[idx] = 2.0
            else:
                raise RuntimeError(
                    f"Fragment {counter}, environment orbital {idx} has occupation "
                    f"{occ:.6f}, which is neither near 0 nor 2 (cutoff={core_cutoff}). "
                    f"The bath size may be too small."
                )

        norb_in_imp = num_imp_orbs + num_bath_orbs
        nelec_in_imp = int(round(self._ints.nelec - np.sum(core_1rdm_dmet)))
        core_1rdm_loc = loc_2_dmet @ np.diag(core_1rdm_dmet) @ loc_2_dmet.T

        if norb_in_imp > self._ints.norb:
            raise RuntimeError(
                f"Cluster size norb_in_imp={norb_in_imp} exceeds norb={self._ints.norb}.")

        dmet_oei = self._ints.dmet_oei(loc_2_dmet, norb_in_imp)
        dmet_fock = self._ints.dmet_fock(loc_2_dmet, norb_in_imp, core_1rdm_loc)
        dmet_tei = self._ints.dmet_tei(loc_2_dmet, norb_in_imp)

        needs_dm = flag_rhf or self._method in self._NEEDS_DM_METHODS
        dm_guess_rhf = None
        if needs_dm:
            dm_guess_rhf = self._ints.dmet_init_guess_rhf(
                loc_2_dmet, norb_in_imp, nelec_in_imp // 2,
                num_imp_orbs, chempot_imp)

        method_key = 'flag_rhf' if flag_rhf else self._method

        dip_mom_ao = None
        if self._method in self._NEEDS_DIP_METHODS:
            dip_mom_ao = self._ints.dmet_dip_mom(loc_2_dmet, norb_in_imp)

        return {
            'counter': counter,
            'flag_rhf': flag_rhf,
            'impurity_orbs': impurity_orbs,
            'num_imp_orbs': num_imp_orbs,
            'norb_in_imp': norb_in_imp,
            'nelec_in_imp': nelec_in_imp,
            'loc_2_dmet': loc_2_dmet,
            'core_1rdm_loc': core_1rdm_loc,
            'core_1rdm_dmet': core_1rdm_dmet,
            'dmet_oei': dmet_oei,
            'dmet_fock': dmet_fock,
            'dmet_tei': dmet_tei,
            'dm_guess_rhf': dm_guess_rhf,
            'dip_mom_ao': dip_mom_ao,
            'method_key': method_key,
        }

    def build_symmetry_bath(self, counter, one_rdm, sym_parent):
        impurity_orbs = np.abs(self._fragments[counter])
        num_imp_orbs = int(np.sum(impurity_orbs))
        bath_request = num_imp_orbs if self._num_bath_orbs is None else self._num_bath_orbs[counter]
        num_bath_orbs, loc_2_dmet, _ = self._helper.construct_bath(
            one_rdm, impurity_orbs, bath_request, threshold=self._bath_tol)
        norb_in_imp = num_imp_orbs + num_bath_orbs
        return {
            'counter': counter,
            'sym_parent': sym_parent,
            'impurity_orbs': impurity_orbs,
            'norb_in_imp': norb_in_imp,
            'loc_2_dmet': loc_2_dmet,
        }
