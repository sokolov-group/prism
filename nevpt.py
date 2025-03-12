# Copyright 2023 Prism Developers. All Rights Reserved.
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
#

import prism.nevpt_integrals as nevpt_integrals
import prism.nevpt_rdms as nevpt_rdms
import prism.nevpt_compute as nevpt_compute

class NEVPT:
    def __init__(self, interface):

        # General info
        self.interface = interface
        self.log = interface.log
        log = self.log

        log.info("\nInitializing state-specific fully internally contracted NEVPT...")

        if (interface.reference not in ("casscf", "sa-casscf")):
            log.info("NEVPT requires CASSCF reference")
            raise Exception("NEVPT requires CASSCF reference")

        self.stdout = interface.stdout
        self.verbose = interface.verbose
        self.max_memory = interface.max_memory
        self.current_memory = interface.current_memory

        self.temp_dir = interface.temp_dir
        self.tmpfile = lambda:None

        self.mo = interface.mo
        self.mo_scf = interface.mo_scf
        self.ovlp = interface.ovlp
        self.nmo = interface.nmo
        self.nelec = interface.nelec
        self.enuc = interface.enuc
        self.e_scf = interface.e_scf

        self.symmetry = interface.symmetry
        self.group_repr_symm = interface.group_repr_symm

        # CASSCF specific
        self.ncore = interface.ncore
        self.ncas = interface.ncas
        self.nextern = interface.nextern
        self.nocc = self.ncas + self.ncore
        self.nelecas = interface.nelecas
        self.e_casscf = interface.e_casscf      # Total reference CASSCF energy
        self.e_cas = interface.e_cas            # Reference active-space CASSCF energy
        self.wfn_casscf = interface.wfn_casscf  # Reference CASSCF wavefunction
        self.wfn_casscf_spin_square = interface.wfn_casscf_spin_square
        self.wfn_casscf_spin = interface.wfn_casscf_spin
        self.wfn_casscf_spin_mult = interface.wfn_casscf_spin_mult

        # NEVPT specific variables
        self.method = "nevpt2"          # Possible methods: nevpt2
        self.compute_singles_amplitudes = False
        self.semi_internal_projector = "gno" # Possible values: gno, gs, only matters when compute_singles_amplitudes is True
        self.s_thresh_singles = 1e-10
        self.s_thresh_doubles = 1e-10

        self.S12 = lambda:None          # Matrices for orthogonalization of excitation spaces

        self.outcore_expensive_tensors = True # Store expensive (ooee) integrals and amplitudes on disk

        # Integrals
        self.mo_energy = lambda:None
        self.h1eff = lambda:None
        self.v2e = lambda:None
        self.rdm = lambda:None
        self.t1 = lambda:None


    def kernel(self):

        log = self.log
        self.method = self.method.lower()

        if self.method not in ("nevpt2"):
            msg = "Unknown method %s" % self.method
            log.error(msg)
            raise Exception(msg)

        # Transform one- and two-electron integrals
        log.info("\nTransforming integrals to MO basis...")
        nevpt_integrals.transform_integrals_1e(self)
        if self.interface.with_df:
            nevpt_integrals.transform_Heff_integrals_2e_df(self)
            nevpt_integrals.transform_integrals_2e_df(self)
        else:
            # TODO: this actually handles out-of-core integrals too, rename the function
            nevpt_integrals.transform_integrals_2e_incore(self)

        # Compute CASCI energies and reduced density matrices
        nevpt_rdms.compute_reference_rdms(self)

        # Run NEVPT computation
        e_tot, e_corr = nevpt_compute.kernel(self)

        return e_tot, e_corr

    @property
    def verbose(self):
        return self._verbose
    @verbose.setter
    def verbose(self, obj):
        self._verbose = obj
        self.log.verbose = obj
