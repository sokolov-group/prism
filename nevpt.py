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
# Authors: Carlos E. V. de Moura <carlosevmoura@gmail.com>
#          Alexander Yu. Sokolov <alexander.y.sokolov@gmail.com>
#

import prism.nevpt_integrals as nevpt_integrals
import prism.nevpt_compute as nevpt_compute

class NEVPT:
    def __init__(self, interface):

        # General info
        self.interface = interface
        self.log = interface.log
        log = self.log

        log.info("Initializing state-specific fully internally contracted NEVPT...")

        if (interface.reference not in ("casscf", "casci", "sa-casscf", "ms-casci")):
            log.info("The NEVPT code does not support %s reference" % interface.reference)
            raise Exception("The NEVPT code does not support %s reference" % interface.reference)

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
        self.ref_nelecas = interface.ref_nelecas
        self.e_ref = interface.e_ref              # Total reference energy
        self.e_ref_cas = interface.e_ref_cas      # Reference active-space energy
        self.ref_wfn = interface.ref_wfn          # Reference wavefunction
        self.ref_wfn_spin_mult = interface.ref_wfn_spin_mult

        # NEVPT specific variables
        self.method = "nevpt2"                    # Possible methods: nevpt2
        self.nfrozen = None                       # Number of lowest-energy (core) orbitals that will be left uncorrelated ("frozen core")
        self.compute_singles_amplitudes = False   # Include singles amplitudes in the NEVPT2 energy?
        self.semi_internal_projector = "gno"      # Possible values: gno, gs, only matters when compute_singles_amplitudes is True
        self.s_thresh_singles = 1e-10
        self.s_thresh_doubles = 1e-10

        self.S12 = lambda:None                    # Matrices for orthogonalization of excitation spaces

        self.outcore_expensive_tensors = True     # Store expensive (ooee) integrals and amplitudes on disk

        # Integrals
        self.mo_energy = lambda:None
        self.h1eff = lambda:None
        self.v2e = lambda:None
        self.rdm = lambda:None
        self.t1 = lambda:None

        self.mo_energy.c = interface.mo_energy[:self.ncore]
        self.mo_energy.e = interface.mo_energy[self.nocc:]

    def kernel(self):

        log = self.log
        self.method = self.method.lower()

        if self.method not in ("nevpt2"):
            msg = "Unknown method %s" % self.method
            log.info(msg)
            raise Exception(msg)

        if self.nfrozen is None:
            self.nfrozen = 0

        if self.nfrozen > self.ncore:
            msg = "The number of frozen orbitals cannot exceed the number of core orbitals"
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
