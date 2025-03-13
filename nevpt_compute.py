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

import sys
import numpy as np
from functools import reduce

import prism.nevpt_amplitudes as nevpt_amplitudes
import prism.lib.logger as logger

def kernel(nevpt):

    cput0 = (logger.process_clock(), logger.perf_counter())
    nevpt.log.info("\nComputing NEVPT energies...\n")

    # Print general information
    nevpt.log.info("Method:                                            %s" % nevpt.method)
#    nevpt.log.info("Number of MR-ADC roots requested:                  %d" % nevpt.nroots)
    nevpt.log.info("Reference wavefunction type:                       %s" % nevpt.interface.reference)
    nevpt.log.info("Reference state active-space energy:         %20.12f" % nevpt.e_ref_cas)
    nevpt.log.info("Nuclear repulsion energy:                    %20.12f" % nevpt.enuc)
    nevpt.log.info("Reference state spin multiplicity:                 %s" % str(nevpt.ref_wfn_spin_mult))
    nevpt.log.info("Number of basis functions:                         %d" % nevpt.nmo)
    nevpt.log.info("Number of core orbitals:                           %d" % nevpt.ncore)
    nevpt.log.info("Number of active orbitals:                         %d" % nevpt.ncas)
    nevpt.log.info("Number of external orbitals:                       %d" % nevpt.nextern)
    nevpt.log.info("Number of electrons:                               %d" % nevpt.nelec)
    nevpt.log.info("Number of active electrons:                        %s" % str(nevpt.ref_nelecas))
    nevpt.log.info("Overlap truncation parameter (singles):            %e" % nevpt.s_thresh_singles)
    nevpt.log.info("Overlap truncation parameter (doubles):            %e" % nevpt.s_thresh_doubles)
    nevpt.log.info("Compute singles amplitudes?                        %s" % str(nevpt.compute_singles_amplitudes))
    if nevpt.compute_singles_amplitudes:
        nevpt.log.info("Projector for the semi-internal ampltiudes:        %s" % nevpt.semi_internal_projector)

#    # Print info about CASCI states
#    nevpt.log.info("Number of CASCI states:                            %d" % nevpt.ncasci)
#
#    if nevpt.e_cas_ci is not None:
#        nevpt.log.extra("CASCI excitation energies (eV):                    %s" % str(27.2114*(nevpt.e_cas_ci - nevpt.e_cas)))

    nevpt.log.debug("Temporary directory path: %s" % nevpt.temp_dir)

    # Compute amplitudes and correlation energy
    e_tot, e_corr = nevpt_amplitudes.compute_amplitudes(nevpt)

#    nevpt.log.note("\n%s-%s excitation energies (a.u.):" % (nevpt.method_type, nevpt.method))
#    print(E.reshape(-1, 1))
#    nevpt.log.note("\n%s-%s excitation energies (eV):" % (nevpt.method_type, nevpt.method))
#    E_ev = E * 27.2114
#    print(E_ev.reshape(-1, 1))
#    sys.stdout.flush()

    nevpt.log.info("\n------------------------------------------------------------------------------")
    nevpt.log.timer0("total NEVPT calculation", *cput0)

    return e_tot, e_corr

