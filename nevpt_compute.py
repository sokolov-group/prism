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

import sys
import numpy as np
from functools import reduce

import prism.lib.logger as logger
import prism.nevpt_amplitudes as nevpt_amplitudes
import prism.nevpt_rdms as nevpt_rdms

def kernel(nevpt):

    cput0 = (logger.process_clock(), logger.perf_counter())
    nevpt.log.info("\nComputing NEVPT energies...\n")

    n_states = len(nevpt.ref_wfn_spin_mult)
    n_micro_states = sum(nevpt.ref_wfn_spin_mult)

    ref_df = False
    df = False
    if nevpt.interface.reference_df:
        ref_df = True
    if nevpt.interface.with_df:
        df = True

    # Print general information
    nevpt.log.info("Method:                                            %s" % nevpt.method)
    nevpt.log.info("Nuclear repulsion energy:                    %20.12f" % nevpt.enuc)
    nevpt.log.info("Number of electrons:                               %d" % nevpt.nelec)
    nevpt.log.info("Number of basis functions:                         %d" % nevpt.nmo)
    nevpt.log.info("Reference wavefunction type:                       %s" % nevpt.interface.reference)
    nevpt.log.info("Number of reference states:                        %d" % n_states)
    nevpt.log.info("Number of reference microstates:                   %d" % n_micro_states)
    nevpt.log.info("Number of frozen orbitals:                         %d" % nevpt.nfrozen)
    nevpt.log.info("Number of core orbitals:                           %d" % nevpt.ncore)
    nevpt.log.info("Number of active orbitals:                         %d" % nevpt.ncas)
    nevpt.log.info("Number of external orbitals:                       %d" % nevpt.nextern)
    if nevpt.compute_singles_amplitudes:
        nevpt.log.info("Projector for the semi-internal amplitudes:        %s" % nevpt.semi_internal_projector)
    nevpt.log.info("Reference density fitting?                         %s" % ref_df)
    nevpt.log.info("Correlation density fitting?                       %s" % df)
    nevpt.log.info("Temporary directory path:                          %s" % nevpt.temp_dir)

    nevpt.log.info("\nInternal contraction:                              %s" % "Full (equivalent to partial)")
    nevpt.log.info("Compute singles amplitudes?                        %s" % str(nevpt.compute_singles_amplitudes))
    nevpt.log.info("Overlap truncation parameter (singles):            %e" % nevpt.s_thresh_singles)
    nevpt.log.info("Overlap truncation parameter (doubles):            %e" % nevpt.s_thresh_doubles)

    e_tot = []
    e_corr = []
    mstate = 0
    for state in range(n_states):
        spin_mult = nevpt.ref_wfn_spin_mult[state]

        nevpt.log.info("\nComputing energy of state #%d..." % (state + 1))
        nevpt.log.info("Reference state active-space energy:         %20.12f" % nevpt.e_ref_cas[state])
        nevpt.log.info("Reference state spin multiplicity:                 %d" % nevpt.ref_wfn_spin_mult[state])
        nevpt.log.info("Number of active electrons:                        %s" % str(nevpt.ref_nelecas[mstate:(mstate+spin_mult)]))

        # Compute reduced density matrices for a specific state
        nevpt_rdms.compute_reference_rdms(nevpt, nevpt.ref_wfn[mstate:(mstate+spin_mult)], nevpt.ref_nelecas[mstate:(mstate+spin_mult)])

        # Compute amplitudes and correlation energy
        e_corr_state = nevpt_amplitudes.compute_amplitudes(nevpt)
        e_tot_state = nevpt.e_ref[state] + e_corr_state

        ref_name = nevpt.interface.reference.upper()
        method_name = nevpt.method.upper()
        nevpt.log.info("%s reference state total energy: %s  %20.12f" % (ref_name.upper(), (12-len(ref_name)) * " ", nevpt.e_ref[state]))
        nevpt.log.info("%s correlation energy:           %s  %20.12f" % (method_name.upper(), (12-len(method_name)) * " ", e_corr_state))
        nevpt.log.info("Total %s energy:                 %s  %20.12f" % (method_name.upper(), (12-len(method_name)) * " ", e_tot_state))

        e_corr.append(e_corr_state)
        e_tot.append(e_tot_state)

        mstate += spin_mult

    if n_states > 1:
        h2ev = nevpt.interface.hartree_to_ev
        h2cm = nevpt.interface.hartree_to_inv_cm

        nevpt.log.info("\nSummary of results for the %s calculation with the %s reference:" % (nevpt.method.upper(), nevpt.interface.reference.upper()))

        nevpt.log.info("------------------------------------------------------------------------------------------------")
        nevpt.log.info("  State    (2S+1)         E(total)            dE(a.u.)        dE(eV)      dE(nm)       dE(cm-1)")
        nevpt.log.info("------------------------------------------------------------------------------------------------")

        e_gs = e_tot[0]

        for p in range(n_states):
            de = e_tot[p] - e_gs
            de_ev = de * h2ev
            de_cm = de * h2cm
            if p == 0:
                nevpt.log.info("%5d       %2d      %20.12f %14.8f %12.4f %12s %14.4f" % ((p+1), nevpt.ref_wfn_spin_mult[p], e_tot[p], de, de_ev, " ", de_cm))
            else:
                de_nm = 10000000 / de_cm
                nevpt.log.info("%5d       %2d      %20.12f %14.8f %12.4f %12.4f %14.4f" % ((p+1), nevpt.ref_wfn_spin_mult[p], e_tot[p], de, de_ev, de_nm, de_cm))

        nevpt.log.info("------------------------------------------------------------------------------------------------")

    sys.stdout.flush()

    nevpt.log.timer0("total %s calculation" % nevpt.method.upper(), *cput0)

    return e_tot, e_corr

