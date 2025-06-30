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
import prism.nevpt_rdms as nevpt_rdms
import prism.nevpt2 as nevpt2
import prism.qd_nevpt2 as qd_nevpt2

def kernel(nevpt):

    cput0 = (logger.process_clock(), logger.perf_counter())
    nevpt.log.info("\nComputing NEVPT energies...\n")

    n_states = len(nevpt.ref_wfn_deg)
    n_micro_states = sum(nevpt.ref_wfn_deg)

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

    nevpt.log.info("Reference density fitting?                         %s" % ref_df)
    nevpt.log.info("Correlation density fitting?                       %s" % df)
    nevpt.log.info("Temporary directory path:                          %s" % nevpt.temp_dir)

    nevpt.log.info("\nInternal contraction:                              %s" % "Full (= Partial)")
    nevpt.log.info("Compute singles amplitudes?                        %s" % str(nevpt.compute_singles_amplitudes))
    nevpt.log.info("Overlap truncation parameter (singles):            %e" % nevpt.s_thresh_singles)
    nevpt.log.info("Overlap truncation parameter (doubles):            %e" % nevpt.s_thresh_doubles)
    if nevpt.compute_singles_amplitudes:
        nevpt.log.info("Projector for the semi-internal amplitudes:        %s" % nevpt.semi_internal_projector)

    # State-specific NEVPT calculation
    e_tot = []
    e_corr = []
    mstate = 0

    e_0 = 0.0
    t1_0 = None
    t1 = []

    ncore = nevpt.ncore - nevpt.nfrozen

    if ncore > 0 and nevpt.nextern > 0:
        e_0, t1_0 = nevpt2.compute_t1_0(nevpt)
    else:
        t1_0 = np.zeros((ncore, ncore, nevpt.nextern, nevpt.nextern))

    for state in range(n_states):
        deg = nevpt.ref_wfn_deg[state]

        nevpt.log.info("\nComputing energy of state #%d..." % (state + 1))
        nevpt.log.info("Reference state active-space energy:         %20.12f" % nevpt.e_ref_cas[state])
        nevpt.log.info("Reference state spin multiplicity:                 %d" % nevpt.ref_wfn_spin_mult[state])
        nevpt.log.info("Number of active electrons:                        %s" % str(nevpt.ref_nelecas[mstate:(mstate+deg)]))

        # Compute reduced density matrices for a specific state
        rdms = nevpt_rdms.compute_reference_rdms(nevpt, nevpt.ref_wfn[mstate:(mstate+deg)], nevpt.ref_nelecas[mstate:(mstate+deg)])

        # Compute amplitudes and correlation energy
        e_corr_state, t1_state = nevpt2.compute_energy(nevpt, rdms, e_0)
        e_tot_state = nevpt.e_ref[state] + e_corr_state

        ref_name = nevpt.interface.reference.upper()
        method_name = "NEVPT2"
        nevpt.log.info("%s reference state total energy: %s  %20.12f" % (ref_name.upper(), (12-len(ref_name)) * " ", nevpt.e_ref[state]))
        nevpt.log.info("%s correlation energy:           %s  %20.12f" % (method_name, (12-len(method_name)) * " ", e_corr_state))
        nevpt.log.info("Total %s energy:                 %s  %20.12f" % (method_name, (12-len(method_name)) * " ", e_tot_state))

        e_corr.append(e_corr_state)
        e_tot.append(e_tot_state)

        if nevpt.method == "qd-nevpt2":
            t1.append(t1_state)
        else:
            del(t1_state)

        del(rdms)

        mstate += deg

    # Quasidegenerate NEVPT2 calculation
    if nevpt.method == "qd-nevpt2":
        nevpt.log.info("\nComputing the QD-NEVPT2 effective Hamiltonian...")

        # Compute and diagonalize the QD-NEVPT2 effective Hamiltonian
        e_tot, h_evec = qd_nevpt2.compute_energy(nevpt, e_tot, t1, t1_0)

        # Update correlation energies
        for state in range(n_states):
            e_corr[state] = e_tot[state] - nevpt.e_ref[state]

        # Return QDNEVPT2 total energies
    else:
        del(t1_0)

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
            if p == 0 or abs(de) < 1e-5:
                nevpt.log.info("%5d       %2d      %20.12f %14.8f %12.4f %12s %14.4f" % ((p+1), nevpt.ref_wfn_spin_mult[p], e_tot[p], de, de_ev, " ", de_cm))
            else:
                de_nm = 10000000 / de_cm
                nevpt.log.info("%5d       %2d      %20.12f %14.8f %12.4f %12.4f %14.4f" % ((p+1), nevpt.ref_wfn_spin_mult[p], e_tot[p], de, de_ev, de_nm, de_cm))

        nevpt.log.info("------------------------------------------------------------------------------------------------")
    
        # Oscillator Strengths
        num_states = len(e_tot) # total states
        col_width = 18  # characters per column

        # Collect all transitions per ground state
        transition_data = [[] for _ in range(num_states - 1)]  # last state has no transitions

        for i in range(num_states - 1):
            osc_str = osc_strength(nevpt, e_tot, h_evec, gs_index=i, ncore=ncore) 
            for j in range(i + 1, num_states):
                f_ij = osc_str[j - i - 1]
                f_val_str = f"{f_ij:.8f}"
                transition_data[i].append(f"{i + 1} -> {j + 1}: {f_val_str}")

        # Compute max rows needed and total line width
        max_len = max(len(col) for col in transition_data)
        total_line_width = (col_width + 4) * len(transition_data) # Needed for adjusting printout

        # Print header and transitions
        separator = "-" * total_line_width
        nevpt.log.info("\n\nOscillator Strengths: state i -> state f")
        nevpt.log.info(separator)

        # Final print
        for row in range(max_len):
            row_data = []
            for col in transition_data:
                if row < len(col):
                    row_data.append(col[row].ljust(col_width))
                else:
                    row_data.append("".ljust(col_width))
            nevpt.log.info("    ".join(row_data))

        nevpt.log.info(separator)

    
    sys.stdout.flush()
    nevpt.log.timer0("total %s calculation" % nevpt.method.upper(), *cput0)

    return e_tot, e_corr

def osc_strength(nevpt, en, evec, gs_index = 0, ncore = None):
    if ncore == None: ncore = nevpt.ncore # Flag for frozen core

    n_micro_states = sum(nevpt.ref_wfn_deg)
    dip_mom_ao = nevpt.interface.dip_mom_ao
    
    if nevpt.pe == True:
        dip_mom_ao = nevpt.ind

    mo_coeff = nevpt.mo
    nmo = nevpt.nmo
    ncas = nevpt.ncas
    nevpt.evec = evec
    dip_mom_mo = np.zeros_like(dip_mom_ao)

    # Transform dipole moments from AO to MO basis
    for d in range(dip_mom_ao.shape[0]):
        dip_mom_mo[d] = mo_coeff.T @ dip_mom_ao[d] @ mo_coeff

    # List to store Osc. Strength Values
    osc_total = []

    # Looping over CAS States
    for state in range(gs_index + 1, n_micro_states):
        # Reset final transformed RDM
        rdm_qd = np.zeros((nmo, nmo))

        # Looping over states I,J
        for I in range(n_micro_states):
            for J in range(n_micro_states):
                rdm_mo = np.zeros((nmo, nmo))  # Reset RDM in MO Basis   
                trdm_ca = nevpt.interface.compute_rdm1(nevpt.ref_wfn[I], nevpt.ref_wfn[J], nevpt.ref_nelecas[I])
                rdm_mo[ncore:ncore + ncas ,ncore:ncore + ncas] = trdm_ca

                if I == J:
                    rdm_mo[:ncore, :ncore] = 2 * np.eye(nevpt.ncore)
                    rdm_qd += np.conj(evec)[I, state] * rdm_mo * evec[J, gs_index]
                else:
                    rdm_qd += np.conj(evec)[I, state] * rdm_mo * evec[J, gs_index]

        # Create Dipole Moment Operator with RDM
        dip_evec_x = np.einsum('pq,pq', dip_mom_mo[0], rdm_qd)
        dip_evec_y = np.einsum('pq,pq', dip_mom_mo[1], rdm_qd)
        dip_evec_z = np.einsum('pq,pq', dip_mom_mo[2], rdm_qd)
 
        osc_x = ((2/3)*(en[state] - en[gs_index]))*(np.conj(dip_evec_x)*dip_evec_x)
        osc_y = ((2/3)*(en[state] - en[gs_index]))*(np.conj(dip_evec_y)*dip_evec_y)
        osc_z = ((2/3)*(en[state] - en[gs_index]))*(np.conj(dip_evec_z)*dip_evec_z)

        # Add Dipole Moment Components
        osc_total.append((osc_x + osc_y + osc_z).real)

    return osc_total

