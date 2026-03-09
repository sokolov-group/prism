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
#          James D. Serna <jserna456@gmail.com>

import numpy as np

# Compute oscillator strengths
# Input: e_diff (excitation energies),
# trdm_mo (transition 1-RDM between reference and excited state)
def osc_strength(interface, e_diff, trdm_mo):

    n_micro_states = len(e_diff) 
    dip_mom_ao = interface.dip_mom_ao
    mo_coeff = interface.mo

    dip_mom_mo = np.zeros_like(dip_mom_ao)

    # Transform dipole moments from AO to MO basis
    for d in range(dip_mom_ao.shape[0]):
        dip_mom_mo[d] = mo_coeff.T @ dip_mom_ao[d] @ mo_coeff

    # List to store Osc. Strength Values
    osc_total = []

    for state in range(n_micro_states):
        # Create Dipole Moment Operator with RDM
        dip_evec_x = np.einsum('pq,pq', dip_mom_mo[0], trdm_mo[state])
        dip_evec_y = np.einsum('pq,pq', dip_mom_mo[1], trdm_mo[state])
        dip_evec_z = np.einsum('pq,pq', dip_mom_mo[2], trdm_mo[state])
        
        osc_x = (2/3)*(e_diff[state])*(np.conj(dip_evec_x) * dip_evec_x)
        osc_y = (2/3)*(e_diff[state])*(np.conj(dip_evec_y) * dip_evec_y)
        osc_z = (2/3)*(e_diff[state])*(np.conj(dip_evec_z) * dip_evec_z)

        osc_total.append((osc_x + osc_y + osc_z).real)
        
    return (np.array(osc_total))


def print_osc_strength(interface, osc_str):

    # Oscillator Strengths
    col_width = 18  # characters per column

    # Collect all transitions per ground state
    transition_data = [[] for _ in range(len(osc_str))] 
    
    for i in range(len(osc_str)):
        transitions = osc_str[i]

        for j in range(len(transitions)):
            f_ij = transitions[j]

            f_val_str = f"{f_ij:.8f}"
            transition_data[i].append(f"{i+1} -> {i+j+2}: {f_val_str}")

    # Compute max rows needed and total line width
    max_len = max(len(col) for col in transition_data)
    total_line_width = (col_width + 4) * len(transition_data) # Needed for adjusting printout

    # Print header and transitions
    separator = "-" * total_line_width
    interface.log.info("\n\nOscillator Strengths: state i -> state f")
    interface.log.info(separator)

    # Final print
    for row in range(max_len):
        row_data = []
        for col in transition_data:
            if row < len(col):
                row_data.append(col[row].ljust(col_width))
            else:
                row_data.append("".ljust(col_width))
        interface.log.info("    ".join(row_data))

    interface.log.info(separator)
