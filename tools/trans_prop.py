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
# Authors: Carlos E. V. de Moura <carlosevmoura@gmail.com>
#          Alexander Yu. Sokolov <alexander.y.sokolov@gmail.com>
#          James D. Serna <jserna456@gmail.com>
#          Donna H. Odhiambo <donna.odhiambo@proton.me>
#          Bryce Pickett <brycepickett22@outlook.com>
#

import os
import sys
import numpy as np

def osc_strength(interface, e_diff, trdm_mo):
    '''
    Computes oscillator strengths between states given
    an interface object (source of MO coefficients
    and dipole moment integrals), an array of excitation energies,
    and a transition density matrix.
    '''
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
    '''
    Prints table of given list of oscillator strengths.
    '''
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

def compute_dyson(interface, X):
    '''
    Computes Dyson orbitals given an interface object (source of MO coefficients
    and PySCF mol object), and an array of spectroscopic amplitudes.
    '''
    interface.log.note("\nComputing Dyson orbitals...")
    dyson_mos = np.dot(interface.mo, X)

    filename = os.path.basename(sys.argv[0])
    name = os.path.splitext(filename)[0]
    interface.molden.from_mo(interface.mol, f'{name}_dyson.molden', dyson_mos)

    interface.log.note(f"Dyson orbitals written to {name}_dyson.molden")

    return dyson_mos

def compute_ntos(interface, trdm, initial_state=0, target_state=1):
    '''
    Computes natural transition orbitals between two given states
    given an interface object (source of MO coefficients
    and PySCF mol object) and a transition density matrix.
    '''
    ## Needs mask dependent on nto_thresh

    interface.log.info(f"\nComputing NTOs...")

    U, s, Vh =  np.linalg.svd(trdm, full_matrices = False)
    V = Vh.T
    weights = s**2

    assert np.allclose(U.T @ U, np.eye(U.shape[1])), "U is not unitary"
    assert np.allclose(Vh @ Vh.T, np.eye(Vh.shape[0])), "Vh is not unitary"

    # NTO Metrics
    omega = np.sum(weights)                # sum of NTO occupation numbers
    w = weights / omega                    # normalized weights
    PR = 1.0 / np.sum(w**2)                # NTO participation ratio
    S = -np.sum(w * np.log(w + 1e-16))     # entanglement entropy
    Z = np.exp(S)                          # number of entangled states

    interface.log.info(f"State {initial_state} -> State {target_state}:")
    interface.log.info(f"   Sum of SVs (Omega):               {omega: .6f}")
    interface.log.info(f"   Participation ratio (PR_NTO):     {PR: .6f}")
    interface.log.info(f"   Entanglement entropy (S_HE):      {S: .6f}")
    interface.log.info(f"   Nr of entangled states (Z_HE):    {Z: .6f}")
    interface.log.info(f"   Renormalized S_HE/Z_HE:  {S/Z:.6f} / {1.0: .6f}")

    # MO to AO
    C_hole = interface.mo @ U
    C_particle = interface.mo @ V
    n_nto = C_hole.shape[1]

    # phase consistency (relative to largest-magnitude AO coefficient)
    for k in range(n_nto):
        idx = np.argmax(np.abs(C_hole[:, k]))
        if C_hole[idx, k] < 0:
            C_hole[:, k] *= -1
            C_particle[:, k] *= -1

    # Interleave pairs: [hole_0, particle_0, hole_1, particle_1, ...]
    C_nto = np.zeros((C_hole.shape[0], 2*n_nto))
    C_nto[:, 0::2] = C_hole
    C_nto[:, 1::2] = C_particle
    occ_nto = np.repeat(weights, 2)

    # Save to molden
    input_file = os.path.splitext(os.path.basename(sys.argv[0]))[0]
    filename = f"{input_file}_nto_S{initial_state}_S{target_state}.molden"
    with open(filename, "w") as f:
        interface.molden.header(interface.mol, f)
        interface.molden.orbital_coeff(interface.mol, f, C_nto, occ=occ_nto)

    interface.log.note(f"NTOs written to {filename}")

    return weights, U, Vh
