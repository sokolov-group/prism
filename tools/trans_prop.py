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

    input_file = os.path.splitext(os.path.basename(sys.argv[0]))[0]
    filename = f"{input_file}_dyson.molden"

    interface.molden.from_mo(interface.mol, filename, dyson_mos)
    interface.log.note("Dyson orbitals written to %s" % filename)

    return dyson_mos

def compute_ntos(interface, trdm, initial_state=1, target_state=2, orb_thresh=None):
    '''
    Computes natural transition orbitals between two given states
    given an interface object (source of MO coefficients
    and PySCF mol object) and a transition density matrix.
    '''

    interface.log.info("\nComputing NTOs...")

    # Threshold
    if orb_thresh is None:
        orb_thresh = getattr(interface, "orb_thresh", 1e-3)

    U, s, Vh =  np.linalg.svd(trdm, full_matrices = False)

    assert np.allclose(U.T @ U, np.eye(U.shape[1])), "U is not unitary"
    assert np.allclose(Vh @ Vh.T, np.eye(Vh.shape[0])), "Vh is not unitary"

    # Apply threshold
    mask = s**2 > orb_thresh
    weights = s[mask]**2
    U, V = U[:, mask], Vh[mask].T

    if weights.size == 0:
        interface.log.note("No significant NTO weights found for S%s -> S%s" % (initial_state, target_state))
        return None

    # NTO Metrics
    omega = np.sum(weights)                # sum of NTO occupation numbers
    w = weights / omega                    # normalized weights
    PR = 1.0 / np.sum(w**2)                # NTO participation ratio
    S = -np.sum(w * np.log(w + 1e-16))     # entanglement entropy
    Z = np.exp(S)                          # number of entangled states

    interface.log.info("State %s -> State %s" % (initial_state, target_state))
    interface.log.info("   Sum of SVs (Omega):               % .6f" % omega)
    interface.log.info("   Participation ratio (PR_NTO):     % .6f" % PR)
    interface.log.info("   Entanglement entropy (S_HE):      % .6f" % S)
    interface.log.info("   Nr of entangled states (Z_HE):    % .6f" % Z)
    interface.log.info("   Renormalized S_HE/Z_HE:  %.6f / % .6f" % (S/Z, 1.0))

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

    interface.log.note("NTOs written to %s" % filename)

    return weights, U, Vh

def compute_exciton_analysis(interface, trdm, initial_state=1, target_state=2, orb_thresh=None):
    '''
    Computes exciton analysis between two given states
    given an interface object (source of MO coefficients
    and PySCF mol object) and a transition density matrix.
    '''

    interface.log.info("\nComputing exciton analysis...")

    # Threshold
    if orb_thresh is None:
        orb_thresh = getattr(interface, "orb_thresh", 1e-3)

    U, s, Vh =  np.linalg.svd(trdm, full_matrices = False)

    assert np.allclose(U.T @ U, np.eye(U.shape[1])), "U is not unitary"
    assert np.allclose(Vh @ Vh.T, np.eye(Vh.shape[0])), "Vh is not unitary"

    # Apply threshold
    mask = s**2 > orb_thresh
    weights = s[mask]**2

    if weights.size == 0:
        interface.log.note("No significant NTO weights found for S%s -> S%s" % (initial_state, target_state))
        return None

    U, V = U[:, mask], Vh[mask].T
    w = weights / np.sum(weights) # normalized weights

    # Integrals: AO to MO to NTO
    mo_coeff = interface.mo
    R_ao = interface.mol.intor("int1e_r")
    R2_ao = interface.mol.intor("int1e_r2")

    R_mo = np.stack([mo_coeff.T @ R_ao[i] @ mo_coeff for i in range(3)])
    R_h = np.stack([U.T @ R_mo[i] @ U for i in range(3)])
    R_e = np.stack([V.T @ R_mo[i] @ V for i in range(3)])

    R2_mo = mo_coeff.T @ R2_ao @ mo_coeff
    R2_h = U.T @ R2_mo @ U
    R2_e = V.T @ R2_mo @ V

    # Diag
    rh_diag = np.array([np.diag(R_h[i]) for i in range(3)])
    re_diag = np.array([np.diag(R_e[i]) for i in range(3)])

    # mean positions: <r>
    rh = rh_diag @ w
    re = re_diag @ w

    # raw second moments: <r^2>
    r2h = np.diag(R2_h) @ w
    r2e = np.diag(R2_e) @ w

    # RMS sizes: sigma
    var_h = r2h - rh @ rh
    var_e = r2e - re @ re

    sigma_h = np.sqrt(max(var_h, 0.0))
    sigma_e = np.sqrt(max(var_e, 0.0))

    # e-h dot product
    s_norm = np.sqrt(w)
    dot_he = np.einsum("j,k,ijk,ijk->", s_norm, s_norm, R_h, R_e)

    # e-h separation metrics
    d_lin = np.linalg.norm(re - rh)
    d_exc = np.sqrt(max(r2e + r2h - 2 * dot_he, 0.0))

    # e-h covariance, correlation
    cov = dot_he - rh @ re
    denom = sigma_h * sigma_e
    corr  = cov / denom if denom > 1e-12 else 0.0

    # center-of-mass size
    sigma_com = 0.5 * np.sqrt(max(var_h + var_e + 2 * cov, 0.0))

    exciton = {
        "rh":        rh        * interface.bohr_to_ang,
        "re":        re        * interface.bohr_to_ang,
        "sigma_h":   sigma_h   * interface.bohr_to_ang,
        "sigma_e":   sigma_e   * interface.bohr_to_ang,
        "d_lin":     d_lin     * interface.bohr_to_ang,
        "d_exc":     d_exc     * interface.bohr_to_ang,
        "cov":       cov       * interface.bohr_to_ang ** 2,
        "sigma_com": sigma_com * interface.bohr_to_ang,
        "corr":      corr,
    }

    interface.log.info("State %s -> State %s" % (initial_state, target_state))
    fmt= lambda x: f"{float(x): .3f}"
    rh_str = np.array2string(exciton['rh'], formatter={'float_kind': fmt})
    re_str = np.array2string(exciton['re'], formatter={'float_kind': fmt})
    interface.log.info("   Mean position of hole:            %s"     % rh_str)
    interface.log.info("   Mean position of electron:        %s"     % re_str)
    interface.log.info("   Linear e-h distance [Ang]:        % .6f"  % exciton['d_lin'])
    interface.log.info("   Hole size [Ang]:                  % .6f"  % exciton['sigma_h'])
    interface.log.info("   Electron size [Ang]:              % .6f"  % exciton['sigma_e'])
    interface.log.info("   RMS e-h separation [Ang]:         % .6f"  % exciton['d_exc'])
    interface.log.info("   Covariance [Ang^2]:               % .6f"  % exciton['cov'])
    interface.log.info("   Correlation coefficient:          % .6f"  % exciton['corr'])
    interface.log.info("   Center-of-mass size [Ang]:        % .6f"  % exciton['sigma_com'])

    return exciton
