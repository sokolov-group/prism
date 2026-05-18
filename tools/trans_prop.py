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

def compute_exciton_analysis(interface, TDM, omega):

    # Variables from kernel
    ncore = interface.ncore
    ncas = interface.ncas
    nextern = interface.nextern

    nroots = interface.nroots

    mo = interface.mo
    mol = interface.mol

    # Hole/Particle
    hole_idx = list(range(ncore + ncas))
    particle_idx = list(range(ncore, ncore + ncas + nextern))

    # AO Integrals
    R_ao = mol.intor("int1e_r")
    R2_ao = mol.intor("int1e_r2")

    # Transform to MO basis
    R_mo  = np.einsum('pi,rpq,qj->rij', mo, R_ao, mo)
    R2_mo = mo.T @ R2_ao @ mo

    # Restrict to hole/particle spaces
    Rh  = R_mo[:, hole_idx][:,:,hole_idx]
    Re  = R_mo[:, particle_idx][:,:,particle_idx]

    R2h = R2_mo[np.ix_(hole_idx, hole_idx)]
    R2e = R2_mo[np.ix_(particle_idx, particle_idx)]

    # Orbital Centroids
    rh_orb = np.stack([np.diag(Rh[a]) for a in range(3)], axis=1)
    re_orb = np.stack([np.diag(Re[a]) for a in range(3)], axis=1)

    ### code below takes TDM & omega from compute_ntos fcn ###

    # e/h densities
    rho_h = TDM @ TDM.T
    rho_e = TDM.T @ TDM

    # raw second moments: <r^2>
    rh2 = np.einsum('ij,ij->', R2h, rho_h) / omega
    re2 = np.einsum('ij,ij->', R2e, rho_e) / omega

    # e-h dot product
    dot_he = np.sum((TDM**2) * (rh_orb @ re_orb.T)) / omega

    # mean positions: <r>
    rh = np.einsum('rij,ij->r', Rh, rho_h) / omega
    re = np.einsum('rij,ij->r', Re, rho_e) / omega

    # RMS sizes: sigma
    sigma_h = np.sqrt(max(rh2 - rh @ rh, 0.0))
    sigma_e = np.sqrt(max(re2 - re @ re, 0.0))

    # e-h separation metrics
    d_lin = np.linalg.norm(re - rh)
    d_exc = np.sqrt(max(re2 + rh2 - 2 * dot_he, 0.0))

    # e-h covariance, correlation
    cov = dot_he - rh @ re
    denom = sigma_h * sigma_e
    corr  = cov / denom if denom > 1e-16 else 0.0

    exciton =  {
        "rh":      rh      * interface.bohr_to_ang,
        "re":      re      * interface.bohr_to_ang,
        "sigma_h": sigma_h * interface.bohr_to_ang,
        "sigma_e": sigma_e * interface.bohr_to_ang,
        "d_lin":   d_lin   * interface.bohr_to_ang,
        "d_exc":   d_exc   * interface.bohr_to_ang,
        "cov":     cov     * interface.bohr_to_ang ** 2,
        "corr":    corr,
    }

    interface.log.info("\n   === Exciton Analysis ===")
    fmt = lambda x: f"{float(x): .3f}"
    rh = np.array2string(exciton['rh'], formatter={'float_kind': fmt})
    re = np.array2string(exciton['re'], formatter={'float_kind': fmt})
    interface.log.info(f"   Mean position of hole:            {rh}")
    interface.log.info(f"   Mean position of electron:        {re}")
    interface.log.info(f"   Linear e-h distance [Ang]:        {exciton['d_lin']: .6f}")
    interface.log.info(f"   Hole size [Ang]:                  {exciton['sigma_h']: .6f}")
    interface.log.info(f"   Electron size [Ang]:              {exciton['sigma_e']: .6f}")
    interface.log.info(f"   RMS e-h separation [Ang]:         {exciton['d_exc']: .6f}")
    interface.log.info(f"   Covariance [Ang^2]:               {exciton['cov']: .6f}")
    interface.log.info(f"   Correlation coefficient:          {exciton['corr']: .6f}")
    interface.log.info("")

