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
#

import sys
import os
import numpy as np
from functools import reduce
from pyscf.lib.parameters import LIGHT_SPEED
from pyscf.x2c import sfx2c1e
from pyscf.x2c import x2c
from sympy.physics.quantum.cg import CG

# Add python path for socutils:
prism_path = os.path.dirname(os.path.abspath(__file__)) 
if prism_path not in sys.path:
      sys.path.insert(0, prism_path)

# Add socutils module
from socutils.somf import somf

def getSOC_integrals(method, unc = True):
    
    mo = method.mo
    nmo = method.nmo
    nao = method.mo.shape[0]

    prefactor = 0.5 / ((LIGHT_SPEED)**2)
    mol = method.interface.mol

    # Build 1e-density matrix
    rdm1ao = method.interface.mc.make_rdm1() 

    if unc:
        xmol, contr_coeff = sfx2c1e.SpinFreeX2C(mol).get_xmol()
        rdm1ao = reduce(np.dot, (contr_coeff, rdm1ao, contr_coeff.T))
    else:
        xmol, contr_coeff = method.mol, np.eye(method.mol.nao_nr())

    nbasis = xmol.nao_nr()
    hsocint = np.zeros((3, nbasis, nbasis))

    if method.soc == "breit-pauli":
        hsocint += prefactor * somf.get_wso(xmol)
        hsocint -= prefactor * somf.get_fso2e_bp(xmol, rdm1ao)
    elif method.soc == "x2c":
        hsocint += prefactor * somf.get_hso1e_x2c1(xmol)
        hsocint -= prefactor * somf.get_fso2e_x2c(xmol, rdm1ao)
    else:
        raise Exception("Incorrect SOC flag in input file!!")

    ###contract########
    h_soc_all_contr = np.zeros((3, nao, nao))
    for comp in range(3):
        h_soc_all_contr[comp] = reduce(np.dot, (contr_coeff.T, hsocint[comp], contr_coeff))

    ##### Convert to MO basis:
    h_soc_all_contr = np.einsum('xpq,pi,qj->xij',h_soc_all_contr, mo, mo)
    
    h_soc_total = np.zeros((3, nmo, nmo), dtype = 'complex')    
    for comp in range(3):
        h_soc_total[comp] =-1j*(h_soc_all_contr[comp].astype('complex'))


    return h_soc_total #hsocint


def generalSOC(method):
    print("\n \n \n Spin-Free Framework: Employ Wigner–Eckart’s theorem")
    print("Consider spin-orbit coupling effect...")
    ref_wfn = method.ref_wfn  
    ncas = method.ncas 
    nmo = method.nmo
    ncore = method.ncore 
    evec = method.evec
    en = method.en
    ref_nelecas = method.ref_nelecas 
    ref_wfn_spin_mult = method.ref_wfn_spin_mult
    S = [round((spin_mult-1)/2,2) for spin_mult in ref_wfn_spin_mult]     
    nstate = len(ref_wfn)

    ##test by using CASSCF######
    #print("This is SOC-CASSCF")
    #evec = np.diag(np.ones(len(ref_wfn)))
    #en = method.e_ref 
    ##test by using CASSCF######

    #Get target state psi (wfn)
    wfn = np.einsum('ij,iab->jab',evec,ref_wfn)
    wfn = list(wfn)
    #Get ms 
    ms = []
    for I in range(nstate):
        sz = method.interface.apply_S_z(wfn[I],ncas,ref_nelecas[I])
        ms.append(np.dot(wfn[I].ravel(), sz.ravel()))

    ms = [round(elem,2) for elem in ms]


    print("calculate rdm...")
    from pyscf.fci.direct_spin1 import trans_rdm1s
    
    rdm_wigner = np.zeros((nstate,nstate,nmo,nmo), dtype='complex')
    for I in range(nstate):
        for J in range(nstate):
            cg = CG(S[I],ms[I], 1, 0, S[J],ms[J]).doit()
            cg = float(cg)
            rdm_aabb = trans_rdm1s(wfn[J],wfn[I],ncas,ref_nelecas[I])
            T_z = 1/np.sqrt(2) * (rdm_aabb[0] - rdm_aabb[1]) / cg
            rdm_wigner[I,J,ncore:ncore + ncas ,ncore:ncore + ncas] = T_z 

    # Get SOC integrals:
    h_soc = getSOC_integrals(method)
    h1_plus = (h_soc[0] + (1j*h_soc[1])) 
    h1_minus = (h_soc[0] - (1j*h_soc[1])) 
    h1_zero = h_soc[2]

    H_minus = np.einsum('pq,ijpq->ij',h1_minus,rdm_wigner) * (-1/2)
    H_zero = np.einsum('pq,ijpq->ij',h1_zero,rdm_wigner) / np.sqrt(2)
    H_plus = np.einsum('pq,ijpq->ij',h1_plus,rdm_wigner) / 2

    S_total = []
    ms_total = []
    multiplicity = []
    I_total = []
    E_spinstate = []
    for i in range(nstate):
        n = int(S[i]*2 + 1)
        multiplicity.append(n)
        for j in range(n):
            S_total.append(S[i])
            m = -S[i]+j
            ms_total.append(m)
            I_total.append(i)
            E_spinstate.append(en[i])
            
    nstate_total = len(S_total)

    # Forming SOC Hamiltonian using WE-Theorem:
    HSOC = np.zeros((nstate_total,nstate_total),dtype='complex')
    for I in range(nstate_total):
        for J in  range(nstate_total):
            m_dif = ms_total[I] - ms_total[J]
            m_dif = int(np.round(m_dif))
            #print(m_dif)
            if m_dif == 0:
                cg = CG(S_total[J],ms_total[J], 1, 0, S_total[I],ms_total[I]).doit()
                cg = float(cg)
                HSOC[I,J] = cg * H_zero[I_total[I],I_total[J]]

            if m_dif == 1:
                cg = CG(S_total[J],ms_total[J], 1, 1, S_total[I],ms_total[I]).doit()
                cg = float(cg)
                HSOC[I,J] = cg * H_minus[I_total[I],I_total[J]]
            
            if m_dif == -1:
                cg = CG(S_total[J],ms_total[J], 1, -1, S_total[I],ms_total[I]).doit()
                cg = float(cg)
                HSOC[I,J] = cg * H_plus[I_total[I],I_total[J]]

    H_sf = np.diag(E_spinstate).astype('complex')
    method.en, method.evec = np.linalg.eigh(HSOC+H_sf)

    print("\n Absolute energies in a.u |||| Excitation energies in a.u ||  eV ||  cm-1\n*****************************")
    for e in method.en:
        print("%14.6f ||||  %14.6f  ||  %14.6f  ||   %8.2f"%((e), (e-method.en[0]),((e-method.en[0])*27.2114),((e-method.en[0])*219474.63)))
    
    return S_total, ms_total, I_total 



def osc_strength_soc(nevpt, en, evec, S_total, ms_total, I_total, gs_index = 0):

    ncore = nevpt.ncore 
    #n_micro_states = sum(nevpt.ref_wfn_deg)
    n_states = sum(nevpt.ref_wfn_deg)
    n_micro_states = len(en)
    dip_mom_ao = nevpt.interface.dip_mom_ao
    mo_coeff = nevpt.mo
    nmo = nevpt.nmo
    ncas = nevpt.ncas

    dip_mom_mo = np.zeros_like(dip_mom_ao)

    # Transform dipole moments from AO to MO basis
    for d in range(dip_mom_ao.shape[0]):
        dip_mom_mo[d] = mo_coeff.T @ dip_mom_ao[d] @ mo_coeff

    # List to store Osc. Strength Values
    osc_total = []

    # Looping over CAS States
    for state in range(gs_index + 1, n_micro_states):
        # Reset final transformed RDM
        rdm_qd = np.zeros((nmo, nmo),dtype='complex')

        # Looping over states I,J
        for I in range(n_micro_states):
            for J in range(n_micro_states):
                #if  (S_total[I]==S_total[J]) and (ms_total[I]==ms_total[J]):
                i = I_total[I]
                j = I_total[J]
                rdm_mo = np.zeros((nmo, nmo),dtype='complex')  # Reset RDM in MO Basis   
                trdm_ca = nevpt.interface.compute_rdm1(nevpt.ref_wfn[i], nevpt.ref_wfn[j], nevpt.ref_nelecas[j])
                rdm_mo[ncore:ncore + ncas ,ncore:ncore + ncas] = trdm_ca
                if I == J:
                    rdm_mo[:ncore, :ncore] = 2 * np.eye(nevpt.ncore)
                
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