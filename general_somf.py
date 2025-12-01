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
from pyscf.fci.direct_spin1 import trans_rdm1s
from prism import qd_nevpt2
from prism import nevpt2

# Add python path for socutils:
prism_path = os.path.dirname(os.path.abspath(__file__)) 
if prism_path not in sys.path:
      sys.path.insert(0, prism_path)

# Add socutils module
from socutils.somf import somf

def getSOC_integrals(interface, soc="breit-pauli", unc=True):
    
    mo = interface.mo
    nmo = interface.nmo
    nao = interface.mo.shape[1]

    prefactor = 0.5 / ((LIGHT_SPEED)**2)
    mol = interface.mol

    # Build 1e-density matrix
    rdm1ao = interface.mc.make_rdm1() 

    xmol, contr_coeff = None, None
    if unc:
        xmol, contr_coeff = sfx2c1e.SpinFreeX2C(mol).get_xmol()
        rdm1ao = reduce(np.dot, (contr_coeff, rdm1ao, contr_coeff.T))
    else:
        xmol, contr_coeff = mol, np.eye(mol.nao_nr())
 
        nbasis = xmol.nao_nr()
        hsocint = np.zeros((3, nbasis, nbasis))
   
    if (soc=="breit-pauli"):
        hsocint = np.zeros((3, nbasis, nbasis))
        hsocint += prefactor * somf.get_wso(xmol)
        hsocint -= prefactor * somf.get_fso2e_bp(xmol, rdm1ao)

    elif (soc=="x2c"):
        hsocint = np.zeros((3, nbasis, nbasis))
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


def generalSOC(interface, en, rdm, S, ms):
    print("\nSpin-Free Framework: Employ Wigner–Eckart’s theorem")
    print("Consider spin-orbit coupling effect...")
    nmo = interface.nmo
    soc = interface.soc
    unc = interface.uncontract
    nstate = len(en)

    print("calculate Wigner's rdm...")
    rdm_wigner = np.zeros((nstate,nstate,nmo,nmo), dtype='complex')
    for I in range(nstate):
        for J in range(nstate):
            cg = CG(S[I],ms[I], 1, 0, S[J],ms[J]).doit()
            cg = float(cg)
            T_z = 1/np.sqrt(2) * (rdm[0,I,J] - rdm[1,I,J]) / cg
            rdm_wigner[I,J] = T_z 

    # Get SOC integrals:
    h_soc = getSOC_integrals(interface, soc=soc, unc=unc)
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
            m = S[i]-j
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
    en_soc, evec_soc = np.linalg.eigh(HSOC+H_sf)

    #calculate Osc Str
    osc_str = interface.osc_strength_general(en_soc,rdm[0]+rdm[1], (I_total,evec_soc))

    h2ev = interface.hartree_to_ev
    h2cm = interface.hartree_to_inv_cm

    #print("\n Absolute energies in a.u |||| Excitation energies in a.u ||  eV ||  cm-1\n*****************************")
    #for e in en_soc:
    #    print("%14.6f ||||  %14.6f  ||  %14.6f  ||   %8.2f"%((e), (e-en_soc[0]),((e-en_soc[0])*27.2114),((e-en_soc[0])*219474.63)))
    print("\nSummary of results for the SOC calculation with the %s Hamiltionian:" % (soc))

    print("-----------------------------------------------------------------------------------------------------------------")
    print("  State            E(total)           dE(a.u.)        dE(eV)      dE(nm)       dE(cm-1)      Osc Str. ")
    print("-----------------------------------------------------------------------------------------------------------------")
    
    e_gs = en_soc[0]

    for p in range(nstate_total):
        de = en_soc[p] - e_gs
        de_ev = de * h2ev
        de_cm = de * h2cm
        if p == 0 or abs(de) < 1e-5:
            print("%5d       %20.12f %14.8f %12.4f %12s %14.4f   %12s" % ((p+1), en_soc[p], de, de_ev, " ", de_cm, " "))
        else:
            de_nm = 10000000 / de_cm
            print("%5d       %20.12f %14.8f %12.4f %12.4f %14.4f   %12.8f" % ((p+1), en_soc[p], de, de_ev, de_nm, de_cm, osc_str[p-1]))
    print("-----------------------------------------------------------------------------------------------------------------")
    return en_soc, evec_soc, S_total, ms_total, I_total 

#def osc_strength_soc(interface, en_soc, evec_soc, rdm,  I_total, gs_index = 0):
#
#   ncore = interface.ncore 
#   n_states = len(rdm[0,0])
#   n_micro_states = len(en_soc)
#   dip_mom_ao = interface.dip_mom_ao
#   mo_coeff = interface.mo
#   nmo = interface.nmo
#   ncas = interface.ncas
#   dip_mom_mo = np.zeros_like(dip_mom_ao)
#
#   # Transform dipole moments from AO to MO basis
#   for d in range(dip_mom_ao.shape[0]):
#       dip_mom_mo[d] = mo_coeff.T @ dip_mom_ao[d] @ mo_coeff
#
#   # List to store Osc. Strength Values
#   osc_total = []
#   # Looping over CAS States
#   for state in range(gs_index + 1, n_micro_states):
#       # Reset final transformed RDM
#       rdm_qd = np.zeros((nmo, nmo),dtype='complex')
#
#       # Looping over states I,J
#       for I in range(n_micro_states):
#           for J in range(n_micro_states):
#               i = I_total[I]
#               j = I_total[J]
#               rdm_mo = rdm[0,i,j] + rdm[1,i,j]
#               rdm_qd += np.conj(evec_soc)[I, state] * rdm_mo * evec_soc[J, gs_index]
#
#       # Create Dipole Moment Operator with RDM
#       dip_evec_x = np.einsum('pq,pq', dip_mom_mo[0], rdm_qd)
#       dip_evec_y = np.einsum('pq,pq', dip_mom_mo[1], rdm_qd)
#       dip_evec_z = np.einsum('pq,pq', dip_mom_mo[2], rdm_qd)
#
#       osc_x = ((2/3)*(en_soc[state] - en_soc[gs_index]))*(np.conj(dip_evec_x)*dip_evec_x)
#       osc_y = ((2/3)*(en_soc[state] - en_soc[gs_index]))*(np.conj(dip_evec_y)*dip_evec_y)
#       osc_z = ((2/3)*(en_soc[state] - en_soc[gs_index]))*(np.conj(dip_evec_z)*dip_evec_z)
#
#       # Add Dipole Moment Components
#       osc_total.append((osc_x + osc_y + osc_z).real)
#
#   return osc_total
            
def gtensor_general(interface, evec_soc, rdm, S_total, I_total,target_index = 0):
    print("Calculating g-tensor(general)...")
    mf = interface.mf
    mo = interface.mo
    ncore = interface.ncore 
    n_states = len(rdm[0,0])
    n_micro_states = len(evec_soc)
    mo_coeff = interface.mo
    nmo = interface.nmo
    ncas = interface.ncas
    
    # Calculate spin-free S
    S = np.zeros(n_states)
    for i in range(n_micro_states):
        I = I_total[i]
        S[I] = S_total[i]

    # Calculate spin-free multiplicity
    multiplicity=[]
    for I in range(n_states):
        multiplicity.append(int(S[I]*2 + 1))
    
    # Calculate ms_total
    ms_total = []
    for I in range(n_states):
        for i in range(multiplicity[I]):
            ms_total.append(S[I]-i)

    #contruct S matrix
    s_mat = np.zeros((3,n_micro_states,n_micro_states),dtype='complex')
    s_mat[2] = np.diag(ms_total)
    A = 0
    for I in range(n_states):
        Sz=np.arange(S[I],-S[I]-1,-1)
        C = []
        for J in range(len(Sz)-1):
            J += 1
            C.append(np.sqrt( (S[I]-Sz[J]) * (S[I]+Sz[J]+1)))
        C =np.array(C)
        #Sx
        Sx = C/2
        Sx = np.diag(Sx,1)
        Sx = Sx + Sx.T
        #print(Sx)
        s_mat[0,A:A+multiplicity[I],A:A+multiplicity[I]]=Sx
        
        #Sy
        Sy = -C/2 * 1j
        Sy = np.diag(Sy,1)
        Sy = Sy - Sy.T
        s_mat[1,A:A+multiplicity[I],A:A+multiplicity[I]]=Sy

        A += multiplicity[I]

    ###L part########
    origin_type = 'charge'
    if origin_type == 'charge':
        origin = [ 0, 0 ,0]
        total_charge = 0
        for atm in range(mf.mol.natm):
            origin += mf.mol.atom_coord(atm) * mf.mol.atom_charge(atm)
            total_charge += mf.mol.atom_charge(atm)
        origin = origin / total_charge
        print("origin=",origin)
        mf.mol.set_common_orig(origin)
    elif origin_type == 'atom1':
        print("origin=",mf.mol.atom_coord(0))
        mf.mol.set_common_orig(mf.mol.atom_coord(0))
    else:
        print("origin=",origin_type)
        mf.mol.set_common_orig(origin_type)
    l1_ao = 0 
    l1_ao += -1j * mf.mol.intor('cint1e_cg_irxp_sph', comp=3)#, hermi=1) #cint1e_cg_irxp_sph int1e_giao_irjxp_sph
    
    # AO -> MO basis:
    l1_mo = np.einsum('xpq,pi,qj->xij',l1_ao,mo,mo) 
    l_mat = np.zeros((3,n_micro_states,n_micro_states),dtype='complex')
    for I in range(n_micro_states):
        for J in range(n_micro_states):
            if J>=I:
                if  (np.abs(S_total[I]-S_total[J])<1e-8) and (np.abs(ms_total[I]-ms_total[J])<1e-8):
                    i = I_total[I]
                    j = I_total[J]

                    rdm_mo = np.zeros((nmo, nmo),dtype='complex')  # Reset RDM in MO Basis   
                    rdm_mo = rdm[0,i,j] + rdm[1,i,j]

                    l_mat[0,I,J] += np.einsum('ij,ij',rdm_mo,l1_mo[0])
                    l_mat[1,I,J] += np.einsum('ij,ij',rdm_mo,l1_mo[1])
                    l_mat[2,I,J] += np.einsum('ij,ij',rdm_mo,l1_mo[2])

    l_mat[0] = l_mat[0] + np.conj(l_mat[0]).T
    l_mat[1] = l_mat[1] + np.conj(l_mat[1]).T
    l_mat[2] = l_mat[2] + np.conj(l_mat[2]).T

    target_index = 0 
    print("Target State index=",target_index)
    S = S[target_index]
    target_multiplicity = multiplicity[target_index]
    print("Calculating g-tensor for multiplicity = ", target_multiplicity)
    Kramer_pair  = evec_soc[:,target_index:target_index+target_multiplicity] 
    
    # define unique Kramer's doublet
    S_old = np.einsum('ai,kib,bj->kaj',np.conj(Kramer_pair).T , s_mat , Kramer_pair)
    L_old = np.einsum('ai,kib,bj->kaj',np.conj(Kramer_pair).T , l_mat , Kramer_pair)
    Hab_old = L_old + S_old * 2.002319

    print("Use Jz to transform...")
    z_en, z_evec = np.linalg.eigh(Hab_old[2])
    Kramer_pair_2 = Kramer_pair @ z_evec
    Kramer_pair_new =  Kramer_pair_2   
    
    # S in Kramer_pair basis
    S_new = np.einsum('ai,kib,bj->kaj',np.conj(Kramer_pair_new).T , s_mat , Kramer_pair_new)
    # L in Kramer_pair basis
    L_new = np.einsum('ai,kib,bj->kaj',np.conj(Kramer_pair_new).T , l_mat , Kramer_pair_new) 
    
    Hab_new = L_new + S_new * 2.002319
    Hab = Hab_new
 
    g=np.zeros([3,3])
    for i in range(3):
      g[i,0] = np.real(Hab[i,0,1]) *2/np.sqrt(S*2)
      g[i,1] = np.imag(Hab[i,0,1]) *(-2)/np.sqrt(S*2)
      g[i,2] = np.real(Hab[i,0,0])/S
    
    G = np.einsum('km,lm->kl',g,g)
    print("g=")
    print(g)
    print("G=")
    print(G)

    G_en, G_evec = np.linalg.eigh(G)
    #G_tensor_en, G_tensor_evec = np.linalg.eigh(G_tensor)
    G_sq_en = np.sqrt(G_en)
    #G_tensor_sq_en = np.sqrt(G_tensor_en)
    print("magnetic axis=")
    print(G_evec)
    print("g-factor=")
    print("%14.6f, %14.6f, %14.6f, ge=2.002319"%(G_sq_en[0],G_sq_en[1],G_sq_en[2]))
    print("%14.6f, %14.6f, %14.6f"%(G_sq_en[0]-2.002319,G_sq_en[1]-2.002319,G_sq_en[2]-2.002319))
    print("%14.3f, %14.3f, %14.3f, ptt(general)"%(1000*(G_sq_en[0]-2.002319),1000*(G_sq_en[1]-2.002319),1000*(G_sq_en[2]-2.002319)))
