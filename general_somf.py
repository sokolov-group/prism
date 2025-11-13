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
    print("\n \n \nSpin-Free Framework: Employ Wigner–Eckart’s theorem")
    print("Consider spin-orbit coupling effect...")
    ref_wfn = method.ref_wfn  
    ncas = method.ncas 
    nmo = method.nmo
    ncore = method.ncore 
    evec = method.evec
    en = method.en
    ref_nelecas = method.ref_nelecas 
    ref_wfn_spin_mult = method.ref_wfn_spin_mult
    S_cas = [round((spin_mult-1)/2,2) for spin_mult in ref_wfn_spin_mult]     
    nstate = len(ref_wfn)

    ##test by using CASSCF######
    #print("This is SOC-CASSCF")
    #evec = np.diag(np.ones(len(ref_wfn)))
    #en = method.e_ref 
    ##test by using CASSCF######

    #Get target state psi (wfn)
    wfn = np.einsum('ij,iab->jab',evec,ref_wfn)
    wfn = list(wfn)
    #method S:
    spin_mult_wfn = qd_nevpt2.determine_spin_mult(method,evec)
    S = [round((spin_mult-1)/2,2) for spin_mult in spin_mult_wfn] 

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
    method.en_soc, method.evec_soc = np.linalg.eigh(HSOC+H_sf)
    print("\n Absolute energies in a.u |||| Excitation energies in a.u ||  eV ||  cm-1\n*****************************")
    for e in method.en_soc:
        print("%14.6f ||||  %14.6f  ||  %14.6f  ||   %8.2f"%((e), (e-method.en_soc[0]),((e-method.en_soc[0])*27.2114),((e-method.en_soc[0])*219474.63)))
    
    return S_total, ms_total, I_total 



def osc_strength_soc(nevpt, en_soc, evec_soc, S_total, ms_total, I_total, gs_index = 0):

    ncore = nevpt.ncore 
    #n_micro_states = sum(nevpt.ref_wfn_deg)
    n_states = sum(nevpt.ref_wfn_deg)
    n_micro_states = len(en_soc)
    dip_mom_ao = nevpt.interface.dip_mom_ao
    mo_coeff = nevpt.mo
    nmo = nevpt.nmo
    ncas = nevpt.ncas
    wfn = np.einsum('ij,iab->jab',nevpt.evec,nevpt.ref_wfn)
    wfn = list(wfn)

    dip_mom_mo = np.zeros_like(dip_mom_ao)

    # Transform dipole moments from AO to MO basis
    for d in range(dip_mom_ao.shape[0]):
        dip_mom_mo[d] = mo_coeff.T @ dip_mom_ao[d] @ mo_coeff

    # List to store Osc. Strength Values
    osc_total = []
    print(ms_total)
    # Looping over CAS States
    for state in range(gs_index + 1, n_micro_states):
        # Reset final transformed RDM
        rdm_qd = np.zeros((nmo, nmo),dtype='complex')

        # Looping over states I,J
        for I in range(n_micro_states):
            for J in range(n_micro_states):
                #if  ((S_total[I]-S_total[J])<1e-8) and ((ms_total[I]-ms_total[J])<1e-8):
                i = I_total[I]
                j = I_total[J]
                rdm_mo = np.zeros((nmo, nmo),dtype='complex')  # Reset RDM in MO Basis   
                trdm_ca = nevpt.interface.compute_rdm1(wfn[i], wfn[j], nevpt.ref_nelecas[j])
                rdm_mo[ncore:ncore + ncas ,ncore:ncore + ncas] = trdm_ca
                if I == J:
                    rdm_mo[:ncore, :ncore] = 2 * np.eye(nevpt.ncore)
                
                rdm_qd += np.conj(evec_soc)[I, state] * rdm_mo * evec_soc[J, gs_index]

        # Create Dipole Moment Operator with RDM
        dip_evec_x = np.einsum('pq,pq', dip_mom_mo[0], rdm_qd)
        dip_evec_y = np.einsum('pq,pq', dip_mom_mo[1], rdm_qd)
        dip_evec_z = np.einsum('pq,pq', dip_mom_mo[2], rdm_qd)
 
        osc_x = ((2/3)*(en_soc[state] - en_soc[gs_index]))*(np.conj(dip_evec_x)*dip_evec_x)
        osc_y = ((2/3)*(en_soc[state] - en_soc[gs_index]))*(np.conj(dip_evec_y)*dip_evec_y)
        osc_z = ((2/3)*(en_soc[state] - en_soc[gs_index]))*(np.conj(dip_evec_z)*dip_evec_z)

        # Add Dipole Moment Components
        osc_total.append((osc_x + osc_y + osc_z).real)

    return osc_total

def gtensor(method, S_total, ms_total, I_total):
    print("Calculating g-tensor...")
    mf = method.interface.mf
    mo = method.mo
    ncore = method.ncore 
    #n_micro_states = sum(method.ref_wfn_deg)
    n_states = sum(method.ref_wfn_deg)
    n_micro_states = len(S_total)
    dip_mom_ao = method.interface.dip_mom_ao
    mo_coeff = method.mo
    nmo = method.nmo
    ncas = method.ncas
    qdnevpt_evec  = method.evec_soc
    
    S = np.zeros(n_states)
    for i in range(n_micro_states):
        I = I_total[i]
        S[I] = S_total[i]
    print(S_total)
    print(S)

    multiplicity=[]
    for I in range(n_states):
        multiplicity.append(int(S[I]*2 + 1))

    wfn = np.einsum('ij,iab->jab',method.evec,method.ref_wfn)
    wfn = list(wfn)

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
        #print(Sy)
        s_mat[1,A:A+multiplicity[I],A:A+multiplicity[I]]=Sy

        A += multiplicity[I]

        
    #print(s_mat[1])
    #print(s_mat[2])

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
    #print("set_different_origin")
    l1_ao += -1j * mf.mol.intor('cint1e_cg_irxp_sph', comp=3)#, hermi=1) #cint1e_cg_irxp_sph int1e_giao_irjxp_sph
    
    # AO -> MO basis:
    l1_mo = np.einsum('xpq,pi,qj->xij',l1_ao,mo,mo) 
    l_mat = np.zeros((3,n_micro_states,n_micro_states),dtype='complex')
    for I in range(n_micro_states):
        for J in range(n_micro_states):
            if J>=I:
                if  ((S_total[I]-S_total[J])<1e-8) and ((ms_total[I]-ms_total[J])<1e-8):
                    i = I_total[I]
                    j = I_total[J]
                    rdm_mo = np.zeros((nmo, nmo),dtype='complex')  # Reset RDM in MO Basis   
                    #rdm_ca = method.interface.compute_rdm1(wfn[i], wfn[j], method.ref_nelecas[j])
                    rdm_aabb = trans_rdm1s(wfn[j],wfn[i],ncas,method.ref_nelecas[i])
                    rdm_ca = rdm_aabb[0] + rdm_aabb[1]

                    rdm_mo[ncore:ncore + ncas ,ncore:ncore + ncas] = rdm_ca
                    if I == J:
                        rdm_mo[:ncore, :ncore] = 2 * np.eye(method.ncore)
                
                    l_mat[0,I,J] += np.einsum('ij,ij',rdm_mo,l1_mo[0])
                    l_mat[1,I,J] += np.einsum('ij,ij',rdm_mo,l1_mo[1])
                    l_mat[2,I,J] += np.einsum('ij,ij',rdm_mo,l1_mo[2])
    l_mat[0] = l_mat[0] + np.conj(l_mat[0]).T
    l_mat[1] = l_mat[1] + np.conj(l_mat[1]).T
    l_mat[2] = l_mat[2] + np.conj(l_mat[2]).T
    #l_mat = -l_mat
    #print("lx=")
    #print(l_mat[0])
    #print("ly=")
    #print(l_mat[1])
    #print("Sz=")
    #print(l_mat[2])
    
    Kramer_index=0
    print("kramer_index(defined by user)=",Kramer_index)
    #Kramer_index = 0 #int(np.sum(qdnevpt_degeneracy[0:de_index]))
    #print("kramer_index(fix alaways set ground state)=",Kramer_index)
    Kramer_pair  = qdnevpt_evec[:,Kramer_index:Kramer_index+2] 
    
    # define unique Kramer's doublet
    test_kramer_z = np.einsum('ai,ib,bj->aj',np.conj(Kramer_pair).T , s_mat[2] , Kramer_pair)
    if test_kramer_z[0,0] < 0:
        print("Change 0,1 -> 1,0")
        Kramer_pair[:, [0, 1]] = Kramer_pair[:, [1, 0]]

    test_kramer_x = np.einsum('ai,ib,bj->aj',np.conj(Kramer_pair).T , s_mat[0] , Kramer_pair)
    if np.real(test_kramer_x[0,1]) < 0:
        print("Change 0 -> -0")
        Kramer_pair[:, 0] = - Kramer_pair[:, 0]

    print("Kramer_pair=")
    print(Kramer_pair)

    # S in Kramer_pair basis
    S_kramer = np.einsum('ai,kib,bj->kaj',np.conj(Kramer_pair).T , s_mat , Kramer_pair)
    # L in Kramer_pair basis
    L_kramer = np.einsum('ai,kib,bj->kaj',np.conj(Kramer_pair).T , l_mat , Kramer_pair) 

    #G_tensor = np.zeros((3,3),dtype='complex')
    #for k in range(3):
    #    for l in range(3):
    #        for u in range(2):
    #            for v in range(2):
    #                G_tensor[k,l] += 2 * (L_kramer[k,u,v] + 2.002319 * S_kramer[k,u,v] ) * (L_kramer[l,v,u] + 2.002319 * S_kramer[l,v,u] )


      
    print("##############################################################")
    
    print("Sx_Kramer=")
    print(S_kramer[0])
    print("Sy_Kramer=")
    print(S_kramer[1])
    print("Sz_Kramer=")
    print(S_kramer[2])

    print("Lx_Kramer=")
    print(L_kramer[0])
    print("Ly_Kramer=")
    print(L_kramer[1])
    print("Lz_Kramer=")
    print(L_kramer[2])
    
    print("Re S_11 (x,y,z) = %14.6f, %14.6f, %14.6f"%(np.real(S_kramer[0,0,0]),np.real(S_kramer[1,0,0]),np.real(S_kramer[2,0,0])))
    print("Re S_12 (x,y,z) = %14.6f, %14.6f, %14.6f"%(np.real(S_kramer[0,0,1]),np.real(S_kramer[1,0,1]),np.real(S_kramer[2,0,1])))
    print("Im S_12 (x,y,z) = %14.6f, %14.6f, %14.6f"%(np.imag(S_kramer[0,0,1]),np.imag(S_kramer[1,0,1]),np.imag(S_kramer[2,0,1])))

    print("Re L_11 (x,y,z) = %14.6f, %14.6f, %14.6f"%(np.real(L_kramer[0,0,0]),np.real(L_kramer[1,0,0]),np.real(L_kramer[2,0,0])))
    print("Re L_12 (x,y,z) = %14.6f, %14.6f, %14.6f"%(np.real(L_kramer[0,0,1]),np.real(L_kramer[1,0,1]),np.real(L_kramer[2,0,1])))
    print("Im L_12 (x,y,z) = %14.6f, %14.6f, %14.6f"%(np.imag(L_kramer[0,0,1]),np.imag(L_kramer[1,0,1]),np.imag(L_kramer[2,0,1])))
    
    # S matrix (Σ)
    Sigma=np.zeros((3,3))
    for k in range(3):
        Sigma[k,0] = 2 * np.real( S_kramer[k,1,0])                  
        Sigma[k,1] = 2 * np.imag( S_kramer[k,1,0])                           
        Sigma[k,2] = 2 * np.real( S_kramer[k,0,0])                       
        
    # L matrix (Λ)
    Lambda=np.zeros((3,3))
    for k in range(3):
        Lambda[k,0] = 2 * np.real( L_kramer[k,1,0])         
        Lambda[k,1] = 2 * np.imag( L_kramer[k,1,0])    
        Lambda[k,2] = 2 * np.real( L_kramer[k,0,0])        

    print("###################################################################################")
    print("Σ=")
    print(Sigma)
    G_sigma = np.einsum('km,lm->kl',2.002319 * Sigma,2.002319 * Sigma)
    G_sigma_en, G_sigma_evec = np.linalg.eigh(G_sigma)    
    G_sigma_en = np.sqrt(G_sigma_en)

    print("g_s=")
    print("%14.6f, %14.6f, %14.6f"%(G_sigma_en[0],G_sigma_en[1],G_sigma_en[2]))
    print("%14.6f, %14.6f, %14.6f"%(G_sigma_en[0]-2.002319,G_sigma_en[1]-2.002319,G_sigma_en[2]-2.002319))
    
    print("Λ=")
    print(Lambda)    
    G_Lambda = np.einsum('km,lm->kl', Lambda,  Lambda)
    G_Lambda_en, G_Lambda_evec = np.linalg.eigh(G_Lambda)
    G_Lambda_en = np.sqrt(G_Lambda_en)
    print("g_l=")
    print("%14.6f, %14.6f, %14.6f"%(G_Lambda_en[0],G_Lambda_en[1],G_Lambda_en[2]))
    
    print("##############################")
    g = 2.002319 * Sigma + Lambda
    G = np.einsum('km,lm->kl',g,g)

    print("g=")
    print(g)
    print("G=")
    print(G)
    #print("G_gtensor=")
    #print(G_tensor)

    G_en, G_evec = np.linalg.eigh(G)
    #G_tensor_en, G_tensor_evec = np.linalg.eigh(G_tensor)
    G_sq_en = np.sqrt(G_en)
    #G_tensor_sq_en = np.sqrt(G_tensor_en)
    print("%14.6f, %14.6f, %14.6f"%(G_sq_en[0],G_sq_en[1],G_sq_en[2]))
    print("%14.6f, %14.6f, %14.6f"%(G_sq_en[0]-2.002319,G_sq_en[1]-2.002319,G_sq_en[2]-2.002319),"g_e=2.002319")
    print("%14.3f, %14.3f, %14.3f, ptt(Kramer)"%(1000*(G_sq_en[0]-2.002319),1000*(G_sq_en[1]-2.002319),1000*(G_sq_en[2]-2.002319)))
    return G_sq_en
    
        
def gtensor_general(method, S_total, ms_total, I_total):
    print("Calculating g-tensor(general)...")
    mf = method.interface.mf
    mo = method.mo
    ncore = method.ncore 
    #n_micro_states = sum(method.ref_wfn_deg)
    n_states = sum(method.ref_wfn_deg)
    n_micro_states = len(S_total)
    dip_mom_ao = method.interface.dip_mom_ao
    mo_coeff = method.mo
    nmo = method.nmo
    ncas = method.ncas
    qdnevpt_evec  = method.evec_soc
    
    S = np.zeros(n_states)
    for i in range(n_micro_states):
        I = I_total[i]
        S[I] = S_total[i]

    multiplicity=[]
    for I in range(n_states):
        multiplicity.append(int(S[I]*2 + 1))

    wfn = np.einsum('ij,iab->jab',method.evec,method.ref_wfn)
    wfn = list(wfn)

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
    #print("set_different_origin")
    l1_ao += -1j * mf.mol.intor('cint1e_cg_irxp_sph', comp=3)#, hermi=1) #cint1e_cg_irxp_sph int1e_giao_irjxp_sph
    
    # AO -> MO basis:
    l1_mo = np.einsum('xpq,pi,qj->xij',l1_ao,mo,mo) 
    l_mat = np.zeros((3,n_micro_states,n_micro_states),dtype='complex')
    for I in range(n_micro_states):
        for J in range(n_micro_states):
            if J>=I:
                if  ((S_total[I]-S_total[J])<1e-8) and ((ms_total[I]-ms_total[J])<1e-8):
                    i = I_total[I]
                    j = I_total[J]
                    rdm_mo = np.zeros((nmo, nmo),dtype='complex')  # Reset RDM in MO Basis   
                    #rdm_ca = method.interface.compute_rdm1(wfn[i], wfn[j], method.ref_nelecas[j])
                    rdm_aabb = trans_rdm1s(wfn[j],wfn[i],ncas,method.ref_nelecas[i])
                    rdm_ca = rdm_aabb[0] + rdm_aabb[1]

                    rdm_mo[ncore:ncore + ncas ,ncore:ncore + ncas] = rdm_ca
                    if I == J:
                        rdm_mo[:ncore, :ncore] = 2 * np.eye(method.ncore)
                
                    l_mat[0,I,J] += np.einsum('ij,ij',rdm_mo,l1_mo[0])
                    l_mat[1,I,J] += np.einsum('ij,ij',rdm_mo,l1_mo[1])
                    l_mat[2,I,J] += np.einsum('ij,ij',rdm_mo,l1_mo[2])
    l_mat[0] = l_mat[0] + np.conj(l_mat[0]).T
    l_mat[1] = l_mat[1] + np.conj(l_mat[1]).T
    l_mat[2] = l_mat[2] + np.conj(l_mat[2]).T

    Kramer_index = 0 #int(np.sum(qdnevpt_degeneracy[0:de_index]))
    print("kramer_index(alaways set ground state)=",Kramer_index)
    S = S_total[0]
    multicity = np.round(S*2+1)
    multicity = int(multicity)
    print("Calculating g-tensor for multicity = ", multicity )
    Kramer_pair  = qdnevpt_evec[:,Kramer_index:Kramer_index+multicity] 
    
    # define unique Kramer's doublet
    test_kramer_z = np.einsum('ai,ib,bj->aj',np.conj(Kramer_pair).T , s_mat[2] , Kramer_pair)
    #test_kramer_lx = np.einsum('ai,ib,bj->aj',np.conj(Kramer_pair).T , l_mat[0] , Kramer_pair)
    #test_kramer_ly = np.einsum('ai,ib,bj->aj',np.conj(Kramer_pair).T , l_mat[1] , Kramer_pair)  
    test_kramer_lz = np.einsum('ai,ib,bj->aj',np.conj(Kramer_pair).T , l_mat[2] , Kramer_pair)


    test_Jz = test_kramer_lz + test_kramer_z * 2.002319


    
    z_en, z_evec = np.linalg.eigh(test_Jz)
    print("Use Jz to transform...")

    Kramer_pair_2 = Kramer_pair @ z_evec
    


    test_kramer_z = np.real(z_en)

    reorder_list  = np.argsort(test_kramer_z)



    Kramer_pair_new = Kramer_pair_2
    test_kramer_z = np.einsum('ai,ib,bj->aj',np.conj(Kramer_pair_new).T , s_mat[2] , Kramer_pair_new)


    test_kramer_x = np.einsum('ai,ib,bj->aj',np.conj(Kramer_pair_new).T , s_mat[0] , Kramer_pair_new)
    

    #print("Kramer_pair_new=")
    #print(Kramer_pair_new)
    
   
    
    # S in Kramer_pair basis
    S_kramer = np.einsum('ai,kib,bj->kaj',np.conj(Kramer_pair_new).T , s_mat , Kramer_pair_new)
    # L in Kramer_pair basis
    L_kramer = np.einsum('ai,kib,bj->kaj',np.conj(Kramer_pair_new).T , l_mat , Kramer_pair_new) 
    
    
    Hab = L_kramer + S_kramer * 2.002319

    g=np.zeros([3,3])
    for i in range(3):
      g[i,0] = np.real(Hab[i,0,1]) *2/np.sqrt(S*2)
      g[i,1] = np.imag(Hab[i,0,1]) *(-2)/np.sqrt(S*2)
      g[i,2] = np.real(Hab[i,0,0])/S
    
    #g[1,1] = - g[1,1]
    #print(g)
    #g = (g + g.T)/2
    G = np.einsum('km,lm->kl',g,g)

    print("g=")
    print(g)
    print("G=")
    print(G)
    #print("G_gtensor=")
    #print(G_tensor)

    G_en, G_evec = np.linalg.eigh(G)
    #G_tensor_en, G_tensor_evec = np.linalg.eigh(G_tensor)
    G_sq_en = np.sqrt(G_en)
    #G_tensor_sq_en = np.sqrt(G_tensor_en)
    print("%14.6f, %14.6f, %14.6f"%(G_sq_en[0],G_sq_en[1],G_sq_en[2]))
    print("%14.6f, %14.6f, %14.6f"%(G_sq_en[0]-2.002319,G_sq_en[1]-2.002319,G_sq_en[2]-2.002319))
    print("%14.3f, %14.3f, %14.3f, ptt(general)"%(1000*(G_sq_en[0]-2.002319),1000*(G_sq_en[1]-2.002319),1000*(G_sq_en[2]-2.002319)))