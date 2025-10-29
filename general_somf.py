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
    h_soc_all_x2c1_contr = np.zeros((3, nao, nao))
    for comp in range(3):
        h_soc_all_x2c1_contr[comp] = reduce(np.dot, (contr_coeff.T, hsocint[comp], contr_coeff))

    ##### Convert to MO basis:
    h_soc_all_x2c1_contr = np.einsum('xpq,pi,qj->xij',h_soc_all_x2c1_contr, mo, mo)
    
    h_soc_total = np.zeros((3, nmo, nmo), dtype = 'complex')    
    for comp in range(3):
        h_soc_total[comp] =-1j*(h_soc_all_x2c1_contr[comp].astype('complex'))



    return h_soc_total #hsocint




def Wigner_SOC(method):
    print("\n \n \n Employ Wigner–Eckart’s theorem")
    print("Consider spin-orbit coupling effect...")
    ref_wfn =method.ref_wfn  
    ncas = method.ncas 
    nmo = method.nmo
    ncore = method.ncore 

    ref_nelecas = method.ref_nelecas 
    ref_wfn_spin_mult = method.ref_wfn_spin_mult
    S =  [round((spin_mult-1)/2,2) for spin_mult in ref_wfn_spin_mult]     
    nstate = len(ref_wfn)


    if method.evec_qdnevpt2 is None:
        print("It is NEVPT2")
        method.evec_qdnevpt2 = np.diag(np.ones(len(ref_wfn)))
    else:
        print("It is QDNEVPT2")

    ##test by using CASSCF######
    #print("This is SOC-CASSCF")
    #method.evec_qdnevpt2 = np.diag(np.ones(len(ref_wfn)))
    #method.en_qdnevpt2 = method.e_ref_cas 
    ##test by using CASSCF######

    evec = method.evec_qdnevpt2
    en = method.en_qdnevpt2

    #Get target state psi (wfn)
    wfn = np.einsum('ij,iab->jab',evec,ref_wfn)
    wfn = list(wfn)
    #Get ms 
    ms = []
    for I in range(nstate):
        sz = method.interface.apply_S_z(wfn[I],ncas,ref_nelecas[I])
        ms.append(np.dot(wfn[I].ravel(), sz.ravel()))

    ms = [round(elem,2) for elem in ms]
    print("ms=")
    print(ms)

    #calculate 1-TRDM
    #trdm_mo = np.zeros((nstate,nstate,nmo, nmo))
    #for I in range(nstate):
    #    for J in range(nstate):
    #        trdm_ca = method.interface.compute_rdm1(wfn[I], wfn[J], ref_nelecas[I])
    #        trdm_mo[I,J,ncore:ncore + ncas,ncore:ncore + ncas] = trdm_ca
    #print(trdm_ca)
    #print(trdm_ca.shape)
 
    #print("finish")
    print("calculate rdm...")

    from pyscf.fci.direct_spin1 import trans_rdm1s
    from pyscf.fci.direct_spin1 import make_rdm1s
    #from pyscf.fci.addons import make_trdm1s
    print("test make_trdm1s")
    
    
    rdm_wigner_2 = np.zeros((nstate,nstate,nmo,nmo), dtype='complex')
    #trdm_wigner = np.zeros((nstate*(nstate-1)//2,nmo,nmo))
    print("ref_nelecas=")
    print(ref_nelecas)
    for I in range(nstate):
        for J in range(nstate):
            cg = CG(S[I],ms[I], 1, 0, S[J],ms[J]).doit()
            cg = float(cg)
            rdm_aabb = trans_rdm1s(wfn[I],wfn[J],ncas,ref_nelecas[I])
            T_z = 1/np.sqrt(2) * (rdm_aabb[0] - rdm_aabb[1]) / cg
            rdm_wigner_2[I,J,ncore:ncore + ncas ,ncore:ncore + ncas] = T_z 
            




    from prism import qd_nevpt2
    ncas_so = ncas * 2
    rdm_ca_so = np.zeros((nstate, ncas_so, ncas_so))    
    for ind in range(nstate):
        rdm_ca_so[ind] =  qd_nevpt2.compute_rdm_ca_so(method.interface, wfn[ind], wfn[ind], ref_nelecas[ind], ref_nelecas[ind])

    print("calculate trdm...")
    trdm_ca_so = qd_nevpt2.compute_rdm_tcat_so(method.interface, wfn, ref_nelecas, offset = -1)
    
    rdm_wigner = np.zeros((nstate,nstate,nmo,nmo), dtype='complex')
    #trdm_wigner = np.zeros((nstate*(nstate-1)//2,nmo,nmo))

    for I in range(nstate):
        for J in range(nstate):
            cg = CG(S[I],ms[I], 1, 0, S[J],ms[J]).doit()
            cg = float(cg)
            if I==J:
                T_z = 1/np.sqrt(2) * (rdm_ca_so[I, ::2, ::2] - rdm_ca_so[J, 1::2, 1::2]) / cg
                #rdm_wigner[I,:ncore,:ncore] += np.identity(ncore)
                rdm_wigner[I,J,ncore:ncore + ncas ,ncore:ncore + ncas] = T_z 
                #rdm_wigner[I] = T_z 
            elif I>J:
                P = (I*(I-1))//2 + J
                T_z = 1/np.sqrt(2) * (trdm_ca_so[P, ::2, ::2] - trdm_ca_so[P, 1::2, 1::2]) / cg
                rdm_wigner[I,J,ncore:ncore + ncas ,ncore:ncore + ncas] = T_z 
            elif J>I:
                P = (J*(J-1))//2 + I
                trdm_ca_so_t = trdm_ca_so[P].T
                T_z = 1/np.sqrt(2) * (trdm_ca_so_t[::2, ::2] - trdm_ca_so_t[1::2, 1::2]) / cg
                rdm_wigner[I,J,ncore:ncore + ncas ,ncore:ncore + ncas] = T_z 
    
    I = 1
    J = 0
    P = (J*(J-1))//2 + I
    trdm_ca_so_t = trdm_ca_so[P].T
    print("trdm_ca_so[P, ::2, ::2]")
    print(trdm_ca_so_t[::2, ::2])
    print("rdm_wigner_2=")
    A = trans_rdm1s(wfn[I],wfn[J],ncas,ref_nelecas[I])
    print(A[0])
    #exit()
    #print(rdm_wigner[0,0])
    #print(np.trace(rdm_wigner[0,0]))
    print("compute Hso_mo...")

    h_soc = getSOC_integrals(method)
    h1_plus = (h_soc[0] + (1j*h_soc[1])) #/np.sqrt(2)
    h1_minus = (h_soc[0] - (1j*h_soc[1])) #/np.sqrt(2)
    h1_zero = h_soc[2]

    H_minus = np.einsum('pq,ijpq->ij',h1_minus,rdm_wigner) * (-1/2)
    H_zero = np.einsum('pq,ijpq->ij',h1_zero,rdm_wigner) / np.sqrt(2)
    H_plus = np.einsum('pq,ijpq->ij',h1_plus,rdm_wigner) / 2

    #n_spinstate = 0
    #for i in range(nstate):
    #    n_spinstate += S[i]*2 + 1
    #n_spinstate = int(n_spinstate)
    

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

    print(S_total)
    print(ms_total)
    print(I_total)
    nstate_total = len(S_total)

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
    print("\n Absolute energies in a.u |||| Excitation energies in a.u ||  eV ||  cm-1\n*****************************")
    for e in en_soc:
        print("%14.6f ||||  %14.6f  ||  %14.6f  ||   %8.2f"%((e), (e-en_soc[0]),((e-en_soc[0])*27.2114),((e-en_soc[0])*219474.63)))

            




