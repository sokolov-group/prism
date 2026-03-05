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
# Authors: Alexander Yu. Sokolov <alexander.y.sokolov@gmail.com>
#          Rajat S. Majumder <majumder.rajat071@gmail.com>
#          Nicholas Y. Chiang <nicholas.yiching.chiang@gmail.com>
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
from prism import qd_nevpt2
from prism import nevpt2
from prism import nevpt_compute
import prism.lib.logger as logger

# Add python path for socutils:
prism_path = os.path.dirname(os.path.abspath(__file__)) 
if prism_path not in sys.path:
      sys.path.insert(0, prism_path)

# Add socutils module
from socutils.somf import somf

def getSOC_integrals(method):
    mo = method.mo
    nmo = method.nmo
    nao = method.mo.shape[1]
    prefactor = 0.5 / ((LIGHT_SPEED)**2)
    mol = method.interface.mol

    # Build 1e-density matrix
    rdm1ao = method.interface.mc.make_rdm1() 

    if (method.soc=="Breit-Pauli" or method.soc=="BP"):
        xmol, contr_coeff = mol, np.eye(mol.nao_nr())
        nbasis = xmol.nao_nr()
        hsocint = np.zeros((3, nbasis, nbasis))
 
        hsocint += prefactor * somf.get_wso(xmol) #, unc=False)
        hsocint -= prefactor * somf.get_fso2e_bp(xmol, rdm1ao)

    elif (method.soc=="x2c1" or method.soc=="DKH1"):   
        xmol, contr_coeff = sfx2c1e.SpinFreeX2C(mol).get_xmol()
        nbasis = xmol.nao_nr()
        hsocint = np.zeros((3, nbasis, nbasis))
 
        rdm1ao = reduce(np.dot, (contr_coeff, rdm1ao, contr_coeff.T))
        hsocint += prefactor * somf.get_hso1e_x2c1(xmol, unc=False)
        hsocint -= prefactor * somf.get_fso2e_x2c(xmol, rdm1ao)
    
    elif (method.soc=="x2c2" or method.soc=="DKH2"):
        raise Exception("The sf-X2C-1e+so-DKH2 implementation is not complete yet in Prism.")
        xmol, contr_coeff = sfx2c1e.SpinFreeX2C(mol).get_xmol()
        nbasis = xmol.nao_nr()
        hsocint = np.zeros((3, nbasis, nbasis))
        
        rdm1ao = reduce(np.dot, (contr_coeff, rdm1ao, contr_coeff.T))
        hsocint += prefactor * somf.get_hso1e_x2c1(xmol, unc=False)
        hsocint += prefactor * get_hso1e_x2c2(xmol) 
        hsocint -= prefactor * somf.get_fso2e_x2c(xmol, rdm1ao)

    else:
        raise Exception("Incorrect SOC flag in input file!!")

    ### Recontract the basis:
    h_soc_all_contr = np.zeros((3, nao, nao))
    for comp in range(3):
        h_soc_all_contr[comp] = reduce(np.dot, (contr_coeff.T, hsocint[comp], contr_coeff))

    ### Convert to MO basis:
    h_soc_all_contr = np.einsum('xpq,pi,qj->xij',h_soc_all_contr, mo, mo)
    h_soc_total = np.zeros((3, nmo, nmo), dtype = 'complex')    
    for comp in range(3):
        h_soc_total[comp] =-1j*(h_soc_all_contr[comp].astype('complex'))

    return h_soc_total 


def state_interaction_SOC(method, en, rdm_aabb, S, ms):
    cput0 = (logger.process_clock(), logger.perf_counter())
    method.log.info("Spin-Free Framework: Employ Wigner–Eckart’s theorem")
    method.log.info("Consider spin-orbit coupling effect...")
    nmo = method.nmo
    nstate = len(en)

    if (method.soc=="Breit-Pauli" or method.soc=="BP"):
        soc = "Breit-Pauli"
    
    elif(method.soc=="x2c-1" or method.soc=="DKH1"):
        method.log.info("\nNote that SOC Hamiltionian is sf-X2C-1e+so-DKH1 instead of usual DKH. \n")  
        soc = "sf-X2C-1e+so-DKH1"
    
    elif(method.soc=="x2c-2" or method.soc=="DKH2"):
        raise Exception("The sf-X2C-1e+so-DKH2 implementation is not complete yet in Prism.")
    
    else:
        raise Exception("Incorrect SOC flag in input file!!")
        
    method.log.info("Calculate Wigner's rdm...")
    rdm_wigner = np.zeros((nstate,nstate,nmo,nmo), dtype='complex')
    for I in range(nstate):
        for J in range(nstate):
            cg = CG(S[I],ms[I], 1, 0, S[J],ms[J]).doit()
            cg = float(cg)
            if np.abs(cg) > 1e-5:               
                T_z = 1/np.sqrt(2) * (rdm_aabb[0,I,J] - rdm_aabb[1,I,J]) / cg
                rdm_wigner[I,J] = T_z 
            
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
    en_soc, evec_soc = np.linalg.eigh(HSOC+H_sf)


    #calculate Osc Str
    rdm_sf = rdm_aabb[0] + rdm_aabb[1]
    rdm_mo = np.zeros((nstate_total, nstate_total, nmo, nmo),dtype='complex')
    for I in range(nstate_total):
        for J in  range(nstate_total):
            rdm_mo[I,J] = rdm_sf[I_total[I], I_total[J]]
    rdm_mo_soc = np.einsum('ai,ibIJ,bj->ajIJ',np.conj(evec_soc).T , rdm_mo , evec_soc)
    osc_str_soc =    nevpt2.osc_strength(method, en_soc, rdm_mo_soc)  

    #debug osc: osc with soc has conflict up to 5 decimal place
    #method.osc_str_soc_test = osc_strength_general(method, en_soc, rdm_sf,(I_total,evec_soc))     
    
    h2ev = method.interface.hartree_to_ev
    h2cm = method.interface.hartree_to_inv_cm

    method.log.info("\nSummary of results for the SOC calculation with the %s Hamiltionian:" % (soc))
    method.log.info("-----------------------------------------------------------------------------------------------------------------")
    method.log.info("  State            E(total)           dE(a.u.)        dE(eV)      dE(nm)       dE(cm-1)      Osc Str. ")
    method.log.info("-----------------------------------------------------------------------------------------------------------------")
    
    e_gs = en_soc[0]

    for p in range(nstate_total):
        de = en_soc[p] - e_gs
        de_ev = de * h2ev
        de_cm = de * h2cm
        if p == 0 or abs(de) < 1e-5:
            method.log.info("%5d       %20.12f %14.8f %12.4f %12s %14.4f   %12s" % ((p+1), en_soc[p], de, de_ev, " ", de_cm, " "))
        else:
            de_nm = 10000000 / de_cm
            method.log.info("%5d       %20.12f %14.8f %12.4f %12.4f %14.4f   %12.8f" % ((p+1), en_soc[p], de, de_ev, de_nm, de_cm, osc_str_soc[p-1]))
    method.log.info("-----------------------------------------------------------------------------------------------------------------")
    
    if method.verbose >= 5:
        osc_str_full = osc_str_soc
        # Compute all transitions starting from each state
        for gs_index in range(1, len(en_soc)): 
            osc_str_full.extend(nevpt2.osc_strength(method, en_soc, rdm_mo_soc, gs_index))

        nevpt_compute.print_osc_str(method, en_soc, osc_str_full)
    
    sys.stdout.flush()
    method.log.timer0("total %s calculation" % soc, *cput0)
    
    return en_soc, evec_soc, osc_str_soc


def gtensor(method, evec_soc, rdm_sf, S,target_index = 0,origin_type = 'charge'):
    method.log.info("\nCalculating g-tensor...")
    mf = method.interface.mf
    mo = method.mo
    ncore = method.ncore 
    n_states = len(rdm_sf[0])
    n_micro_states = len(evec_soc)
    mo_coeff = method.mo
    nmo = method.nmo
    ncas = method.ncas

    # Calculate spin-free multiplicity
    multiplicity=[]
    for I in range(n_states):
        multiplicity.append(int(S[I]*2 + 1))
    
    # Calculate quantum number of spin-orbit state
    ms_total = []
    S_total = []
    I_total = []
    for I in range(n_states):
        for i in range(multiplicity[I]):
            ms_total.append(S[I]-i)
            S_total.append(S[I])
            I_total.append(I)

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
        s_mat[0,A:A+multiplicity[I],A:A+multiplicity[I]]=Sx
        
        #Sy
        Sy = -C/2 * 1j
        Sy = np.diag(Sy,1)
        Sy = Sy - Sy.T
        s_mat[1,A:A+multiplicity[I],A:A+multiplicity[I]]=Sy

        A += multiplicity[I]

    ###L part########
    if origin_type == 'charge':
        origin = [ 0, 0 ,0]
        total_charge = 0
        for atm in range(mf.mol.natm):
            origin += mf.mol.atom_coord(atm) * mf.mol.atom_charge(atm)
            total_charge += mf.mol.atom_charge(atm)
        origin = origin / total_charge
        method.log.info("origin(Bohr)= %s", origin) 
        mf.mol.set_common_orig(origin)
        l1_ao = -1j * mf.mol.intor('cint1e_cg_irxp_sph', comp=3)

    elif origin_type == 'atom1':
        method.log.info("origin(Bohr)= %s", mf.mol.atom_coord(0))
        mf.mol.set_common_orig(mf.mol.atom_coord(0))
        l1_ao = -1j * mf.mol.intor('cint1e_cg_irxp_sph', comp=3)
    
    elif origin_type == 'GIAO':
        method.log.info("origin=GIAO")
        l1_ao = -1j * mf.mol.intor('int1e_giao_irjxp_sph', comp=3)

    else:
        method.log.info("origin(Bohr)= %s", origin_type)
        mf.mol.set_common_orig(origin_type)
        l1_ao = -1j * mf.mol.intor('cint1e_cg_irxp_sph', comp=3)
    
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
                    rdm_mo += rdm_sf[i,j]

                    l_mat[0,I,J] += np.einsum('ij,ij',rdm_mo,l1_mo[0])
                    l_mat[1,I,J] += np.einsum('ij,ij',rdm_mo,l1_mo[1])
                    l_mat[2,I,J] += np.einsum('ij,ij',rdm_mo,l1_mo[2])

    l_mat[0] = l_mat[0] + np.conj(l_mat[0]).T
    l_mat[1] = l_mat[1] + np.conj(l_mat[1]).T
    l_mat[2] = l_mat[2] + np.conj(l_mat[2]).T

    method.log.info("Target State index = %s", target_index)
    S_target = S[target_index]
    target_multiplicity = int(2*S_target+1)

    target_index_soc = 0
    if target_index >= 1:
        for i in range(target_index):
            target_index_soc += int(2*S[i]+1)

    method.log.info("Calculating g-tensor for multiplicity = %s", target_multiplicity)
    Kramer_pair  = evec_soc[:,target_index_soc:target_index_soc+target_multiplicity] 
 
    # define unique Kramer's doublet
    S_old = np.einsum('ai,kib,bj->kaj',np.conj(Kramer_pair).T , s_mat , Kramer_pair)
    L_old = np.einsum('ai,kib,bj->kaj',np.conj(Kramer_pair).T , l_mat , Kramer_pair)
    Hab_old = L_old + S_old * 2.002319

    method.log.info("Use Jz to transform...")
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
      g[i,0] = np.real(Hab[i,0,1]) *2/np.sqrt(S_target*2)
      g[i,1] = np.imag(Hab[i,0,1]) *(-2)/np.sqrt(S_target*2)
      g[i,2] = np.real(Hab[i,0,0])/S_target
    
    G = np.einsum('km,lm->kl',g,g)
    method.log.info("g=")
    method.log.info("%s", np.array2string(g, precision=6, suppress_small=True))
    method.log.info("G=")
    method.log.info("%s", np.array2string(G, precision=6, suppress_small=True))

    G_en, G_evec = np.linalg.eigh(G)
    #G_tensor_en, G_tensor_evec = np.linalg.eigh(G_tensor)
    G_sq_en = np.sqrt(G_en)
    #G_tensor_sq_en = np.sqrt(G_tensor_en)
    method.log.info("magnetic axis=")
    method.log.info("%s", np.array2string(G_evec, precision=6, suppress_small=True))
    method.log.info("g-factor=")
    method.log.info("%14.6f, %14.6f, %14.6f, ge=2.002319"%(G_sq_en[0],G_sq_en[1],G_sq_en[2]))
    method.log.info("%14.6f, %14.6f, %14.6f"%(G_sq_en[0]-2.002319,G_sq_en[1]-2.002319,G_sq_en[2]-2.002319))
    method.log.info("%14.3f, %14.3f, %14.3f, ptt"%(1000*(G_sq_en[0]-2.002319),1000*(G_sq_en[1]-2.002319),1000*(G_sq_en[2]-2.002319)))

    return G_sq_en, G_evec

## For SOC osc debug:
def osc_strength_general(method, en, rdm_mo, I_evec_soc=None, gs_index = 0):
    ncore = method.ncore 
    dip_mom_ao = method.interface.dip_mom_ao
    mo_coeff = method.mo
    nmo = method.nmo
    ncas = method.ncas
    dip_mom_mo = np.zeros_like(dip_mom_ao)
    n_states = len(en)

    #If spin-free:
    if I_evec_soc is None:
        I_total=np.arange(n_states)
        evec_soc = np.identity(n_states)
    else:
        I_total = I_evec_soc[0]
        evec_soc = I_evec_soc[1]
    # Transform dipole moments from AO to MO basis
    for d in range(dip_mom_ao.shape[0]):
        dip_mom_mo[d] = mo_coeff.T @ dip_mom_ao[d] @ mo_coeff
    # List to store Osc. Strength Values
    osc_total = []
    # Looping over CAS States
    for state in range(gs_index + 1, n_states):
        # Reset final transformed RDM
        rdm_qd = np.zeros((nmo, nmo),dtype='complex')
        # Looping over states i,j    
        for i in range(n_states):
            for j in range(n_states):
                I = I_total[i]
                J = I_total[j]
                rdm_qd += np.conj(evec_soc)[i, state] * rdm_mo[I,J] * evec_soc[j, gs_index]
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


## DKH-2 specific functionalities:

def get_hxr_dkh2(mol):

    c = LIGHT_SPEED
    t = mol.intor_symmetric('int1e_kin')
    v = mol.intor_symmetric('int1e_nuc')
    s = mol.intor_symmetric('int1e_ovlp')
    w = mol.intor_symmetric('int1e_pnucp')
    h1p, x, rp, h1m, xb, rm = _x2c1e_hxrmat_dkh2(t, v, w, s, c)
    
    return s, t, h1p, x, rp, h1m, xb, rm


def _x2c1e_hxrmat_dkh2(t, v, w, s, c):
    nao = s.shape[0]
    n2 = nao * 2
    h = np.zeros((n2, n2), dtype=v.dtype)
    m = np.zeros((n2, n2), dtype=v.dtype)
    h[:nao, :nao] = v
    h[:nao, nao:] = t
    h[nao:, :nao] = t
    h[nao:, nao:] = w * (.25 / c**2) - t
    m[:nao, :nao] = s
    m[nao:, nao:] = t * (.5 / c**2)

    e, a = scipy.linalg.eigh(h, m)
    cl = a[:nao, nao:]
    cs = a[nao:, nao:]

    b = np.dot(cl, cl.T.conj())
    x = reduce(np.dot, (cs, cl.T.conj(), np.linalg.inv(b)))

    s1 = s + reduce(np.dot, (x.T.conj(), t, x)) * (.5 / c**2)
    #tx = reduce(np.dot, (t, x))
    h1 = (h[:nao, :nao] + h[:nao, nao:].dot(x) + x.T.conj().dot(h[nao:, :nao]) +
          reduce(np.dot, (x.T.conj(), h[nao:, nao:], x)))

    sa = x2c._invsqrt(s)
    sb = x2c._invsqrt(reduce(np.dot, (sa, s1, sa)))
    r = reduce(np.dot, (sa, sb, sa, s))
    h1_plus = reduce(np.dot, (r.T.conj(), h1, r))

    # Add h_minus, l_minus, r_minus, x_bar: ADDED BY RAJAT
    x_bar = -(reduce(np.dot, (np.linalg.inv(s), x.T.conj(), (0.5 / c**2)*t))) 
    
    s_min_bar = (0.5 / c**2)*t + reduce(np.dot, (x_bar.T.conj(), s, x_bar))
    s2_invsqrt = x2c._invsqrt((0.5/c**2)*t)
    s2_smin_s2_invsqrt = x2c._invsqrt(reduce(np.dot, (s2_invsqrt, s_min_bar, s2_invsqrt)))
    r_min = reduce(np.dot, (s2_invsqrt, s2_smin_s2_invsqrt, s2_invsqrt, ((0.5/c**2)*t)))
    
    l_min = (h[nao:, nao:] + h[nao:, :nao].dot(x_bar) + (x_bar.T.conj()).dot(h[:nao, nao:]) + 
              reduce(np.dot, (x_bar.T.conj(), h[:nao, :nao], x_bar)))
    h1_min = reduce(np.dot, (r_min.T.conj(), l_min, r_min))

    return h1_plus, x, r, h1_min, x_bar, r_min


def get_hso1e_x2c2(mol):
    '''One electron DKH-2 type operator'''
    s, t, h1p, x, rp, h1m, xb, rm = get_hxr_dkh2(mol)
    c = LIGHT_SPEED
    ep, cp = scipy.linalg.eigh(h1p, s)
    em, cm = scipy.linalg.eigh(h1m, ((0.5/c**2)*t))
    
    #Build o1:
    nb = x.shape[0]
    O1 = np.zeros((3, nb, nb))
    o1 = np.zeros((3, nb, nb))
    wso = somf.get_wso(mol)
    for ic in range(3):
        O1[ic] = (0.25/c**2) * reduce(np.dot, (rp.T.conj(), x.T.conj(), wso[ic], rm))
        o1[ic] = reduce(np.dot, (cp.T.conj(), O1[ic], cm))
    
    # Construct w1:
    w1 = np.zeros((3, nb, nb))
    for ic in range(3):
        for p in range(nb):
            for q in range(nb):
                w1[ic, p, q] -= o1[ic, p, q]/(em[q] - ep[p])

    del o1
    # Construct W1:
    W1 = np.zeros((3, nb, nb))
    for ic in range(3):
        W1[ic] = (0.5/c**2) * reduce(np.dot, (s, cp, w1[ic], cm.T.conj(), t))

    
    # Construct h1e_sd2_plus: Eq. 91 
    tinv_w1 = np.zeros((3, nb, nb))
    tinv_o1 = np.zeros((3, nb, nb))
    for ic in range(3):
        tinv_w1[ic] = np.dot(np.linalg.inv(t), W1[ic].T.conj())
        tinv_o1[ic] = np.dot(np.linalg.inv(t), O1[ic].T.conj())

    h1e_sox2c2 = np.zeros((3, nb, nb))
    h1e_sox2c2[0] = np.dot(O1[1],tinv_w1[2]) - np.dot(O1[2],tinv_w1[1])
    h1e_sox2c2[1] = np.dot(O1[2],tinv_w1[0]) - np.dot(O1[0],tinv_w1[2])
    h1e_sox2c2[2] = np.dot(O1[0],tinv_w1[1]) - np.dot(O1[1],tinv_w1[0])
    
    h1e_sox2c2[0] += np.dot(W1[1],tinv_o1[2]) - np.dot(W1[2],tinv_o1[1])
    h1e_sox2c2[1] += np.dot(W1[2],tinv_o1[0]) - np.dot(W1[0],tinv_o1[2])
    h1e_sox2c2[2] += np.dot(W1[0],tinv_o1[1]) - np.dot(W1[1],tinv_o1[0])

    # Sanity checks:
    print(np.linalg.norm(h1e_sox2c2[0] + h1e_sox2c2[0].T.conj()))
    print(np.linalg.norm(h1e_sox2c2[1] + h1e_sox2c2[1].T.conj()))
    print(np.linalg.norm(h1e_sox2c2[2] + h1e_sox2c2[2].T.conj()))

    return h1e_sox2c2


def get_hsf1e_x2c2(mol, unc=True):
    
    if unc:
        xmol, contr_coeff = sfx2c1e.SpinFreeX2C(mol).get_xmol()

    '''One electron DKH-2 type operator'''
    s, t, h1p, x, rp, h1m, xb, rm = get_hxr(mol)
    c = LIGHT_SPEED
    ep, cp = scipy.linalg.eigh(h1p, s)
    em, cm = scipy.linalg.eigh(h1m, ((0.5/c**2)*t))

    #Build o1:
    nb = x.shape[0]
    O1 = np.zeros((3, nb, nb))
    o1 = np.zeros((3, nb, nb))
    wso = somf.get_wso(xmol)
    for ic in range(3):
        O1[ic] = (0.25/c**2) * reduce(np.dot, (rp.T.conj(), x.T.conj(), wso[ic], rm))
        o1[ic] = reduce(np.dot, (cp.T.conj(), O1[ic], cm))
    
    # Construct w1:
    w1 = np.zeros((3, nb, nb))
    for ic in range(3):
        for p in range(nb):
            for q in range(nb):
                w1[ic, p, q] -= o1[ic, p, q]/(em[q] - ep[p])

    del o1
    # Construct W1:
    W1 = np.zeros((3, nb, nb))
    for ic in range(3):
        W1[ic] = (0.5/c**2) * reduce(np.dot, (s, cp, w1[ic], cm.T.conj(), t))


    tinv_w1 = np.zeros((3, nb, nb))
    tinv_o1 = np.zeros((3, nb, nb))
    for ic in range(3):
        tinv_w1[ic] = np.dot(np.linalg.inv(t), W1[ic].T.conj())
        tinv_o1[ic] = np.dot(np.linalg.inv(t), O1[ic].T.conj())

    # Construct h1e_sf2_plus: Eq. 90
    h1e_sf2 = (np.dot(O1[0],tinv_w1[0]) + np.dot(O1[1],tinv_w1[1]) + np.dot(O1[2],tinv_w1[2]))+(np.dot(W1[0],tinv_o1[0]) + np.dot(W1[1],tinv_o1[1]) + np.dot(W1[2],tinv_o1[2]))
    h1e_sf2 = (c**2) * h1e_sf2

    # Recontract:
    h1e_sf2 = reduce(np.dot, (contr_coeff.T, h1e_sf2, contr_coeff))
    h1e_sf1 = reduce(np.dot, (contr_coeff.T, h1p, contr_coeff))
   
    return h1e_sfx2c1, h1e_sfx2c2


