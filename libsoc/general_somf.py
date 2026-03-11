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
from sympy.physics.quantum.cg import CG
from prism.tools import transition
import prism.lib.logger as logger

# Add python path for socutils:
prism_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if prism_path not in sys.path:
    sys.path.insert(0, prism_path)

# Add socutils module
from socutils.somf import somf

def get_soc_integrals(interface,soc):
    mo = interface.mo
    nmo = interface.nmo
    nao = interface.mo.shape[1]
    prefactor = 0.5 / ((interface.light_speed)**2)
    mol = interface.mol
    xmol = interface.xmol
    contr_coeff = interface.contr_coeff

    # Build 1e-density matrix
    rdm1ao = interface.mc.make_rdm1() 

    soc = soc.lower()
    if (soc=="breit-pauli" or soc=="bp"):
        nbasis = mol.nao_nr()
        hsocint = np.zeros((3, nbasis, nbasis)) 
        hsocint += prefactor * somf.get_wso(mol)
        hsocint -= prefactor * somf.get_fso2e_bp(mol, rdm1ao)

    elif (soc=="x2c1" or soc=="dkh1"):
        nbasis_unc = xmol.nao_nr()
        nbasis = mol.nao_nr()
        hsocint = np.zeros((3, nbasis, nbasis))

        if (nbasis != nbasis_unc):
            hsocint_unc = np.zeros((3, nbasis_unc, nbasis_unc))
            rdm1ao = reduce(np.dot, (contr_coeff, rdm1ao, contr_coeff.T))
            hsocint_unc += prefactor * somf.get_hso1e_x2c1(xmol)
            hsocint_unc -= prefactor * somf.get_fso2e_x2c(xmol, rdm1ao)
            hsocint = np.einsum('pi,xpq,qj->xij', contr_coeff, hsocint_unc, contr_coeff)

        else:
            hsocint += prefactor * somf.get_hso1e_x2c1(mol)
            hsocint -= prefactor * somf.get_fso2e_x2c(mol, rdm1ao)
       
    else:
        raise Exception("Incorrect SOC flag in input file!!")

    ### Convert to MO basis:
    hsoc_mo = np.einsum('xpq,pi,qj->xij', hsocint, mo, mo)
    hsoc = np.zeros((3, nmo, nmo), dtype = 'complex')    
    for comp in range(3):
        hsoc[comp] = -1j*(hsoc_mo[comp].astype('complex'))

    return hsoc 


def state_interaction_soc(interface, en, rdm_aabb, S, ms, soc = "breit-pauli",verbose = 4):
    cput0 = (logger.process_clock(), logger.perf_counter())
    interface.log.info("Spin-Free Framework: Employ Wigner–Eckart’s theorem")
    interface.log.info("Consider spin-orbit coupling effect...")
    nmo = interface.nmo
    nstate = len(en)

    #Make sure SOC flag:
    soc = soc.lower()
    if (soc=="breit-pauli" or soc=="bp"):
        soc_name = "Breit-Pauli"
    
    elif(soc=="x2c-1" or soc=="dkh1"):
        interface.log.info("\nNote that SOC Hamiltionian is sf-X2C-1e+so-DKH1 instead of usual DKH. \n")  
        soc_name = "sf-X2C-1e+so-DKH1"
    
    elif(soc=="x2c-2" or soc=="dkh2"):
        raise Exception("The sf-X2C-1e+so-DKH2 implementation in Prism is still incomplete.")
    
    else:
        raise Exception("Incorrect SOC flag in input file!!")
        
    interface.log.info("Calculate Wigner's rdm...")
    rdm_wigner = np.zeros((nstate,nstate,nmo,nmo), dtype='complex')
    for I in range(nstate):
        for J in range(nstate):
            cg = CG(S[I], ms, 1, 0, S[J], ms).doit()
            cg = float(cg)
            if np.abs(cg) > 1e-5:               
                T_z = 1/np.sqrt(2) * (rdm_aabb[0,I,J] - rdm_aabb[1,I,J]) / cg
                rdm_wigner[I,J] = T_z 
            
    # Get SOC integrals:
    h_soc = get_soc_integrals(interface, soc)
    
    h1_plus = (h_soc[0] + (1j*h_soc[1])) 
    h1_minus = (h_soc[0] - (1j*h_soc[1])) 
    h1_zero = h_soc[2]

    H_minus = np.einsum('pq,ijpq->ij',h1_minus,rdm_wigner) * (-1/2)
    H_zero = np.einsum('pq,ijpq->ij',h1_zero,rdm_wigner) / np.sqrt(2)
    H_plus = np.einsum('pq,ijpq->ij',h1_plus,rdm_wigner) / 2

    # Calculate quantum number 
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

    sys.stdout.flush()
    interface.log.timer0("total %s calculation" % soc, *cput0)

    return en_soc, evec_soc


## DKH-2 specific functionalities:
def get_hxr_dkh2(method):

    mol = method.interface.mol
    c = method.interface.light_speed
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

    sa = scipy.linalg.invsqrt(s)
    sb = scipy.linalg.invsqrt(reduce(np.dot, (sa, s1, sa)))
    r = reduce(np.dot, (sa, sb, sa, s))
    h1_plus = reduce(np.dot, (r.T.conj(), h1, r))

    # Add h_minus, l_minus, r_minus, x_bar: ADDED BY RAJAT
    x_bar = -(reduce(np.dot, (np.linalg.inv(s), x.T.conj(), (0.5 / c**2)*t))) 
    
    s_min_bar = (0.5 / c**2)*t + reduce(np.dot, (x_bar.T.conj(), s, x_bar))
    s2_invsqrt = scipy.linalg.invsqrt((0.5/c**2)*t)
    s2_smin_s2_invsqrt = scipy.linalg.invsqrt(reduce(np.dot, (s2_invsqrt, s_min_bar, s2_invsqrt)))
    r_min = reduce(np.dot, (s2_invsqrt, s2_smin_s2_invsqrt, s2_invsqrt, ((0.5/c**2)*t)))
    
    l_min = (h[nao:, nao:] + h[nao:, :nao].dot(x_bar) + (x_bar.T.conj()).dot(h[:nao, nao:]) + 
              reduce(np.dot, (x_bar.T.conj(), h[:nao, :nao], x_bar)))
    h1_min = reduce(np.dot, (r_min.T.conj(), l_min, r_min))

    return h1_plus, x, r, h1_min, x_bar, r_min


def get_hso1e_x2c2(method):
    '''One electron DKH-2 type operator'''
    mol = method.interface.mol
    s, t, h1p, x, rp, h1m, xb, rm = get_hxr_dkh2(mol)
    c = method.interface.light_speed
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
    method.log.info("Check DKH-2 h1e Matrix Hemiticity")
    method.log.info("Norm of X-comp: %14.6f" % (np.linalg.norm(h1e_sox2c2[0] + h1e_sox2c2[0].T.conj())))
    method.log.info("Norm of Y-comp: %14.6f" % (np.linalg.norm(h1e_sox2c2[1] + h1e_sox2c2[1].T.conj())))
    method.log.info("Norm of X-comp: %14.6f" % (np.linalg.norm(h1e_sox2c2[2] + h1e_sox2c2[2].T.conj())))

    return h1e_sox2c2


def get_hsf1e_x2c2(method, unc=True):
    
    mol = method.interface.mol
    if unc:
        xmol = method.xmol
        contr_coeff = method.contr_coeff

    '''One electron DKH-2 type operator'''
    s, t, h1p, x, rp, h1m, xb, rm = get_hxr(mol)
    c = method.interface.light_speed
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


