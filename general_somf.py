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
from pyscf.lib.parameters import LIGHT_SPEED
from pyscf.x2c import sfx2c1e
from pyscf.x2c import x2c
from .socutils.somf import somf

def getSOC_integrals(method, unc = None):
    
    prefactor = 0.5 / ((LIGHT_SPEED)**2)
    mol = method.interface.mol

    # Build 1e-density matrix
    rdm1ao = method.interface.mc.make_rdm1() 

    if unc:
        xmol, contr_coeff = sfx2c1e.SpinFreeX2C(mol).get_xmol()
        rdm1ao = reduce(numpy.dot, (contr_coeff, rdm1ao, contr_coeff.T))
    else:
        xmol, contr_coeff = method.mol, numpy.eye(method.mol.nao_nr())

    nbasis = xmol.nao_nr()
    hsocint = np.zeros((3, nbasis, nbasis))

    if method.soc is "breit-pauli":
        hsocint += prefactor * somf.get_wso(xmol)
        hsocint -= prefactor * somf.get_fso2e_bp(xmol, rdm1ao)
    elif method.soc is "x2c":
        hsocint += prefactor * somf.get_hso1e_x2c1(xmol)
        hsocint -= prefactor * somf.get_fso2e_x2c(xmol, rdm1ao)
    else:
        raise Exception("Incorrect SOC flag in input file!!")

    return hsocint


def Initialize_SOC(method):

    if method.evec_qdnevpt2 is None:
        print("It is NEVPT2")
    else:
        print("It is QDNEVPT2")


    ref_wfn =method.ref_wfn  
    ncas = method.ncas 
    ref_nelecas = method.ref_nelecas 
    ref_wfn_spin_mult = method.ref_wfn_spin_mult
    nstate = len(ref_wfn)
    evec = method.evec_qdnevpt2

    print(type(ref_wfn))
    #wfn = np.zeros((nstate, len(ref_wfn[0]), len(ref_wfn[1])))
    #for I in range(nstate)
    #evec[:,0] = np.array([0,0,1,0,0,0])
    #print(evec)
    wfn = np.einsum('ij,iab->jab',evec,ref_wfn)
    #print(wfn[0]-1*ref_wfn[2])
    #print("!!!!")
    
    ms = []
    print("nstate=",nstate)
    
    for I in range(nstate):
        sz = method.interface.apply_S_z(wfn[I],ncas,ref_nelecas[I])
        ms.append(np.dot(wfn[I].ravel(), sz.ravel()))

    ms = [round(elem,2) for elem in ms]

    print(ms)
    print(">?")
