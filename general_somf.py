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
socutils_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'prism', 'socutils'))
if socutils_path not in sys.path:
    sys.path.insert(0, socuitls_path)

import somf

def getSOC_integrals(method):
    
    mol = method.interface.mol
    socints = []

    # Build 1e-density matrix:


    if method.soc is "breit-pauli":
        socints.append(somf.get_hso1e_bp(mol))
        socints.append(somf.get_fso2e_bp(mol, dm))
    else:
        socints.append(somf.get_hso1e_x2c1(mol))
        socints.append(somf.get_fso2e_x2c(mol, dm))

    return socints


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
