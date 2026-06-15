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
# Authors: Alexander Yu. Sokolov <alexander.y.sokolov@gmail.com>
#          Rajat S. Majumder <majumder.rajat071@gmail.com>
#          Nicholas Y. Chiang <nicholas.yiching.chiang@gmail.com>
#

import numpy as np

# magnetic dipole moment in microstates basis without spin-orbit coupling
def mag_dip(interface, rdm_sf, S, origin_type = 'charge'):
    interface.log.info("Calculating magnetic dipole moment...")
    mf = interface.mf
    mo = interface.mo
    n_states = len(rdm_sf[0])
    
    # Calculate spin-free multiplicity
    multiplicity=[]
    for I in range(n_states):
        multiplicity.append(int(S[I]*2 + 1))
    
    # Calculate n_micro_states
    n_micro_states = sum(multiplicity)

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

    #Sz
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
    if isinstance(origin_type, str):
        origin_type = origin_type.lower()

    if origin_type == 'charge':
        origin = [ 0, 0 ,0]
        total_charge = 0
        for atm in range(mf.mol.natm):
            origin += mf.mol.atom_coord(atm) * mf.mol.atom_charge(atm)
            total_charge += mf.mol.atom_charge(atm)
        origin = origin / total_charge
        interface.log.info("Coordinate system origin (charge, Bohr) = %s", origin) 
        mf.mol.set_common_orig(origin)
        l1_ao = -1j * mf.mol.intor('cint1e_cg_irxp_sph', comp=3)

    elif origin_type == 'atom1':
        interface.log.info("Coordinate system origin (atom1, Bohr) = %s", mf.mol.atom_coord(0))
        mf.mol.set_common_orig(mf.mol.atom_coord(0))
        l1_ao = -1j * mf.mol.intor('cint1e_cg_irxp_sph', comp=3)
    
    elif origin_type == 'giao':
        interface.log.info("Using GIAO to compute gauge-invariant g-tensor...")
        l1_ao = -1j * mf.mol.intor('int1e_giao_irjxp_sph', comp=3)

    else:
        interface.log.info("Coordinate system origin (Bohr) = %s", origin_type)
        mf.mol.set_common_orig(origin_type)
        l1_ao = -1j * mf.mol.intor('cint1e_cg_irxp_sph', comp=3)
    
    # AO -> MO basis:
    l1_mo = np.einsum('xpq,pi,qj->xij',l1_ao,mo,mo) 
    l_mat = np.zeros((3,n_micro_states,n_micro_states), dtype='complex')
    for I in range(n_micro_states):
        for J in range(n_micro_states):
            if J>=I:
                if  (np.abs(S_total[I]-S_total[J])<1e-8) and (np.abs(ms_total[I]-ms_total[J])<1e-8):
                    i = I_total[I]
                    j = I_total[J]
  
                    rdm_mo = rdm_sf[i,j]
                    l_mat[0,I,J] += np.einsum('ij,ij',rdm_mo,l1_mo[0])
                    l_mat[1,I,J] += np.einsum('ij,ij',rdm_mo,l1_mo[1])
                    l_mat[2,I,J] += np.einsum('ij,ij',rdm_mo,l1_mo[2])

    l_mat[0] = l_mat[0] + np.conj(l_mat[0]).T
    l_mat[1] = l_mat[1] + np.conj(l_mat[1]).T
    l_mat[2] = l_mat[2] + np.conj(l_mat[2]).T

    #calculate magnetic dipole moment (without spin-orbit coupling)
    Mu = l_mat + s_mat * interface.g_free_elec 

    return Mu


def gtensor(interface, evec_soc, rdm_sf, S, target_index = 1, origin_type = 'charge'):
    ge = interface.g_free_elec

    interface.log.info("\nTarget State index = %s", target_index)

    if target_index < 1 or target_index > rdm_sf.shape[0]:
        raise ValueError("Target index must be between 1 and the number of states")

    target_index -= 1

    S_target = S[target_index]
    target_multiplicity = int(2*S_target+1)

    if (target_multiplicity == 1):
        interface.log.info("Skip g-tensor calculation due to multiplicity=1")
        G_sq_en = np.array([ge, ge, ge])
        G_evec  = np.identity(3)
        return G_sq_en, G_evec
    
    #Colloect Kramer_pair
    target_index_soc = 0
    if target_index >= 1:
        for i in range(target_index):
            target_index_soc += int(2*S[i]+1)
    Kramer_pair  = evec_soc[:,target_index_soc:target_index_soc+target_multiplicity] 

    interface.log.info("Micro state index: %s to %s", target_index_soc+1, (target_index_soc+target_multiplicity))
    interface.log.info("Calculating g-tensor for multiplicity = %s", target_multiplicity)

    # Calculate magnetic dipole moment
    Mu = mag_dip(interface, rdm_sf, S, origin_type)

    # Define old J
    Hab_old = np.einsum('ai,kib,bj->kaj',np.conj(Kramer_pair).T, Mu, Kramer_pair)
 
    # J in new Kramer_pair basis (Use Jz to transform...)
    z_en, z_evec = np.linalg.eigh(Hab_old[2])
    Hab_new = np.einsum('ai,kib,bj->kaj',np.conj(z_evec).T ,Hab_old, z_evec)
    Hab = Hab_new
 
    g=np.zeros([3,3])
    for i in range(3):
      g[i,0] = np.real(Hab[i,0,1]) *2/np.sqrt(S_target*2)
      g[i,1] = np.imag(Hab[i,0,1]) *(-2)/np.sqrt(S_target*2)
      g[i,2] = np.real(Hab[i,0,0])/S_target
    
    G = np.einsum('km,lm->kl',g,g)
    interface.log.extra("g=")
    interface.log.extra("%s", np.array2string(g, precision=6, suppress_small=True))
    interface.log.extra("G=")
    interface.log.extra("%s", np.array2string(G, precision=6, suppress_small=True))

    G_en, G_evec = np.linalg.eigh(G)
    G_sq_en = np.sqrt(G_en)

    return G_sq_en, G_evec