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

def compute_properties(interface, rdm_sf, en_soc, h_evec_soc, S,  method = None):
    '''
    Calculate magnetic properties required by flag in interface or method
    Note that flag should have gtensor, mag_av, sus_av, mag_vec, sus_tensor
    ********
    INPUT:
    rdm_sf(np.array, nstate*nstate*nmo*nmo): spin free 1st rdm (without spin-orbit coupling)
    en_soc (np.array, n): spin-orbit coupling energy
    h_evec_soc (np.array): spin-orbit coupling eigenvector
    S (list), spin quantum number of each state without spin-orbit coupling
    method: nevpt or qdnevpt object. If "None", using interface (SOC-CASSCF)

    OUTPUT:
    properties_mag(dictionary): All magnetic properties
    '''

    if method is None:
        method = interface

    properties_mag = {}                      # Dictionary to store computed magnetic properties
    properties_mag["magnetic_properties"] = True

    # Calculate magnetic dipole moment without spin-orbit coupling
    Mu_sf = mag_dip(interface, rdm_sf, S, origin_type = method.magnetic_origin_type)
    Mu    = np.einsum('ai,kib,bj->kaj',np.conj(h_evec_soc).T, Mu_sf, h_evec_soc)


    #g-tensor
    if method.gtensor:
        method.log.info("\nCalculating g-tensor...")
        target_state = method.gtensor_target_state
        if isinstance(target_state, int):
            target_state = [target_state]
        g_factor = []
        g_evector = []
        for I in target_state:
            g_fac, g_evec = gtensor(interface, S, Mu, target_index = I )
            g_factor.append(g_fac)
            g_evector.append(g_evec)
        properties_mag["g-factors"] = g_factor
        properties_mag["g-eigenvectors"] = g_evector


    #magnetic susceptibility
    if method.mag_av or  method.sus_av or  method.mag_vec or  method.sus_tensor:
        method.log.info("\nCalculating magnetic susceptibility or magnetization...")

        # Parameter
        h_s =method.step_h_s
        method.log.info("h_s= %.2f T" %(h_s))


        #Powder data information
        method.log.info("Import LebedevGrid in pyscf...")
        Powder_data_xyzw = interface.MakeAngularGrid_266()
        method.log.info("Number of LebedevGrid point: %s" % len(Powder_data_xyzw))


        ###Powder magnetization
        if method.mag_av:
            Bs_list = method.Bs_powder_M
            T_list  = method.T_powder_M
            M_av_all = powder_magnetization(interface,Powder_data_xyzw,Bs_list,T_list,en_soc,Mu,h_s)
            
            properties_mag["M_av"] = M_av_all


        ###Powder susceptibility
        if method.sus_av:
            Bs_list = method.Bs_powder_chi
            T_list  = method.T_powder_chi
            chi_av_all = powder_susceptibility(interface,Powder_data_xyzw,Bs_list,T_list,en_soc,Mu,h_s)
           
            properties_mag["chi_av"] = chi_av_all


        ###Vector magnetization
        if method.mag_vec:
            B_vec = method.B_vec_M
            Bs_list = method.Bs_vec_M
            T_list  = method.T_vec_M
            M_xyz_all = vector_magnetization(interface,B_vec,Bs_list,T_list,en_soc,Mu,h_s)

            properties_mag["M_xyz_all"] = M_xyz_all


        ###Tensor  susceptibility
        if method.sus_tensor:
            B_vec = method.B_vec_chi
            Bs_list = method.Bs_vec_chi
            T_list  = method.T_vec_chi
            chi_T_eval_all = tensor_susceptibility(interface,B_vec,Bs_list,T_list,en_soc,Mu,h_s)

            properties_mag["chi_T_eval_all"] = chi_T_eval_all

    return properties_mag


def print_mag_properties(interface, properties,  method = None):
    '''
    print out magnetic properties result
    ********
    INPUT:
    properties (dictionary): magnetic properties result
    method: nevpt or qdnevpt object. If "None", using interface (SOC-CASSCF)
    '''
    
    if method is None:
        method = interface

    Bs_powder_M   = method.Bs_powder_M
    T_powder_M    = method.T_powder_M
    Bs_powder_chi = method.Bs_powder_chi
    T_powder_chi  = method.T_powder_chi
    B_vec_M       = method.B_vec_M
    Bs_vec_M      = method.Bs_vec_M
    T_vec_M       = method.T_vec_M
    B_vec_chi     = method.B_vec_chi
    Bs_vec_chi    = method.Bs_vec_chi
    T_vec_chi     = method.T_vec_chi


    if "g-eigenvectors" in properties:
        G_evecs = properties["g-eigenvectors"]
        method.log.info("\nMagnetic g-tensor principal axes:")
        for G_evec in G_evecs:
            method.log.info("%s", np.array2string(G_evec, precision=6, suppress_small=True))
            
    if "g-factors" in properties:
        ge = interface.g_free_elec
        G_sq = properties["g-factors"]
        method.log.info("\nMagnetic g-factors (ge = %s):" % ge)
        for G_sq_en in G_sq:
            method.log.info("%14.6f, %14.6f, %14.6f" % (G_sq_en[0], G_sq_en[1], G_sq_en[2]))
            method.log.info("%14.6f, %14.6f, %14.6f (g-shift)" % (G_sq_en[0] - ge, G_sq_en[1] - ge, G_sq_en[2] - ge))
            method.log.info("%14.3f, %14.3f, %14.3f (g-shift, ppt)" % (1000 * (G_sq_en[0] - ge), 1000 * (G_sq_en[1] - ge), 1000 * (G_sq_en[2] - ge)))

    #SUS
    if "M_av" in properties:
        M_av_all = properties["M_av"]

        method.log.info("\nPowder_magnetization(Bohr magneton)" )
        method.log.info("-----------------------------------")
        method.log.info("TEMP(K)   B(T)    M(Bohr magneton)")
        method.log.info("-----------------------------------")
        for  I in range(len(T_powder_M)):
            T = T_powder_M[I]
            for K in range(len(Bs_powder_M)):
                Bs = Bs_powder_M[K]
                method.log.info("%6.2f  %8.2f %14.6f " % (T, Bs, M_av_all[I,K]))

    if "chi_av" in properties:
        chi_av_all = properties["chi_av"]

        method.log.info("\nPowder_susceptibility(cm3/mol)" )
        method.log.info("--------------------------------------------")
        method.log.info("TEMP(K)   B(T)         X_av          X_av*T")
        method.log.info("--------------------------------------------")

        for I in range(len(T_powder_chi)):
            T = T_powder_chi[I]
            for K in range(len(Bs_powder_chi)):
                Bs = Bs_powder_chi[K]
                method.log.info("%6.2f  %8.2f %14.6f %14.6f" % (T, Bs, chi_av_all[I,K],chi_av_all[I,K]*T))

    if "M_xyz_all" in properties:
        M_xyz_all = properties["M_xyz_all"]

        method.log.info("\nMagnetization vector (Bohr magneton) in B vector= %s",B_vec_M)
        method.log.info("--------------------------------------------------------")
        method.log.info("TEMP(K)   B(T)          Mx           My           Mz")
        method.log.info("--------------------------------------------------------")
    
        for I in range(len(T_vec_M)):
            T = T_vec_M[I]
            for K in range(len(Bs_vec_M)):
                Bs = Bs_vec_M[K]
                method.log.info("%6.2f  %8.2f %14.6f %12.6f %12.6f" % (T, Bs, M_xyz_all[I,K,0],M_xyz_all[I,K,1],M_xyz_all[I,K,2]))

    if "chi_T_eval_all" in properties:
        chi_T_eval_all = properties["chi_T_eval_all"]

        method.log.info("Susceptibility tensor X*T (cm3K/mol) in B vector= %s", B_vec_chi)
        method.log.info("\nEigenvalue of Susceptibility tensor * T (cm3K/mol)" )
        method.log.info("--------------------------------------------------------")
        method.log.info("TEMP(K)   B(T)         X1*T         X2*T         X3*T")
        method.log.info("--------------------------------------------------------")
    
        for I in range(len(T_vec_chi)):
            T = T_vec_chi[I]
            for K in range(len(Bs_vec_chi)):
                Bs = Bs_vec_chi[K]
                method.log.info("%6.2f  %8.2f %14.6f %12.6f %12.6f" % (T, Bs, chi_T_eval_all[I,K,0],chi_T_eval_all[I,K,1],chi_T_eval_all[I,K,2]))


# magnetic dipole moment in microstates basis without spin-orbit coupling
def mag_dip(interface, rdm_sf, S, origin_type = 'charge'):
    '''
    Calculate magnetic dipole moment matrix(L+geS) in microstates basis without spin-orbit coupling 
    ********
    INPUT:
    rdm_sf (np.array, nstate*nstate*nmo*nmo): spin free 1st rdm (without spin-orbit coupling)
    S (list): spin quantum number of each state without spin-orbit coupling
    origin_type (str or list, 3): origin_type for computing angular momentum. 

    OUTPUT:
    Mu_sf (np.array, n_micro_states*n_micro_states): magnetic dipole moment matrix
    '''

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
                    l_mat[0,I,J] = np.einsum('ij,ij',rdm_mo,l1_mo[0])
                    l_mat[1,I,J] = np.einsum('ij,ij',rdm_mo,l1_mo[1])
                    l_mat[2,I,J] = np.einsum('ij,ij',rdm_mo,l1_mo[2])

                    l_mat[0,J,I] = np.conj(l_mat[0,I,J]).T
                    l_mat[1,J,I] = np.conj(l_mat[1,I,J]).T
                    l_mat[2,J,I] = np.conj(l_mat[2,I,J]).T

    #calculate magnetic dipole moment (without spin-orbit coupling)
    Mu_sf = l_mat + s_mat * interface.g_free_elec 

    return Mu_sf


#Note that Mu should include spin orbit coupling 
def gtensor(interface, S, Mu, target_index = 1):
    '''
    Calculate gtensor by employing method from:
    J. Chem. Phys. 2026, 164 (17), 174117
    ********
    INPUT:
    S(list): spin quantum number of each state without spin-orbit coupling
    Mu (np.array, n*n);  magnetic dipole moment built by spin-orbit coupling basis
    target_index (integer): target state to calculate g-tensor. Note that the index refers to the spin-free states.

    OUTPUT:
    G_sq_en (np.array, 3): g-factor
    G_evec (np.array, 3*3): g-axis
    '''

    ge = interface.g_free_elec

    interface.log.info("\nTarget State index = %s", target_index)

    if target_index < 1 or target_index > len(S):
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

    interface.log.info("Micro state index: %s to %s", target_index_soc+1, (target_index_soc+target_multiplicity))
    interface.log.info("Calculating g-tensor for multiplicity = %s", target_multiplicity)

    # Define old J
    Hab_old = Mu[:,target_index_soc:target_index_soc+target_multiplicity,target_index_soc:target_index_soc+target_multiplicity]
 
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


def powder_magnetization(interface, powder_data, Bs_list, T_list, en_soc, Mu, h_s):
    '''
    Calculate powder magnetization(Bohr magneton)
    ********
    INPUT:
    powder_data (np.array, n*3): vector data for compute powder properties. Can obtain from interface.MakeAngularGrid_266()
    Bs_list (list): magnetic Field Strength. Unit: Telsa
    T_list (list): temperature. Unit: K
    en_soc (np.array, n): spin-orbit coupling energy
    Mu (np.array, n*n);  magnetic dipole moment built by spin-orbit coupling basis
    h_s: Magnetic field step size used for numerical differentiation. Unit: Telsa

    OUTPUT:
    M_av_all(np.array, T*Bs): powder magnetization(Bohr magneton)
    '''
    
    M_av_all = np.zeros((len(T_list),len(Bs_list)))
    for I in range(len(T_list)):
        T = T_list[I]
        for K in range(len(Bs_list)):
            Bs = Bs_list[K]
            M_av = 0
            for i in range(len(powder_data)):
                B_vec = powder_data[i,0:3]
                B_vec = np.reshape(B_vec,3)
                weight = powder_data[i,3]
                M_av += magnetization(interface,Bs,B_vec,en_soc,Mu,T,h_s) * weight
                
            M_av = float(M_av)
            M_av_all[I,K] =  M_av

    return M_av_all


def powder_susceptibility(interface, powder_data, Bs_list, T_list, en_soc, Mu, h_s):
    '''
    Calculate powder susceptibility(cm3/mol)
    ********
    INPUT:
    powder_data (np.array, n*3): vector data for compute powder properties. Can obtain from interface.MakeAngularGrid_266()
    Bs_list (list): magnetic Field Strength. Unit: Telsa
    T_list (list): temperature. Unit: K
    en_soc (np.array, n): spin-orbit coupling energy
    Mu (np.array, n*n):  magnetic dipole moment built by spin-orbit coupling basis
    h_s: Magnetic field step size used for numerical differentiation. Unit: Telsa

    OUTPUT:
    chi_av_all(np.array, T*Bs): powder susceptibility(cm3/mol)
    '''

    chi_av_all= np.zeros((len(T_list),len(Bs_list)))
    for I in range(len(T_list)):
        T = T_list[I]
        for K in range(len(Bs_list)):
            Bs = Bs_list[K]
            chi_av = 0
            for i in range(len(powder_data)):
                B_vec1  = powder_data[i,0:3]
                B_vec1  = np.reshape(B_vec1,3)
                weight = powder_data[i,3]
                chi_av += susceptibility(interface,Bs,B_vec1,en_soc,Mu,T,h_s,dB_k=B_vec1) * weight
            
            chi_av_all[I,K] = chi_av

    return chi_av_all


def vector_magnetization(interface, B_vec, Bs_list, T_list, en_soc, Mu, h_s):
    '''
    Calculate vector magnetization(Bohr magneton)
    ********
    INPUT:
    B_vec (np.array, 3): direction of magmetic field. "magnetization" function will normalize it. 
    Bs_list (list): magnetic Field Strength. Unit: Telsa
    T_list (list): temperature. Unit: K
    en_soc (np.array, n): spin-orbit coupling energy
    Mu (np.array, n*n);  magnetic dipole moment built by spin-orbit coupling basis
    h_s (float): Magnetic field step size used for numerical differentiation. Unit: Telsa

    OUTPUT:
    M_xyz_all (np.array, T*B*3): vector magnetization(Bohr magneton). The third index is x,y,z
    '''

    B_unit = np.eye(3)
    M_xyz_all= np.zeros((len(T_list),len(Bs_list),3))
    for I in range(len(T_list)):
        T = T_list[I]
        for K in range(len(Bs_list)):
            Bs = Bs_list[K]
            M_xyz_all[I,K,0] = magnetization(interface,Bs,B_vec,en_soc,Mu,T,h_s,dB_k=B_unit[0])
            M_xyz_all[I,K,1] = magnetization(interface,Bs,B_vec,en_soc,Mu,T,h_s,dB_k=B_unit[1])
            M_xyz_all[I,K,2] = magnetization(interface,Bs,B_vec,en_soc,Mu,T,h_s,dB_k=B_unit[2])

    return M_xyz_all


def tensor_susceptibility(interface, B_vec, Bs_list, T_list, en_soc, Mu, h_s):
    '''
    Calculate tensor susceptibility(cm3K/mol)
    ********
    INPUT:
    B_vec (np.array, 3): direction of magmetic field. "susceptibility" function will normalize it. 
    Bs_list (list): magnetic Field Strength. Unit: Telsa
    T_list (list): temperature. Unit: K
    en_soc (np.array, n): spin-orbit coupling energy
    Mu (np.array, n*n);  magnetic dipole moment built by spin-orbit coupling basis
    h_s (float): Magnetic field step size used for numerical differentiation. Unit: Telsa
    
    OUTPUT:
    chi_T_eval_all (np.array, T*Bs*3): tensor susceptibility eigenvalue * temperature(cm3K/mol).  The third index is x,y,z.
    '''

    B_unit = np.eye(3)
    chi_T_eval_all= np.zeros((len(T_list),len(Bs_list),3))
    for I in range(len(T_list)):
        T = T_list[I]
        for K in range(len(Bs_list)):
            Bs = Bs_list[K]
            chi = np.zeros((3,3))
            for k in range(3):
                for l in range(3):
                    if k==l:
                        B_k=B_unit[k]
                        chi[k,k]= susceptibility(interface,Bs,B_vec,en_soc,Mu,T,h_s,dB_k=B_k) 
                    else:
                        B_k=B_unit[k]
                        B_l=B_unit[l]
                        chi[k,l]= susceptibility(interface,Bs,B_vec,en_soc,Mu,T,h_s,dB_k=B_k,dB_l=B_l) 

            chi_T = chi * T
            interface.log.extra("TEMP(K)= %s, Bs(T)= %s" %(T, Bs))
            interface.log.extra("%s", np.array2string(chi_T, precision=8))
            chi_T_eval, chi_T_evec = np.linalg.eigh(chi_T)
            chi_T_eval_all[I,K] = chi_T_eval

    return chi_T_eval_all

    
def magnetization(interface,B_s,B_vec,en_soc,Mu,T,h_s,dB_k=None):
    '''
    Calculate magnetization(Bohr magneton) in specfic B and T
    ********
    INPUT:
    B_s (float): magnetic Field Strength. Unit: Telsa
    B_vec (np.array, 3): direction of magmetic field. "magnetization" function will normalize it. 
    T (float): temperature. Unit: K
    en_soc (np.array, n): spin-orbit coupling energy
    Mu (np.array, n*n);  magnetic dipole moment built by spin-orbit coupling basis
    h_s (float): Magnetic field step size used for numerical differentiation. Unit: Telsa
    dB_k (np.array, 3): Partial differential direction. Default follow  B_vec

    OUTPUT:
    MM (float):  magnetization(Bohr magneton)
    '''
    
    kb = interface.kb 
    mu_B = interface.mu_B_Eh

    n_micro_states = len(en_soc)
    B_vec = B_vec / np.linalg.norm(B_vec)
    if dB_k is None:
        dB_k = B_vec
    else:
        dB_k = dB_k / np.linalg.norm(dB_k)
    
    #Set zero pint energy
    zero = en_soc[0]
    en_soc_diff = en_soc - zero

    B_svec = B_s * B_vec
    en_ze, evec_ze =  E_ze(B_svec,en_soc_diff,Mu, mu_B)
    B1 = B_svec + dB_k * h_s
    B2 = B_svec - dB_k * h_s

    A1, evec_A1 = E_ze(B1,en_soc_diff,Mu, mu_B)
    A2, evec_A2 = E_ze(B2,en_soc_diff,Mu, mu_B)
    
    dE = np.zeros(n_micro_states)
    for i in range(n_micro_states):
        dE[i] = (A1[i]-A2[i]) / (2*h_s)
    #partial function
    Z = partition_function(en_ze,T, kb)
    
    MM=0
    for i in range(n_micro_states):
        MM -= dE[i] * np.exp(-(en_ze[i])/(T*kb))/Z / mu_B
    
    return MM #in Bohr magneton unit
    
   
def susceptibility(interface,B_s,B_vec,en_soc,Mu,T,h_s,dB_k=None,dB_l=None):
    '''
    Calculate susceptibility(cm3/mol) in specfic B and T
    ********
    INPUT:
    B_s (float): magnetic Field Strength. Unit: Telsa
    B_vec (np.array, 3): direction of magmetic field. "magnetization" function will normalize it. 
    T (float): temperature. Unit: K
    en_soc (np.array, n): spin-orbit coupling energy
    Mu (np.array, n*n);  magnetic dipole moment built by spin-orbit coupling basis
    h_s (float): Magnetic field step size used for numerical differentiation. Unit: Telsa
    dB_k (np.array, 3): Partial differential direction. Default follow  B_vec
    dB_l (np.array, 3): Partial differential direction. Default follow  dB_k

    OUTPUT:
    chi(float): susceptibility(cm3/mol)
    '''

    kb = interface.kb 
    mu_B_Eh = interface.mu_B_Eh
    mu_B_erg = interface.mu_B_erg
    NA = interface.NA
    T_to_G = interface.T_to_G

    B_vec = B_vec / np.linalg.norm(B_vec)

    if dB_k is None:
        dB_k = B_vec
    else:
        dB_k = dB_k / np.linalg.norm(dB_k)

    n_micro_states = len(en_soc)
    B_svec = B_s * B_vec
    en_ze, evec_ze =  E_ze(B_svec,en_soc,Mu, mu_B_Eh)

    #Set zero pint energy
    zero = en_soc[0]
    en_soc_diff = en_soc - zero

    #same direction
    if dB_l is None:
        B1 =[]
        B2 =[]
        B3 =[]
        B1 = B_svec + dB_k * h_s
        B2 = B_svec 
        B3 = B_svec - dB_k * h_s
        
        E_ze1, evec_ze1 = E_ze(B1,en_soc_diff,Mu, mu_B_Eh)  
        E_ze2, evec_ze2 = E_ze(B2,en_soc_diff,Mu, mu_B_Eh)  
        E_ze3, evec_ze3 = E_ze(B3,en_soc_diff,Mu, mu_B_Eh)  

        Z1 = partition_function(E_ze1,T, kb)
        Z2 = partition_function(E_ze2,T, kb)
        Z3 = partition_function(E_ze3,T, kb)

        lnZ1 = np.log(Z1)
        lnZ2 = np.log(Z2)
        lnZ3 = np.log(Z3)
        d2lnZ = (lnZ1-2*lnZ2+lnZ3)/(h_s**2)
        chi = kb * T * d2lnZ  / mu_B_Eh * mu_B_erg * NA / T_to_G

    else:
        dB_l = dB_l / np.linalg.norm(dB_l)
        B1 =[]
        B2 =[]
        B3 =[]
        B4 =[]
        
        B1 = B_svec + dB_k*h_s + dB_l*h_s
        B2 = B_svec + dB_k*h_s - dB_l*h_s
        B3 = B_svec - dB_k*h_s + dB_l*h_s
        B4 = B_svec - dB_k*h_s - dB_l*h_s

        E_ze1, evec_ze1 = E_ze(B1,en_soc_diff,Mu, mu_B_Eh)  
        E_ze2, evec_ze2 = E_ze(B2,en_soc_diff,Mu, mu_B_Eh)  
        E_ze3, evec_ze3 = E_ze(B3,en_soc_diff,Mu, mu_B_Eh)  
        E_ze4, evec_ze4 = E_ze(B4,en_soc_diff,Mu, mu_B_Eh)  

        Z1 = partition_function(E_ze1,T,kb)
        Z2 = partition_function(E_ze2,T,kb)
        Z3 = partition_function(E_ze3,T,kb)
        Z4 = partition_function(E_ze4,T,kb)
        

        lnZ1 = np.log(Z1)
        lnZ2 = np.log(Z2)
        lnZ3 = np.log(Z3)
        lnZ4 = np.log(Z4)

        d2lnZ = (lnZ1 - lnZ2 - lnZ3 + lnZ4)/(4*h_s**2)
    
        chi = kb * T * d2lnZ   / mu_B_Eh * mu_B_erg * NA / T_to_G 

    return chi #cm3/mole


def E_ze(B,en_soc,Mu,mu_B):
    '''
    Calculate energy state with magnrtic field
    ********
    INPUT:
    B (float): magnetic Field Strength. Unit: Telsa
    en_soc (np.array, n): spin-orbit coupling energy
    Mu (np.array, n*n):  magnetic dipole moment built by spin-orbit coupling basis
    mu_B: Bohr magneton. Can obtained from interface.mu_B_Eh (atomic unit)

    OUTPUT:
    en_ze (np.array,n_micro_states): energy with magnrtic field
    evec_ze (np.array, n_micro_states*n_micro_states):  wavefunction with magnrtic field
    '''

    B = np.array(B) 
    B = B *mu_B 
    H0 = np.diag(en_soc)
    Hze = np.einsum('k,kij->ij',B,Mu) 
    H_total = H0+Hze
    en_ze,evec_ze = np.linalg.eigh(H_total)

    return en_ze,evec_ze #Hartree

def partition_function(en,T,kb):    
    '''
    Calculate partition_function \sum^{n_state}_{i}exp(-(en_{i})/(T*kb))
    ********
    en (np.array, n): state energy
    T (float): temperature. Unit: K
    kb: Boltzmann constant. Can obtained from interface.kb (1.3806483e-23 / 4.3597447222060e-18 Eh/K)

    OUTPUT:
    Z (float): partition_function
    '''

    Z=0
    for i in range(len(en)):
        Z += np.exp(-(en[i])/(T*kb))

    return Z

