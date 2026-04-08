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

                    l_mat[0,J,I] += np.conj(l_mat[0,I,J]).T
                    l_mat[1,J,I] += np.conj(l_mat[1,I,J]).T
                    l_mat[2,J,I] += np.conj(l_mat[2,I,J]).T

    #calculate magnetic dipole moment (without spin-orbit coupling)
    Mu_sf = l_mat + s_mat * interface.g_free_elec 

    return Mu_sf


def gtensor(interface, S, Mu, target_index = 1, origin_type = 'charge'):
    ge = interface.g_free_elec

    interface.log.info("\nTarget State index = %s", target_index)
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


def Powder_magnetization(interface, powder_data, Bs_list, T_list, en_soc, Mu, h_s):
    
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
                #print(B_vec)
                M_av += magnetization(interface,Bs,B_vec,en_soc,Mu,T,h_s) * weight
                
            M_av = float(M_av)
            M_av_all[I,K] =  M_av

    print("\nPowder_magnetization(Bohr magneton)" )
    print("-----------------------------------")
    print("TEMP(K)   B(T)    M(Bohr magneton)")
    print("-----------------------------------")
    
    for  I in range(len(T_list)):
        T = T_list[I]
        for K in range(len(Bs_list)):
            Bs = Bs_list[K]
            print("%6.2f  %8.2f %14.6f " % (T, Bs, M_av_all[I,K]))
    
    return M_av_all


def Powder_susceptibility(interface, powder_data, Bs_list, T_list, en_soc, Mu, h_s):
    
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
                #M_av += magnetization(H,B_vec,en_soc,J,T) * weight[i]
                chi_av += susceptibility(interface,Bs,B_vec1,en_soc,Mu,T,h_s,dB_k=B_vec1) * weight
            
            chi_av_all[I,K] = chi_av

    
    print("\nPowder_susceptibility(cm3/mol)" )
    print("--------------------------------------------")
    print("TEMP(K)   B(T)         X_av          X_av*T")
    print("--------------------------------------------")
    
    for  I in range(len(T_list)):
        T = T_list[I]
        for K in range(len(Bs_list)):
            Bs = Bs_list[K]
            print("%6.2f  %8.2f %14.6f %14.6f" % (T, Bs, chi_av_all[I,K],chi_av_all[I,K]*T))

    return chi_av_all


def vector_magnetization(interface, B_vec, Bs_list, T_list, en_soc, Mu, h_s):
    B_unit = np.eye(3)
    M_xyz_all= np.zeros((len(T_list),len(Bs_list),3))
    for I in range(len(T_list)):
        T = T_list[I]
        for K in range(len(Bs_list)):
            Bs = Bs_list[K]
            M_xyz_all[I,K,0] = magnetization(interface,Bs,B_vec,en_soc,Mu,T,h_s,dB_k=B_unit[0])
            M_xyz_all[I,K,1] = magnetization(interface,Bs,B_vec,en_soc,Mu,T,h_s,dB_k=B_unit[1])
            M_xyz_all[I,K,2] = magnetization(interface,Bs,B_vec,en_soc,Mu,T,h_s,dB_k=B_unit[2])

    
    print("\nMagnetization vector (Bohr magneton) in B vector=",B_vec)
    print("--------------------------------------------------------")
    print("TEMP(K)   B(T)          Mx           My           Mz")
    print("--------------------------------------------------------")
    
    for  I in range(len(T_list)):
        T = T_list[I]
        for K in range(len(Bs_list)):
            Bs = Bs_list[K]
            print("%6.2f  %8.2f %14.6f %12.6f %12.6f" % (T, Bs, M_xyz_all[I,K,0],M_xyz_all[I,K,1],M_xyz_all[I,K,2]))

    return M_xyz_all


def tensor_susceptibility(interface, B_vec, Bs_list, T_list, en_soc, Mu, h_s):
    print("Susceptibility tensor X*T (cm3/mol) in B vector=", B_vec)
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
            print("TEMP(K)=",T," Bs(T)=",Bs)
            print(chi_T)
            chi_T_eval, chi_T_evec = np.linalg.eigh(chi_T)
            chi_T_eval_all[I,K] = chi_T_eval

    print("\nEigenvalue of Susceptibility tensor * T (cm3K/mol)" )
    print("--------------------------------------------------------")
    print("TEMP(K)   B(T)         X1*T         X2*T         X3*T")
    print("--------------------------------------------------------")
    
    for  I in range(len(T_list)):
        T = T_list[I]
        for K in range(len(Bs_list)):
            Bs = Bs_list[K]
            print("%6.2f  %8.2f %14.6f %12.6f %12.6f" % (T, Bs, chi_T_eval_all[I,K,0],chi_T_eval_all[I,K,1],chi_T_eval_all[I,K,2]))

    return chi_T_eval_all

    
def magnetization(interface,B_s,B_vec,en_soc,Mu,T,h_s,dB_k=None):
    
    kb = interface.kb 
    mu_B = interface.mu_B_Eh

    n_micro_states = len(en_soc)
    B_vec = B_vec / np.linalg.norm(B_vec)
    if dB_k is None:
        dB_k = B_vec
    else:
        dB_k = dB_k / np.linalg.norm(dB_k)
    
    #print("B_vec=",B_vec)
    #print("dB_k=",dB_k)

    #Set zero pint energy
    zero = en_soc[0]
    for i in range(n_micro_states):
        en_soc[i] = (en_soc[i] - zero)


    B_svec = B_s * B_vec
    en_ze, evec_ze =  E_ze(B_svec,en_soc,Mu, mu_B)


    #B1 = [i * (B_s + h_s) for i in B_vec]
    #B2 = [i * (B_s - h_s) for i in B_vec]

    #B1 =[]
    #B2 =[]
    #for i in B_vec:
    #    B1.append(i * (B_s + h_s))
    #    B2.append(i * (B_s - h_s))
    B1 = B_svec + dB_k * h_s
    B2 = B_svec - dB_k * h_s
    #print("B1=",B1)
    #print("B2=",B2)
    '''
    A1 = E_ze(B+h,en_soc,M)
    A2 = E_ze(B-h,en_soc,M)
    A3 = E_ze(B+2*h,en_soc,M)
    A4 = E_ze(B-2*h,en_soc,M)
    dE = np.zeros(n_micro_states)
    for i in range(n_micro_states):
        dE[i] =  4 * ( A1[i] - A2[i] ) / (2*h_s) 
        dE[i] -=  ( A3[i] - A4[i]) / (4*h_s) 
        dE[i] = dE[i] /3
        dE[i] = (A1[i]-A2[i]) / (2*h_s)
    print(dE)
    '''
    A1, evec_A1 = E_ze(B1,en_soc,Mu, mu_B)
    A2, evec_A2 = E_ze(B2,en_soc,Mu, mu_B)
    
    dE = np.zeros(n_micro_states)
    for i in range(n_micro_states):
        dE[i] = (A1[i]-A2[i]) / (2*h_s)
    #partial function
    Z = partial_function(en_ze,T, kb)
    
    MM=0
    for i in range(n_micro_states):
        MM -= dE[i] * np.exp(-(en_ze[i])/(T*kb))/Z / mu_B
    

    #E1, evec_1 = E_ze(B1,en_soc,J,mu_B)
    #E2, evec_2 = E_ze(B2,en_soc,J,mu_B)
    #
    #Z1 = partial_function(E1,T,kb)
    #Z2 = partial_function(E2,T,kb)
    #
    #lnZ1 = np.log(Z1)
    #lnZ2 = np.log(Z2)
    #
    #MM = kb * T * (lnZ1 - lnZ2)/(2*h_s) / mu_B

    return MM #in Bohr magneton unit
    
   
def susceptibility(interface,B_s,B_vec1,en_soc,Mu,T,h_s,dB_k=None,dB_l=None):

    kb = interface.kb 
    mu_B_Eh = interface.mu_B_Eh
    mu_B_erg = interface.mu_B_erg
    NA = interface.NA
    T_to_G = interface.T_to_G

    B_vec1 = B_vec1 / np.linalg.norm(B_vec1)
    dB_k = dB_k / np.linalg.norm(dB_k)

    n_micro_states = len(en_soc)
    B_svec1 = B_s * B_vec1
    en_ze, evec_ze =  E_ze(B_svec1,en_soc,Mu, mu_B_Eh)

    #h=np.array([h_s,0,0])

    #B1 = [i * (B_s + h_s) for i in B_vec]
    #B2 = [i * (B_s - h_s) for i in B_vec]


    #Set zero pint energy
    zero = en_soc[0]
    for i in range(n_micro_states):
        en_soc[i] = (en_soc[i] - zero)

    #same direction
    if dB_l is None:
        B1 =[]
        B2 =[]
        B3 =[]
        #for i in dB_k:
        #    B1.append(i * (B_s + h_s))
        #    B2.append(i * B_s)
        #    B3.append(i * (B_s - h_s))
        B1 = B_svec1 + dB_k * h_s
        B2 = B_svec1 
        B3 = B_svec1 - dB_k * h_s
        
        #for i in range(3):
        #    B1.append(B_svec1[i] + dB_k[i]*h_s )
        #    B2.append(B_svec1[i])
        #    B3.append(B_svec1[i]  - dB_k[i]*h_s)
    
        #print("B1=",B1)
        #print("B2=",B2)
        #print("B3=",B3)

    
        #print(en_soc)
        E_ze1, evec_ze1 = E_ze(B1,en_soc,Mu, mu_B_Eh)  #* 219474.63136314
        E_ze2, evec_ze2 = E_ze(B2,en_soc,Mu, mu_B_Eh)  #* 219474.63136314
        E_ze3, evec_ze3 = E_ze(B3,en_soc,Mu, mu_B_Eh)  #* 219474.63136314

        Z1 = partial_function(E_ze1,T, kb)
        Z2 = partial_function(E_ze2,T, kb)
        Z3 = partial_function(E_ze3,T, kb)

        lnZ1 = np.log(Z1)
        lnZ2 = np.log(Z2)
        lnZ3 = np.log(Z3)

        d2lnZ = (lnZ1-2*lnZ2+lnZ3)/(h_s**2)
    
        chi = kb * T * d2lnZ  / mu_B_Eh * mu_B_erg * NA / T_to_G

        #chi_T = chi * T
        #print("H=",B_s)
        #print("chi_T=",chi_T)

        #dlnZdH = (lnZ1 - lnZ3)/(2*h_s)
        #test = kb * T * dlnZdH * 27.2113862459817/5.7883817982e-5
        #print(test)
        #dlnZdH = (lnZ1 - lnZ3)/(2*h_s)
        #k_b =  scipy.constants.k * 2.294e+17 # 0.69503877
        #B_2 = 4.66864374e-5*10000
        #test = k_b * T * dlnZdH * 27.2113862459817/5.7883817982e-5
        ##test = k_b * T * dlnZdH/B_2 # * 27.2113862459817/5.7883817982e-5
        #print("test=",test)
    else:
        B1 =[]
        B2 =[]
        B3 =[]
        B4 =[]
        #for i in dB_k:
        #    B1.append(i * (B_s + h_s))
        #    B2.append(i * B_s)
        #    B3.append(i * (B_s - h_s))
        
        B1 = B_svec1 + dB_k*h_s + dB_l*h_s
        B2 = B_svec1 + dB_k*h_s - dB_l*h_s
        B3 = B_svec1 - dB_k*h_s + dB_l*h_s
        B4 = B_svec1 - dB_k*h_s - dB_l*h_s

        E_ze1, evec_ze1 = E_ze(B1,en_soc,Mu, mu_B_Eh)  #* 219474.63136314
        E_ze2, evec_ze2 = E_ze(B2,en_soc,Mu, mu_B_Eh)  #* 219474.63136314
        E_ze3, evec_ze3 = E_ze(B3,en_soc,Mu, mu_B_Eh)  #* 219474.63136314
        E_ze4, evec_ze4 = E_ze(B4,en_soc,Mu, mu_B_Eh)  #* 219474.63136314

        Z1 = partial_function(E_ze1,T,kb)
        Z2 = partial_function(E_ze2,T,kb)
        Z3 = partial_function(E_ze3,T,kb)
        Z4 = partial_function(E_ze4,T,kb)
        

        lnZ1 = np.log(Z1)
        lnZ2 = np.log(Z2)
        lnZ3 = np.log(Z3)
        lnZ4 = np.log(Z4)

        d2lnZ = (lnZ1 - lnZ2 - lnZ3 + lnZ4)/(4*h_s**2)
    
        chi = kb * T * d2lnZ   / mu_B_Eh * mu_B_erg * NA / T_to_G 






    return chi #cm3/mole


def E_ze(B,en_soc,Mu,mu_B):
    B = np.array(B) 
    #B = B /10000 #Gauss to Telsa
    B = B *mu_B #5.7883817982e-5/27.2113862459817#*2.14170097E-6# 2.14170097E-6=5.788e-5 (eV/T) * 0.0367493 (Eh/eV)               #/ (2.35051742e+5 ) Telsa to atomic unit from book
    H0 = np.diag(en_soc)
    Hze = np.einsum('k,kij->ij',B,Mu) 
    H_total = H0+Hze
    en_ze,evec_ze = np.linalg.eigh(H_total)

    return en_ze,evec_ze #Hartree

def partial_function(en_ze,T,kb):
    
    Z=0
    ##print("en_ze=",en_ze)
    #print("final=")
    #print((en_ze[1]-en_ze[0] )/(T*kb)*4.3597482E-18)
    for i in range(len(en_ze)):
        Z += np.exp(-(en_ze[i])/(T*kb))

    return Z

