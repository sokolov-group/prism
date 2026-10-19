import sys
import os
import numpy as np
import h5py


def rotate_rdm1s_to_ms(interface):
    '''
    Transform interface.rdm_aabb, interface.S_total and interface.E_sf
    in Ms basis
    ********
    INPUT:
    interface.rdm_aabb (np.array, 2*nstate*nstate*nmo*nmo): spin alpha-alpha and beta-beta 1st rdm,  rdm_aabb[0] is alpha-alpha and rdm_aabb[1] is beta-beta 
    interface.S_total (list), spin quantum number of each state without spin-orbit coupling
    interface.E_sf (np.array, n): reference energy
    '''

    interface.log.info("\n*****rotate_rdm1s_to_ms*****")

    #warning: make sure nmo_b is always equal nmo_a

    #parameter
    rdm_aabb = interface.rdm_aabb
    S_old_total = interface.S_total
    E_old_total = interface.E_sf
    n_state = len(E_old_total)

    #Calculating ms by using rdm_aabb
    Sz_en_tot = np.zeros(n_state)
    Sz_evec_total = np.zeros((n_state,n_state))

    deg_list = degeneracy_list(interface, E_old_total,  atol= interface.deg_atol)
    interface.log.info("deg_list=%s",deg_list)

    Ms_total = np.zeros((n_state,n_state))
    for I in range(n_state):
        for J in range(n_state):
            rdm_sz = 1/2 * ( rdm_aabb[0,I,J] - rdm_aabb[1,I,J] )
            sz = np.trace(rdm_sz)
            Ms_total[I,J] = sz

    interface.log.extra("Ms_old_total=")
    interface.log.extra("%s", Ms_total)

    #Compute rotation matrix to get Ms states
    for K in range(len(deg_list)):
        I = int(np.sum(deg_list[0:K]))
        J = I + deg_list[K]
        Ms = Ms_total[I:J, I:J]
        sz_en, sz_evec = np.linalg.eigh(Ms)
        Sz_evec_total[I:J,I:J] = sz_evec
        Sz_en_tot[I:J] = sz_en

    interface.log.extra("sz_en_tot=")
    interface.log.extra("%s", Sz_en_tot)
    interface.log.extra("Sz_evec_total=")
    interface.log.extra("%s", Sz_evec_total)

    #Rotate rdm_aabb in Ms basis
    interface.rdm_aabb = np.einsum('ai,KibIJ,bj->KajIJ',np.conj(Sz_evec_total).T , rdm_aabb , Sz_evec_total)

    #Rotate S_total in Ms basis
    S_total = np.einsum('ai,ib,bj->aj',np.conj(Sz_evec_total).T , np.diag(S_old_total) , Sz_evec_total)
    interface.log.extra("Make sure S_total:")
    interface.log.extra("%s", S_total)
    if not np.allclose(S_total, np.diag(np.diagonal(S_total)), atol=1e-7, rtol=0):
        interface.log.warn("Warning: S_total is not diagonal")
    else:
        interface.log.info("S_total is diagonal")
    interface.S_total = np.diagonal(S_total) #Save S_total

    #Rotate E_new in Ms basis
    E_new = np.einsum('ai,ib,bj->aj',np.conj(Sz_evec_total).T , np.diag(E_old_total) , Sz_evec_total)
    interface.log.extra("Make sure E:")
    interface.log.extra("%s", E_new)
    E_new = np.diagonal(E_new)
    if np.allclose(E_old_total, E_new, atol=interface.deg_atol, rtol=0):
        interface.log.info("Energy is same after rotation")
        interface.E_sf = E_new #Save E_new
    else:
        raise Exception("E_new change energy")



def get_sf_ref(interface, spin):
    '''
    select specifc Ms state for state-interaction calculation
    ********
    INPUT:
    spin (int): number of unpair electrons
    interface.rdm_aabb in Ms Basis (np.array, 2*nstate*nstate*nmo*nmo): spin alpha-alpha and beta-beta 1st rdm,  rdm_aabb[0] is alpha-alpha and rdm_aabb[1] is beta-beta 
    interface.S_total in Ms Basis (list), spin quantum number of each state without spin-orbit coupling
    interface.E_sf in Ms Basis (np.array, n): reference energy

    OUTPUT (Can directly plug in prism.libsoc.general_somf.state_interaction_soc):
    rdm_aabb_ref(np.array, 2*nstate*nstate*nmo*nmo): spin alpha-alpha and beta-beta 1st rdm,  rdm_aabb[0] is alpha-alpha and rdm_aabb[1] is beta-beta 
    E_ref (np.array, n): reference energy
    S_ref (list), spin quantum number of each state without spin-orbit coupling
    ms_ref (float, should be multiply of 0.5), spin projection quantum number of reference state without spin-orbit coupling
    '''

    interface.log.info("\n*****get_sf_ref*****")

    #Parameter
    rdm_aabb = interface.rdm_aabb
    E = interface.E_sf
    n_state = len(E)
    S_total = interface.S_total

    #Recalculate Ms to make sure MS
    Ms=np.zeros((n_state,n_state))
    for I in range(n_state):
        for J in range(n_state):
            rdm_sz = 1/2 * ( rdm_aabb[0,I,J] - rdm_aabb[1,I,J] )
            sz = np.trace(rdm_sz)
            Ms[I,J] = sz
    interface.log.extra("Make sure MS=")
    interface.log.extra("%s", Ms)
    if not np.allclose(Ms, np.diag(np.diagonal(Ms)), atol=1e-5, rtol=0):
        interface.log.warn("Warning: Ms is not diagonal")
    else:
        interface.log.info("Ms is diagonal")

    #Obtain reasonavle S and Ms
    sz = np.diagonal(Ms)
    sz = np.round(sz* 2) / 2
    sz = sz.tolist()
    sz = [round(elem,2) for elem in sz ]
    S_total = np.round(S_total* 2) / 2
    S_total = S_total.tolist()
    S_total = [round(elem,2) for elem in S_total ]

    #Choice ms as positive minimum
    if interface.ms_ref_select is None:
        interface.log.info("Choicing ms_ref as positive minimum")
        ms_ref = min((x for x in sz if x > 0), default=None) #Choice ms as positive minimum
    
        if ms_ref is None:
            interface.log.info("detect ms_ref is None, it should be 0 ")
            ms_ref = 0
    else:
        interface.log.info("User define ms_ref")
        ms_ref = interface.ms_ref_select
    interface.log.info("ms_ref=%s", ms_ref)

    #Find index of ms=ms_ref microstates
    ref_index = []
    for i in range(n_state):
        if (np.abs(sz[i]-ms_ref)) < 1e-5:
            ref_index.append(i)

    #Discard unreasonable spin state 
    ref_index_S = []
    #even electron, S sohuld be 0,1,2,3....
    if spin % 2 == 0:
        interface.log.info("even electrons...")
        for i in range(len(ref_index)):
            I = ref_index[i]
            S_check = S_total[I]
            if abs(S_check % 1 - 0.5) < 1e-8:
                interface.log.info("Discarding S=%.1f state" %S_check)
            elif abs(S_check % 1) < 1e-8:    
                ref_index_S.append(I)
            else:
                interface.log.info("Discarding no .5 .0  S=%.1f state" %S_check)
        
    #odd electron, S sohuld be 0.5, 1.5, 2.5, 3.5....
    else:
        interface.log.info("odds electrons...")
        for i in range(len(ref_index)):
            I = ref_index[i]
            S_check = S_total[I]
            if abs(S_check % 1 - 0.5) < 1e-8:
                ref_index_S.append(I)
            elif abs(S_check % 1) < 1e-8: 
                interface.log.info("Discarding S=%.1f state" %S_check)
            else:
                interface.log.info("Discarding no .5 .0 S=%.1f state" %S_check)

    #Pick ms=ms_ref microstates
    n_state_ref = len(ref_index_S)
    rdm_aabb_ref = np.zeros((2,n_state_ref, n_state_ref, len(rdm_aabb[0,0,0]),len(rdm_aabb[0,0,0])))
    E_ref = []
    S_ref = []
    E = E.tolist()
    for i in range(n_state_ref):
        I = ref_index_S[i]
        E_ref.append(E[I])
        S_ref.append(S_total[I])
        for j in range(n_state_ref):
            J = ref_index_S[j]
            rdm_aabb_ref[:,i,j] = rdm_aabb[:,I,J]

    #Print final result
    interface.log.info("roots_old=%s",n_state)
    interface.log.info("roots_ref=%s",n_state_ref)
    interface.log.info("S_ref=%s",S_ref)
    interface.log.info("ms_ref=%s", ms_ref)
    interface.log.info("E_ref=%s",E_ref)
    interface.log.info("set up xmol, contr_coeff")
    interface.log.info("max_memory_soc =%s", interface.max_memory_soc)
    interface.mol.max_memory = interface.max_memory_soc
    
    return E_ref, rdm_aabb_ref, S_ref, ms_ref


def degeneracy_list(interface,en, atol= 1e-5):
    '''
    determine degeneracy based on energy in list
    '''
    interface.log.info("degeneracy_atol=%s", atol)
    deg = []
    count2, total_deg = 0,0
    for count1 in range(len(en)):
        if (np.abs(en[count1] - en[count2]) < atol):
            total_deg +=1
        else:
            deg.append(total_deg)
            total_deg = 1
            count2 = count1
    if (total_deg >=1):
        deg.append(total_deg)
    return deg


def collect_sf(interface, E_ref, rdm_aabb_ref, S_ref, ms_ref, mo, name=None):
    '''
    Save spin-free information for state-interaction calculation
    '''
    interface.log.info("\nSave E_ref, rdm_aabb_ref, S_ref, ms_ref... as h5 file...")
    if name:
        input_file = name
    else:    
        input_file = os.path.splitext(os.path.basename(sys.argv[0]))[0]
    
    interface.log.info("file name:"+input_file+ ".h5\n")
    with h5py.File(input_file + ".h5", "w") as f:
        f.create_dataset("E_ref", data=E_ref)
        f.create_dataset("rdm_aabb_ref", data=rdm_aabb_ref)
        f.create_dataset("S_ref", data=S_ref)
        f.create_dataset("ms_ref", data=ms_ref)
        f.create_dataset("mo", data=mo)