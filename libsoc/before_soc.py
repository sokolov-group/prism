import sys
import os
import numpy as np
import h5py

def create_soc_flag(method):
    '''
    Create SOC flag. NEVPT object also uses it.
    '''
        
    # SOC params:
    method.soc = None # Spin–orbit coupling. Possible methods: Breit-Pauli (BP), DKH1 (x2c-1)
    method.analyze_evec_soc = False #For soc eigenvector analysis
    method.unc = None # For DKH2
    method.max_memory_soc = method.max_memory #Max_memory for compute soc mo Hamiltionian
    method.save_sf = False #saving spin-free information for SOC calculation
    method.name_sf = None  #file name for saving spin-free information for SOC calculation

    # Set up X2C function in interface
    method.interface.x2c_setup()

    # For magnetic properties
    method.magnetic_origin_type = 'charge'        # Origin of coordinate system for g-tensor calculations. Possible values: charge, GIAO, atom1 or user-defined point (list)
        
    #g-tensor
    method.gtensor = False                       # Enable calculating g-tensors (requires soc)
    method.gtensor_target_state = 1              # Target state for g-tensor calculation. Default is the ground state (target_state = 1).


    # For magnetic susceptibility
    method.mag_av = False # Enable calculating powder magnetization (requires soc)
    method.sus_av = False # Enable calculating powder magnetic susceptibility (requires soc)
    method.mag_vec = False # Enable calculating vector magnetization (requires soc)
    method.sus_tensor = False # Enable calculating tensor magnetic susceptibility (requires soc)
    method.step_h_s = 0.001 #Magnetic field step size used for numerical differentiation (Unit:T)
    ###Powder magnetization
    Bs_list = []
    for i in range(15):
        H = i * 0.5
        Bs_list.append(H)
    method.Bs_powder_M = Bs_list  # Magnetic field for calculating powder magnetization (Unit:T)
    method.T_powder_M = [1.8]     # Temperature for calculating powder magnetization (Unit:K)
    ###Powder susceptibility
    T_list = []
    for i in range(21):
        T = 14.75 * i + 5
        T_list.append(T)
    method.T_powder_chi = T_list # Temperature for calculating powder magnetic susceptibility (Unit:K)
    method.Bs_powder_chi = [0.1] # Magnetic field for calculating powder magnetic susceptibility (Unit:T)
    ###Vector magnetization
    method.B_vec_M = [0,0,1]     # Magnetic field direction for calculating vector magnetization
    method.Bs_vec_M = Bs_list    # Magnetic field strength for calculating vector magnetization (Unit:T)
    method.T_vec_M = [1.8]       # Temperature for calculating vector magnetization (Unit:K)
    ###Tensor  susceptibility
    method.B_vec_chi = [0,0,1]         # Magnetic field direction for calculating tensor magnetic susceptibility
    method.Bs_vec_chi = [0.1]          # Magnetic field strength for calculating tensor magnetic susceptibility (Unit:T)
    method.T_vec_chi = [5,100,200,250] # Temperature for calculating tensor magnetic susceptibility (Unit:K)

    method.properties_soc = {} #SI soc result


def rotate_rdm1s_to_ms(method):
    '''
    Transform method.rdm_aabb, method.S_total and method.E_ref
    in Ms basis
    ********
    INPUT:
    method.rdm_aabb_ref (np.array, 2*nstate*nstate*nmo*nmo): spin alpha-alpha and beta-beta 1st rdm, rdm_aabb[0] is alpha-alpha and rdm_aabb[1] is beta-beta 
    method.S_ref (list), spin quantum number of each state without spin-orbit coupling
    method.E_ref (np.array, n): reference energy

    OUTPUT:
    Update input parameter (method.rdm_aabb_ref, method.S_ref method.E_ref) into ms basis
    '''

    method.log.info("\n*****rotate_rdm1s_to_ms*****")

    #warning: make sure nmo_b is always equal nmo_a

    #parameter
    rdm_aabb_old = method.rdm_aabb_ref.copy()
    S_ref_old    = method.S_ref.copy()
    E_ref_old     = method.E_ref.copy()
    n_state      = len(E_ref_old)

    #Calculating ms by using rdm_aabb
    Sz_en_tot = np.zeros(n_state)
    Sz_evec_total = np.zeros((n_state,n_state))

    deg_list = degeneracy_list(method.interface, E_ref_old ,  atol= method.deg_atol)
    method.log.info("deg_list=%s",deg_list)

    Ms_ref_old = np.zeros((n_state,n_state))
    for I in range(n_state):
        for J in range(n_state):
            rdm_sz = 1/2 * ( rdm_aabb_old[0,I,J] - rdm_aabb_old[1,I,J] )
            sz = np.trace(rdm_sz)
            Ms_ref_old[I,J] = sz

    method.log.extra("Ms_old_total=")
    method.log.extra("%s", Ms_ref_old)

    #Compute rotation matrix to get Ms states
    for K in range(len(deg_list)):
        I = int(np.sum(deg_list[0:K]))
        J = I + deg_list[K]
        Ms = Ms_ref_old[I:J, I:J]
        sz_en, sz_evec = np.linalg.eigh(Ms)
        Sz_evec_total[I:J,I:J] = sz_evec
        Sz_en_tot[I:J] = sz_en

    method.log.extra("sz_en_tot=")
    method.log.extra("%s", Sz_en_tot)
    method.log.extra("Sz_evec_total=")
    method.log.extra("%s", Sz_evec_total)

    #Rotate rdm_aabb_old in Ms basis
    rdm_aabb_new = np.einsum('ai,KibIJ,bj->KajIJ',np.conj(Sz_evec_total).T , rdm_aabb_old , Sz_evec_total)

    #Rotate S_ref_old in Ms basis
    S_new = np.einsum('ai,ib,bj->aj',np.conj(Sz_evec_total).T , np.diag(S_ref_old) , Sz_evec_total)
    method.log.extra("Make sure S_ref:")
    method.log.extra("%s", S_new)
    if not np.allclose(S_new, np.diag(np.diagonal(S_new)), atol=1e-7, rtol=0):
        method.log.warn("Warning: S_ref is not diagonal")
    else:
        method.log.info("S_ref is diagonal")

    #Rotate E_new in Ms basis
    E_new = np.einsum('ai,ib,bj->aj',np.conj(Sz_evec_total).T , np.diag(E_ref_old) , Sz_evec_total)
    method.log.extra("Make sure E:")
    method.log.extra("%s", E_new)
    E_new = np.diagonal(E_new)
    if np.allclose(E_ref_old, E_new, atol=method.deg_atol, rtol=0):
        method.log.info("Energy is same after rotation")
    else:
        raise Exception("E_new change energy")

    #Update NECESSARY parameter of state-interaction calculations
    method.rdm_aabb_ref = rdm_aabb_new
    method.S_ref = np.diagonal(S_new) #Save S_total
    method.E_ref = E_new #Save E_new


def get_sf_ref(method, spin):
    '''
    Select specifc Ms state for state-interaction calculation
    ********
    INPUT:
    spin (int): number of unpair electrons
    interface.rdm_aabb_ref in Ms Basis (np.array, 2*nstate*nstate*nmo*nmo): spin alpha-alpha and beta-beta 1st rdm,  rdm_aabb[0] is alpha-alpha and rdm_aabb[1] is beta-beta 
    interface.S_ref in Ms Basis (list), spin quantum number of each state without spin-orbit coupling
    interface.E_ref in Ms Basis (np.array, n): reference energy

    OUTPUT (Can directly plug in prism.libsoc.general_somf.state_interaction_soc. They are also saved(updated) in method.):
    rdm_aabb_ref(np.array, 2*nstate*nstate*nmo*nmo): spin alpha-alpha and beta-beta 1st rdm,  rdm_aabb[0] is alpha-alpha and rdm_aabb[1] is beta-beta 
    E_ref (np.array, n): reference energy
    S_ref (list), spin quantum number of each state without spin-orbit coupling
    ms_ref (float, should be multiply of 0.5), spin projection quantum number of reference state without spin-orbit coupling
    '''

    method.log.info("\n*****get_sf_ref*****")

    #Parameter
    rdm_aabb_old = method.rdm_aabb_ref.copy()
    S_ref_old    = method.S_ref.copy()
    E_ref_old     = method.E_ref.copy()
    n_state      = len(E_ref_old)


    #Recalculate Ms to make sure Ms
    Ms=np.zeros((n_state,n_state))
    for I in range(n_state):
        for J in range(n_state):
            rdm_sz = 1/2 * ( rdm_aabb_old[0,I,J] - rdm_aabb_old[1,I,J] )
            sz = np.trace(rdm_sz)
            Ms[I,J] = sz
    method.log.extra("Make sure MS=")
    method.log.extra("%s", Ms)
    if not np.allclose(Ms, np.diag(np.diagonal(Ms)), atol=1e-5, rtol=0):
        method.log.warn("Warning: Ms is not diagonal")
    else:
        method.log.info("Ms is diagonal")

    #Obtain reasonavle S and Ms
    sz = np.diagonal(Ms)
    sz = np.round(sz* 2) / 2
    sz = sz.tolist()
    sz = [round(elem,2) for elem in sz ]
    S_ref_old = np.round(S_ref_old* 2) / 2
    S_ref_old = S_ref_old.tolist()
    S_ref_old = [round(elem,2) for elem in S_ref_old ]

    #Choice ms as positive minimum
    if method.ms_ref_select is None:
        method.log.info("Choicing ms_ref as positive minimum")
        ms_ref = min((x for x in sz if x > 0), default=None) #Choice ms as positive minimum
    
        if ms_ref is None:
            method.log.info("detect ms_ref is None, it should be 0 ")
            ms_ref = 0
    else:
        method.log.info("User define ms_ref")
        ms_ref = method.ms_ref_select
    method.log.info("ms_ref=%s", ms_ref)

    #Find index of ms=ms_ref microstates
    ref_index = []
    for i in range(n_state):
        if (np.abs(sz[i]-ms_ref)) < 1e-5:
            ref_index.append(i)

    #Discard unreasonable spin state 
    ref_index_S = []
    #even electron, S sohuld be 0,1,2,3....
    if spin % 2 == 0:
        method.log.info("even electrons...")
        for i in range(len(ref_index)):
            I = ref_index[i]
            S_check = S_ref_old[I]
            if abs(S_check % 1 - 0.5) < 1e-8:
                method.log.info("Discarding S=%.1f state" %S_check)
            elif abs(S_check % 1) < 1e-8:    
                ref_index_S.append(I)
            else:
                method.log.info("Discarding no .5 .0  S=%.1f state" %S_check)
        
    #odd electron, S sohuld be 0.5, 1.5, 2.5, 3.5....
    else:
        method.log.info("odds electrons...")
        for i in range(len(ref_index)):
            I = ref_index[i]
            S_check = S_ref_old[I]
            if abs(S_check % 1 - 0.5) < 1e-8:
                ref_index_S.append(I)
            elif abs(S_check % 1) < 1e-8: 
                method.log.info("Discarding S=%.1f state" %S_check)
            else:
                method.log.info("Discarding no .5 .0 S=%.1f state" %S_check)

    #Pick ms=ms_ref microstates
    n_state_ref = len(ref_index_S)
    rdm_aabb_ref = np.zeros((2,n_state_ref, n_state_ref, len(rdm_aabb_old[0,0,0]),len(rdm_aabb_old[0,0,0])))
    E_ref = []
    S_ref = []
    E_ref_old = E_ref_old.tolist()
    for i in range(n_state_ref):
        I = ref_index_S[i]
        E_ref.append(E_ref_old[I])
        S_ref.append(S_ref_old[I])
        for j in range(n_state_ref):
            J = ref_index_S[j]
            rdm_aabb_ref[:,i,j] = rdm_aabb_old[:,I,J]

    #Save NECESSARY parameter of state-interaction calculations
    method.E_ref = E_ref
    method.rdm_aabb_ref = rdm_aabb_ref
    method.S_ref = S_ref
    method.ms_ref = ms_ref

    #Print final result
    method.log.info("roots_old=%s",n_state)
    method.log.info("roots_ref=%s",n_state_ref)
    method.log.info("S_ref=%s",S_ref)
    method.log.info("ms_ref=%s", ms_ref)
    method.log.info("E_ref=%s",E_ref)
    
    return method.E_ref, method.rdm_aabb_ref, method.S_ref, method.ms_ref


def degeneracy_list(interface, en, atol= 1e-5):
    '''
    determine degeneracy based on energy in list
    ********
    INPUT:
    en (list, n_states): energy of each state
    atol (float): Absolute tolerance for determining whether two states have the same energy. 

    OUTPUT:
    deg (list): list for each state's degeneracy
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


def write_checkpoint(interface, E_ref, rdm_aabb_ref, S_ref, ms_ref, mo, name=None):
    '''
    Save spin-free information for state-interaction calculation(state_interaction_soc)
    ********
    INPUT (Can directly plug in prism.libsoc.general_somf.state_interaction_soc):
    E_ref (np.array, n): reference energy
    rdm_aabb_ref(np.array, 2*nstate*nstate*nmo*nmo): spin alpha-alpha and beta-beta 1st rdm,  rdm_aabb[0] is alpha-alpha and rdm_aabb[1] is beta-beta 
    S_ref (list): spin quantum number of each state without spin-orbit coupling
    ms_ref (float, should be multiply of 0.5): spin projection quantum number of reference state without spin-orbit coupling
    mo (np.array): molecular orbitals
    name (str): name of saving h5 file. Default is inputfile name
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


def write_checkpoint_ms0(interface, E_ref, rdm_aabb_ref, rdm_aabb_plus_ref, S_ref, mo, name=None):
    '''
    Save spin-free information for Ms=0 state-interaction calculation
    ********
    INPUT (Can directly plug in prism.libsoc.general_somf.state_interaction_soc_ms0):
    E_ref (np.array, n): reference energy
    rdm_aabb_ref(np.array, 2*nstate*nstate*nmo*nmo):  ms=0 spin alpha-alpha and beta-beta 1st rdm for diferent spin coupling,  rdm_aabb[0] is alpha-alpha and rdm_aabb[1] is beta-beta 
    rdm_aabb_plus_ref(np.array, 2*nstate*nstate*nmo*nmo): ms=1 spin alpha-alpha and beta-beta 1st rdm for same spin coupling
    S_ref (list): spin quantum number of each state without spin-orbit coupling
    mo (np.array): molecular orbitals
    name (str): name of saving h5 file. Default is inputfile name
    '''
    interface.log.info("\nSave E_ref, rdm_aabb_ref, rdm_aabb_plus_ref, S_ref ... as h5 file...")
    if name:
        input_file = name
    else:    
        input_file = os.path.splitext(os.path.basename(sys.argv[0]))[0]
    
    interface.log.info("file name:"+input_file+ "_ms0.h5\n")
    with h5py.File(input_file + "_ms0.h5", "w") as f:
        f.create_dataset("E_ref", data=E_ref)
        f.create_dataset("rdm_aabb_ref", data=rdm_aabb_ref)
        f.create_dataset("rdm_aabb_plus_ref", data=rdm_aabb_plus_ref)
        f.create_dataset("S_ref", data=S_ref)
        f.create_dataset("mo", data=mo)


def read_checkpoint(sf_name):
    '''
    Read spin-free information for Ms=0 state-interaction calculation
    ********
    INPUT:
    sf_name (str): name of reading file

    OUTPUT (Can directly plug in prism.libsoc.general_somf.state_interaction_soc):
    E_ref (np.array, n): reference energy
    rdm_aabb_ref(np.array, 2*nstate*nstate*nmo*nmo): spin alpha-alpha and beta-beta 1st rdm,  rdm_aabb[0] is alpha-alpha and rdm_aabb[1] is beta-beta 
    S_ref (list): spin quantum number of each state without spin-orbit coupling
    ms_ref (float, should be multiply of 0.5): spin projection quantum number of reference state without spin-orbit coupling
    mo (np.array): molecular orbitals
    name (str): name of saving h5 file. Default is inputfile name
    '''
    print("\nLoad E_ref, rdm_aabb_ref, S_ref, ms_ref...")
    print("file name:"+sf_name+ "\n")
    with h5py.File(sf_name, "r") as f:
        E_ref        = f["E_ref"][:]
        rdm_aabb_ref = f["rdm_aabb_ref"][:]
        S_ref        = f["S_ref"][:]
        ms_ref       = f["ms_ref"][()]
        mo           = f["mo"][:]

    return E_ref, rdm_aabb_ref, S_ref, ms_ref, mo


def read_checkpoint_ms0(sf_name):
    '''
    Read spin-free information for Ms=0 state-interaction calculation
    ********
    INPUT:
    sf_name (str): name of reading file

    OUTPUT (Can directly plug in prism.libsoc.general_somf.state_interaction_soc_ms0):
    E_ref (np.array, n): reference energy
    rdm_aabb_ref(np.array, 2*nstate*nstate*nmo*nmo):  ms=0 spin alpha-alpha and beta-beta 1st rdm for diferent spin coupling,  rdm_aabb[0] is alpha-alpha and rdm_aabb[1] is beta-beta 
    rdm_aabb_plus_ref(np.array, 2*nstate*nstate*nmo*nmo): ms=1 spin alpha-alpha and beta-beta 1st rdm for same spin coupling
    S_ref (list): spin quantum number of each state without spin-orbit coupling
    mo (np.array): molecular orbitals
    '''
    print("\nE_ref, rdm_aabb_ref, rdm_aabb_plus_ref, S_ref...")
    print("file name:"+sf_name+ "\n")
    with h5py.File(sf_name, "r") as f:
        E_ref        = f["E_ref"][:]
        rdm_aabb_ref = f["rdm_aabb_ref"][:]
        rdm_aabb_plus_ref = f["rdm_aabb_plus_ref"][:]
        S_ref        = f["S_ref"][:]
        mo           = f["mo"][:]

    return E_ref, rdm_aabb_ref, rdm_aabb_plus_ref, S_ref, mo