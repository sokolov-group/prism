import sys
import os
import numpy as np
from prism.libsoc import general_somf
from prism.libsoc import magnetic
from prism.libsoc import before_soc
from prism.libsoc import after_soc
from prism.tools import trans_prop
import h5py


def kernel(interface):
    import prism.lib.logger as logger
    interface.log = logger.Logger(interface.stdout, interface.verbose)
    interface.log.info("Collecting necessary information for SOC")

    if interface.add_ref_ee is True:
        interface.log.info("Employing make_rdm1s_ee")
        make_rdm1s_ee(interface)
    else:
        interface.log.info("Employing make_rdm1s_general")
        make_rdm1s_general(interface)
    sys.stdout.flush()

    before_soc.rotate_rdm1s_to_ms(interface)
    sys.stdout.flush()

    E_ref, rdm_aabb_ref, S_ref,  ms_ref = before_soc.get_sf_ref(interface, interface.spin_adc)
    sys.stdout.flush()

    if interface.save_sf:
       before.collect_sf(E_ref, rdm_aabb_ref, S_ref, ms_ref, interface.mo, name=interface.name_sf)

    #Compute en_soc
    en_soc, evec_soc = general_somf.state_interaction_soc(interface, E_ref, rdm_aabb_ref, S_ref,  ms_ref, interface.soc)
    
    #Compute properties
    after_soc.compute_properties(interface, en_soc, evec_soc, rdm_aabb_ref, S_ref)
        
    # Print results obtained from ADC
    after_soc.print_results(interface, en_soc, interface.soc, "SI-ADC")
    
    if "osc_strengths_full" in interface.properties_soc:
        trans_prop.print_osc_strength(interface, interface.properties_soc["osc_strengths_full"])
    if "magnetic_properties_soc" in interface.properties_soc:
        magnetic.print_mag_properties_soc(interface, interface.properties_soc,  method = None)

    sys.stdout.flush()

    interface.e_tot = en_soc
        

    return interface.e_tot, interface.properties_soc


def make_rdm1s_ee(interface):
    '''
    Obtain rdm_1s for ADC-EE
    '''

    #TO DO: make sure nmo_b is always equal nmo_a
    interface.log.info("\n*****make_rdm1s_ee*****")
    interface.log.info("Condsidering Ref state in rdm...")

    #Parameter
    n_state = len(interface.E_sf)
    n_state += 1
    nmo_a = interface.nmo_a
    nmo_b = interface.nmo_b
    mo_a = interface.mo_a
    mo_b = interface.mo_b
    nocc_a = interface.nocc_a
    nocc_b = interface.nocc_b
    nvir_a = interface.nvir_a
    nvir_b = interface.nvir_b
    ovlp = interface.ovlp
    U = interface.U
    P = interface.P
    X = interface.X
    E_ref_abs = interface.E_ref_abs

    #Check ref Energy
    if np.abs(interface.E_sf[0]) > 0.01:
        interface.log.info("Take MPn E_ref as ground state energy ")
        interface.E_sf = np.insert(interface.E_sf, 0, 0)
    else:
        interface.log.info("Asume E_ref = lowest ADC energy ")
        E_g = interface.E_sf[0]
        interface.E_sf = np.insert(interface.E_sf, 0, E_g)
    
    #add UHF and MPn energy
    interface.E_sf += E_ref_abs
    interface.log.info("E_sf=%s",interface.E_sf)

    #get S quantum number:
    from pyscf.adc import uadc_ee
    spin_square, evec_ne = uadc_ee.get_spin_square(interface.adc)
    S_old_total = (-1 + ( 1 + 4*spin_square)**(1/2) ) /2 
    
    #Include reference state
    # TO DO: Find real ref Spin from mf object 
    ss_ref, s_ref =  interface.mf.spin_square() #ss_ref is S(S+1), s_ref = 2S+1
    S_old_ref = (-1 + ( 1 + 4*ss_ref)**(1/2) ) /2
    S_old_total = np.insert(S_old_total, 0, S_old_ref)
    interface.log.info("S_old_total=")
    interface.log.info("%s", S_old_total)
    interface.S_total = S_old_total

    # Calculate Full RDM
    rdm_aabb = np.zeros((2, n_state, n_state, nmo_b, nmo_b))
    rdm_aa = np.zeros((n_state,n_state,nmo_a,nmo_a))
    rdm_bb = np.zeros((n_state,n_state,nmo_b,nmo_b))
    rdm1_ref = interface.adc.make_ref_rdm1()
    rdm_aa[0,0] = rdm1_ref[0]
    rdm_bb[0,0] = rdm1_ref[1]
    for i in range(n_state-1):   
        #rdm1_a, rdm1_b = interface.make_ref_rdm1s_ee(U[:,i])
        #accroding to NTO in prism, X should be <0| |i>
        rdm1_a = X[0][i]
        rdm1_b = X[1][i]
        rdm_aa[0,i+1] = rdm1_a
        rdm_bb[0,i+1] = rdm1_b
        rdm_aa[i+1,0] = rdm1_a.T
        rdm_bb[i+1,0] = rdm1_b.T
    for i in range(n_state-1):
        for j in range(n_state-1):
            rdm1_a, rdm1_b = interface.make_rdm1_eigenvectors(interface.adc,U[:,i], U[:,j])
            rdm_aa[i+1,j+1] = rdm1_a    
            rdm_bb[i+1,j+1] = rdm1_b

    #Transform rdm_bb in mo_a basis
    interface.log.info("Transforming rdm_bb in mo_a basis")
    rdm_bb_ao = np.einsum('ai,IJib,bj->IJaj',mo_b,rdm_bb,mo_b.T)
    CTS = mo_a.T @ ovlp
    SC =  ovlp @ mo_a
    rdm_bb_mo_a = np.einsum('ai,IJib,bj->IJaj',CTS, rdm_bb_ao ,SC)
    rdm_aabb[0] += rdm_aa
    rdm_aabb[1] += rdm_bb_mo_a
    interface.mo_b = mo_a
    interface.mo = mo_a
    interface.rdm_aabb = rdm_aabb

    return interface.rdm_aabb, interface.S_total, interface.E_sf

    
def make_rdm1s_general(interface):
    '''
    Obtain rdm_1s for ADC-IP, EA, CVS-ADC-IP and ADC-EE(witout ref state)
    '''
    interface.log.info("\n*****make_rdm1s_general*****")
    #TO DO: make sure nmo_b is always equal nmo_a

    #Parameter
    n_state = len(interface.E_sf)
    nmo_a = interface.nmo_a
    nmo_b = interface.nmo_b
    mo_a = interface.mo_a
    mo_b = interface.mo_b
    ovlp = interface.ovlp
    U = interface.U
    P = interface.P
    spin_adc = interface.spin_adc

    # Calculate Full RDM
    rdm_aabb = np.zeros((2, n_state, n_state, nmo_b, nmo_b))
    rdm_aa = np.zeros((n_state,n_state,nmo_a,nmo_a))
    rdm_bb = np.zeros((n_state,n_state,nmo_b,nmo_b))
    for i in range(n_state):
        for j in range(n_state):
            if interface.ncvs:
                rdm1_a, rdm1_b = interface.make_rdm1_eigenvectors(interface.adc,U[:,i], U[:,j])
            else:
                if interface.method_type == "ea" :
                    rdm1_a, rdm1_b = interface.make_rdm1_eigenvectors(interface.adc,U[:,i], U[:,j])
                elif interface.method_type == "ee" :
                    rdm1_a, rdm1_b = interface.make_rdm1_eigenvectors(interface.adc,U[:,i], U[:,j])
                elif interface.method_type == 'ip':
                    #flip make_rdm
                    rdm1_a, rdm1_b = interface.make_rdm1_eigenvectors(interface.adc,U[:,j], U[:,i])
            
            rdm_aa[i,j] = rdm1_a    
            rdm_bb[i,j] = rdm1_b

    #Transform rdm_bb in mo_a basis
    interface.log.info("Transforming rdm_bb in mo_a basis")
    rdm_bb_ao = np.einsum('ai,IJib,bj->IJaj',mo_b,rdm_bb,mo_b.T)
    CTS = mo_a.T @ ovlp
    SC =  ovlp @ mo_a
    rdm_bb_mo_a = np.einsum('ai,IJib,bj->IJaj',CTS, rdm_bb_ao ,SC)
    rdm_aabb[0] += rdm_aa
    rdm_aabb[1] += rdm_bb_mo_a
    interface.mo_b = mo_a
    interface.mo = mo_a
    interface.rdm_aabb = rdm_aabb

    #get S quantum number:
    S_old_total = np.zeros(n_state)
    if interface.method_type == "ee":
        from pyscf.adc import uadc_ee
        spin_square, evec_ne = uadc_ee.get_spin_square(interface.adc)
        S_old_total = (-1 + ( 1 + 4*spin_square)**(1/2) ) /2 

    else:
        S_old = round(spin_adc/2 , 2) 
        # If spec factor too low, spin should be different
        if interface.use_P is True:
            interface.log.info("Use P to find S quantum number")
            interface.log.info("P_thresh=%s", interface.P_thresh) 
            for i in range(n_state):
                if P[i] > interface.P_thresh:
                    S_old_total[i] = S_old
                else:
                    S_old_total[i] = S_old + 1
        else:
            interface.log.info("Use trdm to find S quantum number")
            interface.log.info("P_thresh=%s", interface.P_thresh)
            if P[0] > interface.P_thresh:
                S_old_total[0] = S_old
            else:
                S_old_total[0] = S_old + 1
            
            for i in range(n_state-1):
                trdm = rdm_aabb[0,0,i+1] + rdm_aabb[1,0,i+1]
                if np.linalg.norm(trdm) < 1:
                    S_old_total[i+1] = S_old_total[0]+1
                else:
                    S_old_total[i+1] = S_old_total[0]

    interface.log.info("S_old_total=")
    interface.log.info("%s", S_old_total)
    interface.S_total = S_old_total

    return interface.rdm_aabb, interface.S_total, interface.E_sf
        












        