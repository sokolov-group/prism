import sys
import os
import numpy as np
from prism.libsoc import general_somf
from prism.libsoc import magnetic
from prism.libsoc import before_soc
from prism.libsoc import after_soc
from prism.tools import trans_prop


def collect_sf(adc_prism):
    '''
    collect 
    adc_prism.mo
    adc_prism.rdm_aabb_ref
    adc_prism.S_ref
    Update
    adc_prism.mo_b 
    '''
    import prism.lib.logger as logger
    adc_prism.log = logger.Logger(adc_prism.stdout, adc_prism.verbose)
    adc_prism.log.info("Collecting necessary information for SOC")

    if adc_prism.add_ref_ee is True:
        adc_prism.log.info("Employing collect_sf_ee")
        collect_sf_ee(adc_prism)
    else:
        adc_prism.log.info("Employing collect_sf_general")
        collect_sf_general(adc_prism)
    sys.stdout.flush()

    '''
    before_soc.rotate_rdm1s_to_ms(adc_prism)
    sys.stdout.flush()

    E_ref, rdm_aabb_ref, S_ref,  ms_ref = before_soc.get_sf_ref(adc_prism, adc_prism.spin_adc)
    sys.stdout.flush()

    if adc_prism.save_sf:
       before_soc.write_checkpoint(adc_prism, E_ref, rdm_aabb_ref, S_ref, ms_ref, adc_prism.mo, name=adc_prism.name_sf)

    #Compute en_soc
    en_soc, evec_soc = general_somf.state_interaction_soc(adc_prism.interface, E_ref, rdm_aabb_ref, S_ref,  ms_ref, adc_prism.soc, adc_prism.mo)
    
    #Compute properties
    after_soc.compute_properties(adc_prism, en_soc, evec_soc, rdm_aabb_ref, S_ref)
        
    # Print results obtained from ADC
    after_soc.print_result(adc_prism, en_soc, evec_soc, S_ref, adc_prism.soc, "SI-ADC")
    
    #Print properties
    if "osc_strengths_full" in adc_prism.properties_soc:
        trans_prop.print_osc_strength(adc_prism, adc_prism.properties_soc["osc_strengths_full"])
    if "magnetic_properties" in adc_prism.properties_soc:
        magnetic.print_mag_properties(adc_prism, adc_prism.properties_soc,  method = None)

    sys.stdout.flush()

    adc_prism.e_tot = en_soc
    '''    

    #return adc_prism.e_tot, adc_prism.properties_soc["osc_strengths"]

def collect_sf_ee(adc_prism):
    '''
    Obtain rdm_aabb (in MO_a basis), S, E_ref for ADC-EE
    '''

    #TO DO: make sure nmo_b is always equal nmo_a
    adc_prism.log.info("\n*****collect_sf_ee*****")
    adc_prism.log.info("Condsidering Ref state in rdm...")

    #Parameter
    adc_prism.E_ref = adc_prism.E
    n_state = len(adc_prism.E_ref)
    n_state += 1
    nmo_a = adc_prism.nmo_a
    nmo_b = adc_prism.nmo_b
    mo_a = adc_prism.mo_a
    mo_b = adc_prism.mo_b
    nocc_a = adc_prism.nocc_a
    nocc_b = adc_prism.nocc_b
    nvir_a = adc_prism.nvir_a
    nvir_b = adc_prism.nvir_b
    ovlp = adc_prism.ovlp
    U = adc_prism.U
    P = adc_prism.P
    X = adc_prism.X
    E_ref_abs = adc_prism.E_ref_abs

    #Check ref Energy
    if np.abs(adc_prism.E_ref[0]) > 0.01:
        adc_prism.log.info("Take MPn E_ref as ground state energy ")
        adc_prism.E_ref = np.insert(adc_prism.E_ref, 0, 0)
    else:
        adc_prism.log.info("Asume E_ref = lowest ADC energy ")
        E_g = adc_prism.E_ref[0]
        adc_prism.E_ref = np.insert(adc_prism.E_ref, 0, E_g)
    
    #add UHF and MPn energy
    adc_prism.E_ref += E_ref_abs
    adc_prism.log.info("E_ref=%s",adc_prism.E_ref)

    #get S quantum number:
    from pyscf.adc import uadc_ee
    spin_square, evec_ne = uadc_ee.get_spin_square(adc_prism.adc)
    S_old_total = (-1 + ( 1 + 4*spin_square)**(1/2) ) /2 
    
    #Include reference state 
    ss_ref, s_ref =  adc_prism.interface.mf.spin_square() #ss_ref is S(S+1), s_ref = 2S+1
    S_old_ref = (-1 + ( 1 + 4*ss_ref)**(1/2) ) /2
    S_old_total = np.insert(S_old_total, 0, S_old_ref)
    adc_prism.log.info("S_old_total=")
    adc_prism.log.info("%s", S_old_total)

    # Calculate Full RDM
    rdm_aabb = np.zeros((2, n_state, n_state, nmo_b, nmo_b))
    rdm_aa = np.zeros((n_state,n_state,nmo_a,nmo_a))
    rdm_bb = np.zeros((n_state,n_state,nmo_b,nmo_b))
    rdm1_ref = adc_prism.adc.make_ref_rdm1()
    rdm_aa[0,0] = rdm1_ref[0]
    rdm_bb[0,0] = rdm1_ref[1]
    for i in range(n_state-1):   
        #accroding to NTO in prism, X should be <0| |i>
        rdm1_a = X[0][i]
        rdm1_b = X[1][i]
        rdm_aa[0,i+1] = rdm1_a
        rdm_bb[0,i+1] = rdm1_b
        rdm_aa[i+1,0] = rdm1_a.T
        rdm_bb[i+1,0] = rdm1_b.T
    for i in range(n_state-1):
        for j in range(n_state-1):
            rdm1_a, rdm1_b = adc_prism.make_rdm1_eigenvectors(adc_prism.adc,U[:,i], U[:,j])
            rdm_aa[i+1,j+1] = rdm1_a    
            rdm_bb[i+1,j+1] = rdm1_b

    #Transform rdm_bb in mo_a basis
    adc_prism.log.info("Transforming rdm_bb in mo_a basis")
    rdm_bb_ao = np.einsum('ai,IJib,bj->IJaj',mo_b,rdm_bb,mo_b.T)
    CTS = mo_a.T @ ovlp
    SC =  ovlp @ mo_a
    rdm_bb_mo_a = np.einsum('ai,IJib,bj->IJaj',CTS, rdm_bb_ao ,SC)
    rdm_aabb[0] += rdm_aa
    rdm_aabb[1] += rdm_bb_mo_a

    #Save NECESSARY parameter of state-interaction calculations
    adc_prism.mo_b = mo_a #Update mo_b
    adc_prism.mo = mo_a
    adc_prism.rdm_aabb_ref = rdm_aabb
    adc_prism.S_ref = S_old_total

    return adc_prism.rdm_aabb_ref, adc_prism.S_ref, adc_prism.E_ref
    
def collect_sf_general(adc_prism):
    '''
    Obtain rdm_aabb (in MO_a basis ), S, E_ref for ADC-IP, EA, CVS-ADC-IP and ADC-EE(witout ref state)
    '''
    adc_prism.log.info("\n*****collect_sf_general*****")
    #TO DO: make sure nmo_b is always equal nmo_a

    #Parameter
    adc_prism.E_ref = adc_prism.E #Initialize spin-free energy
    n_state = len(adc_prism.E_ref)
    nmo_a = adc_prism.nmo_a
    nmo_b = adc_prism.nmo_b
    mo_a = adc_prism.mo_a
    mo_b = adc_prism.mo_b
    ovlp = adc_prism.ovlp
    U = adc_prism.U
    P = adc_prism.P
    spin_adc = adc_prism.spin

    # Calculate Full RDM
    rdm_aabb = np.zeros((2, n_state, n_state, nmo_b, nmo_b))
    rdm_aa = np.zeros((n_state,n_state,nmo_a,nmo_a))
    rdm_bb = np.zeros((n_state,n_state,nmo_b,nmo_b))
    for i in range(n_state):
        for j in range(n_state):
            if adc_prism.ncvs:
                rdm1_a, rdm1_b = adc_prism.make_rdm1_eigenvectors(adc_prism.adc,U[:,i], U[:,j])
            else:
                if adc_prism.method_type == "ea" :
                    rdm1_a, rdm1_b = adc_prism.make_rdm1_eigenvectors(adc_prism.adc,U[:,i], U[:,j])
                elif adc_prism.method_type == "ee" :
                    rdm1_a, rdm1_b = adc_prism.make_rdm1_eigenvectors(adc_prism.adc,U[:,i], U[:,j])
                elif adc_prism.method_type == 'ip':
                    #flip make_rdm
                    rdm1_a, rdm1_b = adc_prism.make_rdm1_eigenvectors(adc_prism.adc,U[:,j], U[:,i])
            
            rdm_aa[i,j] = rdm1_a    
            rdm_bb[i,j] = rdm1_b

    #Transform rdm_bb in mo_a basis
    adc_prism.log.info("Transforming rdm_bb in mo_a basis")
    rdm_bb_ao = np.einsum('ai,IJib,bj->IJaj',mo_b,rdm_bb,mo_b.T)
    CTS = mo_a.T @ ovlp
    SC =  ovlp @ mo_a
    rdm_bb_mo_a = np.einsum('ai,IJib,bj->IJaj',CTS, rdm_bb_ao ,SC)
    rdm_aabb[0] += rdm_aa
    rdm_aabb[1] += rdm_bb_mo_a

    #get S quantum number:
    S_old_total = np.zeros(n_state)
    if adc_prism.method_type == "ee":
        spin_square, evec_ne = adc_prism.get_spin_square(adc_prism.adc)
        S_old_total = (-1 + ( 1 + 4*spin_square)**(1/2) ) /2 

    else:
        S_old = round(spin_adc/2 , 2) 
        # If spec factor too low, spin should be different
        if adc_prism.use_P is True:
            adc_prism.log.info("Use P to find S quantum number")
            adc_prism.log.info("P_thresh=%s", adc_prism.P_thresh) 
            for i in range(n_state):
                if P[i] > adc_prism.P_thresh:
                    S_old_total[i] = S_old
                else:
                    S_old_total[i] = S_old + 1
        else:
            adc_prism.log.info("Use trdm to find S quantum number")
            adc_prism.log.info("P_thresh=%s", adc_prism.P_thresh)
            if P[0] > adc_prism.P_thresh:
                S_old_total[0] = S_old
            else:
                S_old_total[0] = S_old + 1
            
            for i in range(n_state-1):
                trdm = rdm_aabb[0,0,i+1] + rdm_aabb[1,0,i+1]
                if np.linalg.norm(trdm) < 1:
                    S_old_total[i+1] = S_old_total[0]+1
                else:
                    S_old_total[i+1] = S_old_total[0]

    adc_prism.log.info("S_ref_old=")
    adc_prism.log.info("%s", S_old_total)
    
    #Save NECESSARY parameter of state-interaction calculations
    adc_prism.mo_b = mo_a #Update mo_b
    adc_prism.mo = mo_a
    adc_prism.rdm_aabb_ref = rdm_aabb
    adc_prism.S_ref = S_old_total

    return adc_prism.rdm_aabb_ref, adc_prism.S_ref, adc_prism.E_ref
        












        