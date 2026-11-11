import sys
import prism
import numpy as np
import prism.lib.logger as logger
from prism.tools import trans_prop
from prism.libsoc import general_somf
from prism.libsoc import before_soc
from prism.libsoc import after_soc
from prism.libsoc import magnetic

def collect_sf(method):
    '''
    Calculate SOC-CASSCF
    '''

    method.log.info("Collecting necessary information for SOC")
    
    #Parameter
    method.interface.mol.max_memory = method.max_memory_soc
    mc                  = method.mc            
    nmo                 = method.nmo               
    ncas                = method.ncas              
    ncore               = method.ncore             
    ref_nelecas         = method.ref_nelecas       
    e_ref               = method.e_ref             
    mo                  = method.mo                
    ref_wfn             = method.ref_wfn           
    ref_wfn_spin_mult   = method.ref_wfn_spin_mult 
    wfn                 = method.ref_wfn
    e_ref               = method.e_ref
    name_sf             = method.name_sf

    # Calculate the total spin (S)
    S = [round((spin_mult-1)/2,2) for spin_mult in ref_wfn_spin_mult]

    # Calculate the spin projections Ms
    ms = []
    nstate = len(wfn)
    for I in range(nstate):
        sz = method.interface.apply_S_z(wfn[I], ncas, ref_nelecas[I])
        ms.append(np.dot(wfn[I].ravel(), sz.ravel()))

    ms = [round(elem,2) for elem in ms]

    # Make sure ms is the same across all states:
    for I in range(nstate):
        if (np.abs(ms[I]-ms[0])>1e-8):
            raise Exception("Each state's ms should be same for state_interaction_SOC function")

    rdm_aabb = np.zeros((2, nstate, nstate, nmo, nmo))
    for I in range(nstate):
        for J in range(nstate):
            tmprdm_aabb = method.interface.trans_rdm1s(wfn[J], wfn[I], ncas, ref_nelecas[I])
            rdm_aabb[0, I, J, ncore:ncore+ncas, ncore:ncore+ncas] = tmprdm_aabb[0]
            rdm_aabb[1, I, J, ncore:ncore+ncas, ncore:ncore+ncas] = tmprdm_aabb[1]

            if I == J:
                #uncorrelated diagonal terms
                rdm_aabb[:,I, J, :ncore, :ncore] = np.identity(ncore)  

    #If Ms=0 , CG coefficent vanish...
    if ms[0] != 0:

        #Save NECESSARY parameter of state-interaction calculations
        method.E_ref = e_ref
        method.rdm_aabb_ref = rdm_aabb
        method.S_ref = S
        method.ms_ref = ms[0]  
    
    else:   

        method.log.info("Apply S_plus due to Ms=0...")

        wfn_plus = wfn.copy()
        wfn_ref_nelecas_plus = ref_nelecas.copy()
        ms_plus = ms.copy()

        for I in range(nstate):
            if S[I] > 0 :
                wfn_plus[I], Sp_ne = method.interface.apply_S_plus(wfn[I], ncas, ref_nelecas[I])
                # Upadate ref_nelecas
                wfn_ref_nelecas_plus[I] = Sp_ne
                # Normalize the wfn
                wfn_plus[I] = wfn_plus[I]/(np.sqrt( S[I]**2 + S[I] ))
                # Upadate ms
                ms_plus[I] = 1
            elif S[I] == 0:     
                wfn_plus[I] = np.zeros_like(wfn_plus[I])


        # Calculate RDM_aabb_plus
        rdm_aabb_plus = np.zeros((2, nstate, nstate, nmo, nmo))
        for I in range(nstate):
            for J in range(nstate):
                if (wfn_ref_nelecas_plus[I] == wfn_ref_nelecas_plus[J]):
                    tmprdm_aabb_plus = method.interface.trans_rdm1s(wfn_plus[J], wfn_plus[I], ncas,  wfn_ref_nelecas_plus[I])
                    rdm_aabb_plus[0, I, J, ncore : ncore + ncas, ncore : ncore + ncas] = tmprdm_aabb_plus[0]
                    rdm_aabb_plus[1, I, J, ncore : ncore + ncas, ncore : ncore + ncas] = tmprdm_aabb_plus[1]

                    if I == J:
                        #uncorrelated diagonal terms
                        rdm_aabb_plus[:,I, J, :ncore, :ncore] = np.identity(ncore)


        #Save NECESSARY parameter of state-interaction calculations
        method.E_ref = e_ref
        method.rdm_aabb_ref = rdm_aabb
        method.S_ref = S
        method.rdm_aabb_plus_ref = rdm_aabb_plus  


