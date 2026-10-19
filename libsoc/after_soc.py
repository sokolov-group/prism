import sys
import prism
import numpy as np
import prism.lib.logger as logger
from prism.tools import trans_prop
from prism.libsoc import general_somf
from prism.libsoc import magnetic

def casscf_soc(interface):
    '''
    Calculate SOC-CASSCF
    '''
    
    interface.mol.max_memory = interface.max_memory_soc

    cput0 = (logger.process_clock(), logger.perf_counter())
    wfn = interface.ref_wfn

    # Calculate the total spin (S)
    spin_mult_wfn = interface.ref_wfn_spin_mult

    S = [round((spin_mult-1)/2,2) for spin_mult in spin_mult_wfn]

    # Calculate the spin projections Ms
    ms = []
    nstate = len(wfn)
    for I in range(nstate):
        sz = interface.apply_S_z(wfn[I],interface.ncas,interface.ref_nelecas[I])
        ms.append(np.dot(wfn[I].ravel(), sz.ravel()))

    ms = [round(elem,2) for elem in ms]

    # Make sure ms is the same across all states:
    for I in range(nstate):
        if (np.abs(ms[I]-ms[0])>1e-8):
            raise Exception("Each state's ms should be same for state_interaction_SOC function")

    rdm_aabb = np.zeros((2, nstate, nstate, interface.nmo, interface.nmo))
    for I in range(nstate):
        for J in range(nstate):
            tmprdm_aabb = interface.trans_rdm1s(wfn[J], wfn[I], interface.ncas, interface.ref_nelecas[I])
            rdm_aabb[0, I, J, interface.ncore:interface.ncore+interface.ncas, interface.ncore:interface.ncore+interface.ncas] = tmprdm_aabb[0]
            rdm_aabb[1, I, J, interface.ncore:interface.ncore+interface.ncas, interface.ncore:interface.ncore+interface.ncas] = tmprdm_aabb[1]

            if I == J:
                #uncorrelated diagonal terms
                rdm_aabb[:,I, J, :interface.ncore, :interface.ncore] = np.identity(interface.ncore)  


    #generalSOC requires spin-free energy...
    en = interface.e_ref

    #If Ms=0 , CG coefficent vanish...
    if ms[0] != 0:
        en_soc, evec_soc = general_somf.state_interaction_soc(interface, en, rdm_aabb, S, ms[0], interface.soc, interface.verbose)
    
    else:    
        interface.log.info("Apply S_plus due to Ms=0...")

        wfn_plus = wfn.copy()
        wfn_ref_nelecas_plus = interface.ref_nelecas.copy()
        ms_plus = ms.copy()

        for I in range(nstate):
            if S[I] > 0 :
                wfn_plus[I], Sp_ne = interface.apply_S_plus(wfn[I],interface.ncas,interface.ref_nelecas[I])
                # Upadate ref_nelecas
                wfn_ref_nelecas_plus[I] = Sp_ne
                # Normalize the wfn
                wfn_plus[I] = wfn_plus[I]/(np.sqrt( S[I]**2 + S[I] ))
                # Upadate ms
                ms_plus[I] = 1
            elif S[I] == 0:     
                wfn_plus[I] = np.zeros_like(wfn_plus[I])


        # Calculate RDM_aabb_plus
        rdm_aabb_plus = np.zeros((2, nstate, nstate, interface.nmo, interface.nmo))
        for I in range(nstate):
            for J in range(nstate):
                if (wfn_ref_nelecas_plus[I] == wfn_ref_nelecas_plus[J]):
                    tmprdm_aabb_plus = interface.trans_rdm1s(wfn_plus[J], wfn_plus[I], interface.ncas,  wfn_ref_nelecas_plus[I])
                    rdm_aabb_plus[0, I, J, interface.ncore:interface.ncore+interface.ncas, interface.ncore:interface.ncore+interface.ncas] = tmprdm_aabb_plus[0]
                    rdm_aabb_plus[1, I, J, interface.ncore:interface.ncore+interface.ncas, interface.ncore:interface.ncore+interface.ncas] = tmprdm_aabb_plus[1]

                    if I == J:
                        #uncorrelated diagonal terms
                        rdm_aabb_plus[:,I, J, :interface.ncore, :interface.ncore] = np.identity(interface.ncore)  

        en_soc, evec_soc = general_somf.state_interaction_soc_ms0(interface, en, rdm_aabb,  rdm_aabb_plus, S, interface.soc, interface.verbose)

    #Compute properties
    compute_properties(interface, en_soc, evec_soc, rdm_aabb, S)

    #Print result
    print_results(interface, en_soc, interface.soc, "CASSCF")

    return  en_soc, interface.properties_soc["osc_strengths"]


def compute_properties(interface, en_soc, evec_soc, rdm_aabb_ref, S_ref):

    #Compute OSC
    #compute soc rdm
    rdm_sf = rdm_aabb_ref[0] + rdm_aabb_ref[1]
    I_total = []
    S_total = []
    ms_total = []
    for i in range(len(S_ref)):
        s = S_ref[i]
        n = int(2*s+1)
        for j in range(n):
            I_total.append(i)
            S_total.append(s)
            m = s-j
            ms_total.append(m)

    nstate_total = len(S_total)
    rdm_mo = np.zeros((nstate_total, nstate_total, interface.nmo, interface.nmo),dtype='complex')
    for I in range(nstate_total):
        for J in  range(nstate_total):
            if (np.abs(S_total[I]-S_total[J])<1e-8) and (np.abs(ms_total[I]-ms_total[J])<1e-8):
                rdm_mo[I,J] = rdm_sf[I_total[I], I_total[J]]
    rdm_soc = np.einsum('ai,ibIJ,bj->ajIJ',np.conj(evec_soc).T , rdm_mo , evec_soc)
    # Calculate ground state degeneracy:
    deg_gs = 1
    for i in range(len(en_soc)-1):
        if (np.abs(en_soc[i+1] - en_soc[0])) < 1e-5:
            deg_gs += 1
        else:
            break

    # Calculate oscillator strengths for transitions from the first state
    osc_str_soc_full=[]
    osc_str_soc = np.zeros(len(en_soc)-1)
    for gs_index in range(deg_gs):
        e_diff = en_soc - en_soc[gs_index]
        e_diff = e_diff[gs_index+1:]
        osc = trans_prop.osc_strength(interface, e_diff, rdm_soc[ gs_index, gs_index+1:])
        osc_str_soc_full.append(osc)
        osc_str_soc[gs_index:] += osc 
    interface.properties_soc["osc_strengths"] = osc_str_soc

    # Compute oscillator strengths starting from each state
    if interface.verbose >= 5:
        for gs_index in range(deg_gs, len(en_soc)):  
            e_diff = en_soc - en_soc[gs_index]
            e_diff = e_diff[gs_index+1:]
            osc_str_soc_full.append(trans_prop.osc_strength(interface, e_diff, rdm_mo[  gs_index, gs_index+1:]))
        interface.properties_soc["osc_strengths_full"] = osc_str_soc_full
    
    # eigenvector analysis
    if interface.analyze_evec_soc:
        population = evec_soc * np.conj(evec_soc)
        population = np.real(population)
        I = 0
        population_state = np.zeros((len(S_ref), len(evec_soc)))
        for i in range(len(S_ref)):
            multicity = int(S_ref[i]*2 +1)
            for j in range(multicity):
                J = I + j
                population_state[i] += population[J]
            I += multicity
        if len(population_state[:,0]) < 5:
            k = len(population_state[:,0])
        else:
            k = 5
        # Step 1: each column find large row index
        row_idx = np.argpartition(population_state, -k, axis=0)[-k:, :]
        # Step 2: value according to row index
        values = population_state[row_idx, np.arange(population_state.shape[1])]
        # Step 3: large to small order 
        order = np.argsort(-values, axis=0)
        # Step 4: after order row index
        sorted_row_idx = np.take_along_axis(row_idx, order, axis=0)
        values_sort = population_state[sorted_row_idx, np.arange(population_state.shape[1])]
        interface.log.info("\nSOC index = [sf index]")
        for i in range(len(sorted_row_idx[0])):
            index_value = np.zeros((2,len(sorted_row_idx[:,i])))
            index_value[0] = sorted_row_idx[:,i] + 1
            index_value[1] = np.round(values_sort[:,i],2)
            interface.log.info("%d = %s %s", i + 1, index_value[0], index_value[1])
        interface.log.info("")
    
    #Compute magnetic properties
    if (interface.gtensor or interface.mag_av or  interface.sus_av or  interface.mag_vec or interface.sus_tensor):
        properties_mag = magnetic.compute_properties(interface, rdm_sf, en_soc, evec_soc, S_ref,  method = None)
        interface.properties_soc.update(properties_mag)
    
    return osc_str_soc


def print_results(interface, en_soc, soc, method):
    # Print results obtained from SOC calculation
    print_soc(interface, en_soc, soc, method)

    if "osc_strengths_full" in interface.properties_soc:
        trans_prop.print_osc_strength(interface, interface.properties_soc["osc_strengths_full"])
    if "magnetic_properties_soc" in interface.properties_soc:
        magnetic.print_mag_properties(interface, interface.properties_soc,  method = None)


def print_soc(interface, en_soc, soc, method):
    '''
    Print energy of SOC-CASSCF
    '''
    
    h2ev = interface.hartree_to_ev
    h2cm = interface.hartree_to_inv_cm
    osc_str_soc = interface.properties_soc["osc_strengths"]

    interface.log.info("\nSummary of SI-%s-%s", soc.upper(),method.upper())
    interface.log.info("------------------------------------------------------------------------------------------------------------------")
    interface.log.info("  State    Degen.            E(total)         E(eV)           dE(eV)     dE(nm)        dE(cm-1)        Osc Str.  ")
    interface.log.info("------------------------------------------------------------------------------------------------------------------")

    e_gs = en_soc[0]
    e_tot = en_soc

    n_states = len(e_tot)

    for p in range(n_states):
        deg = 1
        de = e_tot[p] - e_gs
        e_ev = e_tot[p] * h2ev
        de_ev = de * h2ev
        de_cm = de * h2cm
        if p == 0 or abs(de) < 1e-5:
            interface.log.info("%5d       %2d      %20.12f %14.8f %12.4f %12s %14.4f   %12s" % ((p+1), deg, e_tot[p], e_ev, de_ev, " ", de_cm, " "))
        else:
            de_nm = 10000000 / de_cm
            interface.log.info("%5d       %2d      %20.12f %14.8f %12.4f %12.4f %14.4f    %12.8f" % ((p+1), deg, e_tot[p], e_ev, de_ev, de_nm, de_cm, osc_str_soc[p-1]))
    interface.log.info("----------------------------------------------------------------------------------------------------------------")
    
    sys.stdout.flush()
    
    return 