import sys
import time
import numpy as np

def compute_gs_rdms(mr_adc):

    start_time = time.time()

    print ("Computing ground-state RDMs...")
    sys.stdout.flush()

    # TODO: for open-shells, this needs to perform state-averaging 

    # Compute ground-state RDMs
    if mr_adc.ncas != 0:
        mr_adc.rdm.ca, mr_adc.rdm.ccaa, mr_adc.rdm.cccaaa = mr_adc.interface.compute_rdm123(mr_adc.wfn_casscf, mr_adc.wfn_casscf, mr_adc.nelecas)
    else:
        mr_adc.rdm.ca = np.zeros((mr_adc.ncas, mr_adc.ncas))
        mr_adc.rdm.ccaa =  np.zeros((mr_adc.ncas, mr_adc.ncas, mr_adc.ncas, mr_adc.ncas))
        mr_adc.rdm.cccaaa =  np.zeros((mr_adc.ncas, mr_adc.ncas, mr_adc.ncas, mr_adc.ncas, mr_adc.ncas, mr_adc.ncas))

    print ("Time for computing ground-state RDMs:                          %f sec\n" % (time.time() - start_time))


def compute_es_rdms_so(mr_adc):

    start_time = time.time()

    wfn_casci = None
    e_cas_ci = None

    print ("Computing excited-state CASCI wavefunctions...\n")
    sys.stdout.flush()

    # Compute CASCI wavefunctions for excited states in the active space
    if mr_adc.method_type == "ip":

        mr_adc.nelecasci = (mr_adc.nelecas[0] - 1, mr_adc.nelecas[1])

        if (0 <= mr_adc.nelecasci[0] <= mr_adc.ncas and 0 <= mr_adc.nelecasci[1] <= mr_adc.ncas):
            e_cas_ci, wfn_casci = mr_adc.interface.compute_casci_ip_ea(mr_adc.ncasci, mr_adc.method_type)
        else:
            mr_adc.nelecasci = None

    elif mr_adc.method_type == "ea":

        mr_adc.nelecasci = (mr_adc.nelecas[0] + 1, mr_adc.nelecas[1])

        if (0 <= mr_adc.nelecasci[0] <= mr_adc.ncas and 0 <= mr_adc.nelecasci[1] <= mr_adc.ncas):
            e_cas_ci, wfn_casci = mr_adc.interface.compute_casci_ip_ea(mr_adc.ncasci, mr_adc.method_type)
        else:
            mr_adc.nelecasci = None

    elif mr_adc.method_type == "ee":

        mr_adc.nelecasci = (mr_adc.nelecas[0], mr_adc.nelecas[1])
        
        if (mr_adc.nelecasci[0] != 0 or mr_adc.nelecasci[1] != 0) and (mr_adc.nelecasci[0] != mr_adc.ncas or mr_adc.nelecasci[1] != mr_adc.ncas): 
            e_cas_ci, wfn_casci = mr_adc.interface.compute_casci_ee(mr_adc.ncasci)
        else:
            mr_adc.nelecasci = None

    elif mr_adc.method_type in ("cvs-ip", "cvs-ee"):

        mr_adc.nelecasci = None

    else:
        raise Exception("MR-ADC is not implemented for %s" % mr_adc.method_type)

    if mr_adc.nelecasci is not None:
        mr_adc.ncasci = len(e_cas_ci)
        mr_adc.wfn_casci = wfn_casci
        mr_adc.e_cas_ci = e_cas_ci
    else:
        if mr_adc.method_type in ("cvs-ip", "cvs-ee"):
            print ("Requested method type %s does not require running a CASCI calculation..." % mr_adc.method_type)
        else:
            print ("WARNING: active orbitals are either empty of completely filled...")
        print ("Skipping the CASCI calculation...")
        mr_adc.ncasci = 0
        mr_adc.wfn_casci = None
        mr_adc.e_cas_ci = None

    print ("\nFinal number of excited CASCI states: %d\n" % mr_adc.ncasci)

    if mr_adc.ncasci > 0:

        # Compute transition RDMs between the ground (reference) state and target CASCI states
        print ("Computing transition RDMs between reference and target CASCI states...\n")
        sys.stdout.flush()

        if mr_adc.method_type == "ip":
            # Compute CASCI states with higher MS
            Sp_wfn_casci = []
            Sp_wfn_ne = None
            for wfn in wfn_casci:
                Sp_wfn, Sp_wfn_ne = mr_adc.interface.apply_S_plus(wfn, mr_adc.ncas, mr_adc.nelecasci)
                Sp_wfn_casci.append(Sp_wfn)

            mr_adc.rdm.ct = np.zeros((2 * mr_adc.ncasci, mr_adc.ncas_so))
            mr_adc.rdm.ct[:mr_adc.ncasci] = mr_adc_rdms.compute_rdm_ct_so(mr_adc.interface, wfn_casci, mr_adc.nelecasci).copy()
            mr_adc.rdm.ct[mr_adc.ncasci:] = mr_adc_rdms.compute_rdm_ct_so(mr_adc.interface, Sp_wfn_casci, Sp_wfn_ne).copy()

            mr_adc.rdm.ccat = np.zeros((2 * mr_adc.ncasci, mr_adc.ncas_so, mr_adc.ncas_so, mr_adc.ncas_so))
            mr_adc.rdm.ccat[:mr_adc.ncasci] = mr_adc_rdms.compute_rdm_ccat_so(mr_adc.interface, wfn_casci, mr_adc.nelecasci).copy()
            mr_adc.rdm.ccat[mr_adc.ncasci:] = mr_adc_rdms.compute_rdm_ccat_so(mr_adc.interface, Sp_wfn_casci, Sp_wfn_ne).copy()

            if mr_adc.method in ("mr-adc(1)", "mr-adc(2)", "mr-adc(2)-x"):
                mr_adc.rdm.cccaat = np.zeros((2 * mr_adc.ncasci, mr_adc.ncas_so, mr_adc.ncas_so, mr_adc.ncas_so, mr_adc.ncas_so, mr_adc.ncas_so))
                mr_adc.rdm.cccaat[:mr_adc.ncasci] = mr_adc_rdms.compute_rdm_cccaat_so(mr_adc.interface, wfn_casci, mr_adc.nelecasci).copy()
                mr_adc.rdm.cccaat[mr_adc.ncasci:] = mr_adc_rdms.compute_rdm_cccaat_so(mr_adc.interface, Sp_wfn_casci, Sp_wfn_ne).copy()

        elif mr_adc.method_type == "ea":
            # Compute CASCI states with lower MS
            Sm_wfn_casci = []
            Sm_wfn_ne = None
            for wfn in wfn_casci:
                Sm_wfn, Sm_wfn_ne = mr_adc.interface.apply_S_minus(wfn, mr_adc.ncas, mr_adc.nelecasci)
                Sm_wfn_casci.append(Sm_wfn)

            mr_adc.rdm.tc = np.zeros((2 * mr_adc.ncasci, mr_adc.ncas_so))
            mr_adc.rdm.tc[:mr_adc.ncasci] = mr_adc_rdms.compute_rdm_tc_so(mr_adc.interface, wfn_casci, mr_adc.nelecasci).copy()
            mr_adc.rdm.tc[mr_adc.ncasci:] = mr_adc_rdms.compute_rdm_tc_so(mr_adc.interface, Sm_wfn_casci, Sm_wfn_ne).copy()

            mr_adc.rdm.tcca = np.zeros((2 * mr_adc.ncasci, mr_adc.ncas_so, mr_adc.ncas_so, mr_adc.ncas_so))
            mr_adc.rdm.tcca[:mr_adc.ncasci] = mr_adc_rdms.compute_rdm_tcca_so(mr_adc.interface, wfn_casci, mr_adc.nelecasci).copy()
            mr_adc.rdm.tcca[mr_adc.ncasci:] = mr_adc_rdms.compute_rdm_tcca_so(mr_adc.interface, Sm_wfn_casci, Sm_wfn_ne).copy()

            if mr_adc.method in ("mr-adc(1)", "mr-adc(2)", "mr-adc(2)-x"):
                mr_adc.rdm.tcccaa = np.zeros((2 * mr_adc.ncasci, mr_adc.ncas_so, mr_adc.ncas_so, mr_adc.ncas_so, mr_adc.ncas_so, mr_adc.ncas_so))
                mr_adc.rdm.tcccaa[:mr_adc.ncasci] = mr_adc_rdms.compute_rdm_tcccaa_so(mr_adc.interface, wfn_casci, mr_adc.nelecasci).copy()
                mr_adc.rdm.tcccaa[mr_adc.ncasci:] = mr_adc_rdms.compute_rdm_tcccaa_so(mr_adc.interface, Sm_wfn_casci, Sm_wfn_ne).copy()

        elif mr_adc.method_type == "ee":
            mr_adc.rdm.tca = mr_adc_rdms.compute_rdm_tca_so(mr_adc.interface, wfn_casci, mr_adc.nelecasci).copy()
            mr_adc.rdm.tccaa = mr_adc_rdms.compute_rdm_tccaa_so(mr_adc.interface, wfn_casci, mr_adc.nelecasci).copy()

            if mr_adc.method in ("mr-adc(1)", "mr-adc(2)", "mr-adc(2)-x"):
                mr_adc.rdm.tcccaaa = mr_adc_rdms.compute_rdm_tcccaaa_so(mr_adc.interface, wfn_casci, mr_adc.nelecasci).copy()

        print ("Computing transition RDMs between target CASCI states...\n")
        sys.stdout.flush()

        # Compute transition RDMs between two target CASCI states
        mr_adc.rdm.tcat = mr_adc_rdms.compute_rdm_tcat_so(mr_adc.interface, wfn_casci, mr_adc.nelecasci)
        mr_adc.rdm.tccaat = mr_adc_rdms.compute_rdm_tccaat_so(mr_adc.interface, wfn_casci, mr_adc.nelecasci)

    print ("Time for computing excited-state RDMs:                          %f sec\n" % (time.time() - start_time))

#def compute_rdm_ca_so(interface, bra = None, ket = None, nelecas = None):
#
#    ncas = interface.ncas
#
#    if bra is None:
#        bra = interface.wfn_casscf
#    if ket is None:
#        ket = interface.wfn_casscf
#    if nelecas is None:
#        nelecas = interface.nelecas
#
#    rdm = interface.compute_rdm_ca_si(bra, ket, nelecas)
#
#    rdm_so = np.zeros((2 * ncas, 2 * ncas))
#    rdm_so[::2,::2] = rdm[0].copy()
#    rdm_so[1::2,1::2] = rdm[1].copy()
#
#    return rdm_so
#
#def compute_rdm_ca_general_so(interface, bra = None, ket = None, nelecas_bra = None, nelecas_ket = None):
#
#    ncas = interface.ncas
#
#    if bra is None:
#        bra = interface.wfn_casscf
#    if ket is None:
#        ket = interface.wfn_casscf
#    if nelecas_bra is None:
#        nelecas_bra = interface.nelecas[0]
#    if nelecas_ket is None:
#        nelecas_ket = interface.nelecas[0]
#
#    bra_ne = nelecas_bra
#    ket_ne = nelecas_ket
#    rdm = interface.compute_rdm_ca_general_si(bra, ket, bra_ne, ket_ne)
#
#    rdm_so = np.zeros((2 * ncas, 2 * ncas))
#    rdm_so[::2,::2] = rdm[0].copy()
#    rdm_so[::2,1::2] = rdm[1].copy()
#    rdm_so[1::2,::2] = rdm[2].copy()
#    rdm_so[1::2,1::2] = rdm[3].copy()
#
#    return rdm_so
#
#def compute_rdm_ccaa_general_so(interface, bra = None, ket = None, nelecas_bra = None, nelecas_ket = None):
#
#    ncas = interface.ncas
#    if bra is None:
#        bra = interface.wfn_casscf
#    if ket is None:
#        ket = interface.wfn_casscf
#    if nelecas_bra is None:
#        nelecas_bra = interface.nelecas[0]
#    if nelecas_ket is None:
#        nelecas_ket = interface.nelecas[0]
#
#    bra_ne = nelecas_bra
#    ket_ne = nelecas_ket
#    rdm = interface.compute_general_rdm_ccaa_si(bra, ket, bra_ne, ket_ne)
#
#    rdm_so = np.zeros((2 * ncas, 2 * ncas, 2 * ncas, 2 * ncas))
#    rdm_so[::2,::2,::2,::2] = rdm[0].copy()
#    rdm_so[1::2,1::2,1::2,1::2] = rdm[1].copy()
#    rdm_so[::2,1::2,::2,1::2] = rdm[2].copy()
#    rdm_so[::2,::2,1::2,1::2] = rdm[3].copy()
#    rdm_so[1::2,::2,::2,1::2] = rdm[4].copy()
#    rdm_so[1::2,::2,1::2,::2] = rdm[5].copy()
#    rdm_so[1::2,1::2,::2,::2] = rdm[6].copy()
#    rdm_so[::2,1::2,1::2,::2] = rdm[7].copy()
#    rdm_so[::2,::2,::2,1::2] = rdm[8].copy()
#    rdm_so[::2,::2,1::2,::2] = rdm[9].copy()
#    rdm_so[::2,1::2,::2,::2] = rdm[10].copy()
#    rdm_so[1::2,::2,::2,::2] = rdm[11].copy()
#    rdm_so[1::2,1::2,1::2,::2] = rdm[12].copy()
#    rdm_so[1::2,1::2,::2,1::2] = rdm[13].copy()
#    rdm_so[1::2,::2,1::2,1::2] = rdm[14].copy()
#    rdm_so[::2,1::2,1::2,1::2] = rdm[15].copy()
#
#    return rdm_so
#
#def compute_rdm_ccaa_so(interface, bra = None, ket = None, nelecas = None):
#
#    ncas = interface.ncas
#
#    if bra is None:
#        bra = interface.wfn_casscf
#    if ket is None:
#        ket = interface.wfn_casscf
#    if nelecas is None:
#        nelecas = interface.nelecas
#
#    rdm_aa, rdm_ab, rdm_bb = interface.compute_rdm_ccaa_si(bra, ket, nelecas)
#
#    rdm_so = np.zeros((2 * ncas, 2 * ncas, 2 * ncas, 2 * ncas))
#    rdm_so[::2,::2,::2,::2] = rdm_aa.copy()
#    rdm_so[::2,1::2,1::2,::2] = rdm_ab.copy()
#    rdm_so[1::2,::2,1::2,::2] = -rdm_ab.transpose(1,0,2,3).copy()
#    rdm_so[::2,1::2,::2,1::2] = -rdm_ab.transpose(0,1,3,2).copy()
#    rdm_so[1::2,::2,::2,1::2] = rdm_ab.transpose(1,0,3,2).copy()
#    rdm_so[1::2,1::2,1::2,1::2] = rdm_bb.copy()
#
#    return rdm_so
#
#
#def compute_rdm_cccaaa_so(interface, bra = None, ket = None, nelecas = None):
#
#    ncas = interface.ncas
#
#    if bra is None:
#        bra = interface.wfn_casscf
#    if ket is None:
#        ket = interface.wfn_casscf
#    if nelecas is None:
#        nelecas = interface.nelecas
#
#    rdm_aaa, rdm_aab, rdm_abb, rdm_bbb = interface.compute_rdm_cccaaa_si(bra, ket, nelecas)
#
#    rdm_so = np.zeros((2 * ncas, 2 * ncas, 2 * ncas, 2 * ncas, 2 * ncas, 2 * ncas))
#    rdm_so[::2,::2,::2,::2,::2,::2] = rdm_aaa.copy()
#    rdm_so[1::2,::2,::2,::2,::2,1::2] = rdm_aab.copy()
#    rdm_so[::2,1::2,::2,::2,::2,1::2] = -rdm_aab.transpose(1,0,2,3,4,5).copy()
#    rdm_so[::2,::2,1::2,::2,::2,1::2] = rdm_aab.transpose(1,2,0,3,4,5).copy()
#    rdm_so[1::2,::2,::2,::2,1::2,::2] = -rdm_aab.transpose(0,1,2,3,5,4).copy()
#    rdm_so[::2,1::2,::2,::2,1::2,::2] = rdm_aab.transpose(1,0,2,3,5,4).copy()
#    rdm_so[::2,::2,1::2,::2,1::2,::2] = -rdm_aab.transpose(1,2,0,3,5,4).copy()
#    rdm_so[1::2,::2,::2,1::2,::2,::2] = rdm_aab.transpose(0,1,2,5,3,4).copy()
#    rdm_so[::2,1::2,::2,1::2,::2,::2] = -rdm_aab.transpose(1,0,2,5,3,4).copy()
#    rdm_so[::2,::2,1::2,1::2,::2,::2] = rdm_aab.transpose(1,2,0,5,3,4).copy()
#    rdm_so[1::2,1::2,::2,::2,1::2,1::2] = rdm_abb.copy()
#    rdm_so[1::2,::2,1::2,::2,1::2,1::2] = -rdm_abb.transpose(0,2,1,3,4,5).copy()
#    rdm_so[::2,1::2,1::2,::2,1::2,1::2] = rdm_abb.transpose(2,0,1,3,4,5).copy()
#    rdm_so[1::2,1::2,::2,1::2,::2,1::2] = -rdm_abb.transpose(0,1,2,4,3,5).copy()
#    rdm_so[1::2,::2,1::2,1::2,::2,1::2] = rdm_abb.transpose(0,2,1,4,3,5).copy()
#    rdm_so[::2,1::2,1::2,1::2,::2,1::2] = -rdm_abb.transpose(2,0,1,4,3,5).copy()
#    rdm_so[1::2,1::2,::2,1::2,1::2,::2] = rdm_abb.transpose(0,1,2,4,5,3).copy()
#    rdm_so[1::2,::2,1::2,1::2,1::2,::2] = -rdm_abb.transpose(0,2,1,4,5,3).copy()
#    rdm_so[::2,1::2,1::2,1::2,1::2,::2] = rdm_abb.transpose(2,0,1,4,5,3).copy()
#    rdm_so[1::2,1::2,1::2,1::2,1::2,1::2] = rdm_bbb.copy()
#
#    return rdm_so
#
#
#def compute_rdm_ccccaaaa_so(interface, bra = None, ket = None, nelecas = None):
#
#    ncas = interface.ncas
#
#    if bra is None:
#        bra = interface.wfn_casscf
#    if ket is None:
#        ket = interface.wfn_casscf
#    if nelecas is None:
#        nelecas = interface.nelecas
#
#    rdm_aaaa, rdm_aaab, rdm_aabb, rdm_abbb, rdm_bbbb = interface.compute_rdm_ccccaaaa_si(bra, ket, nelecas)
#
#    rdm_so = np.zeros((2 * ncas, 2 * ncas, 2 * ncas, 2 * ncas, 2 * ncas, 2 * ncas, 2 * ncas, 2 * ncas))
#    rdm_so[::2,::2,::2,::2,::2,::2,::2,::2] = rdm_aaaa.copy()
#    rdm_so[1::2,::2,::2,::2,::2,::2,::2,1::2] =  rdm_aaab.copy()
#    rdm_so[::2,1::2,::2,::2,::2,::2,::2,1::2] = -rdm_aaab.transpose(1,0,2,3,4,5,6,7).copy()
#    rdm_so[::2,::2,1::2,::2,::2,::2,::2,1::2] =  rdm_aaab.transpose(1,2,0,3,4,5,6,7).copy()
#    rdm_so[::2,::2,::2,1::2,::2,::2,::2,1::2] = -rdm_aaab.transpose(1,2,3,0,4,5,6,7).copy()
#    rdm_so[1::2,::2,::2,::2,::2,::2,1::2,::2] = -rdm_aaab.transpose(0,1,2,3,4,5,7,6).copy()
#    rdm_so[::2,1::2,::2,::2,::2,::2,1::2,::2] =  rdm_aaab.transpose(1,0,2,3,4,5,7,6).copy()
#    rdm_so[::2,::2,1::2,::2,::2,::2,1::2,::2] = -rdm_aaab.transpose(1,2,0,3,4,5,7,6).copy()
#    rdm_so[::2,::2,::2,1::2,::2,::2,1::2,::2] =  rdm_aaab.transpose(1,2,3,0,4,5,7,6).copy()
#    rdm_so[1::2,::2,::2,::2,::2,1::2,::2,::2] =  rdm_aaab.transpose(0,1,2,3,4,7,5,6).copy()
#    rdm_so[::2,1::2,::2,::2,::2,1::2,::2,::2] = -rdm_aaab.transpose(1,0,2,3,4,7,5,6).copy()
#    rdm_so[::2,::2,1::2,::2,::2,1::2,::2,::2] =  rdm_aaab.transpose(1,2,0,3,4,7,5,6).copy()
#    rdm_so[::2,::2,::2,1::2,::2,1::2,::2,::2] = -rdm_aaab.transpose(1,2,3,0,4,7,5,6).copy()
#    rdm_so[1::2,::2,::2,::2,1::2,::2,::2,::2] = -rdm_aaab.transpose(0,1,2,3,7,4,5,6).copy()
#    rdm_so[::2,1::2,::2,::2,1::2,::2,::2,::2] =  rdm_aaab.transpose(1,0,2,3,7,4,5,6).copy()
#    rdm_so[::2,::2,1::2,::2,1::2,::2,::2,::2] = -rdm_aaab.transpose(1,2,0,3,7,4,5,6).copy()
#    rdm_so[::2,::2,::2,1::2,1::2,::2,::2,::2] =  rdm_aaab.transpose(1,2,3,0,7,4,5,6).copy()
#    rdm_so[1::2,1::2,::2,::2,::2,::2,1::2,1::2] =  rdm_aabb.copy()
#    rdm_so[1::2,::2,1::2,::2,::2,::2,1::2,1::2] = -rdm_aabb.transpose(0,2,1,3,4,5,6,7).copy()
#    rdm_so[1::2,::2,::2,1::2,::2,::2,1::2,1::2] =  rdm_aabb.transpose(0,2,3,1,4,5,6,7).copy()
#    rdm_so[::2,1::2,1::2,::2,::2,::2,1::2,1::2] =  rdm_aabb.transpose(2,0,1,3,4,5,6,7).copy()
#    rdm_so[::2,1::2,::2,1::2,::2,::2,1::2,1::2] = -rdm_aabb.transpose(2,0,3,1,4,5,6,7).copy()
#    rdm_so[::2,::2,1::2,1::2,::2,::2,1::2,1::2] =  rdm_aabb.transpose(2,3,0,1,4,5,6,7).copy()
#    rdm_so[1::2,1::2,::2,::2,::2,1::2,::2,1::2] = -rdm_aabb.transpose(0,1,2,3,4,6,5,7).copy()
#    rdm_so[1::2,::2,1::2,::2,::2,1::2,::2,1::2] =  rdm_aabb.transpose(0,2,1,3,4,6,5,7).copy()
#    rdm_so[1::2,::2,::2,1::2,::2,1::2,::2,1::2] = -rdm_aabb.transpose(0,2,3,1,4,6,5,7).copy()
#    rdm_so[::2,1::2,1::2,::2,::2,1::2,::2,1::2] = -rdm_aabb.transpose(2,0,1,3,4,6,5,7).copy()
#    rdm_so[::2,1::2,::2,1::2,::2,1::2,::2,1::2] =  rdm_aabb.transpose(2,0,3,1,4,6,5,7).copy()
#    rdm_so[::2,::2,1::2,1::2,::2,1::2,::2,1::2] = -rdm_aabb.transpose(2,3,0,1,4,6,5,7).copy()
#    rdm_so[1::2,1::2,::2,::2,1::2,::2,::2,1::2] =  rdm_aabb.transpose(0,1,2,3,6,4,5,7).copy()
#    rdm_so[1::2,::2,1::2,::2,1::2,::2,::2,1::2] = -rdm_aabb.transpose(0,2,1,3,6,4,5,7).copy()
#    rdm_so[1::2,::2,::2,1::2,1::2,::2,::2,1::2] =  rdm_aabb.transpose(0,2,3,1,6,4,5,7).copy()
#    rdm_so[::2,1::2,1::2,::2,1::2,::2,::2,1::2] =  rdm_aabb.transpose(2,0,1,3,6,4,5,7).copy()
#    rdm_so[::2,1::2,::2,1::2,1::2,::2,::2,1::2] = -rdm_aabb.transpose(2,0,3,1,6,4,5,7).copy()
#    rdm_so[::2,::2,1::2,1::2,1::2,::2,::2,1::2] =  rdm_aabb.transpose(2,3,0,1,6,4,5,7).copy()
#    rdm_so[1::2,1::2,::2,::2,::2,1::2,1::2,::2] =  rdm_aabb.transpose(0,1,2,3,4,6,7,5).copy()
#    rdm_so[1::2,::2,1::2,::2,::2,1::2,1::2,::2] = -rdm_aabb.transpose(0,2,1,3,4,6,7,5).copy()
#    rdm_so[1::2,::2,::2,1::2,::2,1::2,1::2,::2] =  rdm_aabb.transpose(0,2,3,1,4,6,7,5).copy()
#    rdm_so[::2,1::2,1::2,::2,::2,1::2,1::2,::2] =  rdm_aabb.transpose(2,0,1,3,4,6,7,5).copy()
#    rdm_so[::2,1::2,::2,1::2,::2,1::2,1::2,::2] = -rdm_aabb.transpose(2,0,3,1,4,6,7,5).copy()
#    rdm_so[::2,::2,1::2,1::2,::2,1::2,1::2,::2] =  rdm_aabb.transpose(2,3,0,1,4,6,7,5).copy()
#    rdm_so[1::2,1::2,::2,::2,1::2,::2,1::2,::2] = -rdm_aabb.transpose(0,1,2,3,6,4,7,5).copy()
#    rdm_so[1::2,::2,1::2,::2,1::2,::2,1::2,::2] =  rdm_aabb.transpose(0,2,1,3,6,4,7,5).copy()
#    rdm_so[1::2,::2,::2,1::2,1::2,::2,1::2,::2] = -rdm_aabb.transpose(0,2,3,1,6,4,7,5).copy()
#    rdm_so[::2,1::2,1::2,::2,1::2,::2,1::2,::2] = -rdm_aabb.transpose(2,0,1,3,6,4,7,5).copy()
#    rdm_so[::2,1::2,::2,1::2,1::2,::2,1::2,::2] =  rdm_aabb.transpose(2,0,3,1,6,4,7,5).copy()
#    rdm_so[::2,::2,1::2,1::2,1::2,::2,1::2,::2] = -rdm_aabb.transpose(2,3,0,1,6,4,7,5).copy()
#    rdm_so[1::2,1::2,::2,::2,1::2,1::2,::2,::2] =  rdm_aabb.transpose(0,1,2,3,6,7,4,5).copy()
#    rdm_so[1::2,::2,1::2,::2,1::2,1::2,::2,::2] = -rdm_aabb.transpose(0,2,1,3,6,7,4,5).copy()
#    rdm_so[1::2,::2,::2,1::2,1::2,1::2,::2,::2] =  rdm_aabb.transpose(0,2,3,1,6,7,4,5).copy()
#    rdm_so[::2,1::2,1::2,::2,1::2,1::2,::2,::2] =  rdm_aabb.transpose(2,0,1,3,6,7,4,5).copy()
#    rdm_so[::2,1::2,::2,1::2,1::2,1::2,::2,::2] = -rdm_aabb.transpose(2,0,3,1,6,7,4,5).copy()
#    rdm_so[::2,::2,1::2,1::2,1::2,1::2,::2,::2] =  rdm_aabb.transpose(2,3,0,1,6,7,4,5).copy()
#    rdm_so[1::2,1::2,1::2,::2,::2,1::2,1::2,1::2] =  rdm_abbb.copy()
#    rdm_so[1::2,1::2,::2,1::2,::2,1::2,1::2,1::2] = -rdm_abbb.transpose(0,1,3,2,4,5,6,7).copy()
#    rdm_so[1::2,::2,1::2,1::2,::2,1::2,1::2,1::2] =  rdm_abbb.transpose(0,3,1,2,4,5,6,7).copy()
#    rdm_so[::2,1::2,1::2,1::2,::2,1::2,1::2,1::2] = -rdm_abbb.transpose(3,0,1,2,4,5,6,7).copy()
#    rdm_so[1::2,1::2,1::2,::2,1::2,::2,1::2,1::2] = -rdm_abbb.transpose(0,1,2,3,5,4,6,7).copy()
#    rdm_so[1::2,1::2,::2,1::2,1::2,::2,1::2,1::2] =  rdm_abbb.transpose(0,1,3,2,5,4,6,7).copy()
#    rdm_so[1::2,::2,1::2,1::2,1::2,::2,1::2,1::2] = -rdm_abbb.transpose(0,3,1,2,5,4,6,7).copy()
#    rdm_so[::2,1::2,1::2,1::2,1::2,::2,1::2,1::2] =  rdm_abbb.transpose(3,0,1,2,5,4,6,7).copy()
#    rdm_so[1::2,1::2,1::2,::2,1::2,1::2,::2,1::2] =  rdm_abbb.transpose(0,1,2,3,5,6,4,7).copy()
#    rdm_so[1::2,1::2,::2,1::2,1::2,1::2,::2,1::2] = -rdm_abbb.transpose(0,1,3,2,5,6,4,7).copy()
#    rdm_so[1::2,::2,1::2,1::2,1::2,1::2,::2,1::2] =  rdm_abbb.transpose(0,3,1,2,5,6,4,7).copy()
#    rdm_so[::2,1::2,1::2,1::2,1::2,1::2,::2,1::2] = -rdm_abbb.transpose(3,0,1,2,5,6,4,7).copy()
#    rdm_so[1::2,1::2,1::2,::2,1::2,1::2,1::2,::2] = -rdm_abbb.transpose(0,1,2,3,5,6,7,4).copy()
#    rdm_so[1::2,1::2,::2,1::2,1::2,1::2,1::2,::2] =  rdm_abbb.transpose(0,1,3,2,5,6,7,4).copy()
#    rdm_so[1::2,::2,1::2,1::2,1::2,1::2,1::2,::2] = -rdm_abbb.transpose(0,3,1,2,5,6,7,4).copy()
#    rdm_so[::2,1::2,1::2,1::2,1::2,1::2,1::2,::2] =  rdm_abbb.transpose(3,0,1,2,5,6,7,4).copy()
#    rdm_so[1::2,1::2,1::2,1::2,1::2,1::2,1::2,1::2] = rdm_bbbb.copy()
#
#    return rdm_so
#
#
#def compute_rdm_c_so(interface, bra, ket, nelecas_bra, nelecas_ket):
#
#    ncas = interface.ncas
#
#    rdm = interface.compute_rdm_c_si(bra, ket, nelecas_bra, nelecas_ket)
#
#    rdm_so = np.zeros((2 * ncas))
#    rdm_so[::2] = rdm[0].copy()
#    rdm_so[1::2] = rdm[1].copy()
#
#    return rdm_so
#
#
## offset = 0: I >= J
## offset = -1: I > J
#def compute_rdm_tcat_so(interface, wfns, nelecasci, offset = 0):
#
#    ncas = interface.ncas
#
#    nroots = len(wfns)
#
#    dim = (nroots + offset) * (nroots + 1 + offset) // 2
#
#    rdm_so = np.zeros((dim, 2 * ncas, 2 * ncas))
#
#    # Loop over unique combination of states
#    for I in range(nroots):
#        for J in range(I + 1 + offset):
#            P = (I + offset) * (I + 1 + offset) // 2 + J
#            rdm_so[P] = compute_rdm_ca_so(interface, wfns[I], wfns[J], nelecasci)
#
#    return rdm_so
#
#
#def compute_rdm_tccaat_so(interface, wfns, nelecasci, offset = 0):
#
#    ncas = interface.ncas
#
#    nroots = len(wfns)
#
#    dim = (nroots + offset) * (nroots + 1 + offset) // 2
#
#    rdm_so = np.zeros((dim, 2 * ncas, 2 * ncas, 2 * ncas, 2 * ncas))
#
#    # Loop over unique combination of states
#    for I in range(nroots):
#        for J in range(I + 1 + offset):
#            P = (I + offset) * (I + 1 + offset) // 2 + J
#            rdm_so[P] = compute_rdm_ccaa_so(interface, wfns[I], wfns[J], nelecasci)
#
#    return rdm_so
#
#
#def compute_rdm_tcccaaat_so(interface, wfns, nelecasci, offset = 0):
#
#    ncas = interface.ncas
#
#    nroots = len(wfns)
#
#    dim = (nroots + offset) * (nroots + 1 + offset) // 2
#
#    rdm_so = np.zeros((dim, 2 * ncas, 2 * ncas, 2 * ncas, 2 * ncas, 2 * ncas, 2 * ncas))
#
#    # Loop over unique combination of states
#    for I in range(nroots):
#        for J in range(I + 1 + offset):
#            P = (I + offset) * (I + 1 + offset) // 2 + J
#            rdm_so[P] = compute_rdm_cccaaa_so(interface, wfns[I], wfns[J], nelecasci)
#
#    return rdm_so
#
#
#def compute_rdm_tccccaaaat_so(interface, wfns, nelecasci, offset = 0):
#
#    ncas = interface.ncas
#
#    nroots = len(wfns)
#
#    dim = (nroots + offset) * (nroots + 1 + offset) // 2
#
#    rdm_so = np.zeros((dim, 2 * ncas, 2 * ncas, 2 * ncas, 2 * ncas, 2 * ncas, 2 * ncas, 2 * ncas, 2 * ncas))
#
#    # Loop over unique combination of states
#    for I in range(nroots):
#        for J in range(I + 1 + offset):
#            P = (I + offset) * (I + 1 + offset) // 2 + J
#            rdm_so[P] = compute_rdm_ccccaaaa_so(interface, wfns[I], wfns[J], nelecasci)
#
#    return rdm_so
#
#def compute_rdm_tcat_general_so_qdnevpt(interface, wfns, nelecasci, offset = 0):
#
#    ncas = interface.ncas
#
#    nroots = len(wfns)
#
#    dim = (nroots + offset) * (nroots + 1 + offset) // 2
#
#    rdm_so = np.zeros((dim, 2 * ncas, 2 * ncas))
#
#    # Loop over unique combination of states
#    for I in range(nroots):
#        for J in range(I + 1 + offset):
#            P = (I + offset) * (I + 1 + offset) // 2 + J
#            rdm_so[P] = compute_rdm_ca_general_so(interface, wfns[I], wfns[J], nelecasci[I], nelecasci[J])
#
#    return rdm_so
#
#def compute_rdm_tccaat_general_so_qdnevpt(interface, wfns, nelecasci, offset = 0):
#
#    ncas = interface.ncas
#
#    nroots = len(wfns)
#
#    dim =  (nroots + offset) * (nroots + 1 + offset) // 2
#
#    rdm_so = np.zeros((dim, 2 * ncas, 2 * ncas, 2 * ncas, 2 * ncas))
#    # Loop over unique combination of states
#    for I in range(nroots):
#        for J in range(I + 1 + offset):
#            P = (I + offset) * (I + 1 + offset) // 2 + J
#            rdm_so[P] = compute_rdm_ccaa_general_so(interface, wfns[I], wfns[J], nelecasci[I], nelecasci[J])
#
#    return rdm_so
#
#def compute_rdm_tcat_so_qdnevpt(interface, wfns, nelecasci, offset = 0):
#
#    ncas = interface.ncas
#
#    nroots = len(wfns)
#
#    dim = (nroots + offset) * (nroots + 1 + offset) // 2
#
#    rdm_so = np.zeros((dim, 2 * ncas, 2 * ncas))
#
#    # Loop over unique combination of states
#    for I in range(nroots):
#        for J in range(I + 1 + offset):
#            P = (I + offset) * (I + 1 + offset) // 2 + J
#            if(nelecasci[I] == nelecasci[J]):
#                rdm_so[P] = compute_rdm_ca_so(interface, wfns[I], wfns[J], nelecasci[I])
#            else:
#                rdm_so[P] = np.zeros((2 * ncas, 2 * ncas))
#    return rdm_so
#
#
#def compute_rdm_tccaat_so_qdnevpt(interface, wfns, nelecasci, offset = 0):
#
#    ncas = interface.ncas
#
#    nroots = len(wfns)
#
#    dim = (nroots + offset) * (nroots + 1 + offset) // 2
#
#    rdm_so = np.zeros((dim, 2 * ncas, 2 * ncas, 2 * ncas, 2 * ncas))
#
#    # Loop over unique combination of states
#    for I in range(nroots):
#        for J in range(I + 1 + offset):
#            P = (I + offset) * (I + 1 + offset) // 2 + J
#            if(nelecasci[I] == nelecasci[J]):
#                rdm_so[P] = compute_rdm_ccaa_so(interface, wfns[I], wfns[J], nelecasci[I])
#            else:
#                rdm_so[P] = np.zeros((2 * ncas, 2 * ncas, 2 * ncas, 2 * ncas))
#    return rdm_so
#
#
#def compute_rdm_tcccaaat_so_qdnevpt(interface, wfns, nelecasci, offset = 0):
#
#    ncas = interface.ncas
#
#    nroots = len(wfns)
#
#    dim = (nroots + offset) * (nroots + 1 + offset) // 2
#
#    rdm_so = np.zeros((dim, 2 * ncas, 2 * ncas, 2 * ncas, 2 * ncas, 2 * ncas, 2 * ncas))
#
#    # Loop over unique combination of states
#    for I in range(nroots):
#        for J in range(I + 1 + offset):
#            P = (I + offset) * (I + 1 + offset) // 2 + J
#            if(nelecasci[I] == nelecasci[J]):
#                rdm_so[P] = compute_rdm_cccaaa_so(interface, wfns[I], wfns[J], nelecasci[I])
#            else:
#                rdm_so[P] = np.zeros((2 * ncas, 2 * ncas, 2 * ncas, 2 * ncas, 2 * ncas, 2 * ncas))
#
#    return rdm_so
#
#
#def compute_rdm_tccccaaaat_so_qdnevpt(interface, wfns, nelecasci, offset = 0):
#
#    ncas = interface.ncas
#
#    nroots = len(wfns)
#
#    dim = (nroots + offset) * (nroots + 1 + offset) // 2
#
#    rdm_so = np.zeros((dim, 2 * ncas, 2 * ncas, 2 * ncas, 2 * ncas, 2 * ncas, 2 * ncas, 2 * ncas, 2 * ncas))
#
#    # Loop over unique combination of states
#    for I in range(nroots):
#        for J in range(I + 1 + offset):
#            P = (I + offset) * (I + 1 + offset) // 2 + J
#            if(nelecasci[I] == nelecasci[J]):
#                rdm_so[P] = compute_rdm_ccccaaaa_so(interface, wfns[I], wfns[J], nelecasci[I])
#            else:
#                rdm_so[P] = np.zeros((2 * ncas, 2 * ncas, 2 * ncas, 2 * ncas, 2 * ncas, 2 * ncas, 2 * ncas, 2 * ncas))
#               
#    return rdm_so
#
#
#def compute_rdm_ct_so(interface, kets, nelecasci, root = None):
#
#    ncas = interface.ncas
#
#    nroots = len(kets)
#
#    wfn_ref = interface.wfn_casscf
#
#    if root is not None:
#        wfn_ref = wfn_ref[root]
#
#    rdm_so = np.zeros((nroots, 2 * ncas))
#
#    for I in range(nroots):
#        rdm_so[I] = compute_rdm_c_so(interface, wfn_ref, kets[I], interface.nelecas, nelecasci)
#
#    return rdm_so
#
#
#def compute_rdm_ccat_so(interface, kets, nelecasci, root = None):
#
#    ncas = interface.ncas
#    nelecas = interface.nelecas
#
#    nroots = len(kets)
#
#    wfn_ref = interface.wfn_casscf
#
#    if root is not None:
#        wfn_ref = wfn_ref[root]
#
#    rdm_so = np.zeros((nroots, 2 * ncas, 2 * ncas, 2 * ncas))
#
#    # A
#    if (nelecas[0] - 1) == nelecasci[0] and nelecas[1] == nelecasci[1]:
#        bras, bra_ne = interface.apply_a(wfn_ref, ncas, nelecas, "a")
#        if bras is not None:
#            for p in range(ncas):
#                for I in range(nroots):
#                    rdm_so[I, 2 * p] = compute_rdm_ca_so(interface, bras[p], kets[I], nelecasci)
#    # B
#    elif nelecas[0] == nelecasci[0] and (nelecas[1] - 1) == nelecasci[1]:
#        bras, bra_ne = interface.apply_a(wfn_ref, ncas, nelecas, "b")
#        if bras is not None:
#            for p in range(ncas):
#                for I in range(nroots):
#                    rdm_so[I, 2 * p + 1] = compute_rdm_ca_so(interface, bras[p], kets[I], nelecasci)
#    else:
#        raise Exception("Number of electrons doesn't match in bra and ket")
#
#    # Add missing spin cases
#    rdm_so[:, 1::2, ::2, 1::2] = -rdm_so[:, ::2, 1::2, 1::2].transpose(0, 2, 1, 3).copy()
#    rdm_so[:, ::2, 1::2, ::2] = -rdm_so[:, 1::2, ::2, ::2].transpose(0, 2, 1, 3).copy()
#
#    return rdm_so
#
#
#def compute_rdm_cccaat_so(interface, kets, nelecasci, root = None):
#
#    ncas = interface.ncas
#    nelecas = interface.nelecas
#    nroots = len(kets)
#
#    wfn_ref = interface.wfn_casscf
#
#    if root is not None:
#        wfn_ref = wfn_ref[root]
#        nelecas = nelecas[root]
#
#    rdm_so = np.zeros((nroots, 2 * ncas, 2 * ncas, 2 * ncas, 2 * ncas, 2 * ncas))
#
#    # A
#    if (nelecas[0] - 1) == nelecasci[0] and nelecas[1] == nelecasci[1]:
#        bras, bra_ne = interface.apply_a(wfn_ref, ncas, nelecas, "a")
#        if bras is not None:
#            for p in range(ncas):
#                for I in range(nroots):
#                    rdm_so[I, 2 * p] = compute_rdm_ccaa_so(interface, bras[p], kets[I], nelecasci)
#    # B
#    elif nelecas[0] == nelecasci[0] and (nelecas[1] - 1) == nelecasci[1]:
#        bras, bra_ne = interface.apply_a(wfn_ref, ncas, nelecas, "b")
#        if bras is not None:
#            for p in range(ncas):
#                for I in range(nroots):
#                    rdm_so[I, 2 * p + 1] = compute_rdm_ccaa_so(interface, bras[p], kets[I], nelecasci)
#    else:
#        raise Exception("Number of electrons doesn't match in bra and ket")
#
#    # Add missing spin cases
#    rdm_so[:, 1::2, ::2, ::2, ::2, 1::2] =   -rdm_so[:, ::2, 1::2, ::2, ::2, 1::2].transpose(0, 2, 1, 3, 4, 5).copy()
#    rdm_so[:, 1::2, ::2, ::2, 1::2, ::2] =   -rdm_so[:, ::2, 1::2, ::2, 1::2, ::2].transpose(0, 2, 1, 3, 4, 5).copy()
#    rdm_so[:, 1::2, ::2, 1::2, 1::2, 1::2] = -rdm_so[:, ::2, 1::2, 1::2, 1::2, 1::2].transpose(0, 2, 1, 3, 4, 5).copy()
#    rdm_so[:, 1::2, 1::2, ::2, 1::2, 1::2] =  rdm_so[:, ::2, 1::2, 1::2, 1::2, 1::2].transpose(0, 2, 3, 1, 4, 5).copy()
#
#    rdm_so[:, ::2, 1::2, 1::2, 1::2, ::2] = -rdm_so[:, 1::2, ::2, 1::2, 1::2, ::2].transpose(0, 2, 1, 3, 4, 5).copy()
#    rdm_so[:, ::2, 1::2, 1::2, ::2, 1::2] = -rdm_so[:, 1::2, ::2, 1::2, ::2, 1::2].transpose(0, 2, 1, 3, 4, 5).copy()
#    rdm_so[:, ::2, 1::2, ::2, ::2, ::2] =   -rdm_so[:, 1::2, ::2, ::2, ::2, ::2].transpose(0, 2, 1, 3, 4, 5).copy()
#    rdm_so[:, ::2, ::2, 1::2, ::2, ::2] =    rdm_so[:, 1::2, ::2, ::2, ::2, ::2].transpose(0, 2, 3, 1, 4, 5).copy()
#
#    return rdm_so
#
#
#def compute_rdm_ccccaaat_so(interface, kets, nelecasci, root = None):
#
#    ncas = interface.ncas
#    nelecas = interface.nelecas
#
#    nroots = 1
#
#    if isinstance(kets, list):
#        nroots = len(kets)
#    else:
#        kets = [kets]
#
#    wfn_ref = interface.wfn_casscf
#
#    if root is not None:
#        wfn_ref = wfn_ref[root]
#
#    rdm_so = np.zeros((nroots, 2 * ncas, 2 * ncas, 2 * ncas, 2 * ncas, 2 * ncas, 2 * ncas, 2 * ncas))
#
#    # A
#    if (nelecas[0] - 1) == nelecasci[0] and nelecas[1] == nelecasci[1]:
#        bras, bra_ne = interface.apply_a(wfn_ref, ncas, nelecas, "a")
#        if bras is not None:
#            for p in range(ncas):
#                for I in range(nroots):
#                    rdm_so[I, 2 * p] = compute_rdm_cccaaa_so(interface, bras[p], kets[I], nelecasci)
#    # B
#    elif nelecas[0] == nelecasci[0] and (nelecas[1] - 1) == nelecasci[1]:
#        bras, bra_ne = interface.apply_a(wfn_ref, ncas, nelecas, "b")
#        if bras is not None:
#            for p in range(ncas):
#                for I in range(nroots):
#                    rdm_so[I, 2 * p + 1] = compute_rdm_cccaaa_so(interface, bras[p], kets[I], nelecasci)
#    else:
#        raise Exception("Number of electrons doesn't match in bra and ket")
#
#    # Add missing spin cases
#    # alpha-alpha part
#    rdm_so[:,1::2,::2,::2,::2,::2,::2,1::2] = -rdm_so[:,::2,1::2,::2,::2,::2,::2,1::2].transpose(0, 2, 1, 3, 4, 5, 6, 7).copy()
#    rdm_so[:,1::2,::2,::2,::2,::2,1::2,::2] = -rdm_so[:,::2,1::2,::2,::2,::2,1::2,::2].transpose(0, 2, 1, 3, 4, 5, 6, 7).copy()
#    rdm_so[:,1::2,::2,::2,::2,1::2,::2,::2] = -rdm_so[:,::2,1::2,::2,::2,1::2,::2,::2].transpose(0, 2, 1, 3, 4, 5, 6, 7).copy()
#
#    rdm_so[:,1::2,::2,1::2,::2,::2,1::2,1::2] = -rdm_so[:,::2,1::2,1::2,::2,::2,1::2,1::2].transpose(0, 2, 1, 3, 4, 5, 6, 7).copy()
#    rdm_so[:,1::2,::2,1::2,::2,1::2,::2,1::2] = -rdm_so[:,::2,1::2,1::2,::2,1::2,::2,1::2].transpose(0, 2, 1, 3, 4, 5, 6, 7).copy()
#    rdm_so[:,1::2,::2,1::2,::2,1::2,1::2,::2] = -rdm_so[:,::2,1::2,1::2,::2,1::2,1::2,::2].transpose(0, 2, 1, 3, 4, 5, 6, 7).copy()
#    rdm_so[:,1::2,::2,::2,1::2,1::2,::2,1::2] = -rdm_so[:,::2,1::2,::2,1::2,1::2,::2,1::2].transpose(0, 2, 1, 3, 4, 5, 6, 7).copy()
#    rdm_so[:,1::2,::2,::2,1::2,::2,1::2,1::2] = -rdm_so[:,::2,1::2,::2,1::2,::2,1::2,1::2].transpose(0, 2, 1, 3, 4, 5, 6, 7).copy()
#    rdm_so[:,1::2,::2,::2,1::2,1::2,1::2,::2] = -rdm_so[:,::2,1::2,::2,1::2,1::2,1::2,::2].transpose(0, 2, 1, 3, 4, 5, 6, 7).copy()
#    rdm_so[:,1::2,1::2,::2,::2,::2,1::2,1::2] =  rdm_so[:,::2,1::2,1::2,::2,::2,1::2,1::2].transpose(0, 2, 3, 1, 4, 5, 6, 7).copy()
#    rdm_so[:,1::2,1::2,::2,::2,1::2,::2,1::2] =  rdm_so[:,::2,1::2,1::2,::2,1::2,::2,1::2].transpose(0, 2, 3, 1, 4, 5, 6, 7).copy()
#    rdm_so[:,1::2,1::2,::2,::2,1::2,1::2,::2] =  rdm_so[:,::2,1::2,1::2,::2,1::2,1::2,::2].transpose(0, 2, 3, 1, 4, 5, 6, 7).copy()
#
#    rdm_so[:,1::2,::2,1::2,1::2,1::2,1::2,1::2] = -rdm_so[:,::2,1::2,1::2,1::2,1::2,1::2,1::2].transpose(0, 2, 1, 3, 4, 5, 6, 7).copy()
#    rdm_so[:,1::2,1::2,::2,1::2,1::2,1::2,1::2] =  rdm_so[:,::2,1::2,1::2,1::2,1::2,1::2,1::2].transpose(0, 2, 3, 1, 4, 5, 6, 7).copy()
#    rdm_so[:,1::2,1::2,1::2,::2,1::2,1::2,1::2] = -rdm_so[:,::2,1::2,1::2,1::2,1::2,1::2,1::2].transpose(0, 2, 3, 4, 1, 5, 6, 7).copy()
#
#    # beta-beta part
#    rdm_so[:,::2,1::2,1::2,1::2,::2,1::2,1::2] = -rdm_so[:,1::2,::2,1::2,1::2,::2,1::2,1::2].transpose(0, 2, 1, 3, 4, 5, 6, 7).copy()
#    rdm_so[:,::2,1::2,1::2,1::2,1::2,::2,1::2] = -rdm_so[:,1::2,::2,1::2,1::2,1::2,::2,1::2].transpose(0, 2, 1, 3, 4, 5, 6, 7).copy()
#    rdm_so[:,::2,1::2,1::2,1::2,1::2,1::2,::2] = -rdm_so[:,1::2,::2,1::2,1::2,1::2,1::2,::2].transpose(0, 2, 1, 3, 4, 5, 6, 7).copy()
#
#    rdm_so[:,::2,1::2,::2,1::2,1::2,::2,::2] = -rdm_so[:,1::2,::2,::2,1::2,1::2,::2,::2].transpose(0, 2, 1, 3, 4, 5, 6, 7).copy()
#    rdm_so[:,::2,1::2,::2,1::2,::2,1::2,::2] = -rdm_so[:,1::2,::2,::2,1::2,::2,1::2,::2].transpose(0, 2, 1, 3, 4, 5, 6, 7).copy()
#    rdm_so[:,::2,1::2,::2,1::2,::2,::2,1::2] = -rdm_so[:,1::2,::2,::2,1::2,::2,::2,1::2].transpose(0, 2, 1, 3, 4, 5, 6, 7).copy()
#    rdm_so[:,::2,1::2,1::2,::2,::2,1::2,::2] = -rdm_so[:,1::2,::2,1::2,::2,::2,1::2,::2].transpose(0, 2, 1, 3, 4, 5, 6, 7).copy()
#    rdm_so[:,::2,1::2,1::2,::2,1::2,::2,::2] = -rdm_so[:,1::2,::2,1::2,::2,1::2,::2,::2].transpose(0, 2, 1, 3, 4, 5, 6, 7).copy()
#    rdm_so[:,::2,1::2,1::2,::2,::2,::2,1::2] = -rdm_so[:,1::2,::2,1::2,::2,::2,::2,1::2].transpose(0, 2, 1, 3, 4, 5, 6, 7).copy()
#    rdm_so[:,::2,::2,1::2,1::2,1::2,::2,::2] =  rdm_so[:,1::2,::2,::2,1::2,1::2,::2,::2].transpose(0, 2, 3, 1, 4, 5, 6, 7).copy()
#    rdm_so[:,::2,::2,1::2,1::2,::2,1::2,::2] =  rdm_so[:,1::2,::2,::2,1::2,::2,1::2,::2].transpose(0, 2, 3, 1, 4, 5, 6, 7).copy()
#    rdm_so[:,::2,::2,1::2,1::2,::2,::2,1::2] =  rdm_so[:,1::2,::2,::2,1::2,::2,::2,1::2].transpose(0, 2, 3, 1, 4, 5, 6, 7).copy()
#
#    rdm_so[:,::2,1::2,::2,::2,::2,::2,::2] = -rdm_so[:,1::2,::2,::2,::2,::2,::2,::2].transpose(0, 2, 1, 3, 4, 5, 6, 7).copy()
#    rdm_so[:,::2,::2,1::2,::2,::2,::2,::2] =  rdm_so[:,1::2,::2,::2,::2,::2,::2,::2].transpose(0, 2, 3, 1, 4, 5, 6, 7).copy()
#    rdm_so[:,::2,::2,::2,1::2,::2,::2,::2] = -rdm_so[:,1::2,::2,::2,::2,::2,::2,::2].transpose(0, 2, 3, 4, 1, 5, 6, 7).copy()
#
#    if nroots == 1:
#        rdm_so = rdm_so[0]
#
#    return rdm_so
#
#
#def compute_rdm_tc_so(interface, kets, nelecasci):
#
#    ncas = interface.ncas
#
#    nroots = len(kets)
#
#    rdm_so = np.zeros((nroots, 2 * ncas))
#
#    for I in range(nroots):
#        rdm_so[I] = compute_rdm_c_so(interface, kets[I], interface.wfn_casscf, nelecasci, interface.nelecas)
#
#    return rdm_so
#
#
#def compute_rdm_tcca_so(interface, bras_target, nelecasci):
#
#    ncas = interface.ncas
#    nelecas = interface.nelecas
#
#    nroots = len(bras_target)
#
#    rdm_so = np.zeros((nroots, 2 * ncas, 2 * ncas, 2 * ncas))
#
#    # A
#    if (nelecasci[0] - 1) == nelecas[0] and nelecasci[1] == nelecas[1]:
#        for I in range(nroots):
#            bras, bra_ne = interface.apply_a(bras_target[I], ncas, nelecasci, "a")
#            if bras is not None:
#                for p in range(ncas):
#                    rdm_so[I, 2 * p] = compute_rdm_ca_so(interface, bras[p], interface.wfn_casscf, nelecas)
#    # B
#    elif nelecasci[0] == nelecas[0] and (nelecasci[1] - 1) == nelecas[1]:
#        for I in range(nroots):
#            bras, bra_ne = interface.apply_a(bras_target[I], ncas, nelecasci, "b")
#            if bras is not None:
#                for p in range(ncas):
#                    rdm_so[I, 2 * p + 1] = compute_rdm_ca_so(interface, bras[p], interface.wfn_casscf, nelecas)
#    else:
#        raise Exception("Number of electrons doesn't match in bra and ket")
#
#    # Add missing spin cases
#    rdm_so[:, 1::2, ::2, 1::2] = -rdm_so[:, ::2, 1::2, 1::2].transpose(0, 2, 1, 3).copy()
#    rdm_so[:, ::2, 1::2, ::2] = -rdm_so[:, 1::2, ::2, ::2].transpose(0, 2, 1, 3).copy()
#
#    return rdm_so
#
#
#def compute_rdm_tcccaa_so(interface, bras_target, nelecasci):
#
#    ncas = interface.ncas
#    nelecas = interface.nelecas
#
#    nroots = len(bras_target)
#
#    rdm_so = np.zeros((nroots, 2 * ncas, 2 * ncas, 2 * ncas, 2 * ncas, 2 * ncas))
#
#    # A
#    if (nelecasci[0] - 1) == nelecas[0] and nelecasci[1] == nelecas[1]:
#        for I in range(nroots):
#            bras, bra_ne = interface.apply_a(bras_target[I], ncas, nelecasci, "a")
#            if bras is not None:
#                for p in range(ncas):
#                    rdm_so[I, 2 * p] = compute_rdm_ccaa_so(interface, bras[p], interface.wfn_casscf, nelecas)
#    # B
#    elif nelecasci[0] == nelecas[0] and (nelecasci[1] - 1) == nelecas[1]:
#        for I in range(nroots):
#            bras, bra_ne = interface.apply_a(bras_target[I], ncas, nelecasci, "b")
#            if bras is not None:
#                for p in range(ncas):
#                    rdm_so[I, 2 * p + 1] = compute_rdm_ccaa_so(interface, bras[p], interface.wfn_casscf, nelecas)
#    else:
#        raise Exception("Number of electrons doesn't match in bra and ket")
#
#    # Add missing spin cases
#    rdm_so[:, 1::2, ::2, ::2, ::2, 1::2] =   -rdm_so[:, ::2, 1::2, ::2, ::2, 1::2].transpose(0, 2, 1, 3, 4, 5).copy()
#    rdm_so[:, 1::2, ::2, ::2, 1::2, ::2] =   -rdm_so[:, ::2, 1::2, ::2, 1::2, ::2].transpose(0, 2, 1, 3, 4, 5).copy()
#    rdm_so[:, 1::2, ::2, 1::2, 1::2, 1::2] = -rdm_so[:, ::2, 1::2, 1::2, 1::2, 1::2].transpose(0, 2, 1, 3, 4, 5).copy()
#    rdm_so[:, 1::2, 1::2, ::2, 1::2, 1::2] =  rdm_so[:, ::2, 1::2, 1::2, 1::2, 1::2].transpose(0, 2, 3, 1, 4, 5).copy()
#
#    rdm_so[:, ::2, 1::2, 1::2, 1::2, ::2] = -rdm_so[:, 1::2, ::2, 1::2, 1::2, ::2].transpose(0, 2, 1, 3, 4, 5).copy()
#    rdm_so[:, ::2, 1::2, 1::2, ::2, 1::2] = -rdm_so[:, 1::2, ::2, 1::2, ::2, 1::2].transpose(0, 2, 1, 3, 4, 5).copy()
#    rdm_so[:, ::2, 1::2, ::2, ::2, ::2] =   -rdm_so[:, 1::2, ::2, ::2, ::2, ::2].transpose(0, 2, 1, 3, 4, 5).copy()
#    rdm_so[:, ::2, ::2, 1::2, ::2, ::2] =    rdm_so[:, 1::2, ::2, ::2, ::2, ::2].transpose(0, 2, 3, 1, 4, 5).copy()
#
#    return rdm_so
#
#
#def compute_rdm_tccccaaa_so(interface, bras_target, nelecasci):
#
#    ncas = interface.ncas
#    nelecas = interface.nelecas
#
#    nroots = 1
#
#    if isinstance(bras_target, list):
#        nroots = len(bras_target)
#    else:
#        bras_target = [bras_target]
#
#    rdm_so = np.zeros((nroots, 2 * ncas, 2 * ncas, 2 * ncas, 2 * ncas, 2 * ncas, 2 * ncas, 2 * ncas))
#
#    # A
#    if (nelecasci[0] - 1) == nelecas[0] and nelecasci[1] == nelecas[1]:
#        for I in range(nroots):
#            bras, bra_ne = interface.apply_a(bras_target[I], ncas, nelecasci, "a")
#            if bras is not None:
#                for p in range(ncas):
#                    rdm_so[I, 2 * p] = compute_rdm_cccaaa_so(interface, bras[p], interface.wfn_casscf, nelecas)
#    # B
#    elif nelecasci[0] == nelecas[0] and (nelecasci[1] - 1) == nelecas[1]:
#        for I in range(nroots):
#            bras, bra_ne = interface.apply_a(bras_target[I], ncas, nelecasci, "b")
#            if bras is not None:
#                for p in range(ncas):
#                    rdm_so[I, 2 * p + 1] = compute_rdm_cccaaa_so(interface, bras[p], interface.wfn_casscf, nelecas)
#    else:
#        raise Exception("Number of electrons doesn't match in bra and ket")
#
#    # Add missing spin cases
#    # alpha-alpha part
#    rdm_so[:,1::2,::2,::2,::2,::2,::2,1::2] = -rdm_so[:,::2,1::2,::2,::2,::2,::2,1::2].transpose(0, 2, 1, 3, 4, 5, 6, 7).copy()
#    rdm_so[:,1::2,::2,::2,::2,::2,1::2,::2] = -rdm_so[:,::2,1::2,::2,::2,::2,1::2,::2].transpose(0, 2, 1, 3, 4, 5, 6, 7).copy()
#    rdm_so[:,1::2,::2,::2,::2,1::2,::2,::2] = -rdm_so[:,::2,1::2,::2,::2,1::2,::2,::2].transpose(0, 2, 1, 3, 4, 5, 6, 7).copy()
#
#    rdm_so[:,1::2,::2,1::2,::2,::2,1::2,1::2] = -rdm_so[:,::2,1::2,1::2,::2,::2,1::2,1::2].transpose(0, 2, 1, 3, 4, 5, 6, 7).copy()
#    rdm_so[:,1::2,::2,1::2,::2,1::2,::2,1::2] = -rdm_so[:,::2,1::2,1::2,::2,1::2,::2,1::2].transpose(0, 2, 1, 3, 4, 5, 6, 7).copy()
#    rdm_so[:,1::2,::2,1::2,::2,1::2,1::2,::2] = -rdm_so[:,::2,1::2,1::2,::2,1::2,1::2,::2].transpose(0, 2, 1, 3, 4, 5, 6, 7).copy()
#    rdm_so[:,1::2,::2,::2,1::2,1::2,::2,1::2] = -rdm_so[:,::2,1::2,::2,1::2,1::2,::2,1::2].transpose(0, 2, 1, 3, 4, 5, 6, 7).copy()
#    rdm_so[:,1::2,::2,::2,1::2,::2,1::2,1::2] = -rdm_so[:,::2,1::2,::2,1::2,::2,1::2,1::2].transpose(0, 2, 1, 3, 4, 5, 6, 7).copy()
#    rdm_so[:,1::2,::2,::2,1::2,1::2,1::2,::2] = -rdm_so[:,::2,1::2,::2,1::2,1::2,1::2,::2].transpose(0, 2, 1, 3, 4, 5, 6, 7).copy()
#    rdm_so[:,1::2,1::2,::2,::2,::2,1::2,1::2] =  rdm_so[:,::2,1::2,1::2,::2,::2,1::2,1::2].transpose(0, 2, 3, 1, 4, 5, 6, 7).copy()
#    rdm_so[:,1::2,1::2,::2,::2,1::2,::2,1::2] =  rdm_so[:,::2,1::2,1::2,::2,1::2,::2,1::2].transpose(0, 2, 3, 1, 4, 5, 6, 7).copy()
#    rdm_so[:,1::2,1::2,::2,::2,1::2,1::2,::2] =  rdm_so[:,::2,1::2,1::2,::2,1::2,1::2,::2].transpose(0, 2, 3, 1, 4, 5, 6, 7).copy()
#
#    rdm_so[:,1::2,::2,1::2,1::2,1::2,1::2,1::2] = -rdm_so[:,::2,1::2,1::2,1::2,1::2,1::2,1::2].transpose(0, 2, 1, 3, 4, 5, 6, 7).copy()
#    rdm_so[:,1::2,1::2,::2,1::2,1::2,1::2,1::2] =  rdm_so[:,::2,1::2,1::2,1::2,1::2,1::2,1::2].transpose(0, 2, 3, 1, 4, 5, 6, 7).copy()
#    rdm_so[:,1::2,1::2,1::2,::2,1::2,1::2,1::2] = -rdm_so[:,::2,1::2,1::2,1::2,1::2,1::2,1::2].transpose(0, 2, 3, 4, 1, 5, 6, 7).copy()
#
#    # beta-beta part
#    rdm_so[:,::2,1::2,1::2,1::2,::2,1::2,1::2] = -rdm_so[:,1::2,::2,1::2,1::2,::2,1::2,1::2].transpose(0, 2, 1, 3, 4, 5, 6, 7).copy()
#    rdm_so[:,::2,1::2,1::2,1::2,1::2,::2,1::2] = -rdm_so[:,1::2,::2,1::2,1::2,1::2,::2,1::2].transpose(0, 2, 1, 3, 4, 5, 6, 7).copy()
#    rdm_so[:,::2,1::2,1::2,1::2,1::2,1::2,::2] = -rdm_so[:,1::2,::2,1::2,1::2,1::2,1::2,::2].transpose(0, 2, 1, 3, 4, 5, 6, 7).copy()
#
#    rdm_so[:,::2,1::2,::2,1::2,1::2,::2,::2] = -rdm_so[:,1::2,::2,::2,1::2,1::2,::2,::2].transpose(0, 2, 1, 3, 4, 5, 6, 7).copy()
#    rdm_so[:,::2,1::2,::2,1::2,::2,1::2,::2] = -rdm_so[:,1::2,::2,::2,1::2,::2,1::2,::2].transpose(0, 2, 1, 3, 4, 5, 6, 7).copy()
#    rdm_so[:,::2,1::2,::2,1::2,::2,::2,1::2] = -rdm_so[:,1::2,::2,::2,1::2,::2,::2,1::2].transpose(0, 2, 1, 3, 4, 5, 6, 7).copy()
#    rdm_so[:,::2,1::2,1::2,::2,::2,1::2,::2] = -rdm_so[:,1::2,::2,1::2,::2,::2,1::2,::2].transpose(0, 2, 1, 3, 4, 5, 6, 7).copy()
#    rdm_so[:,::2,1::2,1::2,::2,1::2,::2,::2] = -rdm_so[:,1::2,::2,1::2,::2,1::2,::2,::2].transpose(0, 2, 1, 3, 4, 5, 6, 7).copy()
#    rdm_so[:,::2,1::2,1::2,::2,::2,::2,1::2] = -rdm_so[:,1::2,::2,1::2,::2,::2,::2,1::2].transpose(0, 2, 1, 3, 4, 5, 6, 7).copy()
#    rdm_so[:,::2,::2,1::2,1::2,1::2,::2,::2] =  rdm_so[:,1::2,::2,::2,1::2,1::2,::2,::2].transpose(0, 2, 3, 1, 4, 5, 6, 7).copy()
#    rdm_so[:,::2,::2,1::2,1::2,::2,1::2,::2] =  rdm_so[:,1::2,::2,::2,1::2,::2,1::2,::2].transpose(0, 2, 3, 1, 4, 5, 6, 7).copy()
#    rdm_so[:,::2,::2,1::2,1::2,::2,::2,1::2] =  rdm_so[:,1::2,::2,::2,1::2,::2,::2,1::2].transpose(0, 2, 3, 1, 4, 5, 6, 7).copy()
#
#    rdm_so[:,::2,1::2,::2,::2,::2,::2,::2] = -rdm_so[:,1::2,::2,::2,::2,::2,::2,::2].transpose(0, 2, 1, 3, 4, 5, 6, 7).copy()
#    rdm_so[:,::2,::2,1::2,::2,::2,::2,::2] =  rdm_so[:,1::2,::2,::2,::2,::2,::2,::2].transpose(0, 2, 3, 1, 4, 5, 6, 7).copy()
#    rdm_so[:,::2,::2,::2,1::2,::2,::2,::2] = -rdm_so[:,1::2,::2,::2,::2,::2,::2,::2].transpose(0, 2, 3, 4, 1, 5, 6, 7).copy()
#
#    if nroots == 1:
#        rdm_so = rdm_so[0]
#
#    return rdm_so
#
#
#def compute_rdm_tca_so(interface, wfns, nelecasci):
#
#    ncas = interface.ncas
#
#    nroots = len(wfns)
#
#    dim = nroots
#
#    rdm_so = np.zeros((dim, 2 * ncas, 2 * ncas))
#
#    # Loop over unique combination of states
#    for I in range(nroots):
#        rdm_so[I] = compute_rdm_ca_so(interface, wfns[I], interface.wfn_casscf, nelecasci)
#
#    return rdm_so
#
#
#def compute_rdm_tccaa_so(interface, wfns, nelecasci):
#
#    ncas = interface.ncas
#
#    nroots = len(wfns)
#
#    dim = nroots
#
#    rdm_so = np.zeros((dim, 2 * ncas, 2 * ncas, 2 * ncas, 2 * ncas))
#
#    # Loop over unique combination of states
#    for I in range(nroots):
#        rdm_so[I] = compute_rdm_ccaa_so(interface, wfns[I], interface.wfn_casscf, nelecasci)
#
#    return rdm_so
#
#
#def compute_rdm_tcccaaa_so(interface, wfns, nelecasci):
#
#    ncas = interface.ncas
#
#    nroots = len(wfns)
#
#    dim = nroots
#
#    rdm_so = np.zeros((dim, 2 * ncas, 2 * ncas, 2 * ncas, 2 * ncas, 2 * ncas, 2 * ncas))
#
#    # Loop over unique combination of states
#    for I in range(nroots):
#        rdm_so[I] = compute_rdm_cccaaa_so(interface, wfns[I], interface.wfn_casscf, nelecasci)
#
#    return rdm_so
#
#
#def compute_rdm_tccccaaaa_so(interface, wfns, nelecasci):
#
#    ncas = interface.ncas
#
#    nroots = len(wfns)
#
#    dim = nroots
#
#    rdm_so = np.zeros((dim, 2 * ncas, 2 * ncas, 2 * ncas, 2 * ncas, 2 * ncas, 2 * ncas, 2 * ncas, 2 * ncas))
#
#    # Loop over unique combination of states
#    for I in range(nroots):
#        rdm_so[I] = compute_rdm_ccccaaaa_so(interface, wfns[I], interface.wfn_casscf, nelecasci)
#
#    return rdm_so
#
#
#def compute_rdm_cat_so(interface, wfns, nelecasci):
#
#    ncas = interface.ncas
#
#    nroots = len(wfns)
#
#    dim = nroots
#
#    rdm_so = np.zeros((dim, 2 * ncas, 2 * ncas))
#
#    # Loop over unique combination of states
#    for I in range(nroots):
#        rdm_so[I] = compute_rdm_ca_so(interface, interface.wfn_casscf, wfns[I], nelecasci)
#
#    return rdm_so
#
#
#def compute_rdm_ccaat_so(interface, wfns, nelecasci):
#
#    ncas = interface.ncas
#
#    nroots = len(wfns)
#
#    dim = nroots
#
#    rdm_so = np.zeros((dim, 2 * ncas, 2 * ncas, 2 * ncas, 2 * ncas))
#
#    # Loop over unique combination of states
#    for I in range(nroots):
#        rdm_so[I] = compute_rdm_ccaa_so(interface, interface.wfn_casscf, wfns[I], nelecasci)
#
#    return rdm_so
#
#
#def compute_rdm_cccaaat_so(interface, wfns, nelecasci):
#
#    ncas = interface.ncas
#
#    nroots = len(wfns)
#
#    dim = nroots
#
#    rdm_so = np.zeros((dim, 2 * ncas, 2 * ncas, 2 * ncas, 2 * ncas, 2 * ncas, 2 * ncas))
#
#    # Loop over unique combination of states
#    for I in range(nroots):
#        rdm_so[I] = compute_rdm_cccaaa_so(interface, interface.wfn_casscf, wfns[I], nelecasci)
#
#    return rdm_so
#
#
#def compute_rdm_ccccaaaat_so(interface, wfns, nelecasci):
#
#    ncas = interface.ncas
#
#    nroots = len(wfns)
#
#    dim = nroots
#
#    rdm_so = np.zeros((dim, 2 * ncas, 2 * ncas, 2 * ncas, 2 * ncas, 2 * ncas, 2 * ncas, 2 * ncas, 2 * ncas))
#
#    # Loop over unique combination of states
#    for I in range(nroots):
#        rdm_so[I] = compute_rdm_ccccaaaa_so(interface, interface.wfn_casscf, wfns[I], nelecasci)
#
#    return rdm_so
#
#
#####
##### Functions below are mainly used for debugging
#####
#
## Helper function for debugging
#def test_rdm_cccaaa_so(interface, bra = None, ket = None, nelecas = None):
#
#    ncas = interface.ncas
#
#    if bra is None:
#        bra = interface.wfn_casscf
#    if ket is None:
#        ket = interface.wfn_casscf
#    if nelecas is None:
#        nelecas = interface.nelecas
#
#    rdm_so = np.zeros((2 * ncas, 2 * ncas, 2 * ncas, 2 * ncas, 2 * ncas, 2 * ncas))
#
#    # AAAAAA
#    wfns, wfn_ne = interface.apply_aaa(ket, ncas, nelecas, "aaa")
#    if wfns is not None:
#        wfns = wfns.reshape(ncas**3, -1)
#        wfns_I, wfn_I = interface.apply_aaa(bra, ncas, nelecas, "aaa")
#        wfns_I = wfns_I.reshape(ncas**3, -1)
#        rdm = np.dot(wfns_I, wfns.T).reshape((ncas, ncas, ncas, ncas, ncas, ncas)).transpose(2,1,0,3,4,5).copy()
#        rdm_so[::2,::2,::2,::2,::2,::2] = rdm.copy()
#
#    # BAAAAB
#    wfn_str = ["aab", "aba", "baa"]
#    for wfn_str_bra in wfn_str:
#        for wfn_str_ket in wfn_str:
#            wfns, wfn_ne = interface.apply_aaa(ket, ncas, nelecas, wfn_str_ket)
#            wfns_I, wfn_I = interface.apply_aaa(bra, ncas, nelecas, wfn_str_bra)
#            if wfns is None and wfns_I is not None:
#                print (wfn_str_ket, wfn_str_bra, wfns_I.shape)
#            if wfns_I is None and wfns is not None:
#                print (wfn_str_bra, wfn_str_ket, wfns.shape)
#            if wfns is not None and wfns_I is not None:
#                wfns = wfns.reshape(ncas**3, -1)
#                wfns_I = wfns_I.reshape(ncas**3, -1)
#                rdm = np.dot(wfns_I, wfns.T).reshape((ncas, ncas, ncas, ncas, ncas, ncas)).transpose(2,1,0,3,4,5).copy()
#                p = 0 if wfn_str_ket[0] == 'a' else 1
#                q = 0 if wfn_str_ket[1] == 'a' else 1
#                r = 0 if wfn_str_ket[2] == 'a' else 1
#                pp = 0 if wfn_str_bra[0] == 'a' else 1
#                qq = 0 if wfn_str_bra[1] == 'a' else 1
#                rr = 0 if wfn_str_bra[2] == 'a' else 1
#                rdm_so[rr::2,qq::2,pp::2,p::2,q::2,r::2] = rdm.copy()
#
#    # BBAABB
#    wfn_str = ["abb", "bab", "bba"]
#    for wfn_str_bra in wfn_str:
#        for wfn_str_ket in wfn_str:
#            wfns, wfn_ne = interface.apply_aaa(ket, ncas, nelecas, wfn_str_ket)
#            wfns_I, wfn_I = interface.apply_aaa(bra, ncas, nelecas, wfn_str_bra)
#            if wfns is None and wfns_I is not None:
#                print (wfn_str_ket, wfn_str_bra, wfns_I.shape)
#            if wfns_I is None and wfns is not None:
#                print (wfn_str_bra, wfn_str_ket, wfns.shape)
#            if wfns is not None and wfns_I is not None:
#                wfns = wfns.reshape(ncas**3, -1)
#                wfns_I = wfns_I.reshape(ncas**3, -1)
#                rdm = np.dot(wfns_I, wfns.T).reshape((ncas, ncas, ncas, ncas, ncas, ncas)).transpose(2,1,0,3,4,5).copy()
#                p = 0 if wfn_str_ket[0] == 'a' else 1
#                q = 0 if wfn_str_ket[1] == 'a' else 1
#                r = 0 if wfn_str_ket[2] == 'a' else 1
#                pp = 0 if wfn_str_bra[0] == 'a' else 1
#                qq = 0 if wfn_str_bra[1] == 'a' else 1
#                rr = 0 if wfn_str_bra[2] == 'a' else 1
#                rdm_so[rr::2,qq::2,pp::2,p::2,q::2,r::2] = rdm.copy()
#
#    # BBBBBB
#    wfns, wfn_ne = interface.apply_aaa(ket, ncas, nelecas, "bbb")
#    if wfns is not None:
#        wfns = wfns.reshape(ncas**3, -1)
#        wfns_I, wfn_I = interface.apply_aaa(bra, ncas, nelecas, "bbb")
#        wfns_I = wfns_I.reshape(ncas**3, -1)
#        rdm = np.dot(wfns_I, wfns.T).reshape((ncas, ncas, ncas, ncas, ncas, ncas)).transpose(2,1,0,3,4,5).copy()
#        rdm_so[1::2,1::2,1::2,1::2,1::2,1::2] = rdm.copy()
#
#    return rdm_so
#
#
## Helper function for debugging
#def test_rdm_ccccaaaa_so(interface, bra = None, ket = None, nelecas = None):
#
#    ncas = interface.ncas
#
#    if bra is None:
#        bra = interface.wfn_casscf
#    if ket is None:
#        ket = interface.wfn_casscf
#    if nelecas is None:
#        nelecas = interface.nelecas
#
#    rdm_so = np.zeros((2 * ncas, 2 * ncas, 2 * ncas, 2 * ncas, 2 * ncas, 2 * ncas, 2 * ncas, 2 * ncas))
#
#    # AAAAAAAA
#    wfns, wfn_ne = interface.apply_aaaa(ket, ncas, nelecas, "aaaa")
#    if wfns is not None:
#        wfns = wfns.reshape(ncas**4, -1)
#        wfns_I, wfn_I = interface.apply_aaaa(bra, ncas, nelecas, "aaaa")
#        wfns_I = wfns_I.reshape(ncas**4, -1)
#        rdm = np.dot(wfns_I, wfns.T).reshape((ncas, ncas, ncas, ncas, ncas, ncas, ncas, ncas)).transpose(3,2,1,0,4,5,6,7).copy()
#        rdm_so[::2,::2,::2,::2,::2,::2,::2,::2] = rdm.copy()
#
#    # BAAAAAAB
#    wfn_str = ["aaab", "aaba", "abaa", "baaa"]
#    for wfn_str_bra in wfn_str:
#        for wfn_str_ket in wfn_str:
#            wfns, wfn_ne = interface.apply_aaaa(ket, ncas, nelecas, wfn_str_ket)
#            wfns_I, wfn_I = interface.apply_aaaa(bra, ncas, nelecas, wfn_str_bra)
#            if wfns is None and wfns_I is not None:
#                print (wfn_str_ket, wfn_str_bra, wfns_I.shape)
#            if wfns_I is None and wfns is not None:
#                print (wfn_str_bra, wfn_str_ket, wfns.shape)
#            if wfns is not None and wfns_I is not None:
#                wfns = wfns.reshape(ncas**4, -1)
#                wfns_I = wfns_I.reshape(ncas**4, -1)
#                rdm = np.dot(wfns_I, wfns.T).reshape((ncas, ncas, ncas, ncas, ncas, ncas, ncas, ncas)).transpose(3,2,1,0,4,5,6,7).copy()
#                p = 0 if wfn_str_ket[0] == 'a' else 1
#                q = 0 if wfn_str_ket[1] == 'a' else 1
#                r = 0 if wfn_str_ket[2] == 'a' else 1
#                s = 0 if wfn_str_ket[3] == 'a' else 1
#                pp = 0 if wfn_str_bra[0] == 'a' else 1
#                qq = 0 if wfn_str_bra[1] == 'a' else 1
#                rr = 0 if wfn_str_bra[2] == 'a' else 1
#                ss = 0 if wfn_str_bra[3] == 'a' else 1
#                rdm_so[ss::2,rr::2,qq::2,pp::2,p::2,q::2,r::2,s::2] = rdm.copy()
#
#    # BBAAAABB
#    wfn_str = ["aabb", "abab", "baab", "abba", "baba", "bbaa"]
#    for wfn_str_bra in wfn_str:
#        for wfn_str_ket in wfn_str:
#            wfns, wfn_ne = interface.apply_aaaa(ket, ncas, nelecas, wfn_str_ket)
#            wfns_I, wfn_I = interface.apply_aaaa(bra, ncas, nelecas, wfn_str_bra)
#            if wfns is not None and wfns_I is not None:
#                wfns = wfns.reshape(ncas**4, -1)
#                wfns_I = wfns_I.reshape(ncas**4, -1)
#                rdm = np.dot(wfns_I, wfns.T).reshape((ncas, ncas, ncas, ncas, ncas, ncas, ncas, ncas)).transpose(3,2,1,0,4,5,6,7).copy()
#                p = 0 if wfn_str_ket[0] == 'a' else 1
#                q = 0 if wfn_str_ket[1] == 'a' else 1
#                r = 0 if wfn_str_ket[2] == 'a' else 1
#                s = 0 if wfn_str_ket[3] == 'a' else 1
#                pp = 0 if wfn_str_bra[0] == 'a' else 1
#                qq = 0 if wfn_str_bra[1] == 'a' else 1
#                rr = 0 if wfn_str_bra[2] == 'a' else 1
#                ss = 0 if wfn_str_bra[3] == 'a' else 1
#                rdm_so[ss::2,rr::2,qq::2,pp::2,p::2,q::2,r::2,s::2] = rdm.copy()
#
#    # BBBAABBB
#    wfn_str = ["abbb", "babb", "bbab", "bbba"]
#    for wfn_str_bra in wfn_str:
#        for wfn_str_ket in wfn_str:
#            wfns, wfn_ne = interface.apply_aaaa(ket, ncas, nelecas, wfn_str_ket)
#            wfns_I, wfn_I = interface.apply_aaaa(bra, ncas, nelecas, wfn_str_bra)
#            if wfns is not None and wfns_I is not None:
#                wfns = wfns.reshape(ncas**4, -1)
#                wfns_I = wfns_I.reshape(ncas**4, -1)
#                rdm = np.dot(wfns_I, wfns.T).reshape((ncas, ncas, ncas, ncas, ncas, ncas, ncas, ncas)).transpose(3,2,1,0,4,5,6,7).copy()
#                p = 0 if wfn_str_ket[0] == 'a' else 1
#                q = 0 if wfn_str_ket[1] == 'a' else 1
#                r = 0 if wfn_str_ket[2] == 'a' else 1
#                s = 0 if wfn_str_ket[3] == 'a' else 1
#                pp = 0 if wfn_str_bra[0] == 'a' else 1
#                qq = 0 if wfn_str_bra[1] == 'a' else 1
#                rr = 0 if wfn_str_bra[2] == 'a' else 1
#                ss = 0 if wfn_str_bra[3] == 'a' else 1
#                rdm_so[ss::2,rr::2,qq::2,pp::2,p::2,q::2,r::2,s::2] = rdm.copy()
#
#    # BBBBBBBB
#    wfns, wfn_ne = interface.apply_aaaa(ket, ncas, nelecas, "bbbb")
#    if wfns is not None:
#        wfns = wfns.reshape(ncas**4, -1)
#        wfns_I, wfn_I = interface.apply_aaaa(bra, ncas, nelecas, "bbbb")
#        wfns_I = wfns_I.reshape(ncas**4, -1)
#        rdm = np.dot(wfns_I, wfns.T).reshape((ncas, ncas, ncas, ncas, ncas, ncas, ncas, ncas)).transpose(3,2,1,0,4,5,6,7).copy()
#        rdm_so[1::2,1::2,1::2,1::2,1::2,1::2,1::2,1::2] = rdm.copy()
#
#    return rdm_so
#
#
