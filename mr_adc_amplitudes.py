import sys
import time
import numpy as np
from functools import reduce
import prism.mr_adc_intermediates as mr_adc_intermediates
import prism.mr_adc_overlap as mr_adc_overlap

def compute_amplitudes(mr_adc):

    start_time = time.time()

    # First-order amplitudes
    # TODO: Implement Spin-Adapted Semi-internals
    t1_amp = compute_t1_amplitudes(mr_adc)

    # Second-order amplitudes
    # TODO: Implement T2 amplitudes
    t2_amp = compute_t2_amplitudes(mr_adc, t1_amp)

    t1_ce, t1_ca, t1_ae, t1_caea, t1_caae, t1_caaa, t1_aaea, t1_aaae, t1_ccee, t1_ccea, t1_caee, t1_ccaa, t1_aaee = t1_amp
    t2_ce, t2_ca, t2_ae, t2_caea, t2_caaa, t2_aaea, t2_ccee, t2_ccea, t2_caee, t2_ccaa, t2_aaee, t2_aa = t2_amp

    print ("Time for computing amplitudes:                    %f sec\n" % (time.time() - start_time))

    return (t1_ce, t1_ca, t1_ae, t1_caea, t1_caae, t1_caaa, t1_aaea, t1_ccee, t1_ccea, t1_caee, t1_ccaa, t1_aaee,
            t2_ce, t2_ca, t2_ae, t2_caea, t2_caea, t2_caaa, t2_aaea, t2_ccee, t2_ccea, t2_caee, t2_ccaa, t2_aaee, t2_aa)

def compute_t1_amplitudes(mr_adc):

    t1_ce, t1_ca, t1_ae = (None,) * 3
    t1_caea, t1_caaa, t1_aaea, t1_aaae = (None,) * 4
    t1_ccee, t1_ccea, t1_caee, t1_ccaa, t1_aaee = (None,) * 5

    e_0p, e_p1p, e_m1p, e_0, e_p1, e_m1, e_p2, e_m2 = (0.0,) * 8

    ncore = mr_adc.ncore
    ncas = mr_adc.ncas
    nextern = mr_adc.nextern

    ##########################
    # First-order amplitudes #
    ##########################
    if mr_adc.method in ("mr-adc(1)", "mr-adc(2)", "mr-adc(2)-x"):
        if mr_adc.ncore > 0 and mr_adc.nextern > 0 and mr_adc.ncas > 0:
            print ("Computing T[0']^(1) amplitudes...")
            sys.stdout.flush()
            # TODO: Implementation of spin-adapted amplited t1_0p
            e_0p, t1_ce, t1_caea, t1_caae = compute_t1_0p_sanity_check(mr_adc)
            print ("Norm of T[0']^(1):                          %20.12f" % (np.linalg.norm(t1_ce) + np.linalg.norm(t1_caea)))
            print ("Correlation energy [0']:                    %20.12f\n" % e_0p)
        else:
            t1_ce = np.zeros((ncore, nextern))
            t1_caea = np.zeros((ncore, ncas, nextern, ncas))

        if mr_adc.ncore > 0 and mr_adc.ncas > 0:
            print ("Computing T[+1']^(1) amplitudes...")
            sys.stdout.flush()
            # TODO: Implementation of spin-adapted amplited t1_p1p
            e_p1p, t1_ca, t1_caaa = compute_t1_p1p_sanity_check(mr_adc)
            print ("Norm of T[+1']^(1):                         %20.12f" % (np.linalg.norm(t1_ca) + np.linalg.norm(t1_caaa)))
            print ("Correlation energy [+1']:                   %20.12f\n" % e_p1p)
        else:
            t1_ca = np.zeros((ncore, ncas))
            t1_caaa = np.zeros((ncore, ncas, ncas, ncas))

        if mr_adc.nextern > 0 and mr_adc.ncas > 0:
            print ("Computing T[-1']^(1) amplitudes...")
            sys.stdout.flush()
            # TODO: Implementation of spin-adapted amplited t1_m1p
            e_m1p, t1_ae, t1_aaea, t1_aaae = compute_t1_m1p_sanity_check(mr_adc)
            print ("Norm of T[-1']^(1):                         %20.12f" % (np.linalg.norm(t1_ae) + np.linalg.norm(t1_aaea)))
            print ("Correlation energy [-1']:                   %20.12f\n" % e_m1p)
        else:
            t1_ae = np.zeros((ncas, nextern))
            t1_aaea = np.zeros((ncas, ncas, nextern, ncas))
            t1_aaae = np.zeros((ncas, ncas, ncas, nextern))

    if mr_adc.method in ("mr-adc(2)", "mr-adc(2)-x") or (mr_adc.method == "mr-adc(1)" and mr_adc.method_type in ("ee", "cvs-ee")):
        if mr_adc.ncore > 0 and mr_adc.nextern > 0:
            print ("Computing T[0]^(1) amplitudes...")
            sys.stdout.flush()
            e_0, t1_ccee = compute_t1_0(mr_adc)
            print ("Norm of T[0]^(1):                           %20.12f" % np.linalg.norm(t1_ccee))
            print ("Correlation energy [0]:                     %20.12f\n" % e_0)
        else:
            t1_ccee = np.zeros((ncore, ncore, nextern, nextern))

        if mr_adc.ncore > 0 and mr_adc.nextern > 0 and mr_adc.ncas > 0:
            print ("Computing T[+1]^(1) amplitudes...")
            sys.stdout.flush()
            e_p1, t1_ccea = compute_t1_p1(mr_adc)
            print ("Norm of T[+1]^(1):                          %20.12f" % np.linalg.norm(t1_ccea))
            print ("Correlation energy [+1]:                    %20.12f\n" % e_p1)

            print ("Computing T[-1]^(1) amplitudes...")
            sys.stdout.flush()
            e_m1, t1_caee = compute_t1_m1(mr_adc)
            print ("Norm of T[-1]^(1):                          %20.12f" % np.linalg.norm(t1_caee))
            print ("Correlation energy [-1]:                    %20.12f\n" % e_m1)
        else:
            t1_ccea = np.zeros((ncore, ncore, nextern, ncas))
            t1_caee = np.zeros((ncore, ncas, nextern, nextern))

        if mr_adc.ncore > 0 and mr_adc.ncas > 0:
            print ("Computing T[+2]^(1) amplitudes...")
            sys.stdout.flush()
            e_p2, t1_ccaa = compute_t1_p2(mr_adc)
            print ("Norm of T[+2]^(1):                          %20.12f" % np.linalg.norm(t1_ccaa))
            print ("Correlation energy [+2]:                    %20.12f\n" % e_p2)
        else:
            t1_ccaa = np.zeros((ncore, ncore, ncas, ncas))

        if mr_adc.nextern > 0 and mr_adc.ncas > 0:
            print ("Computing T[-2]^(1) amplitudes...")
            sys.stdout.flush()
            e_m2, t1_aaee = compute_t1_m2(mr_adc)
            print ("Norm of T[-2]^(1):                          %20.12f" % np.linalg.norm(t1_aaee))
            print ("Correlation energy [-2]:                    %20.12f\n" % e_m2)
        else:
            t1_aaee = np.zeros((ncas, ncas, nextern, nextern))

    e_corr = e_0p + e_p1p + e_m1p + e_0 + e_p1 + e_m1 + e_p2 + e_m2
    e_tot = mr_adc.e_casscf + e_corr

    # if mr_adc.debug_mode:
    #     print (">>> SA e_0p:  {:}".format(e_0p))
    #     print (">>> SA e_p1p: {:}".format(e_p1p))
    #     print (">>> SA e_m1p: {:}".format(e_m1p))
    #     print (">>> SA e_0:   {:}".format(e_0))
    #     print (">>> SA e_p1:  {:}".format(e_p1))
    #     print (">>> SA e_m1:  {:}".format(e_m1))
    #     print (">>> SA e_p2:  {:}".format(e_p2))
    #     print (">>> SA e_m2:  {:}".format(e_m2))
    #     print (">>> SA e_corr: {:}".format(e_corr))

    print ("CASSCF reference energy:                     %20.12f" % mr_adc.e_casscf)
    print ("PC-NEVPT2 correlation energy:                %20.12f" % e_corr)
    print ("Total PC-NEVPT2 energy:                      %20.12f\n" % e_tot)

    t1_amp = (t1_ce, t1_ca, t1_ae, t1_caea, t1_caae, t1_caaa, t1_aaea, t1_aaae, t1_ccee, t1_ccea, t1_caee, t1_ccaa, t1_aaee)

    return t1_amp

def compute_t2_amplitudes(mr_adc, t1_amp):

    t2_ce, t2_ca, t2_ae = (None,) * 3
    t2_caea, t2_caaa, t2_aaea = (None,) * 3
    t2_ccee, t2_ccea, t2_caee, t2_ccaa, t2_aaee = (None,) * 5
    t2_aa = None

    ncore = mr_adc.ncore
    ncas = mr_adc.ncas
    nextern = mr_adc.nextern

    ###########################
    # Second-order amplitudes #
    ###########################
    # Approximate second-order amplitudes
    if mr_adc.method in ("mr-adc(2)", "mr-adc(2)-x"):

        if ncore > 0 and nextern > 0:
            print ("Computing T[0']^(2) amplitudes...")
            sys.stdout.flush()
            # t2_ce = compute_t2_0p_singles(mr_adc, t1_amp)
            # print ("Norm of T[0']^(2):                         %20.12f\n" % np.linalg.norm(t2_ce))
            # sys.stdout.flush()

            #DEBUG t1_so
            t1_ce_so = np.load("t1_ce_so.npy")
            t1_ca_so = np.load("t1_ca_so.npy")
            t1_ae_so = np.load("t1_ae_so.npy")
            t1_caea_so = np.load("t1_caea_so.npy")
            t1_caaa_so = np.load("t1_caaa_so.npy")
            t1_aaea_so = np.load("t1_aaea_so.npy")
            t1_ccee_so = np.load("t1_ccee_so.npy")
            t1_ccea_so = np.load("t1_ccea_so.npy")
            t1_caee_so = np.load("t1_caee_so.npy")
            t1_ccaa_so = np.load("t1_ccaa_so.npy")
            t1_aaee_so = np.load("t1_aaee_so.npy")

            t1_ce_so = t1_ce_so[::2,::2]
            t1_ca_so = t1_ca_so[::2,::2]
            t1_ae_so = t1_ae_so[::2,::2]

            t1_caae_so = - t1_caea_so[::2,1::2,1::2,::2].transpose(0,1,3,2)
            t1_caea_so = t1_caea_so[::2,1::2,::2,1::2]

            t1_caaa_so = t1_caaa_so[::2,1::2,::2,1::2]

            t1_aaae_so = - t1_aaea_so[::2,1::2,1::2,::2].transpose(0,1,3,2)
            t1_aaea_so = t1_aaea_so[::2,1::2,::2,1::2]

            t1_ccee_so = t1_ccee_so[::2,1::2,::2,1::2]
            t1_ccea_so = t1_ccea_so[::2,1::2,::2,1::2]
            t1_caee_so = t1_caee_so[::2,1::2,::2,1::2]
            t1_ccaa_so = t1_ccaa_so[::2,1::2,::2,1::2]
            t1_aaee_so = t1_aaee_so[::2,1::2,::2,1::2]

            # t1_ce, t1_ca, t1_ae, t1_caea, t1_caae, t1_caaa, t1_aaea, t1_aaae, t1_ccee, t1_ccea, t1_caee, t1_ccaa, t1_aaee = t1_amp

            # print(">>> SO-SA t1_ce diff: {:}".format(np.sum(t1_ce_so - t1_ce)))
            # print(">>> SO-SA t1_ca diff: {:}".format(np.sum(t1_ca_so - t1_ca)))
            # print(">>> SO-SA t1_ae diff: {:}".format(np.sum(t1_ae_so - t1_ae)))

            # with open('SO_t1_ae.out', 'w') as outfile:
            #     outfile.write(repr(t1_ae_so))
            # with open('SA_t1_ae.out', 'w') as outfile:
            #     outfile.write(repr(t1_ae))


            # print(">>> SO-SA t1_caae diff: {:}".format(np.sum(t1_caae_so - t1_caae)))
            # print(">>> SO-SA t1_caea diff: {:}".format(np.sum(t1_caea_so - t1_caea)))
            # print(">>> SO-SA t1_caaa diff: {:}".format(np.sum(t1_caaa_so - t1_caaa)))
            # print(">>> SO-SA t1_aaae diff: {:}".format(np.sum(t1_aaae_so - t1_aaae)))
            # print(">>> SO-SA t1_aaea diff: {:}".format(np.sum(t1_aaea_so - t1_aaea)))
            # print(">>> SO-SA t1_ccee diff: {:}".format(np.sum(t1_ccee_so - t1_ccee)))
            # print(">>> SO-SA t1_ccea diff: {:}".format(np.sum(t1_ccea_so - t1_ccea)))
            # print(">>> SO-SA t1_caee diff: {:}".format(np.sum(t1_caee_so - t1_caee)))
            # print(">>> SO-SA t1_ccaa diff: {:}".format(np.sum(t1_ccaa_so - t1_ccaa)))
            # print(">>> SO-SA t1_aaee diff: {:}".format(np.sum(t1_aaee_so - t1_aaee)))

            t1_amp = (t1_ce_so, t1_ca_so, t1_ae_so, t1_caea_so, t1_caae_so, t1_caaa_so, t1_aaea_so, t1_aaae_so, t1_ccee_so, t1_ccea_so, t1_caee_so, t1_ccaa_so, t1_aaee_so)
            t2_ce = compute_t2_0p_singles(mr_adc, t1_amp)
            # DEBUG

        else:
            t2_ce = np.zeros((ncore, nextern))

        t2_caea = np.zeros((ncore, ncas, nextern, ncas))
        t2_aa = np.zeros((ncas, ncas))
        t2_ca = np.zeros((ncore, ncas))
        t2_ae = np.zeros((ncas, nextern))

        t2_caaa = np.zeros((ncore, ncas, ncas, ncas))
        t2_aaea = np.zeros((ncas, ncas, nextern, ncas))
        t2_ccee = np.zeros((ncore, ncore, nextern, nextern))
        t2_ccea = np.zeros((ncore, ncore, nextern, ncas))
        t2_caee = np.zeros((ncore, ncas, nextern, nextern))
        t2_ccaa = np.zeros((ncore, ncore, ncas, ncas))
        t2_aaee = np.zeros((ncas, ncas, nextern, nextern))

    return t2_ce, t2_ca, t2_ae, t2_caea, t2_caaa, t2_aaea, t2_ccee, t2_ccea, t2_caee, t2_ccaa, t2_aaee, t2_aa

def compute_t1_0(mr_adc):

    # Einsum definition from kernel
    einsum = mr_adc.interface.einsum
    einsum_type = mr_adc.interface.einsum_type

    # Variables from kernel
    e_core = mr_adc.mo_energy.c
    e_extern = mr_adc.mo_energy.e

    ncore = mr_adc.ncore
    nextern = mr_adc.nextern

    v_ccee = mr_adc.v2e.ccee

    # Computing t1_0
    d_ij = e_core[:,None] + e_core
    d_ab = e_extern[:,None] + e_extern
    D2 = -d_ij.reshape(-1,1) + d_ab.reshape(-1)
    D2 = D2.reshape((ncore, ncore, nextern, nextern))

    V1  = 1/4 * einsum('IJAB->IJAB', v_ccee, optimize = einsum_type).copy()

    t1_0 = - (V1/D2).copy()

    t1_ccee = t1_0.copy()
    e_0  = 4 * einsum('ijab,ijab', t1_ccee, v_ccee, optimize = einsum_type)
    e_0 -= 2 * einsum('ijab,jiab', t1_ccee, v_ccee, optimize = einsum_type)
    e_0 -= 2 * einsum('jiab,ijab', t1_ccee, v_ccee, optimize = einsum_type)
    e_0 += 4 * einsum('jiab,jiab', t1_ccee, v_ccee, optimize = einsum_type)

    return e_0, t1_0

def compute_t1_p1(mr_adc):

    # Einsum definition from kernel
    einsum = mr_adc.interface.einsum
    einsum_type = mr_adc.interface.einsum_type

    # Variables from kernel
    rdm_ca = mr_adc.rdm.ca
    v_ccae = mr_adc.v2e.ccae

    ncore = mr_adc.ncore
    nextern = mr_adc.nextern

    e_core = mr_adc.mo_energy.c
    e_extern = mr_adc.mo_energy.e

    # Computing K_ac
    K_ac = mr_adc_intermediates.compute_K_ac(mr_adc)

    # Orthogonalization and overlap truncation only in the active space
    S_p1_12_inv_act = mr_adc_overlap.compute_S12_p1(mr_adc, ignore_print = False)

    # Compute (S_12 K S_12)
    SKS = np.einsum("xy,yn->xn", K_ac, S_p1_12_inv_act)
    SKS = np.einsum("xm,xn->mn", S_p1_12_inv_act, SKS)

    evals, evecs = np.linalg.eigh(SKS)

    # Compute r.h.s. of the equation
    Vp1  = einsum('JIXA->IJAX', v_ccae, optimize = einsum_type).copy()
    Vp1 -= 1/2 * einsum('JIxA,Xx->IJAX', v_ccae, rdm_ca, optimize = einsum_type)

    S_12_Vp1 = np.einsum("IJAX,Xm->IJAm", - 1.0 * Vp1, S_p1_12_inv_act)

    # Multiply r.h.s. by U (e_a - e_i + e_mu)^-1 U^dag
    S_12_Vp1 = np.einsum("mp,IJAm->IJAp", evecs, S_12_Vp1)

    # Compute denominators
    d_ap = (e_extern[:,None] + evals).reshape(-1)
    d_ij = (e_core[:,None] + e_core).reshape(-1)

    d_apij = (d_ap[:,None] - d_ij).reshape(nextern, evals.shape[0], ncore, ncore)
    d_apij = d_apij**(-1)

    S_12_Vp1 = np.einsum("ApIJ,IJAp->IJAp", d_apij, S_12_Vp1)
    S_12_Vp1 = np.einsum("mp,IJAp->IJAm", evecs, S_12_Vp1)

    t1_ccea = np.einsum("IJAm,Xm->IJAX", S_12_Vp1, S_p1_12_inv_act).copy()

    e_p1  = 4 * einsum('ijax,jixa', t1_ccea, v_ccae, optimize = einsum_type)
    e_p1 -= 2 * einsum('ijax,ijxa', t1_ccea, v_ccae, optimize = einsum_type)
    e_p1 -= 2 * einsum('ijax,jiya,xy', t1_ccea, v_ccae, rdm_ca, optimize = einsum_type)
    e_p1 += einsum('ijax,ijya,xy', t1_ccea, v_ccae, rdm_ca, optimize = einsum_type)

    return e_p1, t1_ccea

def compute_t1_m1(mr_adc):

    # Einsum definition from kernel
    einsum = mr_adc.interface.einsum
    einsum_type = mr_adc.interface.einsum_type

    # Variables from kernel
    rdm_ca = mr_adc.rdm.ca
    v_caee = mr_adc.v2e.caee

    e_core = mr_adc.mo_energy.c
    e_extern = mr_adc.mo_energy.e

    ncore = mr_adc.ncore
    nextern = mr_adc.nextern

    # Computing K_ca
    K_ca = mr_adc_intermediates.compute_K_ca(mr_adc)

    # Orthogonalization and overlap truncation only in the active space
    S_m1_12_inv_act = mr_adc_overlap.compute_S12_m1(mr_adc, ignore_print = False)

    # Compute (S_12 K S_12)_{i a mu, j b nu}
    SKS = np.einsum("xy,yn->xn", K_ca, S_m1_12_inv_act)
    SKS = np.einsum("xm,xn->mn", S_m1_12_inv_act, SKS)

    evals, evecs = np.linalg.eigh(SKS)

    # Compute r.h.s. of the equation
    Vm1  = 1/2 * einsum('IxAB,Xx->IXAB', v_caee, rdm_ca, optimize = einsum_type)

    S_12_Vm1 = np.einsum("IXAB,Xm->ImAB", - 1.0 * Vm1, S_m1_12_inv_act)

    # Multiply r.h.s. by U (e_a - e_i + e_mu)^-1 U^dag
    S_12_Vm1 = np.einsum("mp,ImAB->IpAB", evecs, S_12_Vm1)

    # Compute denominators
    d_ab = (e_extern[:,None] + e_extern).reshape(-1)
    d_ix = (e_core[:,None] - evals).reshape(-1)
    d_abix = (d_ab[:,None] - d_ix).reshape(nextern, nextern, ncore, evals.shape[0])
    d_abix = d_abix**(-1)

    S_12_Vm1 = np.einsum("ABIp,IpAB->IpAB", d_abix, S_12_Vm1)
    S_12_Vm1 = np.einsum("mp,IpAB->ImAB", evecs, S_12_Vm1)

    t1_caee = np.einsum("ImAB,Xm->IXAB", S_12_Vm1, S_m1_12_inv_act).copy()

    e_m1  = 2 * einsum('ixab,iyab,yx', t1_caee, v_caee, rdm_ca, optimize = einsum_type)
    e_m1 -= einsum('ixab,iyba,yx', t1_caee, v_caee, rdm_ca, optimize = einsum_type)

    return e_m1, t1_caee

def compute_t1_p2(mr_adc):

    # Einsum definition from kernel
    einsum = mr_adc.interface.einsum
    einsum_type = mr_adc.interface.einsum_type

    # Variables from kernel
    rdm_ca = mr_adc.rdm.ca
    rdm_ccaa = mr_adc.rdm.ccaa

    e_core = mr_adc.mo_energy.c

    ncore = mr_adc.ncore
    ncas = mr_adc.ncas

    # Computing K_aacc
    K_aacc = mr_adc_intermediates.compute_K_aacc(mr_adc)
    K_aacc = K_aacc.reshape(ncas**2, ncas**2)

    # Orthogonalization and overlap truncation only in the active space
    S_p2_12_inv_act = mr_adc_overlap.compute_S12_p2(mr_adc, ignore_print = False)

    # Compute (S_12 K S_12)
    SKS = np.einsum("xy,yn->xn", K_aacc, S_p2_12_inv_act)
    SKS = np.einsum("xm,xn->mn", S_p2_12_inv_act, SKS)

    evals, evecs = np.linalg.eigh(SKS)

    # Compute r.h.s. of the equation
    v_ccaa = mr_adc.v2e.ccaa

    Vp2  = einsum('JIYX->IJXY', v_ccaa, optimize = einsum_type).copy()
    Vp2 -= 1/2 * einsum('JIxX,Yx->IJXY', v_ccaa, rdm_ca, optimize = einsum_type)
    Vp2 -= 1/2 * einsum('JIYx,Xx->IJXY', v_ccaa, rdm_ca, optimize = einsum_type)
    Vp2 += 1/3 * einsum('JIyx,XYxy->IJXY', v_ccaa, rdm_ccaa, optimize = einsum_type)
    Vp2 += 1/6 * einsum('JIyx,XYyx->IJXY', v_ccaa, rdm_ccaa, optimize = einsum_type)

    Vp2 = Vp2.reshape(ncore, ncore, ncas**2)

    S_12_Vp2 = np.einsum("IJX,Xm->IJm", - 1.0 * Vp2, S_p2_12_inv_act)

    # Multiply r.h.s. by U D^-1 U^dag
    S_12_Vp2 = np.einsum("mp,IJm->IJp", evecs, S_12_Vp2)

    # Compute denominators
    d_ij = (e_core[:,None] + e_core).reshape(-1)
    d_pij = (evals[:,None] - d_ij).reshape(evals.shape[0], ncore, ncore)
    d_pij = d_pij**(-1)

    S_12_Vp2 = np.einsum("pIJ,IJp->IJp", d_pij, S_12_Vp2)
    S_12_Vp2 = np.einsum("mp,IJp->IJm", evecs, S_12_Vp2)

    t1_ccaa = np.einsum("IJm,Xm->IJX", S_12_Vp2, S_p2_12_inv_act)
    t1_ccaa = t1_ccaa.reshape(ncore, ncore, ncas, ncas)

    e_p2  = 2 * einsum('ijxy,ijxy', t1_ccaa, v_ccaa, optimize = einsum_type)
    e_p2 -= einsum('ijxy,jixy', t1_ccaa, v_ccaa, optimize = einsum_type)
    e_p2 -= 2 * einsum('ijxy,ijxz,yz', t1_ccaa, v_ccaa, rdm_ca, optimize = einsum_type)
    e_p2 += einsum('ijxy,ijyz,xz', t1_ccaa, v_ccaa, rdm_ca, optimize = einsum_type)
    e_p2 += 1/2 * einsum('ijxy,ijzw,xyzw', t1_ccaa, v_ccaa, rdm_ccaa, optimize = einsum_type)

    return e_p2, t1_ccaa

def compute_t1_m2(mr_adc):

    # Einsum definition from kernel
    einsum = mr_adc.interface.einsum
    einsum_type = mr_adc.interface.einsum_type

    # Variables from kernel
    rdm_ccaa = mr_adc.rdm.ccaa

    e_extern = mr_adc.mo_energy.e

    ncas = mr_adc.ncas
    nextern = mr_adc.nextern

    # Computing K_ccaa
    K_ccaa = mr_adc_intermediates.compute_K_ccaa(mr_adc)
    K_ccaa = K_ccaa.reshape(ncas**2, ncas**2)

    # Orthogonalization and overlap truncation only in the active space
    S_m2_12_inv_act = mr_adc_overlap.compute_S12_m2(mr_adc, ignore_print = False)

    # Compute (S_12 K S_12)_{i a mu, j b nu}
    SKS = np.einsum("xy,yn->xn", K_ccaa, S_m2_12_inv_act)
    SKS = np.einsum("xm,xn->mn", S_m2_12_inv_act, SKS)

    evals, evecs = np.linalg.eigh(SKS)

    # Compute r.h.s. of the equation
    v_aaee = mr_adc.v2e.aaee

    Vm2  = 1/3 * einsum('xyAB,XYxy->XYAB', v_aaee, rdm_ccaa, optimize = einsum_type)
    Vm2 += 1/6 * einsum('xyAB,XYyx->XYAB', v_aaee, rdm_ccaa, optimize = einsum_type)

    Vm2 = Vm2.reshape(ncas**2, nextern, nextern)

    S_12_Vm2 = np.einsum("XAB,Xm->mAB", - 1.0 * Vm2, S_m2_12_inv_act)

    # Multiply r.h.s. by U (e_a - e_i + e_mu)^-1 U^dag
    S_12_Vm2 = np.einsum("mp,mAB->pAB", evecs, S_12_Vm2)

    # Compute denominators
    d_ab = (e_extern[:,None] + e_extern).reshape(-1)
    d_abp = (d_ab[:,None] + evals).reshape(nextern, nextern, evals.shape[0])
    d_abp = d_abp**(-1)

    S_12_Vm2 = np.einsum("ABp,pAB->pAB", d_abp, S_12_Vm2)
    S_12_Vm2 = np.einsum("mp,pAB->mAB", evecs, S_12_Vm2)

    t1_aaee = np.einsum("mAB,Xm->XAB", S_12_Vm2, S_m2_12_inv_act)
    t1_aaee = t1_aaee.reshape(ncas, ncas, nextern, nextern)

    e_m2  = 1/2 * einsum('xyab,zwab,xyzw', t1_aaee, v_aaee, rdm_ccaa, optimize = einsum_type)

    return e_m2, t1_aaee

def compute_t1_p1p_sanity_check(mr_adc):

    # Einsum definition from kernel
    einsum = mr_adc.interface.einsum
    einsum_type = mr_adc.interface.einsum_type

    # Variables from kernel
    rdm_ca = mr_adc.rdm.ca
    rdm_ccaa = mr_adc.rdm.ccaa
    rdm_cccaaa = mr_adc.rdm.cccaaa

    e_core = mr_adc.mo_energy.c

    ncore = mr_adc.ncore
    ncas = mr_adc.ncas
    nocc = mr_adc.nocc

    n_x = ncas * 2
    n_xzw = ncas * 2 * ncas * 2 * (ncas * 2 - 1) // 2
    dim_act = n_x + n_xzw
    aa_ind = np.tril_indices(ncas * 2, k=-1)

    # Computing K_p1p
    K_p1p = mr_adc_intermediates.compute_K_p1p_sanity_check(mr_adc)

    # Orthogonalization and overlap truncation only in the active space
    S_p1p_12_inv_act = mr_adc_overlap.compute_S12_p1p_sanity_check_gno_projector(mr_adc, ignore_print = False)

    SKS = reduce(np.dot, (S_p1p_12_inv_act.T, K_p1p, S_p1p_12_inv_act))
    evals, evecs = np.linalg.eigh(SKS)

    # Compute r.h.s. of the equation
    h_ca = mr_adc.h1eff[:ncore,ncore:nocc].copy()
    v_caaa = mr_adc.v2e.caaa
    v_ccca = mr_adc.v2e.ccca

    V = np.zeros((ncore * 2, dim_act))

    V1 = np.zeros((ncore * 2, ncas * 2))

    V1_a_a =- einsum('IX->IX', h_ca, optimize = einsum_type).copy()
    V1_a_a += 1/2 * einsum('Ix,xX->IX', h_ca, rdm_ca, optimize = einsum_type)
    V1_a_a -= einsum('IxXy,yx->IX', v_caaa, rdm_ca, optimize = einsum_type)
    V1_a_a += 1/2 * einsum('IxyX,yx->IX', v_caaa, rdm_ca, optimize = einsum_type)
    V1_a_a += 1/2 * einsum('Ixyz,yzXx->IX', v_caaa, rdm_ccaa, optimize = einsum_type)

    V1[::2,::2] = V1_a_a.copy()
    V1[1::2,1::2] = V1_a_a.copy()

    V2 = np.zeros((ncore * 2, ncas * 2, ncas * 2, ncas * 2))

    V2_aa_aa =- einsum('IV,UX->IUXV', h_ca, np.identity(ncas), optimize = einsum_type)
    V2_aa_aa += 1/2 * einsum('IV,UX->IUXV', h_ca, rdm_ca, optimize = einsum_type)
    V2_aa_aa -= 1/2 * einsum('IX,UV->IUXV', h_ca, rdm_ca, optimize = einsum_type)
    V2_aa_aa += 1/6 * einsum('Ix,UxVX->IUXV', h_ca, rdm_ccaa, optimize = einsum_type)
    V2_aa_aa -= 1/6 * einsum('Ix,UxXV->IUXV', h_ca, rdm_ccaa, optimize = einsum_type)
    V2_aa_aa += 1/2 * einsum('IxVX,Ux->IUXV', v_caaa, rdm_ca, optimize = einsum_type)
    V2_aa_aa += 1/2 * einsum('IxVy,UyXx->IUXV', v_caaa, rdm_ccaa, optimize = einsum_type)
    V2_aa_aa -= 1/2 * einsum('IxXV,Ux->IUXV', v_caaa, rdm_ca, optimize = einsum_type)
    V2_aa_aa -= 1/2 * einsum('IxXy,UyVx->IUXV', v_caaa, rdm_ccaa, optimize = einsum_type)
    V2_aa_aa -= 1/6 * einsum('IxyV,UyXx->IUXV', v_caaa, rdm_ccaa, optimize = einsum_type)
    V2_aa_aa += 1/6 * einsum('IxyV,UyxX->IUXV', v_caaa, rdm_ccaa, optimize = einsum_type)
    V2_aa_aa += 1/6 * einsum('IxyX,UyVx->IUXV', v_caaa, rdm_ccaa, optimize = einsum_type)
    V2_aa_aa -= 1/6 * einsum('IxyX,UyxV->IUXV', v_caaa, rdm_ccaa, optimize = einsum_type)
    V2_aa_aa += 1/6 * einsum('Ixyz,UyzVXx->IUXV', v_caaa, rdm_cccaaa, optimize = einsum_type)
    V2_aa_aa -= 1/6 * einsum('Ixyz,UyzXVx->IUXV', v_caaa, rdm_cccaaa, optimize = einsum_type)
    V2_aa_aa += 1/2 * einsum('Ix,UX,xV->IUXV', h_ca, np.identity(ncas), rdm_ca, optimize = einsum_type)
    V2_aa_aa -= einsum('UX,IxVy,yx->IUXV', np.identity(ncas), v_caaa, rdm_ca, optimize = einsum_type)
    V2_aa_aa += 1/2 * einsum('UX,IxyV,yx->IUXV', np.identity(ncas), v_caaa, rdm_ca, optimize = einsum_type)
    V2_aa_aa += 1/2 * einsum('UX,Ixyz,yzVx->IUXV', np.identity(ncas), v_caaa, rdm_ccaa, optimize = einsum_type)

    V2_ab_ab =- 1/2 * einsum('IX,UV->IUXV', h_ca, rdm_ca, optimize = einsum_type)
    V2_ab_ab += 1/3 * einsum('Ix,UxVX->IUXV', h_ca, rdm_ccaa, optimize = einsum_type)
    V2_ab_ab += 1/6 * einsum('Ix,UxXV->IUXV', h_ca, rdm_ccaa, optimize = einsum_type)
    V2_ab_ab -= 1/2 * einsum('IxXV,Ux->IUXV', v_caaa, rdm_ca, optimize = einsum_type)
    V2_ab_ab -= 1/2 * einsum('IxXy,UyVx->IUXV', v_caaa, rdm_ccaa, optimize = einsum_type)
    V2_ab_ab += 1/6 * einsum('IxyV,UyXx->IUXV', v_caaa, rdm_ccaa, optimize = einsum_type)
    V2_ab_ab += 1/3 * einsum('IxyV,UyxX->IUXV', v_caaa, rdm_ccaa, optimize = einsum_type)
    V2_ab_ab += 1/3 * einsum('IxyX,UyVx->IUXV', v_caaa, rdm_ccaa, optimize = einsum_type)
    V2_ab_ab += 1/6 * einsum('IxyX,UyxV->IUXV', v_caaa, rdm_ccaa, optimize = einsum_type)
    V2_ab_ab += 1/4 * einsum('Ixyz,UyzVXx->IUXV', v_caaa, rdm_cccaaa, optimize = einsum_type)
    V2_ab_ab -= 1/12 * einsum('Ixyz,UyzVxX->IUXV', v_caaa, rdm_cccaaa, optimize = einsum_type)
    V2_ab_ab += 1/12 * einsum('Ixyz,UyzXVx->IUXV', v_caaa, rdm_cccaaa, optimize = einsum_type)
    V2_ab_ab -= 1/12 * einsum('Ixyz,UyzXxV->IUXV', v_caaa, rdm_cccaaa, optimize = einsum_type)
    V2_ab_ab -= 1/12 * einsum('Ixyz,UyzxVX->IUXV', v_caaa, rdm_cccaaa, optimize = einsum_type)
    V2_ab_ab -= 1/12 * einsum('Ixyz,UyzxXV->IUXV', v_caaa, rdm_cccaaa, optimize = einsum_type)

    V2_ab_ba =- einsum('IV,UX->IUXV', h_ca, np.identity(ncas), optimize = einsum_type)
    V2_ab_ba += 1/2 * einsum('IV,UX->IUXV', h_ca, rdm_ca, optimize = einsum_type)
    V2_ab_ba -= 1/6 * einsum('Ix,UxVX->IUXV', h_ca, rdm_ccaa, optimize = einsum_type)
    V2_ab_ba -= 1/3 * einsum('Ix,UxXV->IUXV', h_ca, rdm_ccaa, optimize = einsum_type)
    V2_ab_ba += 1/2 * einsum('IxVX,Ux->IUXV', v_caaa, rdm_ca, optimize = einsum_type)
    V2_ab_ba += 1/2 * einsum('IxVy,UyXx->IUXV', v_caaa, rdm_ccaa, optimize = einsum_type)
    V2_ab_ba -= 1/3 * einsum('IxyV,UyXx->IUXV', v_caaa, rdm_ccaa, optimize = einsum_type)
    V2_ab_ba -= 1/6 * einsum('IxyV,UyxX->IUXV', v_caaa, rdm_ccaa, optimize = einsum_type)
    V2_ab_ba -= 1/6 * einsum('IxyX,UyVx->IUXV', v_caaa, rdm_ccaa, optimize = einsum_type)
    V2_ab_ba -= 1/3 * einsum('IxyX,UyxV->IUXV', v_caaa, rdm_ccaa, optimize = einsum_type)
    V2_ab_ba -= 1/12 * einsum('Ixyz,UyzVXx->IUXV', v_caaa, rdm_cccaaa, optimize = einsum_type)
    V2_ab_ba += 1/12 * einsum('Ixyz,UyzVxX->IUXV', v_caaa, rdm_cccaaa, optimize = einsum_type)
    V2_ab_ba -= 1/4 * einsum('Ixyz,UyzXVx->IUXV', v_caaa, rdm_cccaaa, optimize = einsum_type)
    V2_ab_ba += 1/12 * einsum('Ixyz,UyzXxV->IUXV', v_caaa, rdm_cccaaa, optimize = einsum_type)
    V2_ab_ba += 1/12 * einsum('Ixyz,UyzxVX->IUXV', v_caaa, rdm_cccaaa, optimize = einsum_type)
    V2_ab_ba += 1/12 * einsum('Ixyz,UyzxXV->IUXV', v_caaa, rdm_cccaaa, optimize = einsum_type)
    V2_ab_ba += 1/2 * einsum('Ix,UX,xV->IUXV', h_ca, np.identity(ncas), rdm_ca, optimize = einsum_type)
    V2_ab_ba -= einsum('UX,IxVy,yx->IUXV', np.identity(ncas), v_caaa, rdm_ca, optimize = einsum_type)
    V2_ab_ba += 1/2 * einsum('UX,IxyV,yx->IUXV', np.identity(ncas), v_caaa, rdm_ca, optimize = einsum_type)
    V2_ab_ba += 1/2 * einsum('UX,Ixyz,yzVx->IUXV', np.identity(ncas), v_caaa, rdm_ccaa, optimize = einsum_type)

    V2[::2,::2,::2,::2,] = V2_aa_aa.copy()
    V2[1::2,1::2,1::2,1::2] = V2_aa_aa.copy()

    V2[::2,1::2,::2,1::2] = V2_ab_ab.copy()
    V2[1::2,::2,1::2,::2] = V2_ab_ab.copy()

    V2[::2,1::2,1::2,::2] = V2_ab_ba.copy()
    V2[1::2,::2,::2,1::2] = V2_ab_ba.copy()
    V2 = V2[:,:,aa_ind[0],aa_ind[1]].reshape(ncore * 2, -1).copy()

    V[:,:n_x] = V1.copy()
    V[:,n_x:] = V2.copy()

    S_12_V = np.einsum("iP,Pm->im", - 1.0 * V, S_p1p_12_inv_act)

    # Multiply r.h.s. by U (- e_i + e_mu)^-1 U^dag
    S_12_V = np.einsum("mp,im->ip", evecs, S_12_V)

    # Compute denominators
    e_core_so = np.zeros(ncore * 2)
    e_core_so[::2] = e_core.copy()
    e_core_so[1::2] = e_core.copy()

    d_ip = (-e_core_so[:,None] + evals)
    d_ip = d_ip**(-1)

    S_12_V *= d_ip
    S_12_V = np.einsum("mp,ip->im", evecs, S_12_V)
    t_p1p = np.einsum("Pm,im->iP", S_p1p_12_inv_act, S_12_V)

    t1_ca = t_p1p[:,:n_x].copy()
    t1_caaa = np.zeros((ncore * 2, ncas * 2, ncas * 2, ncas * 2))
    t1_caaa[:,:,aa_ind[0],aa_ind[1]] =  t_p1p[:,n_x:].reshape(ncore * 2, ncas * 2, -1)
    t1_caaa[:,:,aa_ind[1],aa_ind[0]] = -t_p1p[:,n_x:].reshape(ncore * 2, ncas * 2, -1)

    # Transpose t2 indices to the conventional order
    t1_caaa = t1_caaa.transpose(0,1,3,2).copy()

    t1_ca = t1_ca[::2,::2].copy()
    t1_caaa = t1_caaa[::2,1::2,::2,1::2].copy()
    # if mr_adc.debug_mode:
    #     print (">>> SA t1_ca norm: {:}".format(np.linalg.norm(t1_ca)))
    #     with open('SA_t1_ca.out', 'w') as outfile:
    #         outfile.write(repr(t1_ca))

    #     with open('SA_evals.out', 'w') as outfile:
    #         outfile.write(repr(evals))

    #     with open('SA_S_p1p_12_inv_act.out', 'w') as outfile:
    #         outfile.write(repr(S_p1p_12_inv_act))

    e_p1p  = 2 * einsum('ix,ix', h_ca, t1_ca, optimize = einsum_type)
    e_p1p += 2 * einsum('ix,izxy,yz', h_ca, t1_caaa, rdm_ca, optimize = einsum_type)
    e_p1p -= einsum('ix,iy,yx', h_ca, t1_ca, rdm_ca, optimize = einsum_type)
    e_p1p -= einsum('ix,iyzx,zy', h_ca, t1_caaa, rdm_ca, optimize = einsum_type)
    e_p1p -= einsum('ix,iyzw,zwxy', h_ca, t1_caaa, rdm_ccaa, optimize = einsum_type)
    e_p1p += 2 * einsum('izxy,iwxy,wz', t1_caaa, v_caaa, rdm_ca, optimize = einsum_type)
    e_p1p -= einsum('izxy,iwyx,wz', t1_caaa, v_caaa, rdm_ca, optimize = einsum_type)
    e_p1p += 2 * einsum('izxy,iuxw,yuzw', t1_caaa, v_caaa, rdm_ccaa, optimize = einsum_type)
    e_p1p -= einsum('izxy,iuwx,yuzw', t1_caaa, v_caaa, rdm_ccaa, optimize = einsum_type)
    e_p1p += 2 * einsum('ix,iyxz,yz', t1_ca, v_caaa, rdm_ca, optimize = einsum_type)
    e_p1p -= einsum('ix,iyzx,yz', t1_ca, v_caaa, rdm_ca, optimize = einsum_type)
    e_p1p -= einsum('ix,iyzw,xyzw', t1_ca, v_caaa, rdm_ccaa, optimize = einsum_type)
    e_p1p -= einsum('ixyz,iwzu,ywxu', t1_caaa, v_caaa, rdm_ccaa, optimize = einsum_type)
    e_p1p -= einsum('ixyz,iwuz,ywux', t1_caaa, v_caaa, rdm_ccaa, optimize = einsum_type)
    e_p1p += 1/6 * einsum('ixyz,iwuv,yzwxuv', t1_caaa, v_caaa, rdm_cccaaa, optimize = einsum_type)
    e_p1p += 1/6 * einsum('ixyz,iwuv,yzwxvu', t1_caaa, v_caaa, rdm_cccaaa, optimize = einsum_type)
    e_p1p -= 5/6 * einsum('ixyz,iwuv,yzwuxv', t1_caaa, v_caaa, rdm_cccaaa, optimize = einsum_type)
    e_p1p += 1/6 * einsum('ixyz,iwuv,yzwuvx', t1_caaa, v_caaa, rdm_cccaaa, optimize = einsum_type)
    e_p1p += 1/6 * einsum('ixyz,iwuv,yzwvxu', t1_caaa, v_caaa, rdm_cccaaa, optimize = einsum_type)
    e_p1p += 1/6 * einsum('ixyz,iwuv,yzwvux', t1_caaa, v_caaa, rdm_cccaaa, optimize = einsum_type)

    return e_p1p, t1_ca, t1_caaa

def compute_t1_m1p_sanity_check(mr_adc):

    # Einsum definition from kernel
    einsum = mr_adc.interface.einsum
    einsum_type = mr_adc.interface.einsum_type

    # Variables from kernel
    rdm_ca = mr_adc.rdm.ca
    rdm_ccaa = mr_adc.rdm.ccaa
    rdm_cccaaa = mr_adc.rdm.cccaaa

    e_extern = mr_adc.mo_energy.e

    ncore = mr_adc.ncore
    ncas = mr_adc.ncas
    nocc = mr_adc.nocc
    nextern = mr_adc.nextern

    n_x = ncas * 2
    n_xzw = ncas * 2 * ncas * 2 * (ncas * 2 - 1) // 2
    dim_act = n_x + n_xzw
    aa_ind = np.tril_indices(ncas * 2, k=-1)

    # Computing K_m1p
    K_m1p = mr_adc_intermediates.compute_K_m1p_sanity_check(mr_adc)

    # Orthogonalization and overlap truncation only in the active space
    S_m1p_12_inv_act = mr_adc_overlap.compute_S12_m1p_sanity_check(mr_adc, ignore_print = False)
    # S_m1p_12_inv_act = mr_adc_overlap.compute_S12_m1p_sanity_check_gno_projector(mr_adc, ignore_print = False)

    SKS = reduce(np.dot, (S_m1p_12_inv_act.T, K_m1p, S_m1p_12_inv_act))
    evals, evecs = np.linalg.eigh(SKS)

    # Compute r.h.s. of the equation
    h_ae = mr_adc.h1eff[ncore:nocc,nocc:].copy()
    v_aaae = mr_adc.v2e.aaae

    V = np.zeros((dim_act, nextern * 2))

    V1 = np.zeros((ncas * 2, nextern * 2))

    V1_a_a =- 1/2 * einsum('xA,Xx->XA', h_ae, rdm_ca, optimize = einsum_type)
    V1_a_a -= 1/2 * einsum('xyzA,Xzyx->XA', v_aaae, rdm_ccaa, optimize = einsum_type)

    V1[::2,::2] = V1_a_a.copy()
    V1[1::2,1::2] = V1_a_a.copy()

    V2 = np.zeros((ncas * 2, ncas * 2, ncas * 2, nextern * 2))
    V2_aa_aa  = 1/6 * einsum('xA,UXVx->XUVA', h_ae, rdm_ccaa, optimize = einsum_type)
    V2_aa_aa -= 1/6 * einsum('xA,UXxV->XUVA', h_ae, rdm_ccaa, optimize = einsum_type)
    V2_aa_aa -= 1/6 * einsum('xyVA,UXyx->XUVA', v_aaae, rdm_ccaa, optimize = einsum_type)
    V2_aa_aa += 1/6 * einsum('xyVA,UXxy->XUVA', v_aaae, rdm_ccaa, optimize = einsum_type)
    V2_aa_aa += 1/6 * einsum('xyzA,UXzVyx->XUVA', v_aaae, rdm_cccaaa, optimize = einsum_type)
    V2_aa_aa -= 1/6 * einsum('xyzA,UXzyVx->XUVA', v_aaae, rdm_cccaaa, optimize = einsum_type)
    V2_aa_aa -= 1/2 * einsum('xA,UV,Xx->XUVA', h_ae, np.identity(ncas), rdm_ca, optimize = einsum_type)
    V2_aa_aa -= 1/2 * einsum('UV,xyzA,Xzyx->XUVA', np.identity(ncas), v_aaae, rdm_ccaa, optimize = einsum_type)

    V2_ab_ab =- 1/6 * einsum('xA,UXVx->XUVA', h_ae, rdm_ccaa, optimize = einsum_type)
    V2_ab_ab -= 1/3 * einsum('xA,UXxV->XUVA', h_ae, rdm_ccaa, optimize = einsum_type)
    V2_ab_ab -= 1/3 * einsum('xyVA,UXyx->XUVA', v_aaae, rdm_ccaa, optimize = einsum_type)
    V2_ab_ab -= 1/6 * einsum('xyVA,UXxy->XUVA', v_aaae, rdm_ccaa, optimize = einsum_type)
    V2_ab_ab -= 1/12 * einsum('xyzA,UXzVyx->XUVA', v_aaae, rdm_cccaaa, optimize = einsum_type)
    V2_ab_ab += 1/12 * einsum('xyzA,UXzVxy->XUVA', v_aaae, rdm_cccaaa, optimize = einsum_type)
    V2_ab_ab -= 1/4 * einsum('xyzA,UXzyVx->XUVA', v_aaae, rdm_cccaaa, optimize = einsum_type)
    V2_ab_ab += 1/12 * einsum('xyzA,UXzyxV->XUVA', v_aaae, rdm_cccaaa, optimize = einsum_type)
    V2_ab_ab += 1/12 * einsum('xyzA,UXzxVy->XUVA', v_aaae, rdm_cccaaa, optimize = einsum_type)
    V2_ab_ab += 1/12 * einsum('xyzA,UXzxyV->XUVA', v_aaae, rdm_cccaaa, optimize = einsum_type)

    V2_ab_ba  = 1/3 * einsum('xA,UXVx->XUVA', h_ae, rdm_ccaa, optimize = einsum_type)
    V2_ab_ba += 1/6 * einsum('xA,UXxV->XUVA', h_ae, rdm_ccaa, optimize = einsum_type)
    V2_ab_ba += 1/6 * einsum('xyVA,UXyx->XUVA', v_aaae, rdm_ccaa, optimize = einsum_type)
    V2_ab_ba += 1/3 * einsum('xyVA,UXxy->XUVA', v_aaae, rdm_ccaa, optimize = einsum_type)
    V2_ab_ba += 1/4 * einsum('xyzA,UXzVyx->XUVA', v_aaae, rdm_cccaaa, optimize = einsum_type)
    V2_ab_ba -= 1/12 * einsum('xyzA,UXzVxy->XUVA', v_aaae, rdm_cccaaa, optimize = einsum_type)
    V2_ab_ba += 1/12 * einsum('xyzA,UXzyVx->XUVA', v_aaae, rdm_cccaaa, optimize = einsum_type)
    V2_ab_ba -= 1/12 * einsum('xyzA,UXzyxV->XUVA', v_aaae, rdm_cccaaa, optimize = einsum_type)
    V2_ab_ba -= 1/12 * einsum('xyzA,UXzxVy->XUVA', v_aaae, rdm_cccaaa, optimize = einsum_type)
    V2_ab_ba -= 1/12 * einsum('xyzA,UXzxyV->XUVA', v_aaae, rdm_cccaaa, optimize = einsum_type)
    V2_ab_ba -= 1/2 * einsum('xA,UV,Xx->XUVA', h_ae, np.identity(ncas), rdm_ca, optimize = einsum_type)
    V2_ab_ba -= 1/2 * einsum('UV,xyzA,Xzyx->XUVA', np.identity(ncas), v_aaae, rdm_ccaa, optimize = einsum_type)

    V2[::2,::2,::2,::2] = V2_aa_aa.copy()
    V2[1::2,1::2,1::2,1::2] = V2_aa_aa.copy()

    V2[::2,1::2,::2,1::2] = V2_ab_ab.copy()
    V2[1::2,::2,1::2,::2] = V2_ab_ab.copy()

    V2[::2,1::2,1::2,::2] = V2_ab_ba.copy()
    V2[1::2,::2,::2,1::2] = V2_ab_ba.copy()

    V2 = V2[aa_ind[0],aa_ind[1]].reshape(-1, nextern * 2).copy()

    V[:n_x,:] = - 1.0 * V1.copy()
    V[n_x:,:] = - 1.0 * V2.copy()

    S_12_V = np.einsum("Pa,Pm->ma", V, S_m1p_12_inv_act)

    # Multiply r.h.s. by U (e_mu + e_a)^-1 U^dag
    S_12_V = np.einsum("mp,ma->pa", evecs, S_12_V)

    # Compute denominators
    e_extern_so = np.zeros(nextern * 2)
    e_extern_so[::2] = e_extern.copy()
    e_extern_so[1::2] = e_extern.copy()

    d_pa = (evals[:,None] + e_extern_so)
    d_pa = d_pa**(-1)

    S_12_V *= d_pa
    S_12_V = np.einsum("mp,pa->ma", evecs, S_12_V)
    t_m1p = np.einsum("Pm,ma->Pa", S_m1p_12_inv_act, S_12_V)

    t1_ae = t_m1p[:n_x,:].copy()
    t1_aaea = np.zeros((ncas * 2, ncas * 2, ncas * 2, nextern * 2))
    t1_aaea[aa_ind[0],aa_ind[1], :, :] =  t_m1p[n_x:, :].reshape(-1, ncas * 2, nextern * 2)
    t1_aaea[aa_ind[1],aa_ind[0], :, :] = -t_m1p[n_x:, :].reshape(-1, ncas * 2, nextern * 2)

    # Transpose t2 indices to the conventional order
    t1_aaea = t1_aaea.transpose(0,1,3,2).copy()

    t1_ae = t1_ae[::2,::2].copy()
    #TODO: Is it t1_aaae really needed?
    t1_aaae = - t1_aaea[::2,1::2,1::2,::2].transpose(0,1,3,2).copy()
    t1_aaea =   t1_aaea[::2,1::2,::2,1::2].copy()

    # Compute correlation energy contribution
    e_m1p  = einsum('xa,ya,xy', h_ae, t1_ae, rdm_ca, optimize = einsum_type)
    e_m1p += einsum('xa,zway,xyzw', h_ae, t1_aaea, rdm_ccaa, optimize = einsum_type)
    e_m1p += einsum('xa,wzya,zwxy', t1_ae, v_aaae, rdm_ccaa, optimize = einsum_type)
    e_m1p += einsum('yzax,uwxa,wuyz', t1_aaea, v_aaae, rdm_ccaa, optimize = einsum_type)
    e_m1p -= 1/6 * einsum('yzax,vuwa,xuvyzw', t1_aaea, v_aaae, rdm_cccaaa, optimize = einsum_type)
    e_m1p -= 1/6 * einsum('yzax,vuwa,xuvywz', t1_aaea, v_aaae, rdm_cccaaa, optimize = einsum_type)
    e_m1p += 5/6 * einsum('yzax,vuwa,xuvzyw', t1_aaea, v_aaae, rdm_cccaaa, optimize = einsum_type)
    e_m1p -= 1/6 * einsum('yzax,vuwa,xuvzwy', t1_aaea, v_aaae, rdm_cccaaa, optimize = einsum_type)
    e_m1p -= 1/6 * einsum('yzax,vuwa,xuvwyz', t1_aaea, v_aaae, rdm_cccaaa, optimize = einsum_type)
    e_m1p -= 1/6 * einsum('yzax,vuwa,xuvwzy', t1_aaea, v_aaae, rdm_cccaaa, optimize = einsum_type)

    return e_m1p, t1_ae, t1_aaea, t1_aaae

def compute_t1_0p_sanity_check(mr_adc):

    # Einsum definition from kernel
    einsum = mr_adc.interface.einsum
    einsum_type = mr_adc.interface.einsum_type

    # Variables from kernel
    ncore = mr_adc.ncore
    ncas = mr_adc.ncas
    nocc = mr_adc.nocc
    nextern = mr_adc.nextern

    h_ce = mr_adc.h1eff[:ncore,nocc:]
    v_caea = mr_adc.v2e.caea
    v_caae = mr_adc.v2e.caae

    rdm_ca = mr_adc.rdm.ca
    rdm_ccaa = mr_adc.rdm.ccaa

    e_core = mr_adc.mo_energy.c
    e_extern = mr_adc.mo_energy.e

    # Computing K_caca
    K_caca = mr_adc_intermediates.compute_K_caca_sanity_check(mr_adc)

    # Orthogonalization and overlap truncation only in the active space
    S_0p_12_inv_act = mr_adc_overlap.compute_S12_0p_sanity_check_gno_projector(mr_adc, ignore_print = False)

    # Compute (S_12 K S_12)_{i a mu, j b nu}
    SKS = np.einsum("xywz,zwn->xyn", K_caca, S_0p_12_inv_act[1:,:].reshape(ncas * 2, ncas * 2, -1))
    SKS = np.einsum("xym,xyn->mn", S_0p_12_inv_act[1:,:].reshape(ncas * 2, ncas * 2, -1), SKS)

    evals, evecs = np.linalg.eigh(SKS)

    # Compute r.h.s. of the equation
    V0p = np.zeros((ncore * 2, nextern * 2, ncas * 2 * ncas * 2 + 1))

    V1 = np.zeros((ncore * 2, nextern * 2))
    V1_a_a  = einsum('IA->IA', h_ce, optimize = einsum_type).copy()
    V1_a_a += einsum('IyAx,xy->IA', v_caea, rdm_ca, optimize = einsum_type)
    V1_a_a -= 1/2 * einsum('IyxA,xy->IA', v_caae, rdm_ca, optimize = einsum_type)

    V1[::2,::2] = V1_a_a.copy()
    V1[1::2,1::2] = V1_a_a.copy()

    V0p[:,:,0] = V1.copy()

    V2 = np.zeros((ncore * 2, nextern * 2, ncas * 2, ncas * 2))
    V2_aa_aa =- 1/2 * einsum('IA,XY->IAXY', h_ce, rdm_ca, optimize = einsum_type)
    V2_aa_aa -= 1/2 * einsum('IxAY,Xx->IAXY', v_caea, rdm_ca, optimize = einsum_type)
    V2_aa_aa += 1/2 * einsum('IxYA,Xx->IAXY', v_caae, rdm_ca, optimize = einsum_type)
    V2_aa_aa -= 1/2 * einsum('IyAx,XxYy->IAXY', v_caea, rdm_ccaa, optimize = einsum_type)
    V2_aa_aa += 1/6 * einsum('IyxA,XxYy->IAXY', v_caae, rdm_ccaa, optimize = einsum_type)
    V2_aa_aa -= 1/6 * einsum('IyxA,XxyY->IAXY', v_caae, rdm_ccaa, optimize = einsum_type)

    V2_ab_ba  = 1/2 * einsum('IxYA,Xx->IAXY', v_caae, rdm_ca, optimize = einsum_type)
    V2_ab_ba -= 1/6 * einsum('IyxA,XxYy->IAXY', v_caae, rdm_ccaa, optimize = einsum_type)
    V2_ab_ba -= 1/3 * einsum('IyxA,XxyY->IAXY', v_caae, rdm_ccaa, optimize = einsum_type)

    V2_aa_bb =- 1/2 * einsum('IA,XY->IAXY', h_ce, rdm_ca, optimize = einsum_type)
    V2_aa_bb -= 1/2 * einsum('IxAY,Xx->IAXY', v_caea, rdm_ca, optimize = einsum_type)
    V2_aa_bb -= 1/2 * einsum('IyAx,XxYy->IAXY', v_caea, rdm_ccaa, optimize = einsum_type)
    V2_aa_bb += 1/3 * einsum('IyxA,XxYy->IAXY', v_caae, rdm_ccaa, optimize = einsum_type)
    V2_aa_bb += 1/6 * einsum('IyxA,XxyY->IAXY', v_caae, rdm_ccaa, optimize = einsum_type)

    V2[::2,::2,::2,::2] = V2_aa_aa.copy()
    V2[1::2,1::2,1::2,1::2] = V2_aa_aa.copy()

    V2[::2,1::2,1::2,::2] = V2_ab_ba.copy()
    V2[1::2,::2,::2,1::2] = V2_ab_ba.copy()

    V2[::2,::2,1::2,1::2] = V2_aa_bb.copy()
    V2[1::2,1::2,::2,::2] = V2_aa_bb.copy()

    V0p[:,:,1:] = V2.reshape(ncore * 2, nextern * 2, -1)

    V0p = V0p.reshape(ncore * 2, nextern * 2, -1)

    S_12_V0p = np.einsum("iaP,Pm->iam", V0p, S_0p_12_inv_act)

    # Multiply r.h.s. by U (e_a - e_i + e_mu)^-1 U^dag
    S_12_V0p = np.einsum("mp,iam->iap", evecs, S_12_V0p)

    # Compute denominators
    e_core_so = np.zeros(ncore * 2)
    e_core_so[::2] = e_core.copy()
    e_core_so[1::2] = e_core.copy()

    e_extern_so = np.zeros(nextern * 2)
    e_extern_so[::2] = e_extern.copy()
    e_extern_so[1::2] = e_extern.copy()

    d_ai = (e_extern_so[:,None] - e_core_so).reshape(-1)
    d_aip = (d_ai[:,None] + evals).reshape(nextern * 2, ncore * 2, -1)
    d_aip = d_aip**(-1)

    S_12_V0p = np.einsum("aip,iap->iap", d_aip, S_12_V0p)
    S_12_V0p = np.einsum("mp,iap->iam", evecs, S_12_V0p)

    t0p = np.einsum("iam,Pm->iaP", S_12_V0p, S_0p_12_inv_act)

    t1_ce = t0p[:,:,0].copy()
    t1_caea = t0p[:,:,1:].reshape(ncore * 2, nextern * 2, ncas * 2, ncas * 2)
    t1_caea = t1_caea.transpose(0,2,1,3).copy()

    t1_ce = t1_ce[::2,::2].copy()
    t1_caae = - t1_caea[::2,1::2,1::2,::2].transpose(0,1,3,2).copy()
    t1_caea =   t1_caea[::2,1::2,::2,1::2].copy()

    e_0p  = 2 * einsum('ia,ia', h_ce, t1_ce, optimize = einsum_type)
    e_0p += 2 * einsum('ia,ixay,yx', h_ce, t1_caea, rdm_ca, optimize = einsum_type)
    e_0p -= einsum('ia,ixya,yx', h_ce, t1_caae, rdm_ca, optimize = einsum_type)
    e_0p += 2 * einsum('ixya,izya,zx', t1_caae, v_caae, rdm_ca, optimize = einsum_type)
    e_0p -= einsum('ixya,izay,zx', t1_caae, v_caea, rdm_ca, optimize = einsum_type)
    e_0p -= einsum('ixya,izaw,yzxw', t1_caae, v_caea, rdm_ccaa, optimize = einsum_type)
    e_0p -= einsum('ixya,izwa,yzwx', t1_caae, v_caae, rdm_ccaa, optimize = einsum_type)
    e_0p += 2 * einsum('ixay,izay,zx', t1_caea, v_caea, rdm_ca, optimize = einsum_type)
    e_0p += 2 * einsum('ixay,izaw,yzxw', t1_caea, v_caea, rdm_ccaa, optimize = einsum_type)
    e_0p -= einsum('ixay,izya,zx', t1_caea, v_caae, rdm_ca, optimize = einsum_type)
    e_0p -= einsum('ixay,izwa,yzxw', t1_caea, v_caae, rdm_ccaa, optimize = einsum_type)
    e_0p += 2 * einsum('ia,ixay,xy', t1_ce, v_caea, rdm_ca, optimize = einsum_type)
    e_0p -= einsum('ia,ixya,xy', t1_ce, v_caae, rdm_ca, optimize = einsum_type)

    return e_0p, t1_ce, t1_caea, t1_caae

def compute_t2_0p_singles(mr_adc, t1_amp):

    # Einsum definition from kernel
    einsum = mr_adc.interface.einsum
    einsum_type = mr_adc.interface.einsum_type

    # Variables from kernel
    rdm_ca = mr_adc.rdm.ca
    rdm_ccaa = mr_adc.rdm.ccaa
    rdm_cccaaa = mr_adc.rdm.cccaaa

    e_core = mr_adc.mo_energy.c
    e_extern = mr_adc.mo_energy.e

    ncore = mr_adc.ncore
    nocc = mr_adc.nocc
    nextern = mr_adc.nextern

    # Compute r.h.s. of the equation
    # One-electron integrals
    h_ca = mr_adc.h1eff[:ncore,ncore:nocc].copy()
    h_ce = mr_adc.h1eff[:ncore,nocc:].copy()
    h_ae = mr_adc.h1eff[ncore:nocc,nocc:].copy()
    h_aa = mr_adc.h1eff_act

    # Two-electron integrals
    #TODO: Check if all integrals were calculate before
    v_aaaa = mr_adc.v2e.aaaa
    v_caaa = mr_adc.v2e.caaa
    v_ccee = mr_adc.v2e.ccee
    v_caea = mr_adc.v2e.caea
    v_caae = mr_adc.v2e.caae
    v_ccae = mr_adc.v2e.ccae
    v_caee = mr_adc.v2e.caee
    v_aaae = mr_adc.v2e.aaae
    v_ccca = mr_adc.v2e.ccca
    v_ccce = mr_adc.v2e.ccce
    v_cace = mr_adc.v2e.cace
    v_caec = mr_adc.v2e.caec
    v_caca = mr_adc.v2e.caca
    v_caac = mr_adc.v2e.caac
    v_ceaa = mr_adc.v2e.ceaa

    v_ceae = mr_adc.v2e.ceae
    v_ceea = mr_adc.v2e.ceea
    v_cece = mr_adc.v2e.cece
    v_ceec = mr_adc.v2e.ceec
    v_ceee = mr_adc.v2e.ceee
    v_aeae = mr_adc.v2e.aeae
    v_aeea = mr_adc.v2e.aeea
    v_aeee = mr_adc.v2e.aeee

    # Amplitudes
    t1_ce, t1_ca, t1_ae, t1_caea, t1_caae, t1_caaa, t1_aaea, t1_aaae, t1_ccee, t1_ccea, t1_caee, t1_ccaa, t1_aaee = t1_amp

    V1  = einsum('xA,Ix->IA', h_ae, t1_ca, optimize = einsum_type)
    V1 -= einsum('Ix,xA->IA', h_ca, t1_ae, optimize = einsum_type)
    V1 += 2 * einsum('ix,IiAx->IA', h_ca, t1_ccea, optimize = einsum_type)
    V1 -= einsum('ix,iIAx->IA', h_ca, t1_ccea, optimize = einsum_type)
    V1 += 2 * einsum('ia,IiAa->IA', h_ce, t1_ccee, optimize = einsum_type)
    V1 -= einsum('ia,iIAa->IA', h_ce, t1_ccee, optimize = einsum_type)
    V1 -= 2 * einsum('ijAx,ijIx->IA', t1_ccea, v_ccca, optimize = einsum_type)
    V1 += einsum('ijAx,jiIx->IA', t1_ccea, v_ccca, optimize = einsum_type)
    V1 -= 2 * einsum('ijAa,ijIa->IA', t1_ccee, v_ccce, optimize = einsum_type)
    V1 += einsum('ijAa,jiIa->IA', t1_ccee, v_ccce, optimize = einsum_type)
    V1 += 2 * einsum('Iixy,iAyx->IA', t1_ccaa, v_ceaa, optimize = einsum_type)
    V1 -= einsum('Iixy,iAxy->IA', t1_ccaa, v_ceaa, optimize = einsum_type)
    V1 += 2 * einsum('Iiax,iAxa->IA', t1_ccea, v_ceae, optimize = einsum_type)
    V1 -= einsum('Iiax,iAax->IA', t1_ccea, v_ceea, optimize = einsum_type)
    V1 += 2 * einsum('Iiab,iAba->IA', t1_ccee, v_ceee, optimize = einsum_type)
    V1 -= einsum('Iiab,iAab->IA', t1_ccee, v_ceee, optimize = einsum_type)
    V1 += 2 * einsum('ix,iIxA->IA', t1_ca, v_ccae, optimize = einsum_type)
    V1 -= einsum('ix,IixA->IA', t1_ca, v_ccae, optimize = einsum_type)
    V1 -= einsum('iIax,iAxa->IA', t1_ccea, v_ceae, optimize = einsum_type)
    V1 += 2 * einsum('iIax,iAax->IA', t1_ccea, v_ceea, optimize = einsum_type)
    V1 += 2 * einsum('ix,IxAi->IA', t1_ca, v_caec, optimize = einsum_type)
    V1 -= einsum('ix,IxiA->IA', t1_ca, v_cace, optimize = einsum_type)
    V1 += 2 * einsum('ia,iAaI->IA', t1_ce, v_ceec, optimize = einsum_type)
    V1 -= einsum('ia,iAIa->IA', t1_ce, v_cece, optimize = einsum_type)
    V1 += 2 * einsum('ia,IiAa->IA', t1_ce, v_ccee, optimize = einsum_type)
    V1 -= einsum('ia,iIAa->IA', t1_ce, v_ccee, optimize = einsum_type)
    V1 += 1/2 * einsum('A,xA,Ix->IA', e_extern, t1_ae, t1_ca, optimize = einsum_type)
    V1 += einsum('A,IiAx,ix->IA', e_extern, t1_ccea, t1_ca, optimize = einsum_type)
    V1 -= 1/2 * einsum('A,iIAx,ix->IA', e_extern, t1_ccea, t1_ca, optimize = einsum_type)
    V1 += einsum('A,IiAa,ia->IA', e_extern, t1_ccee, t1_ce, optimize = einsum_type)
    V1 -= 1/2 * einsum('A,iIAa,ia->IA', e_extern, t1_ccee, t1_ce, optimize = einsum_type)
    V1 += 1/2 * einsum('I,xA,Ix->IA', e_core, t1_ae, t1_ca, optimize = einsum_type)
    V1 -= einsum('I,IiAx,ix->IA', e_core, t1_ccea, t1_ca, optimize = einsum_type)
    V1 += 1/2 * einsum('I,iIAx,ix->IA', e_core, t1_ccea, t1_ca, optimize = einsum_type)
    V1 -= einsum('I,IiAa,ia->IA', e_core, t1_ccee, t1_ce, optimize = einsum_type)
    V1 += 1/2 * einsum('I,iIAa,ia->IA', e_core, t1_ccee, t1_ce, optimize = einsum_type)
    V1 -= einsum('i,IiAx,ix->IA', e_core, t1_ccea, t1_ca, optimize = einsum_type)
    V1 += 1/2 * einsum('i,iIAx,ix->IA', e_core, t1_ccea, t1_ca, optimize = einsum_type)
    V1 -= einsum('i,IiAa,ia->IA', e_core, t1_ccee, t1_ce, optimize = einsum_type)
    V1 += 1/2 * einsum('i,iIAa,ia->IA', e_core, t1_ccee, t1_ce, optimize = einsum_type)
    V1 -= einsum('i,ix,IiAx->IA', e_core, t1_ca, t1_ccea, optimize = einsum_type)
    V1 += 1/2 * einsum('i,ix,iIAx->IA', e_core, t1_ca, t1_ccea, optimize = einsum_type)
    V1 -= einsum('i,ia,IiAa->IA', e_core, t1_ce, t1_ccee, optimize = einsum_type)
    V1 += 1/2 * einsum('i,ia,iIAa->IA', e_core, t1_ce, t1_ccee, optimize = einsum_type)
    V1 += 2 * einsum('a,ia,IiAa->IA', e_extern, t1_ce, t1_ccee, optimize = einsum_type)
    V1 -= einsum('a,ia,iIAa->IA', e_extern, t1_ce, t1_ccee, optimize = einsum_type)
    V1 += einsum('xA,Iyxz,zy->IA', h_ae, t1_caaa, rdm_ca, optimize = einsum_type)
    V1 -= 1/2 * einsum('xA,Iyzx,zy->IA', h_ae, t1_caaa, rdm_ca, optimize = einsum_type)
    V1 -= einsum('Ix,xyAz,zy->IA', h_ca, t1_aaea, rdm_ca, optimize = einsum_type)
    V1 += 1/2 * einsum('Ix,yxAz,zy->IA', h_ca, t1_aaea, rdm_ca, optimize = einsum_type)
    V1 -= einsum('ix,IiAy,yx->IA', h_ca, t1_ccea, rdm_ca, optimize = einsum_type)
    V1 += 1/2 * einsum('ix,iIAy,yx->IA', h_ca, t1_ccea, rdm_ca, optimize = einsum_type)
    V1 += einsum('xa,IyAa,xy->IA', h_ae, t1_caee, rdm_ca, optimize = einsum_type)
    V1 -= 1/2 * einsum('xa,IyaA,xy->IA', h_ae, t1_caee, rdm_ca, optimize = einsum_type)
    V1 -= einsum('xy,xA,Iy->IA', h_aa, t1_ae, t1_ca, optimize = einsum_type)
    V1 += 2 * einsum('xy,IiAx,iy->IA', h_aa, t1_ccea, t1_ca, optimize = einsum_type)
    V1 -= einsum('xy,iIAx,iy->IA', h_aa, t1_ccea, t1_ca, optimize = einsum_type)
    V1 -= einsum('xA,Iyxz,zy->IA', t1_ae, v_caaa, rdm_ca, optimize = einsum_type)
    V1 += 1/2 * einsum('xA,Iyzx,zy->IA', t1_ae, v_caaa, rdm_ca, optimize = einsum_type)
    V1 += 2 * einsum('IiAx,iyxz,yz->IA', t1_ccea, v_caaa, rdm_ca, optimize = einsum_type)
    V1 -= einsum('IiAx,iyzx,yz->IA', t1_ccea, v_caaa, rdm_ca, optimize = einsum_type)
    V1 -= einsum('IiAx,iyzw,xyzw->IA', t1_ccea, v_caaa, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('xyAz,Izwu,xywu->IA', t1_aaea, v_caaa, rdm_ccaa, optimize = einsum_type)
    V1 -= einsum('xyAz,Iwxy,zw->IA', t1_aaea, v_caaa, rdm_ca, optimize = einsum_type)
    V1 -= einsum('xyAz,Iwxu,zuyw->IA', t1_aaea, v_caaa, rdm_ccaa, optimize = einsum_type)
    V1 += 1/2 * einsum('xyAz,Iwyx,zw->IA', t1_aaea, v_caaa, rdm_ca, optimize = einsum_type)
    V1 += 1/2 * einsum('xyAz,Iwyu,zuxw->IA', t1_aaea, v_caaa, rdm_ccaa, optimize = einsum_type)
    V1 += 1/2 * einsum('xyAz,Iwux,zuyw->IA', t1_aaea, v_caaa, rdm_ccaa, optimize = einsum_type)
    V1 += 1/2 * einsum('xyAz,Iwuy,zuwx->IA', t1_aaea, v_caaa, rdm_ccaa, optimize = einsum_type)
    V1 -= einsum('iIAx,iyxz,yz->IA', t1_ccea, v_caaa, rdm_ca, optimize = einsum_type)
    V1 += 1/2 * einsum('iIAx,iyzx,yz->IA', t1_ccea, v_caaa, rdm_ca, optimize = einsum_type)
    V1 += 1/2 * einsum('iIAx,iyzw,xyzw->IA', t1_ccea, v_caaa, rdm_ccaa, optimize = einsum_type)
    V1 += einsum('ijAx,ijIy,xy->IA', t1_ccea, v_ccca, rdm_ca, optimize = einsum_type)
    V1 -= 1/2 * einsum('ijAx,jiIy,xy->IA', t1_ccea, v_ccca, rdm_ca, optimize = einsum_type)
    V1 += einsum('iA,Ixiy,xy->IA', t1_ce, v_caca, rdm_ca, optimize = einsum_type)
    V1 -= einsum('iA,Ixiy,yx->IA', t1_ce, v_caca, rdm_ca, optimize = einsum_type)
    V1 -= 1/2 * einsum('iA,Ixyi,xy->IA', t1_ce, v_caac, rdm_ca, optimize = einsum_type)
    V1 += 1/2 * einsum('iA,Ixyi,yx->IA', t1_ce, v_caac, rdm_ca, optimize = einsum_type)
    V1 += einsum('IxAa,yzwa,xwzy->IA', t1_caee, v_aaae, rdm_ccaa, optimize = einsum_type)
    V1 += 2 * einsum('IiAa,ixay,xy->IA', t1_ccee, v_caea, rdm_ca, optimize = einsum_type)
    V1 -= einsum('IiAa,ixya,xy->IA', t1_ccee, v_caae, rdm_ca, optimize = einsum_type)
    V1 -= 1/2 * einsum('xyAa,Iazw,xyzw->IA', t1_aaee, v_ceaa, rdm_ccaa, optimize = einsum_type)
    V1 -= einsum('iIAa,ixay,xy->IA', t1_ccee, v_caea, rdm_ca, optimize = einsum_type)
    V1 += 1/2 * einsum('iIAa,ixya,xy->IA', t1_ccee, v_caae, rdm_ca, optimize = einsum_type)
    V1 -= einsum('ixAa,iyIa,yx->IA', t1_caee, v_cace, rdm_ca, optimize = einsum_type)
    V1 += 1/2 * einsum('ixAa,iyaI,yx->IA', t1_caee, v_caec, rdm_ca, optimize = einsum_type)
    V1 += einsum('Ix,yxzA,zy->IA', t1_ca, v_aaae, rdm_ca, optimize = einsum_type)
    V1 -= 1/2 * einsum('Ix,xyzA,zy->IA', t1_ca, v_aaae, rdm_ca, optimize = einsum_type)
    V1 += 1/2 * einsum('Ixyz,wuxA,yzuw->IA', t1_caaa, v_aaae, rdm_ccaa, optimize = einsum_type)
    V1 += einsum('Ixyz,zywA,wx->IA', t1_caaa, v_aaae, rdm_ca, optimize = einsum_type)
    V1 += einsum('Ixyz,wyuA,xwzu->IA', t1_caaa, v_aaae, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('Ixyz,yzwA,wx->IA', t1_caaa, v_aaae, rdm_ca, optimize = einsum_type)
    V1 -= 1/2 * einsum('Ixyz,wzuA,xwyu->IA', t1_caaa, v_aaae, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('Ixyz,ywuA,xwzu->IA', t1_caaa, v_aaae, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('Ixyz,zwuA,xwuy->IA', t1_caaa, v_aaae, rdm_ccaa, optimize = einsum_type)
    V1 += einsum('Ixya,zAay,zx->IA', t1_caae, v_aeea, rdm_ca, optimize = einsum_type)
    V1 -= 1/2 * einsum('Ixya,zAya,zx->IA', t1_caae, v_aeae, rdm_ca, optimize = einsum_type)
    V1 -= 1/2 * einsum('Ixya,zAwa,xwyz->IA', t1_caae, v_aeae, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('Ixya,zAaw,xwzy->IA', t1_caae, v_aeea, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('IxaA,yzwa,xwzy->IA', t1_caee, v_aaae, rdm_ccaa, optimize = einsum_type)
    V1 += einsum('Ixay,zAya,zx->IA', t1_caea, v_aeae, rdm_ca, optimize = einsum_type)
    V1 += einsum('Ixay,zAwa,xwyz->IA', t1_caea, v_aeae, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('Ixay,zAay,zx->IA', t1_caea, v_aeea, rdm_ca, optimize = einsum_type)
    V1 -= 1/2 * einsum('Ixay,zAaw,xwyz->IA', t1_caea, v_aeea, rdm_ccaa, optimize = einsum_type)
    V1 += einsum('Ixab,yAba,yx->IA', t1_caee, v_aeee, rdm_ca, optimize = einsum_type)
    V1 -= 1/2 * einsum('Ixab,yAab,yx->IA', t1_caee, v_aeee, rdm_ca, optimize = einsum_type)
    V1 -= einsum('Iixy,iAzx,yz->IA', t1_ccaa, v_ceaa, rdm_ca, optimize = einsum_type)
    V1 += 1/2 * einsum('Iixy,iAzy,xz->IA', t1_ccaa, v_ceaa, rdm_ca, optimize = einsum_type)
    V1 += 1/2 * einsum('Iixy,iAxz,yz->IA', t1_ccaa, v_ceaa, rdm_ca, optimize = einsum_type)
    V1 -= einsum('Iixy,iAyz,xz->IA', t1_ccaa, v_ceaa, rdm_ca, optimize = einsum_type)
    V1 += 1/2 * einsum('Iixy,iAzw,xywz->IA', t1_ccaa, v_ceaa, rdm_ccaa, optimize = einsum_type)
    V1 -= einsum('Iiax,iAya,xy->IA', t1_ccea, v_ceae, rdm_ca, optimize = einsum_type)
    V1 += 1/2 * einsum('Iiax,iAay,xy->IA', t1_ccea, v_ceea, rdm_ca, optimize = einsum_type)
    V1 += einsum('Ia,xAya,xy->IA', t1_ce, v_aeae, rdm_ca, optimize = einsum_type)
    V1 -= einsum('Ia,xAya,yx->IA', t1_ce, v_aeae, rdm_ca, optimize = einsum_type)
    V1 -= 1/2 * einsum('Ia,xAay,xy->IA', t1_ce, v_aeea, rdm_ca, optimize = einsum_type)
    V1 += 1/2 * einsum('Ia,xAay,yx->IA', t1_ce, v_aeea, rdm_ca, optimize = einsum_type)
    V1 += 2 * einsum('ixyz,iIyA,xz->IA', t1_caaa, v_ccae, rdm_ca, optimize = einsum_type)
    V1 -= einsum('ixyz,IiyA,xz->IA', t1_caaa, v_ccae, rdm_ca, optimize = einsum_type)
    V1 -= einsum('ixyz,iIzA,xy->IA', t1_caaa, v_ccae, rdm_ca, optimize = einsum_type)
    V1 += 1/2 * einsum('ixyz,IizA,xy->IA', t1_caaa, v_ccae, rdm_ca, optimize = einsum_type)
    V1 += 1/2 * einsum('iIax,iAya,xy->IA', t1_ccea, v_ceae, rdm_ca, optimize = einsum_type)
    V1 -= einsum('iIax,iAay,xy->IA', t1_ccea, v_ceea, rdm_ca, optimize = einsum_type)
    V1 -= einsum('ix,IyAi,xy->IA', t1_ca, v_caec, rdm_ca, optimize = einsum_type)
    V1 += 1/2 * einsum('ix,IyiA,xy->IA', t1_ca, v_cace, rdm_ca, optimize = einsum_type)
    V1 -= einsum('ix,iIyA,yx->IA', t1_ca, v_ccae, rdm_ca, optimize = einsum_type)
    V1 += 1/2 * einsum('ix,IiyA,yx->IA', t1_ca, v_ccae, rdm_ca, optimize = einsum_type)
    V1 -= einsum('ixAy,Iyiz,zx->IA', t1_caea, v_caca, rdm_ca, optimize = einsum_type)
    V1 += 1/2 * einsum('ixAy,Iyzi,zx->IA', t1_caea, v_caac, rdm_ca, optimize = einsum_type)
    V1 -= einsum('ixAy,Iziw,xzyw->IA', t1_caea, v_caca, rdm_ccaa, optimize = einsum_type)
    V1 += 1/2 * einsum('ixAy,Izwi,xzyw->IA', t1_caea, v_caac, rdm_ccaa, optimize = einsum_type)
    V1 += 1/2 * einsum('ixyA,Iyiz,zx->IA', t1_caae, v_caca, rdm_ca, optimize = einsum_type)
    V1 -= einsum('ixyA,Iyzi,zx->IA', t1_caae, v_caac, rdm_ca, optimize = einsum_type)
    V1 += 1/2 * einsum('ixyA,Iziw,xzyw->IA', t1_caae, v_caca, rdm_ccaa, optimize = einsum_type)
    V1 += 1/2 * einsum('ixyA,Izwi,xzwy->IA', t1_caae, v_caac, rdm_ccaa, optimize = einsum_type)
    V1 += 2 * einsum('ixyz,IyAi,zx->IA', t1_caaa, v_caec, rdm_ca, optimize = einsum_type)
    V1 -= einsum('ixyz,IzAi,yx->IA', t1_caaa, v_caec, rdm_ca, optimize = einsum_type)
    V1 -= einsum('ixyz,IwAi,xwzy->IA', t1_caaa, v_caec, rdm_ccaa, optimize = einsum_type)
    V1 -= einsum('ixyz,IyiA,zx->IA', t1_caaa, v_cace, rdm_ca, optimize = einsum_type)
    V1 += 1/2 * einsum('ixyz,IziA,yx->IA', t1_caaa, v_cace, rdm_ca, optimize = einsum_type)
    V1 += 1/2 * einsum('ixyz,IwiA,xwzy->IA', t1_caaa, v_cace, rdm_ccaa, optimize = einsum_type)
    V1 -= einsum('ixyz,iIwA,xwzy->IA', t1_caaa, v_ccae, rdm_ccaa, optimize = einsum_type)
    V1 += 1/2 * einsum('ixyz,IiwA,xwzy->IA', t1_caaa, v_ccae, rdm_ccaa, optimize = einsum_type)
    V1 -= einsum('ixya,iAaI,yx->IA', t1_caae, v_ceec, rdm_ca, optimize = einsum_type)
    V1 += 1/2 * einsum('ixya,iAIa,yx->IA', t1_caae, v_cece, rdm_ca, optimize = einsum_type)
    V1 -= einsum('ixya,IiAa,xy->IA', t1_caae, v_ccee, rdm_ca, optimize = einsum_type)
    V1 += 1/2 * einsum('ixya,iIAa,xy->IA', t1_caae, v_ccee, rdm_ca, optimize = einsum_type)
    V1 += 2 * einsum('ixay,iAaI,yx->IA', t1_caea, v_ceec, rdm_ca, optimize = einsum_type)
    V1 -= einsum('ixay,iAIa,yx->IA', t1_caea, v_cece, rdm_ca, optimize = einsum_type)
    V1 += 2 * einsum('ixay,IiAa,xy->IA', t1_caea, v_ccee, rdm_ca, optimize = einsum_type)
    V1 -= einsum('ixay,iIAa,xy->IA', t1_caea, v_ccee, rdm_ca, optimize = einsum_type)
    V1 += 1/2 * einsum('ixaA,iyIa,yx->IA', t1_caee, v_cace, rdm_ca, optimize = einsum_type)
    V1 -= einsum('ixaA,iyaI,yx->IA', t1_caee, v_caec, rdm_ca, optimize = einsum_type)
    V1 += einsum('xa,IyAa,xy->IA', t1_ae, v_caee, rdm_ca, optimize = einsum_type)
    V1 -= 1/2 * einsum('xa,IyaA,xy->IA', t1_ae, v_caee, rdm_ca, optimize = einsum_type)
    V1 += einsum('xa,IaAy,yx->IA', t1_ae, v_ceea, rdm_ca, optimize = einsum_type)
    V1 -= 1/2 * einsum('xa,IayA,yx->IA', t1_ae, v_ceae, rdm_ca, optimize = einsum_type)
    V1 += einsum('xyaz,IwAa,zwyx->IA', t1_aaea, v_caee, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('xyaz,IwaA,zwyx->IA', t1_aaea, v_caee, rdm_ccaa, optimize = einsum_type)
    V1 += einsum('xyaz,IaAw,zwyx->IA', t1_aaea, v_ceea, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('xyaz,IawA,zwyx->IA', t1_aaea, v_ceae, rdm_ccaa, optimize = einsum_type)
    V1 += 1/2 * einsum('A,xA,Iyxz,zy->IA', e_extern, t1_ae, t1_caaa, rdm_ca, optimize = einsum_type)
    V1 -= 1/4 * einsum('A,xA,Iyzx,zy->IA', e_extern, t1_ae, t1_caaa, rdm_ca, optimize = einsum_type)
    V1 += einsum('A,IiAx,iyxz,yz->IA', e_extern, t1_ccea, t1_caaa, rdm_ca, optimize = einsum_type)
    V1 -= 1/2 * einsum('A,IiAx,iy,xy->IA', e_extern, t1_ccea, t1_ca, rdm_ca, optimize = einsum_type)
    V1 -= 1/2 * einsum('A,IiAx,iyzx,yz->IA', e_extern, t1_ccea, t1_caaa, rdm_ca, optimize = einsum_type)
    V1 -= 1/2 * einsum('A,IiAx,iyzw,xyzw->IA', e_extern, t1_ccea, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 += 1/4 * einsum('A,xyAz,Izwu,xywu->IA', e_extern, t1_aaea, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 += 1/2 * einsum('A,xyAz,Ix,zy->IA', e_extern, t1_aaea, t1_ca, rdm_ca, optimize = einsum_type)
    V1 -= 1/4 * einsum('A,xyAz,Iy,zx->IA', e_extern, t1_aaea, t1_ca, rdm_ca, optimize = einsum_type)
    V1 += 1/2 * einsum('A,xyAz,Iwxy,zw->IA', e_extern, t1_aaea, t1_caaa, rdm_ca, optimize = einsum_type)
    V1 += 1/2 * einsum('A,xyAz,Iwxu,zuyw->IA', e_extern, t1_aaea, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('A,xyAz,Iwyx,zw->IA', e_extern, t1_aaea, t1_caaa, rdm_ca, optimize = einsum_type)
    V1 -= 1/4 * einsum('A,xyAz,Iwyu,zuxw->IA', e_extern, t1_aaea, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('A,xyAz,Iwux,zuyw->IA', e_extern, t1_aaea, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('A,xyAz,Iwuy,zuwx->IA', e_extern, t1_aaea, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('A,iIAx,iyxz,yz->IA', e_extern, t1_ccea, t1_caaa, rdm_ca, optimize = einsum_type)
    V1 += 1/4 * einsum('A,iIAx,iy,xy->IA', e_extern, t1_ccea, t1_ca, rdm_ca, optimize = einsum_type)
    V1 += 1/4 * einsum('A,iIAx,iyzx,yz->IA', e_extern, t1_ccea, t1_caaa, rdm_ca, optimize = einsum_type)
    V1 += 1/4 * einsum('A,iIAx,iyzw,xyzw->IA', e_extern, t1_ccea, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 += 1/2 * einsum('A,IxAa,ya,yx->IA', e_extern, t1_caee, t1_ae, rdm_ca, optimize = einsum_type)
    V1 += 1/2 * einsum('A,IxAa,yzaw,xwyz->IA', e_extern, t1_caee, t1_aaea, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('A,IxaA,ya,yx->IA', e_extern, t1_caee, t1_ae, rdm_ca, optimize = einsum_type)
    V1 -= 1/4 * einsum('A,IxaA,yzaw,xwyz->IA', e_extern, t1_caee, t1_aaea, rdm_ccaa, optimize = einsum_type)
    V1 += einsum('A,IiAa,ixay,xy->IA', e_extern, t1_ccee, t1_caea, rdm_ca, optimize = einsum_type)
    V1 -= 1/2 * einsum('A,IiAa,ixya,xy->IA', e_extern, t1_ccee, t1_caae, rdm_ca, optimize = einsum_type)
    V1 -= 1/2 * einsum('A,IiaA,ixay,xy->IA', e_extern, t1_ccee, t1_caea, rdm_ca, optimize = einsum_type)
    V1 += 1/4 * einsum('A,IiaA,ixya,xy->IA', e_extern, t1_ccee, t1_caae, rdm_ca, optimize = einsum_type)
    V1 += 1/2 * einsum('I,xA,Iyxz,zy->IA', e_core, t1_ae, t1_caaa, rdm_ca, optimize = einsum_type)
    V1 -= 1/4 * einsum('I,xA,Iyzx,zy->IA', e_core, t1_ae, t1_caaa, rdm_ca, optimize = einsum_type)
    V1 -= einsum('I,IiAx,iyxz,yz->IA', e_core, t1_ccea, t1_caaa, rdm_ca, optimize = einsum_type)
    V1 += 1/2 * einsum('I,IiAx,iy,xy->IA', e_core, t1_ccea, t1_ca, rdm_ca, optimize = einsum_type)
    V1 += 1/2 * einsum('I,IiAx,iyzx,yz->IA', e_core, t1_ccea, t1_caaa, rdm_ca, optimize = einsum_type)
    V1 += 1/2 * einsum('I,IiAx,iyzw,xyzw->IA', e_core, t1_ccea, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 += 1/4 * einsum('I,xyAz,Izwu,xywu->IA', e_core, t1_aaea, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 += 1/2 * einsum('I,xyAz,Ix,zy->IA', e_core, t1_aaea, t1_ca, rdm_ca, optimize = einsum_type)
    V1 -= 1/4 * einsum('I,xyAz,Iy,zx->IA', e_core, t1_aaea, t1_ca, rdm_ca, optimize = einsum_type)
    V1 += 1/2 * einsum('I,xyAz,Iwxy,zw->IA', e_core, t1_aaea, t1_caaa, rdm_ca, optimize = einsum_type)
    V1 += 1/2 * einsum('I,xyAz,Iwxu,zuyw->IA', e_core, t1_aaea, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('I,xyAz,Iwyx,zw->IA', e_core, t1_aaea, t1_caaa, rdm_ca, optimize = einsum_type)
    V1 -= 1/4 * einsum('I,xyAz,Iwyu,zuxw->IA', e_core, t1_aaea, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('I,xyAz,Iwux,zuyw->IA', e_core, t1_aaea, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('I,xyAz,Iwuy,zuwx->IA', e_core, t1_aaea, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 += 1/2 * einsum('I,iIAx,iyxz,yz->IA', e_core, t1_ccea, t1_caaa, rdm_ca, optimize = einsum_type)
    V1 -= 1/4 * einsum('I,iIAx,iy,xy->IA', e_core, t1_ccea, t1_ca, rdm_ca, optimize = einsum_type)
    V1 -= 1/4 * einsum('I,iIAx,iyzx,yz->IA', e_core, t1_ccea, t1_caaa, rdm_ca, optimize = einsum_type)
    V1 -= 1/4 * einsum('I,iIAx,iyzw,xyzw->IA', e_core, t1_ccea, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('I,IxAa,ya,yx->IA', e_core, t1_caee, t1_ae, rdm_ca, optimize = einsum_type)
    V1 -= 1/2 * einsum('I,IxAa,yzaw,xwyz->IA', e_core, t1_caee, t1_aaea, rdm_ccaa, optimize = einsum_type)
    V1 += 1/4 * einsum('I,IxaA,ya,yx->IA', e_core, t1_caee, t1_ae, rdm_ca, optimize = einsum_type)
    V1 += 1/4 * einsum('I,IxaA,yzaw,xwyz->IA', e_core, t1_caee, t1_aaea, rdm_ccaa, optimize = einsum_type)
    V1 -= einsum('I,IiAa,ixay,xy->IA', e_core, t1_ccee, t1_caea, rdm_ca, optimize = einsum_type)
    V1 += 1/2 * einsum('I,IiAa,ixya,xy->IA', e_core, t1_ccee, t1_caae, rdm_ca, optimize = einsum_type)
    V1 += 1/2 * einsum('I,IiaA,ixay,xy->IA', e_core, t1_ccee, t1_caea, rdm_ca, optimize = einsum_type)
    V1 -= 1/4 * einsum('I,IiaA,ixya,xy->IA', e_core, t1_ccee, t1_caae, rdm_ca, optimize = einsum_type)
    V1 += einsum('i,IiAx,iy,xy->IA', e_core, t1_ccea, t1_ca, rdm_ca, optimize = einsum_type)
    V1 -= einsum('i,IiAx,iyxz,yz->IA', e_core, t1_ccea, t1_caaa, rdm_ca, optimize = einsum_type)
    V1 += 1/2 * einsum('i,IiAx,iyzx,yz->IA', e_core, t1_ccea, t1_caaa, rdm_ca, optimize = einsum_type)
    V1 += einsum('i,IiAx,iyzw,xyzw->IA', e_core, t1_ccea, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('i,iIAx,iy,xy->IA', e_core, t1_ccea, t1_ca, rdm_ca, optimize = einsum_type)
    V1 += 1/2 * einsum('i,iIAx,iyxz,yz->IA', e_core, t1_ccea, t1_caaa, rdm_ca, optimize = einsum_type)
    V1 -= 1/4 * einsum('i,iIAx,iyzx,yz->IA', e_core, t1_ccea, t1_caaa, rdm_ca, optimize = einsum_type)
    V1 -= 1/2 * einsum('i,iIAx,iyzw,xyzw->IA', e_core, t1_ccea, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 -= einsum('i,IiAa,ixay,xy->IA', e_core, t1_ccee, t1_caea, rdm_ca, optimize = einsum_type)
    V1 += 1/2 * einsum('i,IiAa,ixya,xy->IA', e_core, t1_ccee, t1_caae, rdm_ca, optimize = einsum_type)
    V1 += 1/2 * einsum('i,IiaA,ixay,xy->IA', e_core, t1_ccee, t1_caea, rdm_ca, optimize = einsum_type)
    V1 -= 1/4 * einsum('i,IiaA,ixya,xy->IA', e_core, t1_ccee, t1_caae, rdm_ca, optimize = einsum_type)
    V1 -= einsum('i,ixyz,IiAy,xz->IA', e_core, t1_caaa, t1_ccea, rdm_ca, optimize = einsum_type)
    V1 += 1/2 * einsum('i,ixyz,iIAy,xz->IA', e_core, t1_caaa, t1_ccea, rdm_ca, optimize = einsum_type)
    V1 += 1/2 * einsum('i,ixyz,IiAz,xy->IA', e_core, t1_caaa, t1_ccea, rdm_ca, optimize = einsum_type)
    V1 -= 1/4 * einsum('i,ixyz,iIAz,xy->IA', e_core, t1_caaa, t1_ccea, rdm_ca, optimize = einsum_type)
    V1 += 1/2 * einsum('i,ixya,IiAa,xy->IA', e_core, t1_caae, t1_ccee, rdm_ca, optimize = einsum_type)
    V1 -= 1/4 * einsum('i,ixya,iIAa,xy->IA', e_core, t1_caae, t1_ccee, rdm_ca, optimize = einsum_type)
    V1 -= einsum('i,ixay,IiAa,xy->IA', e_core, t1_caea, t1_ccee, rdm_ca, optimize = einsum_type)
    V1 += 1/2 * einsum('i,ixay,iIAa,xy->IA', e_core, t1_caea, t1_ccee, rdm_ca, optimize = einsum_type)
    V1 += einsum('a,xa,IyAa,xy->IA', e_extern, t1_ae, t1_caee, rdm_ca, optimize = einsum_type)
    V1 -= 1/2 * einsum('a,xa,IyaA,xy->IA', e_extern, t1_ae, t1_caee, rdm_ca, optimize = einsum_type)
    V1 += einsum('a,xyaz,IwAa,zwyx->IA', e_extern, t1_aaea, t1_caee, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('a,xyaz,IwaA,zwyx->IA', e_extern, t1_aaea, t1_caee, rdm_ccaa, optimize = einsum_type)
    V1 -= einsum('a,ixya,IiAa,xy->IA', e_extern, t1_caae, t1_ccee, rdm_ca, optimize = einsum_type)
    V1 += 1/2 * einsum('a,ixya,iIAa,xy->IA', e_extern, t1_caae, t1_ccee, rdm_ca, optimize = einsum_type)
    V1 += 2 * einsum('a,ixay,IiAa,xy->IA', e_extern, t1_caea, t1_ccee, rdm_ca, optimize = einsum_type)
    V1 -= einsum('a,ixay,iIAa,xy->IA', e_extern, t1_caea, t1_ccee, rdm_ca, optimize = einsum_type)
    V1 -= einsum('xy,xA,Izyw,wz->IA', h_aa, t1_ae, t1_caaa, rdm_ca, optimize = einsum_type)
    V1 += 1/2 * einsum('xy,xA,Izwy,wz->IA', h_aa, t1_ae, t1_caaa, rdm_ca, optimize = einsum_type)
    V1 += 2 * einsum('xy,IiAx,izyw,zw->IA', h_aa, t1_ccea, t1_caaa, rdm_ca, optimize = einsum_type)
    V1 -= 1/2 * einsum('xy,IiAx,iz,yz->IA', h_aa, t1_ccea, t1_ca, rdm_ca, optimize = einsum_type)
    V1 -= einsum('xy,IiAx,izwy,zw->IA', h_aa, t1_ccea, t1_caaa, rdm_ca, optimize = einsum_type)
    V1 -= 1/2 * einsum('xy,IiAx,izwu,yzwu->IA', h_aa, t1_ccea, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 += 1/2 * einsum('xy,zwAx,Iyuv,zwuv->IA', h_aa, t1_aaea, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 += 1/2 * einsum('xy,zwAx,Iz,yw->IA', h_aa, t1_aaea, t1_ca, rdm_ca, optimize = einsum_type)
    V1 -= 1/4 * einsum('xy,zwAx,Iw,yz->IA', h_aa, t1_aaea, t1_ca, rdm_ca, optimize = einsum_type)
    V1 += 1/2 * einsum('xy,zwAx,Iuzw,yu->IA', h_aa, t1_aaea, t1_caaa, rdm_ca, optimize = einsum_type)
    V1 += 1/2 * einsum('xy,zwAx,Iuzv,yvwu->IA', h_aa, t1_aaea, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xy,zwAx,Iuwz,yu->IA', h_aa, t1_aaea, t1_caaa, rdm_ca, optimize = einsum_type)
    V1 -= 1/4 * einsum('xy,zwAx,Iuwv,yvzu->IA', h_aa, t1_aaea, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xy,zwAx,Iuvz,yvwu->IA', h_aa, t1_aaea, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xy,zwAx,Iuvw,yvuz->IA', h_aa, t1_aaea, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 -= einsum('xy,iIAx,izyw,zw->IA', h_aa, t1_ccea, t1_caaa, rdm_ca, optimize = einsum_type)
    V1 += 1/4 * einsum('xy,iIAx,iz,yz->IA', h_aa, t1_ccea, t1_ca, rdm_ca, optimize = einsum_type)
    V1 += 1/2 * einsum('xy,iIAx,izwy,zw->IA', h_aa, t1_ccea, t1_caaa, rdm_ca, optimize = einsum_type)
    V1 += 1/4 * einsum('xy,iIAx,izwu,yzwu->IA', h_aa, t1_ccea, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 -= einsum('xy,xzAw,Iy,wz->IA', h_aa, t1_aaea, t1_ca, rdm_ca, optimize = einsum_type)
    V1 -= 1/4 * einsum('xy,xzAw,Iwuv,yzuv->IA', h_aa, t1_aaea, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 += 1/4 * einsum('xy,xzAw,Iz,wy->IA', h_aa, t1_aaea, t1_ca, rdm_ca, optimize = einsum_type)
    V1 -= einsum('xy,xzAw,Iuyz,wu->IA', h_aa, t1_aaea, t1_caaa, rdm_ca, optimize = einsum_type)
    V1 -= einsum('xy,xzAw,Iuyv,wvzu->IA', h_aa, t1_aaea, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 += 1/2 * einsum('xy,xzAw,Iuzy,wu->IA', h_aa, t1_aaea, t1_caaa, rdm_ca, optimize = einsum_type)
    V1 += 1/4 * einsum('xy,xzAw,Iuzv,yuwv->IA', h_aa, t1_aaea, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 += 1/2 * einsum('xy,xzAw,Iuvy,wvzu->IA', h_aa, t1_aaea, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 += 1/4 * einsum('xy,xzAw,Iuvz,yuvw->IA', h_aa, t1_aaea, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 += 1/2 * einsum('xy,zxAw,Iy,wz->IA', h_aa, t1_aaea, t1_ca, rdm_ca, optimize = einsum_type)
    V1 -= 1/4 * einsum('xy,zxAw,Iwuv,yzvu->IA', h_aa, t1_aaea, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('xy,zxAw,Iz,wy->IA', h_aa, t1_aaea, t1_ca, rdm_ca, optimize = einsum_type)
    V1 += 1/2 * einsum('xy,zxAw,Iuyz,wu->IA', h_aa, t1_aaea, t1_caaa, rdm_ca, optimize = einsum_type)
    V1 += 1/2 * einsum('xy,zxAw,Iuyv,wvzu->IA', h_aa, t1_aaea, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 -= einsum('xy,zxAw,Iuzy,wu->IA', h_aa, t1_aaea, t1_caaa, rdm_ca, optimize = einsum_type)
    V1 -= 1/2 * einsum('xy,zxAw,Iuzv,yuwv->IA', h_aa, t1_aaea, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 += 1/2 * einsum('xy,zxAw,Iuvy,wvuz->IA', h_aa, t1_aaea, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 += 1/4 * einsum('xy,zxAw,Iuvz,yuwv->IA', h_aa, t1_aaea, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('xy,IxAa,za,zy->IA', h_aa, t1_caee, t1_ae, rdm_ca, optimize = einsum_type)
    V1 -= 1/2 * einsum('xy,IxAa,zwau,yuzw->IA', h_aa, t1_caee, t1_aaea, rdm_ccaa, optimize = einsum_type)
    V1 += 1/2 * einsum('xy,Ixzw,zA,wy->IA', h_aa, t1_caaa, t1_ae, rdm_ca, optimize = einsum_type)
    V1 -= 1/4 * einsum('xy,Ixzw,wA,zy->IA', h_aa, t1_caaa, t1_ae, rdm_ca, optimize = einsum_type)
    V1 += 1/2 * einsum('xy,Ixzw,zwAu,uy->IA', h_aa, t1_caaa, t1_aaea, rdm_ca, optimize = einsum_type)
    V1 += 1/2 * einsum('xy,Ixzw,zuAv,yuwv->IA', h_aa, t1_caaa, t1_aaea, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xy,Ixzw,wzAu,uy->IA', h_aa, t1_caaa, t1_aaea, rdm_ca, optimize = einsum_type)
    V1 -= 1/4 * einsum('xy,Ixzw,wuAv,yuzv->IA', h_aa, t1_caaa, t1_aaea, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xy,Ixzw,uzAv,yuwv->IA', h_aa, t1_caaa, t1_aaea, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xy,Ixzw,uwAv,yuvz->IA', h_aa, t1_caaa, t1_aaea, rdm_ccaa, optimize = einsum_type)
    V1 += 1/4 * einsum('xy,IxaA,za,zy->IA', h_aa, t1_caee, t1_ae, rdm_ca, optimize = einsum_type)
    V1 += 1/4 * einsum('xy,IxaA,zwau,yuzw->IA', h_aa, t1_caee, t1_aaea, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xy,Izxw,uvAz,ywuv->IA', h_aa, t1_caaa, t1_aaea, rdm_ccaa, optimize = einsum_type)
    V1 += 1/4 * einsum('xy,Izxw,wA,yz->IA', h_aa, t1_caaa, t1_ae, rdm_ca, optimize = einsum_type)
    V1 += 1/4 * einsum('xy,Izxw,wuAv,yvzu->IA', h_aa, t1_caaa, t1_aaea, rdm_ccaa, optimize = einsum_type)
    V1 += 1/4 * einsum('xy,Izxw,uwAv,yvuz->IA', h_aa, t1_caaa, t1_aaea, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xy,Izwx,uvAz,ywvu->IA', h_aa, t1_caaa, t1_aaea, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('xy,Izwx,wA,yz->IA', h_aa, t1_caaa, t1_ae, rdm_ca, optimize = einsum_type)
    V1 -= 1/2 * einsum('xy,Izwx,wuAv,yvzu->IA', h_aa, t1_caaa, t1_aaea, rdm_ccaa, optimize = einsum_type)
    V1 += 1/4 * einsum('xy,Izwx,uwAv,yvzu->IA', h_aa, t1_caaa, t1_aaea, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('xy,izxw,IiAw,zy->IA', h_aa, t1_caaa, t1_ccea, rdm_ca, optimize = einsum_type)
    V1 += 1/4 * einsum('xy,izxw,iIAw,zy->IA', h_aa, t1_caaa, t1_ccea, rdm_ca, optimize = einsum_type)
    V1 -= 1/2 * einsum('xy,izxw,IiAu,ywuz->IA', h_aa, t1_caaa, t1_ccea, rdm_ccaa, optimize = einsum_type)
    V1 += 1/4 * einsum('xy,izxw,iIAu,ywuz->IA', h_aa, t1_caaa, t1_ccea, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('xy,xzaw,IuAa,yzuw->IA', h_aa, t1_aaea, t1_caee, rdm_ccaa, optimize = einsum_type)
    V1 += 1/4 * einsum('xy,xzaw,IuaA,yzuw->IA', h_aa, t1_aaea, t1_caee, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('xy,ix,IiAz,zy->IA', h_aa, t1_ca, t1_ccea, rdm_ca, optimize = einsum_type)
    V1 += 1/4 * einsum('xy,ix,iIAz,zy->IA', h_aa, t1_ca, t1_ccea, rdm_ca, optimize = einsum_type)
    V1 -= einsum('xy,ixaz,IiAa,yz->IA', h_aa, t1_caea, t1_ccee, rdm_ca, optimize = einsum_type)
    V1 += 1/2 * einsum('xy,ixaz,iIAa,yz->IA', h_aa, t1_caea, t1_ccee, rdm_ca, optimize = einsum_type)
    V1 += 1/2 * einsum('xy,ixza,IiAa,yz->IA', h_aa, t1_caae, t1_ccee, rdm_ca, optimize = einsum_type)
    V1 -= 1/4 * einsum('xy,ixza,iIAa,yz->IA', h_aa, t1_caae, t1_ccee, rdm_ca, optimize = einsum_type)
    V1 -= 1/2 * einsum('xy,xa,IzAa,yz->IA', h_aa, t1_ae, t1_caee, rdm_ca, optimize = einsum_type)
    V1 += 1/4 * einsum('xy,xa,IzaA,yz->IA', h_aa, t1_ae, t1_caee, rdm_ca, optimize = einsum_type)
    V1 += einsum('xy,izax,IiAa,zy->IA', h_aa, t1_caea, t1_ccee, rdm_ca, optimize = einsum_type)
    V1 -= 1/2 * einsum('xy,izax,iIAa,zy->IA', h_aa, t1_caea, t1_ccee, rdm_ca, optimize = einsum_type)
    V1 -= 1/2 * einsum('xy,izxa,IiAa,zy->IA', h_aa, t1_caae, t1_ccee, rdm_ca, optimize = einsum_type)
    V1 += 1/4 * einsum('xy,izxa,iIAa,zy->IA', h_aa, t1_caae, t1_ccee, rdm_ca, optimize = einsum_type)
    V1 += einsum('xy,izwx,IiAw,zy->IA', h_aa, t1_caaa, t1_ccea, rdm_ca, optimize = einsum_type)
    V1 -= 1/2 * einsum('xy,izwx,iIAw,zy->IA', h_aa, t1_caaa, t1_ccea, rdm_ca, optimize = einsum_type)
    V1 -= 1/2 * einsum('xy,izwx,IiAu,ywzu->IA', h_aa, t1_caaa, t1_ccea, rdm_ccaa, optimize = einsum_type)
    V1 += 1/4 * einsum('xy,izwx,iIAu,ywzu->IA', h_aa, t1_caaa, t1_ccea, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('xy,zxaw,IuAa,yzwu->IA', h_aa, t1_aaea, t1_caee, rdm_ccaa, optimize = einsum_type)
    V1 += 1/4 * einsum('xy,zxaw,IuaA,yzwu->IA', h_aa, t1_aaea, t1_caee, rdm_ccaa, optimize = einsum_type)
    V1 -= einsum('xy,ixzw,IiAz,yw->IA', h_aa, t1_caaa, t1_ccea, rdm_ca, optimize = einsum_type)
    V1 += 1/2 * einsum('xy,ixzw,iIAz,yw->IA', h_aa, t1_caaa, t1_ccea, rdm_ca, optimize = einsum_type)
    V1 += 1/2 * einsum('xy,ixzw,IiAw,yz->IA', h_aa, t1_caaa, t1_ccea, rdm_ca, optimize = einsum_type)
    V1 -= 1/4 * einsum('xy,ixzw,iIAw,yz->IA', h_aa, t1_caaa, t1_ccea, rdm_ca, optimize = einsum_type)
    V1 += 1/2 * einsum('xy,ixzw,IiAu,yuwz->IA', h_aa, t1_caaa, t1_ccea, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xy,ixzw,iIAu,yuwz->IA', h_aa, t1_caaa, t1_ccea, rdm_ccaa, optimize = einsum_type)
    V1 += 1/2 * einsum('xy,zwax,IuAa,yuwz->IA', h_aa, t1_aaea, t1_caee, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xy,zwax,IuaA,yuwz->IA', h_aa, t1_aaea, t1_caee, rdm_ccaa, optimize = einsum_type)
    V1 += 1/2 * einsum('Ixya,zAwa,wz,yx->IA', t1_caae, v_aeae, rdm_ca, rdm_ca, optimize = einsum_type)
    V1 -= 1/4 * einsum('Ixya,zAaw,wz,yx->IA', t1_caae, v_aeea, rdm_ca, rdm_ca, optimize = einsum_type)
    V1 -= einsum('Ixay,zAwa,wz,yx->IA', t1_caea, v_aeae, rdm_ca, rdm_ca, optimize = einsum_type)
    V1 += 1/2 * einsum('Ixay,zAaw,wz,yx->IA', t1_caea, v_aeea, rdm_ca, rdm_ca, optimize = einsum_type)
    V1 += einsum('ixAy,Iziw,zw,yx->IA', t1_caea, v_caca, rdm_ca, rdm_ca, optimize = einsum_type)
    V1 -= 1/2 * einsum('ixAy,Izwi,zw,yx->IA', t1_caea, v_caac, rdm_ca, rdm_ca, optimize = einsum_type)
    V1 -= 1/2 * einsum('ixyA,Iziw,zw,yx->IA', t1_caae, v_caca, rdm_ca, rdm_ca, optimize = einsum_type)
    V1 += 1/4 * einsum('ixyA,Izwi,zw,yx->IA', t1_caae, v_caac, rdm_ca, rdm_ca, optimize = einsum_type)
    V1 -= 1/2 * einsum('xyzw,xA,Iyuv,zwuv->IA', v_aaaa, t1_ae, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 -= einsum('xyzw,xA,Iz,yw->IA', v_aaaa, t1_ae, t1_ca, rdm_ca, optimize = einsum_type)
    V1 += 1/2 * einsum('xyzw,xA,Iw,yz->IA', v_aaaa, t1_ae, t1_ca, rdm_ca, optimize = einsum_type)
    V1 -= einsum('xyzw,xA,Iuzw,yu->IA', v_aaaa, t1_ae, t1_caaa, rdm_ca, optimize = einsum_type)
    V1 -= einsum('xyzw,xA,Iuzv,yvwu->IA', v_aaaa, t1_ae, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 += 1/2 * einsum('xyzw,xA,Iuwz,yu->IA', v_aaaa, t1_ae, t1_caaa, rdm_ca, optimize = einsum_type)
    V1 += 1/2 * einsum('xyzw,xA,Iuwv,yvzu->IA', v_aaaa, t1_ae, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 += 1/2 * einsum('xyzw,xA,Iuvz,yvwu->IA', v_aaaa, t1_ae, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 += 1/2 * einsum('xyzw,xA,Iuvw,yvuz->IA', v_aaaa, t1_ae, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 += 2 * einsum('xyzw,IiAx,iuzw,uy->IA', v_aaaa, t1_ccea, t1_caaa, rdm_ca, optimize = einsum_type)
    V1 += 2 * einsum('xyzw,IiAx,iz,wy->IA', v_aaaa, t1_ccea, t1_ca, rdm_ca, optimize = einsum_type)
    V1 += 2 * einsum('xyzw,IiAx,iuzv,yvwu->IA', v_aaaa, t1_ccea, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 -= einsum('xyzw,IiAx,iuwz,uy->IA', v_aaaa, t1_ccea, t1_caaa, rdm_ca, optimize = einsum_type)
    V1 -= einsum('xyzw,IiAx,iw,zy->IA', v_aaaa, t1_ccea, t1_ca, rdm_ca, optimize = einsum_type)
    V1 -= einsum('xyzw,IiAx,iuwv,yvzu->IA', v_aaaa, t1_ccea, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 += 1/2 * einsum('xyzw,IiAx,iyuv,zwuv->IA', v_aaaa, t1_ccea, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('xyzw,IiAx,iu,yuwz->IA', v_aaaa, t1_ccea, t1_ca, rdm_ccaa, optimize = einsum_type)
    V1 -= einsum('xyzw,IiAx,iuvz,yvwu->IA', v_aaaa, t1_ccea, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 -= einsum('xyzw,IiAx,iuvw,yvuz->IA', v_aaaa, t1_ccea, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,IiAx,iuvs,zwuyvs->IA', v_aaaa, t1_ccea, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,IiAx,iuvs,zwuysv->IA', v_aaaa, t1_ccea, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 -= 5/12 * einsum('xyzw,IiAx,iuvs,zwuvys->IA', v_aaaa, t1_ccea, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,IiAx,iuvs,zwuvsy->IA', v_aaaa, t1_ccea, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,IiAx,iuvs,zwusyv->IA', v_aaaa, t1_ccea, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,IiAx,iuvs,zwusvy->IA', v_aaaa, t1_ccea, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('xyzw,uvAx,Iy,zwvu->IA', v_aaaa, t1_aaea, t1_ca, rdm_ccaa, optimize = einsum_type)
    V1 += 5/12 * einsum('xyzw,uvAx,Izst,wstyuv->IA', v_aaaa, t1_aaea, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/12 * einsum('xyzw,uvAx,Izst,wstyvu->IA', v_aaaa, t1_aaea, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/12 * einsum('xyzw,uvAx,Izst,wstuyv->IA', v_aaaa, t1_aaea, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/12 * einsum('xyzw,uvAx,Izst,wstuvy->IA', v_aaaa, t1_aaea, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/12 * einsum('xyzw,uvAx,Izst,wstvyu->IA', v_aaaa, t1_aaea, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/12 * einsum('xyzw,uvAx,Izst,wstvuy->IA', v_aaaa, t1_aaea, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/12 * einsum('xyzw,uvAx,Iwst,zstyuv->IA', v_aaaa, t1_aaea, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/12 * einsum('xyzw,uvAx,Iwst,zstyvu->IA', v_aaaa, t1_aaea, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/12 * einsum('xyzw,uvAx,Iwst,zstuyv->IA', v_aaaa, t1_aaea, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/12 * einsum('xyzw,uvAx,Iwst,zstuvy->IA', v_aaaa, t1_aaea, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/12 * einsum('xyzw,uvAx,Iwst,zstvyu->IA', v_aaaa, t1_aaea, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 5/12 * einsum('xyzw,uvAx,Iwst,zstvuy->IA', v_aaaa, t1_aaea, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/2 * einsum('xyzw,uvAx,Iu,yvwz->IA', v_aaaa, t1_aaea, t1_ca, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xyzw,uvAx,Iv,yuwz->IA', v_aaaa, t1_aaea, t1_ca, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xyzw,uvAx,Isyu,zwvs->IA', v_aaaa, t1_aaea, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xyzw,uvAx,Isyv,zwsu->IA', v_aaaa, t1_aaea, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,uvAx,Isyt,zwtuvs->IA', v_aaaa, t1_aaea, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,uvAx,Isyt,zwtusv->IA', v_aaaa, t1_aaea, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 -= 5/12 * einsum('xyzw,uvAx,Isyt,zwtvus->IA', v_aaaa, t1_aaea, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,uvAx,Isyt,zwtvsu->IA', v_aaaa, t1_aaea, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,uvAx,Isyt,zwtsuv->IA', v_aaaa, t1_aaea, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,uvAx,Isyt,zwtsvu->IA', v_aaaa, t1_aaea, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/2 * einsum('xyzw,uvAx,Isuy,zwvs->IA', v_aaaa, t1_aaea, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 += 1/2 * einsum('xyzw,uvAx,Isuv,yswz->IA', v_aaaa, t1_aaea, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/12 * einsum('xyzw,uvAx,Isut,zwtyvs->IA', v_aaaa, t1_aaea, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/12 * einsum('xyzw,uvAx,Isut,zwtysv->IA', v_aaaa, t1_aaea, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 5/12 * einsum('xyzw,uvAx,Isut,zwtvys->IA', v_aaaa, t1_aaea, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/12 * einsum('xyzw,uvAx,Isut,zwtvsy->IA', v_aaaa, t1_aaea, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/12 * einsum('xyzw,uvAx,Isut,zwtsyv->IA', v_aaaa, t1_aaea, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/12 * einsum('xyzw,uvAx,Isut,zwtsvy->IA', v_aaaa, t1_aaea, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xyzw,uvAx,Isvy,zwus->IA', v_aaaa, t1_aaea, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xyzw,uvAx,Isvu,yswz->IA', v_aaaa, t1_aaea, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 += 1/24 * einsum('xyzw,uvAx,Isvt,zwtyus->IA', v_aaaa, t1_aaea, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/24 * einsum('xyzw,uvAx,Isvt,zwtysu->IA', v_aaaa, t1_aaea, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 -= 5/24 * einsum('xyzw,uvAx,Isvt,zwtuys->IA', v_aaaa, t1_aaea, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/24 * einsum('xyzw,uvAx,Isvt,zwtusy->IA', v_aaaa, t1_aaea, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/24 * einsum('xyzw,uvAx,Isvt,zwtsyu->IA', v_aaaa, t1_aaea, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/24 * einsum('xyzw,uvAx,Isvt,zwtsuy->IA', v_aaaa, t1_aaea, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,uvAx,Isty,zwtuvs->IA', v_aaaa, t1_aaea, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,uvAx,Isty,zwtusv->IA', v_aaaa, t1_aaea, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,uvAx,Isty,zwtvus->IA', v_aaaa, t1_aaea, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 -= 5/12 * einsum('xyzw,uvAx,Isty,zwtvsu->IA', v_aaaa, t1_aaea, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,uvAx,Isty,zwtsuv->IA', v_aaaa, t1_aaea, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,uvAx,Isty,zwtsvu->IA', v_aaaa, t1_aaea, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/24 * einsum('xyzw,uvAx,Istu,zwtyvs->IA', v_aaaa, t1_aaea, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/24 * einsum('xyzw,uvAx,Istu,zwtysv->IA', v_aaaa, t1_aaea, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 -= 5/24 * einsum('xyzw,uvAx,Istu,zwtvys->IA', v_aaaa, t1_aaea, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/24 * einsum('xyzw,uvAx,Istu,zwtvsy->IA', v_aaaa, t1_aaea, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/24 * einsum('xyzw,uvAx,Istu,zwtsyv->IA', v_aaaa, t1_aaea, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/24 * einsum('xyzw,uvAx,Istu,zwtsvy->IA', v_aaaa, t1_aaea, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/24 * einsum('xyzw,uvAx,Istv,zwtyus->IA', v_aaaa, t1_aaea, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/24 * einsum('xyzw,uvAx,Istv,zwtysu->IA', v_aaaa, t1_aaea, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/24 * einsum('xyzw,uvAx,Istv,zwtuys->IA', v_aaaa, t1_aaea, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/24 * einsum('xyzw,uvAx,Istv,zwtusy->IA', v_aaaa, t1_aaea, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 -= 5/24 * einsum('xyzw,uvAx,Istv,zwtsyu->IA', v_aaaa, t1_aaea, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/24 * einsum('xyzw,uvAx,Istv,zwtsuy->IA', v_aaaa, t1_aaea, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 -= einsum('xyzw,iIAx,iuzw,uy->IA', v_aaaa, t1_ccea, t1_caaa, rdm_ca, optimize = einsum_type)
    V1 -= einsum('xyzw,iIAx,iz,wy->IA', v_aaaa, t1_ccea, t1_ca, rdm_ca, optimize = einsum_type)
    V1 -= einsum('xyzw,iIAx,iuzv,yvwu->IA', v_aaaa, t1_ccea, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 += 1/2 * einsum('xyzw,iIAx,iuwz,uy->IA', v_aaaa, t1_ccea, t1_caaa, rdm_ca, optimize = einsum_type)
    V1 += 1/2 * einsum('xyzw,iIAx,iw,zy->IA', v_aaaa, t1_ccea, t1_ca, rdm_ca, optimize = einsum_type)
    V1 += 1/2 * einsum('xyzw,iIAx,iuwv,yvzu->IA', v_aaaa, t1_ccea, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xyzw,iIAx,iyuv,zwuv->IA', v_aaaa, t1_ccea, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 += 1/4 * einsum('xyzw,iIAx,iu,yuwz->IA', v_aaaa, t1_ccea, t1_ca, rdm_ccaa, optimize = einsum_type)
    V1 += 1/2 * einsum('xyzw,iIAx,iuvz,yvwu->IA', v_aaaa, t1_ccea, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 += 1/2 * einsum('xyzw,iIAx,iuvw,yvuz->IA', v_aaaa, t1_ccea, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/24 * einsum('xyzw,iIAx,iuvs,zwuyvs->IA', v_aaaa, t1_ccea, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/24 * einsum('xyzw,iIAx,iuvs,zwuysv->IA', v_aaaa, t1_ccea, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 5/24 * einsum('xyzw,iIAx,iuvs,zwuvys->IA', v_aaaa, t1_ccea, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/24 * einsum('xyzw,iIAx,iuvs,zwuvsy->IA', v_aaaa, t1_ccea, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/24 * einsum('xyzw,iIAx,iuvs,zwusyv->IA', v_aaaa, t1_ccea, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/24 * einsum('xyzw,iIAx,iuvs,zwusvy->IA', v_aaaa, t1_ccea, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 -= einsum('xyzw,xyAu,Iz,uw->IA', v_aaaa, t1_aaea, t1_ca, rdm_ca, optimize = einsum_type)
    V1 += 1/2 * einsum('xyzw,xyAu,Iw,uz->IA', v_aaaa, t1_aaea, t1_ca, rdm_ca, optimize = einsum_type)
    V1 -= 1/4 * einsum('xyzw,xyAu,Iuvs,zwvs->IA', v_aaaa, t1_aaea, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 -= einsum('xyzw,xyAu,Ivzw,uv->IA', v_aaaa, t1_aaea, t1_caaa, rdm_ca, optimize = einsum_type)
    V1 -= einsum('xyzw,xyAu,Ivzs,wvus->IA', v_aaaa, t1_aaea, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 += 1/2 * einsum('xyzw,xyAu,Ivwz,uv->IA', v_aaaa, t1_aaea, t1_caaa, rdm_ca, optimize = einsum_type)
    V1 += 1/2 * einsum('xyzw,xyAu,Ivws,zvus->IA', v_aaaa, t1_aaea, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 += 1/2 * einsum('xyzw,xyAu,Ivsz,wvus->IA', v_aaaa, t1_aaea, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 += 1/2 * einsum('xyzw,xyAu,Ivsw,zvsu->IA', v_aaaa, t1_aaea, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xyzw,xuAv,Iyus,zwvs->IA', v_aaaa, t1_aaea, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xyzw,xuAv,Iysu,zwsv->IA', v_aaaa, t1_aaea, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,xuAv,Iyst,vstzwu->IA', v_aaaa, t1_aaea, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,xuAv,Iyst,vstzuw->IA', v_aaaa, t1_aaea, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,xuAv,Iyst,vstwzu->IA', v_aaaa, t1_aaea, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,xuAv,Iyst,vstwuz->IA', v_aaaa, t1_aaea, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 -= 5/12 * einsum('xyzw,xuAv,Iyst,vstuzw->IA', v_aaaa, t1_aaea, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,xuAv,Iyst,vstuwz->IA', v_aaaa, t1_aaea, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 -= einsum('xyzw,xuAv,Iz,yvwu->IA', v_aaaa, t1_aaea, t1_ca, rdm_ccaa, optimize = einsum_type)
    V1 += 1/2 * einsum('xyzw,xuAv,Iw,yvzu->IA', v_aaaa, t1_aaea, t1_ca, rdm_ccaa, optimize = einsum_type)
    V1 += 1/24 * einsum('xyzw,xuAv,Ivst,ystzwu->IA', v_aaaa, t1_aaea, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/24 * einsum('xyzw,xuAv,Ivst,ystzuw->IA', v_aaaa, t1_aaea, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 -= 5/24 * einsum('xyzw,xuAv,Ivst,ystwzu->IA', v_aaaa, t1_aaea, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/24 * einsum('xyzw,xuAv,Ivst,ystwuz->IA', v_aaaa, t1_aaea, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/24 * einsum('xyzw,xuAv,Ivst,ystuzw->IA', v_aaaa, t1_aaea, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/24 * einsum('xyzw,xuAv,Ivst,ystuwz->IA', v_aaaa, t1_aaea, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/4 * einsum('xyzw,xuAv,Iu,yvwz->IA', v_aaaa, t1_aaea, t1_ca, rdm_ccaa, optimize = einsum_type)
    V1 -= einsum('xyzw,xuAv,Iszw,yvsu->IA', v_aaaa, t1_aaea, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 -= einsum('xyzw,xuAv,Iszu,yvws->IA', v_aaaa, t1_aaea, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 -= einsum('xyzw,xuAv,Iszt,yvtwus->IA', v_aaaa, t1_aaea, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/2 * einsum('xyzw,xuAv,Iswz,yvsu->IA', v_aaaa, t1_aaea, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 += 1/2 * einsum('xyzw,xuAv,Iswu,yvzs->IA', v_aaaa, t1_aaea, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 += 1/2 * einsum('xyzw,xuAv,Iswt,yvtzus->IA', v_aaaa, t1_aaea, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/2 * einsum('xyzw,xuAv,Isuz,yvws->IA', v_aaaa, t1_aaea, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 += 1/2 * einsum('xyzw,xuAv,Isuw,yvsz->IA', v_aaaa, t1_aaea, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/24 * einsum('xyzw,xuAv,Isut,yvtzws->IA', v_aaaa, t1_aaea, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/24 * einsum('xyzw,xuAv,Isut,yvtzsw->IA', v_aaaa, t1_aaea, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 5/24 * einsum('xyzw,xuAv,Isut,yvtwzs->IA', v_aaaa, t1_aaea, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/24 * einsum('xyzw,xuAv,Isut,yvtwsz->IA', v_aaaa, t1_aaea, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/24 * einsum('xyzw,xuAv,Isut,yvtszw->IA', v_aaaa, t1_aaea, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/24 * einsum('xyzw,xuAv,Isut,yvtswz->IA', v_aaaa, t1_aaea, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/2 * einsum('xyzw,xuAv,Istz,yvtwus->IA', v_aaaa, t1_aaea, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/6 * einsum('xyzw,xuAv,Istw,yvtzus->IA', v_aaaa, t1_aaea, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/6 * einsum('xyzw,xuAv,Istw,yvtzsu->IA', v_aaaa, t1_aaea, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/6 * einsum('xyzw,xuAv,Istw,yvtuzs->IA', v_aaaa, t1_aaea, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/6 * einsum('xyzw,xuAv,Istw,yvtusz->IA', v_aaaa, t1_aaea, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/6 * einsum('xyzw,xuAv,Istw,yvtszu->IA', v_aaaa, t1_aaea, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/3 * einsum('xyzw,xuAv,Istw,yvtsuz->IA', v_aaaa, t1_aaea, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/24 * einsum('xyzw,xuAv,Istu,yvtzws->IA', v_aaaa, t1_aaea, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/24 * einsum('xyzw,xuAv,Istu,yvtzsw->IA', v_aaaa, t1_aaea, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/24 * einsum('xyzw,xuAv,Istu,yvtwzs->IA', v_aaaa, t1_aaea, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 5/24 * einsum('xyzw,xuAv,Istu,yvtwsz->IA', v_aaaa, t1_aaea, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/24 * einsum('xyzw,xuAv,Istu,yvtszw->IA', v_aaaa, t1_aaea, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/24 * einsum('xyzw,xuAv,Istu,yvtswz->IA', v_aaaa, t1_aaea, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/2 * einsum('xyzw,uxAv,Iyus,zwvs->IA', v_aaaa, t1_aaea, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xyzw,uxAv,Iysu,zwvs->IA', v_aaaa, t1_aaea, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,uxAv,Iyst,vstzwu->IA', v_aaaa, t1_aaea, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 -= 5/12 * einsum('xyzw,uxAv,Iyst,vstzuw->IA', v_aaaa, t1_aaea, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,uxAv,Iyst,vstwzu->IA', v_aaaa, t1_aaea, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,uxAv,Iyst,vstwuz->IA', v_aaaa, t1_aaea, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,uxAv,Iyst,vstuzw->IA', v_aaaa, t1_aaea, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,uxAv,Iyst,vstuwz->IA', v_aaaa, t1_aaea, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/2 * einsum('xyzw,uxAv,Iz,yvwu->IA', v_aaaa, t1_aaea, t1_ca, rdm_ccaa, optimize = einsum_type)
    V1 += 1/2 * einsum('xyzw,uxAv,Iw,yvuz->IA', v_aaaa, t1_aaea, t1_ca, rdm_ccaa, optimize = einsum_type)
    V1 += 1/24 * einsum('xyzw,uxAv,Ivst,ystzwu->IA', v_aaaa, t1_aaea, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/24 * einsum('xyzw,uxAv,Ivst,ystzuw->IA', v_aaaa, t1_aaea, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/24 * einsum('xyzw,uxAv,Ivst,ystwzu->IA', v_aaaa, t1_aaea, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 -= 5/24 * einsum('xyzw,uxAv,Ivst,ystwuz->IA', v_aaaa, t1_aaea, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/24 * einsum('xyzw,uxAv,Ivst,ystuzw->IA', v_aaaa, t1_aaea, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/24 * einsum('xyzw,uxAv,Ivst,ystuwz->IA', v_aaaa, t1_aaea, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('xyzw,uxAv,Iu,yvwz->IA', v_aaaa, t1_aaea, t1_ca, rdm_ccaa, optimize = einsum_type)
    V1 += 1/2 * einsum('xyzw,uxAv,Iszw,yvsu->IA', v_aaaa, t1_aaea, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 += 1/2 * einsum('xyzw,uxAv,Iszu,yvws->IA', v_aaaa, t1_aaea, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 += 1/2 * einsum('xyzw,uxAv,Iszt,yvtwus->IA', v_aaaa, t1_aaea, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/2 * einsum('xyzw,uxAv,Iswz,yvus->IA', v_aaaa, t1_aaea, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 += 1/2 * einsum('xyzw,uxAv,Iswu,yvsz->IA', v_aaaa, t1_aaea, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/6 * einsum('xyzw,uxAv,Iswt,yvtzus->IA', v_aaaa, t1_aaea, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/6 * einsum('xyzw,uxAv,Iswt,yvtzsu->IA', v_aaaa, t1_aaea, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/3 * einsum('xyzw,uxAv,Iswt,yvtuzs->IA', v_aaaa, t1_aaea, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/6 * einsum('xyzw,uxAv,Iswt,yvtusz->IA', v_aaaa, t1_aaea, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/6 * einsum('xyzw,uxAv,Iswt,yvtszu->IA', v_aaaa, t1_aaea, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/6 * einsum('xyzw,uxAv,Iswt,yvtsuz->IA', v_aaaa, t1_aaea, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 -= einsum('xyzw,uxAv,Isuz,yvws->IA', v_aaaa, t1_aaea, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 -= einsum('xyzw,uxAv,Isuw,yvsz->IA', v_aaaa, t1_aaea, t1_caaa, rdm_ccaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,uxAv,Isut,yvtzws->IA', v_aaaa, t1_aaea, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,uxAv,Isut,yvtzsw->IA', v_aaaa, t1_aaea, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 -= 5/12 * einsum('xyzw,uxAv,Isut,yvtwzs->IA', v_aaaa, t1_aaea, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,uxAv,Isut,yvtwsz->IA', v_aaaa, t1_aaea, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,uxAv,Isut,yvtszw->IA', v_aaaa, t1_aaea, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,uxAv,Isut,yvtswz->IA', v_aaaa, t1_aaea, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/6 * einsum('xyzw,uxAv,Istz,yvtwus->IA', v_aaaa, t1_aaea, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/3 * einsum('xyzw,uxAv,Istz,yvtwsu->IA', v_aaaa, t1_aaea, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/6 * einsum('xyzw,uxAv,Istz,yvtuws->IA', v_aaaa, t1_aaea, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/6 * einsum('xyzw,uxAv,Istz,yvtusw->IA', v_aaaa, t1_aaea, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/6 * einsum('xyzw,uxAv,Istz,yvtswu->IA', v_aaaa, t1_aaea, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/6 * einsum('xyzw,uxAv,Istz,yvtsuw->IA', v_aaaa, t1_aaea, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/2 * einsum('xyzw,uxAv,Istw,yvtszu->IA', v_aaaa, t1_aaea, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/24 * einsum('xyzw,uxAv,Istu,yvtzws->IA', v_aaaa, t1_aaea, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/24 * einsum('xyzw,uxAv,Istu,yvtzsw->IA', v_aaaa, t1_aaea, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 += 5/24 * einsum('xyzw,uxAv,Istu,yvtwzs->IA', v_aaaa, t1_aaea, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/24 * einsum('xyzw,uxAv,Istu,yvtwsz->IA', v_aaaa, t1_aaea, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/24 * einsum('xyzw,uxAv,Istu,yvtszw->IA', v_aaaa, t1_aaea, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/24 * einsum('xyzw,uxAv,Istu,yvtswz->IA', v_aaaa, t1_aaea, t1_caaa, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('xyzw,IxAa,uvay,zwuv->IA', v_aaaa, t1_caee, t1_aaea, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('xyzw,IxAa,ua,yuwz->IA', v_aaaa, t1_caee, t1_ae, rdm_ccaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,IxAa,uvas,yuvzws->IA', v_aaaa, t1_caee, t1_aaea, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,IxAa,uvas,yuvzsw->IA', v_aaaa, t1_caee, t1_aaea, rdm_cccaaa, optimize = einsum_type)
    V1 -= 5/12 * einsum('xyzw,IxAa,uvas,yuvwzs->IA', v_aaaa, t1_caee, t1_aaea, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,IxAa,uvas,yuvwsz->IA', v_aaaa, t1_caee, t1_aaea, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,IxAa,uvas,yuvszw->IA', v_aaaa, t1_caee, t1_aaea, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,IxAa,uvas,yuvswz->IA', v_aaaa, t1_caee, t1_aaea, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/2 * einsum('xyzw,Ixuv,uA,yvwz->IA', v_aaaa, t1_caaa, t1_ae, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xyzw,Ixuv,vA,yuwz->IA', v_aaaa, t1_caaa, t1_ae, rdm_ccaa, optimize = einsum_type)
    V1 += 1/2 * einsum('xyzw,Ixuv,uvAs,yswz->IA', v_aaaa, t1_caaa, t1_aaea, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/12 * einsum('xyzw,Ixuv,usAt,yvtzws->IA', v_aaaa, t1_caaa, t1_aaea, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/12 * einsum('xyzw,Ixuv,usAt,yvtzsw->IA', v_aaaa, t1_caaa, t1_aaea, rdm_cccaaa, optimize = einsum_type)
    V1 += 5/12 * einsum('xyzw,Ixuv,usAt,yvtwzs->IA', v_aaaa, t1_caaa, t1_aaea, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/12 * einsum('xyzw,Ixuv,usAt,yvtwsz->IA', v_aaaa, t1_caaa, t1_aaea, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/12 * einsum('xyzw,Ixuv,usAt,yvtszw->IA', v_aaaa, t1_caaa, t1_aaea, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/12 * einsum('xyzw,Ixuv,usAt,yvtswz->IA', v_aaaa, t1_caaa, t1_aaea, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xyzw,Ixuv,vuAs,yswz->IA', v_aaaa, t1_caaa, t1_aaea, rdm_ccaa, optimize = einsum_type)
    V1 += 1/24 * einsum('xyzw,Ixuv,vsAt,yutzws->IA', v_aaaa, t1_caaa, t1_aaea, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/24 * einsum('xyzw,Ixuv,vsAt,yutzsw->IA', v_aaaa, t1_caaa, t1_aaea, rdm_cccaaa, optimize = einsum_type)
    V1 -= 5/24 * einsum('xyzw,Ixuv,vsAt,yutwzs->IA', v_aaaa, t1_caaa, t1_aaea, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/24 * einsum('xyzw,Ixuv,vsAt,yutwsz->IA', v_aaaa, t1_caaa, t1_aaea, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/24 * einsum('xyzw,Ixuv,vsAt,yutszw->IA', v_aaaa, t1_caaa, t1_aaea, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/24 * einsum('xyzw,Ixuv,vsAt,yutswz->IA', v_aaaa, t1_caaa, t1_aaea, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/24 * einsum('xyzw,Ixuv,suAt,yvtzws->IA', v_aaaa, t1_caaa, t1_aaea, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/24 * einsum('xyzw,Ixuv,suAt,yvtzsw->IA', v_aaaa, t1_caaa, t1_aaea, rdm_cccaaa, optimize = einsum_type)
    V1 -= 5/24 * einsum('xyzw,Ixuv,suAt,yvtwzs->IA', v_aaaa, t1_caaa, t1_aaea, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/24 * einsum('xyzw,Ixuv,suAt,yvtwsz->IA', v_aaaa, t1_caaa, t1_aaea, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/24 * einsum('xyzw,Ixuv,suAt,yvtszw->IA', v_aaaa, t1_caaa, t1_aaea, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/24 * einsum('xyzw,Ixuv,suAt,yvtswz->IA', v_aaaa, t1_caaa, t1_aaea, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/24 * einsum('xyzw,Ixuv,svAt,yutzws->IA', v_aaaa, t1_caaa, t1_aaea, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/24 * einsum('xyzw,Ixuv,svAt,yutzsw->IA', v_aaaa, t1_caaa, t1_aaea, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/24 * einsum('xyzw,Ixuv,svAt,yutwzs->IA', v_aaaa, t1_caaa, t1_aaea, rdm_cccaaa, optimize = einsum_type)
    V1 -= 5/24 * einsum('xyzw,Ixuv,svAt,yutwsz->IA', v_aaaa, t1_caaa, t1_aaea, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/24 * einsum('xyzw,Ixuv,svAt,yutszw->IA', v_aaaa, t1_caaa, t1_aaea, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/24 * einsum('xyzw,Ixuv,svAt,yutswz->IA', v_aaaa, t1_caaa, t1_aaea, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/4 * einsum('xyzw,IxaA,uvay,zwuv->IA', v_aaaa, t1_caee, t1_aaea, rdm_ccaa, optimize = einsum_type)
    V1 += 1/4 * einsum('xyzw,IxaA,ua,yuwz->IA', v_aaaa, t1_caee, t1_ae, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/24 * einsum('xyzw,IxaA,uvas,yuvzws->IA', v_aaaa, t1_caee, t1_aaea, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/24 * einsum('xyzw,IxaA,uvas,yuvzsw->IA', v_aaaa, t1_caee, t1_aaea, rdm_cccaaa, optimize = einsum_type)
    V1 += 5/24 * einsum('xyzw,IxaA,uvas,yuvwzs->IA', v_aaaa, t1_caee, t1_aaea, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/24 * einsum('xyzw,IxaA,uvas,yuvwsz->IA', v_aaaa, t1_caee, t1_aaea, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/24 * einsum('xyzw,IxaA,uvas,yuvszw->IA', v_aaaa, t1_caee, t1_aaea, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/24 * einsum('xyzw,IxaA,uvas,yuvswz->IA', v_aaaa, t1_caee, t1_aaea, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xyzw,Iuxy,vsAu,zwvs->IA', v_aaaa, t1_caaa, t1_aaea, rdm_ccaa, optimize = einsum_type)
    V1 += 1/24 * einsum('xyzw,Iuxv,stAu,zwvyst->IA', v_aaaa, t1_caaa, t1_aaea, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/24 * einsum('xyzw,Iuxv,stAu,zwvyts->IA', v_aaaa, t1_caaa, t1_aaea, rdm_cccaaa, optimize = einsum_type)
    V1 -= 5/24 * einsum('xyzw,Iuxv,stAu,zwvsyt->IA', v_aaaa, t1_caaa, t1_aaea, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/24 * einsum('xyzw,Iuxv,stAu,zwvsty->IA', v_aaaa, t1_caaa, t1_aaea, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/24 * einsum('xyzw,Iuxv,stAu,zwvtys->IA', v_aaaa, t1_caaa, t1_aaea, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/24 * einsum('xyzw,Iuxv,stAu,zwvtsy->IA', v_aaaa, t1_caaa, t1_aaea, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/4 * einsum('xyzw,Iuxv,vA,yuwz->IA', v_aaaa, t1_caaa, t1_ae, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/24 * einsum('xyzw,Iuxv,vsAt,zwtyus->IA', v_aaaa, t1_caaa, t1_aaea, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/24 * einsum('xyzw,Iuxv,vsAt,zwtysu->IA', v_aaaa, t1_caaa, t1_aaea, rdm_cccaaa, optimize = einsum_type)
    V1 += 5/24 * einsum('xyzw,Iuxv,vsAt,zwtuys->IA', v_aaaa, t1_caaa, t1_aaea, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/24 * einsum('xyzw,Iuxv,vsAt,zwtusy->IA', v_aaaa, t1_caaa, t1_aaea, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/24 * einsum('xyzw,Iuxv,vsAt,zwtsyu->IA', v_aaaa, t1_caaa, t1_aaea, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/24 * einsum('xyzw,Iuxv,vsAt,zwtsuy->IA', v_aaaa, t1_caaa, t1_aaea, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/24 * einsum('xyzw,Iuxv,svAt,zwtyus->IA', v_aaaa, t1_caaa, t1_aaea, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/24 * einsum('xyzw,Iuxv,svAt,zwtysu->IA', v_aaaa, t1_caaa, t1_aaea, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/24 * einsum('xyzw,Iuxv,svAt,zwtuys->IA', v_aaaa, t1_caaa, t1_aaea, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/24 * einsum('xyzw,Iuxv,svAt,zwtusy->IA', v_aaaa, t1_caaa, t1_aaea, rdm_cccaaa, optimize = einsum_type)
    V1 += 5/24 * einsum('xyzw,Iuxv,svAt,zwtsyu->IA', v_aaaa, t1_caaa, t1_aaea, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/24 * einsum('xyzw,Iuxv,svAt,zwtsuy->IA', v_aaaa, t1_caaa, t1_aaea, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/24 * einsum('xyzw,Iuvx,stAu,zwvyst->IA', v_aaaa, t1_caaa, t1_aaea, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/24 * einsum('xyzw,Iuvx,stAu,zwvyts->IA', v_aaaa, t1_caaa, t1_aaea, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/24 * einsum('xyzw,Iuvx,stAu,zwvsyt->IA', v_aaaa, t1_caaa, t1_aaea, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/24 * einsum('xyzw,Iuvx,stAu,zwvsty->IA', v_aaaa, t1_caaa, t1_aaea, rdm_cccaaa, optimize = einsum_type)
    V1 -= 5/24 * einsum('xyzw,Iuvx,stAu,zwvtys->IA', v_aaaa, t1_caaa, t1_aaea, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/24 * einsum('xyzw,Iuvx,stAu,zwvtsy->IA', v_aaaa, t1_caaa, t1_aaea, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('xyzw,Iuvx,vA,yuwz->IA', v_aaaa, t1_caaa, t1_ae, rdm_ccaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,Iuvx,vsAt,zwtyus->IA', v_aaaa, t1_caaa, t1_aaea, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,Iuvx,vsAt,zwtysu->IA', v_aaaa, t1_caaa, t1_aaea, rdm_cccaaa, optimize = einsum_type)
    V1 -= 5/12 * einsum('xyzw,Iuvx,vsAt,zwtuys->IA', v_aaaa, t1_caaa, t1_aaea, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,Iuvx,vsAt,zwtusy->IA', v_aaaa, t1_caaa, t1_aaea, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,Iuvx,vsAt,zwtsyu->IA', v_aaaa, t1_caaa, t1_aaea, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,Iuvx,vsAt,zwtsuy->IA', v_aaaa, t1_caaa, t1_aaea, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/24 * einsum('xyzw,Iuvx,svAt,zwtyus->IA', v_aaaa, t1_caaa, t1_aaea, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/24 * einsum('xyzw,Iuvx,svAt,zwtysu->IA', v_aaaa, t1_caaa, t1_aaea, rdm_cccaaa, optimize = einsum_type)
    V1 += 5/24 * einsum('xyzw,Iuvx,svAt,zwtuys->IA', v_aaaa, t1_caaa, t1_aaea, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/24 * einsum('xyzw,Iuvx,svAt,zwtusy->IA', v_aaaa, t1_caaa, t1_aaea, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/24 * einsum('xyzw,Iuvx,svAt,zwtsyu->IA', v_aaaa, t1_caaa, t1_aaea, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/24 * einsum('xyzw,Iuvx,svAt,zwtsuy->IA', v_aaaa, t1_caaa, t1_aaea, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('xyzw,iuxy,IiAv,zwvu->IA', v_aaaa, t1_caaa, t1_ccea, rdm_ccaa, optimize = einsum_type)
    V1 += 1/4 * einsum('xyzw,iuxy,iIAv,zwvu->IA', v_aaaa, t1_caaa, t1_ccea, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('xyzw,xyau,IvAa,zwvu->IA', v_aaaa, t1_aaea, t1_caee, rdm_ccaa, optimize = einsum_type)
    V1 += 1/4 * einsum('xyzw,xyau,IvaA,zwvu->IA', v_aaaa, t1_aaea, t1_caee, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('xyzw,iuxv,IiAv,yuwz->IA', v_aaaa, t1_caaa, t1_ccea, rdm_ccaa, optimize = einsum_type)
    V1 += 1/4 * einsum('xyzw,iuxv,iIAv,yuwz->IA', v_aaaa, t1_caaa, t1_ccea, rdm_ccaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,iuxv,IiAs,yuszwv->IA', v_aaaa, t1_caaa, t1_ccea, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,iuxv,IiAs,yuszvw->IA', v_aaaa, t1_caaa, t1_ccea, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,iuxv,IiAs,yuswzv->IA', v_aaaa, t1_caaa, t1_ccea, rdm_cccaaa, optimize = einsum_type)
    V1 -= 5/12 * einsum('xyzw,iuxv,IiAs,yuswvz->IA', v_aaaa, t1_caaa, t1_ccea, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,iuxv,IiAs,yusvzw->IA', v_aaaa, t1_caaa, t1_ccea, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,iuxv,IiAs,yusvwz->IA', v_aaaa, t1_caaa, t1_ccea, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/24 * einsum('xyzw,iuxv,iIAs,yuszwv->IA', v_aaaa, t1_caaa, t1_ccea, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/24 * einsum('xyzw,iuxv,iIAs,yuszvw->IA', v_aaaa, t1_caaa, t1_ccea, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/24 * einsum('xyzw,iuxv,iIAs,yuswzv->IA', v_aaaa, t1_caaa, t1_ccea, rdm_cccaaa, optimize = einsum_type)
    V1 += 5/24 * einsum('xyzw,iuxv,iIAs,yuswvz->IA', v_aaaa, t1_caaa, t1_ccea, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/24 * einsum('xyzw,iuxv,iIAs,yusvzw->IA', v_aaaa, t1_caaa, t1_ccea, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/24 * einsum('xyzw,iuxv,iIAs,yusvwz->IA', v_aaaa, t1_caaa, t1_ccea, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,xuav,IsAa,zwuyvs->IA', v_aaaa, t1_aaea, t1_caee, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,xuav,IsAa,zwuysv->IA', v_aaaa, t1_aaea, t1_caee, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,xuav,IsAa,zwuvys->IA', v_aaaa, t1_aaea, t1_caee, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,xuav,IsAa,zwuvsy->IA', v_aaaa, t1_aaea, t1_caee, rdm_cccaaa, optimize = einsum_type)
    V1 -= 5/12 * einsum('xyzw,xuav,IsAa,zwusyv->IA', v_aaaa, t1_aaea, t1_caee, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,xuav,IsAa,zwusvy->IA', v_aaaa, t1_aaea, t1_caee, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/24 * einsum('xyzw,xuav,IsaA,zwuyvs->IA', v_aaaa, t1_aaea, t1_caee, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/24 * einsum('xyzw,xuav,IsaA,zwuysv->IA', v_aaaa, t1_aaea, t1_caee, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/24 * einsum('xyzw,xuav,IsaA,zwuvys->IA', v_aaaa, t1_aaea, t1_caee, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/24 * einsum('xyzw,xuav,IsaA,zwuvsy->IA', v_aaaa, t1_aaea, t1_caee, rdm_cccaaa, optimize = einsum_type)
    V1 += 5/24 * einsum('xyzw,xuav,IsaA,zwusyv->IA', v_aaaa, t1_aaea, t1_caee, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/24 * einsum('xyzw,xuav,IsaA,zwusvy->IA', v_aaaa, t1_aaea, t1_caee, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('xyzw,ix,IiAu,yuwz->IA', v_aaaa, t1_ca, t1_ccea, rdm_ccaa, optimize = einsum_type)
    V1 += 1/4 * einsum('xyzw,ix,iIAu,yuwz->IA', v_aaaa, t1_ca, t1_ccea, rdm_ccaa, optimize = einsum_type)
    V1 -= einsum('xyzw,ixau,IiAa,yuwz->IA', v_aaaa, t1_caea, t1_ccee, rdm_ccaa, optimize = einsum_type)
    V1 += 1/2 * einsum('xyzw,ixau,iIAa,yuwz->IA', v_aaaa, t1_caea, t1_ccee, rdm_ccaa, optimize = einsum_type)
    V1 += 1/2 * einsum('xyzw,ixua,IiAa,yuwz->IA', v_aaaa, t1_caae, t1_ccee, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xyzw,ixua,iIAa,yuwz->IA', v_aaaa, t1_caae, t1_ccee, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('xyzw,xa,IuAa,yuwz->IA', v_aaaa, t1_ae, t1_caee, rdm_ccaa, optimize = einsum_type)
    V1 += 1/4 * einsum('xyzw,xa,IuaA,yuwz->IA', v_aaaa, t1_ae, t1_caee, rdm_ccaa, optimize = einsum_type)
    V1 += einsum('xyzw,iuax,IiAa,yuwz->IA', v_aaaa, t1_caea, t1_ccee, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('xyzw,iuax,iIAa,yuwz->IA', v_aaaa, t1_caea, t1_ccee, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('xyzw,iuxa,IiAa,yuwz->IA', v_aaaa, t1_caae, t1_ccee, rdm_ccaa, optimize = einsum_type)
    V1 += 1/4 * einsum('xyzw,iuxa,iIAa,yuwz->IA', v_aaaa, t1_caae, t1_ccee, rdm_ccaa, optimize = einsum_type)
    V1 += einsum('xyzw,iuvx,IiAv,yuwz->IA', v_aaaa, t1_caaa, t1_ccea, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/2 * einsum('xyzw,iuvx,iIAv,yuwz->IA', v_aaaa, t1_caaa, t1_ccea, rdm_ccaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,iuvx,IiAs,yuszwv->IA', v_aaaa, t1_caaa, t1_ccea, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,iuvx,IiAs,yuszvw->IA', v_aaaa, t1_caaa, t1_ccea, rdm_cccaaa, optimize = einsum_type)
    V1 -= 5/12 * einsum('xyzw,iuvx,IiAs,yuswzv->IA', v_aaaa, t1_caaa, t1_ccea, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,iuvx,IiAs,yuswvz->IA', v_aaaa, t1_caaa, t1_ccea, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,iuvx,IiAs,yusvzw->IA', v_aaaa, t1_caaa, t1_ccea, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,iuvx,IiAs,yusvwz->IA', v_aaaa, t1_caaa, t1_ccea, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/24 * einsum('xyzw,iuvx,iIAs,yuszwv->IA', v_aaaa, t1_caaa, t1_ccea, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/24 * einsum('xyzw,iuvx,iIAs,yuszvw->IA', v_aaaa, t1_caaa, t1_ccea, rdm_cccaaa, optimize = einsum_type)
    V1 += 5/24 * einsum('xyzw,iuvx,iIAs,yuswzv->IA', v_aaaa, t1_caaa, t1_ccea, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/24 * einsum('xyzw,iuvx,iIAs,yuswvz->IA', v_aaaa, t1_caaa, t1_ccea, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/24 * einsum('xyzw,iuvx,iIAs,yusvzw->IA', v_aaaa, t1_caaa, t1_ccea, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/24 * einsum('xyzw,iuvx,iIAs,yusvwz->IA', v_aaaa, t1_caaa, t1_ccea, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,uxav,IsAa,zwuyvs->IA', v_aaaa, t1_aaea, t1_caee, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,uxav,IsAa,zwuysv->IA', v_aaaa, t1_aaea, t1_caee, rdm_cccaaa, optimize = einsum_type)
    V1 -= 5/12 * einsum('xyzw,uxav,IsAa,zwuvys->IA', v_aaaa, t1_aaea, t1_caee, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,uxav,IsAa,zwuvsy->IA', v_aaaa, t1_aaea, t1_caee, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,uxav,IsAa,zwusyv->IA', v_aaaa, t1_aaea, t1_caee, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/12 * einsum('xyzw,uxav,IsAa,zwusvy->IA', v_aaaa, t1_aaea, t1_caee, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/24 * einsum('xyzw,uxav,IsaA,zwuyvs->IA', v_aaaa, t1_aaea, t1_caee, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/24 * einsum('xyzw,uxav,IsaA,zwuysv->IA', v_aaaa, t1_aaea, t1_caee, rdm_cccaaa, optimize = einsum_type)
    V1 += 5/24 * einsum('xyzw,uxav,IsaA,zwuvys->IA', v_aaaa, t1_aaea, t1_caee, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/24 * einsum('xyzw,uxav,IsaA,zwuvsy->IA', v_aaaa, t1_aaea, t1_caee, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/24 * einsum('xyzw,uxav,IsaA,zwusyv->IA', v_aaaa, t1_aaea, t1_caee, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/24 * einsum('xyzw,uxav,IsaA,zwusvy->IA', v_aaaa, t1_aaea, t1_caee, rdm_cccaaa, optimize = einsum_type)
    V1 -= einsum('xyzw,ixuv,IiAu,yvwz->IA', v_aaaa, t1_caaa, t1_ccea, rdm_ccaa, optimize = einsum_type)
    V1 += 1/2 * einsum('xyzw,ixuv,iIAu,yvwz->IA', v_aaaa, t1_caaa, t1_ccea, rdm_ccaa, optimize = einsum_type)
    V1 += 1/2 * einsum('xyzw,ixuv,IiAv,yuwz->IA', v_aaaa, t1_caaa, t1_ccea, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/4 * einsum('xyzw,ixuv,iIAv,yuwz->IA', v_aaaa, t1_caaa, t1_ccea, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/12 * einsum('xyzw,ixuv,IiAs,zwsyuv->IA', v_aaaa, t1_caaa, t1_ccea, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/12 * einsum('xyzw,ixuv,IiAs,zwsyvu->IA', v_aaaa, t1_caaa, t1_ccea, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/12 * einsum('xyzw,ixuv,IiAs,zwsuyv->IA', v_aaaa, t1_caaa, t1_ccea, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/12 * einsum('xyzw,ixuv,IiAs,zwsuvy->IA', v_aaaa, t1_caaa, t1_ccea, rdm_cccaaa, optimize = einsum_type)
    V1 += 5/12 * einsum('xyzw,ixuv,IiAs,zwsvyu->IA', v_aaaa, t1_caaa, t1_ccea, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/12 * einsum('xyzw,ixuv,IiAs,zwsvuy->IA', v_aaaa, t1_caaa, t1_ccea, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/24 * einsum('xyzw,ixuv,iIAs,zwsyuv->IA', v_aaaa, t1_caaa, t1_ccea, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/24 * einsum('xyzw,ixuv,iIAs,zwsyvu->IA', v_aaaa, t1_caaa, t1_ccea, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/24 * einsum('xyzw,ixuv,iIAs,zwsuyv->IA', v_aaaa, t1_caaa, t1_ccea, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/24 * einsum('xyzw,ixuv,iIAs,zwsuvy->IA', v_aaaa, t1_caaa, t1_ccea, rdm_cccaaa, optimize = einsum_type)
    V1 -= 5/24 * einsum('xyzw,ixuv,iIAs,zwsvyu->IA', v_aaaa, t1_caaa, t1_ccea, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/24 * einsum('xyzw,ixuv,iIAs,zwsvuy->IA', v_aaaa, t1_caaa, t1_ccea, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/12 * einsum('xyzw,uvax,IsAa,yuvzws->IA', v_aaaa, t1_aaea, t1_caee, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/12 * einsum('xyzw,uvax,IsAa,yuvzsw->IA', v_aaaa, t1_aaea, t1_caee, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/12 * einsum('xyzw,uvax,IsAa,yuvwzs->IA', v_aaaa, t1_aaea, t1_caee, rdm_cccaaa, optimize = einsum_type)
    V1 += 5/12 * einsum('xyzw,uvax,IsAa,yuvwsz->IA', v_aaaa, t1_aaea, t1_caee, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/12 * einsum('xyzw,uvax,IsAa,yuvszw->IA', v_aaaa, t1_aaea, t1_caee, rdm_cccaaa, optimize = einsum_type)
    V1 -= 1/12 * einsum('xyzw,uvax,IsAa,yuvswz->IA', v_aaaa, t1_aaea, t1_caee, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/24 * einsum('xyzw,uvax,IsaA,yuvzws->IA', v_aaaa, t1_aaea, t1_caee, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/24 * einsum('xyzw,uvax,IsaA,yuvzsw->IA', v_aaaa, t1_aaea, t1_caee, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/24 * einsum('xyzw,uvax,IsaA,yuvwzs->IA', v_aaaa, t1_aaea, t1_caee, rdm_cccaaa, optimize = einsum_type)
    V1 -= 5/24 * einsum('xyzw,uvax,IsaA,yuvwsz->IA', v_aaaa, t1_aaea, t1_caee, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/24 * einsum('xyzw,uvax,IsaA,yuvszw->IA', v_aaaa, t1_aaea, t1_caee, rdm_cccaaa, optimize = einsum_type)
    V1 += 1/24 * einsum('xyzw,uvax,IsaA,yuvswz->IA', v_aaaa, t1_aaea, t1_caee, rdm_cccaaa, optimize = einsum_type)

    # Compute denominators
    d_ai = (e_extern[:,None] - e_core)
    d_ai = d_ai**(-1)

    t2_ce = np.einsum("ai,ia->ia", d_ai, - 1.0 * V1)

    return t2_ce

### Under Development
def compute_t1_0p(mr_adc):

    K_caca_so = mr_adc_intermediates.compute_K_caca_so(mr_adc)

    rdm_ca_so = mr_adc.rdm_so.ca
    rdm_ccaa_so = mr_adc.rdm_so.ccaa
    rdm_cccaaa_so = mr_adc.rdm_so.cccaaa

    e_core_so = mr_adc.mo_energy_so.c
    e_extern_so = mr_adc.mo_energy_so.e

    ncore_so = mr_adc.ncore_so
    ncas_so = mr_adc.ncas_so
    nocc_so = mr_adc.nocc_so
    nextern_so = mr_adc.nextern_so

    n_tia = ncore_so * nextern_so
    n_tixay = ncore_so * ncas_so * ncas_so * nextern_so

    # Orthogonalization and overlap truncation only in the active space
    # S_0p_12_inv_act = mr_adc_overlap.compute_S12_0p(mr_adc, ignore_print = False)
    # S_0p_12_inv_act = mr_adc_overlap.compute_S12_0p_projector(mr_adc, ignore_print = False)
    S_0p_12_inv_act = mr_adc_overlap.compute_S12_0p_gno_projector(mr_adc, ignore_print = False)

    # Compute (S_12 K S_12)_{i a mu, j b nu}
    SKS = np.einsum("xywz,zwn->xyn", K_caca_so, S_0p_12_inv_act[1:,:].reshape(ncas_so, ncas_so, -1))
    SKS = np.einsum("xym,xyn->mn", S_0p_12_inv_act[1:,:].reshape(ncas_so, ncas_so, -1), SKS)

    evals, evecs = np.linalg.eigh(SKS)

    # Compute r.h.s. of the equation
    # h1_ce_so = mr_adc.h1eff_so[:ncore_so,nocc_so:]
    h1eff_ce_so = mr_adc.h1eff_so[:ncore_so,nocc_so:]
    v_caea_so = mr_adc.v2e_so.caea

    rdm_caca_so = rdm_ccaa_so.transpose(1,2,0,3).copy()
    rdm_caca_so += np.einsum('wy,xz->xywz', np.identity(ncas_so), rdm_ca_so)

    V0p = np.zeros((ncore_so, nextern_so, ncas_so * ncas_so + 1))
    V0p[:,:,0] = -h1eff_ce_so
    V0p[:,:,0] -= np.einsum("izaw,wz->ia", v_caea_so, rdm_ca_so)

    V0p[:,:,1:] = -np.einsum("ia,xy->iaxy", h1eff_ce_so, rdm_ca_so).reshape(ncore_so, nextern_so, -1)
    V0p[:,:,1:] -= np.einsum("izaw,xywz->iaxy", v_caea_so, rdm_caca_so).reshape(ncore_so, nextern_so, -1)

    V0p = V0p.reshape(ncore_so, nextern_so, -1)

    S_12_V0p = np.einsum("iaP,Pm->iam", V0p, S_0p_12_inv_act)

    # Multiply r.h.s. by U (e_a - e_i + e_mu)^-1 U^dag
    S_12_V0p = np.einsum("mp,iam->iap", evecs, S_12_V0p)

    # Compute denominators
    d_ai = (e_extern_so[:,None] - e_core_so).reshape(-1)
    d_aip = (d_ai[:,None] + evals).reshape(nextern_so, ncore_so, -1)
    d_aip = d_aip**(-1)

    S_12_V0p = np.einsum("aip,iap->iap", d_aip, S_12_V0p)
    S_12_V0p = np.einsum("mp,iap->iam", evecs, S_12_V0p)

    t0p = np.einsum("iam,Pm->iaP", S_12_V0p, S_0p_12_inv_act)

    t_ia = t0p[:,:,0].copy()
    t_ixay = t0p[:,:,1:].reshape(ncore_so, nextern_so, ncas_so, ncas_so)
    t_ixay = t_ixay.transpose(0,2,1,3).copy()

    e_0p = np.einsum("ia,ia", h1eff_ce_so, t_ia)
    e_0p += np.einsum("ia,izaw,wz", h1eff_ce_so, t_ixay, rdm_ca_so, optimize = True)
    e_0p += np.einsum("ixay,ia,xy", v_caea_so, t_ia, rdm_ca_so, optimize = True)
    e_0p += np.einsum("ixay,izaw,xywz", v_caea_so, t_ixay, rdm_caca_so, optimize = True)

    return e_0p, t_ia, t_ixay

def compute_t1_0p(mr_adc):

    # Einsum definition from kernel
    einsum = mr_adc.interface.einsum
    einsum_type = mr_adc.interface.einsum_type

    # Variables from kernel
    K_caca = mr_adc_intermediates.compute_K_caca(mr_adc)

    rdm_ca = mr_adc.rdm.ca
    rdm_ccaa = mr_adc.rdm.ccaa

    e_core = mr_adc.mo_energy.c
    e_extern = mr_adc.mo_energy.e

    ncore = mr_adc.ncore
    ncas = mr_adc.ncas
    nocc = mr_adc.nocc
    nextern = mr_adc.nextern

    # Orthogonalization and overlap truncation only in the active space
    S_0p_12_inv_act = mr_adc_overlap.compute_S12_0p_gno_projector(mr_adc, ignore_print = False)

    # Compute (S_12 K S_12)_{i a mu, j b nu}
    SKS = np.einsum("xywz,zwn->xyn", K_caca, S_0p_12_inv_act[1:,:].reshape(ncas, ncas, -1))
    SKS = np.einsum("xym,xyn->mn", S_0p_12_inv_act[1:,:].reshape(ncas, ncas, -1), SKS)

    evals, evecs = np.linalg.eigh(SKS)

    # Compute r.h.s. of the equation
    h_ce = mr_adc.h1e[:ncore,nocc:]
    v_caea = mr_adc.v2e.caea
    v_ccce = mr_adc.v2e.ccce

    import prism.mr_adc_integrals as mr_adc_integrals
    mo_c = mr_adc.mo[:, :mr_adc.ncore].copy()
    mo_a = mr_adc.mo[:, mr_adc.ncore:mr_adc.nocc].copy()
    mo_e = mr_adc.mo[:, mr_adc.nocc:].copy()
    v_caae = mr_adc_integrals.transform_2e_phys_incore(mr_adc.interface, mo_c, mo_a, mo_a, mo_e)
    v_acae = mr_adc_integrals.transform_2e_phys_incore(mr_adc.interface, mo_a, mo_c, mo_a, mo_e)
    v_ccec = mr_adc_integrals.transform_2e_phys_incore(mr_adc.interface, mo_c, mo_c, mo_e, mo_c)

    V0p = np.zeros((ncore, nextern, ncas * ncas + 1))

    V0p_ce  = einsum('IA->IA', h_ce, optimize = einsum_type).copy()
    V0p_ce -= einsum('IjjA->IA', v_ccce, optimize = einsum_type).copy()
    V0p_ce += 2 * einsum('jIjA->IA', v_ccce, optimize = einsum_type).copy()
    V0p_ce -= 1/2 * einsum('IxyA,yx->IA', v_caae, rdm_ca, optimize = einsum_type)
    V0p_ce += einsum('xIyA,yx->IA', v_acae, rdm_ca, optimize = einsum_type)
    V0p[:,:,0] = V0p_ce.copy()

    # Spin-summed
    V0p_ceaa =- einsum('IA,XY->IAXY', h_ce, rdm_ca, optimize = einsum_type)
    V0p_ceaa -= einsum('IjAj,XY->IAXY', v_ccec, rdm_ca, optimize = einsum_type)
    V0p_ceaa += einsum('IjjA,XY->IAXY', v_ccce, rdm_ca, optimize = einsum_type)
    V0p_ceaa -= 1/2 * einsum('IzAY,Xz->IAXY', v_caea, rdm_ca, optimize = einsum_type)
    V0p_ceaa -= 1/2 * einsum('IzAw,XwYz->IAXY', v_caea, rdm_ccaa, optimize = einsum_type)
    V0p_ceaa += 1/2 * einsum('IzYA,Xz->IAXY', v_caae, rdm_ca, optimize = einsum_type)
    V0p_ceaa += 1/6 * einsum('IzwA,XwYz->IAXY', v_caae, rdm_ccaa, optimize = einsum_type)
    V0p_ceaa -= 1/6 * einsum('IzwA,XwzY->IAXY', v_caae, rdm_ccaa, optimize = einsum_type)
    V0p_ceaa += 1/6 * einsum('IzwA,wXYz->IAXY', v_caae, rdm_ccaa, optimize = einsum_type)
    V0p_ceaa += 1/3 * einsum('IzwA,wXzY->IAXY', v_caae, rdm_ccaa, optimize = einsum_type)
    V0p_ceaa -= einsum('jIjA,XY->IAXY', v_ccce, rdm_ca, optimize = einsum_type)
    V0p_ceaa -= 1/2 * einsum('zIYA,Xz->IAXY', v_acae, rdm_ca, optimize = einsum_type)
    V0p_ceaa -= 1/6 * einsum('zIwA,XwYz->IAXY', v_acae, rdm_ccaa, optimize = einsum_type)
    V0p_ceaa += 1/6 * einsum('zIwA,XwzY->IAXY', v_acae, rdm_ccaa, optimize = einsum_type)
    V0p_ceaa -= 1/6 * einsum('zIwA,wXYz->IAXY', v_acae, rdm_ccaa, optimize = einsum_type)
    V0p_ceaa -= 1/3 * einsum('zIwA,wXzY->IAXY', v_acae, rdm_ccaa, optimize = einsum_type)

    V0p[:,:,1:] = V0p_ceaa.reshape(ncore, nextern, -1)

    V0p = V0p.reshape(ncore, nextern, -1)

    S_12_V0p = np.einsum("iaP,Pm->iam", V0p, S_0p_12_inv_act)

    # Multiply r.h.s. by U (e_a - e_i + e_mu)^-1 U^dag
    S_12_V0p = np.einsum("mp,iam->iap", evecs, S_12_V0p)

    # Compute denominators
    d_ai = (e_extern[:,None] - e_core).reshape(-1)

    d_aip = (d_ai[:,None] + evals).reshape(nextern, ncore, -1)
    d_aip = d_aip**(-1)

    S_12_V0p = np.einsum("aip,iap->iap", d_aip, S_12_V0p)
    S_12_V0p = np.einsum("mp,iap->iam", evecs, S_12_V0p)

    t0p = np.einsum("iam,Pm->iaP", S_12_V0p, S_0p_12_inv_act)

    t1_ce = t0p[:,:,0].copy()
    t1_caea = t0p[:,:,1:].reshape(ncore, nextern, ncas, ncas)
    t1_caea = t1_caea.transpose(0,2,1,3).copy()

    e_0p  = einsum('ia,ia', h_ce, t1_ce, optimize = einsum_type)
    e_0p += einsum('ia,ijaj', t1_ce, v_ccec, optimize = einsum_type)
    e_0p -= einsum('ia,ijja', t1_ce, v_ccce, optimize = einsum_type)
    e_0p += einsum('ia,jija', t1_ce, v_ccce, optimize = einsum_type)

    e_0p += 1/2 * einsum('ia,ixay,yx', h_ce, t1_caea, rdm_ca, optimize = einsum_type)
    e_0p += 1/2 * einsum('ixay,ijaj,yx', t1_caea, v_ccec, rdm_ca, optimize = einsum_type)
    e_0p -= 1/2 * einsum('ixay,ijja,yx', t1_caea, v_ccce, rdm_ca, optimize = einsum_type)
    e_0p += 1/2 * einsum('ixay,izay,zx', t1_caea, v_caea, rdm_ca, optimize = einsum_type)
    e_0p += 1/2 * einsum('ixay,jija,yx', t1_caea, v_ccce, rdm_ca, optimize = einsum_type)
    e_0p += 1/2 * einsum('ia,ixay,xy', t1_ce, v_caea, rdm_ca, optimize = einsum_type)
    e_0p -= 1/2 * einsum('ia,ixya,xy', t1_ce, v_caae, rdm_ca, optimize = einsum_type)
    e_0p += 1/2 * einsum('ia,xiya,xy', t1_ce, v_acae, rdm_ca, optimize = einsum_type)

    e_0p -= 1/6 * einsum('ixay,izaw,yzwx', t1_caea, v_caea, rdm_ccaa, optimize = einsum_type)
    e_0p += 1/6 * einsum('ixay,izaw,yzxw', t1_caea, v_caea, rdm_ccaa, optimize = einsum_type)
    e_0p -= 1/3 * einsum('ixay,izwa,zywx', t1_caea, v_caae, rdm_ccaa, optimize = einsum_type)
    e_0p -= 1/6 * einsum('ixay,izwa,zyxw', t1_caea, v_caae, rdm_ccaa, optimize = einsum_type)
    e_0p += 1/3 * einsum('ixay,ziwa,zywx', t1_caea, v_acae, rdm_ccaa, optimize = einsum_type)
    e_0p += 1/6 * einsum('ixay,ziwa,zyxw', t1_caea, v_acae, rdm_ccaa, optimize = einsum_type)

    return e_0p, t1_ce, t1_caea

def compute_t1_p1p(mr_adc):

    # Einsum definition from kernel
    einsum = mr_adc.interface.einsum
    einsum_type = mr_adc.interface.einsum_type

    # Variables from kernel
    K_p1p = mr_adc_intermediates.compute_K_p1p(mr_adc)

    rdm_ca = mr_adc.rdm.ca
    rdm_ccaa = mr_adc.rdm.ccaa
    rdm_cccaaa = mr_adc.rdm.cccaaa

    e_core = mr_adc.mo_energy.c

    ncore = mr_adc.ncore
    ncas = mr_adc.ncas
    nocc = mr_adc.nocc

    n_x = ncas
    n_zwx = ncas**3
    dim_act = n_x + n_zwx

    aa_ind = np.tril_indices(ncas, k=-1)

    S_p1p_12_inv_act = mr_adc_overlap.compute_S12_p1p(mr_adc, ignore_print = False, half_transform = True)
    # S_p1p_12_inv_act = mr_adc_overlap.compute_S12_p1p_gno_projector(mr_adc, ignore_print = False, half_transform = True)

    SKS = reduce(np.dot, (S_p1p_12_inv_act.T, K_p1p, S_p1p_12_inv_act))

    evals, evecs = np.linalg.eigh(SKS)

    # Compute r.h.s. of the equation
    h_ca = mr_adc.h1e[:ncore,ncore:nocc].copy()
    v_caaa = mr_adc.v2e.caaa
    v_ccca = mr_adc.v2e.ccca

    import prism.mr_adc_integrals as mr_adc_integrals
    mo_c = mr_adc.mo[:, :mr_adc.ncore].copy()
    mo_a = mr_adc.mo[:, mr_adc.ncore:mr_adc.nocc].copy()
    v_acaa = mr_adc_integrals.transform_2e_phys_incore(mr_adc.interface, mo_a, mo_c, mo_a, mo_a)

    V = np.zeros((ncore, dim_act))

    V1  = 2 * einsum('IX->IX', h_ca, optimize = einsum_type).copy()
    V1 -= 2 * einsum('IjjX->IX', v_ccca, optimize = einsum_type).copy()
    V1 += 4 * einsum('jIjX->IX', v_ccca, optimize = einsum_type).copy()
    V1 -= einsum('Iy,yX->IX', h_ca, rdm_ca, optimize = einsum_type)
    V1 += einsum('Ijjy,yX->IX', v_ccca, rdm_ca, optimize = einsum_type)
    V1 += 3/2 * einsum('IyXz,zy->IX', v_caaa, rdm_ca, optimize = einsum_type)
    V1 -= 1/12 * einsum('Iyzw,wzXy->IX', v_caaa, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/6 * einsum('Iyzw,wzyX->IX', v_caaa, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/3 * einsum('Iyzw,zwXy->IX', v_caaa, rdm_ccaa, optimize = einsum_type)
    V1 += 1/12 * einsum('Iyzw,zwyX->IX', v_caaa, rdm_ccaa, optimize = einsum_type)
    V1 -= 2 * einsum('jIjy,yX->IX', v_ccca, rdm_ca, optimize = einsum_type)
    V1 -= einsum('yIXz,zy->IX', v_acaa, rdm_ca, optimize = einsum_type)
    V1 += 1/2 * einsum('yIzX,zy->IX', v_acaa, rdm_ca, optimize = einsum_type)
    V1 -= 1/6 * einsum('yIzw,wzXy->IX', v_acaa, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/12 * einsum('yIzw,wzyX->IX', v_acaa, rdm_ccaa, optimize = einsum_type)
    V1 += 1/12 * einsum('yIzw,zwXy->IX', v_acaa, rdm_ccaa, optimize = einsum_type)
    V1 -= 1/3 * einsum('yIzw,zwyX->IX', v_acaa, rdm_ccaa, optimize = einsum_type)

    V2  = 2 * einsum('IV,UX->IUXV', h_ca, np.identity(ncas), optimize = einsum_type)
    V2 -= einsum('IV,UX->IUXV', h_ca, rdm_ca, optimize = einsum_type)
    V2 += 3/2 * einsum('IX,UV->IUXV', h_ca, rdm_ca, optimize = einsum_type)
    V2 -= 2/3 * einsum('Iy,UyVX->IUXV', h_ca, rdm_ccaa, optimize = einsum_type)
    V2 += 1/6 * einsum('Iy,UyXV->IUXV', h_ca, rdm_ccaa, optimize = einsum_type)
    V2 -= 2 * einsum('UX,IjjV->IUXV', np.identity(ncas), v_ccca, optimize = einsum_type)
    V2 += 4 * einsum('UX,jIjV->IUXV', np.identity(ncas), v_ccca, optimize = einsum_type)
    V2 += einsum('IjjV,UX->IUXV', v_ccca, rdm_ca, optimize = einsum_type)
    V2 -= 3/2 * einsum('IjjX,UV->IUXV', v_ccca, rdm_ca, optimize = einsum_type)
    V2 += 2/3 * einsum('Ijjy,UyVX->IUXV', v_ccca, rdm_ccaa, optimize = einsum_type)
    V2 -= 1/6 * einsum('Ijjy,UyXV->IUXV', v_ccca, rdm_ccaa, optimize = einsum_type)
    V2 -= einsum('IyVX,Uy->IUXV', v_caaa, rdm_ca, optimize = einsum_type)
    V2 -= 2/3 * einsum('IyVz,UzXy->IUXV', v_caaa, rdm_ccaa, optimize = einsum_type)
    V2 += 1/6 * einsum('IyVz,UzyX->IUXV', v_caaa, rdm_ccaa, optimize = einsum_type)
    V2 += 1/2 * einsum('IyXV,Uy->IUXV', v_caaa, rdm_ca, optimize = einsum_type)
    V2 += 7/6 * einsum('IyXz,UzVy->IUXV', v_caaa, rdm_ccaa, optimize = einsum_type)
    V2 -= 1/6 * einsum('IyXz,UzyV->IUXV', v_caaa, rdm_ccaa, optimize = einsum_type)
    V2 -= 1/6 * einsum('IyzV,UzXy->IUXV', v_caaa, rdm_ccaa, optimize = einsum_type)
    V2 -= 1/3 * einsum('IyzV,UzyX->IUXV', v_caaa, rdm_ccaa, optimize = einsum_type)
    V2 -= 1/12 * einsum('Iyzw,UwzVyX->IUXV', v_caaa, rdm_cccaaa, optimize = einsum_type)
    V2 += 1/12 * einsum('Iyzw,UwzXVy->IUXV', v_caaa, rdm_cccaaa, optimize = einsum_type)
    V2 += 1/12 * einsum('Iyzw,UwzXyV->IUXV', v_caaa, rdm_cccaaa, optimize = einsum_type)
    V2 += 1/12 * einsum('Iyzw,UwzyVX->IUXV', v_caaa, rdm_cccaaa, optimize = einsum_type)
    V2 += 1/12 * einsum('Iyzw,UwzyXV->IUXV', v_caaa, rdm_cccaaa, optimize = einsum_type)
    V2 -= 1/4 * einsum('Iyzw,UzwVXy->IUXV', v_caaa, rdm_cccaaa, optimize = einsum_type)
    V2 += 1/24 * einsum('Iyzw,UzwVyX->IUXV', v_caaa, rdm_cccaaa, optimize = einsum_type)
    V2 += 1/24 * einsum('Iyzw,UzwXVy->IUXV', v_caaa, rdm_cccaaa, optimize = einsum_type)
    V2 += 1/24 * einsum('Iyzw,UzwyXV->IUXV', v_caaa, rdm_cccaaa, optimize = einsum_type)
    V2 -= 2 * einsum('jIjV,UX->IUXV', v_ccca, rdm_ca, optimize = einsum_type)
    V2 += 3 * einsum('jIjX,UV->IUXV', v_ccca, rdm_ca, optimize = einsum_type)
    V2 -= 4/3 * einsum('jIjy,UyVX->IUXV', v_ccca, rdm_ccaa, optimize = einsum_type)
    V2 += 1/3 * einsum('jIjy,UyXV->IUXV', v_ccca, rdm_ccaa, optimize = einsum_type)
    V2 += einsum('yIVX,Uy->IUXV', v_acaa, rdm_ca, optimize = einsum_type)
    V2 += 1/3 * einsum('yIVz,UzXy->IUXV', v_acaa, rdm_ccaa, optimize = einsum_type)
    V2 -= 1/3 * einsum('yIVz,UzyX->IUXV', v_acaa, rdm_ccaa, optimize = einsum_type)
    V2 -= 2/3 * einsum('yIXz,UzVy->IUXV', v_acaa, rdm_ccaa, optimize = einsum_type)
    V2 += 1/6 * einsum('yIXz,UzyV->IUXV', v_acaa, rdm_ccaa, optimize = einsum_type)
    V2 -= 1/3 * einsum('yIzV,UzXy->IUXV', v_acaa, rdm_ccaa, optimize = einsum_type)
    V2 -= 1/6 * einsum('yIzV,UzyX->IUXV', v_acaa, rdm_ccaa, optimize = einsum_type)
    V2 += 1/3 * einsum('yIzX,UzVy->IUXV', v_acaa, rdm_ccaa, optimize = einsum_type)
    V2 += 1/6 * einsum('yIzX,UzyV->IUXV', v_acaa, rdm_ccaa, optimize = einsum_type)
    V2 -= 1/12 * einsum('yIzw,UwzVXy->IUXV', v_acaa, rdm_cccaaa, optimize = einsum_type)
    V2 -= 1/24 * einsum('yIzw,UwzVyX->IUXV', v_acaa, rdm_cccaaa, optimize = einsum_type)
    V2 += 1/24 * einsum('yIzw,UwzXVy->IUXV', v_acaa, rdm_cccaaa, optimize = einsum_type)
    V2 -= 1/24 * einsum('yIzw,UwzyXV->IUXV', v_acaa, rdm_cccaaa, optimize = einsum_type)
    V2 += 1/12 * einsum('yIzw,UzwVXy->IUXV', v_acaa, rdm_cccaaa, optimize = einsum_type)
    V2 -= 1/6 * einsum('yIzw,UzwVyX->IUXV', v_acaa, rdm_cccaaa, optimize = einsum_type)
    V2 += 1/12 * einsum('yIzw,UzwXyV->IUXV', v_acaa, rdm_cccaaa, optimize = einsum_type)
    V2 -= einsum('Iy,UX,yV->IUXV', h_ca, np.identity(ncas), rdm_ca, optimize = einsum_type)
    V2 += einsum('UX,Ijjy,yV->IUXV', np.identity(ncas), v_ccca, rdm_ca, optimize = einsum_type)
    V2 += 3/2 * einsum('UX,IyVz,zy->IUXV', np.identity(ncas), v_caaa, rdm_ca, optimize = einsum_type)
    V2 -= 1/12 * einsum('UX,Iyzw,wzVy->IUXV', np.identity(ncas), v_caaa, rdm_ccaa, optimize = einsum_type)
    V2 -= 1/6 * einsum('UX,Iyzw,wzyV->IUXV', np.identity(ncas), v_caaa, rdm_ccaa, optimize = einsum_type)
    V2 -= 1/3 * einsum('UX,Iyzw,zwVy->IUXV', np.identity(ncas), v_caaa, rdm_ccaa, optimize = einsum_type)
    V2 += 1/12 * einsum('UX,Iyzw,zwyV->IUXV', np.identity(ncas), v_caaa, rdm_ccaa, optimize = einsum_type)
    V2 -= 2 * einsum('UX,jIjy,yV->IUXV', np.identity(ncas), v_ccca, rdm_ca, optimize = einsum_type)
    V2 -= einsum('UX,yIVz,zy->IUXV', np.identity(ncas), v_acaa, rdm_ca, optimize = einsum_type)
    V2 += 1/2 * einsum('UX,yIzV,zy->IUXV', np.identity(ncas), v_acaa, rdm_ca, optimize = einsum_type)
    V2 -= 1/6 * einsum('UX,yIzw,wzVy->IUXV', np.identity(ncas), v_acaa, rdm_ccaa, optimize = einsum_type)
    V2 -= 1/12 * einsum('UX,yIzw,wzyV->IUXV', np.identity(ncas), v_acaa, rdm_ccaa, optimize = einsum_type)
    V2 += 1/12 * einsum('UX,yIzw,zwVy->IUXV', np.identity(ncas), v_acaa, rdm_ccaa, optimize = einsum_type)
    V2 -= 1/3 * einsum('UX,yIzw,zwyV->IUXV', np.identity(ncas), v_acaa, rdm_ccaa, optimize = einsum_type)

    V[:,:n_x] = V1.copy()
    V[:,n_x:] = V2.reshape(ncore, n_zwx).copy()

    S_12_V = np.einsum("iP,Pm->im", - V, S_p1p_12_inv_act)

    # Multiply r.h.s. by U (- e_i + e_mu)^-1 U^dag
    S_12_V = np.einsum("mp,im->ip", evecs, S_12_V)

    # Compute denominators
    d_ip = (-e_core[:,None] + evals)
    d_ip = d_ip**(-1)

    S_12_V *= d_ip
    S_12_V = np.einsum("mp,ip->im", evecs, S_12_V)
    t_p1p = np.einsum("Pm,im->iP", S_p1p_12_inv_act, S_12_V)

    t1_ca = t_p1p[:,:n_x].copy()
    t1_caaa = t_p1p[:,n_x:].reshape(ncore, ncas, ncas, ncas).copy()

    # Transpose t2 indices to the conventional order
    t1_caaa = t1_caaa.transpose(0,1,3,2).copy()

    # Compute correlation energy contribution
    e_p1p  = 2 * einsum('ix,ix', h_ca, t1_ca, optimize = einsum_type)
    e_p1p -= 2 * einsum('ix,ijjx', t1_ca, v_ccca, optimize = einsum_type)
    e_p1p += 4 * einsum('ix,jijx', t1_ca, v_ccca, optimize = einsum_type)
    e_p1p -= einsum('ix,iy,yx', h_ca, t1_ca, rdm_ca, optimize = einsum_type)
    e_p1p += 1/6 * einsum('ix,iywz,zwxy', h_ca, t1_caaa, rdm_ccaa, optimize = einsum_type)
    e_p1p -= 1/6 * einsum('ix,iywz,zwyx', h_ca, t1_caaa, rdm_ccaa, optimize = einsum_type)
    e_p1p += 3/2 * einsum('ix,iyxz,zy', h_ca, t1_caaa, rdm_ca, optimize = einsum_type)
    e_p1p -= 1/2 * einsum('ix,iyzw,zwxy', h_ca, t1_caaa, rdm_ccaa, optimize = einsum_type)
    e_p1p -= einsum('ix,iyzx,zy', h_ca, t1_caaa, rdm_ca, optimize = einsum_type)
    e_p1p += einsum('ix,ijjy,xy', t1_ca, v_ccca, rdm_ca, optimize = einsum_type)
    e_p1p += 3/2 * einsum('ix,iyxz,yz', t1_ca, v_caaa, rdm_ca, optimize = einsum_type)
    e_p1p -= 1/2 * einsum('ix,iyzw,xyzw', t1_ca, v_caaa, rdm_ccaa, optimize = einsum_type)
    e_p1p -= 2 * einsum('ix,jijy,xy', t1_ca, v_ccca, rdm_ca, optimize = einsum_type)
    e_p1p -= einsum('ix,yixz,yz', t1_ca, v_acaa, rdm_ca, optimize = einsum_type)
    e_p1p -= 1/2 * einsum('ix,yizw,xywz', t1_ca, v_acaa, rdm_ccaa, optimize = einsum_type)
    e_p1p += 1/2 * einsum('ix,yizx,yz', t1_ca, v_acaa, rdm_ca, optimize = einsum_type)
    e_p1p += 1/2 * einsum('ixyz,ijjw,yzwx', t1_caaa, v_ccca, rdm_ccaa, optimize = einsum_type)
    e_p1p -= 3/2 * einsum('ixyz,ijjy,zx', t1_caaa, v_ccca, rdm_ca, optimize = einsum_type)
    e_p1p += 1/24 * einsum('ixyz,iwuv,yzwuvx', t1_caaa, v_caaa, rdm_cccaaa, optimize = einsum_type)
    e_p1p -= 5/24 * einsum('ixyz,iwuv,yzwuxv', t1_caaa, v_caaa, rdm_cccaaa, optimize = einsum_type)
    e_p1p += 1/24 * einsum('ixyz,iwuv,yzwvux', t1_caaa, v_caaa, rdm_cccaaa, optimize = einsum_type)
    e_p1p += 1/24 * einsum('ixyz,iwuv,yzwvxu', t1_caaa, v_caaa, rdm_cccaaa, optimize = einsum_type)
    e_p1p += 1/24 * einsum('ixyz,iwuv,yzwxuv', t1_caaa, v_caaa, rdm_cccaaa, optimize = einsum_type)
    e_p1p += 1/24 * einsum('ixyz,iwuv,yzwxvu', t1_caaa, v_caaa, rdm_cccaaa, optimize = einsum_type)
    e_p1p -= 1/12 * einsum('ixyz,iwyu,zwux', t1_caaa, v_caaa, rdm_ccaa, optimize = einsum_type)
    e_p1p += 13/12 * einsum('ixyz,iwyu,zwxu', t1_caaa, v_caaa, rdm_ccaa, optimize = einsum_type)
    e_p1p += 3/4 * einsum('ixyz,iwyz,wx', t1_caaa, v_caaa, rdm_ca, optimize = einsum_type)
    e_p1p -= einsum('ixyz,jijw,yzwx', t1_caaa, v_ccca, rdm_ccaa, optimize = einsum_type)
    e_p1p += 3 * einsum('ixyz,jijy,zx', t1_caaa, v_ccca, rdm_ca, optimize = einsum_type)
    e_p1p += 1/24 * einsum('ixyz,wiuv,yzwuvx', t1_caaa, v_acaa, rdm_cccaaa, optimize = einsum_type)
    e_p1p += 1/24 * einsum('ixyz,wiuv,yzwuxv', t1_caaa, v_acaa, rdm_cccaaa, optimize = einsum_type)
    e_p1p += 1/24 * einsum('ixyz,wiuv,yzwvux', t1_caaa, v_acaa, rdm_cccaaa, optimize = einsum_type)
    e_p1p -= 5/24 * einsum('ixyz,wiuv,yzwvxu', t1_caaa, v_acaa, rdm_cccaaa, optimize = einsum_type)
    e_p1p += 1/24 * einsum('ixyz,wiuv,yzwxuv', t1_caaa, v_acaa, rdm_cccaaa, optimize = einsum_type)
    e_p1p += 1/24 * einsum('ixyz,wiuv,yzwxvu', t1_caaa, v_acaa, rdm_cccaaa, optimize = einsum_type)
    e_p1p += 1/12 * einsum('ixyz,wiuy,zwux', t1_caaa, v_acaa, rdm_ccaa, optimize = einsum_type)
    e_p1p += 5/12 * einsum('ixyz,wiuy,zwxu', t1_caaa, v_acaa, rdm_ccaa, optimize = einsum_type)
    e_p1p += 1/6 * einsum('ixyz,wiyu,zwux', t1_caaa, v_acaa, rdm_ccaa, optimize = einsum_type)
    e_p1p -= 2/3 * einsum('ixyz,wiyu,zwxu', t1_caaa, v_acaa, rdm_ccaa, optimize = einsum_type)
    e_p1p -= 1/2 * einsum('ixyz,wiyz,wx', t1_caaa, v_acaa, rdm_ca, optimize = einsum_type)
    e_p1p += 1/4 * einsum('ixyz,wizy,wx', t1_caaa, v_acaa, rdm_ca, optimize = einsum_type)
    e_p1p -= 1/6 * einsum('ixzy,ijjw,yzwx', t1_caaa, v_ccca, rdm_ccaa, optimize = einsum_type)
    e_p1p += 1/6 * einsum('ixzy,ijjw,yzxw', t1_caaa, v_ccca, rdm_ccaa, optimize = einsum_type)
    e_p1p += einsum('ixzy,ijjy,zx', t1_caaa, v_ccca, rdm_ca, optimize = einsum_type)
    e_p1p += 1/48 * einsum('ixzy,iwuv,yzwuvx', t1_caaa, v_caaa, rdm_cccaaa, optimize = einsum_type)
    e_p1p += 5/48 * einsum('ixzy,iwuv,yzwuxv', t1_caaa, v_caaa, rdm_cccaaa, optimize = einsum_type)
    e_p1p += 1/48 * einsum('ixzy,iwuv,yzwvux', t1_caaa, v_caaa, rdm_cccaaa, optimize = einsum_type)
    e_p1p += 1/48 * einsum('ixzy,iwuv,yzwvxu', t1_caaa, v_caaa, rdm_cccaaa, optimize = einsum_type)
    e_p1p -= 1/16 * einsum('ixzy,iwuv,yzwxuv', t1_caaa, v_caaa, rdm_cccaaa, optimize = einsum_type)
    e_p1p += 1/48 * einsum('ixzy,iwuv,yzwxvu', t1_caaa, v_caaa, rdm_cccaaa, optimize = einsum_type)
    e_p1p -= 1/6 * einsum('ixzy,iwuy,zwux', t1_caaa, v_caaa, rdm_ccaa, optimize = einsum_type)
    e_p1p -= 1/12 * einsum('ixzy,iwuy,zwxu', t1_caaa, v_caaa, rdm_ccaa, optimize = einsum_type)
    e_p1p += 1/6 * einsum('ixzy,iwyu,zwux', t1_caaa, v_caaa, rdm_ccaa, optimize = einsum_type)
    e_p1p -= 2/3 * einsum('ixzy,iwyu,zwxu', t1_caaa, v_caaa, rdm_ccaa, optimize = einsum_type)
    e_p1p -= 1/2 * einsum('ixzy,iwyz,wx', t1_caaa, v_caaa, rdm_ca, optimize = einsum_type)
    e_p1p += 1/3 * einsum('ixzy,jijw,yzwx', t1_caaa, v_ccca, rdm_ccaa, optimize = einsum_type)
    e_p1p -= 1/3 * einsum('ixzy,jijw,yzxw', t1_caaa, v_ccca, rdm_ccaa, optimize = einsum_type)
    e_p1p -= 2 * einsum('ixzy,jijy,zx', t1_caaa, v_ccca, rdm_ca, optimize = einsum_type)
    e_p1p -= 1/48 * einsum('ixzy,wiuv,yzwuvx', t1_caaa, v_acaa, rdm_cccaaa, optimize = einsum_type)
    e_p1p -= 1/48 * einsum('ixzy,wiuv,yzwuxv', t1_caaa, v_acaa, rdm_cccaaa, optimize = einsum_type)
    e_p1p -= 1/48 * einsum('ixzy,wiuv,yzwvux', t1_caaa, v_acaa, rdm_cccaaa, optimize = einsum_type)
    e_p1p += 1/16 * einsum('ixzy,wiuv,yzwvxu', t1_caaa, v_acaa, rdm_cccaaa, optimize = einsum_type)
    e_p1p -= 1/48 * einsum('ixzy,wiuv,yzwxuv', t1_caaa, v_acaa, rdm_cccaaa, optimize = einsum_type)
    e_p1p -= 5/48 * einsum('ixzy,wiuv,yzwxvu', t1_caaa, v_acaa, rdm_cccaaa, optimize = einsum_type)
    e_p1p -= 1/6 * einsum('ixzy,wiuy,zwux', t1_caaa, v_acaa, rdm_ccaa, optimize = einsum_type)
    e_p1p -= 1/3 * einsum('ixzy,wiuy,zwxu', t1_caaa, v_acaa, rdm_ccaa, optimize = einsum_type)
    e_p1p -= 1/2 * einsum('ixzy,wiyu,zwux', t1_caaa, v_acaa, rdm_ccaa, optimize = einsum_type)
    e_p1p += 1/4 * einsum('ixzy,wiyu,zwxu', t1_caaa, v_acaa, rdm_ccaa, optimize = einsum_type)
    e_p1p += 1/2 * einsum('ixzy,wiyz,wx', t1_caaa, v_acaa, rdm_ca, optimize = einsum_type)

    return e_p1p, t1_ca, t1_caaa

def compute_t1_p1p_singles(mr_adc):

    # Einsum definition from kernel
    einsum = mr_adc.interface.einsum
    einsum_type = mr_adc.interface.einsum_type

    # Variables from kernel
    K_p1p = mr_adc_intermediates.compute_K_p1p_singles(mr_adc)

    rdm_ca = mr_adc.rdm.ca
    rdm_ccaa = mr_adc.rdm.ccaa

    e_core = mr_adc.mo_energy.c

    ncore = mr_adc.ncore
    nocc = mr_adc.nocc

    S_p1p_12_inv_act = mr_adc_overlap.compute_S12_p1p_singles(mr_adc, ignore_print = False)

    SKS = reduce(np.dot, (S_p1p_12_inv_act.T, K_p1p, S_p1p_12_inv_act))

    evals, evecs = np.linalg.eigh(SKS)

    # Compute r.h.s. of the equation
    h_ca = mr_adc.h1eff[:ncore,ncore:nocc].copy()
    v_caaa = mr_adc.v2e.caaa

    V1 =- 2 * einsum('IX->IX', h_ca, optimize = einsum_type).copy()
    V1 += einsum('Ix,xX->IX', h_ca, rdm_ca, optimize = einsum_type)
    V1 -= 2 * einsum('IxXy,yx->IX', v_caaa, rdm_ca, optimize = einsum_type)
    V1 += einsum('IxyX,yx->IX', v_caaa, rdm_ca, optimize = einsum_type)
    V1 += einsum('Ixyz,yzXx->IX', v_caaa, rdm_ccaa, optimize = einsum_type)

    S_12_V = np.einsum("iP,Pm->im", - V1, S_p1p_12_inv_act)

    # Multiply r.h.s. by U (- e_i + e_mu)^-1 U^dag
    S_12_V = np.einsum("mp,im->ip", evecs, S_12_V)

    # Compute denominators
    d_ip = (-e_core[:,None] + evals)
    d_ip = d_ip**(-1)

    S_12_V *= d_ip
    S_12_V = np.einsum("mp,ip->im", evecs, S_12_V)

    t1_ca = np.einsum("Pm,im->iP", S_p1p_12_inv_act, S_12_V)
    # if mr_adc.debug_mode:
    #     print (">>> SA t1_ca (singles only) norm: {:}".format(np.linalg.norm(t1_ca)))
    #     with open('SA_t1_ca_singles.out', 'w') as outfile:
    #         outfile.write(repr(t1_ca))

    #     with open('SA_evals_singles.out', 'w') as outfile:
    #         outfile.write(repr(evals))

    #     with open('SA_S_p1p_12_inv_act_singles.out', 'w') as outfile:
    #         outfile.write(repr(S_p1p_12_inv_act))

    return t1_ca

def compute_t1_p1p_singles_sanity_check(mr_adc):

    # Einsum definition from kernel
    einsum = mr_adc.interface.einsum
    einsum_type = mr_adc.interface.einsum_type

    # Variables from kernel
    K_p1p = mr_adc_intermediates.compute_K_p1p_singles_sanity_check(mr_adc)

    rdm_ca = mr_adc.rdm.ca
    rdm_ccaa = mr_adc.rdm.ccaa
    rdm_cccaaa = mr_adc.rdm.cccaaa

    e_core = mr_adc.mo_energy.c

    ncore = mr_adc.ncore
    ncas = mr_adc.ncas
    nocc = mr_adc.nocc

    n_x = ncas * 2
    n_xzw = ncas * 2 * ncas * 2 * (ncas * 2 - 1) // 2
    dim_act = n_x + n_xzw
    aa_ind = np.tril_indices(ncas * 2, k=-1)

    S_p1p_12_inv_act = mr_adc_overlap.compute_S12_p1p_singles_sanity_check_gno_projector(mr_adc, ignore_print = False)

    SKS = reduce(np.dot, (S_p1p_12_inv_act.T, K_p1p, S_p1p_12_inv_act))

    evals, evecs = np.linalg.eigh(SKS)

    # Compute r.h.s. of the equation
    h_ca = mr_adc.h1eff[:ncore,ncore:nocc].copy()
    v_caaa = mr_adc.v2e.caaa

    V1 = np.zeros((ncore * 2, ncas * 2))

    V1_a_a =- einsum('IX->IX', h_ca, optimize = einsum_type).copy()
    V1_a_a += 1/2 * einsum('Ix,xX->IX', h_ca, rdm_ca, optimize = einsum_type)
    V1_a_a -= einsum('IxXy,yx->IX', v_caaa, rdm_ca, optimize = einsum_type)
    V1_a_a += 1/2 * einsum('IxyX,yx->IX', v_caaa, rdm_ca, optimize = einsum_type)
    V1_a_a += 1/2 * einsum('Ixyz,yzXx->IX', v_caaa, rdm_ccaa, optimize = einsum_type)

    V1[::2,::2] = V1_a_a.copy()
    V1[1::2,1::2] = V1_a_a.copy()

    S_12_V = np.einsum("iP,Pm->im", - 1.0 * V1, S_p1p_12_inv_act)

    # Multiply r.h.s. by U (- e_i + e_mu)^-1 U^dag
    S_12_V = np.einsum("mp,im->ip", evecs, S_12_V)

    # Compute denominators
    e_core_so = np.zeros(ncore * 2)
    e_core_so[::2] = e_core.copy()
    e_core_so[1::2] = e_core.copy()

    d_ip = (-e_core_so[:,None] + evals)
    d_ip = d_ip**(-1)

    S_12_V *= d_ip
    S_12_V = np.einsum("mp,ip->im", evecs, S_12_V)
    t1_ca = np.einsum("Pm,im->iP", S_p1p_12_inv_act, S_12_V)

    # Transpose t2 indices to the conventional order

    t1_ca = t1_ca[::2,::2].copy()
    # if mr_adc.debug_mode:
    #     print (">>> SA->SO t1_ca norm: {:}".format(np.linalg.norm(t1_ca)))
    #     with open('SA_SO_t1_ca.out', 'w') as outfile:
    #         outfile.write(repr(t1_ca))

    #     with open('SA_SO_evals.out', 'w') as outfile:
    #         outfile.write(repr(evals))

    #     with open('SA_SO_S_p1p_12_inv_act.out', 'w') as outfile:
    #         outfile.write(repr(S_p1p_12_inv_act))

    return t1_ca

def compute_t1_p1p_definition(mr_adc):

    # Einsum definition from kernel
    einsum = mr_adc.interface.einsum
    einsum_type = mr_adc.interface.einsum_type

    # Variables from kernel
    K_p1p = mr_adc_intermediates.compute_K_p1p_definition(mr_adc)

    rdm_ca = mr_adc.rdm.ca
    rdm_ccaa = mr_adc.rdm.ccaa
    rdm_cccaaa = mr_adc.rdm.cccaaa

    e_core = mr_adc.mo_energy.c
    h_ca = mr_adc.h1eff[:ncore,ncore:nocc]
    v_caaa = mr_adc.v_caaa

    ncore = mr_adc.ncore
    ncas = mr_adc.ncas
    nocc = mr_adc.nocc

    S_p1p_act = mr_adc_overlap.compute_S12_p1p_definition(mr_adc, ignore_print = False, half_transform = True)
    # S_p1p_12_inv_act = mr_adc_overlap.compute_S12_p1p_gno_projector(mr_adc, ignore_print = False, half_transform = True)

    KS = reduce(np.dot, (K_p1p, - 1.0 * S_p1p_act))

    evals, evecs = np.linalg.eigh(KS)

    S12 =- einsum('WYZX->XZWY', rdm_ccaa, optimize = einsum_type).copy()
    S12 -= einsum('WX,YZ->XZWY', np.identity(ncas), rdm_ca, optimize = einsum_type)
    S12 += 2 * einsum('XY,WZ->XZWY', np.identity(ncas), rdm_ca, optimize = einsum_type)
    S21 = S12.T.copy()

    S_abc  = np.einsum("mp,ip->ip", evecs, S_p1p_act)
    S_abc = np.einsum('icba,am->im', v_caaa, S_abc)

    S_a = np.einsum("mp,ip->ip", evecs, S21)
    S_a = np.einsum('ia,am->im', h_ca, S_a)

    amp = (S_a + S_abc)**2

    d_ip = (-e_core[:,None] + evals)
    d_ip = d_ip**(-1)

    e_p1p = np.einsum('im,im', amp, d_ip)

    return e_p1p
