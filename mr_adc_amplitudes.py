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
    # t2_amp = compute_t2_amplitudes(mr_adc, t1_amp)

    t1_ce, t1_ca, t1_ae, t1_caea, t1_caae, t1_caaa, t1_aaea, t1_ccee, t1_ccea, t1_caee, t1_ccaa, t1_aaee = t1_amp
    # t2_ce, t2_ca, t2_ae, t2_caea, t2_caaa, t2_aaea, t2_ccee, t2_ccea, t2_caee, t2_ccaa, t2_aaee, t2_aa = t2_amp

    print ("Time for computing amplitudes:                    %f sec\n" % (time.time() - start_time))

    # return (t1_ce, t1_ca, t1_ae, t1_caea, t1_caae, t1_caaa, t1_aaea, t1_ccee, t1_ccea, t1_caee, t1_ccaa, t1_aaee,
    #         t2_ce, t2_ca, t2_ae, t2_caea, t2_caea, t2_caaa, t2_aaea, t2_ccee, t2_ccea, t2_caee, t2_ccaa, t2_aaee, t2_aa)

    return (t1_ce, t1_ca, t1_ae, t1_caea, t1_caae, t1_caaa, t1_aaea, t1_ccee, t1_ccea, t1_caee, t1_ccaa, t1_aaee)

def compute_t1_amplitudes(mr_adc):

    t1_ce, t1_ca, t1_ae = (None,) * 3
    t1_caea, t1_caaa, t1_aaea = (None,) * 3
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
            e_m1p, t1_ae, t1_aaea = compute_t1_m1p_sanity_check(mr_adc)
            print ("Norm of T[-1']^(1):                         %20.12f" % (np.linalg.norm(t1_ae) + np.linalg.norm(t1_aaea)))
            print ("Correlation energy [-1']:                   %20.12f\n" % e_m1p)
        else:
            t1_ae = np.zeros((ncas, nextern))
            t1_aaea = np.zeros((ncas, ncas, nextern, ncas))

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
    print ("CASSCF reference energy:                     %20.12f" % mr_adc.e_casscf)
    print ("PC-NEVPT2 correlation energy:                %20.12f" % e_corr)
    print ("Total PC-NEVPT2 energy:                      %20.12f\n" % e_tot)

    t1_amp = (t1_ce, t1_ca, t1_ae, t1_caea, t1_caae, t1_caaa, t1_aaea, t1_ccee, t1_ccea, t1_caee, t1_ccaa, t1_aaee)

    return t1_amp

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

    d_ij = e_core[:,None] + e_core
    d_ab = e_extern[:,None] + e_extern
    D2 = -d_ij.reshape(-1,1) + d_ab.reshape(-1)
    D2 = D2.reshape((ncore, ncore, nextern, nextern))

    V1  = 1/4 * einsum('IJAB->IJAB', v_ccee, optimize = einsum_type).copy()

    t1_0 = - (V1/D2).copy()

    t1_ccee = t1_0.copy()
    e_0  = 4 * einsum('ijab,ijab', t1_ccee, v_ccee, optimize = einsum_type)
    e_0 -= 2 * einsum('ijab,ijba', t1_ccee, v_ccee, optimize = einsum_type)
    e_0 -= 2 * einsum('ijba,ijab', t1_ccee, v_ccee, optimize = einsum_type)
    e_0 += 4 * einsum('ijba,ijba', t1_ccee, v_ccee, optimize = einsum_type)

    return e_0, t1_0

def compute_t1_p1(mr_adc):

    # Einsum definition from kernel
    einsum = mr_adc.interface.einsum
    einsum_type = mr_adc.interface.einsum_type

    # Variables from kernel
    rdm_ca = mr_adc.rdm.ca
    v_ccea = mr_adc.v2e.ccea
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

    S_12_Vp1 = np.einsum("IJAX,Xm->IJAm", - Vp1, S_p1_12_inv_act)

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
    e_p1 -= 2 * einsum('ijax,jiax', t1_ccea, v_ccea, optimize = einsum_type)
    e_p1 -= 2 * einsum('ijax,jiya,xy', t1_ccea, v_ccae, rdm_ca, optimize = einsum_type)
    e_p1 += einsum('ijax,jiay,xy', t1_ccea, v_ccea, rdm_ca, optimize = einsum_type)

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

    S_12_Vm1 = np.einsum("IXAB,Xm->ImAB", - Vm1, S_m1_12_inv_act)

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

    e_m1  = 2 * einsum('ixab,iyab,xy', t1_caee, v_caee, rdm_ca, optimize = einsum_type)
    e_m1 -= einsum('ixab,iyba,xy', t1_caee, v_caee, rdm_ca, optimize = einsum_type)

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

    S_12_Vp2 = np.einsum("IJX,Xm->IJm", - Vp2, S_p2_12_inv_act)

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

    e_p2  = 2 * einsum('ijxy,jiyx', t1_ccaa, v_ccaa, optimize = einsum_type)
    e_p2 -= einsum('ijxy,jixy', t1_ccaa, v_ccaa, optimize = einsum_type)
    e_p2 -= einsum('ijxy,jizx,yz', t1_ccaa, v_ccaa, rdm_ca, optimize = einsum_type)
    e_p2 += 1/2 * einsum('ijxy,jizy,xz', t1_ccaa, v_ccaa, rdm_ca, optimize = einsum_type)
    e_p2 += 1/2 * einsum('ijxy,jixz,yz', t1_ccaa, v_ccaa, rdm_ca, optimize = einsum_type)
    e_p2 -= einsum('ijxy,jiyz,xz', t1_ccaa, v_ccaa, rdm_ca, optimize = einsum_type)
    e_p2 += 1/2 * einsum('ijxy,jiwz,xyzw', t1_ccaa, v_ccaa, rdm_ccaa, optimize = einsum_type)

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

    S_12_Vm2 = np.einsum("XAB,Xm->mAB", - Vm2, S_m2_12_inv_act)

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

def compute_t1_p1p_sanity_check(mr_adc):

    # Einsum definition from kernel
    einsum = mr_adc.interface.einsum
    einsum_type = mr_adc.interface.einsum_type

    # Variables from kernel
    K_p1p = mr_adc_intermediates.compute_K_p1p_sanity_check(mr_adc)

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

    S_p1p_12_inv_act = mr_adc_overlap.compute_S12_p1p_sanity_check(mr_adc, ignore_print = False, half_transform = True)

    SKS = reduce(np.dot, (S_p1p_12_inv_act.T, K_p1p, S_p1p_12_inv_act))

    evals, evecs = np.linalg.eigh(SKS)

    # Compute r.h.s. of the equation
    h_ca = mr_adc.h1e[:ncore,ncore:nocc].copy()
    v_caaa = mr_adc.v2e.caaa
    v_ccca = mr_adc.v2e.ccca

    V = np.zeros((ncore * 2, dim_act))

    V1 = np.zeros((ncore * 2, ncas * 2))

    V1_a_a =- einsum('IX->IX', h_ca, optimize = einsum_type).copy()
    V1_a_a -= 2 * einsum('iIiX->IX', v_ccca, optimize = einsum_type).copy()
    V1_a_a += einsum('IiiX->IX', v_ccca, optimize = einsum_type).copy()
    V1_a_a += 1/2 * einsum('Ix,Xx->IX', h_ca, rdm_ca, optimize = einsum_type)
    V1_a_a -= einsum('IxXy,xy->IX', v_caaa, rdm_ca, optimize = einsum_type)
    V1_a_a += 1/2 * einsum('IxyX,xy->IX', v_caaa, rdm_ca, optimize = einsum_type)
    V1_a_a += 1/2 * einsum('Ixyz,Xxyz->IX', v_caaa, rdm_ccaa, optimize = einsum_type)
    V1_a_a -= 1/2 * einsum('Iiix,Xx->IX', v_ccca, rdm_ca, optimize = einsum_type)
    V1_a_a += einsum('iIix,Xx->IX', v_ccca, rdm_ca, optimize = einsum_type)

    V1[::2,::2] = V1_a_a.copy()
    V1[1::2,1::2] = V1_a_a.copy()

    V2 = np.zeros((ncore * 2, ncas * 2, ncas * 2, ncas * 2))

    V2_aa_aa =- einsum('IV,UX->IUXV', h_ca, np.identity(ncas), optimize = einsum_type)
    V2_aa_aa += 1/2 * einsum('IV,UX->IUXV', h_ca, rdm_ca, optimize = einsum_type)
    V2_aa_aa -= 1/2 * einsum('IX,UV->IUXV', h_ca, rdm_ca, optimize = einsum_type)
    V2_aa_aa += 1/6 * einsum('Ix,UxVX->IUXV', h_ca, rdm_ccaa, optimize = einsum_type)
    V2_aa_aa -= 1/6 * einsum('Ix,UxXV->IUXV', h_ca, rdm_ccaa, optimize = einsum_type)
    V2_aa_aa -= 2 * einsum('UX,iIiV->IUXV', np.identity(ncas), v_ccca, optimize = einsum_type)
    V2_aa_aa += einsum('UX,IiiV->IUXV', np.identity(ncas), v_ccca, optimize = einsum_type)
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
    V2_aa_aa += einsum('iIiV,UX->IUXV', v_ccca, rdm_ca, optimize = einsum_type)
    V2_aa_aa -= einsum('iIiX,UV->IUXV', v_ccca, rdm_ca, optimize = einsum_type)
    V2_aa_aa -= 1/2 * einsum('IiiV,UX->IUXV', v_ccca, rdm_ca, optimize = einsum_type)
    V2_aa_aa += 1/2 * einsum('IiiX,UV->IUXV', v_ccca, rdm_ca, optimize = einsum_type)
    V2_aa_aa -= 1/6 * einsum('Iiix,UxVX->IUXV', v_ccca, rdm_ccaa, optimize = einsum_type)
    V2_aa_aa += 1/6 * einsum('Iiix,UxXV->IUXV', v_ccca, rdm_ccaa, optimize = einsum_type)
    V2_aa_aa += 1/3 * einsum('iIix,UxVX->IUXV', v_ccca, rdm_ccaa, optimize = einsum_type)
    V2_aa_aa -= 1/3 * einsum('iIix,UxXV->IUXV', v_ccca, rdm_ccaa, optimize = einsum_type)
    V2_aa_aa += 1/2 * einsum('Ix,UX,Vx->IUXV', h_ca, np.identity(ncas), rdm_ca, optimize = einsum_type)
    V2_aa_aa -= einsum('UX,IxVy,xy->IUXV', np.identity(ncas), v_caaa, rdm_ca, optimize = einsum_type)
    V2_aa_aa += 1/2 * einsum('UX,IxyV,xy->IUXV', np.identity(ncas), v_caaa, rdm_ca, optimize = einsum_type)
    V2_aa_aa += 1/2 * einsum('UX,Ixyz,Vxyz->IUXV', np.identity(ncas), v_caaa, rdm_ccaa, optimize = einsum_type)
    V2_aa_aa -= 1/2 * einsum('UX,Iiix,Vx->IUXV', np.identity(ncas), v_ccca, rdm_ca, optimize = einsum_type)
    V2_aa_aa += einsum('UX,iIix,Vx->IUXV', np.identity(ncas), v_ccca, rdm_ca, optimize = einsum_type)

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
    V2_ab_ab -= einsum('iIiX,UV->IUXV', v_ccca, rdm_ca, optimize = einsum_type)
    V2_ab_ab += 1/2 * einsum('IiiX,UV->IUXV', v_ccca, rdm_ca, optimize = einsum_type)
    V2_ab_ab -= 1/3 * einsum('Iiix,UxVX->IUXV', v_ccca, rdm_ccaa, optimize = einsum_type)
    V2_ab_ab -= 1/6 * einsum('Iiix,UxXV->IUXV', v_ccca, rdm_ccaa, optimize = einsum_type)
    V2_ab_ab += 2/3 * einsum('iIix,UxVX->IUXV', v_ccca, rdm_ccaa, optimize = einsum_type)
    V2_ab_ab += 1/3 * einsum('iIix,UxXV->IUXV', v_ccca, rdm_ccaa, optimize = einsum_type)

    V2_ab_ba =- einsum('IV,UX->IUXV', h_ca, np.identity(ncas), optimize = einsum_type)
    V2_ab_ba += 1/2 * einsum('IV,UX->IUXV', h_ca, rdm_ca, optimize = einsum_type)
    V2_ab_ba -= 1/6 * einsum('Ix,UxVX->IUXV', h_ca, rdm_ccaa, optimize = einsum_type)
    V2_ab_ba -= 1/3 * einsum('Ix,UxXV->IUXV', h_ca, rdm_ccaa, optimize = einsum_type)
    V2_ab_ba -= 2 * einsum('UX,iIiV->IUXV', np.identity(ncas), v_ccca, optimize = einsum_type)
    V2_ab_ba += einsum('UX,IiiV->IUXV', np.identity(ncas), v_ccca, optimize = einsum_type)
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
    V2_ab_ba += einsum('iIiV,UX->IUXV', v_ccca, rdm_ca, optimize = einsum_type)
    V2_ab_ba -= 1/2 * einsum('IiiV,UX->IUXV', v_ccca, rdm_ca, optimize = einsum_type)
    V2_ab_ba += 1/6 * einsum('Iiix,UxVX->IUXV', v_ccca, rdm_ccaa, optimize = einsum_type)
    V2_ab_ba += 1/3 * einsum('Iiix,UxXV->IUXV', v_ccca, rdm_ccaa, optimize = einsum_type)
    V2_ab_ba -= 1/3 * einsum('iIix,UxVX->IUXV', v_ccca, rdm_ccaa, optimize = einsum_type)
    V2_ab_ba -= 2/3 * einsum('iIix,UxXV->IUXV', v_ccca, rdm_ccaa, optimize = einsum_type)
    V2_ab_ba += 1/2 * einsum('Ix,UX,Vx->IUXV', h_ca, np.identity(ncas), rdm_ca, optimize = einsum_type)
    V2_ab_ba -= einsum('UX,IxVy,xy->IUXV', np.identity(ncas), v_caaa, rdm_ca, optimize = einsum_type)
    V2_ab_ba += 1/2 * einsum('UX,IxyV,xy->IUXV', np.identity(ncas), v_caaa, rdm_ca, optimize = einsum_type)
    V2_ab_ba += 1/2 * einsum('UX,Ixyz,Vxyz->IUXV', np.identity(ncas), v_caaa, rdm_ccaa, optimize = einsum_type)
    V2_ab_ba -= 1/2 * einsum('UX,Iiix,Vx->IUXV', np.identity(ncas), v_ccca, rdm_ca, optimize = einsum_type)
    V2_ab_ba += einsum('UX,iIix,Vx->IUXV', np.identity(ncas), v_ccca, rdm_ca, optimize = einsum_type)

    V2[::2,::2,::2,::2,] = V2_aa_aa.copy()
    V2[1::2,1::2,1::2,1::2] = V2_aa_aa.copy()

    V2[::2,1::2,::2,1::2] = V2_ab_ab.copy()
    V2[1::2,::2,1::2,::2] = V2_ab_ab.copy()

    V2[::2,1::2,1::2,::2] = V2_ab_ba.copy()
    V2[1::2,::2,::2,1::2] = V2_ab_ba.copy()
    V2 = V2[:,:,aa_ind[0],aa_ind[1]].reshape(ncore * 2, -1).copy()

    V[:,:n_x] = V1.copy()
    V[:,n_x:] = V2.copy()

    S_12_V = np.einsum("iP,Pm->im", - V, S_p1p_12_inv_act)

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

    e_p1p  = 2 * einsum('ix,ix', h_ca, t1_ca, optimize = einsum_type)
    e_p1p += 4 * einsum('ix,jijx', t1_ca, v_ccca, optimize = einsum_type)
    e_p1p -= 2 * einsum('ix,ijjx', t1_ca, v_ccca, optimize = einsum_type)
    e_p1p -= einsum('ix,iy,xy', h_ca, t1_ca, rdm_ca, optimize = einsum_type)
    e_p1p += 2 * einsum('ix,iyxz,yz', h_ca, t1_caaa, rdm_ca, optimize = einsum_type)
    e_p1p -= einsum('ix,iyzx,yz', h_ca, t1_caaa, rdm_ca, optimize = einsum_type)
    e_p1p -= einsum('ix,iyzw,xyzw', h_ca, t1_caaa, rdm_ccaa, optimize = einsum_type)
    e_p1p += 2 * einsum('ix,iyxz,yz', t1_ca, v_caaa, rdm_ca, optimize = einsum_type)
    e_p1p -= einsum('ix,iyzx,yz', t1_ca, v_caaa, rdm_ca, optimize = einsum_type)
    e_p1p -= einsum('ix,iyzw,xyzw', t1_ca, v_caaa, rdm_ccaa, optimize = einsum_type)
    e_p1p += einsum('ix,ijjy,xy', t1_ca, v_ccca, rdm_ca, optimize = einsum_type)
    e_p1p -= 2 * einsum('ix,jijy,xy', t1_ca, v_ccca, rdm_ca, optimize = einsum_type)
    e_p1p += 2 * einsum('ixyz,iwyz,xw', t1_caaa, v_caaa, rdm_ca, optimize = einsum_type)
    e_p1p += 2 * einsum('ixyz,iwyu,xuzw', t1_caaa, v_caaa, rdm_ccaa, optimize = einsum_type)
    e_p1p -= einsum('ixyz,iwzy,xw', t1_caaa, v_caaa, rdm_ca, optimize = einsum_type)
    e_p1p -= einsum('ixyz,iwzu,xuyw', t1_caaa, v_caaa, rdm_ccaa, optimize = einsum_type)
    e_p1p -= einsum('ixyz,iwuy,xuzw', t1_caaa, v_caaa, rdm_ccaa, optimize = einsum_type)
    e_p1p -= einsum('ixyz,iwuz,xuwy', t1_caaa, v_caaa, rdm_ccaa, optimize = einsum_type)
    e_p1p += 1/6 * einsum('ixyz,iwuv,xuvyzw', t1_caaa, v_caaa, rdm_cccaaa, optimize = einsum_type)
    e_p1p += 1/6 * einsum('ixyz,iwuv,xuvywz', t1_caaa, v_caaa, rdm_cccaaa, optimize = einsum_type)
    e_p1p -= 5/6 * einsum('ixyz,iwuv,xuvzyw', t1_caaa, v_caaa, rdm_cccaaa, optimize = einsum_type)
    e_p1p += 1/6 * einsum('ixyz,iwuv,xuvzwy', t1_caaa, v_caaa, rdm_cccaaa, optimize = einsum_type)
    e_p1p += 1/6 * einsum('ixyz,iwuv,xuvwyz', t1_caaa, v_caaa, rdm_cccaaa, optimize = einsum_type)
    e_p1p += 1/6 * einsum('ixyz,iwuv,xuvwzy', t1_caaa, v_caaa, rdm_cccaaa, optimize = einsum_type)
    e_p1p += 4 * einsum('ixyz,jijy,xz', t1_caaa, v_ccca, rdm_ca, optimize = einsum_type)
    e_p1p -= 2 * einsum('ixyz,jijz,xy', t1_caaa, v_ccca, rdm_ca, optimize = einsum_type)
    e_p1p -= 2 * einsum('ixyz,ijjy,xz', t1_caaa, v_ccca, rdm_ca, optimize = einsum_type)
    e_p1p += einsum('ixyz,ijjz,xy', t1_caaa, v_ccca, rdm_ca, optimize = einsum_type)
    e_p1p += einsum('ixyz,ijjw,xwzy', t1_caaa, v_ccca, rdm_ccaa, optimize = einsum_type)
    e_p1p -= 2 * einsum('ixyz,jijw,xwzy', t1_caaa, v_ccca, rdm_ccaa, optimize = einsum_type)

    return e_p1p, t1_ca, t1_caaa

def compute_t1_m1p_sanity_check(mr_adc):

    # Einsum definition from kernel
    einsum = mr_adc.interface.einsum
    einsum_type = mr_adc.interface.einsum_type

    # Variables from kernel
    K_m1p = mr_adc_intermediates.compute_K_m1p_sanity_check(mr_adc)

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

    S_m1p_12_inv_act = mr_adc_overlap.compute_S12_m1p_sanity_check(mr_adc, ignore_print = False, half_transform = True)
    SKS = reduce(np.dot, (S_m1p_12_inv_act.T, K_m1p, S_m1p_12_inv_act))
    evals, evecs = np.linalg.eigh(SKS)

    # Compute r.h.s. of the equation
    h_ae = mr_adc.h1e[ncore:nocc,nocc:].copy()
    v_aaae = mr_adc.v2e.aaae
    v_cace = mr_adc.v2e.cace
    v_caec = mr_adc.v2e.caec

    V = np.zeros((dim_act, nextern * 2))

    V1 = np.zeros((ncas * 2, nextern * 2))

    V1_a_a =- 1/2 * einsum('xA,Xx->XA', h_ae, rdm_ca, optimize = einsum_type)
    V1_a_a -= 1/2 * einsum('yxzA,Xzxy->XA', v_aaae, rdm_ccaa, optimize = einsum_type)
    V1_a_a -= einsum('ixiA,Xx->XA', v_cace, rdm_ca, optimize = einsum_type)
    V1_a_a += 1/2 * einsum('ixAi,Xx->XA', v_caec, rdm_ca, optimize = einsum_type)

    V1[::2,::2] = V1_a_a.copy()
    V1[1::2,1::2] = V1_a_a.copy()

    V2 = np.zeros((ncas * 2, ncas * 2, ncas * 2, nextern * 2))

    V2_aa_aa  = 1/6 * einsum('xA,UXVx->XUVA', h_ae, rdm_ccaa, optimize = einsum_type)
    V2_aa_aa -= 1/6 * einsum('xA,UXxV->XUVA', h_ae, rdm_ccaa, optimize = einsum_type)
    V2_aa_aa -= 1/6 * einsum('yxVA,UXxy->XUVA', v_aaae, rdm_ccaa, optimize = einsum_type)
    V2_aa_aa += 1/6 * einsum('yxVA,UXyx->XUVA', v_aaae, rdm_ccaa, optimize = einsum_type)
    V2_aa_aa += 1/6 * einsum('yxzA,UXzVxy->XUVA', v_aaae, rdm_cccaaa, optimize = einsum_type)
    V2_aa_aa -= 1/6 * einsum('yxzA,UXzxVy->XUVA', v_aaae, rdm_cccaaa, optimize = einsum_type)
    V2_aa_aa += 1/3 * einsum('ixiA,UXVx->XUVA', v_cace, rdm_ccaa, optimize = einsum_type)
    V2_aa_aa -= 1/3 * einsum('ixiA,UXxV->XUVA', v_cace, rdm_ccaa, optimize = einsum_type)
    V2_aa_aa -= 1/6 * einsum('ixAi,UXVx->XUVA', v_caec, rdm_ccaa, optimize = einsum_type)
    V2_aa_aa += 1/6 * einsum('ixAi,UXxV->XUVA', v_caec, rdm_ccaa, optimize = einsum_type)
    V2_aa_aa -= 1/2 * einsum('xA,UV,Xx->XUVA', h_ae, np.identity(ncas), rdm_ca, optimize = einsum_type)
    V2_aa_aa -= 1/2 * einsum('UV,yxzA,Xzxy->XUVA', np.identity(ncas), v_aaae, rdm_ccaa, optimize = einsum_type)
    V2_aa_aa -= einsum('UV,ixiA,Xx->XUVA', np.identity(ncas), v_cace, rdm_ca, optimize = einsum_type)
    V2_aa_aa += 1/2 * einsum('UV,ixAi,Xx->XUVA', np.identity(ncas), v_caec, rdm_ca, optimize = einsum_type)

    V2_ab_ab =- 1/6 * einsum('xA,UXVx->XUVA', h_ae, rdm_ccaa, optimize = einsum_type)
    V2_ab_ab -= 1/3 * einsum('xA,UXxV->XUVA', h_ae, rdm_ccaa, optimize = einsum_type)
    V2_ab_ab -= 1/3 * einsum('yxVA,UXxy->XUVA', v_aaae, rdm_ccaa, optimize = einsum_type)
    V2_ab_ab -= 1/6 * einsum('yxVA,UXyx->XUVA', v_aaae, rdm_ccaa, optimize = einsum_type)
    V2_ab_ab -= 1/12 * einsum('yxzA,UXzVxy->XUVA', v_aaae, rdm_cccaaa, optimize = einsum_type)
    V2_ab_ab += 1/12 * einsum('yxzA,UXzVyx->XUVA', v_aaae, rdm_cccaaa, optimize = einsum_type)
    V2_ab_ab -= 1/4 * einsum('yxzA,UXzxVy->XUVA', v_aaae, rdm_cccaaa, optimize = einsum_type)
    V2_ab_ab += 1/12 * einsum('yxzA,UXzxyV->XUVA', v_aaae, rdm_cccaaa, optimize = einsum_type)
    V2_ab_ab += 1/12 * einsum('yxzA,UXzyVx->XUVA', v_aaae, rdm_cccaaa, optimize = einsum_type)
    V2_ab_ab += 1/12 * einsum('yxzA,UXzyxV->XUVA', v_aaae, rdm_cccaaa, optimize = einsum_type)
    V2_ab_ab -= 1/3 * einsum('ixiA,UXVx->XUVA', v_cace, rdm_ccaa, optimize = einsum_type)
    V2_ab_ab -= 2/3 * einsum('ixiA,UXxV->XUVA', v_cace, rdm_ccaa, optimize = einsum_type)
    V2_ab_ab += 1/6 * einsum('ixAi,UXVx->XUVA', v_caec, rdm_ccaa, optimize = einsum_type)
    V2_ab_ab += 1/3 * einsum('ixAi,UXxV->XUVA', v_caec, rdm_ccaa, optimize = einsum_type)

    V2_ab_ba  = 1/3 * einsum('xA,UXVx->XUVA', h_ae, rdm_ccaa, optimize = einsum_type)
    V2_ab_ba += 1/6 * einsum('xA,UXxV->XUVA', h_ae, rdm_ccaa, optimize = einsum_type)
    V2_ab_ba += 1/6 * einsum('yxVA,UXxy->XUVA', v_aaae, rdm_ccaa, optimize = einsum_type)
    V2_ab_ba += 1/3 * einsum('yxVA,UXyx->XUVA', v_aaae, rdm_ccaa, optimize = einsum_type)
    V2_ab_ba += 1/4 * einsum('yxzA,UXzVxy->XUVA', v_aaae, rdm_cccaaa, optimize = einsum_type)
    V2_ab_ba -= 1/12 * einsum('yxzA,UXzVyx->XUVA', v_aaae, rdm_cccaaa, optimize = einsum_type)
    V2_ab_ba += 1/12 * einsum('yxzA,UXzxVy->XUVA', v_aaae, rdm_cccaaa, optimize = einsum_type)
    V2_ab_ba -= 1/12 * einsum('yxzA,UXzxyV->XUVA', v_aaae, rdm_cccaaa, optimize = einsum_type)
    V2_ab_ba -= 1/12 * einsum('yxzA,UXzyVx->XUVA', v_aaae, rdm_cccaaa, optimize = einsum_type)
    V2_ab_ba -= 1/12 * einsum('yxzA,UXzyxV->XUVA', v_aaae, rdm_cccaaa, optimize = einsum_type)
    V2_ab_ba += 2/3 * einsum('ixiA,UXVx->XUVA', v_cace, rdm_ccaa, optimize = einsum_type)
    V2_ab_ba += 1/3 * einsum('ixiA,UXxV->XUVA', v_cace, rdm_ccaa, optimize = einsum_type)
    V2_ab_ba -= 1/3 * einsum('ixAi,UXVx->XUVA', v_caec, rdm_ccaa, optimize = einsum_type)
    V2_ab_ba -= 1/6 * einsum('ixAi,UXxV->XUVA', v_caec, rdm_ccaa, optimize = einsum_type)
    V2_ab_ba -= 1/2 * einsum('xA,UV,Xx->XUVA', h_ae, np.identity(ncas), rdm_ca, optimize = einsum_type)
    V2_ab_ba -= 1/2 * einsum('UV,yxzA,Xzxy->XUVA', np.identity(ncas), v_aaae, rdm_ccaa, optimize = einsum_type)
    V2_ab_ba -= einsum('UV,ixiA,Xx->XUVA', np.identity(ncas), v_cace, rdm_ca, optimize = einsum_type)
    V2_ab_ba += 1/2 * einsum('UV,ixAi,Xx->XUVA', np.identity(ncas), v_caec, rdm_ca, optimize = einsum_type)

    V2[::2,::2,::2,::2] = V2_aa_aa.copy()
    V2[1::2,1::2,1::2,1::2] = V2_aa_aa.copy()

    V2[::2,1::2,::2,1::2] = V2_ab_ab.copy()
    V2[1::2,::2,1::2,::2] = V2_ab_ab.copy()

    V2[::2,1::2,1::2,::2] = V2_ab_ba.copy()
    V2[1::2,::2,::2,1::2] = V2_ab_ba.copy()

    V2 = V2[aa_ind[0],aa_ind[1]].reshape(-1, nextern * 2).copy()

    V[:n_x,:] = V1.copy()
    V[n_x:,:] = V2.copy()

    S_12_V = np.einsum("Pa,Pm->ma", - V, S_m1p_12_inv_act)

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
    t1_aaea = t1_aaea[::2,1::2,::2,1::2].copy()

    # Compute correlation energy contribution
    e_m1p  = einsum('xa,ya,xy', h_ae, t1_ae, rdm_ca, optimize = einsum_type)
    e_m1p += einsum('xa,yzaw,xwyz', h_ae, t1_aaea, rdm_ccaa, optimize = einsum_type)
    e_m1p += einsum('xa,zywa,xwyz', t1_ae, v_aaae, rdm_ccaa, optimize = einsum_type)
    e_m1p += 2 * einsum('xa,iyia,xy', t1_ae, v_cace, rdm_ca, optimize = einsum_type)
    e_m1p -= einsum('xa,iyai,xy', t1_ae, v_caec, rdm_ca, optimize = einsum_type)
    e_m1p += einsum('xyaz,uwza,xywu', t1_aaea, v_aaae, rdm_ccaa, optimize = einsum_type)
    e_m1p -= 1/6 * einsum('xyaz,uwva,xyvzwu', t1_aaea, v_aaae, rdm_cccaaa, optimize = einsum_type)
    e_m1p -= 1/6 * einsum('xyaz,uwva,xyvzuw', t1_aaea, v_aaae, rdm_cccaaa, optimize = einsum_type)
    e_m1p += 5/6 * einsum('xyaz,uwva,xyvwzu', t1_aaea, v_aaae, rdm_cccaaa, optimize = einsum_type)
    e_m1p -= 1/6 * einsum('xyaz,uwva,xyvwuz', t1_aaea, v_aaae, rdm_cccaaa, optimize = einsum_type)
    e_m1p -= 1/6 * einsum('xyaz,uwva,xyvuzw', t1_aaea, v_aaae, rdm_cccaaa, optimize = einsum_type)
    e_m1p -= 1/6 * einsum('xyaz,uwva,xyvuwz', t1_aaea, v_aaae, rdm_cccaaa, optimize = einsum_type)
    e_m1p += 2 * einsum('xyaz,iwia,xywz', t1_aaea, v_cace, rdm_ccaa, optimize = einsum_type)
    e_m1p -= einsum('xyaz,iwai,xywz', t1_aaea, v_caec, rdm_ccaa, optimize = einsum_type)

    return e_m1p, t1_ae, t1_aaea

def compute_t1_0p_sanity_check(mr_adc):

    # Einsum definition from kernel
    einsum = mr_adc.interface.einsum
    einsum_type = mr_adc.interface.einsum_type

    # Variables from kernel
    K_caca = mr_adc_intermediates.compute_K_caca_sanity_check(mr_adc)

    ncore = mr_adc.ncore
    ncas = mr_adc.ncas
    nocc = mr_adc.nocc
    nextern = mr_adc.nextern

    h_ce = mr_adc.h1e[:ncore,nocc:]
    v_caea = mr_adc.v2e.caea
    v_caae = mr_adc.v2e.caae
    v_ccce = mr_adc.v2e.ccce

    rdm_ca = mr_adc.rdm.ca
    rdm_ccaa = mr_adc.rdm.ccaa

    e_core = mr_adc.mo_energy.c
    e_extern = mr_adc.mo_energy.e

    # Orthogonalization and overlap truncation only in the active space
    S_0p_12_inv_act = mr_adc_overlap.compute_S12_0p_sanity_check(mr_adc, ignore_print = False)

    # Compute (S_12 K S_12)_{i a mu, j b nu}
    SKS = np.einsum("xywz,zwn->xyn", K_caca, S_0p_12_inv_act[1:,:].reshape(ncas * 2, ncas * 2, -1))
    SKS = np.einsum("xym,xyn->mn", S_0p_12_inv_act[1:,:].reshape(ncas * 2, ncas * 2, -1), SKS)

    evals, evecs = np.linalg.eigh(SKS)

    # Compute r.h.s. of the equation
    V0p = np.zeros((ncore * 2, nextern * 2, ncas * 2 * ncas * 2 + 1))

    V1 = np.zeros((ncore * 2, nextern * 2))
    V1_a_a  = einsum('IA->IA', h_ce, optimize = einsum_type).copy()
    V1_a_a += 2 * einsum('iIiA->IA', v_ccce, optimize = einsum_type).copy()
    V1_a_a -= einsum('IiiA->IA', v_ccce, optimize = einsum_type).copy()
    V1_a_a += einsum('IxAy,xy->IA', v_caea, rdm_ca, optimize = einsum_type)
    V1_a_a -= 1/2 * einsum('IxyA,xy->IA', v_caae, rdm_ca, optimize = einsum_type)

    V1[::2,::2] = V1_a_a.copy()
    V1[1::2,1::2] = V1_a_a.copy()

    V0p[:,:,0] = V1.copy()

    V2 = np.zeros((ncore * 2, nextern * 2, ncas * 2, ncas * 2))
    V2_aa_aa =- 1/2 * einsum('IA,XY->IAXY', h_ce, rdm_ca, optimize = einsum_type)
    V2_aa_aa -= 1/2 * einsum('IxAY,Xx->IAXY', v_caea, rdm_ca, optimize = einsum_type)
    V2_aa_aa -= 1/2 * einsum('IxAy,XyYx->IAXY', v_caea, rdm_ccaa, optimize = einsum_type)
    V2_aa_aa += 1/2 * einsum('IxYA,Xx->IAXY', v_caae, rdm_ca, optimize = einsum_type)
    V2_aa_aa += 1/6 * einsum('IxyA,XyYx->IAXY', v_caae, rdm_ccaa, optimize = einsum_type)
    V2_aa_aa -= 1/6 * einsum('IxyA,XyxY->IAXY', v_caae, rdm_ccaa, optimize = einsum_type)
    V2_aa_aa -= einsum('iIiA,XY->IAXY', v_ccce, rdm_ca, optimize = einsum_type)
    V2_aa_aa += 1/2 * einsum('IiiA,XY->IAXY', v_ccce, rdm_ca, optimize = einsum_type)

    V2_ab_ba  = 1/2 * einsum('IxYA,Xx->IAXY', v_caae, rdm_ca, optimize = einsum_type)
    V2_ab_ba -= 1/6 * einsum('IxyA,XyYx->IAXY', v_caae, rdm_ccaa, optimize = einsum_type)
    V2_ab_ba -= 1/3 * einsum('IxyA,XyxY->IAXY', v_caae, rdm_ccaa, optimize = einsum_type)

    V2_aa_bb =- 1/2 * einsum('IA,XY->IAXY', h_ce, rdm_ca, optimize = einsum_type)
    V2_aa_bb -= 1/2 * einsum('IxAY,Xx->IAXY', v_caea, rdm_ca, optimize = einsum_type)
    V2_aa_bb -= 1/2 * einsum('IxAy,XyYx->IAXY', v_caea, rdm_ccaa, optimize = einsum_type)
    V2_aa_bb += 1/3 * einsum('IxyA,XyYx->IAXY', v_caae, rdm_ccaa, optimize = einsum_type)
    V2_aa_bb += 1/6 * einsum('IxyA,XyxY->IAXY', v_caae, rdm_ccaa, optimize = einsum_type)
    V2_aa_bb -= einsum('iIiA,XY->IAXY', v_ccce, rdm_ca, optimize = einsum_type)
    V2_aa_bb += 1/2 * einsum('IiiA,XY->IAXY', v_ccce, rdm_ca, optimize = einsum_type)

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
    e_0p += 4 * einsum('ia,jija', t1_ce, v_ccce, optimize = einsum_type)
    e_0p -= 2 * einsum('ia,ijja', t1_ce, v_ccce, optimize = einsum_type)
    e_0p -= einsum('ia,ixya,xy', h_ce, t1_caae, rdm_ca, optimize = einsum_type)
    e_0p += 2 * einsum('ia,ixay,xy', h_ce, t1_caea, rdm_ca, optimize = einsum_type)
    e_0p += 2 * einsum('ia,ixay,xy', t1_ce, v_caea, rdm_ca, optimize = einsum_type)
    e_0p -= einsum('ia,ixya,xy', t1_ce, v_caae, rdm_ca, optimize = einsum_type)
    e_0p += 2 * einsum('ixya,izya,xz', t1_caae, v_caae, rdm_ca, optimize = einsum_type)
    e_0p -= einsum('ixya,izay,xz', t1_caae, v_caea, rdm_ca, optimize = einsum_type)
    e_0p -= einsum('ixya,izaw,xwyz', t1_caae, v_caea, rdm_ccaa, optimize = einsum_type)
    e_0p -= einsum('ixya,izwa,xwzy', t1_caae, v_caae, rdm_ccaa, optimize = einsum_type)
    e_0p -= 2 * einsum('ixya,jija,xy', t1_caae, v_ccce, rdm_ca, optimize = einsum_type)
    e_0p += einsum('ixya,ijja,xy', t1_caae, v_ccce, rdm_ca, optimize = einsum_type)
    e_0p += 2 * einsum('ixay,izay,xz', t1_caea, v_caea, rdm_ca, optimize = einsum_type)
    e_0p += 2 * einsum('ixay,izaw,xwyz', t1_caea, v_caea, rdm_ccaa, optimize = einsum_type)
    e_0p -= einsum('ixay,izya,xz', t1_caea, v_caae, rdm_ca, optimize = einsum_type)
    e_0p -= einsum('ixay,izwa,xwyz', t1_caea, v_caae, rdm_ccaa, optimize = einsum_type)
    e_0p += 4 * einsum('ixay,jija,xy', t1_caea, v_ccce, rdm_ca, optimize = einsum_type)
    e_0p -= 2 * einsum('ixay,ijja,xy', t1_caea, v_ccce, rdm_ca, optimize = einsum_type)

    return e_0p, t1_ce, t1_caea, t1_caae

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
