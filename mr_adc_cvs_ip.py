import sys
import time
import numpy as np
import prism.mr_adc_overlap as mr_adc_overlap
import prism.mr_adc_rdms as mr_adc_rdms

def compute_excitation_manifolds(mr_adc):

    # MR-ADC(0) and MR-ADC(1)
    mr_adc.h0.n_c = mr_adc.ncvs

    ## Total dimension of h0
    mr_adc.h0.dim = mr_adc.h0.n_c

    mr_adc.h0.s_c = 0
    mr_adc.h0.f_c = mr_adc.h0.s_c + mr_adc.h0.n_c

    print("Dimension of h0 excitation manifold:                       %d" % mr_adc.h0.dim)

    # MR-ADC(2)
    mr_adc.h1.dim = 0
    mr_adc.h_orth.dim = mr_adc.h0.dim

    if mr_adc.method in ("mr-adc(2)", "mr-adc(2)-x"):
        mr_adc.h1.n_caa = mr_adc.ncvs * mr_adc.ncas * mr_adc.ncas
        mr_adc.h1.n_cce_aaa = mr_adc.ncvs * (mr_adc.ncvs - 1) * mr_adc.nextern // 2
        mr_adc.h1.n_cce_abb = mr_adc.ncvs * mr_adc.ncvs * mr_adc.nextern
        mr_adc.h1.n_cae = mr_adc.ncvs * mr_adc.ncas * mr_adc.nextern
        mr_adc.h1.n_cca_aaa = mr_adc.ncvs * (mr_adc.ncvs - 1) * mr_adc.ncas // 2
        mr_adc.h1.n_cca_abb = mr_adc.ncvs * mr_adc.ncvs * mr_adc.ncas

        mr_adc.h1.dim_caa = 3 * mr_adc.h1.n_caa
        mr_adc.h1.dim_cce = mr_adc.h1.n_cce_aaa + mr_adc.h1.n_cce_abb
        mr_adc.h1.dim_cae = 3 * mr_adc.h1.n_cae
        mr_adc.h1.dim_cca = mr_adc.h1.n_cca_aaa + mr_adc.h1.n_cca_abb

        if mr_adc.nval > 0:
            mr_adc.h1.n_cve = mr_adc.ncvs * mr_adc.nval * mr_adc.nextern
            mr_adc.h1.n_cva = mr_adc.ncvs * mr_adc.nval * mr_adc.ncas

            mr_adc.h1.dim_cve = 3 * mr_adc.h1.n_cve
            mr_adc.h1.dim_cva = 3 * mr_adc.h1.n_cva

            mr_adc.h1.dim = (mr_adc.h1.dim_caa + mr_adc.h1.dim_cce + mr_adc.h1.dim_cve +
                             mr_adc.h1.dim_cae + mr_adc.h1.dim_cca + mr_adc.h1.dim_cva)
        else:
            mr_adc.h1.dim = (mr_adc.h1.dim_caa + mr_adc.h1.dim_cce + mr_adc.h1.dim_cae + mr_adc.h1.dim_cca)

        if mr_adc.nval > 0:
            mr_adc.h1.s_caa = mr_adc.h0.f_c
            mr_adc.h1.f_caa = mr_adc.h1.s_caa + mr_adc.h1.dim_caa
            mr_adc.h1.s_cce = mr_adc.h1.f_caa
            mr_adc.h1.f_cce = mr_adc.h1.s_cce + mr_adc.h1.dim_cce
            mr_adc.h1.s_cve = mr_adc.h1.f_cce
            mr_adc.h1.f_cve = mr_adc.h1.s_cve + mr_adc.h1.dim_cve
            mr_adc.h1.s_cae = mr_adc.h1.f_cve
            mr_adc.h1.f_cae = mr_adc.h1.s_cae + mr_adc.h1.dim_cae
            mr_adc.h1.s_cca = mr_adc.h1.f_cae
            mr_adc.h1.f_cca = mr_adc.h1.s_cca + mr_adc.h1.dim_cca
            mr_adc.h1.s_cva = mr_adc.h1.f_cca
            mr_adc.h1.f_cva = mr_adc.h1.s_cva + mr_adc.h1.dim_cva

            mr_adc.h1.s_caa_aaa = mr_adc.h1.s_caa
            mr_adc.h1.f_caa_aaa = mr_adc.h1.s_caa_aaa + mr_adc.h1.n_caa
            mr_adc.h1.s_caa_abb = mr_adc.h1.f_caa_aaa
            mr_adc.h1.f_caa_abb = mr_adc.h1.s_caa_abb + mr_adc.h1.n_caa
            mr_adc.h1.s_caa_bab = mr_adc.h1.f_caa_abb
            mr_adc.h1.f_caa_bab = mr_adc.h1.s_caa_bab + mr_adc.h1.n_caa

            mr_adc.h1.s_cce_aaa = mr_adc.h1.s_cce
            mr_adc.h1.f_cce_aaa = mr_adc.h1.s_cce_aaa + mr_adc.h1.n_cce_aaa
            mr_adc.h1.s_cce_abb = mr_adc.h1.f_cce_aaa
            mr_adc.h1.f_cce_abb = mr_adc.h1.s_cce_abb + mr_adc.h1.n_cce_abb

            mr_adc.h1.s_cve_aaa = mr_adc.h1.s_cve
            mr_adc.h1.f_cve_aaa = mr_adc.h1.s_cve_aaa + mr_adc.h1.n_cve
            mr_adc.h1.s_cve_abb = mr_adc.h1.f_cve_aaa
            mr_adc.h1.f_cve_abb = mr_adc.h1.s_cve_abb + mr_adc.h1.n_cve
            mr_adc.h1.s_cve_bab = mr_adc.h1.f_cve_abb
            mr_adc.h1.f_cve_bab = mr_adc.h1.s_cve_bab + mr_adc.h1.n_cve

            mr_adc.h1.s_cae_aaa = mr_adc.h1.s_cae
            mr_adc.h1.f_cae_aaa = mr_adc.h1.s_cae_aaa + mr_adc.h1.n_cae
            mr_adc.h1.s_cae_abb = mr_adc.h1.f_cae_aaa
            mr_adc.h1.f_cae_abb = mr_adc.h1.s_cae_abb + mr_adc.h1.n_cae
            mr_adc.h1.s_cae_bab = mr_adc.h1.f_cae_abb
            mr_adc.h1.f_cae_bab = mr_adc.h1.s_cae_bab + mr_adc.h1.n_cae

            mr_adc.h1.s_cca_aaa = mr_adc.h1.s_cca
            mr_adc.h1.f_cca_aaa = mr_adc.h1.s_cca_aaa + mr_adc.h1.n_cca_aaa
            mr_adc.h1.s_cca_abb = mr_adc.h1.f_cca_aaa
            mr_adc.h1.f_cca_abb = mr_adc.h1.s_cca_abb + mr_adc.h1.n_cca_abb

            mr_adc.h1.s_cva_aaa = mr_adc.h1.s_cva
            mr_adc.h1.f_cva_aaa = mr_adc.h1.s_cva_aaa + mr_adc.h1.n_cva
            mr_adc.h1.s_cva_abb = mr_adc.h1.f_cva_aaa
            mr_adc.h1.f_cva_abb = mr_adc.h1.s_cva_abb + mr_adc.h1.n_cva
            mr_adc.h1.s_cva_bab = mr_adc.h1.f_cva_abb
            mr_adc.h1.f_cva_bab = mr_adc.h1.s_cva_bab + mr_adc.h1.n_cva

        else:
            mr_adc.h1.s_caa = mr_adc.h0.f_c
            mr_adc.h1.f_caa = mr_adc.h1.s_caa + mr_adc.h1.dim_caa
            mr_adc.h1.s_cce = mr_adc.h1.f_caa
            mr_adc.h1.f_cce = mr_adc.h1.s_cce + mr_adc.h1.dim_cce
            mr_adc.h1.s_cae = mr_adc.h1.f_cce
            mr_adc.h1.f_cae = mr_adc.h1.s_cae + mr_adc.h1.dim_cae
            mr_adc.h1.s_cca = mr_adc.h1.f_cae
            mr_adc.h1.f_cca = mr_adc.h1.s_cca + mr_adc.h1.dim_cca

            mr_adc.h1.s_caa_aaa = mr_adc.h1.s_caa
            mr_adc.h1.f_caa_aaa = mr_adc.h1.s_caa_aaa + mr_adc.h1.n_caa
            mr_adc.h1.s_caa_abb = mr_adc.h1.f_caa_aaa
            mr_adc.h1.f_caa_abb = mr_adc.h1.s_caa_abb + mr_adc.h1.n_caa
            mr_adc.h1.s_caa_bab = mr_adc.h1.f_caa_abb
            mr_adc.h1.f_caa_bab = mr_adc.h1.s_caa_bab + mr_adc.h1.n_caa

            mr_adc.h1.s_cce_aaa = mr_adc.h1.s_cce
            mr_adc.h1.f_cce_aaa = mr_adc.h1.s_cce_aaa + mr_adc.h1.n_cce_aaa
            mr_adc.h1.s_cce_abb = mr_adc.h1.f_cce_aaa
            mr_adc.h1.f_cce_abb = mr_adc.h1.s_cce_abb + mr_adc.h1.n_cce_abb

            mr_adc.h1.s_cae_aaa = mr_adc.h1.s_cae
            mr_adc.h1.f_cae_aaa = mr_adc.h1.s_cae_aaa + mr_adc.h1.n_cae
            mr_adc.h1.s_cae_abb = mr_adc.h1.f_cae_aaa
            mr_adc.h1.f_cae_abb = mr_adc.h1.s_cae_abb + mr_adc.h1.n_cae
            mr_adc.h1.s_cae_bab = mr_adc.h1.f_cae_abb
            mr_adc.h1.f_cae_bab = mr_adc.h1.s_cae_bab + mr_adc.h1.n_cae

            mr_adc.h1.s_cca_aaa = mr_adc.h1.s_cca
            mr_adc.h1.f_cca_aaa = mr_adc.h1.s_cca_aaa + mr_adc.h1.n_cca_aaa
            mr_adc.h1.s_cca_abb = mr_adc.h1.f_cca_aaa
            mr_adc.h1.f_cca_abb = mr_adc.h1.s_cca_abb + mr_adc.h1.n_cca_abb

        print("Dimension of h1 excitation manifold:                       %d" % mr_adc.h1.dim)

        # Overlap for c - caa
        mr_adc.S12.c_caa = mr_adc_overlap.compute_S12_0p_projector(mr_adc)
        mr_adc.S12.cae = mr_adc_overlap.compute_S12_m1(mr_adc)
        mr_adc.S12.cca = mr_adc_overlap.compute_S12_p1(mr_adc)

        # Determine dimensions of orthogonalized excitation spaces
        mr_adc.h_orth.n_c_caa = mr_adc.ncvs * mr_adc.S12.c_caa.shape[1]
        mr_adc.h_orth.n_cce_aaa = mr_adc.h1.n_cce_aaa
        mr_adc.h_orth.n_cce_abb = mr_adc.h1.n_cce_abb
        mr_adc.h_orth.n_cae = mr_adc.ncvs * mr_adc.S12.cae.shape[1] * mr_adc.nextern
        mr_adc.h_orth.n_cca_aaa = mr_adc.ncvs * (mr_adc.ncvs - 1) * mr_adc.S12.cca.shape[1] // 2
        mr_adc.h_orth.n_cca_abb = mr_adc.ncvs * mr_adc.ncvs * mr_adc.S12.cca.shape[1]

        mr_adc.h_orth.dim_c_caa = mr_adc.h_orth.n_c_caa
        mr_adc.h_orth.dim_cce = mr_adc.h1.dim_cce
        mr_adc.h_orth.dim_cae = 3 * mr_adc.h_orth.n_cae
        mr_adc.h_orth.dim_cca = mr_adc.h_orth.n_cca_aaa + mr_adc.h_orth.n_cca_abb

        if mr_adc.nval > 0:
            mr_adc.h_orth.n_cve = mr_adc.h1.n_cve
            mr_adc.h_orth.n_cva = mr_adc.ncvs * mr_adc.nval * mr_adc.S12.cca.shape[1]

            mr_adc.h_orth.dim_cve = mr_adc.h1.dim_cve
            mr_adc.h_orth.dim_cva = 3 * mr_adc.h_orth.n_cva

            mr_adc.h_orth.dim = (mr_adc.h_orth.dim_c_caa + mr_adc.h_orth.dim_cce + mr_adc.h_orth.dim_cve +
                                 mr_adc.h_orth.dim_cae + mr_adc.h_orth.dim_cca + mr_adc.h_orth.dim_cva)
        else:
            mr_adc.h_orth.dim = mr_adc.h_orth.dim_c_caa + mr_adc.h_orth.dim_cce + mr_adc.h_orth.dim_cae + mr_adc.h_orth.dim_cca

        if mr_adc.nval > 0:
            mr_adc.h_orth.s_c_caa = 0
            mr_adc.h_orth.f_c_caa = mr_adc.h_orth.s_c_caa + mr_adc.h_orth.dim_c_caa
            mr_adc.h_orth.s_cce = mr_adc.h_orth.f_c_caa
            mr_adc.h_orth.f_cce = mr_adc.h_orth.s_cce + mr_adc.h_orth.dim_cce
            mr_adc.h_orth.s_cve = mr_adc.h_orth.f_cce
            mr_adc.h_orth.f_cve = mr_adc.h_orth.s_cve + mr_adc.h_orth.dim_cve
            mr_adc.h_orth.s_cae = mr_adc.h_orth.f_cve
            mr_adc.h_orth.f_cae = mr_adc.h_orth.s_cae + mr_adc.h_orth.dim_cae
            mr_adc.h_orth.s_cca = mr_adc.h_orth.f_cae
            mr_adc.h_orth.f_cca = mr_adc.h_orth.s_cca + mr_adc.h_orth.dim_cca
            mr_adc.h_orth.s_cva = mr_adc.h_orth.f_cca
            mr_adc.h_orth.f_cva = mr_adc.h_orth.s_cva + mr_adc.h_orth.dim_cva

            mr_adc.h_orth.s_cce_aaa = mr_adc.h_orth.s_cce
            mr_adc.h_orth.f_cce_aaa = mr_adc.h_orth.s_cce_aaa + mr_adc.h_orth.n_cce_aaa
            mr_adc.h_orth.s_cce_abb = mr_adc.h_orth.f_cce_aaa
            mr_adc.h_orth.f_cce_abb = mr_adc.h_orth.s_cce_abb + mr_adc.h_orth.n_cce_abb

            mr_adc.h_orth.s_cve_aaa = mr_adc.h_orth.s_cve
            mr_adc.h_orth.f_cve_aaa = mr_adc.h_orth.s_cve_aaa + mr_adc.h_orth.n_cve
            mr_adc.h_orth.s_cve_abb = mr_adc.h_orth.f_cve_aaa
            mr_adc.h_orth.f_cve_abb = mr_adc.h_orth.s_cve_abb + mr_adc.h_orth.n_cve
            mr_adc.h_orth.s_cve_bab = mr_adc.h_orth.f_cve_abb
            mr_adc.h_orth.f_cve_bab = mr_adc.h_orth.s_cve_bab + mr_adc.h_orth.n_cve

            mr_adc.h_orth.s_cae_aaa = mr_adc.h_orth.s_cae
            mr_adc.h_orth.f_cae_aaa = mr_adc.h_orth.s_cae_aaa + mr_adc.h_orth.n_cae
            mr_adc.h_orth.s_cae_abb = mr_adc.h_orth.f_cae_aaa
            mr_adc.h_orth.f_cae_abb = mr_adc.h_orth.s_cae_abb + mr_adc.h_orth.n_cae
            mr_adc.h_orth.s_cae_bab = mr_adc.h_orth.f_cae_abb
            mr_adc.h_orth.f_cae_bab = mr_adc.h_orth.s_cae_bab + mr_adc.h_orth.n_cae

            mr_adc.h_orth.s_cca_aaa = mr_adc.h_orth.s_cca
            mr_adc.h_orth.f_cca_aaa = mr_adc.h_orth.s_cca_aaa + mr_adc.h_orth.n_cca_aaa
            mr_adc.h_orth.s_cca_abb = mr_adc.h_orth.f_cca_aaa
            mr_adc.h_orth.f_cca_abb = mr_adc.h_orth.s_cca_abb + mr_adc.h_orth.n_cca_abb

            mr_adc.h_orth.s_cva_aaa = mr_adc.h_orth.s_cva
            mr_adc.h_orth.f_cva_aaa = mr_adc.h_orth.s_cva_aaa + mr_adc.h_orth.n_cva
            mr_adc.h_orth.s_cva_abb = mr_adc.h_orth.f_cva_aaa
            mr_adc.h_orth.f_cva_abb = mr_adc.h_orth.s_cva_abb + mr_adc.h_orth.n_cva
            mr_adc.h_orth.s_cva_bab = mr_adc.h_orth.f_cva_abb
            mr_adc.h_orth.f_cva_bab = mr_adc.h_orth.s_cva_bab + mr_adc.h_orth.n_cva

        else:
            mr_adc.h_orth.s_c_caa = 0
            mr_adc.h_orth.f_c_caa = mr_adc.h_orth.s_c_caa + mr_adc.h_orth.dim_c_caa
            mr_adc.h_orth.s_cce = mr_adc.h_orth.f_c_caa
            mr_adc.h_orth.f_cce = mr_adc.h_orth.s_cce + mr_adc.h_orth.dim_cce
            mr_adc.h_orth.s_cae = mr_adc.h_orth.f_cce
            mr_adc.h_orth.f_cae = mr_adc.h_orth.s_cae + mr_adc.h_orth.dim_cae
            mr_adc.h_orth.s_cca = mr_adc.h_orth.f_cae
            mr_adc.h_orth.f_cca = mr_adc.h_orth.s_cca + mr_adc.h_orth.dim_cca

            mr_adc.h_orth.s_cce_aaa = mr_adc.h_orth.s_cce
            mr_adc.h_orth.f_cce_aaa = mr_adc.h_orth.s_cce_aaa + mr_adc.h_orth.n_cce_aaa
            mr_adc.h_orth.s_cce_abb = mr_adc.h_orth.f_cce_aaa
            mr_adc.h_orth.f_cce_abb = mr_adc.h_orth.s_cce_abb + mr_adc.h_orth.n_cce_abb

            mr_adc.h_orth.s_cae_aaa = mr_adc.h_orth.s_cae
            mr_adc.h_orth.f_cae_aaa = mr_adc.h_orth.s_cae_aaa + mr_adc.h_orth.n_cae
            mr_adc.h_orth.s_cae_abb = mr_adc.h_orth.f_cae_aaa
            mr_adc.h_orth.f_cae_abb = mr_adc.h_orth.s_cae_abb + mr_adc.h_orth.n_cae
            mr_adc.h_orth.s_cae_bab = mr_adc.h_orth.f_cae_abb
            mr_adc.h_orth.f_cae_bab = mr_adc.h_orth.s_cae_bab + mr_adc.h_orth.n_cae

            mr_adc.h_orth.s_cca_aaa = mr_adc.h_orth.s_cca
            mr_adc.h_orth.f_cca_aaa = mr_adc.h_orth.s_cca_aaa + mr_adc.h_orth.n_cca_aaa
            mr_adc.h_orth.s_cca_abb = mr_adc.h_orth.f_cca_aaa
            mr_adc.h_orth.f_cca_abb = mr_adc.h_orth.s_cca_abb + mr_adc.h_orth.n_cca_abb

    print("Total dimension of the excitation manifold:                %d" % (mr_adc.h0.dim + mr_adc.h1.dim))
    print("Dimension of the orthogonalized excitation manifold:       %d\n" % (mr_adc.h_orth.dim))
    sys.stdout.flush()

    if (mr_adc.h_orth.dim < mr_adc.nroots):
        mr_adc.nroots = mr_adc.h_orth.dim

    return mr_adc

def compute_M_00(mr_adc):

    start_time = time.time()

    print("Computing M(h0-h0) block...")
    sys.stdout.flush()

    # Einsum definition from kernel
    einsum = mr_adc.interface.einsum
    einsum_type = mr_adc.interface.einsum_type

    # Variables from kernel
    ncvs = mr_adc.ncvs

    s_c = mr_adc.h0.s_c
    f_c = mr_adc.h0.f_c
    dim = mr_adc.h0.dim

    e_cvs = mr_adc.mo_energy.x
    e_core = mr_adc.mo_energy.c
    e_extern = mr_adc.mo_energy.e

    # M00 matrix
    M = np.zeros((dim, dim))

    # Zeroth-order terms
    # C - C
    M[s_c:f_c, s_c:f_c]  = einsum('J,IJ->IJ', e_cvs, np.identity(ncvs), optimize = einsum_type)

    # Second-order terms
    if mr_adc.method in ("mr-adc(2)", "mr-adc(2)-x"):
        # Amplitudes
        t1_ce = mr_adc.t1.ce
        t1_ca = mr_adc.t1.ca
        t1_ae = mr_adc.t1.ae
        t1_caea = mr_adc.t1.caea
        t1_caae = mr_adc.t1.caae
        t1_caaa = mr_adc.t1.caaa
        t1_aaea = mr_adc.t1.aaea

        t1_xe = mr_adc.t1.xe
        t1_xa = mr_adc.t1.xa
        t1_xaea = mr_adc.t1.xaea
        t1_xaae = mr_adc.t1.xaae
        t1_xaaa = mr_adc.t1.xaaa
        t1_xcee = mr_adc.t1.xcee
        t1_xcea = mr_adc.t1.xcea
        t1_cxea = mr_adc.t1.cxea
        t1_xaee = mr_adc.t1.xaee
        t1_xcaa = mr_adc.t1.xcaa

        # One-electron integrals
        h_aa = mr_adc.h1eff.aa

        h_xa = mr_adc.h1eff.xa
        h_xe = mr_adc.h1eff.xe

        # Two-electrons integrals
        v_aaaa = mr_adc.v2e.aaaa

        v_xaaa = mr_adc.v2e.xaaa
        v_xcaa = mr_adc.v2e.xcaa
        v_xcee = mr_adc.v2e.xcee
        v_xaea = mr_adc.v2e.xaea
        v_xaae = mr_adc.v2e.xaae
        v_xcae = mr_adc.v2e.xcae
        v_cxae = mr_adc.v2e.cxae
        v_xaee = mr_adc.v2e.xaee
        v_xcxa = mr_adc.v2e.xcxa
        v_cxxa = mr_adc.v2e.cxxa
        v_xcxe = mr_adc.v2e.xcxe
        v_cxxe = mr_adc.v2e.cxxe
        v_xaxe = mr_adc.v2e.xaxe
        v_xaex = mr_adc.v2e.xaex

        # Reduced Density Matrices
        rdm_ca = mr_adc.rdm.ca
        rdm_ccaa = mr_adc.rdm.ccaa
        rdm_cccaaa = mr_adc.rdm.cccaaa

        # C - C
        M[s_c:f_c, s_c:f_c] += einsum('Ix,Jx->IJ', h_xa, t1_xa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += einsum('Ia,Ja->IJ', h_xe, t1_xe, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += einsum('Jx,Ix->IJ', h_xa, t1_xa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += einsum('Ja,Ia->IJ', h_xe, t1_xe, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 2 * einsum('Iixy,Jixy->IJ', t1_xcaa, v_xcaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= einsum('Iixy,Jiyx->IJ', t1_xcaa, v_xcaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 2 * einsum('Iiax,iJxa->IJ', t1_xcea, v_cxae, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= einsum('Iiax,Jixa->IJ', t1_xcea, v_xcae, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 2 * einsum('Iiab,Jiab->IJ', t1_xcee, v_xcee, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= einsum('Iiab,Jiba->IJ', t1_xcee, v_xcee, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 2 * einsum('Jixy,Iixy->IJ', t1_xcaa, v_xcaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= einsum('Jixy,Iiyx->IJ', t1_xcaa, v_xcaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 2 * einsum('Jiax,iIxa->IJ', t1_xcea, v_cxae, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= einsum('Jiax,Iixa->IJ', t1_xcea, v_xcae, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 2 * einsum('Jiab,Iiab->IJ', t1_xcee, v_xcee, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= einsum('Jiab,Iiba->IJ', t1_xcee, v_xcee, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 2 * einsum('ix,JiIx->IJ', t1_ca, v_xcxa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= einsum('ix,iJIx->IJ', t1_ca, v_cxxa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= einsum('iIax,iJxa->IJ', t1_cxea, v_cxae, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 2 * einsum('iIax,Jixa->IJ', t1_cxea, v_xcae, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= einsum('iJax,iIxa->IJ', t1_cxea, v_cxae, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 2 * einsum('iJax,Iixa->IJ', t1_cxea, v_xcae, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 2 * einsum('ix,IiJx->IJ', t1_ca, v_xcxa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= einsum('ix,iIJx->IJ', t1_ca, v_cxxa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 2 * einsum('ia,IiJa->IJ', t1_ce, v_xcxe, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= einsum('ia,iIJa->IJ', t1_ce, v_cxxe, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 2 * einsum('ia,JiIa->IJ', t1_ce, v_xcxe, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= einsum('ia,iJIa->IJ', t1_ce, v_cxxe, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/2 * einsum('I,Ix,Jx->IJ', e_cvs, t1_xa, t1_xa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= einsum('I,Iixy,Jixy->IJ', e_cvs, t1_xcaa, t1_xcaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 1/2 * einsum('I,Iixy,Jiyx->IJ', e_cvs, t1_xcaa, t1_xcaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= einsum('I,Iiax,Jiax->IJ', e_cvs, t1_xcea, t1_xcea, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 1/2 * einsum('I,Iiax,iJax->IJ', e_cvs, t1_xcea, t1_cxea, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= einsum('I,Iiab,Jiab->IJ', e_cvs, t1_xcee, t1_xcee, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 1/2 * einsum('I,Iiab,Jiba->IJ', e_cvs, t1_xcee, t1_xcee, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/2 * einsum('I,Ia,Ja->IJ', e_cvs, t1_xe, t1_xe, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 1/2 * einsum('I,Jiax,iIax->IJ', e_cvs, t1_xcea, t1_cxea, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= einsum('I,iIax,iJax->IJ', e_cvs, t1_cxea, t1_cxea, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/2 * einsum('J,Ix,Jx->IJ', e_cvs, t1_xa, t1_xa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= einsum('J,Iixy,Jixy->IJ', e_cvs, t1_xcaa, t1_xcaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 1/2 * einsum('J,Iixy,Jiyx->IJ', e_cvs, t1_xcaa, t1_xcaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= einsum('J,Iiax,Jiax->IJ', e_cvs, t1_xcea, t1_xcea, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 1/2 * einsum('J,Iiax,iJax->IJ', e_cvs, t1_xcea, t1_cxea, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= einsum('J,Iiab,Jiab->IJ', e_cvs, t1_xcee, t1_xcee, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 1/2 * einsum('J,Iiab,Jiba->IJ', e_cvs, t1_xcee, t1_xcee, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/2 * einsum('J,Ia,Ja->IJ', e_cvs, t1_xe, t1_xe, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 1/2 * einsum('J,Jiax,iIax->IJ', e_cvs, t1_xcea, t1_cxea, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= einsum('J,iIax,iJax->IJ', e_cvs, t1_cxea, t1_cxea, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= einsum('i,Iixy,Jixy->IJ', e_core, t1_xcaa, t1_xcaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 1/2 * einsum('i,Iixy,Jiyx->IJ', e_core, t1_xcaa, t1_xcaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= einsum('i,Iiax,Jiax->IJ', e_core, t1_xcea, t1_xcea, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 1/2 * einsum('i,Iiax,iJax->IJ', e_core, t1_xcea, t1_cxea, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= einsum('i,Iiab,Jiab->IJ', e_core, t1_xcee, t1_xcee, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 1/2 * einsum('i,Iiab,Jiba->IJ', e_core, t1_xcee, t1_xcee, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= einsum('i,Jixy,Iixy->IJ', e_core, t1_xcaa, t1_xcaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 1/2 * einsum('i,Jixy,Iiyx->IJ', e_core, t1_xcaa, t1_xcaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= einsum('i,Jiax,Iiax->IJ', e_core, t1_xcea, t1_xcea, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 1/2 * einsum('i,Jiax,iIax->IJ', e_core, t1_xcea, t1_cxea, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= einsum('i,Jiab,Iiab->IJ', e_core, t1_xcee, t1_xcee, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 1/2 * einsum('i,Jiab,Iiba->IJ', e_core, t1_xcee, t1_xcee, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 1/2 * einsum('i,iIax,Jiax->IJ', e_core, t1_cxea, t1_xcea, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= einsum('i,iIax,iJax->IJ', e_core, t1_cxea, t1_cxea, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 1/2 * einsum('i,iJax,Iiax->IJ', e_core, t1_cxea, t1_xcea, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= einsum('i,iJax,iIax->IJ', e_core, t1_cxea, t1_cxea, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += einsum('a,Ia,Ja->IJ', e_extern, t1_xe, t1_xe, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 2 * einsum('a,Iiax,Jiax->IJ', e_extern, t1_xcea, t1_xcea, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= einsum('a,Iiax,iJax->IJ', e_extern, t1_xcea, t1_cxea, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 2 * einsum('a,Iiab,Jiab->IJ', e_extern, t1_xcee, t1_xcee, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= einsum('a,Iiab,Jiba->IJ', e_extern, t1_xcee, t1_xcee, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= einsum('a,Iiba,Jiab->IJ', e_extern, t1_xcee, t1_xcee, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 2 * einsum('a,Iiba,Jiba->IJ', e_extern, t1_xcee, t1_xcee, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= einsum('a,iIax,Jiax->IJ', e_extern, t1_cxea, t1_xcea, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 2 * einsum('a,iIax,iJax->IJ', e_extern, t1_cxea, t1_cxea, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += einsum('Ix,Jyxz,zy->IJ', h_xa, t1_xaaa, rdm_ca, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/2 * einsum('Ix,Jyzx,zy->IJ', h_xa, t1_xaaa, rdm_ca, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += einsum('Ia,Jxay,yx->IJ', h_xe, t1_xaea, rdm_ca, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/2 * einsum('Ia,Jxya,yx->IJ', h_xe, t1_xaae, rdm_ca, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += einsum('Jx,Iyxz,yz->IJ', h_xa, t1_xaaa, rdm_ca, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/2 * einsum('Jx,Iyzx,yz->IJ', h_xa, t1_xaaa, rdm_ca, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += einsum('Ja,Ixay,xy->IJ', h_xe, t1_xaea, rdm_ca, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/2 * einsum('Ja,Ixya,xy->IJ', h_xe, t1_xaae, rdm_ca, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += einsum('xy,Ix,Jy->IJ', h_aa, t1_xa, t1_xa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 2 * einsum('xy,Iixz,Jiyz->IJ', h_aa, t1_xcaa, t1_xcaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= einsum('xy,Iixz,Jizy->IJ', h_aa, t1_xcaa, t1_xcaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= einsum('xy,Iizx,Jiyz->IJ', h_aa, t1_xcaa, t1_xcaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 2 * einsum('xy,Iizx,Jizy->IJ', h_aa, t1_xcaa, t1_xcaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 2 * einsum('xy,Iiax,Jiay->IJ', h_aa, t1_xcea, t1_xcea, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= einsum('xy,Iiax,iJay->IJ', h_aa, t1_xcea, t1_cxea, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= einsum('xy,Jiax,iIay->IJ', h_aa, t1_xcea, t1_cxea, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 2 * einsum('xy,iIax,iJay->IJ', h_aa, t1_cxea, t1_cxea, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += einsum('Ix,Jyxz,zy->IJ', t1_xa, v_xaaa, rdm_ca, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/2 * einsum('Ix,Jyzx,zy->IJ', t1_xa, v_xaaa, rdm_ca, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 1/2 * einsum('Ixyz,Jxwu,yzwu->IJ', t1_xaaa, v_xaaa, rdm_ccaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += einsum('Ixyz,Jwyz,xw->IJ', t1_xaaa, v_xaaa, rdm_ca, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += einsum('Ixyz,Jwyu,xuzw->IJ', t1_xaaa, v_xaaa, rdm_ccaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/2 * einsum('Ixyz,Jwzy,xw->IJ', t1_xaaa, v_xaaa, rdm_ca, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/2 * einsum('Ixyz,Jwzu,xuyw->IJ', t1_xaaa, v_xaaa, rdm_ccaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/2 * einsum('Ixyz,Jwuy,xuzw->IJ', t1_xaaa, v_xaaa, rdm_ccaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/2 * einsum('Ixyz,Jwuz,xuwy->IJ', t1_xaaa, v_xaaa, rdm_ccaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += einsum('Ixya,Jzya,xz->IJ', t1_xaae, v_xaae, rdm_ca, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/2 * einsum('Ixya,Jzay,xz->IJ', t1_xaae, v_xaea, rdm_ca, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/2 * einsum('Ixya,Jzaw,xwyz->IJ', t1_xaae, v_xaea, rdm_ccaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/2 * einsum('Ixya,Jzwa,xwzy->IJ', t1_xaae, v_xaae, rdm_ccaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += einsum('Ixay,Jzay,xz->IJ', t1_xaea, v_xaea, rdm_ca, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += einsum('Ixay,Jzaw,xwyz->IJ', t1_xaea, v_xaea, rdm_ccaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/2 * einsum('Ixay,Jzya,xz->IJ', t1_xaea, v_xaae, rdm_ca, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/2 * einsum('Ixay,Jzwa,xwyz->IJ', t1_xaea, v_xaae, rdm_ccaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += einsum('Ixab,Jyab,xy->IJ', t1_xaee, v_xaee, rdm_ca, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/2 * einsum('Ixab,Jyba,xy->IJ', t1_xaee, v_xaee, rdm_ca, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= einsum('Iixy,Jixz,zy->IJ', t1_xcaa, v_xcaa, rdm_ca, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 1/2 * einsum('Iixy,Jiyz,zx->IJ', t1_xcaa, v_xcaa, rdm_ca, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 1/2 * einsum('Iixy,Jizx,zy->IJ', t1_xcaa, v_xcaa, rdm_ca, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= einsum('Iixy,Jizy,zx->IJ', t1_xcaa, v_xcaa, rdm_ca, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 1/2 * einsum('Iixy,Jizw,xyzw->IJ', t1_xcaa, v_xcaa, rdm_ccaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= einsum('Iiax,iJya,yx->IJ', t1_xcea, v_cxae, rdm_ca, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 1/2 * einsum('Iiax,Jiya,yx->IJ', t1_xcea, v_xcae, rdm_ca, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += einsum('Ia,Jxay,yx->IJ', t1_xe, v_xaea, rdm_ca, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/2 * einsum('Ia,Jxya,yx->IJ', t1_xe, v_xaae, rdm_ca, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += einsum('Jx,Iyxz,yz->IJ', t1_xa, v_xaaa, rdm_ca, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/2 * einsum('Jx,Iyzx,yz->IJ', t1_xa, v_xaaa, rdm_ca, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 1/2 * einsum('Jxyz,Ixwu,yzwu->IJ', t1_xaaa, v_xaaa, rdm_ccaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += einsum('Jxyz,Iwyz,wx->IJ', t1_xaaa, v_xaaa, rdm_ca, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += einsum('Jxyz,Iwyu,xuzw->IJ', t1_xaaa, v_xaaa, rdm_ccaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/2 * einsum('Jxyz,Iwzy,wx->IJ', t1_xaaa, v_xaaa, rdm_ca, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/2 * einsum('Jxyz,Iwzu,xuyw->IJ', t1_xaaa, v_xaaa, rdm_ccaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/2 * einsum('Jxyz,Iwuy,xuzw->IJ', t1_xaaa, v_xaaa, rdm_ccaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/2 * einsum('Jxyz,Iwuz,xuwy->IJ', t1_xaaa, v_xaaa, rdm_ccaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += einsum('Jxya,Izya,zx->IJ', t1_xaae, v_xaae, rdm_ca, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/2 * einsum('Jxya,Izay,zx->IJ', t1_xaae, v_xaea, rdm_ca, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/2 * einsum('Jxya,Izaw,xwyz->IJ', t1_xaae, v_xaea, rdm_ccaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/2 * einsum('Jxya,Izwa,xwzy->IJ', t1_xaae, v_xaae, rdm_ccaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += einsum('Jxay,Izay,zx->IJ', t1_xaea, v_xaea, rdm_ca, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += einsum('Jxay,Izaw,xwyz->IJ', t1_xaea, v_xaea, rdm_ccaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/2 * einsum('Jxay,Izya,zx->IJ', t1_xaea, v_xaae, rdm_ca, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/2 * einsum('Jxay,Izwa,xwyz->IJ', t1_xaea, v_xaae, rdm_ccaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += einsum('Jxab,Iyab,yx->IJ', t1_xaee, v_xaee, rdm_ca, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/2 * einsum('Jxab,Iyba,yx->IJ', t1_xaee, v_xaee, rdm_ca, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= einsum('Jixy,Iixz,yz->IJ', t1_xcaa, v_xcaa, rdm_ca, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 1/2 * einsum('Jixy,Iiyz,xz->IJ', t1_xcaa, v_xcaa, rdm_ca, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 1/2 * einsum('Jixy,Iizx,yz->IJ', t1_xcaa, v_xcaa, rdm_ca, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= einsum('Jixy,Iizy,xz->IJ', t1_xcaa, v_xcaa, rdm_ca, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 1/2 * einsum('Jixy,Iizw,xyzw->IJ', t1_xcaa, v_xcaa, rdm_ccaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= einsum('Jiax,iIya,xy->IJ', t1_xcea, v_cxae, rdm_ca, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 1/2 * einsum('Jiax,Iiya,xy->IJ', t1_xcea, v_xcae, rdm_ca, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += einsum('Ja,Ixay,xy->IJ', t1_xe, v_xaea, rdm_ca, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/2 * einsum('Ja,Ixya,xy->IJ', t1_xe, v_xaae, rdm_ca, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 2 * einsum('ixyz,JiIy,xz->IJ', t1_caaa, v_xcxa, rdm_ca, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= einsum('ixyz,iJIy,xz->IJ', t1_caaa, v_cxxa, rdm_ca, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= einsum('ixyz,JiIz,xy->IJ', t1_caaa, v_xcxa, rdm_ca, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 1/2 * einsum('ixyz,iJIz,xy->IJ', t1_caaa, v_cxxa, rdm_ca, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 1/2 * einsum('iIax,iJya,yx->IJ', t1_cxea, v_cxae, rdm_ca, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= einsum('iIax,Jiya,yx->IJ', t1_cxea, v_xcae, rdm_ca, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 1/2 * einsum('iJax,iIya,xy->IJ', t1_cxea, v_cxae, rdm_ca, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= einsum('iJax,Iiya,xy->IJ', t1_cxea, v_xcae, rdm_ca, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= einsum('ix,IiJy,xy->IJ', t1_ca, v_xcxa, rdm_ca, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 1/2 * einsum('ix,iIJy,xy->IJ', t1_ca, v_cxxa, rdm_ca, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= einsum('ix,JiIy,yx->IJ', t1_ca, v_xcxa, rdm_ca, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 1/2 * einsum('ix,iJIy,yx->IJ', t1_ca, v_cxxa, rdm_ca, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 2 * einsum('ixyz,IiJy,zx->IJ', t1_caaa, v_xcxa, rdm_ca, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= einsum('ixyz,IiJz,yx->IJ', t1_caaa, v_xcxa, rdm_ca, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= einsum('ixyz,IiJw,xwzy->IJ', t1_caaa, v_xcxa, rdm_ccaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= einsum('ixyz,iIJy,zx->IJ', t1_caaa, v_cxxa, rdm_ca, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 1/2 * einsum('ixyz,iIJz,yx->IJ', t1_caaa, v_cxxa, rdm_ca, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 1/2 * einsum('ixyz,iIJw,xwzy->IJ', t1_caaa, v_cxxa, rdm_ccaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= einsum('ixyz,JiIw,xwzy->IJ', t1_caaa, v_xcxa, rdm_ccaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 1/2 * einsum('ixyz,iJIw,xwzy->IJ', t1_caaa, v_cxxa, rdm_ccaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= einsum('ixya,IiJa,yx->IJ', t1_caae, v_xcxe, rdm_ca, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 1/2 * einsum('ixya,iIJa,yx->IJ', t1_caae, v_cxxe, rdm_ca, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= einsum('ixya,JiIa,xy->IJ', t1_caae, v_xcxe, rdm_ca, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 1/2 * einsum('ixya,iJIa,xy->IJ', t1_caae, v_cxxe, rdm_ca, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 2 * einsum('ixay,IiJa,yx->IJ', t1_caea, v_xcxe, rdm_ca, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= einsum('ixay,iIJa,yx->IJ', t1_caea, v_cxxe, rdm_ca, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 2 * einsum('ixay,JiIa,xy->IJ', t1_caea, v_xcxe, rdm_ca, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= einsum('ixay,iJIa,xy->IJ', t1_caea, v_cxxe, rdm_ca, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += einsum('xa,JyIa,xy->IJ', t1_ae, v_xaxe, rdm_ca, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/2 * einsum('xa,JyaI,xy->IJ', t1_ae, v_xaex, rdm_ca, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += einsum('xa,IyJa,yx->IJ', t1_ae, v_xaxe, rdm_ca, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/2 * einsum('xa,IyaJ,yx->IJ', t1_ae, v_xaex, rdm_ca, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += einsum('xyaz,JwIa,zwyx->IJ', t1_aaea, v_xaxe, rdm_ccaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/2 * einsum('xyaz,JwaI,zwyx->IJ', t1_aaea, v_xaex, rdm_ccaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += einsum('xyaz,IwJa,zwyx->IJ', t1_aaea, v_xaxe, rdm_ccaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/2 * einsum('xyaz,IwaJ,zwyx->IJ', t1_aaea, v_xaex, rdm_ccaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 2 * einsum('xyzw,Iixy,Jizw->IJ', v_aaaa, t1_xcaa, t1_xcaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= einsum('xyzw,Iixy,Jiwz->IJ', v_aaaa, t1_xcaa, t1_xcaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/2 * einsum('I,Ix,Jyxz,zy->IJ', e_cvs, t1_xa, t1_xaaa, rdm_ca, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 1/4 * einsum('I,Ix,Jyzx,zy->IJ', e_cvs, t1_xa, t1_xaaa, rdm_ca, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/4 * einsum('I,Ixyz,Jxwu,yzwu->IJ', e_cvs, t1_xaaa, t1_xaaa, rdm_ccaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/2 * einsum('I,Ixyz,Jy,xz->IJ', e_cvs, t1_xaaa, t1_xa, rdm_ca, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 1/4 * einsum('I,Ixyz,Jz,xy->IJ', e_cvs, t1_xaaa, t1_xa, rdm_ca, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/2 * einsum('I,Ixyz,Jwyz,xw->IJ', e_cvs, t1_xaaa, t1_xaaa, rdm_ca, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/2 * einsum('I,Ixyz,Jwyu,xuzw->IJ', e_cvs, t1_xaaa, t1_xaaa, rdm_ccaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 1/4 * einsum('I,Ixyz,Jwzy,xw->IJ', e_cvs, t1_xaaa, t1_xaaa, rdm_ca, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 1/4 * einsum('I,Ixyz,Jwzu,xuyw->IJ', e_cvs, t1_xaaa, t1_xaaa, rdm_ccaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 1/4 * einsum('I,Ixyz,Jwuy,xuzw->IJ', e_cvs, t1_xaaa, t1_xaaa, rdm_ccaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 1/4 * einsum('I,Ixyz,Jwuz,xuwy->IJ', e_cvs, t1_xaaa, t1_xaaa, rdm_ccaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 1/4 * einsum('I,Ixya,Ja,xy->IJ', e_cvs, t1_xaae, t1_xe, rdm_ca, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/2 * einsum('I,Ixya,Jzya,xz->IJ', e_cvs, t1_xaae, t1_xaae, rdm_ca, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 1/4 * einsum('I,Ixya,Jzay,xz->IJ', e_cvs, t1_xaae, t1_xaea, rdm_ca, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 1/4 * einsum('I,Ixya,Jzaw,xwyz->IJ', e_cvs, t1_xaae, t1_xaea, rdm_ccaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 1/4 * einsum('I,Ixya,Jzwa,xwzy->IJ', e_cvs, t1_xaae, t1_xaae, rdm_ccaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/2 * einsum('I,Ixay,Ja,xy->IJ', e_cvs, t1_xaea, t1_xe, rdm_ca, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/2 * einsum('I,Ixay,Jzay,xz->IJ', e_cvs, t1_xaea, t1_xaea, rdm_ca, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/2 * einsum('I,Ixay,Jzaw,xwyz->IJ', e_cvs, t1_xaea, t1_xaea, rdm_ccaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 1/4 * einsum('I,Ixay,Jzya,xz->IJ', e_cvs, t1_xaea, t1_xaae, rdm_ca, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 1/4 * einsum('I,Ixay,Jzwa,xwyz->IJ', e_cvs, t1_xaea, t1_xaae, rdm_ccaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/2 * einsum('I,Ixab,Jyab,xy->IJ', e_cvs, t1_xaee, t1_xaee, rdm_ca, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 1/4 * einsum('I,Ixab,Jyba,xy->IJ', e_cvs, t1_xaee, t1_xaee, rdm_ca, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 1/2 * einsum('I,Iixy,Jixz,zy->IJ', e_cvs, t1_xcaa, t1_xcaa, rdm_ca, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/4 * einsum('I,Iixy,Jiyz,zx->IJ', e_cvs, t1_xcaa, t1_xcaa, rdm_ca, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/4 * einsum('I,Iixy,Jizx,zy->IJ', e_cvs, t1_xcaa, t1_xcaa, rdm_ca, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 1/2 * einsum('I,Iixy,Jizy,zx->IJ', e_cvs, t1_xcaa, t1_xcaa, rdm_ca, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/4 * einsum('I,Iixy,Jizw,xyzw->IJ', e_cvs, t1_xcaa, t1_xcaa, rdm_ccaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 1/2 * einsum('I,Iiax,Jiay,yx->IJ', e_cvs, t1_xcea, t1_xcea, rdm_ca, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/4 * einsum('I,Iiax,iJay,yx->IJ', e_cvs, t1_xcea, t1_cxea, rdm_ca, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/2 * einsum('I,Ia,Jxay,yx->IJ', e_cvs, t1_xe, t1_xaea, rdm_ca, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 1/4 * einsum('I,Ia,Jxya,yx->IJ', e_cvs, t1_xe, t1_xaae, rdm_ca, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/4 * einsum('I,Jiax,iIay,xy->IJ', e_cvs, t1_xcea, t1_cxea, rdm_ca, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 1/2 * einsum('I,iIax,iJay,yx->IJ', e_cvs, t1_cxea, t1_cxea, rdm_ca, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/2 * einsum('J,Ix,Jyxz,zy->IJ', e_cvs, t1_xa, t1_xaaa, rdm_ca, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 1/4 * einsum('J,Ix,Jyzx,zy->IJ', e_cvs, t1_xa, t1_xaaa, rdm_ca, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/4 * einsum('J,Ixyz,Jxwu,yzwu->IJ', e_cvs, t1_xaaa, t1_xaaa, rdm_ccaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/2 * einsum('J,Ixyz,Jy,xz->IJ', e_cvs, t1_xaaa, t1_xa, rdm_ca, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 1/4 * einsum('J,Ixyz,Jz,xy->IJ', e_cvs, t1_xaaa, t1_xa, rdm_ca, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/2 * einsum('J,Ixyz,Jwyz,xw->IJ', e_cvs, t1_xaaa, t1_xaaa, rdm_ca, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/2 * einsum('J,Ixyz,Jwyu,xuzw->IJ', e_cvs, t1_xaaa, t1_xaaa, rdm_ccaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 1/4 * einsum('J,Ixyz,Jwzy,xw->IJ', e_cvs, t1_xaaa, t1_xaaa, rdm_ca, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 1/4 * einsum('J,Ixyz,Jwzu,xuyw->IJ', e_cvs, t1_xaaa, t1_xaaa, rdm_ccaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 1/4 * einsum('J,Ixyz,Jwuy,xuzw->IJ', e_cvs, t1_xaaa, t1_xaaa, rdm_ccaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 1/4 * einsum('J,Ixyz,Jwuz,xuwy->IJ', e_cvs, t1_xaaa, t1_xaaa, rdm_ccaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 1/4 * einsum('J,Ixya,Ja,xy->IJ', e_cvs, t1_xaae, t1_xe, rdm_ca, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/2 * einsum('J,Ixya,Jzya,xz->IJ', e_cvs, t1_xaae, t1_xaae, rdm_ca, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 1/4 * einsum('J,Ixya,Jzay,xz->IJ', e_cvs, t1_xaae, t1_xaea, rdm_ca, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 1/4 * einsum('J,Ixya,Jzaw,xwyz->IJ', e_cvs, t1_xaae, t1_xaea, rdm_ccaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 1/4 * einsum('J,Ixya,Jzwa,xwzy->IJ', e_cvs, t1_xaae, t1_xaae, rdm_ccaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/2 * einsum('J,Ixay,Ja,xy->IJ', e_cvs, t1_xaea, t1_xe, rdm_ca, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/2 * einsum('J,Ixay,Jzay,xz->IJ', e_cvs, t1_xaea, t1_xaea, rdm_ca, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/2 * einsum('J,Ixay,Jzaw,xwyz->IJ', e_cvs, t1_xaea, t1_xaea, rdm_ccaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 1/4 * einsum('J,Ixay,Jzya,xz->IJ', e_cvs, t1_xaea, t1_xaae, rdm_ca, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 1/4 * einsum('J,Ixay,Jzwa,xwyz->IJ', e_cvs, t1_xaea, t1_xaae, rdm_ccaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/2 * einsum('J,Ixab,Jyab,xy->IJ', e_cvs, t1_xaee, t1_xaee, rdm_ca, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 1/4 * einsum('J,Ixab,Jyba,xy->IJ', e_cvs, t1_xaee, t1_xaee, rdm_ca, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 1/2 * einsum('J,Iixy,Jixz,zy->IJ', e_cvs, t1_xcaa, t1_xcaa, rdm_ca, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/4 * einsum('J,Iixy,Jiyz,zx->IJ', e_cvs, t1_xcaa, t1_xcaa, rdm_ca, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/4 * einsum('J,Iixy,Jizx,zy->IJ', e_cvs, t1_xcaa, t1_xcaa, rdm_ca, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 1/2 * einsum('J,Iixy,Jizy,zx->IJ', e_cvs, t1_xcaa, t1_xcaa, rdm_ca, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/4 * einsum('J,Iixy,Jizw,xyzw->IJ', e_cvs, t1_xcaa, t1_xcaa, rdm_ccaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 1/2 * einsum('J,Iiax,Jiay,yx->IJ', e_cvs, t1_xcea, t1_xcea, rdm_ca, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/4 * einsum('J,Iiax,iJay,yx->IJ', e_cvs, t1_xcea, t1_cxea, rdm_ca, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/2 * einsum('J,Ia,Jxay,yx->IJ', e_cvs, t1_xe, t1_xaea, rdm_ca, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 1/4 * einsum('J,Ia,Jxya,yx->IJ', e_cvs, t1_xe, t1_xaae, rdm_ca, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/4 * einsum('J,Jiax,iIay,xy->IJ', e_cvs, t1_xcea, t1_cxea, rdm_ca, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 1/2 * einsum('J,iIax,iJay,yx->IJ', e_cvs, t1_cxea, t1_cxea, rdm_ca, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 1/2 * einsum('i,Iixy,Jixz,zy->IJ', e_core, t1_xcaa, t1_xcaa, rdm_ca, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/4 * einsum('i,Iixy,Jiyz,zx->IJ', e_core, t1_xcaa, t1_xcaa, rdm_ca, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/4 * einsum('i,Iixy,Jizx,zy->IJ', e_core, t1_xcaa, t1_xcaa, rdm_ca, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 1/2 * einsum('i,Iixy,Jizy,zx->IJ', e_core, t1_xcaa, t1_xcaa, rdm_ca, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 1/2 * einsum('i,Iiax,Jiay,yx->IJ', e_core, t1_xcea, t1_xcea, rdm_ca, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/4 * einsum('i,Iiax,iJay,yx->IJ', e_core, t1_xcea, t1_cxea, rdm_ca, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 1/2 * einsum('i,Jixy,Iixz,yz->IJ', e_core, t1_xcaa, t1_xcaa, rdm_ca, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/4 * einsum('i,Jixy,Iiyz,xz->IJ', e_core, t1_xcaa, t1_xcaa, rdm_ca, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/4 * einsum('i,Jixy,Iizx,yz->IJ', e_core, t1_xcaa, t1_xcaa, rdm_ca, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 1/2 * einsum('i,Jixy,Iizy,xz->IJ', e_core, t1_xcaa, t1_xcaa, rdm_ca, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/2 * einsum('i,Jixy,Iizw,xyzw->IJ', e_core, t1_xcaa, t1_xcaa, rdm_ccaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 1/2 * einsum('i,Jiax,Iiay,xy->IJ', e_core, t1_xcea, t1_xcea, rdm_ca, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/4 * einsum('i,Jiax,iIay,xy->IJ', e_core, t1_xcea, t1_cxea, rdm_ca, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/4 * einsum('i,iIax,Jiay,yx->IJ', e_core, t1_cxea, t1_xcea, rdm_ca, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 1/2 * einsum('i,iIax,iJay,yx->IJ', e_core, t1_cxea, t1_cxea, rdm_ca, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/4 * einsum('i,iJax,Iiay,xy->IJ', e_core, t1_cxea, t1_xcea, rdm_ca, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 1/2 * einsum('i,iJax,iIay,xy->IJ', e_core, t1_cxea, t1_cxea, rdm_ca, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += einsum('a,Ia,Jxay,yx->IJ', e_extern, t1_xe, t1_xaea, rdm_ca, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/2 * einsum('a,Ia,Jxya,yx->IJ', e_extern, t1_xe, t1_xaae, rdm_ca, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += einsum('a,Ixay,Ja,xy->IJ', e_extern, t1_xaea, t1_xe, rdm_ca, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += einsum('a,Ixay,Jzay,xz->IJ', e_extern, t1_xaea, t1_xaea, rdm_ca, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += einsum('a,Ixay,Jzaw,xwyz->IJ', e_extern, t1_xaea, t1_xaea, rdm_ccaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/2 * einsum('a,Ixay,Jzya,xz->IJ', e_extern, t1_xaea, t1_xaae, rdm_ca, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/2 * einsum('a,Ixay,Jzwa,xwyz->IJ', e_extern, t1_xaea, t1_xaae, rdm_ccaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += einsum('a,Ixab,Jyab,xy->IJ', e_extern, t1_xaee, t1_xaee, rdm_ca, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/2 * einsum('a,Ixab,Jyba,xy->IJ', e_extern, t1_xaee, t1_xaee, rdm_ca, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/2 * einsum('a,Ixya,Ja,xy->IJ', e_extern, t1_xaae, t1_xe, rdm_ca, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/2 * einsum('a,Ixya,Jzay,xz->IJ', e_extern, t1_xaae, t1_xaea, rdm_ca, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/2 * einsum('a,Ixya,Jzaw,xwyz->IJ', e_extern, t1_xaae, t1_xaea, rdm_ccaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += einsum('a,Ixya,Jzya,xz->IJ', e_extern, t1_xaae, t1_xaae, rdm_ca, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/2 * einsum('a,Ixya,Jzwa,xwzy->IJ', e_extern, t1_xaae, t1_xaae, rdm_ccaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/2 * einsum('a,Ixba,Jyab,xy->IJ', e_extern, t1_xaee, t1_xaee, rdm_ca, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += einsum('a,Ixba,Jyba,xy->IJ', e_extern, t1_xaee, t1_xaee, rdm_ca, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= einsum('a,Iiax,Jiay,yx->IJ', e_extern, t1_xcea, t1_xcea, rdm_ca, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 1/2 * einsum('a,Iiax,iJay,yx->IJ', e_extern, t1_xcea, t1_cxea, rdm_ca, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 1/2 * einsum('a,iIax,Jiay,yx->IJ', e_extern, t1_cxea, t1_xcea, rdm_ca, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= einsum('a,iIax,iJay,yx->IJ', e_extern, t1_cxea, t1_cxea, rdm_ca, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += einsum('xy,Ix,Jzyw,wz->IJ', h_aa, t1_xa, t1_xaaa, rdm_ca, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/2 * einsum('xy,Ix,Jzwy,wz->IJ', h_aa, t1_xa, t1_xaaa, rdm_ca, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/2 * einsum('xy,Ixzw,Jyuv,zwuv->IJ', h_aa, t1_xaaa, t1_xaaa, rdm_ccaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/2 * einsum('xy,Ixzw,Jz,yw->IJ', h_aa, t1_xaaa, t1_xa, rdm_ca, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 1/4 * einsum('xy,Ixzw,Jw,yz->IJ', h_aa, t1_xaaa, t1_xa, rdm_ca, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/2 * einsum('xy,Ixzw,Juzw,yu->IJ', h_aa, t1_xaaa, t1_xaaa, rdm_ca, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/2 * einsum('xy,Ixzw,Juzv,yvwu->IJ', h_aa, t1_xaaa, t1_xaaa, rdm_ccaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 1/4 * einsum('xy,Ixzw,Juwz,yu->IJ', h_aa, t1_xaaa, t1_xaaa, rdm_ca, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 1/4 * einsum('xy,Ixzw,Juwv,yvzu->IJ', h_aa, t1_xaaa, t1_xaaa, rdm_ccaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 1/4 * einsum('xy,Ixzw,Juvz,yvwu->IJ', h_aa, t1_xaaa, t1_xaaa, rdm_ccaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 1/4 * einsum('xy,Ixzw,Juvw,yvuz->IJ', h_aa, t1_xaaa, t1_xaaa, rdm_ccaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 1/4 * einsum('xy,Ixza,Ja,yz->IJ', h_aa, t1_xaae, t1_xe, rdm_ca, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/2 * einsum('xy,Ixza,Jwza,yw->IJ', h_aa, t1_xaae, t1_xaae, rdm_ca, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 1/4 * einsum('xy,Ixza,Jwaz,yw->IJ', h_aa, t1_xaae, t1_xaea, rdm_ca, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 1/4 * einsum('xy,Ixza,Jwau,yuzw->IJ', h_aa, t1_xaae, t1_xaea, rdm_ccaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 1/4 * einsum('xy,Ixza,Jwua,yuwz->IJ', h_aa, t1_xaae, t1_xaae, rdm_ccaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/2 * einsum('xy,Ixaz,Ja,yz->IJ', h_aa, t1_xaea, t1_xe, rdm_ca, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/2 * einsum('xy,Ixaz,Jwaz,yw->IJ', h_aa, t1_xaea, t1_xaea, rdm_ca, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/2 * einsum('xy,Ixaz,Jwau,yuzw->IJ', h_aa, t1_xaea, t1_xaea, rdm_ccaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 1/4 * einsum('xy,Ixaz,Jwza,yw->IJ', h_aa, t1_xaea, t1_xaae, rdm_ca, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 1/4 * einsum('xy,Ixaz,Jwua,yuzw->IJ', h_aa, t1_xaea, t1_xaae, rdm_ccaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/2 * einsum('xy,Ixab,Jzab,yz->IJ', h_aa, t1_xaee, t1_xaee, rdm_ca, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 1/4 * einsum('xy,Ixab,Jzba,yz->IJ', h_aa, t1_xaee, t1_xaee, rdm_ca, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += einsum('xy,Izxw,Jy,zw->IJ', h_aa, t1_xaaa, t1_xa, rdm_ca, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 1/4 * einsum('xy,Izxw,Jzuv,ywuv->IJ', h_aa, t1_xaaa, t1_xaaa, rdm_ccaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/4 * einsum('xy,Izxw,Jw,zy->IJ', h_aa, t1_xaaa, t1_xa, rdm_ca, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += einsum('xy,Izxw,Juyw,zu->IJ', h_aa, t1_xaaa, t1_xaaa, rdm_ca, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += einsum('xy,Izxw,Juyv,zvwu->IJ', h_aa, t1_xaaa, t1_xaaa, rdm_ccaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/2 * einsum('xy,Izxw,Juwy,zu->IJ', h_aa, t1_xaaa, t1_xaaa, rdm_ca, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/4 * einsum('xy,Izxw,Juwv,yuzv->IJ', h_aa, t1_xaaa, t1_xaaa, rdm_ccaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/2 * einsum('xy,Izxw,Juvy,zvwu->IJ', h_aa, t1_xaaa, t1_xaaa, rdm_ccaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/4 * einsum('xy,Izxw,Juvw,yuvz->IJ', h_aa, t1_xaaa, t1_xaaa, rdm_ccaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/4 * einsum('xy,Izxa,Ja,zy->IJ', h_aa, t1_xaae, t1_xe, rdm_ca, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += einsum('xy,Izxa,Jwya,zw->IJ', h_aa, t1_xaae, t1_xaae, rdm_ca, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/2 * einsum('xy,Izxa,Jway,zw->IJ', h_aa, t1_xaae, t1_xaea, rdm_ca, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/4 * einsum('xy,Izxa,Jwau,ywzu->IJ', h_aa, t1_xaae, t1_xaea, rdm_ccaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/4 * einsum('xy,Izxa,Jwua,ywuz->IJ', h_aa, t1_xaae, t1_xaae, rdm_ccaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/2 * einsum('xy,Izwx,Jy,zw->IJ', h_aa, t1_xaaa, t1_xa, rdm_ca, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 1/4 * einsum('xy,Izwx,Jzuv,ywvu->IJ', h_aa, t1_xaaa, t1_xaaa, rdm_ccaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 1/2 * einsum('xy,Izwx,Jw,zy->IJ', h_aa, t1_xaaa, t1_xa, rdm_ca, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/2 * einsum('xy,Izwx,Juyw,zu->IJ', h_aa, t1_xaaa, t1_xaaa, rdm_ca, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/2 * einsum('xy,Izwx,Juyv,zvwu->IJ', h_aa, t1_xaaa, t1_xaaa, rdm_ccaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += einsum('xy,Izwx,Juwy,zu->IJ', h_aa, t1_xaaa, t1_xaaa, rdm_ca, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 1/2 * einsum('xy,Izwx,Juwv,yuzv->IJ', h_aa, t1_xaaa, t1_xaaa, rdm_ccaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/2 * einsum('xy,Izwx,Juvy,zvuw->IJ', h_aa, t1_xaaa, t1_xaaa, rdm_ccaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/4 * einsum('xy,Izwx,Juvw,yuzv->IJ', h_aa, t1_xaaa, t1_xaaa, rdm_ccaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 1/2 * einsum('xy,Izax,Ja,zy->IJ', h_aa, t1_xaea, t1_xe, rdm_ca, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/2 * einsum('xy,Izax,Jwya,zw->IJ', h_aa, t1_xaea, t1_xaae, rdm_ca, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += einsum('xy,Izax,Jway,zw->IJ', h_aa, t1_xaea, t1_xaea, rdm_ca, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 1/2 * einsum('xy,Izax,Jwau,ywzu->IJ', h_aa, t1_xaea, t1_xaea, rdm_ccaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/4 * einsum('xy,Izax,Jwua,ywzu->IJ', h_aa, t1_xaea, t1_xaae, rdm_ccaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= einsum('xy,Iixz,Jiyw,wz->IJ', h_aa, t1_xcaa, t1_xcaa, rdm_ca, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 1/4 * einsum('xy,Iixz,Jizw,wy->IJ', h_aa, t1_xcaa, t1_xcaa, rdm_ca, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 1/2 * einsum('xy,Iixz,Jiwy,wz->IJ', h_aa, t1_xcaa, t1_xcaa, rdm_ca, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/2 * einsum('xy,Iixz,Jiwz,wy->IJ', h_aa, t1_xcaa, t1_xcaa, rdm_ca, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 1/4 * einsum('xy,Iixz,Jiwu,yzwu->IJ', h_aa, t1_xcaa, t1_xcaa, rdm_ccaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 1/2 * einsum('xy,Iizx,Jiyw,wz->IJ', h_aa, t1_xcaa, t1_xcaa, rdm_ca, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/2 * einsum('xy,Iizx,Jizw,wy->IJ', h_aa, t1_xcaa, t1_xcaa, rdm_ca, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= einsum('xy,Iizx,Jiwy,wz->IJ', h_aa, t1_xcaa, t1_xcaa, rdm_ca, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 1/4 * einsum('xy,Iizx,Jiwz,wy->IJ', h_aa, t1_xcaa, t1_xcaa, rdm_ca, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 1/4 * einsum('xy,Iizx,Jiwu,yzuw->IJ', h_aa, t1_xcaa, t1_xcaa, rdm_ccaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/2 * einsum('xy,Iiax,Jiaz,zy->IJ', h_aa, t1_xcea, t1_xcea, rdm_ca, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 1/4 * einsum('xy,Iiax,iJaz,zy->IJ', h_aa, t1_xcea, t1_cxea, rdm_ca, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/2 * einsum('xy,Jxzw,Iz,wy->IJ', h_aa, t1_xaaa, t1_xa, rdm_ca, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 1/4 * einsum('xy,Jxzw,Iw,zy->IJ', h_aa, t1_xaaa, t1_xa, rdm_ca, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/2 * einsum('xy,Jxzw,Iuzw,uy->IJ', h_aa, t1_xaaa, t1_xaaa, rdm_ca, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/2 * einsum('xy,Jxzw,Iuzv,yvwu->IJ', h_aa, t1_xaaa, t1_xaaa, rdm_ccaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 1/4 * einsum('xy,Jxzw,Iuwz,uy->IJ', h_aa, t1_xaaa, t1_xaaa, rdm_ca, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 1/4 * einsum('xy,Jxzw,Iuwv,yvzu->IJ', h_aa, t1_xaaa, t1_xaaa, rdm_ccaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 1/4 * einsum('xy,Jxzw,Iuvz,yvwu->IJ', h_aa, t1_xaaa, t1_xaaa, rdm_ccaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 1/4 * einsum('xy,Jxzw,Iuvw,yvuz->IJ', h_aa, t1_xaaa, t1_xaaa, rdm_ccaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 1/4 * einsum('xy,Jxza,Ia,zy->IJ', h_aa, t1_xaae, t1_xe, rdm_ca, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/2 * einsum('xy,Jxza,Iwza,wy->IJ', h_aa, t1_xaae, t1_xaae, rdm_ca, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 1/4 * einsum('xy,Jxza,Iwaz,wy->IJ', h_aa, t1_xaae, t1_xaea, rdm_ca, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 1/4 * einsum('xy,Jxza,Iwau,yuzw->IJ', h_aa, t1_xaae, t1_xaea, rdm_ccaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 1/4 * einsum('xy,Jxza,Iwua,yuwz->IJ', h_aa, t1_xaae, t1_xaae, rdm_ccaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/2 * einsum('xy,Jxaz,Ia,zy->IJ', h_aa, t1_xaea, t1_xe, rdm_ca, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/2 * einsum('xy,Jxaz,Iwaz,wy->IJ', h_aa, t1_xaea, t1_xaea, rdm_ca, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/2 * einsum('xy,Jxaz,Iwau,yuzw->IJ', h_aa, t1_xaea, t1_xaea, rdm_ccaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 1/4 * einsum('xy,Jxaz,Iwza,wy->IJ', h_aa, t1_xaea, t1_xaae, rdm_ca, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 1/4 * einsum('xy,Jxaz,Iwua,yuzw->IJ', h_aa, t1_xaea, t1_xaae, rdm_ccaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/2 * einsum('xy,Jxab,Izab,zy->IJ', h_aa, t1_xaee, t1_xaee, rdm_ca, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 1/4 * einsum('xy,Jxab,Izba,zy->IJ', h_aa, t1_xaee, t1_xaee, rdm_ca, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 1/4 * einsum('xy,Jzxw,Izuv,ywuv->IJ', h_aa, t1_xaaa, t1_xaaa, rdm_ccaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/4 * einsum('xy,Jzxw,Iw,yz->IJ', h_aa, t1_xaaa, t1_xa, rdm_ca, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/4 * einsum('xy,Jzxw,Iuwv,yuzv->IJ', h_aa, t1_xaaa, t1_xaaa, rdm_ccaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/4 * einsum('xy,Jzxw,Iuvw,yuvz->IJ', h_aa, t1_xaaa, t1_xaaa, rdm_ccaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/4 * einsum('xy,Jzxa,Ia,yz->IJ', h_aa, t1_xaae, t1_xe, rdm_ca, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/4 * einsum('xy,Jzxa,Iwau,ywzu->IJ', h_aa, t1_xaae, t1_xaea, rdm_ccaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/4 * einsum('xy,Jzxa,Iwua,ywuz->IJ', h_aa, t1_xaae, t1_xaae, rdm_ccaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 1/4 * einsum('xy,Jzwx,Izuv,ywvu->IJ', h_aa, t1_xaaa, t1_xaaa, rdm_ccaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 1/2 * einsum('xy,Jzwx,Iw,yz->IJ', h_aa, t1_xaaa, t1_xa, rdm_ca, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 1/2 * einsum('xy,Jzwx,Iuwv,yuzv->IJ', h_aa, t1_xaaa, t1_xaaa, rdm_ccaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/4 * einsum('xy,Jzwx,Iuvw,yuzv->IJ', h_aa, t1_xaaa, t1_xaaa, rdm_ccaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 1/2 * einsum('xy,Jzax,Ia,yz->IJ', h_aa, t1_xaea, t1_xe, rdm_ca, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 1/2 * einsum('xy,Jzax,Iwau,ywzu->IJ', h_aa, t1_xaea, t1_xaea, rdm_ccaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/4 * einsum('xy,Jzax,Iwua,ywzu->IJ', h_aa, t1_xaea, t1_xaae, rdm_ccaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 1/4 * einsum('xy,Jixz,Iizw,yw->IJ', h_aa, t1_xcaa, t1_xcaa, rdm_ca, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/2 * einsum('xy,Jixz,Iiwz,yw->IJ', h_aa, t1_xcaa, t1_xcaa, rdm_ca, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 1/4 * einsum('xy,Jixz,Iiwu,yzwu->IJ', h_aa, t1_xcaa, t1_xcaa, rdm_ccaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/2 * einsum('xy,Jizx,Iizw,yw->IJ', h_aa, t1_xcaa, t1_xcaa, rdm_ca, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 1/4 * einsum('xy,Jizx,Iiwz,yw->IJ', h_aa, t1_xcaa, t1_xcaa, rdm_ca, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 1/4 * einsum('xy,Jizx,Iiwu,yzuw->IJ', h_aa, t1_xcaa, t1_xcaa, rdm_ccaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/2 * einsum('xy,Jiax,Iiaz,yz->IJ', h_aa, t1_xcea, t1_xcea, rdm_ca, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 1/4 * einsum('xy,Jiax,iIaz,yz->IJ', h_aa, t1_xcea, t1_cxea, rdm_ca, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 1/4 * einsum('xy,iIax,Jiaz,zy->IJ', h_aa, t1_cxea, t1_xcea, rdm_ca, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/2 * einsum('xy,iIax,iJaz,zy->IJ', h_aa, t1_cxea, t1_cxea, rdm_ca, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 1/4 * einsum('xy,iJax,Iiaz,yz->IJ', h_aa, t1_cxea, t1_xcea, rdm_ca, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/2 * einsum('xy,iJax,iIaz,yz->IJ', h_aa, t1_cxea, t1_cxea, rdm_ca, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 1/2 * einsum('xyzw,Ix,Jyuv,zwuv->IJ', v_aaaa, t1_xa, t1_xaaa, rdm_ccaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += einsum('xyzw,Ix,Jz,yw->IJ', v_aaaa, t1_xa, t1_xa, rdm_ca, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/2 * einsum('xyzw,Ix,Jw,yz->IJ', v_aaaa, t1_xa, t1_xa, rdm_ca, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += einsum('xyzw,Ix,Juzw,yu->IJ', v_aaaa, t1_xa, t1_xaaa, rdm_ca, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += einsum('xyzw,Ix,Juzv,yvwu->IJ', v_aaaa, t1_xa, t1_xaaa, rdm_ccaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/2 * einsum('xyzw,Ix,Juwz,yu->IJ', v_aaaa, t1_xa, t1_xaaa, rdm_ca, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/2 * einsum('xyzw,Ix,Juwv,yvzu->IJ', v_aaaa, t1_xa, t1_xaaa, rdm_ccaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/2 * einsum('xyzw,Ix,Juvz,yvwu->IJ', v_aaaa, t1_xa, t1_xaaa, rdm_ccaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/2 * einsum('xyzw,Ix,Juvw,yvuz->IJ', v_aaaa, t1_xa, t1_xaaa, rdm_ccaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 1/2 * einsum('xyzw,Ixuv,Jy,zwvu->IJ', v_aaaa, t1_xaaa, t1_xa, rdm_ccaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 5/12 * einsum('xyzw,Ixuv,Jzst,wstyuv->IJ', v_aaaa, t1_xaaa, t1_xaaa, rdm_cccaaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 1/12 * einsum('xyzw,Ixuv,Jzst,wstyvu->IJ', v_aaaa, t1_xaaa, t1_xaaa, rdm_cccaaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 1/12 * einsum('xyzw,Ixuv,Jzst,wstuyv->IJ', v_aaaa, t1_xaaa, t1_xaaa, rdm_cccaaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 1/12 * einsum('xyzw,Ixuv,Jzst,wstuvy->IJ', v_aaaa, t1_xaaa, t1_xaaa, rdm_cccaaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 1/12 * einsum('xyzw,Ixuv,Jzst,wstvyu->IJ', v_aaaa, t1_xaaa, t1_xaaa, rdm_cccaaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 1/12 * einsum('xyzw,Ixuv,Jzst,wstvuy->IJ', v_aaaa, t1_xaaa, t1_xaaa, rdm_cccaaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 1/12 * einsum('xyzw,Ixuv,Jwst,zstyuv->IJ', v_aaaa, t1_xaaa, t1_xaaa, rdm_cccaaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 1/12 * einsum('xyzw,Ixuv,Jwst,zstyvu->IJ', v_aaaa, t1_xaaa, t1_xaaa, rdm_cccaaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 1/12 * einsum('xyzw,Ixuv,Jwst,zstuyv->IJ', v_aaaa, t1_xaaa, t1_xaaa, rdm_cccaaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 1/12 * einsum('xyzw,Ixuv,Jwst,zstuvy->IJ', v_aaaa, t1_xaaa, t1_xaaa, rdm_cccaaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 1/12 * einsum('xyzw,Ixuv,Jwst,zstvyu->IJ', v_aaaa, t1_xaaa, t1_xaaa, rdm_cccaaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 5/12 * einsum('xyzw,Ixuv,Jwst,zstvuy->IJ', v_aaaa, t1_xaaa, t1_xaaa, rdm_cccaaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/2 * einsum('xyzw,Ixuv,Ju,yvwz->IJ', v_aaaa, t1_xaaa, t1_xa, rdm_ccaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 1/4 * einsum('xyzw,Ixuv,Jv,yuwz->IJ', v_aaaa, t1_xaaa, t1_xa, rdm_ccaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 1/4 * einsum('xyzw,Ixuv,Jsyu,zwvs->IJ', v_aaaa, t1_xaaa, t1_xaaa, rdm_ccaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 1/4 * einsum('xyzw,Ixuv,Jsyv,zwsu->IJ', v_aaaa, t1_xaaa, t1_xaaa, rdm_ccaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/12 * einsum('xyzw,Ixuv,Jsyt,zwtuvs->IJ', v_aaaa, t1_xaaa, t1_xaaa, rdm_cccaaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/12 * einsum('xyzw,Ixuv,Jsyt,zwtusv->IJ', v_aaaa, t1_xaaa, t1_xaaa, rdm_cccaaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 5/12 * einsum('xyzw,Ixuv,Jsyt,zwtvus->IJ', v_aaaa, t1_xaaa, t1_xaaa, rdm_cccaaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/12 * einsum('xyzw,Ixuv,Jsyt,zwtvsu->IJ', v_aaaa, t1_xaaa, t1_xaaa, rdm_cccaaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/12 * einsum('xyzw,Ixuv,Jsyt,zwtsuv->IJ', v_aaaa, t1_xaaa, t1_xaaa, rdm_cccaaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/12 * einsum('xyzw,Ixuv,Jsyt,zwtsvu->IJ', v_aaaa, t1_xaaa, t1_xaaa, rdm_cccaaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/2 * einsum('xyzw,Ixuv,Jsuy,zwvs->IJ', v_aaaa, t1_xaaa, t1_xaaa, rdm_ccaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/2 * einsum('xyzw,Ixuv,Jsuv,yswz->IJ', v_aaaa, t1_xaaa, t1_xaaa, rdm_ccaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 1/12 * einsum('xyzw,Ixuv,Jsut,zwtyvs->IJ', v_aaaa, t1_xaaa, t1_xaaa, rdm_cccaaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 1/12 * einsum('xyzw,Ixuv,Jsut,zwtysv->IJ', v_aaaa, t1_xaaa, t1_xaaa, rdm_cccaaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 5/12 * einsum('xyzw,Ixuv,Jsut,zwtvys->IJ', v_aaaa, t1_xaaa, t1_xaaa, rdm_cccaaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 1/12 * einsum('xyzw,Ixuv,Jsut,zwtvsy->IJ', v_aaaa, t1_xaaa, t1_xaaa, rdm_cccaaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 1/12 * einsum('xyzw,Ixuv,Jsut,zwtsyv->IJ', v_aaaa, t1_xaaa, t1_xaaa, rdm_cccaaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 1/12 * einsum('xyzw,Ixuv,Jsut,zwtsvy->IJ', v_aaaa, t1_xaaa, t1_xaaa, rdm_cccaaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 1/4 * einsum('xyzw,Ixuv,Jsvy,zwus->IJ', v_aaaa, t1_xaaa, t1_xaaa, rdm_ccaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 1/4 * einsum('xyzw,Ixuv,Jsvu,yswz->IJ', v_aaaa, t1_xaaa, t1_xaaa, rdm_ccaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/24 * einsum('xyzw,Ixuv,Jsvt,zwtyus->IJ', v_aaaa, t1_xaaa, t1_xaaa, rdm_cccaaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/24 * einsum('xyzw,Ixuv,Jsvt,zwtysu->IJ', v_aaaa, t1_xaaa, t1_xaaa, rdm_cccaaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 5/24 * einsum('xyzw,Ixuv,Jsvt,zwtuys->IJ', v_aaaa, t1_xaaa, t1_xaaa, rdm_cccaaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/24 * einsum('xyzw,Ixuv,Jsvt,zwtusy->IJ', v_aaaa, t1_xaaa, t1_xaaa, rdm_cccaaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/24 * einsum('xyzw,Ixuv,Jsvt,zwtsyu->IJ', v_aaaa, t1_xaaa, t1_xaaa, rdm_cccaaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/24 * einsum('xyzw,Ixuv,Jsvt,zwtsuy->IJ', v_aaaa, t1_xaaa, t1_xaaa, rdm_cccaaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/12 * einsum('xyzw,Ixuv,Jsty,zwtuvs->IJ', v_aaaa, t1_xaaa, t1_xaaa, rdm_cccaaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/12 * einsum('xyzw,Ixuv,Jsty,zwtusv->IJ', v_aaaa, t1_xaaa, t1_xaaa, rdm_cccaaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/12 * einsum('xyzw,Ixuv,Jsty,zwtvus->IJ', v_aaaa, t1_xaaa, t1_xaaa, rdm_cccaaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 5/12 * einsum('xyzw,Ixuv,Jsty,zwtvsu->IJ', v_aaaa, t1_xaaa, t1_xaaa, rdm_cccaaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/12 * einsum('xyzw,Ixuv,Jsty,zwtsuv->IJ', v_aaaa, t1_xaaa, t1_xaaa, rdm_cccaaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/12 * einsum('xyzw,Ixuv,Jsty,zwtsvu->IJ', v_aaaa, t1_xaaa, t1_xaaa, rdm_cccaaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/24 * einsum('xyzw,Ixuv,Jstu,zwtyvs->IJ', v_aaaa, t1_xaaa, t1_xaaa, rdm_cccaaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/24 * einsum('xyzw,Ixuv,Jstu,zwtysv->IJ', v_aaaa, t1_xaaa, t1_xaaa, rdm_cccaaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 5/24 * einsum('xyzw,Ixuv,Jstu,zwtvys->IJ', v_aaaa, t1_xaaa, t1_xaaa, rdm_cccaaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/24 * einsum('xyzw,Ixuv,Jstu,zwtvsy->IJ', v_aaaa, t1_xaaa, t1_xaaa, rdm_cccaaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/24 * einsum('xyzw,Ixuv,Jstu,zwtsyv->IJ', v_aaaa, t1_xaaa, t1_xaaa, rdm_cccaaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/24 * einsum('xyzw,Ixuv,Jstu,zwtsvy->IJ', v_aaaa, t1_xaaa, t1_xaaa, rdm_cccaaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/24 * einsum('xyzw,Ixuv,Jstv,zwtyus->IJ', v_aaaa, t1_xaaa, t1_xaaa, rdm_cccaaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/24 * einsum('xyzw,Ixuv,Jstv,zwtysu->IJ', v_aaaa, t1_xaaa, t1_xaaa, rdm_cccaaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/24 * einsum('xyzw,Ixuv,Jstv,zwtuys->IJ', v_aaaa, t1_xaaa, t1_xaaa, rdm_cccaaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/24 * einsum('xyzw,Ixuv,Jstv,zwtusy->IJ', v_aaaa, t1_xaaa, t1_xaaa, rdm_cccaaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 5/24 * einsum('xyzw,Ixuv,Jstv,zwtsyu->IJ', v_aaaa, t1_xaaa, t1_xaaa, rdm_cccaaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/24 * einsum('xyzw,Ixuv,Jstv,zwtsuy->IJ', v_aaaa, t1_xaaa, t1_xaaa, rdm_cccaaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 1/4 * einsum('xyzw,Ixua,Ja,yuwz->IJ', v_aaaa, t1_xaae, t1_xe, rdm_ccaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 1/4 * einsum('xyzw,Ixua,Jvya,zwvu->IJ', v_aaaa, t1_xaae, t1_xaae, rdm_ccaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/2 * einsum('xyzw,Ixua,Jvua,yvwz->IJ', v_aaaa, t1_xaae, t1_xaae, rdm_ccaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 1/4 * einsum('xyzw,Ixua,Jvay,zwuv->IJ', v_aaaa, t1_xaae, t1_xaea, rdm_ccaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 1/4 * einsum('xyzw,Ixua,Jvau,yvwz->IJ', v_aaaa, t1_xaae, t1_xaea, rdm_ccaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/24 * einsum('xyzw,Ixua,Jvas,zwsyuv->IJ', v_aaaa, t1_xaae, t1_xaea, rdm_cccaaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/24 * einsum('xyzw,Ixua,Jvas,zwsyvu->IJ', v_aaaa, t1_xaae, t1_xaea, rdm_cccaaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 5/24 * einsum('xyzw,Ixua,Jvas,zwsuyv->IJ', v_aaaa, t1_xaae, t1_xaea, rdm_cccaaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/24 * einsum('xyzw,Ixua,Jvas,zwsuvy->IJ', v_aaaa, t1_xaae, t1_xaea, rdm_cccaaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/24 * einsum('xyzw,Ixua,Jvas,zwsvyu->IJ', v_aaaa, t1_xaae, t1_xaea, rdm_cccaaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/24 * einsum('xyzw,Ixua,Jvas,zwsvuy->IJ', v_aaaa, t1_xaae, t1_xaea, rdm_cccaaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/24 * einsum('xyzw,Ixua,Jvsa,zwsyuv->IJ', v_aaaa, t1_xaae, t1_xaae, rdm_cccaaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/24 * einsum('xyzw,Ixua,Jvsa,zwsyvu->IJ', v_aaaa, t1_xaae, t1_xaae, rdm_cccaaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/24 * einsum('xyzw,Ixua,Jvsa,zwsuyv->IJ', v_aaaa, t1_xaae, t1_xaae, rdm_cccaaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/24 * einsum('xyzw,Ixua,Jvsa,zwsuvy->IJ', v_aaaa, t1_xaae, t1_xaae, rdm_cccaaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 5/24 * einsum('xyzw,Ixua,Jvsa,zwsvyu->IJ', v_aaaa, t1_xaae, t1_xaae, rdm_cccaaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/24 * einsum('xyzw,Ixua,Jvsa,zwsvuy->IJ', v_aaaa, t1_xaae, t1_xaae, rdm_cccaaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/2 * einsum('xyzw,Ixau,Ja,yuwz->IJ', v_aaaa, t1_xaea, t1_xe, rdm_ccaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 1/4 * einsum('xyzw,Ixau,Jvya,zwuv->IJ', v_aaaa, t1_xaea, t1_xaae, rdm_ccaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/2 * einsum('xyzw,Ixau,Jvay,zwuv->IJ', v_aaaa, t1_xaea, t1_xaea, rdm_ccaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/2 * einsum('xyzw,Ixau,Jvau,yvwz->IJ', v_aaaa, t1_xaea, t1_xaea, rdm_ccaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 1/12 * einsum('xyzw,Ixau,Jvas,zwsyuv->IJ', v_aaaa, t1_xaea, t1_xaea, rdm_cccaaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 1/12 * einsum('xyzw,Ixau,Jvas,zwsyvu->IJ', v_aaaa, t1_xaea, t1_xaea, rdm_cccaaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 5/12 * einsum('xyzw,Ixau,Jvas,zwsuyv->IJ', v_aaaa, t1_xaea, t1_xaea, rdm_cccaaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 1/12 * einsum('xyzw,Ixau,Jvas,zwsuvy->IJ', v_aaaa, t1_xaea, t1_xaea, rdm_cccaaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 1/12 * einsum('xyzw,Ixau,Jvas,zwsvyu->IJ', v_aaaa, t1_xaea, t1_xaea, rdm_cccaaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 1/12 * einsum('xyzw,Ixau,Jvas,zwsvuy->IJ', v_aaaa, t1_xaea, t1_xaea, rdm_cccaaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 1/4 * einsum('xyzw,Ixau,Jvua,yvwz->IJ', v_aaaa, t1_xaea, t1_xaae, rdm_ccaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/24 * einsum('xyzw,Ixau,Jvsa,zwsyuv->IJ', v_aaaa, t1_xaea, t1_xaae, rdm_cccaaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/24 * einsum('xyzw,Ixau,Jvsa,zwsyvu->IJ', v_aaaa, t1_xaea, t1_xaae, rdm_cccaaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 5/24 * einsum('xyzw,Ixau,Jvsa,zwsuyv->IJ', v_aaaa, t1_xaea, t1_xaae, rdm_cccaaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/24 * einsum('xyzw,Ixau,Jvsa,zwsuvy->IJ', v_aaaa, t1_xaea, t1_xaae, rdm_cccaaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/24 * einsum('xyzw,Ixau,Jvsa,zwsvyu->IJ', v_aaaa, t1_xaea, t1_xaae, rdm_cccaaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/24 * einsum('xyzw,Ixau,Jvsa,zwsvuy->IJ', v_aaaa, t1_xaea, t1_xaae, rdm_cccaaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/2 * einsum('xyzw,Ixab,Juab,yuwz->IJ', v_aaaa, t1_xaee, t1_xaee, rdm_ccaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 1/4 * einsum('xyzw,Ixab,Juba,yuwz->IJ', v_aaaa, t1_xaee, t1_xaee, rdm_ccaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += einsum('xyzw,Iuxy,Jz,uw->IJ', v_aaaa, t1_xaaa, t1_xa, rdm_ca, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/2 * einsum('xyzw,Iuxy,Jw,uz->IJ', v_aaaa, t1_xaaa, t1_xa, rdm_ca, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 1/4 * einsum('xyzw,Iuxy,Juvs,zwvs->IJ', v_aaaa, t1_xaaa, t1_xaaa, rdm_ccaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += einsum('xyzw,Iuxy,Jvzw,uv->IJ', v_aaaa, t1_xaaa, t1_xaaa, rdm_ca, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += einsum('xyzw,Iuxy,Jvzs,wvus->IJ', v_aaaa, t1_xaaa, t1_xaaa, rdm_ccaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/2 * einsum('xyzw,Iuxy,Jvwz,uv->IJ', v_aaaa, t1_xaaa, t1_xaaa, rdm_ca, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/2 * einsum('xyzw,Iuxy,Jvws,zvus->IJ', v_aaaa, t1_xaaa, t1_xaaa, rdm_ccaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/2 * einsum('xyzw,Iuxy,Jvsz,wvus->IJ', v_aaaa, t1_xaaa, t1_xaaa, rdm_ccaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/2 * einsum('xyzw,Iuxy,Jvsw,zvsu->IJ', v_aaaa, t1_xaaa, t1_xaaa, rdm_ccaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 1/4 * einsum('xyzw,Iuxv,Jyvs,zwus->IJ', v_aaaa, t1_xaaa, t1_xaaa, rdm_ccaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 1/4 * einsum('xyzw,Iuxv,Jysv,zwsu->IJ', v_aaaa, t1_xaaa, t1_xaaa, rdm_ccaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/12 * einsum('xyzw,Iuxv,Jyst,ustzwv->IJ', v_aaaa, t1_xaaa, t1_xaaa, rdm_cccaaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/12 * einsum('xyzw,Iuxv,Jyst,ustzvw->IJ', v_aaaa, t1_xaaa, t1_xaaa, rdm_cccaaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/12 * einsum('xyzw,Iuxv,Jyst,ustwzv->IJ', v_aaaa, t1_xaaa, t1_xaaa, rdm_cccaaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/12 * einsum('xyzw,Iuxv,Jyst,ustwvz->IJ', v_aaaa, t1_xaaa, t1_xaaa, rdm_cccaaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 5/12 * einsum('xyzw,Iuxv,Jyst,ustvzw->IJ', v_aaaa, t1_xaaa, t1_xaaa, rdm_cccaaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/12 * einsum('xyzw,Iuxv,Jyst,ustvwz->IJ', v_aaaa, t1_xaaa, t1_xaaa, rdm_cccaaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += einsum('xyzw,Iuxv,Jz,yuwv->IJ', v_aaaa, t1_xaaa, t1_xa, rdm_ccaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/2 * einsum('xyzw,Iuxv,Jw,yuzv->IJ', v_aaaa, t1_xaaa, t1_xa, rdm_ccaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/24 * einsum('xyzw,Iuxv,Just,ystzwv->IJ', v_aaaa, t1_xaaa, t1_xaaa, rdm_cccaaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/24 * einsum('xyzw,Iuxv,Just,ystzvw->IJ', v_aaaa, t1_xaaa, t1_xaaa, rdm_cccaaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 5/24 * einsum('xyzw,Iuxv,Just,ystwzv->IJ', v_aaaa, t1_xaaa, t1_xaaa, rdm_cccaaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/24 * einsum('xyzw,Iuxv,Just,ystwvz->IJ', v_aaaa, t1_xaaa, t1_xaaa, rdm_cccaaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/24 * einsum('xyzw,Iuxv,Just,ystvzw->IJ', v_aaaa, t1_xaaa, t1_xaaa, rdm_cccaaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/24 * einsum('xyzw,Iuxv,Just,ystvwz->IJ', v_aaaa, t1_xaaa, t1_xaaa, rdm_cccaaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/4 * einsum('xyzw,Iuxv,Jv,yuwz->IJ', v_aaaa, t1_xaaa, t1_xa, rdm_ccaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += einsum('xyzw,Iuxv,Jszw,yusv->IJ', v_aaaa, t1_xaaa, t1_xaaa, rdm_ccaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += einsum('xyzw,Iuxv,Jszv,yuws->IJ', v_aaaa, t1_xaaa, t1_xaaa, rdm_ccaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += einsum('xyzw,Iuxv,Jszt,yutwvs->IJ', v_aaaa, t1_xaaa, t1_xaaa, rdm_cccaaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/2 * einsum('xyzw,Iuxv,Jswz,yusv->IJ', v_aaaa, t1_xaaa, t1_xaaa, rdm_ccaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/2 * einsum('xyzw,Iuxv,Jswv,yuzs->IJ', v_aaaa, t1_xaaa, t1_xaaa, rdm_ccaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/2 * einsum('xyzw,Iuxv,Jswt,yutzvs->IJ', v_aaaa, t1_xaaa, t1_xaaa, rdm_cccaaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/2 * einsum('xyzw,Iuxv,Jsvz,yuws->IJ', v_aaaa, t1_xaaa, t1_xaaa, rdm_ccaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/2 * einsum('xyzw,Iuxv,Jsvw,yusz->IJ', v_aaaa, t1_xaaa, t1_xaaa, rdm_ccaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 1/24 * einsum('xyzw,Iuxv,Jsvt,yutzws->IJ', v_aaaa, t1_xaaa, t1_xaaa, rdm_cccaaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 1/24 * einsum('xyzw,Iuxv,Jsvt,yutzsw->IJ', v_aaaa, t1_xaaa, t1_xaaa, rdm_cccaaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 5/24 * einsum('xyzw,Iuxv,Jsvt,yutwzs->IJ', v_aaaa, t1_xaaa, t1_xaaa, rdm_cccaaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 1/24 * einsum('xyzw,Iuxv,Jsvt,yutwsz->IJ', v_aaaa, t1_xaaa, t1_xaaa, rdm_cccaaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 1/24 * einsum('xyzw,Iuxv,Jsvt,yutszw->IJ', v_aaaa, t1_xaaa, t1_xaaa, rdm_cccaaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 1/24 * einsum('xyzw,Iuxv,Jsvt,yutswz->IJ', v_aaaa, t1_xaaa, t1_xaaa, rdm_cccaaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/2 * einsum('xyzw,Iuxv,Jstz,yutwvs->IJ', v_aaaa, t1_xaaa, t1_xaaa, rdm_cccaaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 1/6 * einsum('xyzw,Iuxv,Jstw,yutzvs->IJ', v_aaaa, t1_xaaa, t1_xaaa, rdm_cccaaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 1/6 * einsum('xyzw,Iuxv,Jstw,yutzsv->IJ', v_aaaa, t1_xaaa, t1_xaaa, rdm_cccaaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 1/6 * einsum('xyzw,Iuxv,Jstw,yutvzs->IJ', v_aaaa, t1_xaaa, t1_xaaa, rdm_cccaaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 1/6 * einsum('xyzw,Iuxv,Jstw,yutvsz->IJ', v_aaaa, t1_xaaa, t1_xaaa, rdm_cccaaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 1/6 * einsum('xyzw,Iuxv,Jstw,yutszv->IJ', v_aaaa, t1_xaaa, t1_xaaa, rdm_cccaaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/3 * einsum('xyzw,Iuxv,Jstw,yutsvz->IJ', v_aaaa, t1_xaaa, t1_xaaa, rdm_cccaaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 1/24 * einsum('xyzw,Iuxv,Jstv,yutzws->IJ', v_aaaa, t1_xaaa, t1_xaaa, rdm_cccaaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 1/24 * einsum('xyzw,Iuxv,Jstv,yutzsw->IJ', v_aaaa, t1_xaaa, t1_xaaa, rdm_cccaaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 1/24 * einsum('xyzw,Iuxv,Jstv,yutwzs->IJ', v_aaaa, t1_xaaa, t1_xaaa, rdm_cccaaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 5/24 * einsum('xyzw,Iuxv,Jstv,yutwsz->IJ', v_aaaa, t1_xaaa, t1_xaaa, rdm_cccaaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 1/24 * einsum('xyzw,Iuxv,Jstv,yutszw->IJ', v_aaaa, t1_xaaa, t1_xaaa, rdm_cccaaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 1/24 * einsum('xyzw,Iuxv,Jstv,yutswz->IJ', v_aaaa, t1_xaaa, t1_xaaa, rdm_cccaaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 1/4 * einsum('xyzw,Iuxa,Jyav,zwuv->IJ', v_aaaa, t1_xaae, t1_xaea, rdm_ccaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 1/4 * einsum('xyzw,Iuxa,Jyva,zwvu->IJ', v_aaaa, t1_xaae, t1_xaae, rdm_ccaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/4 * einsum('xyzw,Iuxa,Ja,yuwz->IJ', v_aaaa, t1_xaae, t1_xe, rdm_ccaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += einsum('xyzw,Iuxa,Jvza,yuwv->IJ', v_aaaa, t1_xaae, t1_xaae, rdm_ccaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/2 * einsum('xyzw,Iuxa,Jvwa,yuzv->IJ', v_aaaa, t1_xaae, t1_xaae, rdm_ccaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/2 * einsum('xyzw,Iuxa,Jvaz,yuwv->IJ', v_aaaa, t1_xaae, t1_xaea, rdm_ccaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/2 * einsum('xyzw,Iuxa,Jvaw,yuvz->IJ', v_aaaa, t1_xaae, t1_xaea, rdm_ccaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 1/24 * einsum('xyzw,Iuxa,Jvas,yuszwv->IJ', v_aaaa, t1_xaae, t1_xaea, rdm_cccaaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 1/24 * einsum('xyzw,Iuxa,Jvas,yuszvw->IJ', v_aaaa, t1_xaae, t1_xaea, rdm_cccaaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 5/24 * einsum('xyzw,Iuxa,Jvas,yuswzv->IJ', v_aaaa, t1_xaae, t1_xaea, rdm_cccaaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 1/24 * einsum('xyzw,Iuxa,Jvas,yuswvz->IJ', v_aaaa, t1_xaae, t1_xaea, rdm_cccaaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 1/24 * einsum('xyzw,Iuxa,Jvas,yusvzw->IJ', v_aaaa, t1_xaae, t1_xaea, rdm_cccaaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 1/24 * einsum('xyzw,Iuxa,Jvas,yusvwz->IJ', v_aaaa, t1_xaae, t1_xaea, rdm_cccaaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 1/24 * einsum('xyzw,Iuxa,Jvsa,yuszwv->IJ', v_aaaa, t1_xaae, t1_xaae, rdm_cccaaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 1/24 * einsum('xyzw,Iuxa,Jvsa,yuszvw->IJ', v_aaaa, t1_xaae, t1_xaae, rdm_cccaaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 1/24 * einsum('xyzw,Iuxa,Jvsa,yuswzv->IJ', v_aaaa, t1_xaae, t1_xaae, rdm_cccaaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 5/24 * einsum('xyzw,Iuxa,Jvsa,yuswvz->IJ', v_aaaa, t1_xaae, t1_xaae, rdm_cccaaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 1/24 * einsum('xyzw,Iuxa,Jvsa,yusvzw->IJ', v_aaaa, t1_xaae, t1_xaae, rdm_cccaaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 1/24 * einsum('xyzw,Iuxa,Jvsa,yusvwz->IJ', v_aaaa, t1_xaae, t1_xaae, rdm_cccaaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/2 * einsum('xyzw,Iuvx,Jyvs,zwus->IJ', v_aaaa, t1_xaaa, t1_xaaa, rdm_ccaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 1/4 * einsum('xyzw,Iuvx,Jysv,zwus->IJ', v_aaaa, t1_xaaa, t1_xaaa, rdm_ccaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/12 * einsum('xyzw,Iuvx,Jyst,ustzwv->IJ', v_aaaa, t1_xaaa, t1_xaaa, rdm_cccaaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 5/12 * einsum('xyzw,Iuvx,Jyst,ustzvw->IJ', v_aaaa, t1_xaaa, t1_xaaa, rdm_cccaaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/12 * einsum('xyzw,Iuvx,Jyst,ustwzv->IJ', v_aaaa, t1_xaaa, t1_xaaa, rdm_cccaaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/12 * einsum('xyzw,Iuvx,Jyst,ustwvz->IJ', v_aaaa, t1_xaaa, t1_xaaa, rdm_cccaaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/12 * einsum('xyzw,Iuvx,Jyst,ustvzw->IJ', v_aaaa, t1_xaaa, t1_xaaa, rdm_cccaaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/12 * einsum('xyzw,Iuvx,Jyst,ustvwz->IJ', v_aaaa, t1_xaaa, t1_xaaa, rdm_cccaaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/2 * einsum('xyzw,Iuvx,Jz,yuwv->IJ', v_aaaa, t1_xaaa, t1_xa, rdm_ccaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/2 * einsum('xyzw,Iuvx,Jw,yuvz->IJ', v_aaaa, t1_xaaa, t1_xa, rdm_ccaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/24 * einsum('xyzw,Iuvx,Just,ystzwv->IJ', v_aaaa, t1_xaaa, t1_xaaa, rdm_cccaaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/24 * einsum('xyzw,Iuvx,Just,ystzvw->IJ', v_aaaa, t1_xaaa, t1_xaaa, rdm_cccaaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/24 * einsum('xyzw,Iuvx,Just,ystwzv->IJ', v_aaaa, t1_xaaa, t1_xaaa, rdm_cccaaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 5/24 * einsum('xyzw,Iuvx,Just,ystwvz->IJ', v_aaaa, t1_xaaa, t1_xaaa, rdm_cccaaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/24 * einsum('xyzw,Iuvx,Just,ystvzw->IJ', v_aaaa, t1_xaaa, t1_xaaa, rdm_cccaaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/24 * einsum('xyzw,Iuvx,Just,ystvwz->IJ', v_aaaa, t1_xaaa, t1_xaaa, rdm_cccaaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 1/2 * einsum('xyzw,Iuvx,Jv,yuwz->IJ', v_aaaa, t1_xaaa, t1_xa, rdm_ccaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/2 * einsum('xyzw,Iuvx,Jszw,yusv->IJ', v_aaaa, t1_xaaa, t1_xaaa, rdm_ccaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/2 * einsum('xyzw,Iuvx,Jszv,yuws->IJ', v_aaaa, t1_xaaa, t1_xaaa, rdm_ccaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/2 * einsum('xyzw,Iuvx,Jszt,yutwvs->IJ', v_aaaa, t1_xaaa, t1_xaaa, rdm_cccaaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/2 * einsum('xyzw,Iuvx,Jswz,yuvs->IJ', v_aaaa, t1_xaaa, t1_xaaa, rdm_ccaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/2 * einsum('xyzw,Iuvx,Jswv,yusz->IJ', v_aaaa, t1_xaaa, t1_xaaa, rdm_ccaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 1/6 * einsum('xyzw,Iuvx,Jswt,yutzvs->IJ', v_aaaa, t1_xaaa, t1_xaaa, rdm_cccaaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 1/6 * einsum('xyzw,Iuvx,Jswt,yutzsv->IJ', v_aaaa, t1_xaaa, t1_xaaa, rdm_cccaaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/3 * einsum('xyzw,Iuvx,Jswt,yutvzs->IJ', v_aaaa, t1_xaaa, t1_xaaa, rdm_cccaaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 1/6 * einsum('xyzw,Iuvx,Jswt,yutvsz->IJ', v_aaaa, t1_xaaa, t1_xaaa, rdm_cccaaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 1/6 * einsum('xyzw,Iuvx,Jswt,yutszv->IJ', v_aaaa, t1_xaaa, t1_xaaa, rdm_cccaaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 1/6 * einsum('xyzw,Iuvx,Jswt,yutsvz->IJ', v_aaaa, t1_xaaa, t1_xaaa, rdm_cccaaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += einsum('xyzw,Iuvx,Jsvz,yuws->IJ', v_aaaa, t1_xaaa, t1_xaaa, rdm_ccaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += einsum('xyzw,Iuvx,Jsvw,yusz->IJ', v_aaaa, t1_xaaa, t1_xaaa, rdm_ccaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/12 * einsum('xyzw,Iuvx,Jsvt,yutzws->IJ', v_aaaa, t1_xaaa, t1_xaaa, rdm_cccaaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/12 * einsum('xyzw,Iuvx,Jsvt,yutzsw->IJ', v_aaaa, t1_xaaa, t1_xaaa, rdm_cccaaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 5/12 * einsum('xyzw,Iuvx,Jsvt,yutwzs->IJ', v_aaaa, t1_xaaa, t1_xaaa, rdm_cccaaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/12 * einsum('xyzw,Iuvx,Jsvt,yutwsz->IJ', v_aaaa, t1_xaaa, t1_xaaa, rdm_cccaaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/12 * einsum('xyzw,Iuvx,Jsvt,yutszw->IJ', v_aaaa, t1_xaaa, t1_xaaa, rdm_cccaaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/12 * einsum('xyzw,Iuvx,Jsvt,yutswz->IJ', v_aaaa, t1_xaaa, t1_xaaa, rdm_cccaaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 1/6 * einsum('xyzw,Iuvx,Jstz,yutwvs->IJ', v_aaaa, t1_xaaa, t1_xaaa, rdm_cccaaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/3 * einsum('xyzw,Iuvx,Jstz,yutwsv->IJ', v_aaaa, t1_xaaa, t1_xaaa, rdm_cccaaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 1/6 * einsum('xyzw,Iuvx,Jstz,yutvws->IJ', v_aaaa, t1_xaaa, t1_xaaa, rdm_cccaaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 1/6 * einsum('xyzw,Iuvx,Jstz,yutvsw->IJ', v_aaaa, t1_xaaa, t1_xaaa, rdm_cccaaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 1/6 * einsum('xyzw,Iuvx,Jstz,yutswv->IJ', v_aaaa, t1_xaaa, t1_xaaa, rdm_cccaaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 1/6 * einsum('xyzw,Iuvx,Jstz,yutsvw->IJ', v_aaaa, t1_xaaa, t1_xaaa, rdm_cccaaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/2 * einsum('xyzw,Iuvx,Jstw,yutszv->IJ', v_aaaa, t1_xaaa, t1_xaaa, rdm_cccaaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 1/24 * einsum('xyzw,Iuvx,Jstv,yutzws->IJ', v_aaaa, t1_xaaa, t1_xaaa, rdm_cccaaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 1/24 * einsum('xyzw,Iuvx,Jstv,yutzsw->IJ', v_aaaa, t1_xaaa, t1_xaaa, rdm_cccaaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 5/24 * einsum('xyzw,Iuvx,Jstv,yutwzs->IJ', v_aaaa, t1_xaaa, t1_xaaa, rdm_cccaaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 1/24 * einsum('xyzw,Iuvx,Jstv,yutwsz->IJ', v_aaaa, t1_xaaa, t1_xaaa, rdm_cccaaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 1/24 * einsum('xyzw,Iuvx,Jstv,yutszw->IJ', v_aaaa, t1_xaaa, t1_xaaa, rdm_cccaaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 1/24 * einsum('xyzw,Iuvx,Jstv,yutswz->IJ', v_aaaa, t1_xaaa, t1_xaaa, rdm_cccaaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/2 * einsum('xyzw,Iuax,Jyav,zwuv->IJ', v_aaaa, t1_xaea, t1_xaea, rdm_ccaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 1/4 * einsum('xyzw,Iuax,Jyva,zwuv->IJ', v_aaaa, t1_xaea, t1_xaae, rdm_ccaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 1/2 * einsum('xyzw,Iuax,Ja,yuwz->IJ', v_aaaa, t1_xaea, t1_xe, rdm_ccaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/2 * einsum('xyzw,Iuax,Jvza,yuwv->IJ', v_aaaa, t1_xaea, t1_xaae, rdm_ccaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/2 * einsum('xyzw,Iuax,Jvwa,yuvz->IJ', v_aaaa, t1_xaea, t1_xaae, rdm_ccaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += einsum('xyzw,Iuax,Jvaz,yuwv->IJ', v_aaaa, t1_xaea, t1_xaea, rdm_ccaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += einsum('xyzw,Iuax,Jvaw,yuvz->IJ', v_aaaa, t1_xaea, t1_xaea, rdm_ccaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/12 * einsum('xyzw,Iuax,Jvas,yuszwv->IJ', v_aaaa, t1_xaea, t1_xaea, rdm_cccaaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/12 * einsum('xyzw,Iuax,Jvas,yuszvw->IJ', v_aaaa, t1_xaea, t1_xaea, rdm_cccaaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 5/12 * einsum('xyzw,Iuax,Jvas,yuswzv->IJ', v_aaaa, t1_xaea, t1_xaea, rdm_cccaaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/12 * einsum('xyzw,Iuax,Jvas,yuswvz->IJ', v_aaaa, t1_xaea, t1_xaea, rdm_cccaaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/12 * einsum('xyzw,Iuax,Jvas,yusvzw->IJ', v_aaaa, t1_xaea, t1_xaea, rdm_cccaaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/12 * einsum('xyzw,Iuax,Jvas,yusvwz->IJ', v_aaaa, t1_xaea, t1_xaea, rdm_cccaaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 1/24 * einsum('xyzw,Iuax,Jvsa,yuszwv->IJ', v_aaaa, t1_xaea, t1_xaae, rdm_cccaaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 1/24 * einsum('xyzw,Iuax,Jvsa,yuszvw->IJ', v_aaaa, t1_xaea, t1_xaae, rdm_cccaaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 5/24 * einsum('xyzw,Iuax,Jvsa,yuswzv->IJ', v_aaaa, t1_xaea, t1_xaae, rdm_cccaaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 1/24 * einsum('xyzw,Iuax,Jvsa,yuswvz->IJ', v_aaaa, t1_xaea, t1_xaae, rdm_cccaaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 1/24 * einsum('xyzw,Iuax,Jvsa,yusvzw->IJ', v_aaaa, t1_xaea, t1_xaae, rdm_cccaaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 1/24 * einsum('xyzw,Iuax,Jvsa,yusvwz->IJ', v_aaaa, t1_xaea, t1_xaae, rdm_cccaaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= einsum('xyzw,Iixy,Jizu,uw->IJ', v_aaaa, t1_xcaa, t1_xcaa, rdm_ca, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 1/2 * einsum('xyzw,Iixy,Jiwu,uz->IJ', v_aaaa, t1_xcaa, t1_xcaa, rdm_ca, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 1/2 * einsum('xyzw,Iixy,Jiuz,uw->IJ', v_aaaa, t1_xcaa, t1_xcaa, rdm_ca, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= einsum('xyzw,Iixy,Jiuw,uz->IJ', v_aaaa, t1_xcaa, t1_xcaa, rdm_ca, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 1/4 * einsum('xyzw,Iixy,Jiuv,zwuv->IJ', v_aaaa, t1_xcaa, t1_xcaa, rdm_ccaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= einsum('xyzw,Iixu,Jizw,yu->IJ', v_aaaa, t1_xcaa, t1_xcaa, rdm_ca, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 2 * einsum('xyzw,Iixu,Jizu,yw->IJ', v_aaaa, t1_xcaa, t1_xcaa, rdm_ca, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= einsum('xyzw,Iixu,Jizv,yvwu->IJ', v_aaaa, t1_xcaa, t1_xcaa, rdm_ccaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 1/2 * einsum('xyzw,Iixu,Jiwz,yu->IJ', v_aaaa, t1_xcaa, t1_xcaa, rdm_ca, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= einsum('xyzw,Iixu,Jiwu,yz->IJ', v_aaaa, t1_xcaa, t1_xcaa, rdm_ca, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 1/2 * einsum('xyzw,Iixu,Jiwv,yvzu->IJ', v_aaaa, t1_xcaa, t1_xcaa, rdm_ccaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= einsum('xyzw,Iixu,Jiuz,yw->IJ', v_aaaa, t1_xcaa, t1_xcaa, rdm_ca, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 1/2 * einsum('xyzw,Iixu,Jiuw,yz->IJ', v_aaaa, t1_xcaa, t1_xcaa, rdm_ca, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 1/4 * einsum('xyzw,Iixu,Jiuv,yvwz->IJ', v_aaaa, t1_xcaa, t1_xcaa, rdm_ccaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 1/2 * einsum('xyzw,Iixu,Jivz,yvwu->IJ', v_aaaa, t1_xcaa, t1_xcaa, rdm_ccaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 1/2 * einsum('xyzw,Iixu,Jivw,yvuz->IJ', v_aaaa, t1_xcaa, t1_xcaa, rdm_ccaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/2 * einsum('xyzw,Iixu,Jivu,yvwz->IJ', v_aaaa, t1_xcaa, t1_xcaa, rdm_ccaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/24 * einsum('xyzw,Iixu,Jivs,yvszwu->IJ', v_aaaa, t1_xcaa, t1_xcaa, rdm_cccaaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/24 * einsum('xyzw,Iixu,Jivs,yvszuw->IJ', v_aaaa, t1_xcaa, t1_xcaa, rdm_cccaaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 5/24 * einsum('xyzw,Iixu,Jivs,yvswzu->IJ', v_aaaa, t1_xcaa, t1_xcaa, rdm_cccaaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/24 * einsum('xyzw,Iixu,Jivs,yvswuz->IJ', v_aaaa, t1_xcaa, t1_xcaa, rdm_cccaaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/24 * einsum('xyzw,Iixu,Jivs,yvsuzw->IJ', v_aaaa, t1_xcaa, t1_xcaa, rdm_cccaaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/24 * einsum('xyzw,Iixu,Jivs,yvsuwz->IJ', v_aaaa, t1_xcaa, t1_xcaa, rdm_cccaaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 1/2 * einsum('xyzw,Iiux,Jizw,yu->IJ', v_aaaa, t1_xcaa, t1_xcaa, rdm_ca, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= einsum('xyzw,Iiux,Jizu,yw->IJ', v_aaaa, t1_xcaa, t1_xcaa, rdm_ca, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 1/2 * einsum('xyzw,Iiux,Jizv,yvwu->IJ', v_aaaa, t1_xcaa, t1_xcaa, rdm_ccaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= einsum('xyzw,Iiux,Jiwz,yu->IJ', v_aaaa, t1_xcaa, t1_xcaa, rdm_ca, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 1/2 * einsum('xyzw,Iiux,Jiwu,yz->IJ', v_aaaa, t1_xcaa, t1_xcaa, rdm_ca, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 1/2 * einsum('xyzw,Iiux,Jiwv,yvuz->IJ', v_aaaa, t1_xcaa, t1_xcaa, rdm_ccaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 2 * einsum('xyzw,Iiux,Jiuz,yw->IJ', v_aaaa, t1_xcaa, t1_xcaa, rdm_ca, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= einsum('xyzw,Iiux,Jiuw,yz->IJ', v_aaaa, t1_xcaa, t1_xcaa, rdm_ca, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/2 * einsum('xyzw,Iiux,Jiuv,yvwz->IJ', v_aaaa, t1_xcaa, t1_xcaa, rdm_ccaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= einsum('xyzw,Iiux,Jivz,yvwu->IJ', v_aaaa, t1_xcaa, t1_xcaa, rdm_ccaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 1/2 * einsum('xyzw,Iiux,Jivw,yvzu->IJ', v_aaaa, t1_xcaa, t1_xcaa, rdm_ccaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 1/4 * einsum('xyzw,Iiux,Jivu,yvwz->IJ', v_aaaa, t1_xcaa, t1_xcaa, rdm_ccaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/24 * einsum('xyzw,Iiux,Jivs,yvszwu->IJ', v_aaaa, t1_xcaa, t1_xcaa, rdm_cccaaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/24 * einsum('xyzw,Iiux,Jivs,yvszuw->IJ', v_aaaa, t1_xcaa, t1_xcaa, rdm_cccaaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/24 * einsum('xyzw,Iiux,Jivs,yvswzu->IJ', v_aaaa, t1_xcaa, t1_xcaa, rdm_cccaaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 5/24 * einsum('xyzw,Iiux,Jivs,yvswuz->IJ', v_aaaa, t1_xcaa, t1_xcaa, rdm_cccaaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/24 * einsum('xyzw,Iiux,Jivs,yvsuzw->IJ', v_aaaa, t1_xcaa, t1_xcaa, rdm_cccaaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/24 * einsum('xyzw,Iiux,Jivs,yvsuwz->IJ', v_aaaa, t1_xcaa, t1_xcaa, rdm_cccaaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 2 * einsum('xyzw,Iiax,Jiaz,yw->IJ', v_aaaa, t1_xcea, t1_xcea, rdm_ca, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= einsum('xyzw,Iiax,Jiaw,yz->IJ', v_aaaa, t1_xcea, t1_xcea, rdm_ca, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/2 * einsum('xyzw,Iiax,Jiau,yuwz->IJ', v_aaaa, t1_xcea, t1_xcea, rdm_ccaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= einsum('xyzw,Iiax,iJaz,yw->IJ', v_aaaa, t1_xcea, t1_cxea, rdm_ca, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 1/2 * einsum('xyzw,Iiax,iJaw,yz->IJ', v_aaaa, t1_xcea, t1_cxea, rdm_ca, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 1/4 * einsum('xyzw,Iiax,iJau,yuwz->IJ', v_aaaa, t1_xcea, t1_cxea, rdm_ccaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/2 * einsum('xyzw,Jxuv,Iu,yvwz->IJ', v_aaaa, t1_xaaa, t1_xa, rdm_ccaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 1/4 * einsum('xyzw,Jxuv,Iv,yuwz->IJ', v_aaaa, t1_xaaa, t1_xa, rdm_ccaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/2 * einsum('xyzw,Jxuv,Isuv,yswz->IJ', v_aaaa, t1_xaaa, t1_xaaa, rdm_ccaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 1/12 * einsum('xyzw,Jxuv,Isut,yvszwt->IJ', v_aaaa, t1_xaaa, t1_xaaa, rdm_cccaaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 1/12 * einsum('xyzw,Jxuv,Isut,yvsztw->IJ', v_aaaa, t1_xaaa, t1_xaaa, rdm_cccaaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 5/12 * einsum('xyzw,Jxuv,Isut,yvswzt->IJ', v_aaaa, t1_xaaa, t1_xaaa, rdm_cccaaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 1/12 * einsum('xyzw,Jxuv,Isut,yvswtz->IJ', v_aaaa, t1_xaaa, t1_xaaa, rdm_cccaaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 1/12 * einsum('xyzw,Jxuv,Isut,yvstzw->IJ', v_aaaa, t1_xaaa, t1_xaaa, rdm_cccaaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 1/12 * einsum('xyzw,Jxuv,Isut,yvstwz->IJ', v_aaaa, t1_xaaa, t1_xaaa, rdm_cccaaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 1/4 * einsum('xyzw,Jxuv,Isvu,yswz->IJ', v_aaaa, t1_xaaa, t1_xaaa, rdm_ccaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/24 * einsum('xyzw,Jxuv,Isvt,yuszwt->IJ', v_aaaa, t1_xaaa, t1_xaaa, rdm_cccaaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/24 * einsum('xyzw,Jxuv,Isvt,yusztw->IJ', v_aaaa, t1_xaaa, t1_xaaa, rdm_cccaaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 5/24 * einsum('xyzw,Jxuv,Isvt,yuswzt->IJ', v_aaaa, t1_xaaa, t1_xaaa, rdm_cccaaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/24 * einsum('xyzw,Jxuv,Isvt,yuswtz->IJ', v_aaaa, t1_xaaa, t1_xaaa, rdm_cccaaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/24 * einsum('xyzw,Jxuv,Isvt,yustzw->IJ', v_aaaa, t1_xaaa, t1_xaaa, rdm_cccaaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/24 * einsum('xyzw,Jxuv,Isvt,yustwz->IJ', v_aaaa, t1_xaaa, t1_xaaa, rdm_cccaaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/24 * einsum('xyzw,Jxuv,Istu,yvszwt->IJ', v_aaaa, t1_xaaa, t1_xaaa, rdm_cccaaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/24 * einsum('xyzw,Jxuv,Istu,yvsztw->IJ', v_aaaa, t1_xaaa, t1_xaaa, rdm_cccaaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 5/24 * einsum('xyzw,Jxuv,Istu,yvswzt->IJ', v_aaaa, t1_xaaa, t1_xaaa, rdm_cccaaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/24 * einsum('xyzw,Jxuv,Istu,yvswtz->IJ', v_aaaa, t1_xaaa, t1_xaaa, rdm_cccaaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/24 * einsum('xyzw,Jxuv,Istu,yvstzw->IJ', v_aaaa, t1_xaaa, t1_xaaa, rdm_cccaaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/24 * einsum('xyzw,Jxuv,Istu,yvstwz->IJ', v_aaaa, t1_xaaa, t1_xaaa, rdm_cccaaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/24 * einsum('xyzw,Jxuv,Istv,yuszwt->IJ', v_aaaa, t1_xaaa, t1_xaaa, rdm_cccaaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/24 * einsum('xyzw,Jxuv,Istv,yusztw->IJ', v_aaaa, t1_xaaa, t1_xaaa, rdm_cccaaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/24 * einsum('xyzw,Jxuv,Istv,yuswzt->IJ', v_aaaa, t1_xaaa, t1_xaaa, rdm_cccaaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 5/24 * einsum('xyzw,Jxuv,Istv,yuswtz->IJ', v_aaaa, t1_xaaa, t1_xaaa, rdm_cccaaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/24 * einsum('xyzw,Jxuv,Istv,yustzw->IJ', v_aaaa, t1_xaaa, t1_xaaa, rdm_cccaaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/24 * einsum('xyzw,Jxuv,Istv,yustwz->IJ', v_aaaa, t1_xaaa, t1_xaaa, rdm_cccaaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 1/4 * einsum('xyzw,Jxua,Ia,yuwz->IJ', v_aaaa, t1_xaae, t1_xe, rdm_ccaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/2 * einsum('xyzw,Jxua,Ivua,yvwz->IJ', v_aaaa, t1_xaae, t1_xaae, rdm_ccaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 1/4 * einsum('xyzw,Jxua,Ivau,yvwz->IJ', v_aaaa, t1_xaae, t1_xaea, rdm_ccaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/24 * einsum('xyzw,Jxua,Ivas,yuvzws->IJ', v_aaaa, t1_xaae, t1_xaea, rdm_cccaaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/24 * einsum('xyzw,Jxua,Ivas,yuvzsw->IJ', v_aaaa, t1_xaae, t1_xaea, rdm_cccaaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 5/24 * einsum('xyzw,Jxua,Ivas,yuvwzs->IJ', v_aaaa, t1_xaae, t1_xaea, rdm_cccaaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/24 * einsum('xyzw,Jxua,Ivas,yuvwsz->IJ', v_aaaa, t1_xaae, t1_xaea, rdm_cccaaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/24 * einsum('xyzw,Jxua,Ivas,yuvszw->IJ', v_aaaa, t1_xaae, t1_xaea, rdm_cccaaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/24 * einsum('xyzw,Jxua,Ivas,yuvswz->IJ', v_aaaa, t1_xaae, t1_xaea, rdm_cccaaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/24 * einsum('xyzw,Jxua,Ivsa,yuvzws->IJ', v_aaaa, t1_xaae, t1_xaae, rdm_cccaaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/24 * einsum('xyzw,Jxua,Ivsa,yuvzsw->IJ', v_aaaa, t1_xaae, t1_xaae, rdm_cccaaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/24 * einsum('xyzw,Jxua,Ivsa,yuvwzs->IJ', v_aaaa, t1_xaae, t1_xaae, rdm_cccaaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 5/24 * einsum('xyzw,Jxua,Ivsa,yuvwsz->IJ', v_aaaa, t1_xaae, t1_xaae, rdm_cccaaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/24 * einsum('xyzw,Jxua,Ivsa,yuvszw->IJ', v_aaaa, t1_xaae, t1_xaae, rdm_cccaaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/24 * einsum('xyzw,Jxua,Ivsa,yuvswz->IJ', v_aaaa, t1_xaae, t1_xaae, rdm_cccaaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/2 * einsum('xyzw,Jxau,Ia,yuwz->IJ', v_aaaa, t1_xaea, t1_xe, rdm_ccaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/2 * einsum('xyzw,Jxau,Ivau,yvwz->IJ', v_aaaa, t1_xaea, t1_xaea, rdm_ccaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 1/12 * einsum('xyzw,Jxau,Ivas,yuvzws->IJ', v_aaaa, t1_xaea, t1_xaea, rdm_cccaaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 1/12 * einsum('xyzw,Jxau,Ivas,yuvzsw->IJ', v_aaaa, t1_xaea, t1_xaea, rdm_cccaaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 5/12 * einsum('xyzw,Jxau,Ivas,yuvwzs->IJ', v_aaaa, t1_xaea, t1_xaea, rdm_cccaaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 1/12 * einsum('xyzw,Jxau,Ivas,yuvwsz->IJ', v_aaaa, t1_xaea, t1_xaea, rdm_cccaaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 1/12 * einsum('xyzw,Jxau,Ivas,yuvszw->IJ', v_aaaa, t1_xaea, t1_xaea, rdm_cccaaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 1/12 * einsum('xyzw,Jxau,Ivas,yuvswz->IJ', v_aaaa, t1_xaea, t1_xaea, rdm_cccaaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 1/4 * einsum('xyzw,Jxau,Ivua,yvwz->IJ', v_aaaa, t1_xaea, t1_xaae, rdm_ccaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/24 * einsum('xyzw,Jxau,Ivsa,yuvzws->IJ', v_aaaa, t1_xaea, t1_xaae, rdm_cccaaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/24 * einsum('xyzw,Jxau,Ivsa,yuvzsw->IJ', v_aaaa, t1_xaea, t1_xaae, rdm_cccaaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 5/24 * einsum('xyzw,Jxau,Ivsa,yuvwzs->IJ', v_aaaa, t1_xaea, t1_xaae, rdm_cccaaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/24 * einsum('xyzw,Jxau,Ivsa,yuvwsz->IJ', v_aaaa, t1_xaea, t1_xaae, rdm_cccaaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/24 * einsum('xyzw,Jxau,Ivsa,yuvszw->IJ', v_aaaa, t1_xaea, t1_xaae, rdm_cccaaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/24 * einsum('xyzw,Jxau,Ivsa,yuvswz->IJ', v_aaaa, t1_xaea, t1_xaae, rdm_cccaaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/2 * einsum('xyzw,Jxab,Iuab,yuwz->IJ', v_aaaa, t1_xaee, t1_xaee, rdm_ccaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 1/4 * einsum('xyzw,Jxab,Iuba,yuwz->IJ', v_aaaa, t1_xaee, t1_xaee, rdm_ccaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 1/4 * einsum('xyzw,Juxy,Iuvs,zwvs->IJ', v_aaaa, t1_xaaa, t1_xaaa, rdm_ccaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/24 * einsum('xyzw,Juxv,Iust,zwvyst->IJ', v_aaaa, t1_xaaa, t1_xaaa, rdm_cccaaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/24 * einsum('xyzw,Juxv,Iust,zwvyts->IJ', v_aaaa, t1_xaaa, t1_xaaa, rdm_cccaaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 5/24 * einsum('xyzw,Juxv,Iust,zwvsyt->IJ', v_aaaa, t1_xaaa, t1_xaaa, rdm_cccaaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/24 * einsum('xyzw,Juxv,Iust,zwvsty->IJ', v_aaaa, t1_xaaa, t1_xaaa, rdm_cccaaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/24 * einsum('xyzw,Juxv,Iust,zwvtys->IJ', v_aaaa, t1_xaaa, t1_xaaa, rdm_cccaaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/24 * einsum('xyzw,Juxv,Iust,zwvtsy->IJ', v_aaaa, t1_xaaa, t1_xaaa, rdm_cccaaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/4 * einsum('xyzw,Juxv,Iv,yuwz->IJ', v_aaaa, t1_xaaa, t1_xa, rdm_ccaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 1/24 * einsum('xyzw,Juxv,Isvt,zwsyut->IJ', v_aaaa, t1_xaaa, t1_xaaa, rdm_cccaaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 1/24 * einsum('xyzw,Juxv,Isvt,zwsytu->IJ', v_aaaa, t1_xaaa, t1_xaaa, rdm_cccaaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 5/24 * einsum('xyzw,Juxv,Isvt,zwsuyt->IJ', v_aaaa, t1_xaaa, t1_xaaa, rdm_cccaaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 1/24 * einsum('xyzw,Juxv,Isvt,zwsuty->IJ', v_aaaa, t1_xaaa, t1_xaaa, rdm_cccaaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 1/24 * einsum('xyzw,Juxv,Isvt,zwstyu->IJ', v_aaaa, t1_xaaa, t1_xaaa, rdm_cccaaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 1/24 * einsum('xyzw,Juxv,Isvt,zwstuy->IJ', v_aaaa, t1_xaaa, t1_xaaa, rdm_cccaaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 1/24 * einsum('xyzw,Juxv,Istv,zwsyut->IJ', v_aaaa, t1_xaaa, t1_xaaa, rdm_cccaaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 1/24 * einsum('xyzw,Juxv,Istv,zwsytu->IJ', v_aaaa, t1_xaaa, t1_xaaa, rdm_cccaaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 1/24 * einsum('xyzw,Juxv,Istv,zwsuyt->IJ', v_aaaa, t1_xaaa, t1_xaaa, rdm_cccaaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 1/24 * einsum('xyzw,Juxv,Istv,zwsuty->IJ', v_aaaa, t1_xaaa, t1_xaaa, rdm_cccaaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 5/24 * einsum('xyzw,Juxv,Istv,zwstyu->IJ', v_aaaa, t1_xaaa, t1_xaaa, rdm_cccaaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 1/24 * einsum('xyzw,Juxv,Istv,zwstuy->IJ', v_aaaa, t1_xaaa, t1_xaaa, rdm_cccaaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/4 * einsum('xyzw,Juxa,Ia,yuwz->IJ', v_aaaa, t1_xaae, t1_xe, rdm_ccaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 1/24 * einsum('xyzw,Juxa,Ivas,zwvyus->IJ', v_aaaa, t1_xaae, t1_xaea, rdm_cccaaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 1/24 * einsum('xyzw,Juxa,Ivas,zwvysu->IJ', v_aaaa, t1_xaae, t1_xaea, rdm_cccaaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 5/24 * einsum('xyzw,Juxa,Ivas,zwvuys->IJ', v_aaaa, t1_xaae, t1_xaea, rdm_cccaaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 1/24 * einsum('xyzw,Juxa,Ivas,zwvusy->IJ', v_aaaa, t1_xaae, t1_xaea, rdm_cccaaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 1/24 * einsum('xyzw,Juxa,Ivas,zwvsyu->IJ', v_aaaa, t1_xaae, t1_xaea, rdm_cccaaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 1/24 * einsum('xyzw,Juxa,Ivas,zwvsuy->IJ', v_aaaa, t1_xaae, t1_xaea, rdm_cccaaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 1/24 * einsum('xyzw,Juxa,Ivsa,zwvyus->IJ', v_aaaa, t1_xaae, t1_xaae, rdm_cccaaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 1/24 * einsum('xyzw,Juxa,Ivsa,zwvysu->IJ', v_aaaa, t1_xaae, t1_xaae, rdm_cccaaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 1/24 * einsum('xyzw,Juxa,Ivsa,zwvuys->IJ', v_aaaa, t1_xaae, t1_xaae, rdm_cccaaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 1/24 * einsum('xyzw,Juxa,Ivsa,zwvusy->IJ', v_aaaa, t1_xaae, t1_xaae, rdm_cccaaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 5/24 * einsum('xyzw,Juxa,Ivsa,zwvsyu->IJ', v_aaaa, t1_xaae, t1_xaae, rdm_cccaaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 1/24 * einsum('xyzw,Juxa,Ivsa,zwvsuy->IJ', v_aaaa, t1_xaae, t1_xaae, rdm_cccaaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/24 * einsum('xyzw,Juvx,Iust,zwvyst->IJ', v_aaaa, t1_xaaa, t1_xaaa, rdm_cccaaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/24 * einsum('xyzw,Juvx,Iust,zwvyts->IJ', v_aaaa, t1_xaaa, t1_xaaa, rdm_cccaaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/24 * einsum('xyzw,Juvx,Iust,zwvsyt->IJ', v_aaaa, t1_xaaa, t1_xaaa, rdm_cccaaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/24 * einsum('xyzw,Juvx,Iust,zwvsty->IJ', v_aaaa, t1_xaaa, t1_xaaa, rdm_cccaaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 5/24 * einsum('xyzw,Juvx,Iust,zwvtys->IJ', v_aaaa, t1_xaaa, t1_xaaa, rdm_cccaaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/24 * einsum('xyzw,Juvx,Iust,zwvtsy->IJ', v_aaaa, t1_xaaa, t1_xaaa, rdm_cccaaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 1/2 * einsum('xyzw,Juvx,Iv,yuwz->IJ', v_aaaa, t1_xaaa, t1_xa, rdm_ccaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/12 * einsum('xyzw,Juvx,Isvt,zwsyut->IJ', v_aaaa, t1_xaaa, t1_xaaa, rdm_cccaaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/12 * einsum('xyzw,Juvx,Isvt,zwsytu->IJ', v_aaaa, t1_xaaa, t1_xaaa, rdm_cccaaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 5/12 * einsum('xyzw,Juvx,Isvt,zwsuyt->IJ', v_aaaa, t1_xaaa, t1_xaaa, rdm_cccaaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/12 * einsum('xyzw,Juvx,Isvt,zwsuty->IJ', v_aaaa, t1_xaaa, t1_xaaa, rdm_cccaaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/12 * einsum('xyzw,Juvx,Isvt,zwstyu->IJ', v_aaaa, t1_xaaa, t1_xaaa, rdm_cccaaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/12 * einsum('xyzw,Juvx,Isvt,zwstuy->IJ', v_aaaa, t1_xaaa, t1_xaaa, rdm_cccaaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 1/24 * einsum('xyzw,Juvx,Istv,zwsyut->IJ', v_aaaa, t1_xaaa, t1_xaaa, rdm_cccaaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 1/24 * einsum('xyzw,Juvx,Istv,zwsytu->IJ', v_aaaa, t1_xaaa, t1_xaaa, rdm_cccaaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 5/24 * einsum('xyzw,Juvx,Istv,zwsuyt->IJ', v_aaaa, t1_xaaa, t1_xaaa, rdm_cccaaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 1/24 * einsum('xyzw,Juvx,Istv,zwsuty->IJ', v_aaaa, t1_xaaa, t1_xaaa, rdm_cccaaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 1/24 * einsum('xyzw,Juvx,Istv,zwstyu->IJ', v_aaaa, t1_xaaa, t1_xaaa, rdm_cccaaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 1/24 * einsum('xyzw,Juvx,Istv,zwstuy->IJ', v_aaaa, t1_xaaa, t1_xaaa, rdm_cccaaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 1/2 * einsum('xyzw,Juax,Ia,yuwz->IJ', v_aaaa, t1_xaea, t1_xe, rdm_ccaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/12 * einsum('xyzw,Juax,Ivas,zwvyus->IJ', v_aaaa, t1_xaea, t1_xaea, rdm_cccaaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/12 * einsum('xyzw,Juax,Ivas,zwvysu->IJ', v_aaaa, t1_xaea, t1_xaea, rdm_cccaaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 5/12 * einsum('xyzw,Juax,Ivas,zwvuys->IJ', v_aaaa, t1_xaea, t1_xaea, rdm_cccaaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/12 * einsum('xyzw,Juax,Ivas,zwvusy->IJ', v_aaaa, t1_xaea, t1_xaea, rdm_cccaaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/12 * einsum('xyzw,Juax,Ivas,zwvsyu->IJ', v_aaaa, t1_xaea, t1_xaea, rdm_cccaaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/12 * einsum('xyzw,Juax,Ivas,zwvsuy->IJ', v_aaaa, t1_xaea, t1_xaea, rdm_cccaaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 1/24 * einsum('xyzw,Juax,Ivsa,zwvyus->IJ', v_aaaa, t1_xaea, t1_xaae, rdm_cccaaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 1/24 * einsum('xyzw,Juax,Ivsa,zwvysu->IJ', v_aaaa, t1_xaea, t1_xaae, rdm_cccaaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 5/24 * einsum('xyzw,Juax,Ivsa,zwvuys->IJ', v_aaaa, t1_xaea, t1_xaae, rdm_cccaaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 1/24 * einsum('xyzw,Juax,Ivsa,zwvusy->IJ', v_aaaa, t1_xaea, t1_xaae, rdm_cccaaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 1/24 * einsum('xyzw,Juax,Ivsa,zwvsyu->IJ', v_aaaa, t1_xaea, t1_xaae, rdm_cccaaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 1/24 * einsum('xyzw,Juax,Ivsa,zwvsuy->IJ', v_aaaa, t1_xaea, t1_xaae, rdm_cccaaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 1/4 * einsum('xyzw,Jixy,Iiuv,zwuv->IJ', v_aaaa, t1_xcaa, t1_xcaa, rdm_ccaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 1/4 * einsum('xyzw,Jixu,Iiuv,yvwz->IJ', v_aaaa, t1_xcaa, t1_xcaa, rdm_ccaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/2 * einsum('xyzw,Jixu,Iivu,yvwz->IJ', v_aaaa, t1_xcaa, t1_xcaa, rdm_ccaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/24 * einsum('xyzw,Jixu,Iivs,zwuyvs->IJ', v_aaaa, t1_xcaa, t1_xcaa, rdm_cccaaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/24 * einsum('xyzw,Jixu,Iivs,zwuysv->IJ', v_aaaa, t1_xcaa, t1_xcaa, rdm_cccaaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 5/24 * einsum('xyzw,Jixu,Iivs,zwuvys->IJ', v_aaaa, t1_xcaa, t1_xcaa, rdm_cccaaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/24 * einsum('xyzw,Jixu,Iivs,zwuvsy->IJ', v_aaaa, t1_xcaa, t1_xcaa, rdm_cccaaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/24 * einsum('xyzw,Jixu,Iivs,zwusyv->IJ', v_aaaa, t1_xcaa, t1_xcaa, rdm_cccaaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/24 * einsum('xyzw,Jixu,Iivs,zwusvy->IJ', v_aaaa, t1_xcaa, t1_xcaa, rdm_cccaaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/2 * einsum('xyzw,Jiux,Iiuv,yvwz->IJ', v_aaaa, t1_xcaa, t1_xcaa, rdm_ccaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 1/4 * einsum('xyzw,Jiux,Iivu,yvwz->IJ', v_aaaa, t1_xcaa, t1_xcaa, rdm_ccaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/24 * einsum('xyzw,Jiux,Iivs,zwuyvs->IJ', v_aaaa, t1_xcaa, t1_xcaa, rdm_cccaaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/24 * einsum('xyzw,Jiux,Iivs,zwuysv->IJ', v_aaaa, t1_xcaa, t1_xcaa, rdm_cccaaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/24 * einsum('xyzw,Jiux,Iivs,zwuvys->IJ', v_aaaa, t1_xcaa, t1_xcaa, rdm_cccaaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/24 * einsum('xyzw,Jiux,Iivs,zwuvsy->IJ', v_aaaa, t1_xcaa, t1_xcaa, rdm_cccaaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 5/24 * einsum('xyzw,Jiux,Iivs,zwusyv->IJ', v_aaaa, t1_xcaa, t1_xcaa, rdm_cccaaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/24 * einsum('xyzw,Jiux,Iivs,zwusvy->IJ', v_aaaa, t1_xcaa, t1_xcaa, rdm_cccaaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/2 * einsum('xyzw,Jiax,Iiau,yuwz->IJ', v_aaaa, t1_xcea, t1_xcea, rdm_ccaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= einsum('xyzw,Jiax,iIaz,wy->IJ', v_aaaa, t1_xcea, t1_cxea, rdm_ca, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 1/2 * einsum('xyzw,Jiax,iIaw,zy->IJ', v_aaaa, t1_xcea, t1_cxea, rdm_ca, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 1/4 * einsum('xyzw,Jiax,iIau,yuwz->IJ', v_aaaa, t1_xcea, t1_cxea, rdm_ccaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 1/4 * einsum('xyzw,iIax,Jiau,yuwz->IJ', v_aaaa, t1_cxea, t1_xcea, rdm_ccaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 2 * einsum('xyzw,iIax,iJaz,yw->IJ', v_aaaa, t1_cxea, t1_cxea, rdm_ca, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= einsum('xyzw,iIax,iJaw,yz->IJ', v_aaaa, t1_cxea, t1_cxea, rdm_ca, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/2 * einsum('xyzw,iIax,iJau,yuwz->IJ', v_aaaa, t1_cxea, t1_cxea, rdm_ccaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] += 1/4 * einsum('xyzw,iJax,Iiau,yuwz->IJ', v_aaaa, t1_cxea, t1_xcea, rdm_ccaa, optimize = einsum_type)
        M[s_c:f_c, s_c:f_c] -= 1/2 * einsum('xyzw,iJax,iIau,yuwz->IJ', v_aaaa, t1_cxea, t1_cxea, rdm_ccaa, optimize = einsum_type)

    print("Time for computing M(h0-h0) block:                %f sec\n" % (time.time() - start_time))
    sys.stdout.flush()

    return M

def compute_M_01(mr_adc):

    start_time = time.time()

    print ("Computing M(h0-h1) blocks...")
    sys.stdout.flush()

    # Einsum definition from kernel
    einsum = mr_adc.interface.einsum
    einsum_type = mr_adc.interface.einsum_type

    # Dimensions
    ncore = mr_adc.ncore
    ncas = mr_adc.ncas
    nocc = mr_adc.nocc
    nextern = mr_adc.nextern

    ncvs = mr_adc.ncvs
    nval = mr_adc.nval

    # Indices
    cvs_tril_ind = np.tril_indices(ncvs, k=-1)

    # MOs Energy
    e_cvs = mr_adc.mo_energy.x
    e_core = mr_adc.mo_energy.c
    e_extern = mr_adc.mo_energy.e

    if nval > 0:
        e_val = mr_adc.mo_energy.v

    # Amplitudes
    t1_ce = mr_adc.t1.ce
    t1_ca = mr_adc.t1.ca
    t1_ae = mr_adc.t1.ae
    t1_caea = mr_adc.t1.caea
    t1_caae = mr_adc.t1.caae
    t1_caaa = mr_adc.t1.caaa
    t1_aaea = mr_adc.t1.aaea
    t1_aaae = mr_adc.t1.aaae

    t1_xa = mr_adc.t1.xa
    t1_xaaa = mr_adc.t1.xaaa

    t1_xe = mr_adc.t1.xe
    t1_xaea = mr_adc.t1.xaea
    t1_xaae = mr_adc.t1.xaae

    if nval > 0:
        t1_ve = mr_adc.t1.ve
        t1_vaea = mr_adc.t1.vaea
        t1_vaae = mr_adc.t1.vaae

        t1_va = mr_adc.t1.va
        t1_vaaa = mr_adc.t1.vaaa

    # One-electron integrals
    h_aa = mr_adc.h1eff.aa
    h_ae = mr_adc.h1eff.ae

    h_xe = mr_adc.h1eff.xe
    h_xa = mr_adc.h1eff.xa

    if nval > 0:
        h_ve = mr_adc.h1eff.ve
        h_va = mr_adc.h1eff.va

    # Two-electrons integrals
    v_aaaa = mr_adc.v2e.aaaa
    v_aaae = mr_adc.v2e.aaae

    v_xaxa = mr_adc.v2e.xaxa
    v_xaax = mr_adc.v2e.xaax

    v_xaxe = mr_adc.v2e.xaxe
    v_xaex = mr_adc.v2e.xaex

    v_xaea = mr_adc.v2e.xaea
    v_xaae = mr_adc.v2e.xaae
    v_xxxe = mr_adc.v2e.xxxe

    v_xxxa = mr_adc.v2e.xxxa
    v_xaaa = mr_adc.v2e.xaaa

    if nval > 0:
        v_vxxe = mr_adc.v2e.vxxe
        v_xvxe = mr_adc.v2e.xvxe

        v_vxxa = mr_adc.v2e.vxxa
        v_xvxa = mr_adc.v2e.xvxa

        v_vaea = mr_adc.v2e.vaea
        v_vaae = mr_adc.v2e.vaae

        v_vaaa = mr_adc.v2e.vaaa

    # Reduced Density Matrices
    rdm_ca = mr_adc.rdm.ca
    rdm_ccaa = mr_adc.rdm.ccaa
    rdm_cccaaa = mr_adc.rdm.cccaaa

    # Excitation Spaces
    dim_caa = mr_adc.h1.dim_caa
    dim_cce = mr_adc.h1.dim_cce
    dim_cae = mr_adc.h1.dim_cae
    dim_cca = mr_adc.h1.dim_cca

    n_caa = mr_adc.h1.n_caa
    n_cce_aaa = mr_adc.h1.n_cce_aaa
    n_cce_abb = mr_adc.h1.n_cce_abb
    n_cae = mr_adc.h1.n_cae
    n_cca_aaa = mr_adc.h1.n_cca_aaa
    n_cca_abb = mr_adc.h1.n_cca_abb

    if nval > 0:
        dim_cve = mr_adc.h1.dim_cve
        dim_cva = mr_adc.h1.dim_cva

        n_cve = mr_adc.h1.n_cve
        n_cva = mr_adc.h1.n_cva

    # C - CAA
    # Oth-order
    M_C_CAA_a_aaa  = 1/2 * einsum('J,IJ,WZ->IJWZ', e_cvs, np.identity(ncvs), rdm_ca, optimize = einsum_type)
    M_C_CAA_a_aaa += 1/2 * einsum('Wx,IJ,xZ->IJWZ', h_aa, np.identity(ncvs), rdm_ca, optimize = einsum_type)
    M_C_CAA_a_aaa -= 1/2 * einsum('Zx,IJ,Wx->IJWZ', h_aa, np.identity(ncvs), rdm_ca, optimize = einsum_type)
    M_C_CAA_a_aaa += 1/2 * einsum('IJ,Wxyz,Zxyz->IJWZ', np.identity(ncvs), v_aaaa, rdm_ccaa, optimize = einsum_type)
    M_C_CAA_a_aaa -= 1/2 * einsum('IJ,Zxyz,Wxyz->IJWZ', np.identity(ncvs), v_aaaa, rdm_ccaa, optimize = einsum_type)

    M_C_CAA_a_abb  = 1/2 * einsum('J,IJ,WZ->IJWZ', e_cvs, np.identity(ncvs), rdm_ca, optimize = einsum_type)
    M_C_CAA_a_abb += 1/2 * einsum('Wx,IJ,xZ->IJWZ', h_aa, np.identity(ncvs), rdm_ca, optimize = einsum_type)
    M_C_CAA_a_abb -= 1/2 * einsum('Zx,IJ,Wx->IJWZ', h_aa, np.identity(ncvs), rdm_ca, optimize = einsum_type)
    M_C_CAA_a_abb += 1/2 * einsum('IJ,Wxyz,Zxyz->IJWZ', np.identity(ncvs), v_aaaa, rdm_ccaa, optimize = einsum_type)
    M_C_CAA_a_abb -= 1/2 * einsum('IJ,Zxyz,Wxyz->IJWZ', np.identity(ncvs), v_aaaa, rdm_ccaa, optimize = einsum_type)

    # 1st-order
    M_C_CAA_a_aaa += 1/2 * einsum('IZJx,Wx->IJWZ', v_xaxa, rdm_ca, optimize = einsum_type)
    M_C_CAA_a_aaa -= 1/2 * einsum('IZxJ,Wx->IJWZ', v_xaax, rdm_ca, optimize = einsum_type)
    M_C_CAA_a_aaa += 1/2 * einsum('IxJy,WxZy->IJWZ', v_xaxa, rdm_ccaa, optimize = einsum_type)
    M_C_CAA_a_aaa -= 1/6 * einsum('IxyJ,WxZy->IJWZ', v_xaax, rdm_ccaa, optimize = einsum_type)
    M_C_CAA_a_aaa += 1/6 * einsum('IxyJ,WxyZ->IJWZ', v_xaax, rdm_ccaa, optimize = einsum_type)
    M_C_CAA_a_aaa -= 1/2 * einsum('IxJy,yx,WZ->IJWZ', v_xaxa, rdm_ca, rdm_ca, optimize = einsum_type)
    M_C_CAA_a_aaa += 1/4 * einsum('IxyJ,yx,WZ->IJWZ', v_xaax, rdm_ca, rdm_ca, optimize = einsum_type)

    M_C_CAA_a_abb += 1/2 * einsum('IZJx,Wx->IJWZ', v_xaxa, rdm_ca, optimize = einsum_type)
    M_C_CAA_a_abb += 1/2 * einsum('IxJy,WxZy->IJWZ', v_xaxa, rdm_ccaa, optimize = einsum_type)
    M_C_CAA_a_abb -= 1/3 * einsum('IxyJ,WxZy->IJWZ', v_xaax, rdm_ccaa, optimize = einsum_type)
    M_C_CAA_a_abb -= 1/6 * einsum('IxyJ,WxyZ->IJWZ', v_xaax, rdm_ccaa, optimize = einsum_type)
    M_C_CAA_a_abb -= 1/2 * einsum('IxJy,yx,WZ->IJWZ', v_xaxa, rdm_ca, rdm_ca, optimize = einsum_type)
    M_C_CAA_a_abb += 1/4 * einsum('IxyJ,yx,WZ->IJWZ', v_xaax, rdm_ca, rdm_ca, optimize = einsum_type)

    M_C_CAA_a_bab =- 1/2 * einsum('IZxJ,Wx->IJWZ', v_xaax, rdm_ca, optimize = einsum_type)
    M_C_CAA_a_bab += 1/6 * einsum('IxyJ,WxZy->IJWZ', v_xaax, rdm_ccaa, optimize = einsum_type)
    M_C_CAA_a_bab += 1/3 * einsum('IxyJ,WxyZ->IJWZ', v_xaax, rdm_ccaa, optimize = einsum_type)

    M_C_CAA_a_aaa = M_C_CAA_a_aaa.reshape(ncvs, -1)
    M_C_CAA_a_abb = M_C_CAA_a_abb.reshape(ncvs, -1)
    M_C_CAA_a_bab = M_C_CAA_a_bab.reshape(ncvs, -1)

    ## Building C-CAA matrix
    m_c_caa_aa_i = 0
    m_c_caa_aa_f = m_c_caa_aa_i + n_caa
    m_c_caa_bb_i = m_c_caa_aa_f
    m_c_caa_bb_f = m_c_caa_bb_i + n_caa
    m_c_caa_ab_i = m_c_caa_bb_f
    m_c_caa_ab_f = m_c_caa_ab_i + n_caa

    M_C_CAA = np.zeros((ncvs, dim_caa))
    M_C_CAA[:, m_c_caa_aa_i:m_c_caa_aa_f] = M_C_CAA_a_aaa.reshape(ncvs, -1)
    M_C_CAA[:, m_c_caa_bb_i:m_c_caa_bb_f] = M_C_CAA_a_abb.reshape(ncvs, -1)
    M_C_CAA[:, m_c_caa_ab_i:m_c_caa_ab_f] = M_C_CAA_a_bab.reshape(ncvs, -1)

    # C - CCE
    M_C_CCE_a_abb  = einsum('KLIB->IKLB', v_xxxe, optimize = einsum_type).copy()
    M_C_CCE_a_abb -= einsum('LB,IK->IKLB', h_xe, np.identity(ncvs), optimize = einsum_type)
    M_C_CCE_a_abb -= einsum('B,IK,LB->IKLB', e_extern, np.identity(ncvs), t1_xe, optimize = einsum_type)
    M_C_CCE_a_abb += einsum('L,IK,LB->IKLB', e_cvs, np.identity(ncvs), t1_xe, optimize = einsum_type)
    M_C_CCE_a_abb -= einsum('IK,LxBy,yx->IKLB', np.identity(ncvs), v_xaea, rdm_ca, optimize = einsum_type)
    M_C_CCE_a_abb += 1/2 * einsum('IK,LxyB,yx->IKLB', np.identity(ncvs), v_xaae, rdm_ca, optimize = einsum_type)
    M_C_CCE_a_abb -= einsum('B,IK,LxBy,yx->IKLB', e_extern, np.identity(ncvs), t1_xaea, rdm_ca, optimize = einsum_type)
    M_C_CCE_a_abb += 1/2 * einsum('B,IK,LxyB,yx->IKLB', e_extern, np.identity(ncvs), t1_xaae, rdm_ca, optimize = einsum_type)
    M_C_CCE_a_abb += einsum('L,IK,LxBy,yx->IKLB', e_cvs, np.identity(ncvs), t1_xaea, rdm_ca, optimize = einsum_type)
    M_C_CCE_a_abb -= 1/2 * einsum('L,IK,LxyB,yx->IKLB', e_cvs, np.identity(ncvs), t1_xaae, rdm_ca, optimize = einsum_type)
    M_C_CCE_a_abb += einsum('xy,IK,LxBz,zy->IKLB', h_aa, np.identity(ncvs), t1_xaea, rdm_ca, optimize = einsum_type)
    M_C_CCE_a_abb -= 1/2 * einsum('xy,IK,LxzB,zy->IKLB', h_aa, np.identity(ncvs), t1_xaae, rdm_ca, optimize = einsum_type)
    M_C_CCE_a_abb -= einsum('xy,IK,LzBx,yz->IKLB', h_aa, np.identity(ncvs), t1_xaea, rdm_ca, optimize = einsum_type)
    M_C_CCE_a_abb += 1/2 * einsum('xy,IK,LzxB,yz->IKLB', h_aa, np.identity(ncvs), t1_xaae, rdm_ca, optimize = einsum_type)
    M_C_CCE_a_abb += einsum('IK,LxBy,xzwu,yzwu->IKLB', np.identity(ncvs), t1_xaea, v_aaaa, rdm_ccaa, optimize = einsum_type)
    M_C_CCE_a_abb -= einsum('IK,LxBy,yzwu,xzwu->IKLB', np.identity(ncvs), t1_xaea, v_aaaa, rdm_ccaa, optimize = einsum_type)
    M_C_CCE_a_abb -= 1/2 * einsum('IK,LxyB,xzwu,yzwu->IKLB', np.identity(ncvs), t1_xaae, v_aaaa, rdm_ccaa, optimize = einsum_type)
    M_C_CCE_a_abb += 1/2 * einsum('IK,LxyB,yzwu,xzwu->IKLB', np.identity(ncvs), t1_xaae, v_aaaa, rdm_ccaa, optimize = einsum_type)

    M_C_CCE_a_aaa = M_C_CCE_a_abb - M_C_CCE_a_abb.transpose(0,2,1,3)

    ## Reshape tensors to matrix form
    M_C_CCE_a_aaa = M_C_CCE_a_aaa[:, cvs_tril_ind[0], cvs_tril_ind[1]]

    M_C_CCE_a_aaa = M_C_CCE_a_aaa.reshape(ncvs, -1)
    M_C_CCE_a_abb = M_C_CCE_a_abb.reshape(ncvs, -1)

    ## Building C-CCE matrix
    s_c_cce_aaa = 0
    f_c_cce_aaa = s_c_cce_aaa + n_cce_aaa
    s_c_cce_abb = f_c_cce_aaa
    f_c_cce_abb = s_c_cce_abb + n_cce_abb

    M_C_CCE = np.zeros((ncvs, dim_cce))
    M_C_CCE[:, s_c_cce_aaa:f_c_cce_aaa] = M_C_CCE_a_aaa.copy()
    M_C_CCE[:, s_c_cce_abb:f_c_cce_abb] = M_C_CCE_a_abb.copy()

    if nval > 0:
        # C - CVE
        M_C_CVE_a_abb  = einsum('KLIB->IKLB', v_xvxe, optimize = einsum_type).copy()
        M_C_CVE_a_abb -= einsum('LB,IK->IKLB', h_ve, np.identity(ncvs), optimize = einsum_type)
        M_C_CVE_a_abb -= einsum('B,IK,LB->IKLB', e_extern, np.identity(ncvs), t1_ve, optimize = einsum_type)
        M_C_CVE_a_abb += einsum('L,IK,LB->IKLB', e_val, np.identity(ncvs), t1_ve, optimize = einsum_type)
        M_C_CVE_a_abb -= einsum('IK,LxBy,yx->IKLB', np.identity(ncvs), v_vaea, rdm_ca, optimize = einsum_type)
        M_C_CVE_a_abb += 1/2 * einsum('IK,LxyB,yx->IKLB', np.identity(ncvs), v_vaae, rdm_ca, optimize = einsum_type)
        M_C_CVE_a_abb -= einsum('B,IK,LxBy,yx->IKLB', e_extern, np.identity(ncvs), t1_vaea, rdm_ca, optimize = einsum_type)
        M_C_CVE_a_abb += 1/2 * einsum('B,IK,LxyB,yx->IKLB', e_extern, np.identity(ncvs), t1_vaae, rdm_ca, optimize = einsum_type)
        M_C_CVE_a_abb += einsum('L,IK,LxBy,yx->IKLB', e_val, np.identity(ncvs), t1_vaea, rdm_ca, optimize = einsum_type)
        M_C_CVE_a_abb -= 1/2 * einsum('L,IK,LxyB,yx->IKLB', e_val, np.identity(ncvs), t1_vaae, rdm_ca, optimize = einsum_type)
        M_C_CVE_a_abb += einsum('xy,IK,LxBz,zy->IKLB', h_aa, np.identity(ncvs), t1_vaea, rdm_ca, optimize = einsum_type)
        M_C_CVE_a_abb -= 1/2 * einsum('xy,IK,LxzB,zy->IKLB', h_aa, np.identity(ncvs), t1_vaae, rdm_ca, optimize = einsum_type)
        M_C_CVE_a_abb -= einsum('xy,IK,LzBx,yz->IKLB', h_aa, np.identity(ncvs), t1_vaea, rdm_ca, optimize = einsum_type)
        M_C_CVE_a_abb += 1/2 * einsum('xy,IK,LzxB,yz->IKLB', h_aa, np.identity(ncvs), t1_vaae, rdm_ca, optimize = einsum_type)
        M_C_CVE_a_abb += einsum('IK,LxBy,xzwu,yzwu->IKLB', np.identity(ncvs), t1_vaea, v_aaaa, rdm_ccaa, optimize = einsum_type)
        M_C_CVE_a_abb -= einsum('IK,LxBy,yzwu,xzwu->IKLB', np.identity(ncvs), t1_vaea, v_aaaa, rdm_ccaa, optimize = einsum_type)
        M_C_CVE_a_abb -= 1/2 * einsum('IK,LxyB,xzwu,yzwu->IKLB', np.identity(ncvs), t1_vaae, v_aaaa, rdm_ccaa, optimize = einsum_type)
        M_C_CVE_a_abb += 1/2 * einsum('IK,LxyB,yzwu,xzwu->IKLB', np.identity(ncvs), t1_vaae, v_aaaa, rdm_ccaa, optimize = einsum_type)

        M_C_CVE_a_bab =- einsum('LKIB->IKLB', v_vxxe, optimize = einsum_type).copy()

        M_C_CVE_a_aaa = np.ascontiguousarray(M_C_CVE_a_abb + M_C_CVE_a_bab)

        ## Reshape tensors to matrix form
        M_C_CVE_a_aaa = M_C_CVE_a_aaa.reshape(ncvs, -1)
        M_C_CVE_a_abb = M_C_CVE_a_abb.reshape(ncvs, -1)
        M_C_CVE_a_bab = M_C_CVE_a_bab.reshape(ncvs, -1)

        ## Building C-CVE matrix
        n_cve = mr_adc.h1.n_cve

        s_c_cve_aaa = 0
        f_c_cve_aaa = s_c_cve_aaa + n_cve
        s_c_cve_abb = f_c_cve_aaa
        f_c_cve_abb = s_c_cve_abb + n_cve
        s_c_cve_bab = f_c_cve_abb
        f_c_cve_bab = s_c_cve_bab + n_cve

        M_C_CVE = np.zeros((ncvs, dim_cve))
        M_C_CVE[:, s_c_cve_aaa:f_c_cve_aaa] = M_C_CVE_a_aaa.copy()
        M_C_CVE[:, s_c_cve_abb:f_c_cve_abb] = M_C_CVE_a_abb.copy()
        M_C_CVE[:, s_c_cve_bab:f_c_cve_bab] = M_C_CVE_a_bab.copy()

    # C - CAE
    M_C_CAE_a_aaa =- 1/2 * einsum('KxBI,Yx->IKYB', v_xaex, rdm_ca, optimize = einsum_type)
    M_C_CAE_a_aaa += 1/2 * einsum('KxIB,Yx->IKYB', v_xaxe, rdm_ca, optimize = einsum_type)
    M_C_CAE_a_aaa -= 1/2 * einsum('xB,IK,Yx->IKYB', h_ae, np.identity(ncvs), rdm_ca, optimize = einsum_type)
    M_C_CAE_a_aaa -= 1/2 * einsum('IK,xyzB,Yzyx->IKYB', np.identity(ncvs), v_aaae, rdm_ccaa, optimize = einsum_type)
    M_C_CAE_a_aaa -= 1/2 * einsum('B,IK,xB,Yx->IKYB', e_extern, np.identity(ncvs), t1_ae, rdm_ca, optimize = einsum_type)
    M_C_CAE_a_aaa -= 1/2 * einsum('B,IK,xyzB,Yzyx->IKYB', e_extern, np.identity(ncvs), t1_aaae, rdm_ccaa, optimize = einsum_type)
    M_C_CAE_a_aaa += 1/2 * einsum('xy,IK,xB,Yy->IKYB', h_aa, np.identity(ncvs), t1_ae, rdm_ca, optimize = einsum_type)
    M_C_CAE_a_aaa -= 1/2 * einsum('xy,IK,zwxB,Yywz->IKYB', h_aa, np.identity(ncvs), t1_aaae, rdm_ccaa, optimize = einsum_type)
    M_C_CAE_a_aaa += 1/2 * einsum('xy,IK,xzwB,Ywzy->IKYB', h_aa, np.identity(ncvs), t1_aaae, rdm_ccaa, optimize = einsum_type)
    M_C_CAE_a_aaa += 1/2 * einsum('xy,IK,zxwB,Ywyz->IKYB', h_aa, np.identity(ncvs), t1_aaae, rdm_ccaa, optimize = einsum_type)
    M_C_CAE_a_aaa += 1/2 * einsum('IK,xB,xyzw,Yyzw->IKYB', np.identity(ncvs), t1_ae, v_aaaa, rdm_ccaa, optimize = einsum_type)
    M_C_CAE_a_aaa += 1/12 * einsum('IK,xyzB,zwuv,Yuvxyw->IKYB', np.identity(ncvs), t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    M_C_CAE_a_aaa += 1/12 * einsum('IK,xyzB,zwuv,Yuvxwy->IKYB', np.identity(ncvs), t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    M_C_CAE_a_aaa -= 5/12 * einsum('IK,xyzB,zwuv,Yuvyxw->IKYB', np.identity(ncvs), t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    M_C_CAE_a_aaa += 1/12 * einsum('IK,xyzB,zwuv,Yuvywx->IKYB', np.identity(ncvs), t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    M_C_CAE_a_aaa += 1/12 * einsum('IK,xyzB,zwuv,Yuvwxy->IKYB', np.identity(ncvs), t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    M_C_CAE_a_aaa += 1/12 * einsum('IK,xyzB,zwuv,Yuvwyx->IKYB', np.identity(ncvs), t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    M_C_CAE_a_aaa += 1/2 * einsum('IK,xyzB,xywu,Yzuw->IKYB', np.identity(ncvs), t1_aaae, v_aaaa, rdm_ccaa, optimize = einsum_type)
    M_C_CAE_a_aaa += 5/12 * einsum('IK,xyzB,xwuv,Yzwyuv->IKYB', np.identity(ncvs), t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    M_C_CAE_a_aaa -= 1/12 * einsum('IK,xyzB,xwuv,Yzwyvu->IKYB', np.identity(ncvs), t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    M_C_CAE_a_aaa -= 1/12 * einsum('IK,xyzB,xwuv,Yzwuyv->IKYB', np.identity(ncvs), t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    M_C_CAE_a_aaa -= 1/12 * einsum('IK,xyzB,xwuv,Yzwuvy->IKYB', np.identity(ncvs), t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    M_C_CAE_a_aaa -= 1/12 * einsum('IK,xyzB,xwuv,Yzwvyu->IKYB', np.identity(ncvs), t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    M_C_CAE_a_aaa -= 1/12 * einsum('IK,xyzB,xwuv,Yzwvuy->IKYB', np.identity(ncvs), t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    M_C_CAE_a_aaa -= 1/12 * einsum('IK,xyzB,ywuv,Yzwxuv->IKYB', np.identity(ncvs), t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    M_C_CAE_a_aaa -= 1/12 * einsum('IK,xyzB,ywuv,Yzwxvu->IKYB', np.identity(ncvs), t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    M_C_CAE_a_aaa += 5/12 * einsum('IK,xyzB,ywuv,Yzwuxv->IKYB', np.identity(ncvs), t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    M_C_CAE_a_aaa -= 1/12 * einsum('IK,xyzB,ywuv,Yzwuvx->IKYB', np.identity(ncvs), t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    M_C_CAE_a_aaa -= 1/12 * einsum('IK,xyzB,ywuv,Yzwvxu->IKYB', np.identity(ncvs), t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    M_C_CAE_a_aaa -= 1/12 * einsum('IK,xyzB,ywuv,Yzwvux->IKYB', np.identity(ncvs), t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)

    M_C_CAE_a_abb  = 1/2 * einsum('KxIB,Yx->IKYB', v_xaxe, rdm_ca, optimize = einsum_type)
    M_C_CAE_a_abb -= 1/2 * einsum('xB,IK,Yx->IKYB', h_ae, np.identity(ncvs), rdm_ca, optimize = einsum_type)
    M_C_CAE_a_abb -= 1/2 * einsum('IK,xyzB,Yzyx->IKYB', np.identity(ncvs), v_aaae, rdm_ccaa, optimize = einsum_type)
    M_C_CAE_a_abb -= 1/2 * einsum('B,IK,xB,Yx->IKYB', e_extern, np.identity(ncvs), t1_ae, rdm_ca, optimize = einsum_type)
    M_C_CAE_a_abb -= 1/2 * einsum('B,IK,xyzB,Yzyx->IKYB', e_extern, np.identity(ncvs), t1_aaae, rdm_ccaa, optimize = einsum_type)
    M_C_CAE_a_abb += 1/2 * einsum('xy,IK,xB,Yy->IKYB', h_aa, np.identity(ncvs), t1_ae, rdm_ca, optimize = einsum_type)
    M_C_CAE_a_abb -= 1/2 * einsum('xy,IK,zwxB,Yywz->IKYB', h_aa, np.identity(ncvs), t1_aaae, rdm_ccaa, optimize = einsum_type)
    M_C_CAE_a_abb += 1/2 * einsum('xy,IK,xzwB,Ywzy->IKYB', h_aa, np.identity(ncvs), t1_aaae, rdm_ccaa, optimize = einsum_type)
    M_C_CAE_a_abb += 1/2 * einsum('xy,IK,zxwB,Ywyz->IKYB', h_aa, np.identity(ncvs), t1_aaae, rdm_ccaa, optimize = einsum_type)
    M_C_CAE_a_abb += 1/2 * einsum('IK,xB,xyzw,Yyzw->IKYB', np.identity(ncvs), t1_ae, v_aaaa, rdm_ccaa, optimize = einsum_type)
    M_C_CAE_a_abb += 1/12 * einsum('IK,xyzB,zwuv,Yuvxyw->IKYB', np.identity(ncvs), t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    M_C_CAE_a_abb += 1/12 * einsum('IK,xyzB,zwuv,Yuvxwy->IKYB', np.identity(ncvs), t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    M_C_CAE_a_abb -= 5/12 * einsum('IK,xyzB,zwuv,Yuvyxw->IKYB', np.identity(ncvs), t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    M_C_CAE_a_abb += 1/12 * einsum('IK,xyzB,zwuv,Yuvywx->IKYB', np.identity(ncvs), t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    M_C_CAE_a_abb += 1/12 * einsum('IK,xyzB,zwuv,Yuvwxy->IKYB', np.identity(ncvs), t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    M_C_CAE_a_abb += 1/12 * einsum('IK,xyzB,zwuv,Yuvwyx->IKYB', np.identity(ncvs), t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    M_C_CAE_a_abb += 1/2 * einsum('IK,xyzB,xywu,Yzuw->IKYB', np.identity(ncvs), t1_aaae, v_aaaa, rdm_ccaa, optimize = einsum_type)
    M_C_CAE_a_abb += 5/12 * einsum('IK,xyzB,xwuv,Yzwyuv->IKYB', np.identity(ncvs), t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    M_C_CAE_a_abb -= 1/12 * einsum('IK,xyzB,xwuv,Yzwyvu->IKYB', np.identity(ncvs), t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    M_C_CAE_a_abb -= 1/12 * einsum('IK,xyzB,xwuv,Yzwuyv->IKYB', np.identity(ncvs), t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    M_C_CAE_a_abb -= 1/12 * einsum('IK,xyzB,xwuv,Yzwuvy->IKYB', np.identity(ncvs), t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    M_C_CAE_a_abb -= 1/12 * einsum('IK,xyzB,xwuv,Yzwvyu->IKYB', np.identity(ncvs), t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    M_C_CAE_a_abb -= 1/12 * einsum('IK,xyzB,xwuv,Yzwvuy->IKYB', np.identity(ncvs), t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    M_C_CAE_a_abb -= 1/12 * einsum('IK,xyzB,ywuv,Yzwxuv->IKYB', np.identity(ncvs), t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    M_C_CAE_a_abb -= 1/12 * einsum('IK,xyzB,ywuv,Yzwxvu->IKYB', np.identity(ncvs), t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    M_C_CAE_a_abb += 5/12 * einsum('IK,xyzB,ywuv,Yzwuxv->IKYB', np.identity(ncvs), t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    M_C_CAE_a_abb -= 1/12 * einsum('IK,xyzB,ywuv,Yzwuvx->IKYB', np.identity(ncvs), t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    M_C_CAE_a_abb -= 1/12 * einsum('IK,xyzB,ywuv,Yzwvxu->IKYB', np.identity(ncvs), t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    M_C_CAE_a_abb -= 1/12 * einsum('IK,xyzB,ywuv,Yzwvux->IKYB', np.identity(ncvs), t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)

    M_C_CAE_a_bab = M_C_CAE_a_aaa - M_C_CAE_a_abb

    ## Reshape tensors to matrix form
    M_C_CAE_a_aaa = M_C_CAE_a_aaa.reshape(ncvs, -1)
    M_C_CAE_a_abb = M_C_CAE_a_abb.reshape(ncvs, -1)
    M_C_CAE_a_bab = M_C_CAE_a_bab.reshape(ncvs, -1)

    ## Building C-CCE matrix
    s_c_cae_aaa = 0
    f_c_cae_aaa = s_c_cae_aaa + n_cae
    s_c_cae_abb = f_c_cae_aaa
    f_c_cae_abb = s_c_cae_abb + n_cae
    s_c_cae_bab = f_c_cae_abb
    f_c_cae_bab = s_c_cae_bab + n_cae

    M_C_CAE = np.zeros((ncvs, dim_cae))
    M_C_CAE[:, s_c_cae_aaa:f_c_cae_aaa] = M_C_CAE_a_aaa.copy()
    M_C_CAE[:, s_c_cae_abb:f_c_cae_abb] = M_C_CAE_a_abb.copy()
    M_C_CAE[:, s_c_cae_bab:f_c_cae_bab] = M_C_CAE_a_bab.copy()

    # C - CCA
    M_C_CCA_a_abb  = einsum('KLIY->IKLY', v_xxxa, optimize = einsum_type).copy()
    M_C_CCA_a_abb -= einsum('LY,IK->IKLY', h_xa, np.identity(ncvs), optimize = einsum_type)
    M_C_CCA_a_abb -= 1/2 * einsum('KLIx,xY->IKLY', v_xxxa, rdm_ca, optimize = einsum_type)
    M_C_CCA_a_abb += einsum('L,IK,LY->IKLY', e_cvs, np.identity(ncvs), t1_xa, optimize = einsum_type)
    M_C_CCA_a_abb += 1/2 * einsum('Lx,IK,xY->IKLY', h_xa, np.identity(ncvs), rdm_ca, optimize = einsum_type)
    M_C_CCA_a_abb -= einsum('Yx,IK,Lx->IKLY', h_aa, np.identity(ncvs), t1_xa, optimize = einsum_type)
    M_C_CCA_a_abb -= einsum('IK,LxYy,yx->IKLY', np.identity(ncvs), v_xaaa, rdm_ca, optimize = einsum_type)
    M_C_CCA_a_abb += 1/2 * einsum('IK,LxyY,yx->IKLY', np.identity(ncvs), v_xaaa, rdm_ca, optimize = einsum_type)
    M_C_CCA_a_abb += 1/2 * einsum('IK,Lxyz,Yxyz->IKLY', np.identity(ncvs), v_xaaa, rdm_ccaa, optimize = einsum_type)
    M_C_CCA_a_abb -= 1/2 * einsum('L,IK,Lx,xY->IKLY', e_cvs, np.identity(ncvs), t1_xa, rdm_ca, optimize = einsum_type)
    M_C_CCA_a_abb += einsum('L,IK,LxYy,yx->IKLY', e_cvs, np.identity(ncvs), t1_xaaa, rdm_ca, optimize = einsum_type)
    M_C_CCA_a_abb -= 1/2 * einsum('L,IK,LxyY,yx->IKLY', e_cvs, np.identity(ncvs), t1_xaaa, rdm_ca, optimize = einsum_type)
    M_C_CCA_a_abb -= 1/2 * einsum('L,IK,Lxyz,Yxyz->IKLY', e_cvs, np.identity(ncvs), t1_xaaa, rdm_ccaa, optimize = einsum_type)
    M_C_CCA_a_abb -= einsum('Yx,IK,Lyxz,zy->IKLY', h_aa, np.identity(ncvs), t1_xaaa, rdm_ca, optimize = einsum_type)
    M_C_CCA_a_abb += 1/2 * einsum('Yx,IK,Lyzx,zy->IKLY', h_aa, np.identity(ncvs), t1_xaaa, rdm_ca, optimize = einsum_type)
    M_C_CCA_a_abb += 1/2 * einsum('xy,IK,Lx,yY->IKLY', h_aa, np.identity(ncvs), t1_xa, rdm_ca, optimize = einsum_type)
    M_C_CCA_a_abb += einsum('xy,IK,LxYz,zy->IKLY', h_aa, np.identity(ncvs), t1_xaaa, rdm_ca, optimize = einsum_type)
    M_C_CCA_a_abb -= 1/2 * einsum('xy,IK,LxzY,zy->IKLY', h_aa, np.identity(ncvs), t1_xaaa, rdm_ca, optimize = einsum_type)
    M_C_CCA_a_abb -= 1/2 * einsum('xy,IK,Lxzw,Yyzw->IKLY', h_aa, np.identity(ncvs), t1_xaaa, rdm_ccaa, optimize = einsum_type)
    M_C_CCA_a_abb -= einsum('xy,IK,LzYx,yz->IKLY', h_aa, np.identity(ncvs), t1_xaaa, rdm_ca, optimize = einsum_type)
    M_C_CCA_a_abb += 1/2 * einsum('xy,IK,LzxY,yz->IKLY', h_aa, np.identity(ncvs), t1_xaaa, rdm_ca, optimize = einsum_type)
    M_C_CCA_a_abb += 1/2 * einsum('xy,IK,Lzxw,Yzyw->IKLY', h_aa, np.identity(ncvs), t1_xaaa, rdm_ccaa, optimize = einsum_type)
    M_C_CCA_a_abb += 1/2 * einsum('xy,IK,Lzwx,Yzwy->IKLY', h_aa, np.identity(ncvs), t1_xaaa, rdm_ccaa, optimize = einsum_type)
    M_C_CCA_a_abb -= einsum('IK,Lx,Yyxz,yz->IKLY', np.identity(ncvs), t1_xa, v_aaaa, rdm_ca, optimize = einsum_type)
    M_C_CCA_a_abb += 1/2 * einsum('IK,Lx,Yyzx,yz->IKLY', np.identity(ncvs), t1_xa, v_aaaa, rdm_ca, optimize = einsum_type)
    M_C_CCA_a_abb += 1/2 * einsum('IK,Lx,xyzw,Yyzw->IKLY', np.identity(ncvs), t1_xa, v_aaaa, rdm_ccaa, optimize = einsum_type)
    M_C_CCA_a_abb += einsum('IK,LxYy,xzwu,yzwu->IKLY', np.identity(ncvs), t1_xaaa, v_aaaa, rdm_ccaa, optimize = einsum_type)
    M_C_CCA_a_abb -= einsum('IK,LxYy,yzwu,xzwu->IKLY', np.identity(ncvs), t1_xaaa, v_aaaa, rdm_ccaa, optimize = einsum_type)
    M_C_CCA_a_abb -= 1/2 * einsum('IK,LxyY,xzwu,yzwu->IKLY', np.identity(ncvs), t1_xaaa, v_aaaa, rdm_ccaa, optimize = einsum_type)
    M_C_CCA_a_abb += 1/2 * einsum('IK,LxyY,yzwu,xzwu->IKLY', np.identity(ncvs), t1_xaaa, v_aaaa, rdm_ccaa, optimize = einsum_type)
    M_C_CCA_a_abb -= 1/2 * einsum('IK,Lxyz,Yxwu,yzwu->IKLY', np.identity(ncvs), t1_xaaa, v_aaaa, rdm_ccaa, optimize = einsum_type)
    M_C_CCA_a_abb -= einsum('IK,Lxyz,Ywyz,wx->IKLY', np.identity(ncvs), t1_xaaa, v_aaaa, rdm_ca, optimize = einsum_type)
    M_C_CCA_a_abb -= einsum('IK,Lxyz,Ywyu,xuzw->IKLY', np.identity(ncvs), t1_xaaa, v_aaaa, rdm_ccaa, optimize = einsum_type)
    M_C_CCA_a_abb += 1/2 * einsum('IK,Lxyz,Ywzy,wx->IKLY', np.identity(ncvs), t1_xaaa, v_aaaa, rdm_ca, optimize = einsum_type)
    M_C_CCA_a_abb += 1/2 * einsum('IK,Lxyz,Ywzu,xuyw->IKLY', np.identity(ncvs), t1_xaaa, v_aaaa, rdm_ccaa, optimize = einsum_type)
    M_C_CCA_a_abb += 1/2 * einsum('IK,Lxyz,Ywuy,xuzw->IKLY', np.identity(ncvs), t1_xaaa, v_aaaa, rdm_ccaa, optimize = einsum_type)
    M_C_CCA_a_abb += 1/2 * einsum('IK,Lxyz,Ywuz,xuwy->IKLY', np.identity(ncvs), t1_xaaa, v_aaaa, rdm_ccaa, optimize = einsum_type)
    M_C_CCA_a_abb -= 5/12 * einsum('IK,Lxyz,xwuv,yzwYuv->IKLY', np.identity(ncvs), t1_xaaa, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    M_C_CCA_a_abb += 1/12 * einsum('IK,Lxyz,xwuv,yzwYvu->IKLY', np.identity(ncvs), t1_xaaa, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    M_C_CCA_a_abb += 1/12 * einsum('IK,Lxyz,xwuv,yzwuYv->IKLY', np.identity(ncvs), t1_xaaa, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    M_C_CCA_a_abb += 1/12 * einsum('IK,Lxyz,xwuv,yzwuvY->IKLY', np.identity(ncvs), t1_xaaa, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    M_C_CCA_a_abb += 1/12 * einsum('IK,Lxyz,xwuv,yzwvYu->IKLY', np.identity(ncvs), t1_xaaa, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    M_C_CCA_a_abb += 1/12 * einsum('IK,Lxyz,xwuv,yzwvuY->IKLY', np.identity(ncvs), t1_xaaa, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    M_C_CCA_a_abb += 1/2 * einsum('IK,Lxyz,yzwu,Yxwu->IKLY', np.identity(ncvs), t1_xaaa, v_aaaa, rdm_ccaa, optimize = einsum_type)
    M_C_CCA_a_abb -= 1/12 * einsum('IK,Lxyz,ywuv,zuvYxw->IKLY', np.identity(ncvs), t1_xaaa, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    M_C_CCA_a_abb -= 1/12 * einsum('IK,Lxyz,ywuv,zuvYwx->IKLY', np.identity(ncvs), t1_xaaa, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    M_C_CCA_a_abb += 5/12 * einsum('IK,Lxyz,ywuv,zuvxYw->IKLY', np.identity(ncvs), t1_xaaa, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    M_C_CCA_a_abb -= 1/12 * einsum('IK,Lxyz,ywuv,zuvxwY->IKLY', np.identity(ncvs), t1_xaaa, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    M_C_CCA_a_abb -= 1/12 * einsum('IK,Lxyz,ywuv,zuvwYx->IKLY', np.identity(ncvs), t1_xaaa, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    M_C_CCA_a_abb -= 1/12 * einsum('IK,Lxyz,ywuv,zuvwxY->IKLY', np.identity(ncvs), t1_xaaa, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    M_C_CCA_a_abb += 5/12 * einsum('IK,Lxyz,zwuv,yuvYxw->IKLY', np.identity(ncvs), t1_xaaa, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    M_C_CCA_a_abb -= 1/12 * einsum('IK,Lxyz,zwuv,yuvYwx->IKLY', np.identity(ncvs), t1_xaaa, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    M_C_CCA_a_abb -= 1/12 * einsum('IK,Lxyz,zwuv,yuvxYw->IKLY', np.identity(ncvs), t1_xaaa, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    M_C_CCA_a_abb -= 1/12 * einsum('IK,Lxyz,zwuv,yuvxwY->IKLY', np.identity(ncvs), t1_xaaa, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    M_C_CCA_a_abb -= 1/12 * einsum('IK,Lxyz,zwuv,yuvwYx->IKLY', np.identity(ncvs), t1_xaaa, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    M_C_CCA_a_abb -= 1/12 * einsum('IK,Lxyz,zwuv,yuvwxY->IKLY', np.identity(ncvs), t1_xaaa, v_aaaa, rdm_cccaaa, optimize = einsum_type)

    M_C_CCA_a_aaa = M_C_CCA_a_abb - M_C_CCA_a_abb.transpose(0,2,1,3)

    ## Reshape tensors to matrix form
    M_C_CCA_a_aaa = M_C_CCA_a_aaa[:, cvs_tril_ind[0], cvs_tril_ind[1]]

    M_C_CCA_a_aaa = M_C_CCA_a_aaa.reshape(ncvs, -1)
    M_C_CCA_a_abb = M_C_CCA_a_abb.reshape(ncvs, -1)

    ## Building C-CCA matrix
    s_c_cca_aaa = 0
    f_c_cca_aaa = s_c_cca_aaa + n_cca_aaa
    s_c_cca_abb = f_c_cca_aaa
    f_c_cca_abb = s_c_cca_abb + n_cca_abb

    M_C_CCA = np.zeros((ncvs, dim_cca))
    M_C_CCA[:, s_c_cca_aaa:f_c_cca_aaa] = M_C_CCA_a_aaa.copy()
    M_C_CCA[:, s_c_cca_abb:f_c_cca_abb] = M_C_CCA_a_abb.copy()

    if nval > 0:
        M_C_CVA_a_abb  = einsum('KLIY->IKLY', v_xvxa, optimize = einsum_type).copy()
        M_C_CVA_a_abb -= einsum('LY,IK->IKLY', h_va, np.identity(ncvs), optimize = einsum_type)
        M_C_CVA_a_abb -= 1/2 * einsum('KLIx,xY->IKLY', v_xvxa, rdm_ca, optimize = einsum_type)
        M_C_CVA_a_abb += einsum('L,IK,LY->IKLY', e_val, np.identity(ncvs), t1_va, optimize = einsum_type)
        M_C_CVA_a_abb += 1/2 * einsum('Lx,IK,xY->IKLY', h_va, np.identity(ncvs), rdm_ca, optimize = einsum_type)
        M_C_CVA_a_abb -= einsum('Yx,IK,Lx->IKLY', h_aa, np.identity(ncvs), t1_va, optimize = einsum_type)
        M_C_CVA_a_abb -= einsum('IK,LxYy,yx->IKLY', np.identity(ncvs), v_vaaa, rdm_ca, optimize = einsum_type)
        M_C_CVA_a_abb += 1/2 * einsum('IK,LxyY,yx->IKLY', np.identity(ncvs), v_vaaa, rdm_ca, optimize = einsum_type)
        M_C_CVA_a_abb += 1/2 * einsum('IK,Lxyz,Yxyz->IKLY', np.identity(ncvs), v_vaaa, rdm_ccaa, optimize = einsum_type)
        M_C_CVA_a_abb -= 1/2 * einsum('L,IK,Lx,xY->IKLY', e_val, np.identity(ncvs), t1_va, rdm_ca, optimize = einsum_type)
        M_C_CVA_a_abb += einsum('L,IK,LxYy,yx->IKLY', e_val, np.identity(ncvs), t1_vaaa, rdm_ca, optimize = einsum_type)
        M_C_CVA_a_abb -= 1/2 * einsum('L,IK,LxyY,yx->IKLY', e_val, np.identity(ncvs), t1_vaaa, rdm_ca, optimize = einsum_type)
        M_C_CVA_a_abb -= 1/2 * einsum('L,IK,Lxyz,Yxyz->IKLY', e_val, np.identity(ncvs), t1_vaaa, rdm_ccaa, optimize = einsum_type)
        M_C_CVA_a_abb -= einsum('Yx,IK,Lyxz,zy->IKLY', h_aa, np.identity(ncvs), t1_vaaa, rdm_ca, optimize = einsum_type)
        M_C_CVA_a_abb += 1/2 * einsum('Yx,IK,Lyzx,zy->IKLY', h_aa, np.identity(ncvs), t1_vaaa, rdm_ca, optimize = einsum_type)
        M_C_CVA_a_abb += 1/2 * einsum('xy,IK,Lx,yY->IKLY', h_aa, np.identity(ncvs), t1_va, rdm_ca, optimize = einsum_type)
        M_C_CVA_a_abb += einsum('xy,IK,LxYz,zy->IKLY', h_aa, np.identity(ncvs), t1_vaaa, rdm_ca, optimize = einsum_type)
        M_C_CVA_a_abb -= 1/2 * einsum('xy,IK,LxzY,zy->IKLY', h_aa, np.identity(ncvs), t1_vaaa, rdm_ca, optimize = einsum_type)
        M_C_CVA_a_abb -= 1/2 * einsum('xy,IK,Lxzw,Yyzw->IKLY', h_aa, np.identity(ncvs), t1_vaaa, rdm_ccaa, optimize = einsum_type)
        M_C_CVA_a_abb -= einsum('xy,IK,LzYx,yz->IKLY', h_aa, np.identity(ncvs), t1_vaaa, rdm_ca, optimize = einsum_type)
        M_C_CVA_a_abb += 1/2 * einsum('xy,IK,LzxY,yz->IKLY', h_aa, np.identity(ncvs), t1_vaaa, rdm_ca, optimize = einsum_type)
        M_C_CVA_a_abb += 1/2 * einsum('xy,IK,Lzxw,Yzyw->IKLY', h_aa, np.identity(ncvs), t1_vaaa, rdm_ccaa, optimize = einsum_type)
        M_C_CVA_a_abb += 1/2 * einsum('xy,IK,Lzwx,Yzwy->IKLY', h_aa, np.identity(ncvs), t1_vaaa, rdm_ccaa, optimize = einsum_type)
        M_C_CVA_a_abb -= einsum('IK,Lx,Yyxz,yz->IKLY', np.identity(ncvs), t1_va, v_aaaa, rdm_ca, optimize = einsum_type)
        M_C_CVA_a_abb += 1/2 * einsum('IK,Lx,Yyzx,yz->IKLY', np.identity(ncvs), t1_va, v_aaaa, rdm_ca, optimize = einsum_type)
        M_C_CVA_a_abb += 1/2 * einsum('IK,Lx,xyzw,Yyzw->IKLY', np.identity(ncvs), t1_va, v_aaaa, rdm_ccaa, optimize = einsum_type)
        M_C_CVA_a_abb += einsum('IK,LxYy,xzwu,yzwu->IKLY', np.identity(ncvs), t1_vaaa, v_aaaa, rdm_ccaa, optimize = einsum_type)
        M_C_CVA_a_abb -= einsum('IK,LxYy,yzwu,xzwu->IKLY', np.identity(ncvs), t1_vaaa, v_aaaa, rdm_ccaa, optimize = einsum_type)
        M_C_CVA_a_abb -= 1/2 * einsum('IK,LxyY,xzwu,yzwu->IKLY', np.identity(ncvs), t1_vaaa, v_aaaa, rdm_ccaa, optimize = einsum_type)
        M_C_CVA_a_abb += 1/2 * einsum('IK,LxyY,yzwu,xzwu->IKLY', np.identity(ncvs), t1_vaaa, v_aaaa, rdm_ccaa, optimize = einsum_type)
        M_C_CVA_a_abb -= 1/2 * einsum('IK,Lxyz,Yxwu,yzwu->IKLY', np.identity(ncvs), t1_vaaa, v_aaaa, rdm_ccaa, optimize = einsum_type)
        M_C_CVA_a_abb -= einsum('IK,Lxyz,Ywyz,wx->IKLY', np.identity(ncvs), t1_vaaa, v_aaaa, rdm_ca, optimize = einsum_type)
        M_C_CVA_a_abb -= einsum('IK,Lxyz,Ywyu,xuzw->IKLY', np.identity(ncvs), t1_vaaa, v_aaaa, rdm_ccaa, optimize = einsum_type)
        M_C_CVA_a_abb += 1/2 * einsum('IK,Lxyz,Ywzy,wx->IKLY', np.identity(ncvs), t1_vaaa, v_aaaa, rdm_ca, optimize = einsum_type)
        M_C_CVA_a_abb += 1/2 * einsum('IK,Lxyz,Ywzu,xuyw->IKLY', np.identity(ncvs), t1_vaaa, v_aaaa, rdm_ccaa, optimize = einsum_type)
        M_C_CVA_a_abb += 1/2 * einsum('IK,Lxyz,Ywuy,xuzw->IKLY', np.identity(ncvs), t1_vaaa, v_aaaa, rdm_ccaa, optimize = einsum_type)
        M_C_CVA_a_abb += 1/2 * einsum('IK,Lxyz,Ywuz,xuwy->IKLY', np.identity(ncvs), t1_vaaa, v_aaaa, rdm_ccaa, optimize = einsum_type)
        M_C_CVA_a_abb -= 5/12 * einsum('IK,Lxyz,xwuv,yzwYuv->IKLY', np.identity(ncvs), t1_vaaa, v_aaaa, rdm_cccaaa, optimize = einsum_type)
        M_C_CVA_a_abb += 1/12 * einsum('IK,Lxyz,xwuv,yzwYvu->IKLY', np.identity(ncvs), t1_vaaa, v_aaaa, rdm_cccaaa, optimize = einsum_type)
        M_C_CVA_a_abb += 1/12 * einsum('IK,Lxyz,xwuv,yzwuYv->IKLY', np.identity(ncvs), t1_vaaa, v_aaaa, rdm_cccaaa, optimize = einsum_type)
        M_C_CVA_a_abb += 1/12 * einsum('IK,Lxyz,xwuv,yzwuvY->IKLY', np.identity(ncvs), t1_vaaa, v_aaaa, rdm_cccaaa, optimize = einsum_type)
        M_C_CVA_a_abb += 1/12 * einsum('IK,Lxyz,xwuv,yzwvYu->IKLY', np.identity(ncvs), t1_vaaa, v_aaaa, rdm_cccaaa, optimize = einsum_type)
        M_C_CVA_a_abb += 1/12 * einsum('IK,Lxyz,xwuv,yzwvuY->IKLY', np.identity(ncvs), t1_vaaa, v_aaaa, rdm_cccaaa, optimize = einsum_type)
        M_C_CVA_a_abb += 1/2 * einsum('IK,Lxyz,yzwu,Yxwu->IKLY', np.identity(ncvs), t1_vaaa, v_aaaa, rdm_ccaa, optimize = einsum_type)
        M_C_CVA_a_abb -= 1/12 * einsum('IK,Lxyz,ywuv,zuvYxw->IKLY', np.identity(ncvs), t1_vaaa, v_aaaa, rdm_cccaaa, optimize = einsum_type)
        M_C_CVA_a_abb -= 1/12 * einsum('IK,Lxyz,ywuv,zuvYwx->IKLY', np.identity(ncvs), t1_vaaa, v_aaaa, rdm_cccaaa, optimize = einsum_type)
        M_C_CVA_a_abb += 5/12 * einsum('IK,Lxyz,ywuv,zuvxYw->IKLY', np.identity(ncvs), t1_vaaa, v_aaaa, rdm_cccaaa, optimize = einsum_type)
        M_C_CVA_a_abb -= 1/12 * einsum('IK,Lxyz,ywuv,zuvxwY->IKLY', np.identity(ncvs), t1_vaaa, v_aaaa, rdm_cccaaa, optimize = einsum_type)
        M_C_CVA_a_abb -= 1/12 * einsum('IK,Lxyz,ywuv,zuvwYx->IKLY', np.identity(ncvs), t1_vaaa, v_aaaa, rdm_cccaaa, optimize = einsum_type)
        M_C_CVA_a_abb -= 1/12 * einsum('IK,Lxyz,ywuv,zuvwxY->IKLY', np.identity(ncvs), t1_vaaa, v_aaaa, rdm_cccaaa, optimize = einsum_type)
        M_C_CVA_a_abb += 5/12 * einsum('IK,Lxyz,zwuv,yuvYxw->IKLY', np.identity(ncvs), t1_vaaa, v_aaaa, rdm_cccaaa, optimize = einsum_type)
        M_C_CVA_a_abb -= 1/12 * einsum('IK,Lxyz,zwuv,yuvYwx->IKLY', np.identity(ncvs), t1_vaaa, v_aaaa, rdm_cccaaa, optimize = einsum_type)
        M_C_CVA_a_abb -= 1/12 * einsum('IK,Lxyz,zwuv,yuvxYw->IKLY', np.identity(ncvs), t1_vaaa, v_aaaa, rdm_cccaaa, optimize = einsum_type)
        M_C_CVA_a_abb -= 1/12 * einsum('IK,Lxyz,zwuv,yuvxwY->IKLY', np.identity(ncvs), t1_vaaa, v_aaaa, rdm_cccaaa, optimize = einsum_type)
        M_C_CVA_a_abb -= 1/12 * einsum('IK,Lxyz,zwuv,yuvwYx->IKLY', np.identity(ncvs), t1_vaaa, v_aaaa, rdm_cccaaa, optimize = einsum_type)
        M_C_CVA_a_abb -= 1/12 * einsum('IK,Lxyz,zwuv,yuvwxY->IKLY', np.identity(ncvs), t1_vaaa, v_aaaa, rdm_cccaaa, optimize = einsum_type)

        M_C_CVA_a_bab =- einsum('LKIY->IKLY', v_vxxa, optimize = einsum_type).copy()
        M_C_CVA_a_bab += 1/2 * einsum('LKIx,xY->IKLY', v_vxxa, rdm_ca, optimize = einsum_type)

        M_C_CVA_a_aaa = M_C_CVA_a_abb + M_C_CVA_a_bab

        M_C_CVA_a_aaa = M_C_CVA_a_aaa.reshape(ncvs, -1)
        M_C_CVA_a_abb = M_C_CVA_a_abb.reshape(ncvs, -1)
        M_C_CVA_a_bab = M_C_CVA_a_bab.reshape(ncvs, -1)

        ## Building C-CVA matrix
        s_c_cva_aaa = 0
        f_c_cva_aaa = s_c_cva_aaa + n_cva
        s_c_cva_abb = f_c_cva_aaa
        f_c_cva_abb = s_c_cva_abb + n_cva
        s_c_cva_bab = f_c_cva_abb
        f_c_cva_bab = s_c_cva_bab + n_cva

        M_C_CVA = np.zeros((ncvs, dim_cva))
        M_C_CVA[:, s_c_cva_aaa:f_c_cva_aaa] = M_C_CVA_a_aaa.copy()
        M_C_CVA[:, s_c_cva_abb:f_c_cva_abb] = M_C_CVA_a_abb.copy()
        M_C_CVA[:, s_c_cva_bab:f_c_cva_bab] = M_C_CVA_a_bab.copy()

    print("Time for computing M(h0-h1) blocks:               %f sec\n" % (time.time() - start_time))
    sys.stdout.flush()

    if nval > 0:
        M_01 = (M_C_CAA, M_C_CCE, M_C_CVE, M_C_CAE, M_C_CCA, M_C_CVA)
    else:
        M_01 = (M_C_CAA, M_C_CCE, M_C_CAE, M_C_CCA)

    return M_01

def define_effective_hamiltonian(mr_adc, M_00, M_01 = None, M_11 = None):

    apply_M = None

    # MR-ADC(0) and MR-ADC(1)
    if mr_adc.method in ("mr-adc(0)", "mr-adc(1)"):
        def apply_M(X):
            sigma = np.dot(M_00, X)

            # Multiply by -1.0, since we are solving for -M C = -S C E
            sigma *= -1.0

            return sigma

    # MR-ADC(2)
    else:
        def apply_M(X):
            # Xt = S_12 X
            Xt = apply_S_12(mr_adc, X)

            # Apply M: Sigma = M Xt
            sigma = compute_sigma_vector(mr_adc, M_00, M_01, M_11, Xt)

            # Multiply by -1.0, since we are solving for -M C = -S C E
            sigma *= -1.0

            # X_new = S_12.T Sigma
            X_new = apply_S_12(mr_adc, sigma, transpose = True)

            return X_new

    return apply_M

def compute_preconditioner(mr_adc, M_00):

    start_time = time.time()

    # Einsum definition from kernel
    einsum = mr_adc.interface.einsum
    einsum_type = mr_adc.interface.einsum_type

    if mr_adc.method in ("mr-adc(0)", "mr-adc(1)"):

        # Multiply by -1.0, since we are solving for -M C = -S C E
        return (-1.0 * np.diag(M_00))

    # Variables from kernel
    ncvs = mr_adc.ncvs
    nval = mr_adc.nval
    ncas = mr_adc.ncas
    nextern = mr_adc.nextern

    e_cvs = mr_adc.mo_energy.x
    e_extern = mr_adc.mo_energy.e

    if nval > 0:
        e_val = mr_adc.mo_energy.v

    # One-electron integrals
    h_aa = mr_adc.h1eff.aa

    # Two-electrons integrals
    v_aaaa = mr_adc.v2e.aaaa

    v_xaxa = mr_adc.v2e.xaxa
    v_xaax = mr_adc.v2e.xaax

    # Reduced Density Matrices
    rdm_ca = mr_adc.rdm.ca
    rdm_ccaa = mr_adc.rdm.ccaa
    rdm_cccaaa = mr_adc.rdm.cccaaa

    # Overlap Matrices
    S12_c_caa = mr_adc.S12.c_caa
    S12_cae = mr_adc.S12.cae
    S12_cca = mr_adc.S12.cca

    # Excitation Spaces
    ho_s_c_caa = mr_adc.h_orth.s_c_caa
    ho_f_c_caa = mr_adc.h_orth.f_c_caa

    ho_s_cce_aaa = mr_adc.h_orth.s_cce_aaa
    ho_f_cce_aaa = mr_adc.h_orth.f_cce_aaa
    ho_s_cce_abb = mr_adc.h_orth.s_cce_abb
    ho_f_cce_abb = mr_adc.h_orth.f_cce_abb

    ho_s_cae_aaa = mr_adc.h_orth.s_cae_aaa
    ho_f_cae_aaa = mr_adc.h_orth.f_cae_aaa
    ho_s_cae_abb = mr_adc.h_orth.s_cae_abb
    ho_f_cae_abb = mr_adc.h_orth.f_cae_abb
    ho_s_cae_bab = mr_adc.h_orth.s_cae_bab
    ho_f_cae_bab = mr_adc.h_orth.f_cae_bab

    ho_s_cca_aaa = mr_adc.h_orth.s_cca_aaa
    ho_f_cca_aaa = mr_adc.h_orth.f_cca_aaa
    ho_s_cca_abb = mr_adc.h_orth.s_cca_abb
    ho_f_cca_abb = mr_adc.h_orth.f_cca_abb

    if nval > 0:
        ho_s_cve = mr_adc.h_orth.s_cve
        ho_f_cve = mr_adc.h_orth.f_cve

        ho_s_cva = mr_adc.h_orth.s_cva
        ho_f_cva = mr_adc.h_orth.f_cva

        ho_s_cve_aaa = mr_adc.h_orth.s_cve_aaa
        ho_f_cve_aaa = mr_adc.h_orth.f_cve_aaa
        ho_s_cve_abb = mr_adc.h_orth.s_cve_abb
        ho_f_cve_abb = mr_adc.h_orth.f_cve_abb
        ho_s_cve_bab = mr_adc.h_orth.s_cve_bab
        ho_f_cve_bab = mr_adc.h_orth.f_cve_bab

        ho_s_cva_aaa = mr_adc.h_orth.s_cva_aaa
        ho_f_cva_aaa = mr_adc.h_orth.f_cva_aaa
        ho_s_cva_abb = mr_adc.h_orth.s_cva_abb
        ho_f_cva_abb = mr_adc.h_orth.f_cva_abb
        ho_s_cva_bab = mr_adc.h_orth.s_cva_bab
        ho_f_cva_bab = mr_adc.h_orth.f_cva_bab

    s_c = mr_adc.h0.s_c
    f_c = mr_adc.h0.f_c

    cvs_tril_ind = np.tril_indices(ncvs, k=-1)

    # Build the preconditioner
    precond = np.zeros(mr_adc.h_orth.dim)

    # C and CAA
    # 0th-order
    precond_c_caa_a_aaa  = 1/2 * einsum('I,II,XY->IXY', e_cvs, np.identity(ncvs), rdm_ca, optimize = einsum_type)
    precond_c_caa_a_abb  = 1/2 * einsum('I,II,XY->IXY', e_cvs, np.identity(ncvs), rdm_ca, optimize = einsum_type)

    # 1st-order
    precond_c_caa_a_aaa += 1/2 * einsum('IxIY,Xx->IXY', v_xaxa, rdm_ca, optimize = einsum_type)
    precond_c_caa_a_aaa += 1/2 * einsum('IxIy,XyYx->IXY', v_xaxa, rdm_ccaa, optimize = einsum_type)
    precond_c_caa_a_aaa -= 1/2 * einsum('IxYI,Xx->IXY', v_xaax, rdm_ca, optimize = einsum_type)
    precond_c_caa_a_aaa -= 1/6 * einsum('IxyI,XyYx->IXY', v_xaax, rdm_ccaa, optimize = einsum_type)
    precond_c_caa_a_aaa += 1/6 * einsum('IxyI,XyxY->IXY', v_xaax, rdm_ccaa, optimize = einsum_type)
    precond_c_caa_a_aaa -= 1/2 * einsum('IxIy,xy,XY->IXY', v_xaxa, rdm_ca, rdm_ca, optimize = einsum_type)
    precond_c_caa_a_aaa += 1/4 * einsum('IxyI,xy,XY->IXY', v_xaax, rdm_ca, rdm_ca, optimize = einsum_type)

    precond_c_caa_a_abb += 1/2 * einsum('IxIY,Xx->IXY', v_xaxa, rdm_ca, optimize = einsum_type)
    precond_c_caa_a_abb += 1/2 * einsum('IxIy,XyYx->IXY', v_xaxa, rdm_ccaa, optimize = einsum_type)
    precond_c_caa_a_abb -= 1/3 * einsum('IxyI,XyYx->IXY', v_xaax, rdm_ccaa, optimize = einsum_type)
    precond_c_caa_a_abb -= 1/6 * einsum('IxyI,XyxY->IXY', v_xaax, rdm_ccaa, optimize = einsum_type)
    precond_c_caa_a_abb -= 1/2 * einsum('IxIy,xy,XY->IXY', v_xaxa, rdm_ca, rdm_ca, optimize = einsum_type)
    precond_c_caa_a_abb += 1/4 * einsum('IxyI,xy,XY->IXY', v_xaax, rdm_ca, rdm_ca, optimize = einsum_type)

    precond_caa_aaa =- 1/6 * einsum('I,II,WYXZ->IWZXY', e_cvs, np.identity(ncvs), rdm_ccaa, optimize = einsum_type)
    precond_caa_aaa += 1/6 * einsum('I,II,WYZX->IWZXY', e_cvs, np.identity(ncvs), rdm_ccaa, optimize = einsum_type)
    precond_caa_aaa += 1/6 * einsum('Xx,II,WYZx->IWZXY', h_aa, np.identity(ncvs), rdm_ccaa, optimize = einsum_type)
    precond_caa_aaa -= 1/6 * einsum('Xx,II,WYxZ->IWZXY', h_aa, np.identity(ncvs), rdm_ccaa, optimize = einsum_type)
    precond_caa_aaa -= 1/2 * einsum('YZ,II,XW->IWZXY', h_aa, np.identity(ncvs), rdm_ca, optimize = einsum_type)
    precond_caa_aaa += 1/6 * einsum('Yx,II,WxXZ->IWZXY', h_aa, np.identity(ncvs), rdm_ccaa, optimize = einsum_type)
    precond_caa_aaa -= 1/6 * einsum('Yx,II,WxZX->IWZXY', h_aa, np.identity(ncvs), rdm_ccaa, optimize = einsum_type)
    precond_caa_aaa -= 1/6 * einsum('II,XZxy,WYxy->IWZXY', np.identity(ncvs), v_aaaa, rdm_ccaa, optimize = einsum_type)
    precond_caa_aaa += 1/6 * einsum('II,XZxy,WYyx->IWZXY', np.identity(ncvs), v_aaaa, rdm_ccaa, optimize = einsum_type)
    precond_caa_aaa += 1/6 * einsum('II,Xxyz,ZyzWYx->IWZXY', np.identity(ncvs), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    precond_caa_aaa -= 1/6 * einsum('II,Xxyz,ZyzYWx->IWZXY', np.identity(ncvs), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    precond_caa_aaa -= 1/2 * einsum('II,YxZy,WyXx->IWZXY', np.identity(ncvs), v_aaaa, rdm_ccaa, optimize = einsum_type)
    precond_caa_aaa += 1/6 * einsum('II,YxyZ,WyXx->IWZXY', np.identity(ncvs), v_aaaa, rdm_ccaa, optimize = einsum_type)
    precond_caa_aaa -= 1/6 * einsum('II,YxyZ,WyxX->IWZXY', np.identity(ncvs), v_aaaa, rdm_ccaa, optimize = einsum_type)
    precond_caa_aaa += 1/6 * einsum('II,Yxyz,XZxWyz->IWZXY', np.identity(ncvs), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    precond_caa_aaa -= 1/6 * einsum('II,Yxyz,XZxyWz->IWZXY', np.identity(ncvs), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    precond_caa_aaa += 1/2 * einsum('I,II,YZ,XW->IWZXY', e_cvs, np.identity(ncvs), np.identity(ncas), rdm_ca, optimize = einsum_type)
    precond_caa_aaa += 1/2 * einsum('Xx,II,YZ,xW->IWZXY', h_aa, np.identity(ncvs), np.identity(ncas), rdm_ca, optimize = einsum_type)
    precond_caa_aaa += 1/2 * einsum('Xxyz,II,YZ,Wxyz->IWZXY', v_aaaa, np.identity(ncvs), np.identity(ncas), rdm_ccaa, optimize = einsum_type)

    precond_caa_abb =- 1/6 * einsum('I,II,WYXZ->IWZXY', e_cvs, np.identity(ncvs), rdm_ccaa, optimize = einsum_type)
    precond_caa_abb += 1/6 * einsum('I,II,WYZX->IWZXY', e_cvs, np.identity(ncvs), rdm_ccaa, optimize = einsum_type)
    precond_caa_abb += 1/6 * einsum('Xx,II,WYZx->IWZXY', h_aa, np.identity(ncvs), rdm_ccaa, optimize = einsum_type)
    precond_caa_abb -= 1/6 * einsum('Xx,II,WYxZ->IWZXY', h_aa, np.identity(ncvs), rdm_ccaa, optimize = einsum_type)
    precond_caa_abb -= 1/2 * einsum('YZ,II,XW->IWZXY', h_aa, np.identity(ncvs), rdm_ca, optimize = einsum_type)
    precond_caa_abb += 1/6 * einsum('Yx,II,WxXZ->IWZXY', h_aa, np.identity(ncvs), rdm_ccaa, optimize = einsum_type)
    precond_caa_abb -= 1/6 * einsum('Yx,II,WxZX->IWZXY', h_aa, np.identity(ncvs), rdm_ccaa, optimize = einsum_type)
    precond_caa_abb -= 1/6 * einsum('II,XZxy,WYxy->IWZXY', np.identity(ncvs), v_aaaa, rdm_ccaa, optimize = einsum_type)
    precond_caa_abb += 1/6 * einsum('II,XZxy,WYyx->IWZXY', np.identity(ncvs), v_aaaa, rdm_ccaa, optimize = einsum_type)
    precond_caa_abb += 1/6 * einsum('II,Xxyz,ZyzWYx->IWZXY', np.identity(ncvs), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    precond_caa_abb -= 1/6 * einsum('II,Xxyz,ZyzYWx->IWZXY', np.identity(ncvs), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    precond_caa_abb -= 1/2 * einsum('II,YxZy,WyXx->IWZXY', np.identity(ncvs), v_aaaa, rdm_ccaa, optimize = einsum_type)
    precond_caa_abb += 1/6 * einsum('II,YxyZ,WyXx->IWZXY', np.identity(ncvs), v_aaaa, rdm_ccaa, optimize = einsum_type)
    precond_caa_abb -= 1/6 * einsum('II,YxyZ,WyxX->IWZXY', np.identity(ncvs), v_aaaa, rdm_ccaa, optimize = einsum_type)
    precond_caa_abb += 1/6 * einsum('II,Yxyz,XZxWyz->IWZXY', np.identity(ncvs), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    precond_caa_abb -= 1/6 * einsum('II,Yxyz,XZxyWz->IWZXY', np.identity(ncvs), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    precond_caa_abb += 1/2 * einsum('I,II,YZ,XW->IWZXY', e_cvs, np.identity(ncvs), np.identity(ncas), rdm_ca, optimize = einsum_type)
    precond_caa_abb += 1/2 * einsum('Xx,II,YZ,xW->IWZXY', h_aa, np.identity(ncvs), np.identity(ncas), rdm_ca, optimize = einsum_type)
    precond_caa_abb += 1/2 * einsum('Xxyz,II,YZ,Wxyz->IWZXY', v_aaaa, np.identity(ncvs), np.identity(ncas), rdm_ccaa, optimize = einsum_type)

    precond_caa_bab =- 1/3 * einsum('I,II,WYXZ->IWZXY', e_cvs, np.identity(ncvs), rdm_ccaa, optimize = einsum_type)
    precond_caa_bab -= 1/6 * einsum('I,II,WYZX->IWZXY', e_cvs, np.identity(ncvs), rdm_ccaa, optimize = einsum_type)
    precond_caa_bab -= 1/6 * einsum('Xx,II,WYZx->IWZXY', h_aa, np.identity(ncvs), rdm_ccaa, optimize = einsum_type)
    precond_caa_bab -= 1/3 * einsum('Xx,II,WYxZ->IWZXY', h_aa, np.identity(ncvs), rdm_ccaa, optimize = einsum_type)
    precond_caa_bab -= 1/2 * einsum('YZ,II,XW->IWZXY', h_aa, np.identity(ncvs), rdm_ca, optimize = einsum_type)
    precond_caa_bab += 1/3 * einsum('Yx,II,WxXZ->IWZXY', h_aa, np.identity(ncvs), rdm_ccaa, optimize = einsum_type)
    precond_caa_bab += 1/6 * einsum('Yx,II,WxZX->IWZXY', h_aa, np.identity(ncvs), rdm_ccaa, optimize = einsum_type)
    precond_caa_bab -= 1/3 * einsum('II,XZxy,WYxy->IWZXY', np.identity(ncvs), v_aaaa, rdm_ccaa, optimize = einsum_type)
    precond_caa_bab -= 1/6 * einsum('II,XZxy,WYyx->IWZXY', np.identity(ncvs), v_aaaa, rdm_ccaa, optimize = einsum_type)
    precond_caa_bab -= 1/12 * einsum('II,Xxyz,ZyzWYx->IWZXY', np.identity(ncvs), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    precond_caa_bab += 1/12 * einsum('II,Xxyz,ZyzWxY->IWZXY', np.identity(ncvs), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    precond_caa_bab -= 1/4 * einsum('II,Xxyz,ZyzYWx->IWZXY', np.identity(ncvs), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    precond_caa_bab += 1/12 * einsum('II,Xxyz,ZyzYxW->IWZXY', np.identity(ncvs), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    precond_caa_bab += 1/12 * einsum('II,Xxyz,ZyzxWY->IWZXY', np.identity(ncvs), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    precond_caa_bab += 1/12 * einsum('II,Xxyz,ZyzxYW->IWZXY', np.identity(ncvs), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    precond_caa_bab -= 1/2 * einsum('II,YxZy,WyXx->IWZXY', np.identity(ncvs), v_aaaa, rdm_ccaa, optimize = einsum_type)
    precond_caa_bab += 1/3 * einsum('II,YxyZ,WyXx->IWZXY', np.identity(ncvs), v_aaaa, rdm_ccaa, optimize = einsum_type)
    precond_caa_bab += 1/6 * einsum('II,YxyZ,WyxX->IWZXY', np.identity(ncvs), v_aaaa, rdm_ccaa, optimize = einsum_type)
    precond_caa_bab += 1/4 * einsum('II,Yxyz,XZxWyz->IWZXY', np.identity(ncvs), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    precond_caa_bab -= 1/12 * einsum('II,Yxyz,XZxWzy->IWZXY', np.identity(ncvs), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    precond_caa_bab += 1/12 * einsum('II,Yxyz,XZxyWz->IWZXY', np.identity(ncvs), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    precond_caa_bab -= 1/12 * einsum('II,Yxyz,XZxyzW->IWZXY', np.identity(ncvs), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    precond_caa_bab -= 1/12 * einsum('II,Yxyz,XZxzWy->IWZXY', np.identity(ncvs), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    precond_caa_bab -= 1/12 * einsum('II,Yxyz,XZxzyW->IWZXY', np.identity(ncvs), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    precond_caa_bab += 1/2 * einsum('I,II,YZ,XW->IWZXY', e_cvs, np.identity(ncvs), np.identity(ncas), rdm_ca, optimize = einsum_type)
    precond_caa_bab += 1/2 * einsum('Xx,II,YZ,xW->IWZXY', h_aa, np.identity(ncvs), np.identity(ncas), rdm_ca, optimize = einsum_type)
    precond_caa_bab += 1/2 * einsum('Xxyz,II,YZ,Wxyz->IWZXY', v_aaaa, np.identity(ncvs), np.identity(ncas), rdm_ccaa, optimize = einsum_type)

    # Off-diagonal terms
    precond_caa_aaa_abb  = 1/6 * einsum('I,II,WYXZ->IWZXY', e_cvs, np.identity(ncvs), rdm_ccaa, optimize = einsum_type)
    precond_caa_aaa_abb += 1/3 * einsum('I,II,WYZX->IWZXY', e_cvs, np.identity(ncvs), rdm_ccaa, optimize = einsum_type)
    precond_caa_aaa_abb += 1/3 * einsum('Xx,II,WYZx->IWZXY', h_aa, np.identity(ncvs), rdm_ccaa, optimize = einsum_type)
    precond_caa_aaa_abb += 1/6 * einsum('Xx,II,WYxZ->IWZXY', h_aa, np.identity(ncvs), rdm_ccaa, optimize = einsum_type)
    precond_caa_aaa_abb -= 1/6 * einsum('Yx,II,WxXZ->IWZXY', h_aa, np.identity(ncvs), rdm_ccaa, optimize = einsum_type)
    precond_caa_aaa_abb -= 1/3 * einsum('Yx,II,WxZX->IWZXY', h_aa, np.identity(ncvs), rdm_ccaa, optimize = einsum_type)
    precond_caa_aaa_abb += 1/6 * einsum('II,XZxy,WYxy->IWZXY', np.identity(ncvs), v_aaaa, rdm_ccaa, optimize = einsum_type)
    precond_caa_aaa_abb += 1/3 * einsum('II,XZxy,WYyx->IWZXY', np.identity(ncvs), v_aaaa, rdm_ccaa, optimize = einsum_type)
    precond_caa_aaa_abb += 1/4 * einsum('II,Xxyz,ZyzWYx->IWZXY', np.identity(ncvs), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    precond_caa_aaa_abb -= 1/12 * einsum('II,Xxyz,ZyzWxY->IWZXY', np.identity(ncvs), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    precond_caa_aaa_abb += 1/12 * einsum('II,Xxyz,ZyzYWx->IWZXY', np.identity(ncvs), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    precond_caa_aaa_abb -= 1/12 * einsum('II,Xxyz,ZyzYxW->IWZXY', np.identity(ncvs), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    precond_caa_aaa_abb -= 1/12 * einsum('II,Xxyz,ZyzxWY->IWZXY', np.identity(ncvs), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    precond_caa_aaa_abb -= 1/12 * einsum('II,Xxyz,ZyzxYW->IWZXY', np.identity(ncvs), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    precond_caa_aaa_abb -= 1/6 * einsum('II,YxyZ,WyXx->IWZXY', np.identity(ncvs), v_aaaa, rdm_ccaa, optimize = einsum_type)
    precond_caa_aaa_abb -= 1/3 * einsum('II,YxyZ,WyxX->IWZXY', np.identity(ncvs), v_aaaa, rdm_ccaa, optimize = einsum_type)
    precond_caa_aaa_abb -= 1/12 * einsum('II,Yxyz,XZxWyz->IWZXY', np.identity(ncvs), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    precond_caa_aaa_abb += 1/12 * einsum('II,Yxyz,XZxWzy->IWZXY', np.identity(ncvs), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    precond_caa_aaa_abb -= 1/4 * einsum('II,Yxyz,XZxyWz->IWZXY', np.identity(ncvs), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    precond_caa_aaa_abb += 1/12 * einsum('II,Yxyz,XZxyzW->IWZXY', np.identity(ncvs), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    precond_caa_aaa_abb += 1/12 * einsum('II,Yxyz,XZxzWy->IWZXY', np.identity(ncvs), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    precond_caa_aaa_abb += 1/12 * einsum('II,Yxyz,XZxzyW->IWZXY', np.identity(ncvs), v_aaaa, rdm_cccaaa, optimize = einsum_type)

    precond_caa_abb_aaa  = 1/6 * einsum('I,II,WYXZ->IWZXY', e_cvs, np.identity(ncvs), rdm_ccaa, optimize = einsum_type)
    precond_caa_abb_aaa += 1/3 * einsum('I,II,WYZX->IWZXY', e_cvs, np.identity(ncvs), rdm_ccaa, optimize = einsum_type)
    precond_caa_abb_aaa += 1/3 * einsum('Xx,II,WYZx->IWZXY', h_aa, np.identity(ncvs), rdm_ccaa, optimize = einsum_type)
    precond_caa_abb_aaa += 1/6 * einsum('Xx,II,WYxZ->IWZXY', h_aa, np.identity(ncvs), rdm_ccaa, optimize = einsum_type)
    precond_caa_abb_aaa -= 1/6 * einsum('Yx,II,WxXZ->IWZXY', h_aa, np.identity(ncvs), rdm_ccaa, optimize = einsum_type)
    precond_caa_abb_aaa -= 1/3 * einsum('Yx,II,WxZX->IWZXY', h_aa, np.identity(ncvs), rdm_ccaa, optimize = einsum_type)
    precond_caa_abb_aaa += 1/6 * einsum('II,XZxy,WYxy->IWZXY', np.identity(ncvs), v_aaaa, rdm_ccaa, optimize = einsum_type)
    precond_caa_abb_aaa += 1/3 * einsum('II,XZxy,WYyx->IWZXY', np.identity(ncvs), v_aaaa, rdm_ccaa, optimize = einsum_type)
    precond_caa_abb_aaa += 1/4 * einsum('II,Xxyz,ZyzWYx->IWZXY', np.identity(ncvs), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    precond_caa_abb_aaa -= 1/12 * einsum('II,Xxyz,ZyzWxY->IWZXY', np.identity(ncvs), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    precond_caa_abb_aaa += 1/12 * einsum('II,Xxyz,ZyzYWx->IWZXY', np.identity(ncvs), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    precond_caa_abb_aaa -= 1/12 * einsum('II,Xxyz,ZyzYxW->IWZXY', np.identity(ncvs), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    precond_caa_abb_aaa -= 1/12 * einsum('II,Xxyz,ZyzxWY->IWZXY', np.identity(ncvs), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    precond_caa_abb_aaa -= 1/12 * einsum('II,Xxyz,ZyzxYW->IWZXY', np.identity(ncvs), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    precond_caa_abb_aaa -= 1/6 * einsum('II,YxyZ,WyXx->IWZXY', np.identity(ncvs), v_aaaa, rdm_ccaa, optimize = einsum_type)
    precond_caa_abb_aaa -= 1/3 * einsum('II,YxyZ,WyxX->IWZXY', np.identity(ncvs), v_aaaa, rdm_ccaa, optimize = einsum_type)
    precond_caa_abb_aaa -= 1/12 * einsum('II,Yxyz,XZxWyz->IWZXY', np.identity(ncvs), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    precond_caa_abb_aaa += 1/12 * einsum('II,Yxyz,XZxWzy->IWZXY', np.identity(ncvs), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    precond_caa_abb_aaa -= 1/4 * einsum('II,Yxyz,XZxyWz->IWZXY', np.identity(ncvs), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    precond_caa_abb_aaa += 1/12 * einsum('II,Yxyz,XZxyzW->IWZXY', np.identity(ncvs), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    precond_caa_abb_aaa += 1/12 * einsum('II,Yxyz,XZxzWy->IWZXY', np.identity(ncvs), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    precond_caa_abb_aaa += 1/12 * einsum('II,Yxyz,XZxzyW->IWZXY', np.identity(ncvs), v_aaaa, rdm_cccaaa, optimize = einsum_type)

    ## Building C-CAA matrix
    dim_XY = ncas * ncas
    dim_c_caa = 3 * dim_XY

    precond_aa_i = 1
    precond_aa_f = precond_aa_i + dim_XY
    precond_bb_i = precond_aa_f
    precond_bb_f = precond_bb_i + dim_XY
    precond_ab_i = precond_bb_f
    precond_ab_f = precond_ab_i + dim_XY

    precond_temp = np.zeros((ncvs, (1 + dim_c_caa), (1 + dim_c_caa)))
    precond_temp[:, 0, 0] = np.diag(M_00[s_c:f_c, s_c:f_c]).copy()

    precond_temp[:, 0, precond_aa_i:precond_aa_f] = precond_c_caa_a_aaa.reshape(ncvs, ncas * ncas).copy()
    precond_temp[:, 0, precond_bb_i:precond_bb_f] = precond_c_caa_a_abb.reshape(ncvs, ncas * ncas).copy()
    precond_temp[:, precond_aa_i:precond_ab_f, 0] = precond_temp[:, 0, precond_aa_i:precond_ab_f].copy()

    precond_temp[:, precond_aa_i:precond_aa_f, precond_aa_i:precond_aa_f] = precond_caa_aaa.reshape(ncvs, ncas * ncas, ncas * ncas).copy()
    precond_temp[:, precond_aa_i:precond_aa_f, precond_bb_i:precond_bb_f] = precond_caa_aaa_abb.reshape(ncvs, ncas * ncas, ncas * ncas).copy()

    precond_temp[:, precond_bb_i:precond_bb_f, precond_bb_i:precond_bb_f] = precond_caa_abb.reshape(ncvs, ncas * ncas, ncas * ncas).copy()
    precond_temp[:, precond_bb_i:precond_bb_f, precond_aa_i:precond_aa_f] = precond_caa_abb_aaa.reshape(ncvs, ncas * ncas, ncas * ncas).copy()

    precond_temp[:, precond_ab_i:precond_ab_f, precond_ab_i:precond_ab_f] = precond_caa_bab.reshape(ncvs, ncas * ncas, ncas * ncas).copy()

    precond_temp = einsum('IXY,XP,YP->IP', precond_temp, S12_c_caa, S12_c_caa, optimize = einsum_type)

    precond[ho_s_c_caa:ho_f_c_caa] = precond_temp.reshape(-1)

    # CCE
    precond_cce =- einsum('A,AA,II,JJ->IJA', e_extern, np.identity(nextern), np.identity(ncvs), np.identity(ncvs), optimize = einsum_type)
    precond_cce += einsum('I,AA,II,JJ->IJA', e_cvs, np.identity(nextern), np.identity(ncvs), np.identity(ncvs), optimize = einsum_type)
    precond_cce += einsum('J,AA,II,JJ->IJA', e_cvs, np.identity(nextern), np.identity(ncvs), np.identity(ncvs), optimize = einsum_type)
    precond[ho_s_cce_aaa:ho_f_cce_aaa] = precond_cce[cvs_tril_ind[0], cvs_tril_ind[1]].reshape(-1).copy()
    precond[ho_s_cce_abb:ho_f_cce_abb] = precond_cce.reshape(-1).copy()

    if nval > 0:
        # CVE
        precond_cve =- einsum('A,AA,II,JJ->IJA', e_extern, np.identity(nextern), np.identity(ncvs), np.identity(nval), optimize = einsum_type)
        precond_cve += einsum('I,AA,II,JJ->IJA', e_cvs, np.identity(nextern), np.identity(ncvs), np.identity(nval), optimize = einsum_type)
        precond_cve += einsum('J,AA,II,JJ->IJA', e_val, np.identity(nextern), np.identity(ncvs), np.identity(nval), optimize = einsum_type)

        precond[ho_s_cve_aaa:ho_f_cve_aaa] = precond_cve.reshape(-1).copy()
        precond[ho_s_cve_abb:ho_f_cve_abb] = precond_cve.reshape(-1).copy()
        precond[ho_s_cve_bab:ho_f_cve_bab] = precond_cve.reshape(-1).copy()

    # CAE
    precond_cae =- 1/2 * einsum('A,AA,II,XY->IAXY', e_extern, np.identity(nextern), np.identity(ncvs), rdm_ca, optimize = einsum_type)
    precond_cae += 1/2 * einsum('I,AA,II,XY->IAXY', e_cvs, np.identity(nextern), np.identity(ncvs), rdm_ca, optimize = einsum_type)
    precond_cae += 1/2 * einsum('Xx,AA,II,xY->IAXY', h_aa, np.identity(nextern), np.identity(ncvs), rdm_ca, optimize = einsum_type)
    precond_cae += 1/2 * einsum('Xxyz,AA,II,Yxyz->IAXY', v_aaaa, np.identity(nextern), np.identity(ncvs), rdm_ccaa, optimize = einsum_type)

    precond_cae = einsum("IAXY,XP,YP->IPA", precond_cae, S12_cae, S12_cae, optimize = einsum_type)
    precond[ho_s_cae_aaa:ho_f_cae_aaa] = precond_cae.reshape(-1).copy()
    precond[ho_s_cae_abb:ho_f_cae_abb] = precond_cae.reshape(-1).copy()
    precond[ho_s_cae_bab:ho_f_cae_bab] = precond_cae.reshape(-1).copy()

    # CCA
    precond_cca =- einsum('XY,II,JJ->IJXY', h_aa, np.identity(ncvs), np.identity(ncvs), optimize = einsum_type)
    precond_cca += einsum('I,II,JJ,XY->IJXY', e_cvs, np.identity(ncvs), np.identity(ncvs), np.identity(ncas), optimize = einsum_type)
    precond_cca += einsum('J,II,JJ,XY->IJXY', e_cvs, np.identity(ncvs), np.identity(ncvs), np.identity(ncas), optimize = einsum_type)
    precond_cca -= 1/2 * einsum('I,II,JJ,YX->IJXY', e_cvs, np.identity(ncvs), np.identity(ncvs), rdm_ca, optimize = einsum_type)
    precond_cca -= 1/2 * einsum('J,II,JJ,YX->IJXY', e_cvs, np.identity(ncvs), np.identity(ncvs), rdm_ca, optimize = einsum_type)
    precond_cca += 1/2 * einsum('Xx,II,JJ,Yx->IJXY', h_aa, np.identity(ncvs), np.identity(ncvs), rdm_ca, optimize = einsum_type)
    precond_cca -= einsum('XxYy,II,JJ,xy->IJXY', v_aaaa, np.identity(ncvs), np.identity(ncvs), rdm_ca, optimize = einsum_type)
    precond_cca += 1/2 * einsum('XxyY,II,JJ,xy->IJXY', v_aaaa, np.identity(ncvs), np.identity(ncvs), rdm_ca, optimize = einsum_type)
    precond_cca += 1/2 * einsum('Xxyz,II,JJ,Yxyz->IJXY', v_aaaa, np.identity(ncvs), np.identity(ncvs), rdm_ccaa, optimize = einsum_type)

    precond_cca = einsum("IJXY,XP,YP->IJP", precond_cca, S12_cca, S12_cca, optimize = einsum_type)
    precond[ho_s_cca_aaa:ho_f_cca_aaa] = precond_cca[cvs_tril_ind[0], cvs_tril_ind[1]].reshape(-1).copy()
    precond[ho_s_cca_abb:ho_f_cca_abb] = precond_cca.reshape(-1).copy()

    if nval > 0:
        # CVA
        precond_cva =- einsum('XY,II,JJ->IJXY', h_aa, np.identity(ncvs), np.identity(nval), optimize = einsum_type)
        precond_cva += einsum('I,II,JJ,XY->IJXY', e_cvs, np.identity(ncvs), np.identity(nval), np.identity(ncas), optimize = einsum_type)
        precond_cva += einsum('J,II,JJ,XY->IJXY', e_val, np.identity(ncvs), np.identity(nval), np.identity(ncas), optimize = einsum_type)
        precond_cva -= 1/2 * einsum('I,II,JJ,YX->IJXY', e_cvs, np.identity(ncvs), np.identity(nval), rdm_ca, optimize = einsum_type)
        precond_cva -= 1/2 * einsum('J,II,JJ,YX->IJXY', e_val, np.identity(ncvs), np.identity(nval), rdm_ca, optimize = einsum_type)
        precond_cva += 1/2 * einsum('Xx,II,JJ,Yx->IJXY', h_aa, np.identity(ncvs), np.identity(nval), rdm_ca, optimize = einsum_type)
        precond_cva -= einsum('XxYy,II,JJ,xy->IJXY', v_aaaa, np.identity(ncvs), np.identity(nval), rdm_ca, optimize = einsum_type)
        precond_cva += 1/2 * einsum('XxyY,II,JJ,xy->IJXY', v_aaaa, np.identity(ncvs), np.identity(nval), rdm_ca, optimize = einsum_type)
        precond_cva += 1/2 * einsum('Xxyz,II,JJ,Yxyz->IJXY', v_aaaa, np.identity(ncvs), np.identity(nval), rdm_ccaa, optimize = einsum_type)

        precond_cva = einsum("IJXY,XP,YP->IJP", precond_cva, S12_cca, S12_cca, optimize = einsum_type)
        precond[ho_s_cva_aaa:ho_f_cva_aaa] = precond_cva.reshape(-1).copy()
        precond[ho_s_cva_abb:ho_f_cva_abb] = precond_cva.reshape(-1).copy()
        precond[ho_s_cva_bab:ho_f_cva_bab] = precond_cva.reshape(-1).copy()

    # Multiply by -1.0, since we are solving for -M C = -S C E
    precond *= (-1.0)

    print ("Time for computing preconditioner:                %f sec\n" % (time.time() - start_time))
    sys.stdout.flush()

    return precond

def apply_S_12(mr_adc, X, transpose = False):

    # Einsum definition from kernel
    einsum = mr_adc.interface.einsum
    einsum_type = mr_adc.interface.einsum_type

    # Dimensions
    nextern = mr_adc.nextern
    ncvs = mr_adc.ncvs
    nval = mr_adc.nval

    ho_s_c_caa = mr_adc.h_orth.s_c_caa
    ho_f_c_caa = mr_adc.h_orth.f_c_caa
    ho_s_cce = mr_adc.h_orth.s_cce
    ho_f_cce = mr_adc.h_orth.f_cce
    ho_s_cae = mr_adc.h_orth.s_cae
    ho_f_cae = mr_adc.h_orth.f_cae
    ho_s_cca = mr_adc.h_orth.s_cca
    ho_f_cca = mr_adc.h_orth.f_cca

    ho_s_cae_aaa = mr_adc.h_orth.s_cae_aaa
    ho_f_cae_aaa = mr_adc.h_orth.f_cae_aaa
    ho_s_cae_abb = mr_adc.h_orth.s_cae_abb
    ho_f_cae_abb = mr_adc.h_orth.f_cae_abb
    ho_s_cae_bab = mr_adc.h_orth.s_cae_bab
    ho_f_cae_bab = mr_adc.h_orth.f_cae_bab

    s_c = mr_adc.h0.s_c
    f_c = mr_adc.h0.f_c
    s_caa = mr_adc.h1.s_caa
    f_caa = mr_adc.h1.f_caa
    s_cce = mr_adc.h1.s_cce
    f_cce = mr_adc.h1.f_cce
    s_cae = mr_adc.h1.s_cae
    f_cae = mr_adc.h1.f_cae
    s_cca = mr_adc.h1.s_cca
    f_cca = mr_adc.h1.f_cca

    s_cae_aaa = mr_adc.h1.s_cae_aaa
    f_cae_aaa = mr_adc.h1.f_cae_aaa
    s_cae_abb = mr_adc.h1.s_cae_abb
    f_cae_abb = mr_adc.h1.f_cae_abb
    s_cae_bab = mr_adc.h1.s_cae_bab
    f_cae_bab = mr_adc.h1.f_cae_bab

    if nval > 0:
        ho_s_cve = mr_adc.h_orth.s_cve
        ho_f_cve = mr_adc.h_orth.f_cve

        ho_s_cva = mr_adc.h_orth.s_cva
        ho_f_cva = mr_adc.h_orth.f_cva

        ho_s_cva_aaa = mr_adc.h_orth.s_cva_aaa
        ho_f_cva_aaa = mr_adc.h_orth.f_cva_aaa
        ho_s_cva_abb = mr_adc.h_orth.s_cva_abb
        ho_f_cva_abb = mr_adc.h_orth.f_cva_abb
        ho_s_cva_bab = mr_adc.h_orth.s_cva_bab
        ho_f_cva_bab = mr_adc.h_orth.f_cva_bab

        s_cve = mr_adc.h1.s_cve
        f_cve = mr_adc.h1.f_cve

        s_cva = mr_adc.h1.s_cva
        f_cva = mr_adc.h1.f_cva

        s_cva_aaa = mr_adc.h1.s_cva_aaa
        f_cva_aaa = mr_adc.h1.f_cva_aaa
        s_cva_abb = mr_adc.h1.s_cva_abb
        f_cva_abb = mr_adc.h1.f_cva_abb
        s_cva_bab = mr_adc.h1.s_cva_bab
        f_cva_bab = mr_adc.h1.f_cva_bab

    # Overlap matrices
    S12_c_caa = mr_adc.S12.c_caa
    S12_cae = mr_adc.S12.cae
    S12_cca = mr_adc.S12.cca

    Xt = None

    if transpose:
        if (X.shape[0] != (mr_adc.h0.dim + mr_adc.h1.dim)):
            raise Exception("Dimensions do not match when applying S_12 transpose")

        Xt = np.zeros(mr_adc.h_orth.dim)

        # C and CAA -> C_CAA
        ncas = mr_adc.ncas
        n_caa = ncvs * ncas * ncas
        n_aa = ncas * ncas
        s_caa_aaa = s_caa
        f_caa_aaa = s_caa_aaa + n_caa
        s_caa_abb = f_caa_aaa
        f_caa_abb = s_caa_abb + n_caa
        s_caa_bab = f_caa_abb
        f_caa_bab = s_caa_bab + n_caa

        temp = np.zeros((ncvs, S12_c_caa.shape[0]))
        temp[:,0] = X[s_c:f_c].copy()
        temp[:,1:n_aa+1] = X[s_caa_aaa:f_caa_aaa].reshape(ncvs, -1).copy()
        temp[:,n_aa+1:(2*n_aa)+1] = X[s_caa_abb:f_caa_abb].reshape(ncvs, -1).copy()
        temp[:,(2*n_aa)+1:] = X[s_caa_bab:f_caa_bab].reshape(ncvs, -1).copy()
        Xt[ho_s_c_caa:ho_f_c_caa] = np.dot(temp, S12_c_caa).reshape(-1).copy()

        # CCE
        Xt[ho_s_cce:ho_f_cce] = X[s_cce:f_cce].copy()

        if nval > 0:
            # CVE
            Xt[ho_s_cve:ho_f_cve] = X[s_cve:f_cve].copy()

        # CAE
        temp = X[s_cae_aaa:f_cae_aaa].reshape(ncvs, S12_cae.shape[0], nextern).copy()
        Xt[ho_s_cae_aaa:ho_f_cae_aaa] = einsum("IXA,XP->IPA", temp, S12_cae).reshape(-1).copy()

        temp = X[s_cae_abb:f_cae_abb].reshape(ncvs, S12_cae.shape[0], nextern).copy()
        Xt[ho_s_cae_abb:ho_f_cae_abb] = einsum("IXA,XP->IPA", temp, S12_cae).reshape(-1).copy()

        temp = X[s_cae_bab:f_cae_bab].reshape(ncvs, S12_cae.shape[0], nextern).copy()
        Xt[ho_s_cae_bab:ho_f_cae_bab] = einsum("IXA,XP->IPA", temp, S12_cae).reshape(-1).copy()

        # CCA
        temp = X[s_cca:f_cca].reshape(-1, S12_cca.shape[0]).copy()
        Xt[ho_s_cca:ho_f_cca] = einsum("IX,XP->IP", temp, S12_cca).reshape(-1).copy()

        if nval > 0:
            # CVA
            temp = X[s_cva:f_cva].reshape(-1, S12_cca.shape[0]).copy()
            Xt[ho_s_cva:ho_f_cva] = einsum("IX,XP->IP", temp, S12_cca).reshape(-1).copy()

    else:
        if (X.shape[0] != (mr_adc.h_orth.dim)):
            raise Exception("Dimensions do not match when applying S_12")

        Xt = np.zeros(mr_adc.h0.dim + mr_adc.h1.dim)

        # C_CAA -> C and CAA
        temp = X[ho_s_c_caa:ho_f_c_caa].reshape(ncvs, -1).copy()
        temp = np.dot(temp, S12_c_caa.T)
        Xt[s_c:f_c] = temp[:,0].copy()

        ncas = mr_adc.ncas
        n_caa = ncvs * ncas * ncas
        n_aa = ncas * ncas
        s_caa_aaa = s_caa
        f_caa_aaa = s_caa_aaa + n_caa
        s_caa_abb = f_caa_aaa
        f_caa_abb = s_caa_abb + n_caa
        s_caa_bab = f_caa_abb
        f_caa_bab = s_caa_bab + n_caa

        Xt[s_caa_aaa:f_caa_aaa] = temp[:,1:(n_aa)+1].reshape(-1).copy()
        Xt[s_caa_abb:f_caa_abb] = temp[:,(n_aa)+1:(2*n_aa)+1].reshape(-1).copy()
        Xt[s_caa_bab:f_caa_bab] = temp[:,(2*n_aa)+1:].reshape(-1).copy()

        # CCE
        Xt[s_cce:f_cce] = X[ho_s_cce:ho_f_cce].copy()

        if nval > 0:
            # CVE
            Xt[s_cve:f_cve] = X[ho_s_cve:ho_f_cve].copy()

        # CAE
        temp = X[ho_s_cae_aaa:ho_f_cae_aaa].reshape(ncvs, S12_cae.shape[1], nextern).copy()
        Xt[s_cae_aaa:f_cae_aaa] = einsum("IPA,XP->IXA", temp, S12_cae).reshape(-1).copy()

        temp = X[ho_s_cae_abb:ho_f_cae_abb].reshape(ncvs, S12_cae.shape[1], nextern).copy()
        Xt[s_cae_abb:f_cae_abb] = einsum("IPA,XP->IXA", temp, S12_cae).reshape(-1).copy()

        temp = X[ho_s_cae_bab:ho_f_cae_bab].reshape(ncvs, S12_cae.shape[1], nextern).copy()
        Xt[s_cae_bab:f_cae_bab] = einsum("IPA,XP->IXA", temp, S12_cae).reshape(-1).copy()

        # CCA
        temp = X[ho_s_cca:ho_f_cca].reshape(-1, S12_cca.shape[1]).copy()
        Xt[s_cca:f_cca] = einsum("IP,XP->IX", temp, S12_cca).reshape(-1).copy()

        if nval > 0:
            # CVA
            temp = X[ho_s_cva:ho_f_cva].reshape(-1, S12_cca.shape[1]).copy()
            Xt[s_cva:f_cva] = einsum("IP,XP->IX", temp, S12_cca).reshape(-1).copy()

    return Xt

def compute_sigma_vector(mr_adc, M_00, M_01, M_11, Xt):

    # Einsum definition from kernel
    einsum = mr_adc.interface.einsum
    einsum_type = mr_adc.interface.einsum_type

    # Variables from kernel
    ncvs = mr_adc.ncvs
    nval = mr_adc.nval
    ncas = mr_adc.ncas
    nextern = mr_adc.nextern

    e_cvs = mr_adc.mo_energy.x
    e_core = mr_adc.mo_energy.c
    e_extern = mr_adc.mo_energy.e

    if nval > 0:
        e_val = mr_adc.mo_energy.v

    # One-electron integrals
    h_aa = mr_adc.h1eff.aa

    # Two-electrons integrals
    v_aaaa = mr_adc.v2e.aaaa

    # Reduced Density Matrices
    rdm_ca = mr_adc.rdm.ca
    rdm_ccaa = mr_adc.rdm.ccaa
    rdm_cccaaa = mr_adc.rdm.cccaaa

    # Dimensions
    s_c = mr_adc.h0.s_c
    f_c = mr_adc.h0.f_c
    s_caa = mr_adc.h1.s_caa
    f_caa = mr_adc.h1.f_caa
    s_cce = mr_adc.h1.s_cce
    f_cce = mr_adc.h1.f_cce
    s_cae = mr_adc.h1.s_cae
    f_cae = mr_adc.h1.f_cae
    s_cca = mr_adc.h1.s_cca
    f_cca = mr_adc.h1.f_cca

    s_caa_aaa = mr_adc.h1.s_caa_aaa
    f_caa_aaa = mr_adc.h1.f_caa_aaa
    s_caa_abb = mr_adc.h1.s_caa_abb
    f_caa_abb = mr_adc.h1.f_caa_abb
    s_caa_bab = mr_adc.h1.s_caa_bab
    f_caa_bab = mr_adc.h1.f_caa_bab

    s_cce_aaa = mr_adc.h1.s_cce_aaa
    f_cce_aaa = mr_adc.h1.f_cce_aaa
    s_cce_abb = mr_adc.h1.s_cce_abb
    f_cce_abb = mr_adc.h1.f_cce_abb

    s_cae_aaa = mr_adc.h1.s_cae_aaa
    f_cae_aaa = mr_adc.h1.f_cae_aaa
    s_cae_abb = mr_adc.h1.s_cae_abb
    f_cae_abb = mr_adc.h1.f_cae_abb
    s_cae_bab = mr_adc.h1.s_cae_bab
    f_cae_bab = mr_adc.h1.f_cae_bab

    s_cca_aaa = mr_adc.h1.s_cca_aaa
    f_cca_aaa = mr_adc.h1.f_cca_aaa
    s_cca_abb = mr_adc.h1.s_cca_abb
    f_cca_abb = mr_adc.h1.f_cca_abb

    if nval > 0:
        s_cve = mr_adc.h1.s_cve
        f_cve = mr_adc.h1.f_cve

        s_cva = mr_adc.h1.s_cva
        f_cva = mr_adc.h1.f_cva

        s_cve_aaa = mr_adc.h1.s_cve_aaa
        f_cve_aaa = mr_adc.h1.f_cve_aaa
        s_cve_abb = mr_adc.h1.s_cve_abb
        f_cve_abb = mr_adc.h1.f_cve_abb
        s_cve_bab = mr_adc.h1.s_cve_bab
        f_cve_bab = mr_adc.h1.f_cve_bab

        s_cva_aaa = mr_adc.h1.s_cva_aaa
        f_cva_aaa = mr_adc.h1.f_cva_aaa
        s_cva_abb = mr_adc.h1.s_cva_abb
        f_cva_abb = mr_adc.h1.f_cva_abb
        s_cva_bab = mr_adc.h1.s_cva_bab
        f_cva_bab = mr_adc.h1.f_cva_bab

    cvs_tril_ind = np.tril_indices(ncvs, k=-1)

    # (CASCI + C) -> (CASCI + C)
    sigma = np.zeros_like(Xt)

    # h0-h0 contributions
    sigma[:mr_adc.h0.dim] = np.dot(M_00, Xt[:mr_adc.h0.dim])

    # h0-h1 and h1-h0 contributions
    if nval > 0:
        M_C_CAA, M_C_CCE, M_C_CVE, M_C_CAE, M_C_CCA, M_C_CVA = M_01
    else:
        M_C_CAA, M_C_CCE, M_C_CAE, M_C_CCA = M_01

    # C <-> CAA
    sigma[s_c:f_c] += np.dot(M_C_CAA, Xt[s_caa:f_caa])
    sigma[s_caa:f_caa] += np.dot(M_C_CAA.T, Xt[s_c:f_c])

    # C <-> CCE
    sigma[s_c:f_c] += np.dot(M_C_CCE, Xt[s_cce:f_cce])
    sigma[s_cce:f_cce] += np.dot(M_C_CCE.T, Xt[s_c:f_c])

    # C <-> CVE
    if nval > 0:
        sigma[s_c:f_c] += np.dot(M_C_CVE, Xt[s_cve:f_cve])
        sigma[s_cve:f_cve] += np.dot(M_C_CVE.T, Xt[s_c:f_c])

    # C <-> CAE
    sigma[s_c:f_c] += np.dot(M_C_CAE, Xt[s_cae:f_cae])
    sigma[s_cae:f_cae] += np.dot(M_C_CAE.T, Xt[s_c:f_c])

    # C <-> CCA
    sigma[s_c:f_c] += np.dot(M_C_CCA, Xt[s_cca:f_cca])
    sigma[s_cca:f_cca] += np.dot(M_C_CCA.T, Xt[s_c:f_c])

    # C <-> CVA
    if nval > 0:
        sigma[s_c:f_c] += np.dot(M_C_CVA, Xt[s_cva:f_cva])
        sigma[s_cva:f_cva] += np.dot(M_C_CVA.T, Xt[s_c:f_c])

    # h1-h1 contributions
    # CAA <- CAA
    n_caa = ncas * ncas * ncvs
    s_caa_aaa = mr_adc.h1.s_caa
    f_caa_aaa = s_caa_aaa + n_caa
    s_caa_abb = f_caa_aaa
    f_caa_abb = s_caa_abb + n_caa
    s_caa_bab = f_caa_abb
    f_caa_bab = s_caa_bab + n_caa

    X_aaa = Xt[s_caa_aaa:f_caa_aaa].reshape(ncvs, ncas, ncas).copy()
    X_abb = Xt[s_caa_abb:f_caa_abb].reshape(ncvs, ncas, ncas).copy()
    X_bab = Xt[s_caa_bab:f_caa_bab].reshape(ncvs, ncas, ncas).copy()

    sigma_caa_aaa  = 1/2 * einsum('KxZ,K,xW->KWZ', X_aaa, e_cvs, rdm_ca, optimize = einsum_type)
    sigma_caa_aaa += 1/6 * einsum('Kxy,K,WyZx->KWZ', X_aaa, e_cvs, rdm_ccaa, optimize = einsum_type)
    sigma_caa_aaa -= 1/6 * einsum('Kxy,K,WyxZ->KWZ', X_aaa, e_cvs, rdm_ccaa, optimize = einsum_type)
    sigma_caa_aaa += 1/2 * einsum('KxZ,xy,yW->KWZ', X_aaa, h_aa, rdm_ca, optimize = einsum_type)
    sigma_caa_aaa -= 1/2 * einsum('Kxy,Zy,xW->KWZ', X_aaa, h_aa, rdm_ca, optimize = einsum_type)
    sigma_caa_aaa += 1/6 * einsum('Kxy,xz,WyZz->KWZ', X_aaa, h_aa, rdm_ccaa, optimize = einsum_type)
    sigma_caa_aaa -= 1/6 * einsum('Kxy,xz,WyzZ->KWZ', X_aaa, h_aa, rdm_ccaa, optimize = einsum_type)
    sigma_caa_aaa -= 1/6 * einsum('Kxy,yz,WzZx->KWZ', X_aaa, h_aa, rdm_ccaa, optimize = einsum_type)
    sigma_caa_aaa += 1/6 * einsum('Kxy,yz,WzxZ->KWZ', X_aaa, h_aa, rdm_ccaa, optimize = einsum_type)
    sigma_caa_aaa += 1/2 * einsum('KxZ,xyzw,Wyzw->KWZ', X_aaa, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_caa_aaa += 1/6 * einsum('Kxy,Zxzw,Wyzw->KWZ', X_aaa, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_caa_aaa -= 1/6 * einsum('Kxy,Zxzw,Wywz->KWZ', X_aaa, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_caa_aaa -= 1/2 * einsum('Kxy,Zzyw,Wzxw->KWZ', X_aaa, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_caa_aaa += 1/6 * einsum('Kxy,Zzwy,Wzxw->KWZ', X_aaa, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_caa_aaa -= 1/6 * einsum('Kxy,Zzwy,Wzwx->KWZ', X_aaa, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_caa_aaa += 1/6 * einsum('Kxy,xzwu,ZwuWyz->KWZ', X_aaa, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_caa_aaa -= 1/6 * einsum('Kxy,xzwu,ZwuyWz->KWZ', X_aaa, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_caa_aaa -= 1/6 * einsum('Kxy,yzwu,ZxzWwu->KWZ', X_aaa, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_caa_aaa += 1/6 * einsum('Kxy,yzwu,ZxzwWu->KWZ', X_aaa, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_caa_aaa += 1/3 * einsum('Kxy,K,WyZx->KWZ', X_abb, e_cvs, rdm_ccaa, optimize = einsum_type)
    sigma_caa_aaa += 1/6 * einsum('Kxy,K,WyxZ->KWZ', X_abb, e_cvs, rdm_ccaa, optimize = einsum_type)
    sigma_caa_aaa += 1/3 * einsum('Kxy,xz,WyZz->KWZ', X_abb, h_aa, rdm_ccaa, optimize = einsum_type)
    sigma_caa_aaa += 1/6 * einsum('Kxy,xz,WyzZ->KWZ', X_abb, h_aa, rdm_ccaa, optimize = einsum_type)
    sigma_caa_aaa -= 1/3 * einsum('Kxy,yz,WzZx->KWZ', X_abb, h_aa, rdm_ccaa, optimize = einsum_type)
    sigma_caa_aaa -= 1/6 * einsum('Kxy,yz,WzxZ->KWZ', X_abb, h_aa, rdm_ccaa, optimize = einsum_type)
    sigma_caa_aaa += 1/3 * einsum('Kxy,Zxzw,Wyzw->KWZ', X_abb, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_caa_aaa += 1/6 * einsum('Kxy,Zxzw,Wywz->KWZ', X_abb, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_caa_aaa -= 1/6 * einsum('Kxy,Zzwy,Wzxw->KWZ', X_abb, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_caa_aaa -= 1/3 * einsum('Kxy,Zzwy,Wzwx->KWZ', X_abb, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_caa_aaa += 1/4 * einsum('Kxy,xzwu,ZwuWyz->KWZ', X_abb, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_caa_aaa -= 1/12 * einsum('Kxy,xzwu,ZwuWzy->KWZ', X_abb, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_caa_aaa += 1/12 * einsum('Kxy,xzwu,ZwuyWz->KWZ', X_abb, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_caa_aaa -= 1/12 * einsum('Kxy,xzwu,ZwuyzW->KWZ', X_abb, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_caa_aaa -= 1/12 * einsum('Kxy,xzwu,ZwuzWy->KWZ', X_abb, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_caa_aaa -= 1/12 * einsum('Kxy,xzwu,ZwuzyW->KWZ', X_abb, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_caa_aaa -= 1/4 * einsum('Kxy,yzwu,ZxzWwu->KWZ', X_abb, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_caa_aaa += 1/12 * einsum('Kxy,yzwu,ZxzWuw->KWZ', X_abb, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_caa_aaa -= 1/12 * einsum('Kxy,yzwu,ZxzwWu->KWZ', X_abb, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_caa_aaa += 1/12 * einsum('Kxy,yzwu,ZxzwuW->KWZ', X_abb, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_caa_aaa += 1/12 * einsum('Kxy,yzwu,ZxzuWw->KWZ', X_abb, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_caa_aaa += 1/12 * einsum('Kxy,yzwu,ZxzuwW->KWZ', X_abb, v_aaaa, rdm_cccaaa, optimize = einsum_type)

    sigma_caa_abb  = 1/3 * einsum('Kxy,K,WyZx->KWZ', X_aaa, e_cvs, rdm_ccaa, optimize = einsum_type)
    sigma_caa_abb += 1/6 * einsum('Kxy,K,WyxZ->KWZ', X_aaa, e_cvs, rdm_ccaa, optimize = einsum_type)
    sigma_caa_abb += 1/3 * einsum('Kxy,xz,WyZz->KWZ', X_aaa, h_aa, rdm_ccaa, optimize = einsum_type)
    sigma_caa_abb += 1/6 * einsum('Kxy,xz,WyzZ->KWZ', X_aaa, h_aa, rdm_ccaa, optimize = einsum_type)
    sigma_caa_abb -= 1/3 * einsum('Kxy,yz,WzZx->KWZ', X_aaa, h_aa, rdm_ccaa, optimize = einsum_type)
    sigma_caa_abb -= 1/6 * einsum('Kxy,yz,WzxZ->KWZ', X_aaa, h_aa, rdm_ccaa, optimize = einsum_type)
    sigma_caa_abb += 1/3 * einsum('Kxy,Zxzw,Wyzw->KWZ', X_aaa, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_caa_abb += 1/6 * einsum('Kxy,Zxzw,Wywz->KWZ', X_aaa, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_caa_abb -= 1/6 * einsum('Kxy,Zzwy,Wzxw->KWZ', X_aaa, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_caa_abb -= 1/3 * einsum('Kxy,Zzwy,Wzwx->KWZ', X_aaa, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_caa_abb += 1/4 * einsum('Kxy,xzwu,ZwuWyz->KWZ', X_aaa, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_caa_abb -= 1/12 * einsum('Kxy,xzwu,ZwuWzy->KWZ', X_aaa, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_caa_abb += 1/12 * einsum('Kxy,xzwu,ZwuyWz->KWZ', X_aaa, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_caa_abb -= 1/12 * einsum('Kxy,xzwu,ZwuyzW->KWZ', X_aaa, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_caa_abb -= 1/12 * einsum('Kxy,xzwu,ZwuzWy->KWZ', X_aaa, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_caa_abb -= 1/12 * einsum('Kxy,xzwu,ZwuzyW->KWZ', X_aaa, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_caa_abb -= 1/4 * einsum('Kxy,yzwu,ZxzWwu->KWZ', X_aaa, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_caa_abb += 1/12 * einsum('Kxy,yzwu,ZxzWuw->KWZ', X_aaa, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_caa_abb -= 1/12 * einsum('Kxy,yzwu,ZxzwWu->KWZ', X_aaa, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_caa_abb += 1/12 * einsum('Kxy,yzwu,ZxzwuW->KWZ', X_aaa, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_caa_abb += 1/12 * einsum('Kxy,yzwu,ZxzuWw->KWZ', X_aaa, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_caa_abb += 1/12 * einsum('Kxy,yzwu,ZxzuwW->KWZ', X_aaa, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_caa_abb += 1/2 * einsum('KxZ,K,xW->KWZ', X_abb, e_cvs, rdm_ca, optimize = einsum_type)
    sigma_caa_abb += 1/6 * einsum('Kxy,K,WyZx->KWZ', X_abb, e_cvs, rdm_ccaa, optimize = einsum_type)
    sigma_caa_abb -= 1/6 * einsum('Kxy,K,WyxZ->KWZ', X_abb, e_cvs, rdm_ccaa, optimize = einsum_type)
    sigma_caa_abb += 1/2 * einsum('KxZ,xy,yW->KWZ', X_abb, h_aa, rdm_ca, optimize = einsum_type)
    sigma_caa_abb -= 1/2 * einsum('Kxy,Zy,xW->KWZ', X_abb, h_aa, rdm_ca, optimize = einsum_type)
    sigma_caa_abb += 1/6 * einsum('Kxy,xz,WyZz->KWZ', X_abb, h_aa, rdm_ccaa, optimize = einsum_type)
    sigma_caa_abb -= 1/6 * einsum('Kxy,xz,WyzZ->KWZ', X_abb, h_aa, rdm_ccaa, optimize = einsum_type)
    sigma_caa_abb -= 1/6 * einsum('Kxy,yz,WzZx->KWZ', X_abb, h_aa, rdm_ccaa, optimize = einsum_type)
    sigma_caa_abb += 1/6 * einsum('Kxy,yz,WzxZ->KWZ', X_abb, h_aa, rdm_ccaa, optimize = einsum_type)
    sigma_caa_abb += 1/2 * einsum('KxZ,xyzw,Wyzw->KWZ', X_abb, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_caa_abb += 1/6 * einsum('Kxy,Zxzw,Wyzw->KWZ', X_abb, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_caa_abb -= 1/6 * einsum('Kxy,Zxzw,Wywz->KWZ', X_abb, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_caa_abb -= 1/2 * einsum('Kxy,Zzyw,Wzxw->KWZ', X_abb, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_caa_abb += 1/6 * einsum('Kxy,Zzwy,Wzxw->KWZ', X_abb, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_caa_abb -= 1/6 * einsum('Kxy,Zzwy,Wzwx->KWZ', X_abb, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_caa_abb += 1/6 * einsum('Kxy,xzwu,ZwuWyz->KWZ', X_abb, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_caa_abb -= 1/6 * einsum('Kxy,xzwu,ZwuyWz->KWZ', X_abb, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_caa_abb -= 1/6 * einsum('Kxy,yzwu,ZxzWwu->KWZ', X_abb, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_caa_abb += 1/6 * einsum('Kxy,yzwu,ZxzwWu->KWZ', X_abb, v_aaaa, rdm_cccaaa, optimize = einsum_type)

    sigma_caa_bab  = 1/2 * einsum('KxZ,K,xW->KWZ', X_bab, e_cvs, rdm_ca, optimize = einsum_type)
    sigma_caa_bab -= 1/6 * einsum('Kxy,K,WyZx->KWZ', X_bab, e_cvs, rdm_ccaa, optimize = einsum_type)
    sigma_caa_bab -= 1/3 * einsum('Kxy,K,WyxZ->KWZ', X_bab, e_cvs, rdm_ccaa, optimize = einsum_type)
    sigma_caa_bab += 1/2 * einsum('KxZ,xy,yW->KWZ', X_bab, h_aa, rdm_ca, optimize = einsum_type)
    sigma_caa_bab -= 1/2 * einsum('Kxy,Zy,xW->KWZ', X_bab, h_aa, rdm_ca, optimize = einsum_type)
    sigma_caa_bab -= 1/6 * einsum('Kxy,xz,WyZz->KWZ', X_bab, h_aa, rdm_ccaa, optimize = einsum_type)
    sigma_caa_bab -= 1/3 * einsum('Kxy,xz,WyzZ->KWZ', X_bab, h_aa, rdm_ccaa, optimize = einsum_type)
    sigma_caa_bab += 1/6 * einsum('Kxy,yz,WzZx->KWZ', X_bab, h_aa, rdm_ccaa, optimize = einsum_type)
    sigma_caa_bab += 1/3 * einsum('Kxy,yz,WzxZ->KWZ', X_bab, h_aa, rdm_ccaa, optimize = einsum_type)
    sigma_caa_bab += 1/2 * einsum('KxZ,xyzw,Wyzw->KWZ', X_bab, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_caa_bab -= 1/6 * einsum('Kxy,Zxzw,Wyzw->KWZ', X_bab, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_caa_bab -= 1/3 * einsum('Kxy,Zxzw,Wywz->KWZ', X_bab, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_caa_bab -= 1/2 * einsum('Kxy,Zzyw,Wzxw->KWZ', X_bab, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_caa_bab += 1/3 * einsum('Kxy,Zzwy,Wzxw->KWZ', X_bab, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_caa_bab += 1/6 * einsum('Kxy,Zzwy,Wzwx->KWZ', X_bab, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_caa_bab -= 1/12 * einsum('Kxy,xzwu,ZwuWyz->KWZ', X_bab, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_caa_bab += 1/12 * einsum('Kxy,xzwu,ZwuWzy->KWZ', X_bab, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_caa_bab -= 1/4 * einsum('Kxy,xzwu,ZwuyWz->KWZ', X_bab, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_caa_bab += 1/12 * einsum('Kxy,xzwu,ZwuyzW->KWZ', X_bab, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_caa_bab += 1/12 * einsum('Kxy,xzwu,ZwuzWy->KWZ', X_bab, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_caa_bab += 1/12 * einsum('Kxy,xzwu,ZwuzyW->KWZ', X_bab, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_caa_bab += 1/12 * einsum('Kxy,yzwu,ZxzWwu->KWZ', X_bab, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_caa_bab -= 1/12 * einsum('Kxy,yzwu,ZxzWuw->KWZ', X_bab, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_caa_bab += 1/4 * einsum('Kxy,yzwu,ZxzwWu->KWZ', X_bab, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_caa_bab -= 1/12 * einsum('Kxy,yzwu,ZxzwuW->KWZ', X_bab, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_caa_bab -= 1/12 * einsum('Kxy,yzwu,ZxzuWw->KWZ', X_bab, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_caa_bab -= 1/12 * einsum('Kxy,yzwu,ZxzuwW->KWZ', X_bab, v_aaaa, rdm_cccaaa, optimize = einsum_type)

    sigma[s_caa_aaa:f_caa_aaa] += sigma_caa_aaa.reshape(-1).copy()
    sigma[s_caa_abb:f_caa_abb] += sigma_caa_abb.reshape(-1).copy()
    sigma[s_caa_bab:f_caa_bab] += sigma_caa_bab.reshape(-1).copy()

    # CCE <- CCE
    X_aaa = np.zeros((ncvs, ncvs, nextern))
    X_aaa[cvs_tril_ind[0], cvs_tril_ind[1]] =  Xt[s_cce_aaa:f_cce_aaa].reshape(-1, nextern).copy()
    X_aaa[cvs_tril_ind[1], cvs_tril_ind[0]] =- Xt[s_cce_aaa:f_cce_aaa].reshape(-1, nextern).copy()

    X_abb = Xt[s_cce_abb:f_cce_abb].reshape(ncvs, ncvs, nextern).copy()
    X_bab =- X_abb.transpose(1,0,2)

    sigma_cce =- 1/2 * einsum('KLB,B->KLB', X_aaa, e_extern, optimize = einsum_type)
    sigma_cce += 1/2 * einsum('KLB,K->KLB', X_aaa, e_cvs, optimize = einsum_type)
    sigma_cce += 1/2 * einsum('KLB,L->KLB', X_aaa, e_cvs, optimize = einsum_type)
    sigma_cce += 1/2 * einsum('LKB,B->KLB', X_aaa, e_extern, optimize = einsum_type)
    sigma_cce -= 1/2 * einsum('LKB,K->KLB', X_aaa, e_cvs, optimize = einsum_type)
    sigma_cce -= 1/2 * einsum('LKB,L->KLB', X_aaa, e_cvs, optimize = einsum_type)
    sigma[s_cce_aaa:f_cce_aaa] += sigma_cce[cvs_tril_ind[0], cvs_tril_ind[1]].reshape(-1).copy()

    sigma_cce =- 1/2 * einsum('KLB,B->KLB', X_abb, e_extern, optimize = einsum_type)
    sigma_cce += 1/2 * einsum('KLB,K->KLB', X_abb, e_cvs, optimize = einsum_type)
    sigma_cce += 1/2 * einsum('KLB,L->KLB', X_abb, e_cvs, optimize = einsum_type)
    sigma_cce += 1/2 * einsum('LKB,B->KLB', X_bab, e_extern, optimize = einsum_type)
    sigma_cce -= 1/2 * einsum('LKB,K->KLB', X_bab, e_cvs, optimize = einsum_type)
    sigma_cce -= 1/2 * einsum('LKB,L->KLB', X_bab, e_cvs, optimize = einsum_type)
    sigma[s_cce_abb:f_cce_abb] += sigma_cce.reshape(-1).copy()

    if nval > 0:
        # CVE <- CVE
        X_aaa = Xt[s_cve_aaa:f_cve_aaa].reshape(ncvs, nval, nextern).copy()
        X_abb = Xt[s_cve_abb:f_cve_abb].reshape(ncvs, nval, nextern).copy()
        X_bab = Xt[s_cve_bab:f_cve_bab].reshape(ncvs, nval, nextern).copy()

        sigma_cve =- einsum('KLB,B->KLB', X_aaa, e_extern, optimize = einsum_type)
        sigma_cve += einsum('KLB,K->KLB', X_aaa, e_cvs, optimize = einsum_type)
        sigma_cve += einsum('KLB,L->KLB', X_aaa, e_val, optimize = einsum_type)
        sigma[s_cve_aaa:f_cve_aaa] += sigma_cve.reshape(-1).copy()

        sigma_cve =- einsum('KLB,B->KLB', X_abb, e_extern, optimize = einsum_type)
        sigma_cve += einsum('KLB,K->KLB', X_abb, e_cvs, optimize = einsum_type)
        sigma_cve += einsum('KLB,L->KLB', X_abb, e_val, optimize = einsum_type)
        sigma[s_cve_abb:f_cve_abb] += sigma_cve.reshape(-1).copy()

        sigma_cve =- einsum('KLB,B->KLB', X_bab, e_extern, optimize = einsum_type)
        sigma_cve += einsum('KLB,K->KLB', X_bab, e_cvs, optimize = einsum_type)
        sigma_cve += einsum('KLB,L->KLB', X_bab, e_val, optimize = einsum_type)
        sigma[s_cve_bab:f_cve_bab] += sigma_cve.reshape(-1).copy()

    # CAE <- CAE
    X_aaa = Xt[s_cae_aaa:f_cae_aaa].reshape(ncvs, ncas, nextern).copy()
    X_abb = Xt[s_cae_abb:f_cae_abb].reshape(ncvs, ncas, nextern).copy()
    X_bab = Xt[s_cae_bab:f_cae_bab].reshape(ncvs, ncas, nextern).copy()

    sigma_cae =- 1/2 * einsum('KxB,B,xZ->KZB', X_aaa, e_extern, rdm_ca, optimize = einsum_type)
    sigma_cae += 1/2 * einsum('KxB,K,xZ->KZB', X_aaa, e_cvs, rdm_ca, optimize = einsum_type)
    sigma_cae += 1/2 * einsum('KxB,xy,yZ->KZB', X_aaa, h_aa, rdm_ca, optimize = einsum_type)
    sigma_cae += 1/2 * einsum('KxB,xyzw,Zyzw->KZB', X_aaa, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma[s_cae_aaa:f_cae_aaa] += sigma_cae.reshape(-1).copy()

    sigma_cae =- 1/2 * einsum('KxB,B,xZ->KZB', X_abb, e_extern, rdm_ca, optimize = einsum_type)
    sigma_cae += 1/2 * einsum('KxB,K,xZ->KZB', X_abb, e_cvs, rdm_ca, optimize = einsum_type)
    sigma_cae += 1/2 * einsum('KxB,xy,yZ->KZB', X_abb, h_aa, rdm_ca, optimize = einsum_type)
    sigma_cae += 1/2 * einsum('KxB,xyzw,Zyzw->KZB', X_abb, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma[s_cae_abb:f_cae_abb] += sigma_cae.reshape(-1).copy()

    sigma_cae =- 1/2 * einsum('KxB,B,xZ->KZB', X_bab, e_extern, rdm_ca, optimize = einsum_type)
    sigma_cae += 1/2 * einsum('KxB,K,xZ->KZB', X_bab, e_cvs, rdm_ca, optimize = einsum_type)
    sigma_cae += 1/2 * einsum('KxB,xy,yZ->KZB', X_bab, h_aa, rdm_ca, optimize = einsum_type)
    sigma_cae += 1/2 * einsum('KxB,xyzw,Zyzw->KZB', X_bab, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma[s_cae_bab:f_cae_bab] += sigma_cae.reshape(-1).copy()

    # CCA <- CCA
    X_aaa = np.zeros((ncvs, ncvs, ncas))
    X_aaa[cvs_tril_ind[0], cvs_tril_ind[1]] =  Xt[s_cca_aaa:f_cca_aaa].reshape(-1, ncas).copy()
    X_aaa[cvs_tril_ind[1], cvs_tril_ind[0]] =- Xt[s_cca_aaa:f_cca_aaa].reshape(-1, ncas).copy()

    X_abb = Xt[s_cca_abb:f_cca_abb].reshape(ncvs, ncvs, ncas).copy()

    sigma_cca  = einsum('KLW,K->KLW', X_aaa, e_cvs, optimize = einsum_type)
    sigma_cca += einsum('KLW,L->KLW', X_aaa, e_cvs, optimize = einsum_type)
    sigma_cca -= einsum('KLx,Wx->KLW', X_aaa, h_aa, optimize = einsum_type)
    sigma_cca -= 1/2 * einsum('KLx,K,Wx->KLW', X_aaa, e_cvs, rdm_ca, optimize = einsum_type)
    sigma_cca -= 1/2 * einsum('KLx,L,Wx->KLW', X_aaa, e_cvs, rdm_ca, optimize = einsum_type)
    sigma_cca += 1/2 * einsum('KLx,xy,Wy->KLW', X_aaa, h_aa, rdm_ca, optimize = einsum_type)
    sigma_cca -= einsum('KLx,Wyxz,zy->KLW', X_aaa, v_aaaa, rdm_ca, optimize = einsum_type)
    sigma_cca += 1/2 * einsum('KLx,Wyzx,zy->KLW', X_aaa, v_aaaa, rdm_ca, optimize = einsum_type)
    sigma_cca += 1/2 * einsum('KLx,xyzw,Wyzw->KLW', X_aaa, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma[s_cca_aaa:f_cca_aaa] += sigma_cca[cvs_tril_ind[0], cvs_tril_ind[1]].reshape(-1).copy()

    sigma_cca  = einsum('KLW,K->KLW', X_abb, e_cvs, optimize = einsum_type)
    sigma_cca += einsum('KLW,L->KLW', X_abb, e_cvs, optimize = einsum_type)
    sigma_cca -= einsum('KLx,Wx->KLW', X_abb, h_aa, optimize = einsum_type)
    sigma_cca -= 1/2 * einsum('KLx,K,Wx->KLW', X_abb, e_cvs, rdm_ca, optimize = einsum_type)
    sigma_cca -= 1/2 * einsum('KLx,L,Wx->KLW', X_abb, e_cvs, rdm_ca, optimize = einsum_type)
    sigma_cca += 1/2 * einsum('KLx,xy,Wy->KLW', X_abb, h_aa, rdm_ca, optimize = einsum_type)
    sigma_cca -= einsum('KLx,Wyxz,zy->KLW', X_abb, v_aaaa, rdm_ca, optimize = einsum_type)
    sigma_cca += 1/2 * einsum('KLx,Wyzx,zy->KLW', X_abb, v_aaaa, rdm_ca, optimize = einsum_type)
    sigma_cca += 1/2 * einsum('KLx,xyzw,Wyzw->KLW', X_abb, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma[s_cca_abb:f_cca_abb] += sigma_cca.reshape(-1).copy()

    if nval > 0:
        # CVA <- CVA
        X_aaa = Xt[s_cva_aaa:f_cva_aaa].reshape(ncvs, nval, ncas).copy()
        X_abb = Xt[s_cva_abb:f_cva_abb].reshape(ncvs, nval, ncas).copy()
        X_bab = Xt[s_cva_bab:f_cva_bab].reshape(ncvs, nval, ncas).copy()

        sigma_cva  = einsum('KLW,K->KLW', X_aaa, e_cvs, optimize = einsum_type)
        sigma_cva += einsum('KLW,L->KLW', X_aaa, e_val, optimize = einsum_type)
        sigma_cva -= einsum('KLx,Wx->KLW', X_aaa, h_aa, optimize = einsum_type)
        sigma_cva -= 1/2 * einsum('KLx,K,Wx->KLW', X_aaa, e_cvs, rdm_ca, optimize = einsum_type)
        sigma_cva -= 1/2 * einsum('KLx,L,Wx->KLW', X_aaa, e_val, rdm_ca, optimize = einsum_type)
        sigma_cva += 1/2 * einsum('KLx,xy,Wy->KLW', X_aaa, h_aa, rdm_ca, optimize = einsum_type)
        sigma_cva -= einsum('KLx,Wyxz,zy->KLW', X_aaa, v_aaaa, rdm_ca, optimize = einsum_type)
        sigma_cva += 1/2 * einsum('KLx,Wyzx,zy->KLW', X_aaa, v_aaaa, rdm_ca, optimize = einsum_type)
        sigma_cva += 1/2 * einsum('KLx,xyzw,Wyzw->KLW', X_aaa, v_aaaa, rdm_ccaa, optimize = einsum_type)
        sigma[s_cva_aaa:f_cva_aaa] += sigma_cva.reshape(-1).copy()

        sigma_cva  = einsum('KLW,K->KLW', X_abb, e_cvs, optimize = einsum_type)
        sigma_cva += einsum('KLW,L->KLW', X_abb, e_val, optimize = einsum_type)
        sigma_cva -= einsum('KLx,Wx->KLW', X_abb, h_aa, optimize = einsum_type)
        sigma_cva -= 1/2 * einsum('KLx,K,Wx->KLW', X_abb, e_cvs, rdm_ca, optimize = einsum_type)
        sigma_cva -= 1/2 * einsum('KLx,L,Wx->KLW', X_abb, e_val, rdm_ca, optimize = einsum_type)
        sigma_cva += 1/2 * einsum('KLx,xy,Wy->KLW', X_abb, h_aa, rdm_ca, optimize = einsum_type)
        sigma_cva -= einsum('KLx,Wyxz,zy->KLW', X_abb, v_aaaa, rdm_ca, optimize = einsum_type)
        sigma_cva += 1/2 * einsum('KLx,Wyzx,zy->KLW', X_abb, v_aaaa, rdm_ca, optimize = einsum_type)
        sigma_cva += 1/2 * einsum('KLx,xyzw,Wyzw->KLW', X_abb, v_aaaa, rdm_ccaa, optimize = einsum_type)
        sigma[s_cva_abb:f_cva_abb] += sigma_cva.reshape(-1).copy()

        sigma_cva  = einsum('KLW,K->KLW', X_bab, e_cvs, optimize = einsum_type)
        sigma_cva += einsum('KLW,L->KLW', X_bab, e_val, optimize = einsum_type)
        sigma_cva -= einsum('KLx,Wx->KLW', X_bab, h_aa, optimize = einsum_type)
        sigma_cva -= 1/2 * einsum('KLx,K,Wx->KLW', X_bab, e_cvs, rdm_ca, optimize = einsum_type)
        sigma_cva -= 1/2 * einsum('KLx,L,Wx->KLW', X_bab, e_val, rdm_ca, optimize = einsum_type)
        sigma_cva += 1/2 * einsum('KLx,xy,Wy->KLW', X_bab, h_aa, rdm_ca, optimize = einsum_type)
        sigma_cva -= einsum('KLx,Wyxz,zy->KLW', X_bab, v_aaaa, rdm_ca, optimize = einsum_type)
        sigma_cva += 1/2 * einsum('KLx,Wyzx,zy->KLW', X_bab, v_aaaa, rdm_ca, optimize = einsum_type)
        sigma_cva += 1/2 * einsum('KLx,xyzw,Wyzw->KLW', X_bab, v_aaaa, rdm_ccaa, optimize = einsum_type)
        sigma[s_cva_bab:f_cva_bab] += sigma_cva.reshape(-1).copy()

    return sigma

# Working
def compute_excitation_manifolds_dev(mr_adc):

    # MR-ADC(0) and MR-ADC(1)
    mr_adc.h0.n_c = mr_adc.ncvs
    mr_adc.h0.dim = mr_adc.h0.n_c # Total dimension of h0

    mr_adc.h0.s_c = 0
    mr_adc.h0.f_c = mr_adc.h0.s_c + mr_adc.h0.n_c

    print("Dimension of h0 excitation manifold:                       %d" % mr_adc.h0.dim)

    # MR-ADC(2)
    mr_adc.h1.dim = 0
    mr_adc.h_orth.dim = mr_adc.h0.dim

    if mr_adc.method in ("mr-adc(2)", "mr-adc(2)-x"):
        # mr_adc.h1.n_caa = 2 * mr_adc.ncas * mr_adc.ncas * mr_adc.ncvs
        mr_adc.h1.n_caa = 0
        # mr_adc.h1.n_cce = mr_adc.nextern * mr_adc.ncvs * mr_adc.ncvs
        mr_adc.h1.n_cce = 0
        mr_adc.h1.n_cae = mr_adc.nextern * mr_adc.ncas * mr_adc.ncvs
        # mr_adc.h1.n_cae = 0
        # mr_adc.h1.n_ace = mr_adc.nextern * mr_adc.ncas * mr_adc.ncvs
        mr_adc.h1.n_ace = 0
        # mr_adc.h1.n_cca = mr_adc.ncas * mr_adc.ncvs * mr_adc.ncvs
        mr_adc.h1.n_cca = 0
        if mr_adc.nval > 0:
            # mr_adc.h1.n_cve = mr_adc.nextern * mr_adc.ncvs * mr_adc.nval
            mr_adc.h1.n_cve = 0
            # mr_adc.h1.n_vce = mr_adc.nextern * mr_adc.ncvs * mr_adc.nval
            mr_adc.h1.n_vce = 0
            mr_adc.h1.n_cva = 0
            # mr_adc.h1.n_cva = mr_adc.ncas * mr_adc.ncvs * mr_adc.nval
            mr_adc.h1.n_vca = 0
            # mr_adc.h1.n_vca = mr_adc.ncas * mr_adc.ncvs * mr_adc.nval
            mr_adc.h1.dim = (mr_adc.h1.n_caa + mr_adc.h1.n_cce + mr_adc.h1.n_cve + mr_adc.h1.n_vce + 
                             mr_adc.h1.n_cae + mr_adc.h1.n_ace + mr_adc.h1.n_cca + mr_adc.h1.n_cva + mr_adc.h1.n_vca)
        else:
            mr_adc.h1.dim = mr_adc.h1.n_caa + mr_adc.h1.n_cce + mr_adc.h1.n_cae + mr_adc.h1.n_cae + mr_adc.h1.n_cca

        if mr_adc.nval > 0:
            mr_adc.h1.s_caa = mr_adc.h0.f_c
            mr_adc.h1.f_caa = mr_adc.h1.s_caa + mr_adc.h1.n_caa
            mr_adc.h1.s_cce = mr_adc.h1.f_caa
            mr_adc.h1.f_cce = mr_adc.h1.s_cce + mr_adc.h1.n_cce
            mr_adc.h1.s_cve = mr_adc.h1.f_cce
            mr_adc.h1.f_cve = mr_adc.h1.s_cve + mr_adc.h1.n_cve
            mr_adc.h1.s_vce = mr_adc.h1.f_cve
            mr_adc.h1.f_vce = mr_adc.h1.s_vce + mr_adc.h1.n_vce
            mr_adc.h1.s_cae = mr_adc.h1.f_vce
            mr_adc.h1.f_cae = mr_adc.h1.s_cae + mr_adc.h1.n_cae
            mr_adc.h1.s_ace = mr_adc.h1.f_cae
            mr_adc.h1.f_ace = mr_adc.h1.s_ace + mr_adc.h1.n_ace
            mr_adc.h1.s_cca = mr_adc.h1.f_ace
            mr_adc.h1.f_cca = mr_adc.h1.s_cca + mr_adc.h1.n_cca
            mr_adc.h1.s_cva = mr_adc.h1.f_cca
            mr_adc.h1.f_cva = mr_adc.h1.s_cva + mr_adc.h1.n_cva
            mr_adc.h1.s_vca = mr_adc.h1.f_cva
            mr_adc.h1.f_vca = mr_adc.h1.s_vca + mr_adc.h1.n_vca
        else:
            mr_adc.h1.s_caa = mr_adc.h0.f_c
            mr_adc.h1.f_caa = mr_adc.h1.s_caa + mr_adc.h1.n_caa
            mr_adc.h1.s_cce = mr_adc.h1.f_caa
            mr_adc.h1.f_cce = mr_adc.h1.s_cce + mr_adc.h1.n_cce
            mr_adc.h1.s_cae = mr_adc.h1.f_cce
            mr_adc.h1.f_cae = mr_adc.h1.s_cae + mr_adc.h1.n_cae
            mr_adc.h1.s_ace = mr_adc.h1.f_cae
            mr_adc.h1.f_ace = mr_adc.h1.s_ace + mr_adc.h1.n_ace
            mr_adc.h1.s_cca = mr_adc.h1.f_ace
            mr_adc.h1.f_cca = mr_adc.h1.s_cca + mr_adc.h1.n_cca

        print("Dimension of h1 excitation manifold:                       %d" % mr_adc.h1.dim)

        # Overlap for c - caa
        mr_adc.S12.c_caa = mr_adc_overlap.compute_S12_0p_projector(mr_adc)
        mr_adc.S12.cae = mr_adc_overlap.compute_S12_m1(mr_adc)
        mr_adc.S12.cca = mr_adc_overlap.compute_S12_p1(mr_adc)

        # Determine dimensions of orthogonalized excitation spaces
        mr_adc.h_orth.n_c = mr_adc.ncvs
        # mr_adc.h_orth.n_c_caa = mr_adc.ncvs * mr_adc.S12.c_caa.shape[1]
        mr_adc.h_orth.n_c_caa = 0
        mr_adc.h_orth.n_cce = 0
        # mr_adc.h_orth.n_cce = mr_adc.h1.n_cce
        mr_adc.h_orth.n_cae = mr_adc.nextern * mr_adc.ncvs * mr_adc.S12.cae.shape[1]
        # mr_adc.h_orth.n_cae = 0
        # mr_adc.h_orth.n_ace = mr_adc.nextern * mr_adc.ncvs * mr_adc.S12.cae.shape[1]
        mr_adc.h_orth.n_ace = 0
        # mr_adc.h_orth.n_cca = mr_adc.S12.cca.shape[1] * mr_adc.ncvs * mr_adc.ncvs
        mr_adc.h_orth.n_cca = 0
        if mr_adc.nval > 0:
            mr_adc.h_orth.n_cve = 0
            mr_adc.h_orth.n_vce = 0
            # mr_adc.h_orth.n_cve = mr_adc.h1.n_cve
            # mr_adc.h_orth.n_vce = mr_adc.h1.n_vce
            mr_adc.h_orth.n_cva = 0
            mr_adc.h_orth.n_vca = 0
            # mr_adc.h_orth.n_cva = mr_adc.S12.cca.shape[1] * mr_adc.ncvs * mr_adc.nval
            # mr_adc.h_orth.n_vca = mr_adc.S12.cca.shape[1] * mr_adc.ncvs * mr_adc.nval
            # mr_adc.h_orth.dim = (mr_adc.h_orth.n_c_caa + mr_adc.h_orth.n_cce + mr_adc.h_orth.n_cve +
            #                      mr_adc.h_orth.n_cae + mr_adc.h_orth.n_cca + mr_adc.h_orth.n_cva)
            mr_adc.h_orth.dim = (mr_adc.h_orth.n_c + mr_adc.h_orth.n_c_caa + mr_adc.h_orth.n_cce + mr_adc.h_orth.n_cve + mr_adc.h_orth.n_vce +
                                 mr_adc.h_orth.n_cae + mr_adc.h_orth.n_ace + mr_adc.h_orth.n_cca + mr_adc.h_orth.n_cva + mr_adc.h_orth.n_vca)
        else:
            # mr_adc.h_orth.dim = mr_adc.h_orth.n_c_caa + mr_adc.h_orth.n_cce + mr_adc.h_orth.n_cae + mr_adc.h_orth.n_cca
            mr_adc.h_orth.dim = mr_adc.h_orth.n_c + mr_adc.h_orth.n_c_caa + mr_adc.h_orth.n_cce + mr_adc.h_orth.n_cae + mr_adc.h_orth.n_ace + mr_adc.h_orth.n_cca

        if mr_adc.nval > 0:
            mr_adc.h_orth.s_c = 0
            mr_adc.h_orth.f_c = mr_adc.h_orth.n_c
            # mr_adc.h_orth.s_c_caa = 0
            # mr_adc.h_orth.f_c_caa = mr_adc.h_orth.n_c_caa
            mr_adc.h_orth.s_c_caa = mr_adc.h_orth.f_c
            mr_adc.h_orth.f_c_caa = mr_adc.h_orth.s_c_caa + mr_adc.h_orth.n_c_caa
            # mr_adc.h_orth.s_cce = mr_adc.h_orth.f_c_caa
            mr_adc.h_orth.s_cce = mr_adc.h_orth.f_c_caa
            mr_adc.h_orth.f_cce = mr_adc.h_orth.s_cce + mr_adc.h_orth.n_cce
            mr_adc.h_orth.s_cve = mr_adc.h_orth.f_cce
            mr_adc.h_orth.f_cve = mr_adc.h_orth.s_cve + mr_adc.h_orth.n_cve
            mr_adc.h_orth.s_vce = mr_adc.h_orth.f_cve
            mr_adc.h_orth.f_vce = mr_adc.h_orth.s_vce + mr_adc.h_orth.n_vce
            mr_adc.h_orth.s_cae = mr_adc.h_orth.f_vce
            mr_adc.h_orth.f_cae = mr_adc.h_orth.s_cae + mr_adc.h_orth.n_cae
            mr_adc.h_orth.s_ace = mr_adc.h_orth.f_cae
            mr_adc.h_orth.f_ace = mr_adc.h_orth.s_ace + mr_adc.h_orth.n_ace
            mr_adc.h_orth.s_cca = mr_adc.h_orth.f_ace
            mr_adc.h_orth.f_cca = mr_adc.h_orth.s_cca + mr_adc.h_orth.n_cca
            mr_adc.h_orth.s_cva = mr_adc.h_orth.f_cca
            mr_adc.h_orth.f_cva = mr_adc.h_orth.s_cva + mr_adc.h_orth.n_cva
            mr_adc.h_orth.s_vca = mr_adc.h_orth.f_cva
            mr_adc.h_orth.f_vca = mr_adc.h_orth.s_vca + mr_adc.h_orth.n_vca
        else:
            mr_adc.h_orth.s_c = 0
            mr_adc.h_orth.f_c = mr_adc.h_orth.n_c
            # mr_adc.h_orth.s_c_caa = 0
            # mr_adc.h_orth.f_c_caa = mr_adc.h_orth.n_c_caa
            mr_adc.h_orth.s_cce = mr_adc.h_orth.f_c_caa
            mr_adc.h_orth.f_cce = mr_adc.h_orth.s_cce + mr_adc.h_orth.n_cce
            mr_adc.h_orth.s_cae = mr_adc.h_orth.f_cce
            mr_adc.h_orth.f_cae = mr_adc.h_orth.s_cae + mr_adc.h_orth.n_cae
            mr_adc.h_orth.s_ace = mr_adc.h_orth.f_ace
            mr_adc.h_orth.f_ace = mr_adc.h_orth.s_ace + mr_adc.h_orth.n_ace
            mr_adc.h_orth.s_cca = mr_adc.h_orth.f_ace
            mr_adc.h_orth.f_cca = mr_adc.h_orth.s_cca + mr_adc.h_orth.n_cca

    print("Total dimension of the excitation manifold:                %d" % (mr_adc.h0.dim + mr_adc.h1.dim))
    print("Dimension of the orthogonalized excitation manifold:       %d\n" % (mr_adc.h_orth.dim))
    sys.stdout.flush()

    if (mr_adc.h_orth.dim < mr_adc.nroots):
        mr_adc.nroots = mr_adc.h_orth.dim

    return mr_adc

def compute_M_01_dev0_old(mr_adc):

    start_time = time.time()

    print ("Computing M(h0-h1) blocks...")
    sys.stdout.flush()

    # Einsum definition from kernel
    einsum = mr_adc.interface.einsum
    einsum_type = mr_adc.interface.einsum_type

    # Dimensions
    ncore = mr_adc.ncore
    ncas = mr_adc.ncas
    nocc = mr_adc.nocc
    nextern = mr_adc.nextern

    ncvs = mr_adc.ncvs
    nval = mr_adc.nval

    n_caa = mr_adc.h1.n_caa
    n_cce = mr_adc.h1.n_cce
    n_cae = mr_adc.h1.n_cae
    n_cca = mr_adc.h1.n_cca
    if nval > 0:
        n_cve = mr_adc.h1.n_cve
        n_cva = mr_adc.h1.n_cva

    # cvs_tril_ind = np.tril_indices(ncvs, k=-1)

    # MOs Energy
    e_cvs = mr_adc.mo_energy.x
    e_val = mr_adc.mo_energy.v
    e_core = mr_adc.mo_energy.c
    e_extern = mr_adc.mo_energy.e

    # Amplitudes
    t1_ce = mr_adc.t1.ce
    t1_ca = mr_adc.t1.ca
    t1_ae = mr_adc.t1.ae
    t1_caea = mr_adc.t1.caea
    t1_caae = mr_adc.t1.caae
    t1_caaa = mr_adc.t1.caaa
    t1_aaea = mr_adc.t1.aaea

    t1_xe = mr_adc.t1.xe
    t1_xaea = mr_adc.t1.xaea
    t1_xaae = mr_adc.t1.xaae

    t1_ve = mr_adc.t1.ve
    t1_vaea = mr_adc.t1.vaea
    t1_vaae = mr_adc.t1.vaae

    # One-electron integrals
    h_aa = mr_adc.h1eff.aa

    h_xe = mr_adc.h1eff.xe

    h_ve = mr_adc.h1eff.ve

    # Two-electrons integrals
    v_aaaa = mr_adc.v2e.aaaa

    v_xaxa = mr_adc.v2e.xaxa
    v_xaax = mr_adc.v2e.xaax

    v_vxxe = mr_adc.v2e.vxxe
    v_xvxe = mr_adc.v2e.xvxe

    v_xaea = mr_adc.v2e.xaea
    v_xaae = mr_adc.v2e.xaae
    v_xxxe = mr_adc.v2e.xxxe

    v_vaea = mr_adc.v2e.vaea
    v_vaae = mr_adc.v2e.vaae

    # Reduced Density Matrices
    rdm_ca = mr_adc.rdm.ca
    rdm_ccaa = mr_adc.rdm.ccaa
    rdm_cccaaa = mr_adc.rdm.cccaaa

    # C - CAA
    # Oth-order
    # M_C_CAA_a_abb  = 1/2 * einsum('J,IJ,WZ->IJWZ', e_cvs, np.identity(ncvs), rdm_ca, optimize = einsum_type)
    # M_C_CAA_a_abb += 1/2 * einsum('Wx,IJ,xZ->IJWZ', h_aa, np.identity(ncvs), rdm_ca, optimize = einsum_type)
    # M_C_CAA_a_abb -= 1/2 * einsum('Zx,IJ,Wx->IJWZ', h_aa, np.identity(ncvs), rdm_ca, optimize = einsum_type)
    # M_C_CAA_a_abb += 1/2 * einsum('IJ,Wxyz,Zxyz->IJWZ', np.identity(ncvs), v_aaaa, rdm_ccaa, optimize = einsum_type)
    # M_C_CAA_a_abb -= 1/2 * einsum('IJ,Zxyz,Wxyz->IJWZ', np.identity(ncvs), v_aaaa, rdm_ccaa, optimize = einsum_type)

    # 1st-order
    # M_C_CAA_a_abb += 1/2 * einsum('IZJx,Wx->IJWZ', v_xaxa, rdm_ca, optimize = einsum_type)
    # M_C_CAA_a_abb += 1/2 * einsum('IxJy,WxZy->IJWZ', v_xaxa, rdm_ccaa, optimize = einsum_type)
    # M_C_CAA_a_abb -= 1/3 * einsum('IxyJ,WxZy->IJWZ', v_xaax, rdm_ccaa, optimize = einsum_type)
    # M_C_CAA_a_abb -= 1/6 * einsum('IxyJ,WxyZ->IJWZ', v_xaax, rdm_ccaa, optimize = einsum_type)
    # M_C_CAA_a_abb -= 1/2 * einsum('IxJy,yx,WZ->IJWZ', v_xaxa, rdm_ca, rdm_ca, optimize = einsum_type)
    # M_C_CAA_a_abb += 1/4 * einsum('IxyJ,yx,WZ->IJWZ', v_xaax, rdm_ca, rdm_ca, optimize = einsum_type)

    # M_C_CAA_a_bab =- 1/2 * einsum('IZxJ,Wx->IJWZ', v_xaax, rdm_ca, optimize = einsum_type)
    # M_C_CAA_a_bab += 1/6 * einsum('IxyJ,WxZy->IJWZ', v_xaax, rdm_ccaa, optimize = einsum_type)
    # M_C_CAA_a_bab += 1/3 * einsum('IxyJ,WxyZ->IJWZ', v_xaax, rdm_ccaa, optimize = einsum_type)

    # M_C_CAA_a_abb = M_C_CAA_a_abb.reshape(ncvs, -1)
    # M_C_CAA_a_bab = M_C_CAA_a_bab.reshape(ncvs, -1)

    # M_C_CAA = np.zeros((ncvs * 2, ncvs * 2, ncas * 2, ncas * 2))

    # M_C_CAA[::2,::2,1::2,1::2] = M_C_CAA_a_abb.copy()
    # M_C_CAA[1::2,1::2,::2,::2] = M_C_CAA_a_abb.copy()

    # M_C_CAA[::2,1::2,::2,1::2] = M_C_CAA_a_bab.copy()
    # M_C_CAA[1::2,::2,1::2,::2] = M_C_CAA_a_bab.copy()

    # M_C_CAA[::2,::2,::2,::2]  = M_C_CAA_a_abb.copy()
    # M_C_CAA[::2,::2,::2,::2] += M_C_CAA_a_bab.copy()
    # M_C_CAA[1::2,1::2,1::2,1::2] = M_C_CAA[::2,::2,::2,::2].copy()

    # M_C_CAA = M_C_CAA.reshape(ncvs * 2, -1)

    # C - CCE
    # M_C_CCE_a_abb  = einsum('KLIB->IKLB', v_xxxe, optimize = einsum_type).copy()
    # M_C_CCE_a_abb -= einsum('LB,IK->IKLB', h_xe, np.identity(ncvs), optimize = einsum_type)
    # M_C_CCE_a_abb -= einsum('B,IK,LB->IKLB', e_extern, np.identity(ncvs), t1_xe, optimize = einsum_type)
    # M_C_CCE_a_abb += einsum('L,IK,LB->IKLB', e_cvs, np.identity(ncvs), t1_xe, optimize = einsum_type)
    # M_C_CCE_a_abb -= einsum('IK,LxBy,yx->IKLB', np.identity(ncvs), v_xaea, rdm_ca, optimize = einsum_type)
    # M_C_CCE_a_abb += 1/2 * einsum('IK,LxyB,yx->IKLB', np.identity(ncvs), v_xaae, rdm_ca, optimize = einsum_type)
    # M_C_CCE_a_abb -= einsum('B,IK,LxBy,yx->IKLB', e_extern, np.identity(ncvs), t1_xaea, rdm_ca, optimize = einsum_type)
    # M_C_CCE_a_abb += 1/2 * einsum('B,IK,LxyB,yx->IKLB', e_extern, np.identity(ncvs), t1_xaae, rdm_ca, optimize = einsum_type)
    # M_C_CCE_a_abb += einsum('L,IK,LxBy,yx->IKLB', e_cvs, np.identity(ncvs), t1_xaea, rdm_ca, optimize = einsum_type)
    # M_C_CCE_a_abb -= 1/2 * einsum('L,IK,LxyB,yx->IKLB', e_cvs, np.identity(ncvs), t1_xaae, rdm_ca, optimize = einsum_type)
    # M_C_CCE_a_abb += einsum('xy,IK,LxBz,zy->IKLB', h_aa, np.identity(ncvs), t1_xaea, rdm_ca, optimize = einsum_type)
    # M_C_CCE_a_abb -= 1/2 * einsum('xy,IK,LxzB,zy->IKLB', h_aa, np.identity(ncvs), t1_xaae, rdm_ca, optimize = einsum_type)
    # M_C_CCE_a_abb -= einsum('xy,IK,LzBx,yz->IKLB', h_aa, np.identity(ncvs), t1_xaea, rdm_ca, optimize = einsum_type)
    # M_C_CCE_a_abb += 1/2 * einsum('xy,IK,LzxB,yz->IKLB', h_aa, np.identity(ncvs), t1_xaae, rdm_ca, optimize = einsum_type)
    # M_C_CCE_a_abb += einsum('IK,LxBy,xzwu,yzwu->IKLB', np.identity(ncvs), t1_xaea, v_aaaa, rdm_ccaa, optimize = einsum_type)
    # M_C_CCE_a_abb -= einsum('IK,LxBy,yzwu,xzwu->IKLB', np.identity(ncvs), t1_xaea, v_aaaa, rdm_ccaa, optimize = einsum_type)
    # M_C_CCE_a_abb -= 1/2 * einsum('IK,LxyB,xzwu,yzwu->IKLB', np.identity(ncvs), t1_xaae, v_aaaa, rdm_ccaa, optimize = einsum_type)
    # M_C_CCE_a_abb += 1/2 * einsum('IK,LxyB,yzwu,xzwu->IKLB', np.identity(ncvs), t1_xaae, v_aaaa, rdm_ccaa, optimize = einsum_type)
    # M_C_CCE_a_abb = M_C_CCE_a_abb.reshape(ncvs, -1).copy()

    # M_C_CCE = np.zeros((ncvs * 2, ncvs * 2, ncvs * 2, nextern * 2))

    # M_C_CCE[::2,::2,1::2,1::2] = M_C_CCE_a_abb.copy()
    # M_C_CCE[1::2,1::2,::2,::2] = M_C_CCE_a_abb.copy()

    # M_C_CCE[::2,1::2,::2,1::2] -= M_C_CCE_a_abb.transpose(0,2,1,3).copy()
    # M_C_CCE[1::2,::2,1::2,::2] = M_C_CCE[::2,1::2,::2,1::2].copy()

    # M_C_CCE_so = np.load('M_C_CCE_so.npy')
    # print(">>> SO-SA M_C_CCE diff: {:}".format(np.sum(M_C_CCE_so - M_C_CCE)))

    # M_C_CCE = M_C_CCE[:,cc_ind[0], cc_ind[1]].reshape(M_C_CCE.shape[0], -1).copy()

    # if nval > 0:
    #     # C - CVE
    #     M_C_CVE_a_abb  = einsum('KLIB->IKLB', v_xvxe, optimize = einsum_type).copy()
    #     M_C_CVE_a_abb -= einsum('LB,IK->IKLB', h_ve, np.identity(ncvs), optimize = einsum_type)
    #     M_C_CVE_a_abb -= einsum('B,IK,LB->IKLB', e_extern, np.identity(ncvs), t1_ve, optimize = einsum_type)
    #     M_C_CVE_a_abb += einsum('L,IK,LB->IKLB', e_val, np.identity(ncvs), t1_ve, optimize = einsum_type)
    #     M_C_CVE_a_abb -= einsum('IK,LxBy,yx->IKLB', np.identity(ncvs), v_vaea, rdm_ca, optimize = einsum_type)
    #     M_C_CVE_a_abb += 1/2 * einsum('IK,LxyB,yx->IKLB', np.identity(ncvs), v_vaae, rdm_ca, optimize = einsum_type)
    #     M_C_CVE_a_abb -= einsum('B,IK,LxBy,yx->IKLB', e_extern, np.identity(ncvs), t1_vaea, rdm_ca, optimize = einsum_type)
    #     M_C_CVE_a_abb += 1/2 * einsum('B,IK,LxyB,yx->IKLB', e_extern, np.identity(ncvs), t1_vaae, rdm_ca, optimize = einsum_type)
    #     M_C_CVE_a_abb += einsum('L,IK,LxBy,yx->IKLB', e_val, np.identity(ncvs), t1_vaea, rdm_ca, optimize = einsum_type)
    #     M_C_CVE_a_abb -= 1/2 * einsum('L,IK,LxyB,yx->IKLB', e_val, np.identity(ncvs), t1_vaae, rdm_ca, optimize = einsum_type)
    #     M_C_CVE_a_abb += einsum('xy,IK,LxBz,zy->IKLB', h_aa, np.identity(ncvs), t1_vaea, rdm_ca, optimize = einsum_type)
    #     M_C_CVE_a_abb -= 1/2 * einsum('xy,IK,LxzB,zy->IKLB', h_aa, np.identity(ncvs), t1_vaae, rdm_ca, optimize = einsum_type)
    #     M_C_CVE_a_abb -= einsum('xy,IK,LzBx,yz->IKLB', h_aa, np.identity(ncvs), t1_vaea, rdm_ca, optimize = einsum_type)
    #     M_C_CVE_a_abb += 1/2 * einsum('xy,IK,LzxB,yz->IKLB', h_aa, np.identity(ncvs), t1_vaae, rdm_ca, optimize = einsum_type)
    #     M_C_CVE_a_abb += einsum('IK,LxBy,xzwu,yzwu->IKLB', np.identity(ncvs), t1_vaea, v_aaaa, rdm_ccaa, optimize = einsum_type)
    #     M_C_CVE_a_abb -= einsum('IK,LxBy,yzwu,xzwu->IKLB', np.identity(ncvs), t1_vaea, v_aaaa, rdm_ccaa, optimize = einsum_type)
    #     M_C_CVE_a_abb -= 1/2 * einsum('IK,LxyB,xzwu,yzwu->IKLB', np.identity(ncvs), t1_vaae, v_aaaa, rdm_ccaa, optimize = einsum_type)
    #     M_C_CVE_a_abb += 1/2 * einsum('IK,LxyB,yzwu,xzwu->IKLB', np.identity(ncvs), t1_vaae, v_aaaa, rdm_ccaa, optimize = einsum_type)
    #     M_C_CVE = M_C_CVE_a_abb.reshape(ncvs, -1).copy()

        # M_C_CVE = np.zeros((ncvs * 2, ncvs * 2, nval * 2, nextern * 2))

        # M_C_CVE[::2,::2,::2,::2] = M_C_CVE_a_aaa.copy()
        # M_C_CVE[1::2,1::2,1::2,1::2] = M_C_CVE_a_aaa.copy()

        # M_C_CVE[::2,::2,1::2,1::2] = M_C_CVE_a_abb.copy()
        # M_C_CVE[1::2,1::2,::2,::2] = M_C_CVE_a_abb.copy()

        # M_C_CVE[::2,1::2,::2,1::2] = M_C_CVE_a_bab.copy()
        # M_C_CVE[1::2,::2,1::2,::2] = M_C_CVE_a_bab.copy()

        # M_C_CVE_so = np.load('M_C_CVE_so.npy')
        # print(">>> SO-SA M_C_CVE diff: {:}".format(np.sum(M_C_CVE_so - M_C_CVE)))

    # C - CAE
    # M_C_CAE_a_abb  = 1/2 * einsum('JxIB,Yx->IJYB', v_xaxe, rdm_ca, optimize = einsum_type)
    # M_C_CAE_a_abb -= 1/2 * einsum('xB,IJ,Yx->IJYB', h_ae, np.identity(ncvs), rdm_ca, optimize = einsum_type)
    # M_C_CAE_a_abb -= 1/2 * einsum('IJ,xyzB,Yzyx->IJYB', np.identity(ncvs), v_aaae, rdm_ccaa, optimize = einsum_type)
    # M_C_CAE_a_abb -= 1/2 * einsum('B,IJ,xB,Yx->IJYB', e_extern, np.identity(ncvs), t1_ae, rdm_ca, optimize = einsum_type)
    # M_C_CAE_a_abb -= 1/2 * einsum('B,IJ,xyzB,Yzyx->IJYB', e_extern, np.identity(ncvs), t1_aaae, rdm_ccaa, optimize = einsum_type)
    # M_C_CAE_a_abb += 1/2 * einsum('xy,IJ,xB,Yy->IJYB', h_aa, np.identity(ncvs), t1_ae, rdm_ca, optimize = einsum_type)
    # M_C_CAE_a_abb -= 1/2 * einsum('xy,IJ,zwxB,Yywz->IJYB', h_aa, np.identity(ncvs), t1_aaae, rdm_ccaa, optimize = einsum_type)
    # M_C_CAE_a_abb += 1/2 * einsum('xy,IJ,xzwB,Ywzy->IJYB', h_aa, np.identity(ncvs), t1_aaae, rdm_ccaa, optimize = einsum_type)
    # M_C_CAE_a_abb += 1/2 * einsum('xy,IJ,zxwB,Ywyz->IJYB', h_aa, np.identity(ncvs), t1_aaae, rdm_ccaa, optimize = einsum_type)
    # M_C_CAE_a_abb += 1/2 * einsum('IJ,xB,xyzw,Yyzw->IJYB', np.identity(ncvs), t1_ae, v_aaaa, rdm_ccaa, optimize = einsum_type)
    # M_C_CAE_a_abb += 1/12 * einsum('IJ,xyzB,zwuv,Yuvxyw->IJYB', np.identity(ncvs), t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    # M_C_CAE_a_abb += 1/12 * einsum('IJ,xyzB,zwuv,Yuvxwy->IJYB', np.identity(ncvs), t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    # M_C_CAE_a_abb -= 5/12 * einsum('IJ,xyzB,zwuv,Yuvyxw->IJYB', np.identity(ncvs), t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    # M_C_CAE_a_abb += 1/12 * einsum('IJ,xyzB,zwuv,Yuvywx->IJYB', np.identity(ncvs), t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    # M_C_CAE_a_abb += 1/12 * einsum('IJ,xyzB,zwuv,Yuvwxy->IJYB', np.identity(ncvs), t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    # M_C_CAE_a_abb += 1/12 * einsum('IJ,xyzB,zwuv,Yuvwyx->IJYB', np.identity(ncvs), t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    # M_C_CAE_a_abb += 1/2 * einsum('IJ,xyzB,xywu,Yzuw->IJYB', np.identity(ncvs), t1_aaae, v_aaaa, rdm_ccaa, optimize = einsum_type)
    # M_C_CAE_a_abb += 5/12 * einsum('IJ,xyzB,xwuv,Yzwyuv->IJYB', np.identity(ncvs), t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    # M_C_CAE_a_abb -= 1/12 * einsum('IJ,xyzB,xwuv,Yzwyvu->IJYB', np.identity(ncvs), t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    # M_C_CAE_a_abb -= 1/12 * einsum('IJ,xyzB,xwuv,Yzwuyv->IJYB', np.identity(ncvs), t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    # M_C_CAE_a_abb -= 1/12 * einsum('IJ,xyzB,xwuv,Yzwuvy->IJYB', np.identity(ncvs), t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    # M_C_CAE_a_abb -= 1/12 * einsum('IJ,xyzB,xwuv,Yzwvyu->IJYB', np.identity(ncvs), t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    # M_C_CAE_a_abb -= 1/12 * einsum('IJ,xyzB,xwuv,Yzwvuy->IJYB', np.identity(ncvs), t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    # M_C_CAE_a_abb -= 1/12 * einsum('IJ,xyzB,ywuv,Yzwxuv->IJYB', np.identity(ncvs), t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    # M_C_CAE_a_abb -= 1/12 * einsum('IJ,xyzB,ywuv,Yzwxvu->IJYB', np.identity(ncvs), t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    # M_C_CAE_a_abb += 5/12 * einsum('IJ,xyzB,ywuv,Yzwuxv->IJYB', np.identity(ncvs), t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    # M_C_CAE_a_abb -= 1/12 * einsum('IJ,xyzB,ywuv,Yzwuvx->IJYB', np.identity(ncvs), t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    # M_C_CAE_a_abb -= 1/12 * einsum('IJ,xyzB,ywuv,Yzwvxu->IJYB', np.identity(ncvs), t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    # M_C_CAE_a_abb -= 1/12 * einsum('IJ,xyzB,ywuv,Yzwvux->IJYB', np.identity(ncvs), t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    # M_C_CAE_a_abb = M_C_CAE_a_abb.reshape(ncvs, -1).copy()

    # M_C_CAE_a_bab =- 1/2 * einsum('JxBI,Yx->IJYB', v_xaex, rdm_ca, optimize = einsum_type)
    # M_C_CAE_a_bab = M_C_CAE_a_bab.reshape(ncvs, -1).copy()

    # C - CCA
    # M_C_CCA_a_abb  = einsum('KLIY->IKLY', v_xxxa, optimize = einsum_type).copy()
    # M_C_CCA_a_abb -= einsum('LY,IK->IKLY', h_xa, np.identity(ncvs), optimize = einsum_type)
    # M_C_CCA_a_abb -= 1/2 * einsum('KLIx,xY->IKLY', v_xxxa, rdm_ca, optimize = einsum_type)
    # M_C_CCA_a_abb += einsum('L,IK,LY->IKLY', e_cvs, np.identity(ncvs), t1_xa, optimize = einsum_type)
    # M_C_CCA_a_abb += 1/2 * einsum('Lx,IK,xY->IKLY', h_xa, np.identity(ncvs), rdm_ca, optimize = einsum_type)
    # M_C_CCA_a_abb -= einsum('Yx,IK,Lx->IKLY', h_aa, np.identity(ncvs), t1_xa, optimize = einsum_type)
    # M_C_CCA_a_abb -= einsum('IK,LxYy,yx->IKLY', np.identity(ncvs), v_xaaa, rdm_ca, optimize = einsum_type)
    # M_C_CCA_a_abb += 1/2 * einsum('IK,LxyY,yx->IKLY', np.identity(ncvs), v_xaaa, rdm_ca, optimize = einsum_type)
    # M_C_CCA_a_abb += 1/2 * einsum('IK,Lxyz,Yxyz->IKLY', np.identity(ncvs), v_xaaa, rdm_ccaa, optimize = einsum_type)
    # M_C_CCA_a_abb -= 1/2 * einsum('L,IK,Lx,xY->IKLY', e_cvs, np.identity(ncvs), t1_xa, rdm_ca, optimize = einsum_type)
    # M_C_CCA_a_abb += einsum('L,IK,LxYy,yx->IKLY', e_cvs, np.identity(ncvs), t1_xaaa, rdm_ca, optimize = einsum_type)
    # M_C_CCA_a_abb -= 1/2 * einsum('L,IK,LxyY,yx->IKLY', e_cvs, np.identity(ncvs), t1_xaaa, rdm_ca, optimize = einsum_type)
    # M_C_CCA_a_abb -= 1/2 * einsum('L,IK,Lxyz,Yxyz->IKLY', e_cvs, np.identity(ncvs), t1_xaaa, rdm_ccaa, optimize = einsum_type)
    # M_C_CCA_a_abb -= einsum('Yx,IK,Lyxz,zy->IKLY', h_aa, np.identity(ncvs), t1_xaaa, rdm_ca, optimize = einsum_type)
    # M_C_CCA_a_abb += 1/2 * einsum('Yx,IK,Lyzx,zy->IKLY', h_aa, np.identity(ncvs), t1_xaaa, rdm_ca, optimize = einsum_type)
    # M_C_CCA_a_abb += 1/2 * einsum('xy,IK,Lx,yY->IKLY', h_aa, np.identity(ncvs), t1_xa, rdm_ca, optimize = einsum_type)
    # M_C_CCA_a_abb += einsum('xy,IK,LxYz,zy->IKLY', h_aa, np.identity(ncvs), t1_xaaa, rdm_ca, optimize = einsum_type)
    # M_C_CCA_a_abb -= 1/2 * einsum('xy,IK,LxzY,zy->IKLY', h_aa, np.identity(ncvs), t1_xaaa, rdm_ca, optimize = einsum_type)
    # M_C_CCA_a_abb -= 1/2 * einsum('xy,IK,Lxzw,Yyzw->IKLY', h_aa, np.identity(ncvs), t1_xaaa, rdm_ccaa, optimize = einsum_type)
    # M_C_CCA_a_abb -= einsum('xy,IK,LzYx,yz->IKLY', h_aa, np.identity(ncvs), t1_xaaa, rdm_ca, optimize = einsum_type)
    # M_C_CCA_a_abb += 1/2 * einsum('xy,IK,LzxY,yz->IKLY', h_aa, np.identity(ncvs), t1_xaaa, rdm_ca, optimize = einsum_type)
    # M_C_CCA_a_abb += 1/2 * einsum('xy,IK,Lzxw,Yzyw->IKLY', h_aa, np.identity(ncvs), t1_xaaa, rdm_ccaa, optimize = einsum_type)
    # M_C_CCA_a_abb += 1/2 * einsum('xy,IK,Lzwx,Yzwy->IKLY', h_aa, np.identity(ncvs), t1_xaaa, rdm_ccaa, optimize = einsum_type)
    # M_C_CCA_a_abb -= einsum('IK,Lx,Yyxz,yz->IKLY', np.identity(ncvs), t1_xa, v_aaaa, rdm_ca, optimize = einsum_type)
    # M_C_CCA_a_abb += 1/2 * einsum('IK,Lx,Yyzx,yz->IKLY', np.identity(ncvs), t1_xa, v_aaaa, rdm_ca, optimize = einsum_type)
    # M_C_CCA_a_abb += 1/2 * einsum('IK,Lx,xyzw,Yyzw->IKLY', np.identity(ncvs), t1_xa, v_aaaa, rdm_ccaa, optimize = einsum_type)
    # M_C_CCA_a_abb += einsum('IK,LxYy,xzwu,yzwu->IKLY', np.identity(ncvs), t1_xaaa, v_aaaa, rdm_ccaa, optimize = einsum_type)
    # M_C_CCA_a_abb -= einsum('IK,LxYy,yzwu,xzwu->IKLY', np.identity(ncvs), t1_xaaa, v_aaaa, rdm_ccaa, optimize = einsum_type)
    # M_C_CCA_a_abb -= 1/2 * einsum('IK,LxyY,xzwu,yzwu->IKLY', np.identity(ncvs), t1_xaaa, v_aaaa, rdm_ccaa, optimize = einsum_type)
    # M_C_CCA_a_abb += 1/2 * einsum('IK,LxyY,yzwu,xzwu->IKLY', np.identity(ncvs), t1_xaaa, v_aaaa, rdm_ccaa, optimize = einsum_type)
    # M_C_CCA_a_abb -= 1/2 * einsum('IK,Lxyz,Yxwu,yzwu->IKLY', np.identity(ncvs), t1_xaaa, v_aaaa, rdm_ccaa, optimize = einsum_type)
    # M_C_CCA_a_abb -= einsum('IK,Lxyz,Ywyz,wx->IKLY', np.identity(ncvs), t1_xaaa, v_aaaa, rdm_ca, optimize = einsum_type)
    # M_C_CCA_a_abb -= einsum('IK,Lxyz,Ywyu,xuzw->IKLY', np.identity(ncvs), t1_xaaa, v_aaaa, rdm_ccaa, optimize = einsum_type)
    # M_C_CCA_a_abb += 1/2 * einsum('IK,Lxyz,Ywzy,wx->IKLY', np.identity(ncvs), t1_xaaa, v_aaaa, rdm_ca, optimize = einsum_type)
    # M_C_CCA_a_abb += 1/2 * einsum('IK,Lxyz,Ywzu,xuyw->IKLY', np.identity(ncvs), t1_xaaa, v_aaaa, rdm_ccaa, optimize = einsum_type)
    # M_C_CCA_a_abb += 1/2 * einsum('IK,Lxyz,Ywuy,xuzw->IKLY', np.identity(ncvs), t1_xaaa, v_aaaa, rdm_ccaa, optimize = einsum_type)
    # M_C_CCA_a_abb += 1/2 * einsum('IK,Lxyz,Ywuz,xuwy->IKLY', np.identity(ncvs), t1_xaaa, v_aaaa, rdm_ccaa, optimize = einsum_type)
    # M_C_CCA_a_abb -= 5/12 * einsum('IK,Lxyz,xwuv,yzwYuv->IKLY', np.identity(ncvs), t1_xaaa, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    # M_C_CCA_a_abb += 1/12 * einsum('IK,Lxyz,xwuv,yzwYvu->IKLY', np.identity(ncvs), t1_xaaa, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    # M_C_CCA_a_abb += 1/12 * einsum('IK,Lxyz,xwuv,yzwuYv->IKLY', np.identity(ncvs), t1_xaaa, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    # M_C_CCA_a_abb += 1/12 * einsum('IK,Lxyz,xwuv,yzwuvY->IKLY', np.identity(ncvs), t1_xaaa, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    # M_C_CCA_a_abb += 1/12 * einsum('IK,Lxyz,xwuv,yzwvYu->IKLY', np.identity(ncvs), t1_xaaa, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    # M_C_CCA_a_abb += 1/12 * einsum('IK,Lxyz,xwuv,yzwvuY->IKLY', np.identity(ncvs), t1_xaaa, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    # M_C_CCA_a_abb += 1/2 * einsum('IK,Lxyz,yzwu,Yxwu->IKLY', np.identity(ncvs), t1_xaaa, v_aaaa, rdm_ccaa, optimize = einsum_type)
    # M_C_CCA_a_abb -= 1/12 * einsum('IK,Lxyz,ywuv,zuvYxw->IKLY', np.identity(ncvs), t1_xaaa, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    # M_C_CCA_a_abb -= 1/12 * einsum('IK,Lxyz,ywuv,zuvYwx->IKLY', np.identity(ncvs), t1_xaaa, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    # M_C_CCA_a_abb += 5/12 * einsum('IK,Lxyz,ywuv,zuvxYw->IKLY', np.identity(ncvs), t1_xaaa, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    # M_C_CCA_a_abb -= 1/12 * einsum('IK,Lxyz,ywuv,zuvxwY->IKLY', np.identity(ncvs), t1_xaaa, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    # M_C_CCA_a_abb -= 1/12 * einsum('IK,Lxyz,ywuv,zuvwYx->IKLY', np.identity(ncvs), t1_xaaa, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    # M_C_CCA_a_abb -= 1/12 * einsum('IK,Lxyz,ywuv,zuvwxY->IKLY', np.identity(ncvs), t1_xaaa, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    # M_C_CCA_a_abb += 5/12 * einsum('IK,Lxyz,zwuv,yuvYxw->IKLY', np.identity(ncvs), t1_xaaa, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    # M_C_CCA_a_abb -= 1/12 * einsum('IK,Lxyz,zwuv,yuvYwx->IKLY', np.identity(ncvs), t1_xaaa, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    # M_C_CCA_a_abb -= 1/12 * einsum('IK,Lxyz,zwuv,yuvxYw->IKLY', np.identity(ncvs), t1_xaaa, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    # M_C_CCA_a_abb -= 1/12 * einsum('IK,Lxyz,zwuv,yuvxwY->IKLY', np.identity(ncvs), t1_xaaa, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    # M_C_CCA_a_abb -= 1/12 * einsum('IK,Lxyz,zwuv,yuvwYx->IKLY', np.identity(ncvs), t1_xaaa, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    # M_C_CCA_a_abb -= 1/12 * einsum('IK,Lxyz,zwuv,yuvwxY->IKLY', np.identity(ncvs), t1_xaaa, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    # M_C_CCA_a_abb = M_C_CCA_a_abb.reshape(ncvs, -1).copy()

    # if nval > 0:
    #     # C - CVA
    #     M_C_CVA_a_abb  = einsum('KLIY->IKLY', v_xvxa, optimize = einsum_type).copy()
    #     M_C_CVA_a_abb -= einsum('LY,IK->IKLY', h_va, np.identity(ncvs), optimize = einsum_type)
    #     M_C_CVA_a_abb -= 1/2 * einsum('KLIx,xY->IKLY', v_xvxa, rdm_ca, optimize = einsum_type)
    #     M_C_CVA_a_abb += einsum('L,IK,LY->IKLY', e_val, np.identity(ncvs), t1_va, optimize = einsum_type)
    #     M_C_CVA_a_abb += 1/2 * einsum('Lx,IK,xY->IKLY', h_va, np.identity(ncvs), rdm_ca, optimize = einsum_type)
    #     M_C_CVA_a_abb -= einsum('Yx,IK,Lx->IKLY', h_aa, np.identity(ncvs), t1_va, optimize = einsum_type)
    #     M_C_CVA_a_abb -= einsum('IK,LxYy,yx->IKLY', np.identity(ncvs), v_vaaa, rdm_ca, optimize = einsum_type)
    #     M_C_CVA_a_abb += 1/2 * einsum('IK,LxyY,yx->IKLY', np.identity(ncvs), v_vaaa, rdm_ca, optimize = einsum_type)
    #     M_C_CVA_a_abb += 1/2 * einsum('IK,Lxyz,Yxyz->IKLY', np.identity(ncvs), v_vaaa, rdm_ccaa, optimize = einsum_type)
    #     M_C_CVA_a_abb -= 1/2 * einsum('L,IK,Lx,xY->IKLY', e_val, np.identity(ncvs), t1_va, rdm_ca, optimize = einsum_type)
    #     M_C_CVA_a_abb += einsum('L,IK,LxYy,yx->IKLY', e_val, np.identity(ncvs), t1_vaaa, rdm_ca, optimize = einsum_type)
    #     M_C_CVA_a_abb -= 1/2 * einsum('L,IK,LxyY,yx->IKLY', e_val, np.identity(ncvs), t1_vaaa, rdm_ca, optimize = einsum_type)
    #     M_C_CVA_a_abb -= 1/2 * einsum('L,IK,Lxyz,Yxyz->IKLY', e_val, np.identity(ncvs), t1_vaaa, rdm_ccaa, optimize = einsum_type)
    #     M_C_CVA_a_abb -= einsum('Yx,IK,Lyxz,zy->IKLY', h_aa, np.identity(ncvs), t1_vaaa, rdm_ca, optimize = einsum_type)
    #     M_C_CVA_a_abb += 1/2 * einsum('Yx,IK,Lyzx,zy->IKLY', h_aa, np.identity(ncvs), t1_vaaa, rdm_ca, optimize = einsum_type)
    #     M_C_CVA_a_abb += 1/2 * einsum('xy,IK,Lx,yY->IKLY', h_aa, np.identity(ncvs), t1_va, rdm_ca, optimize = einsum_type)
    #     M_C_CVA_a_abb += einsum('xy,IK,LxYz,zy->IKLY', h_aa, np.identity(ncvs), t1_vaaa, rdm_ca, optimize = einsum_type)
    #     M_C_CVA_a_abb -= 1/2 * einsum('xy,IK,LxzY,zy->IKLY', h_aa, np.identity(ncvs), t1_vaaa, rdm_ca, optimize = einsum_type)
    #     M_C_CVA_a_abb -= 1/2 * einsum('xy,IK,Lxzw,Yyzw->IKLY', h_aa, np.identity(ncvs), t1_vaaa, rdm_ccaa, optimize = einsum_type)
    #     M_C_CVA_a_abb -= einsum('xy,IK,LzYx,yz->IKLY', h_aa, np.identity(ncvs), t1_vaaa, rdm_ca, optimize = einsum_type)
    #     M_C_CVA_a_abb += 1/2 * einsum('xy,IK,LzxY,yz->IKLY', h_aa, np.identity(ncvs), t1_vaaa, rdm_ca, optimize = einsum_type)
    #     M_C_CVA_a_abb += 1/2 * einsum('xy,IK,Lzxw,Yzyw->IKLY', h_aa, np.identity(ncvs), t1_vaaa, rdm_ccaa, optimize = einsum_type)
    #     M_C_CVA_a_abb += 1/2 * einsum('xy,IK,Lzwx,Yzwy->IKLY', h_aa, np.identity(ncvs), t1_vaaa, rdm_ccaa, optimize = einsum_type)
    #     M_C_CVA_a_abb -= einsum('IK,Lx,Yyxz,yz->IKLY', np.identity(ncvs), t1_va, v_aaaa, rdm_ca, optimize = einsum_type)
    #     M_C_CVA_a_abb += 1/2 * einsum('IK,Lx,Yyzx,yz->IKLY', np.identity(ncvs), t1_va, v_aaaa, rdm_ca, optimize = einsum_type)
    #     M_C_CVA_a_abb += 1/2 * einsum('IK,Lx,xyzw,Yyzw->IKLY', np.identity(ncvs), t1_va, v_aaaa, rdm_ccaa, optimize = einsum_type)
    #     M_C_CVA_a_abb += einsum('IK,LxYy,xzwu,yzwu->IKLY', np.identity(ncvs), t1_vaaa, v_aaaa, rdm_ccaa, optimize = einsum_type)
    #     M_C_CVA_a_abb -= einsum('IK,LxYy,yzwu,xzwu->IKLY', np.identity(ncvs), t1_vaaa, v_aaaa, rdm_ccaa, optimize = einsum_type)
    #     M_C_CVA_a_abb -= 1/2 * einsum('IK,LxyY,xzwu,yzwu->IKLY', np.identity(ncvs), t1_vaaa, v_aaaa, rdm_ccaa, optimize = einsum_type)
    #     M_C_CVA_a_abb += 1/2 * einsum('IK,LxyY,yzwu,xzwu->IKLY', np.identity(ncvs), t1_vaaa, v_aaaa, rdm_ccaa, optimize = einsum_type)
    #     M_C_CVA_a_abb -= 1/2 * einsum('IK,Lxyz,Yxwu,yzwu->IKLY', np.identity(ncvs), t1_vaaa, v_aaaa, rdm_ccaa, optimize = einsum_type)
    #     M_C_CVA_a_abb -= einsum('IK,Lxyz,Ywyz,wx->IKLY', np.identity(ncvs), t1_vaaa, v_aaaa, rdm_ca, optimize = einsum_type)
    #     M_C_CVA_a_abb -= einsum('IK,Lxyz,Ywyu,xuzw->IKLY', np.identity(ncvs), t1_vaaa, v_aaaa, rdm_ccaa, optimize = einsum_type)
    #     M_C_CVA_a_abb += 1/2 * einsum('IK,Lxyz,Ywzy,wx->IKLY', np.identity(ncvs), t1_vaaa, v_aaaa, rdm_ca, optimize = einsum_type)
    #     M_C_CVA_a_abb += 1/2 * einsum('IK,Lxyz,Ywzu,xuyw->IKLY', np.identity(ncvs), t1_vaaa, v_aaaa, rdm_ccaa, optimize = einsum_type)
    #     M_C_CVA_a_abb += 1/2 * einsum('IK,Lxyz,Ywuy,xuzw->IKLY', np.identity(ncvs), t1_vaaa, v_aaaa, rdm_ccaa, optimize = einsum_type)
    #     M_C_CVA_a_abb += 1/2 * einsum('IK,Lxyz,Ywuz,xuwy->IKLY', np.identity(ncvs), t1_vaaa, v_aaaa, rdm_ccaa, optimize = einsum_type)
    #     M_C_CVA_a_abb -= 5/12 * einsum('IK,Lxyz,xwuv,yzwYuv->IKLY', np.identity(ncvs), t1_vaaa, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    #     M_C_CVA_a_abb += 1/12 * einsum('IK,Lxyz,xwuv,yzwYvu->IKLY', np.identity(ncvs), t1_vaaa, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    #     M_C_CVA_a_abb += 1/12 * einsum('IK,Lxyz,xwuv,yzwuYv->IKLY', np.identity(ncvs), t1_vaaa, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    #     M_C_CVA_a_abb += 1/12 * einsum('IK,Lxyz,xwuv,yzwuvY->IKLY', np.identity(ncvs), t1_vaaa, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    #     M_C_CVA_a_abb += 1/12 * einsum('IK,Lxyz,xwuv,yzwvYu->IKLY', np.identity(ncvs), t1_vaaa, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    #     M_C_CVA_a_abb += 1/12 * einsum('IK,Lxyz,xwuv,yzwvuY->IKLY', np.identity(ncvs), t1_vaaa, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    #     M_C_CVA_a_abb += 1/2 * einsum('IK,Lxyz,yzwu,Yxwu->IKLY', np.identity(ncvs), t1_vaaa, v_aaaa, rdm_ccaa, optimize = einsum_type)
    #     M_C_CVA_a_abb -= 1/12 * einsum('IK,Lxyz,ywuv,zuvYxw->IKLY', np.identity(ncvs), t1_vaaa, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    #     M_C_CVA_a_abb -= 1/12 * einsum('IK,Lxyz,ywuv,zuvYwx->IKLY', np.identity(ncvs), t1_vaaa, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    #     M_C_CVA_a_abb += 5/12 * einsum('IK,Lxyz,ywuv,zuvxYw->IKLY', np.identity(ncvs), t1_vaaa, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    #     M_C_CVA_a_abb -= 1/12 * einsum('IK,Lxyz,ywuv,zuvxwY->IKLY', np.identity(ncvs), t1_vaaa, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    #     M_C_CVA_a_abb -= 1/12 * einsum('IK,Lxyz,ywuv,zuvwYx->IKLY', np.identity(ncvs), t1_vaaa, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    #     M_C_CVA_a_abb -= 1/12 * einsum('IK,Lxyz,ywuv,zuvwxY->IKLY', np.identity(ncvs), t1_vaaa, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    #     M_C_CVA_a_abb += 5/12 * einsum('IK,Lxyz,zwuv,yuvYxw->IKLY', np.identity(ncvs), t1_vaaa, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    #     M_C_CVA_a_abb -= 1/12 * einsum('IK,Lxyz,zwuv,yuvYwx->IKLY', np.identity(ncvs), t1_vaaa, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    #     M_C_CVA_a_abb -= 1/12 * einsum('IK,Lxyz,zwuv,yuvxYw->IKLY', np.identity(ncvs), t1_vaaa, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    #     M_C_CVA_a_abb -= 1/12 * einsum('IK,Lxyz,zwuv,yuvxwY->IKLY', np.identity(ncvs), t1_vaaa, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    #     M_C_CVA_a_abb -= 1/12 * einsum('IK,Lxyz,zwuv,yuvwYx->IKLY', np.identity(ncvs), t1_vaaa, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    #     M_C_CVA_a_abb -= 1/12 * einsum('IK,Lxyz,zwuv,yuvwxY->IKLY', np.identity(ncvs), t1_vaaa, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    #     M_C_CVA_a_abb = M_C_CVA_a_abb.reshape(ncvs, -1).copy()

    print("Time for computing M(h0-h1) blocks:               %f sec\n" % (time.time() - start_time))
    sys.stdout.flush()

    # if nval > 0:
    #     M_01 = (M_C_CAA, M_C_CCE, M_C_CVE, M_C_CAE, M_C_CCA, M_C_CVA)
    # else:
    #     M_01 = (M_C_CAA, M_C_CCE, M_C_CAE, M_C_CCA)

    return 'M_01'

def compute_M_01_dev(mr_adc):

    start_time = time.time()

    print ("Computing M(h0-h1) blocks...")
    sys.stdout.flush()

    shift = 100000.0
    M_C_CAA = shift
    M_C_CCE = shift
    M_C_CAE = shift
    M_C_CCA = shift

    nval = mr_adc.nval
    if nval > 0:
        M_C_CVE = shift
        M_C_CVA = shift

    print ("Time for computing M(h0-h1) blocks:               %f sec\n" % (time.time() - start_time))
    sys.stdout.flush()

    if nval > 0:
        return M_C_CAA, M_C_CCE, M_C_CVE, M_C_CAE, M_C_CCA, M_C_CVA
    else:
        return M_C_CAA, M_C_CCE, M_C_CAE, M_C_CCA

def compute_preconditioner_dev(mr_adc, M_00):

    start_time = time.time()

    # Einsum definition from kernel
    einsum = mr_adc.interface.einsum
    einsum_type = mr_adc.interface.einsum_type

    if mr_adc.method in ("mr-adc(0)", "mr-adc(1)"):

        # Multiply by -1.0, since we are solving for -M C = -S C E
        return (-1.0 * np.diag(M_00))

    # Variables from kernel
    ncvs = mr_adc.ncvs
    nval = mr_adc.nval
    ncas = mr_adc.ncas
    nextern = mr_adc.nextern

    e_cvs = mr_adc.mo_energy.x
    e_val = mr_adc.mo_energy.v
    e_extern = mr_adc.mo_energy.e

    # One-electron integrals
    h_aa = mr_adc.h1eff.aa

    # Two-electrons integrals
    v_aaaa = mr_adc.v2e.aaaa

    # Reduced Density Matrices
    rdm_ca = mr_adc.rdm.ca
    rdm_ccaa = mr_adc.rdm.ccaa

    # Overlap matrices
    S12_c_caa = mr_adc.S12.c_caa
    S12_cae = mr_adc.S12.cae
    S12_cca = mr_adc.S12.cca

    # Dimensions
    ho_s_c = mr_adc.h_orth.s_c
    ho_f_c = mr_adc.h_orth.f_c
    ho_s_c_caa = mr_adc.h_orth.s_c_caa
    ho_f_c_caa = mr_adc.h_orth.f_c_caa
    ho_s_cce = mr_adc.h_orth.s_cce
    ho_f_cce = mr_adc.h_orth.f_cce
    ho_s_cae = mr_adc.h_orth.s_cae
    ho_f_cae = mr_adc.h_orth.f_cae
    ho_s_ace = mr_adc.h_orth.s_ace
    ho_f_ace = mr_adc.h_orth.f_ace
    ho_s_cca = mr_adc.h_orth.s_cca
    ho_f_cca = mr_adc.h_orth.f_cca
    if nval > 0:
        ho_s_cve = mr_adc.h_orth.s_cve
        ho_f_cve = mr_adc.h_orth.f_cve
        ho_s_vce = mr_adc.h_orth.s_vce
        ho_f_vce = mr_adc.h_orth.f_vce

        ho_s_cva = mr_adc.h_orth.s_cva
        ho_f_cva = mr_adc.h_orth.f_cva
        ho_s_vca = mr_adc.h_orth.s_vca
        ho_f_vca = mr_adc.h_orth.f_vca

    s_c = mr_adc.h0.s_c
    f_c = mr_adc.h0.f_c

    cvs_tril_ind = np.tril_indices(ncvs, k=-1)
    # cas_ind = np.tril_indices(ncas, k=-1)

    # Build the preconditioner
    precond = np.zeros(mr_adc.h_orth.dim)

    # C and CAA
    # precond[ho_s_c_caa:ho_f_c_caa] += shift
    # temp = np.zeros((ncvs, (1 + ncas * ncas), (1 + ncas * ncas)))
    # temp[:,0,0] = np.diag(M_00[s_c:f_c, s_c:f_c]).copy()
    # temp[:,0,1:] = 0.0
    # temp[:,1:,0] = 0.0
    # temp[:,1:,1:] = shift
    # dim_temp = range(1+ncas*ncas)

    # precond[ho_s_c_caa:ho_f_c_caa] = einsum('IXY,XP,YP->IP', temp, S12_c_caa, S12_c_caa, optimize = einsum_type).reshape(-1)
    # precond[ho_s_c_caa:ho_f_c_caa] = temp[:,dim_temp,dim_temp].reshape(-1)

    # C-C debug
    precond[ho_s_c:ho_f_c] = np.diag(M_00[s_c:f_c, s_c:f_c]).copy()

    # CCE
    # precond_cce =- einsum('A,AA,II,JJ->IJA', e_extern, np.identity(nextern), np.identity(ncvs), np.identity(ncvs), optimize = einsum_type)
    # precond_cce += einsum('I,AA,II,JJ->IJA', e_cvs, np.identity(nextern), np.identity(ncvs), np.identity(ncvs), optimize = einsum_type)
    # precond_cce += einsum('J,AA,II,JJ->IJA', e_cvs, np.identity(nextern), np.identity(ncvs), np.identity(ncvs), optimize = einsum_type)
    # precond[ho_s_cce:ho_f_cce] = precond_cce.reshape(-1).copy()

    # if nval > 0:
    #     # CVE
    #     precond_cve =- einsum('A,AA,II,JJ->IJA', e_extern, np.identity(nextern), np.identity(ncvs), np.identity(nval), optimize = einsum_type)
    #     precond_cve += einsum('I,AA,II,JJ->IJA', e_cvs, np.identity(nextern), np.identity(ncvs), np.identity(nval), optimize = einsum_type)
    #     precond_cve += einsum('J,AA,II,JJ->IJA', e_val, np.identity(nextern), np.identity(ncvs), np.identity(nval), optimize = einsum_type)
    #     precond[ho_s_cve:ho_f_cve] = precond_cve.reshape(-1).copy()

    #     # VCE
    #     precond_vce =- einsum('A,AA,II,JJ->IJA', e_extern, np.identity(nextern), np.identity(nval), np.identity(ncvs), optimize = einsum_type)
    #     precond_vce += einsum('I,AA,II,JJ->IJA', e_val, np.identity(nextern), np.identity(nval), np.identity(ncvs), optimize = einsum_type)
    #     precond_vce += einsum('J,AA,II,JJ->IJA', e_cvs, np.identity(nextern), np.identity(nval), np.identity(ncvs), optimize = einsum_type)
    #     precond[ho_s_vce:ho_f_vce] = precond_vce.reshape(-1).copy()

    # CAE
    precond_cae =- 1/2 * einsum('A,AA,II,XY->IAXY', e_extern, np.identity(nextern), np.identity(ncvs), rdm_ca, optimize = einsum_type)
    precond_cae += 1/2 * einsum('I,AA,II,XY->IAXY', e_cvs, np.identity(nextern), np.identity(ncvs), rdm_ca, optimize = einsum_type)
    precond_cae += 1/2 * einsum('Xx,AA,II,xY->IAXY', h_aa, np.identity(nextern), np.identity(ncvs), rdm_ca, optimize = einsum_type)
    precond_cae += 1/2 * einsum('Xxyz,AA,II,Yxyz->IAXY', v_aaaa, np.identity(nextern), np.identity(ncvs), rdm_ccaa, optimize = einsum_type)

    precond_cae = einsum("IAXY,XP,YP->IPA", precond_cae, S12_cae, S12_cae, optimize = einsum_type)
    print(">>> SA CAE: {:}".format(np.linalg.norm(precond_cae)))
    precond[ho_s_cae:ho_f_cae] = precond_cae.reshape(-1).copy()

    # ACE
    # precond_ace =- 1/2 * einsum('A,AA,II,XY->XYIA', e_extern, np.identity(nextern), np.identity(ncvs), rdm_ca, optimize = einsum_type)
    # precond_ace += 1/2 * einsum('I,AA,II,XY->XYIA', e_cvs, np.identity(nextern), np.identity(ncvs), rdm_ca, optimize = einsum_type)
    # precond_ace += 1/2 * einsum('Xx,AA,II,xY->XYIA', h_aa, np.identity(nextern), np.identity(ncvs), rdm_ca, optimize = einsum_type)
    # precond_ace += 1/2 * einsum('Xxyz,AA,II,Yxyz->XYIA', v_aaaa, np.identity(nextern), np.identity(ncvs), rdm_ccaa, optimize = einsum_type)

    # precond_ace = einsum("XYIA,XP,YP->PIA", precond_ace, S12_cae, S12_cae, optimize = einsum_type)
    # print(">>> SA ACE: {:}".format(np.linalg.norm(precond_ace)))
    # precond[ho_s_ace:ho_f_ace] = precond_ace.reshape(-1).copy()

    # CCA
    precond_cca =- einsum('XY,II,JJ->IJXY', h_aa, np.identity(ncvs), np.identity(ncvs), optimize = einsum_type)
    precond_cca += einsum('I,II,JJ,XY->IJXY', e_cvs, np.identity(ncvs), np.identity(ncvs), np.identity(ncas), optimize = einsum_type)
    precond_cca += einsum('J,II,JJ,XY->IJXY', e_cvs, np.identity(ncvs), np.identity(ncvs), np.identity(ncas), optimize = einsum_type)
    precond_cca -= 1/2 * einsum('I,II,JJ,YX->IJXY', e_cvs, np.identity(ncvs), np.identity(ncvs), rdm_ca, optimize = einsum_type)
    precond_cca -= 1/2 * einsum('J,II,JJ,YX->IJXY', e_cvs, np.identity(ncvs), np.identity(ncvs), rdm_ca, optimize = einsum_type)
    precond_cca += 1/2 * einsum('Xx,II,JJ,Yx->IJXY', h_aa, np.identity(ncvs), np.identity(ncvs), rdm_ca, optimize = einsum_type)
    precond_cca -= einsum('XxYy,II,JJ,xy->IJXY', v_aaaa, np.identity(ncvs), np.identity(ncvs), rdm_ca, optimize = einsum_type)
    precond_cca += 1/2 * einsum('XxyY,II,JJ,xy->IJXY', v_aaaa, np.identity(ncvs), np.identity(ncvs), rdm_ca, optimize = einsum_type)
    precond_cca += 1/2 * einsum('Xxyz,II,JJ,Yxyz->IJXY', v_aaaa, np.identity(ncvs), np.identity(ncvs), rdm_ccaa, optimize = einsum_type)

    precond_cca = einsum("IJXY,XP,YP->IJP", precond_cca, S12_cca, S12_cca, optimize = einsum_type)
    print(">>> SA CCA: {:}".format(np.linalg.norm(precond_cca)))
    precond[ho_s_cca:ho_f_cca] = precond_cca.reshape(-1).copy()

    # if nval > 0:
    #     # CVA
    #     precond_cva =- einsum('XY,II,JJ->IJXY', h_aa, np.identity(ncvs), np.identity(nval), optimize = einsum_type)
    #     precond_cva += einsum('I,II,JJ,XY->IJXY', e_cvs, np.identity(ncvs), np.identity(nval), np.identity(ncas), optimize = einsum_type)
    #     precond_cva += einsum('J,II,JJ,XY->IJXY', e_val, np.identity(ncvs), np.identity(nval), np.identity(ncas), optimize = einsum_type)
    #     precond_cva -= 1/2 * einsum('I,II,JJ,YX->IJXY', e_cvs, np.identity(ncvs), np.identity(nval), rdm_ca, optimize = einsum_type)
    #     precond_cva -= 1/2 * einsum('J,II,JJ,YX->IJXY', e_val, np.identity(ncvs), np.identity(nval), rdm_ca, optimize = einsum_type)
    #     precond_cva += 1/2 * einsum('Xx,II,JJ,Yx->IJXY', h_aa, np.identity(ncvs), np.identity(nval), rdm_ca, optimize = einsum_type)
    #     precond_cva -= einsum('XxYy,II,JJ,xy->IJXY', v_aaaa, np.identity(ncvs), np.identity(nval), rdm_ca, optimize = einsum_type)
    #     precond_cva += 1/2 * einsum('XxyY,II,JJ,xy->IJXY', v_aaaa, np.identity(ncvs), np.identity(nval), rdm_ca, optimize = einsum_type)
    #     precond_cva += 1/2 * einsum('Xxyz,II,JJ,Yxyz->IJXY', v_aaaa, np.identity(ncvs), np.identity(nval), rdm_ccaa, optimize = einsum_type)
    #     precond_cva = einsum("IJXY,XP,YP->IJP", precond_cva, S12_cca, S12_cca, optimize = einsum_type)
    #     precond[ho_s_cva:ho_f_cva] = precond_cva.reshape(-1).copy()

    #     precond_vca =- einsum('XY,II,JJ->IJXY', h_aa, np.identity(nval), np.identity(ncvs), optimize = einsum_type)
    #     precond_vca += einsum('I,II,JJ,XY->IJXY', e_val, np.identity(nval), np.identity(ncvs), np.identity(ncas), optimize = einsum_type)
    #     precond_vca += einsum('J,II,JJ,XY->IJXY', e_cvs, np.identity(nval), np.identity(ncvs), np.identity(ncas), optimize = einsum_type)
    #     precond_vca -= 1/2 * einsum('I,II,JJ,YX->IJXY', e_val, np.identity(nval), np.identity(ncvs), rdm_ca, optimize = einsum_type)
    #     precond_vca -= 1/2 * einsum('J,II,JJ,YX->IJXY', e_cvs, np.identity(nval), np.identity(ncvs), rdm_ca, optimize = einsum_type)
    #     precond_vca += 1/2 * einsum('Xx,II,JJ,Yx->IJXY', h_aa, np.identity(nval), np.identity(ncvs), rdm_ca, optimize = einsum_type)
    #     precond_vca -= einsum('XxYy,II,JJ,xy->IJXY', v_aaaa, np.identity(nval), np.identity(ncvs), rdm_ca, optimize = einsum_type)
    #     precond_vca += 1/2 * einsum('XxyY,II,JJ,xy->IJXY', v_aaaa, np.identity(nval), np.identity(ncvs), rdm_ca, optimize = einsum_type)
    #     precond_vca += 1/2 * einsum('Xxyz,II,JJ,Yxyz->IJXY', v_aaaa, np.identity(nval), np.identity(ncvs), rdm_ccaa, optimize = einsum_type)
    #     precond_vca = einsum("IJXY,XP,YP->IJP", precond_vca, S12_cca, S12_cca, optimize = einsum_type)
    #     precond[ho_s_vca:ho_f_vca] = precond_vca.reshape(-1).copy()

    # Multiply by -1.0, since we are solving for -M C = -S C E
    precond *= (-1.0)

    print ("Time for computing preconditioner:                %f sec\n" % (time.time() - start_time))
    sys.stdout.flush()

    return precond

def apply_S_12_dev(mr_adc, X, transpose = False):

    # Einsum definition from kernel
    einsum = mr_adc.interface.einsum
    einsum_type = mr_adc.interface.einsum_type

    # Dimensions
    nextern = mr_adc.nextern
    ncvs = mr_adc.ncvs
    nval = mr_adc.nval

    ho_s_c = mr_adc.h_orth.s_c
    ho_f_c = mr_adc.h_orth.f_c
    ho_s_c_caa = mr_adc.h_orth.s_c_caa
    ho_f_c_caa = mr_adc.h_orth.f_c_caa
    ho_s_cce = mr_adc.h_orth.s_cce
    ho_f_cce = mr_adc.h_orth.f_cce
    ho_s_cae = mr_adc.h_orth.s_cae
    ho_f_cae = mr_adc.h_orth.f_cae
    ho_s_ace = mr_adc.h_orth.s_ace
    ho_f_ace = mr_adc.h_orth.f_ace
    ho_s_cca = mr_adc.h_orth.s_cca
    ho_f_cca = mr_adc.h_orth.f_cca
    s_c = mr_adc.h0.s_c
    f_c = mr_adc.h0.f_c
    s_caa = mr_adc.h1.s_caa
    f_caa = mr_adc.h1.f_caa
    s_cce = mr_adc.h1.s_cce
    f_cce = mr_adc.h1.f_cce
    s_cae = mr_adc.h1.s_cae
    f_cae = mr_adc.h1.f_cae
    s_ace = mr_adc.h1.s_ace
    f_ace = mr_adc.h1.f_ace
    s_cca = mr_adc.h1.s_cca
    f_cca = mr_adc.h1.f_cca

    if nval > 0:
        ho_s_cve = mr_adc.h_orth.s_cve
        ho_f_cve = mr_adc.h_orth.f_cve
        ho_s_vce = mr_adc.h_orth.s_vce
        ho_f_vce = mr_adc.h_orth.f_vce

        ho_s_cva = mr_adc.h_orth.s_cva
        ho_f_cva = mr_adc.h_orth.f_cva
        ho_s_vca = mr_adc.h_orth.s_vca
        ho_f_vca = mr_adc.h_orth.f_vca

        s_cve = mr_adc.h1.s_cve
        f_cve = mr_adc.h1.f_cve
        s_vce = mr_adc.h1.s_vce
        f_vce = mr_adc.h1.f_vce

        s_cva = mr_adc.h1.s_cva
        f_cva = mr_adc.h1.f_cva
        s_vca = mr_adc.h1.s_vca
        f_vca = mr_adc.h1.f_vca

    # Overlap matrices
    S12_c_caa = mr_adc.S12.c_caa
    S12_cae = mr_adc.S12.cae
    S12_cca = mr_adc.S12.cca

    Xt = None

    if transpose:
        if (X.shape[0] != (mr_adc.h0.dim + mr_adc.h1.dim)):
            raise Exception("Dimensions do not match when applying S_12 transpose")

        Xt = np.zeros(mr_adc.h_orth.dim)

        # C and CAA -> C_CAA
        # temp = np.zeros((ncvs, S12_c_caa.shape[0]))
        # temp[:,0] = X[s_c:f_c].copy()
        # temp[:,1:] = X[s_caa:f_caa].reshape(ncvs, -1).copy()
        # Xt[ho_s_c_caa:ho_f_c_caa] = np.dot(temp, S12_c_caa).reshape(-1).copy()

        # C-C DEBUG
        Xt[ho_s_c:ho_f_c] = X[s_c:f_c].copy()

        # CCE
        # Xt[ho_s_cce:ho_f_cce] = X[s_cce:f_cce].copy()

        # if nval > 0:
        #     # CVE
        #     Xt[ho_s_cve:ho_f_cve] = X[s_cve:f_cve].copy()

        #     # VCE
        #     Xt[ho_s_vce:ho_f_vce] = X[s_vce:f_vce].copy()

        # CAE
        temp = X[s_cae:f_cae].reshape(ncvs, S12_cae.shape[0], nextern).copy()
        Xt[ho_s_cae:ho_f_cae] = einsum("IXA,XP->IPA", temp, S12_cae).reshape(-1).copy()

        # ACE
        # temp = X[s_ace:f_ace].reshape(S12_cae.shape[0], ncvs, nextern).copy()
        # Xt[ho_s_ace:ho_f_ace] = einsum("XIA,XP->PIA", temp, S12_cae).reshape(-1).copy()

        # CCA
        n_cc = ncvs * ncvs
        temp = X[s_cca:f_cca].reshape(n_cc, S12_cca.shape[0]).copy()
        Xt[ho_s_cca:ho_f_cca] = einsum("IX,XP->IP", temp, S12_cca).reshape(-1).copy()

        # if nval > 0:
        #     # CVA
        #     n_cv = ncvs * nval
        #     temp = X[s_cva:f_cva].reshape(n_cv, S12_cca.shape[0]).copy()
        #     Xt[ho_s_cva:ho_f_cva] = einsum("IX,XP->IP", temp, S12_cca).reshape(-1).copy()

        #     # VCA
        #     temp = X[s_vca:f_vca].reshape(n_cv, S12_cca.shape[0]).copy()
        #     Xt[ho_s_vca:ho_f_vca] = einsum("IX,XP->IP", temp, S12_cca).reshape(-1).copy()

    else:
        if (X.shape[0] != (mr_adc.h_orth.dim)):
            raise Exception("Dimensions do not match when applying S_12")

        Xt = np.zeros(mr_adc.h0.dim + mr_adc.h1.dim)

        # C_CAA -> C and CAA
        # temp = X[ho_s_c_caa:ho_f_c_caa].reshape(ncvs, -1).copy()
        # temp = np.dot(temp, S12_c_caa.T)
        # Xt[s_c:f_c] = temp[:,0].copy()
        # Xt[s_caa:f_caa] = temp[:,1:].reshape(-1).copy()

        # C-C DEBUG
        Xt[s_c:f_c] = X[ho_s_c:ho_f_c].copy()

        # CCE
        # Xt[s_cce:f_cce] = X[ho_s_cce:ho_f_cce].copy()

        # if nval > 0:
        #     # CVE
        #     Xt[s_cve:f_cve] = X[ho_s_cve:ho_f_cve].copy()

        #     # VCE
        #     Xt[s_vce:f_vce] = X[ho_s_vce:ho_f_vce].copy()

        # CAE
        temp = X[ho_s_cae:ho_f_cae].reshape(ncvs, S12_cae.shape[1], nextern).copy()
        Xt[s_cae:f_cae] = einsum("IPA,XP->IXA", temp, S12_cae).reshape(-1).copy()

        # ACE
        # temp = X[ho_s_ace:ho_f_ace].reshape(S12_cae.shape[1], ncvs, nextern).copy()
        # Xt[s_ace:f_ace] = einsum("PIA,XP->XIA", temp, S12_cae).reshape(-1).copy()

        # CCA
        n_cc = ncvs * ncvs
        temp = X[ho_s_cca:ho_f_cca].reshape(n_cc, S12_cca.shape[1]).copy()
        Xt[s_cca:f_cca] = einsum("IP,XP->IX", temp, S12_cca).reshape(-1).copy()

        # if nval > 0:
        #     # CVA
        #     n_cv = ncvs * nval
        #     temp = X[ho_s_cva:ho_f_cva].reshape(n_cv, S12_cca.shape[1]).copy()
        #     Xt[s_cva:f_cva] = einsum("IP,XP->IX", temp, S12_cca).reshape(-1).copy()

        #     # VCA
        #     temp = X[ho_s_vca:ho_f_vca].reshape(n_cv, S12_cca.shape[1]).copy()
        #     Xt[s_vca:f_vca] = einsum("IP,XP->IX", temp, S12_cca).reshape(-1).copy()

    return Xt

def compute_sigma_vector_dev(mr_adc, M_00, M_01, M_11, Xt):

    # Einsum definition from kernel
    einsum = mr_adc.interface.einsum
    einsum_type = mr_adc.interface.einsum_type

    # Variables from kernel
    ncvs = mr_adc.ncvs
    nval = mr_adc.nval
    ncas = mr_adc.ncas
    nextern = mr_adc.nextern

    e_cvs = mr_adc.mo_energy.x
    e_core = mr_adc.mo_energy.c
    e_val = mr_adc.mo_energy.v
    e_extern = mr_adc.mo_energy.e

    # One-electron integrals
    h_aa = mr_adc.h1eff.aa

    # Two-electrons integrals
    v_aaaa = mr_adc.v2e.aaaa

    # Reduced Density Matrices
    rdm_ca = mr_adc.rdm.ca
    rdm_ccaa = mr_adc.rdm.ccaa

    # Dimensions
    s_c = mr_adc.h0.s_c
    f_c = mr_adc.h0.f_c
    s_caa = mr_adc.h1.s_caa
    f_caa = mr_adc.h1.f_caa
    s_cce = mr_adc.h1.s_cce
    f_cce = mr_adc.h1.f_cce
    s_cae = mr_adc.h1.s_cae
    f_cae = mr_adc.h1.f_cae
    s_ace = mr_adc.h1.s_ace
    f_ace = mr_adc.h1.f_ace
    s_cca = mr_adc.h1.s_cca
    f_cca = mr_adc.h1.f_cca
    if nval > 0:
        s_cve = mr_adc.h1.s_cve
        f_cve = mr_adc.h1.f_cve
        s_vce = mr_adc.h1.s_vce
        f_vce = mr_adc.h1.f_vce

        s_cva = mr_adc.h1.s_cva
        f_cva = mr_adc.h1.f_cva
        s_vca = mr_adc.h1.s_vca
        f_vca = mr_adc.h1.f_vca

    cvs_tril_ind = np.tril_indices(ncvs, k=-1)

    # (CASCI + C) -> (CASCI + C)
    sigma = np.zeros_like(Xt)

    # h0-h0 contributions
    sigma[:mr_adc.h0.dim] = np.dot(M_00, Xt[:mr_adc.h0.dim])

    # h0-h1 and h1-h0 contributions
    # if nval > 0:
    #     M_C_CAA, M_C_CCE, M_C_CVE, M_C_CAE, M_C_CCA, M_C_CVA = M_01
    # else:
    #     M_C_CAA, M_C_CCE, M_C_CAE, M_C_CCA = M_01

    # C <-> CAA
    # sigma[s_c:f_c] += np.dot(M_C_CAA, Xt[s_caa:f_caa])
    # sigma[s_caa:f_caa] += np.dot(M_C_CAA.T, Xt[s_c:f_c])

    # C <-> CCE
    # sigma[s_c:f_c] += np.dot(M_C_CCE, Xt[s_cce:f_cce])
    # sigma[s_cce:f_cce] += np.dot(M_C_CCE.T, Xt[s_c:f_c])

    # C <-> CVE
    # if nval > 0:
    #     sigma[s_c:f_c] += np.dot(M_C_CVE, Xt[s_cve:f_cve])
    #     sigma[s_cve:f_cve] += np.dot(M_C_CVE.T, Xt[s_c:f_c])

    # C <-> CAE
    # sigma[s_cae:f_cae] = 0.0

    # C <-> CCA
    # sigma[s_cca:f_cca] = 0.0

    # C <-> CVA
    # if nval > 0:
    #     sigma[s_cva:f_cva] = 0.0

    # h1-h1 contributions
    # CAA <- CAA
    # sigma[s_caa:f_caa] = 0.0

    # CCE <- CCE
    # X = Xt[s_cce:f_cce].reshape(ncvs, ncvs, nextern).copy()

    sigma_cce =- einsum('KLB,B->KLB', X, e_extern, optimize = einsum_type)
    sigma_cce += einsum('KLB,K->KLB', X, e_cvs, optimize = einsum_type)
    sigma_cce += einsum('KLB,L->KLB', X, e_cvs, optimize = einsum_type)

    # print(">>> SA sigma CCE: {:}".format(np.allclose(sigma_cce, sigma_cce_old, atol=1e-12)))
    # sigma[s_cce:f_cce] += sigma_cce.reshape(-1).copy()

    # if nval > 0:
    #     # CVE <- CVE
    #     X = Xt[s_cve:f_cve].reshape(ncvs, nval, nextern).copy()

    #     sigma_cve =- einsum('IJA,A,AA,II,JJ->IJA', X, e_extern, np.identity(nextern), np.identity(ncvs), np.identity(nval), optimize = einsum_type)
    #     sigma_cve += einsum('IJA,I,AA,II,JJ->IJA', X, e_cvs, np.identity(nextern), np.identity(ncvs), np.identity(nval), optimize = einsum_type)
    #     sigma_cve += einsum('IJA,J,AA,II,JJ->IJA', X, e_val, np.identity(nextern), np.identity(ncvs), np.identity(nval), optimize = einsum_type)
    #     sigma[s_cve:f_cve] += sigma_cve.reshape(-1).copy()

    #     # VCE <- VCE
    #     X = Xt[s_vce:f_vce].reshape(nval, ncvs, nextern).copy()

    #     sigma_vce =- einsum('IJA,A,AA,II,JJ->IJA', X, e_extern, np.identity(nextern), np.identity(nval), np.identity(ncvs), optimize = einsum_type)
    #     sigma_vce += einsum('IJA,I,AA,II,JJ->IJA', X, e_val, np.identity(nextern), np.identity(nval), np.identity(ncvs), optimize = einsum_type)
    #     sigma_vce += einsum('IJA,J,AA,II,JJ->IJA', X, e_cvs, np.identity(nextern), np.identity(nval), np.identity(ncvs), optimize = einsum_type)
    #     sigma[s_vce:f_vce] += sigma_vce.reshape(-1).copy()

    # CAE <- CAE
    X = Xt[s_cae:f_cae].reshape(ncvs, ncas, nextern).copy()

    sigma_cae =- 1/2 * einsum('IXA,A,AA,II,XX->IXA', X, e_extern, np.identity(nextern), np.identity(ncvs), rdm_ca, optimize = einsum_type)
    sigma_cae += 1/2 * einsum('IXA,I,AA,II,XX->IXA', X, e_cvs, np.identity(nextern), np.identity(ncvs), rdm_ca, optimize = einsum_type)
    sigma_cae += 1/2 * einsum('IXA,Xx,AA,II,xX->IXA', X, h_aa, np.identity(nextern), np.identity(ncvs), rdm_ca, optimize = einsum_type)
    sigma_cae += 1/2 * einsum('IXA,Xxyz,AA,II,Xxyz->IXA', X, v_aaaa, np.identity(nextern), np.identity(ncvs), rdm_ccaa, optimize = einsum_type)
    print(">>> SA sigma CAE: {:}".format(np.linalg.norm(sigma_cae)))
    sigma[s_cae:f_cae] += sigma_cae.reshape(-1).copy()

    # ACE <- ACE
    # X = Xt[s_ace:f_ace].reshape(ncas, ncvs, nextern).copy()

    # sigma_ace =- 1/2 * einsum('XIA,A,AA,II,XX->IXA', X, e_extern, np.identity(nextern), np.identity(ncvs), rdm_ca, optimize = einsum_type)
    # sigma_ace += 1/2 * einsum('XIA,I,AA,II,XX->IXA', X, e_cvs, np.identity(nextern), np.identity(ncvs), rdm_ca, optimize = einsum_type)
    # sigma_ace += 1/2 * einsum('XIA,Xx,AA,II,xX->IXA', X, h_aa, np.identity(nextern), np.identity(ncvs), rdm_ca, optimize = einsum_type)
    # sigma_ace += 1/2 * einsum('XIA,Xxyz,AA,II,Xxyz->IXA', X, v_aaaa, np.identity(nextern), np.identity(ncvs), rdm_ccaa, optimize = einsum_type)
    # print(">>> SA sigma ACE: {:}".format(np.linalg.norm(sigma_ace)))
    # sigma[s_ace:f_ace] += sigma_cae.reshape(-1).copy()

    # CCA <- CCA
    X = Xt[s_cca:f_cca].reshape(ncvs, ncvs, ncas).copy()
    print(">>> SA X CCA: {:}".format(np.linalg.norm(X)))

    sigma_cca =- einsum('IJX,XX,II,JJ->IJX', X, h_aa, np.identity(ncvs), np.identity(ncvs), optimize = einsum_type)
    sigma_cca += einsum('IJX,I,II,JJ,XX->IJX', X, e_cvs, np.identity(ncvs), np.identity(ncvs), np.identity(ncas), optimize = einsum_type)
    sigma_cca += einsum('IJX,J,II,JJ,XX->IJX', X, e_cvs, np.identity(ncvs), np.identity(ncvs), np.identity(ncas), optimize = einsum_type)
    sigma_cca -= 1/2 * einsum('IJX,I,II,JJ,XX->IJX', X, e_cvs, np.identity(ncvs), np.identity(ncvs), rdm_ca, optimize = einsum_type)
    sigma_cca -= 1/2 * einsum('IJX,J,II,JJ,XX->IJX', X, e_cvs, np.identity(ncvs), np.identity(ncvs), rdm_ca, optimize = einsum_type)
    sigma_cca += 1/2 * einsum('IJX,Xx,II,JJ,Xx->IJX', X, h_aa, np.identity(ncvs), np.identity(ncvs), rdm_ca, optimize = einsum_type)
    sigma_cca -= einsum('IJX,XxXy,II,JJ,xy->IJX', X, v_aaaa, np.identity(ncvs), np.identity(ncvs), rdm_ca, optimize = einsum_type)
    sigma_cca += 1/2 * einsum('IJX,XxyX,II,JJ,xy->IJX', X, v_aaaa, np.identity(ncvs), np.identity(ncvs), rdm_ca, optimize = einsum_type)
    sigma_cca += 1/2 * einsum('IJX,Xxyz,II,JJ,Xxyz->IJX', X, v_aaaa, np.identity(ncvs), np.identity(ncvs), rdm_ccaa, optimize = einsum_type)
    print(">>> SA sigma CCA: {:}".format(np.linalg.norm(sigma_cca)))
    sigma[s_cca:f_cca] += sigma_cca.reshape(-1).copy()

    # if nval > 0:
    #     # CVA <- CVA
    #     X = Xt[s_cva:f_cva].reshape(ncvs, nval, ncas).copy()

    #     sigma_cva =- einsum('IJX,XX,II,JJ->IJX', X, h_aa, np.identity(ncvs), np.identity(nval), optimize = einsum_type)
    #     sigma_cva += einsum('IJX,I,II,JJ,XX->IJX', X, e_cvs, np.identity(ncvs), np.identity(nval), np.identity(ncas), optimize = einsum_type)
    #     sigma_cva += einsum('IJX,J,II,JJ,XX->IJX', X, e_val, np.identity(ncvs), np.identity(nval), np.identity(ncas), optimize = einsum_type)
    #     sigma_cva -= 1/2 * einsum('IJX,I,II,JJ,XX->IJX', X, e_cvs, np.identity(ncvs), np.identity(nval), rdm_ca, optimize = einsum_type)
    #     sigma_cva -= 1/2 * einsum('IJX,J,II,JJ,XX->IJX', X, e_val, np.identity(ncvs), np.identity(nval), rdm_ca, optimize = einsum_type)
    #     sigma_cva += 1/2 * einsum('IJX,Xx,II,JJ,Xx->IJX', X, h_aa, np.identity(ncvs), np.identity(nval), rdm_ca, optimize = einsum_type)
    #     sigma_cva -= einsum('IJX,XxXy,II,JJ,xy->IJX', X, v_aaaa, np.identity(ncvs), np.identity(nval), rdm_ca, optimize = einsum_type)
    #     sigma_cva += 1/2 * einsum('IJX,XxyX,II,JJ,xy->IJX', X, v_aaaa, np.identity(ncvs), np.identity(nval), rdm_ca, optimize = einsum_type)
    #     sigma_cva += 1/2 * einsum('IJX,Xxyz,II,JJ,Xxyz->IJX', X, v_aaaa, np.identity(ncvs), np.identity(nval), rdm_ccaa, optimize = einsum_type)
    #     sigma[s_cva:f_cva] += sigma_cva.reshape(-1).copy()

    #     # VCA <- VCA
    #     X = Xt[s_vca:f_vca].reshape(nval, ncvs, ncas).copy()

    #     sigma_vca =- einsum('IJX,XX,II,JJ->IJX', X, h_aa, np.identity(nval), np.identity(ncvs), optimize = einsum_type)
    #     sigma_vca += einsum('IJX,I,II,JJ,XX->IJX', X, e_val, np.identity(nval), np.identity(ncvs), np.identity(ncas), optimize = einsum_type)
    #     sigma_vca += einsum('IJX,J,II,JJ,XX->IJX', X, e_cvs, np.identity(nval), np.identity(ncvs), np.identity(ncas), optimize = einsum_type)
    #     sigma_vca -= 1/2 * einsum('IJX,I,II,JJ,XX->IJX', X, e_val, np.identity(nval), np.identity(ncvs), rdm_ca, optimize = einsum_type)
    #     sigma_vca -= 1/2 * einsum('IJX,J,II,JJ,XX->IJX', X, e_cvs, np.identity(nval), np.identity(ncvs), rdm_ca, optimize = einsum_type)
    #     sigma_vca += 1/2 * einsum('IJX,Xx,II,JJ,Xx->IJX', X, h_aa, np.identity(nval), np.identity(ncvs), rdm_ca, optimize = einsum_type)
    #     sigma_vca -= einsum('IJX,XxXy,II,JJ,xy->IJX', X, v_aaaa, np.identity(nval), np.identity(ncvs), rdm_ca, optimize = einsum_type)
    #     sigma_vca += 1/2 * einsum('IJX,XxyX,II,JJ,xy->IJX', X, v_aaaa, np.identity(nval), np.identity(ncvs), rdm_ca, optimize = einsum_type)
    #     sigma_vca += 1/2 * einsum('IJX,Xxyz,II,JJ,Xxyz->IJX', X, v_aaaa, np.identity(nval), np.identity(ncvs), rdm_ccaa, optimize = einsum_type)
    #     sigma[s_vca:f_vca] += sigma_vca.reshape(-1).copy()

    return sigma

## CCE block (CCE + CVE + VCE)
def compute_excitation_manifolds_cce(mr_adc):

    # MR-ADC(0) and MR-ADC(1)
    mr_adc.h0.n_c = mr_adc.ncvs
    mr_adc.h0.dim = mr_adc.h0.n_c # Total dimension of h0

    mr_adc.h0.s_c = 0
    mr_adc.h0.f_c = mr_adc.h0.s_c + mr_adc.h0.n_c

    print("Dimension of h0 excitation manifold:                       %d" % mr_adc.h0.dim)

    # MR-ADC(2)
    mr_adc.h1.dim = 0
    mr_adc.h_orth.dim = mr_adc.h0.dim

    if mr_adc.method in ("mr-adc(2)", "mr-adc(2)-x"):
        mr_adc.h1.n_caa = 0
        mr_adc.h1.n_cce = mr_adc.nextern * mr_adc.ncvs * mr_adc.ncvs
        mr_adc.h1.n_cae = 0
        mr_adc.h1.n_ace = 0
        mr_adc.h1.n_cca = 0
        if mr_adc.nval > 0:
            mr_adc.h1.n_cve = mr_adc.nextern * mr_adc.ncvs * mr_adc.nval
            mr_adc.h1.n_vce = mr_adc.nextern * mr_adc.ncvs * mr_adc.nval
            mr_adc.h1.n_cva = 0
            mr_adc.h1.n_vca = 0
            mr_adc.h1.dim = (mr_adc.h1.n_caa + mr_adc.h1.n_cce + mr_adc.h1.n_cve + mr_adc.h1.n_vce +
                             mr_adc.h1.n_cae + mr_adc.h1.n_ace + mr_adc.h1.n_cca + mr_adc.h1.n_cva + mr_adc.h1.n_vca)
        else:
            mr_adc.h1.dim = mr_adc.h1.n_caa + mr_adc.h1.n_cce + mr_adc.h1.n_cae + mr_adc.h1.n_cae + mr_adc.h1.n_cca

        if mr_adc.nval > 0:
            mr_adc.h1.s_caa = mr_adc.h0.f_c
            mr_adc.h1.f_caa = mr_adc.h1.s_caa + mr_adc.h1.n_caa
            mr_adc.h1.s_cce = mr_adc.h1.f_caa
            mr_adc.h1.f_cce = mr_adc.h1.s_cce + mr_adc.h1.n_cce
            mr_adc.h1.s_cve = mr_adc.h1.f_cce
            mr_adc.h1.f_cve = mr_adc.h1.s_cve + mr_adc.h1.n_cve
            mr_adc.h1.s_vce = mr_adc.h1.f_cve
            mr_adc.h1.f_vce = mr_adc.h1.s_vce + mr_adc.h1.n_vce
            mr_adc.h1.s_cae = mr_adc.h1.f_vce
            mr_adc.h1.f_cae = mr_adc.h1.s_cae + mr_adc.h1.n_cae
            mr_adc.h1.s_ace = mr_adc.h1.f_cae
            mr_adc.h1.f_ace = mr_adc.h1.s_ace + mr_adc.h1.n_ace
            mr_adc.h1.s_cca = mr_adc.h1.f_ace
            mr_adc.h1.f_cca = mr_adc.h1.s_cca + mr_adc.h1.n_cca
            mr_adc.h1.s_cva = mr_adc.h1.f_cca
            mr_adc.h1.f_cva = mr_adc.h1.s_cva + mr_adc.h1.n_cva
            mr_adc.h1.s_vca = mr_adc.h1.f_cva
            mr_adc.h1.f_vca = mr_adc.h1.s_vca + mr_adc.h1.n_vca
        else:
            mr_adc.h1.s_caa = mr_adc.h0.f_c
            mr_adc.h1.f_caa = mr_adc.h1.s_caa + mr_adc.h1.n_caa
            mr_adc.h1.s_cce = mr_adc.h1.f_caa
            mr_adc.h1.f_cce = mr_adc.h1.s_cce + mr_adc.h1.n_cce
            mr_adc.h1.s_cae = mr_adc.h1.f_cce
            mr_adc.h1.f_cae = mr_adc.h1.s_cae + mr_adc.h1.n_cae
            mr_adc.h1.s_ace = mr_adc.h1.f_cae
            mr_adc.h1.f_ace = mr_adc.h1.s_ace + mr_adc.h1.n_ace
            mr_adc.h1.s_cca = mr_adc.h1.f_ace
            mr_adc.h1.f_cca = mr_adc.h1.s_cca + mr_adc.h1.n_cca

        print("Dimension of h1 excitation manifold:                       %d" % mr_adc.h1.dim)

        # Overlap for c - caa
        mr_adc.S12.c_caa = mr_adc_overlap.compute_S12_0p_projector(mr_adc)
        mr_adc.S12.cae = mr_adc_overlap.compute_S12_m1(mr_adc)
        mr_adc.S12.cca = mr_adc_overlap.compute_S12_p1(mr_adc)

        # Determine dimensions of orthogonalized excitation spaces
        mr_adc.h_orth.n_c = mr_adc.ncvs
        mr_adc.h_orth.n_c_caa = 0
        mr_adc.h_orth.n_cce = 0
        mr_adc.h_orth.n_cce = mr_adc.h1.n_cce
        mr_adc.h_orth.n_cae = 0
        mr_adc.h_orth.n_ace = 0
        mr_adc.h_orth.n_cca = 0
        if mr_adc.nval > 0:
            mr_adc.h_orth.n_cve = mr_adc.h1.n_cve
            mr_adc.h_orth.n_vce = mr_adc.h1.n_vce
            mr_adc.h_orth.n_cva = 0
            mr_adc.h_orth.n_vca = 0
            mr_adc.h_orth.dim = (mr_adc.h_orth.n_c + mr_adc.h_orth.n_c_caa + mr_adc.h_orth.n_cce + mr_adc.h_orth.n_cve + mr_adc.h_orth.n_vce +
                                 mr_adc.h_orth.n_cae + mr_adc.h_orth.n_ace + mr_adc.h_orth.n_cca + mr_adc.h_orth.n_cva + mr_adc.h_orth.n_vca)
        else:
            mr_adc.h_orth.dim = mr_adc.h_orth.n_c + mr_adc.h_orth.n_c_caa + mr_adc.h_orth.n_cce + mr_adc.h_orth.n_cae + mr_adc.h_orth.n_ace + mr_adc.h_orth.n_cca

        if mr_adc.nval > 0:
            mr_adc.h_orth.s_c = 0
            mr_adc.h_orth.f_c = mr_adc.h_orth.n_c
            mr_adc.h_orth.s_c_caa = mr_adc.h_orth.f_c
            mr_adc.h_orth.f_c_caa = mr_adc.h_orth.s_c_caa + mr_adc.h_orth.n_c_caa
            mr_adc.h_orth.s_cce = mr_adc.h_orth.f_c_caa
            mr_adc.h_orth.f_cce = mr_adc.h_orth.s_cce + mr_adc.h_orth.n_cce
            mr_adc.h_orth.s_cve = mr_adc.h_orth.f_cce
            mr_adc.h_orth.f_cve = mr_adc.h_orth.s_cve + mr_adc.h_orth.n_cve
            mr_adc.h_orth.s_vce = mr_adc.h_orth.f_cve
            mr_adc.h_orth.f_vce = mr_adc.h_orth.s_vce + mr_adc.h_orth.n_vce
            mr_adc.h_orth.s_cae = mr_adc.h_orth.f_vce
            mr_adc.h_orth.f_cae = mr_adc.h_orth.s_cae + mr_adc.h_orth.n_cae
            mr_adc.h_orth.s_ace = mr_adc.h_orth.f_cae
            mr_adc.h_orth.f_ace = mr_adc.h_orth.s_ace + mr_adc.h_orth.n_ace
            mr_adc.h_orth.s_cca = mr_adc.h_orth.f_ace
            mr_adc.h_orth.f_cca = mr_adc.h_orth.s_cca + mr_adc.h_orth.n_cca
            mr_adc.h_orth.s_cva = mr_adc.h_orth.f_cca
            mr_adc.h_orth.f_cva = mr_adc.h_orth.s_cva + mr_adc.h_orth.n_cva
            mr_adc.h_orth.s_vca = mr_adc.h_orth.f_cva
            mr_adc.h_orth.f_vca = mr_adc.h_orth.s_vca + mr_adc.h_orth.n_vca
        else:
            mr_adc.h_orth.s_c = 0
            mr_adc.h_orth.f_c = mr_adc.h_orth.n_c
            mr_adc.h_orth.s_c_caa = mr_adc.h_orth.f_c
            mr_adc.h_orth.f_c_caa = mr_adc.h_orth.s_c_caa + mr_adc.h_orth.n_c_caa
            mr_adc.h_orth.s_cce = mr_adc.h_orth.f_c_caa
            mr_adc.h_orth.f_cce = mr_adc.h_orth.s_cce + mr_adc.h_orth.n_cce
            mr_adc.h_orth.s_cae = mr_adc.h_orth.f_cce
            mr_adc.h_orth.f_cae = mr_adc.h_orth.s_cae + mr_adc.h_orth.n_cae
            mr_adc.h_orth.s_ace = mr_adc.h_orth.f_cae
            mr_adc.h_orth.f_ace = mr_adc.h_orth.s_ace + mr_adc.h_orth.n_ace
            mr_adc.h_orth.s_cca = mr_adc.h_orth.f_ace
            mr_adc.h_orth.f_cca = mr_adc.h_orth.s_cca + mr_adc.h_orth.n_cca

    print("Total dimension of the excitation manifold:                %d" % (mr_adc.h0.dim + mr_adc.h1.dim))
    print("Dimension of the orthogonalized excitation manifold:       %d\n" % (mr_adc.h_orth.dim))
    sys.stdout.flush()

    if (mr_adc.h_orth.dim < mr_adc.nroots):
        mr_adc.nroots = mr_adc.h_orth.dim

    return mr_adc

def compute_preconditioner_cce(mr_adc, M_00):

    start_time = time.time()

    # Einsum definition from kernel
    einsum = mr_adc.interface.einsum
    einsum_type = mr_adc.interface.einsum_type

    if mr_adc.method in ("mr-adc(0)", "mr-adc(1)"):

        # Multiply by -1.0, since we are solving for -M C = -S C E
        return (-1.0 * np.diag(M_00))

    # Variables from kernel
    ncvs = mr_adc.ncvs
    nval = mr_adc.nval
    ncas = mr_adc.ncas
    nextern = mr_adc.nextern

    e_cvs = mr_adc.mo_energy.x
    if nval > 0:
        e_val = mr_adc.mo_energy.v
    e_extern = mr_adc.mo_energy.e

    # One-electron integrals
    h_aa = mr_adc.h1eff.aa

    # Two-electrons integrals
    v_aaaa = mr_adc.v2e.aaaa

    # Reduced Density Matrices
    rdm_ca = mr_adc.rdm.ca
    rdm_ccaa = mr_adc.rdm.ccaa

    # Overlap matrices
    S12_c_caa = mr_adc.S12.c_caa
    S12_cae = mr_adc.S12.cae
    S12_cca = mr_adc.S12.cca

    # Dimensions
    ho_s_c = mr_adc.h_orth.s_c
    ho_f_c = mr_adc.h_orth.f_c
    ho_s_c_caa = mr_adc.h_orth.s_c_caa
    ho_f_c_caa = mr_adc.h_orth.f_c_caa
    ho_s_cce = mr_adc.h_orth.s_cce
    ho_f_cce = mr_adc.h_orth.f_cce
    ho_s_cae = mr_adc.h_orth.s_cae
    ho_f_cae = mr_adc.h_orth.f_cae
    ho_s_ace = mr_adc.h_orth.s_ace
    ho_f_ace = mr_adc.h_orth.f_ace
    ho_s_cca = mr_adc.h_orth.s_cca
    ho_f_cca = mr_adc.h_orth.f_cca
    if nval > 0:
        ho_s_cve = mr_adc.h_orth.s_cve
        ho_f_cve = mr_adc.h_orth.f_cve
        ho_s_vce = mr_adc.h_orth.s_vce
        ho_f_vce = mr_adc.h_orth.f_vce

        ho_s_cva = mr_adc.h_orth.s_cva
        ho_f_cva = mr_adc.h_orth.f_cva
        ho_s_vca = mr_adc.h_orth.s_vca
        ho_f_vca = mr_adc.h_orth.f_vca

    s_c = mr_adc.h0.s_c
    f_c = mr_adc.h0.f_c

    cvs_tril_ind = np.tril_indices(ncvs, k=-1)
    # cas_ind = np.tril_indices(ncas, k=-1)

    # Build the preconditioner
    precond = np.zeros(mr_adc.h_orth.dim)

    # C-C debug
    precond[ho_s_c:ho_f_c] = np.diag(M_00[s_c:f_c, s_c:f_c]).copy()

    # CCE
    precond_cce =- einsum('A,AA,II,JJ->IJA', e_extern, np.identity(nextern), np.identity(ncvs), np.identity(ncvs), optimize = einsum_type)
    precond_cce += einsum('I,AA,II,JJ->IJA', e_cvs, np.identity(nextern), np.identity(ncvs), np.identity(ncvs), optimize = einsum_type)
    precond_cce += einsum('J,AA,II,JJ->IJA', e_cvs, np.identity(nextern), np.identity(ncvs), np.identity(ncvs), optimize = einsum_type)
    precond[ho_s_cce:ho_f_cce] = precond_cce.reshape(-1).copy()

    if nval > 0:
        # CVE
        precond_cve =- einsum('A,AA,II,JJ->IJA', e_extern, np.identity(nextern), np.identity(ncvs), np.identity(nval), optimize = einsum_type)
        precond_cve += einsum('I,AA,II,JJ->IJA', e_cvs, np.identity(nextern), np.identity(ncvs), np.identity(nval), optimize = einsum_type)
        precond_cve += einsum('J,AA,II,JJ->IJA', e_val, np.identity(nextern), np.identity(ncvs), np.identity(nval), optimize = einsum_type)
        precond[ho_s_cve:ho_f_cve] = precond_cve.reshape(-1).copy()

        # VCE
        precond_vce =- einsum('A,AA,II,JJ->IJA', e_extern, np.identity(nextern), np.identity(nval), np.identity(ncvs), optimize = einsum_type)
        precond_vce += einsum('I,AA,II,JJ->IJA', e_val, np.identity(nextern), np.identity(nval), np.identity(ncvs), optimize = einsum_type)
        precond_vce += einsum('J,AA,II,JJ->IJA', e_cvs, np.identity(nextern), np.identity(nval), np.identity(ncvs), optimize = einsum_type)
        precond[ho_s_vce:ho_f_vce] = precond_vce.reshape(-1).copy()

    # Multiply by -1.0, since we are solving for -M C = -S C E
    precond *= (-1.0)

    print ("Time for computing preconditioner:                %f sec\n" % (time.time() - start_time))
    sys.stdout.flush()

    return precond

def apply_S_12_cce(mr_adc, X, transpose = False):

    # Einsum definition from kernel
    einsum = mr_adc.interface.einsum
    einsum_type = mr_adc.interface.einsum_type

    # Dimensions
    nextern = mr_adc.nextern
    ncvs = mr_adc.ncvs
    nval = mr_adc.nval

    ho_s_c = mr_adc.h_orth.s_c
    ho_f_c = mr_adc.h_orth.f_c
    ho_s_c_caa = mr_adc.h_orth.s_c_caa
    ho_f_c_caa = mr_adc.h_orth.f_c_caa
    ho_s_cce = mr_adc.h_orth.s_cce
    ho_f_cce = mr_adc.h_orth.f_cce
    ho_s_cae = mr_adc.h_orth.s_cae
    ho_f_cae = mr_adc.h_orth.f_cae
    ho_s_ace = mr_adc.h_orth.s_ace
    ho_f_ace = mr_adc.h_orth.f_ace
    ho_s_cca = mr_adc.h_orth.s_cca
    ho_f_cca = mr_adc.h_orth.f_cca
    s_c = mr_adc.h0.s_c
    f_c = mr_adc.h0.f_c
    s_caa = mr_adc.h1.s_caa
    f_caa = mr_adc.h1.f_caa
    s_cce = mr_adc.h1.s_cce
    f_cce = mr_adc.h1.f_cce
    s_cae = mr_adc.h1.s_cae
    f_cae = mr_adc.h1.f_cae
    s_ace = mr_adc.h1.s_ace
    f_ace = mr_adc.h1.f_ace
    s_cca = mr_adc.h1.s_cca
    f_cca = mr_adc.h1.f_cca

    if nval > 0:
        ho_s_cve = mr_adc.h_orth.s_cve
        ho_f_cve = mr_adc.h_orth.f_cve
        ho_s_vce = mr_adc.h_orth.s_vce
        ho_f_vce = mr_adc.h_orth.f_vce

        ho_s_cva = mr_adc.h_orth.s_cva
        ho_f_cva = mr_adc.h_orth.f_cva
        ho_s_vca = mr_adc.h_orth.s_vca
        ho_f_vca = mr_adc.h_orth.f_vca

        s_cve = mr_adc.h1.s_cve
        f_cve = mr_adc.h1.f_cve
        s_vce = mr_adc.h1.s_vce
        f_vce = mr_adc.h1.f_vce

        s_cva = mr_adc.h1.s_cva
        f_cva = mr_adc.h1.f_cva
        s_vca = mr_adc.h1.s_vca
        f_vca = mr_adc.h1.f_vca

    # Overlap matrices
    S12_c_caa = mr_adc.S12.c_caa
    S12_cae = mr_adc.S12.cae
    S12_cca = mr_adc.S12.cca

    Xt = None

    if transpose:
        if (X.shape[0] != (mr_adc.h0.dim + mr_adc.h1.dim)):
            raise Exception("Dimensions do not match when applying S_12 transpose")

        Xt = np.zeros(mr_adc.h_orth.dim)


        # C-C DEBUG
        Xt[ho_s_c:ho_f_c] = X[s_c:f_c].copy()

        # CCE
        Xt[ho_s_cce:ho_f_cce] = X[s_cce:f_cce].copy()

        if nval > 0:
            # CVE
            Xt[ho_s_cve:ho_f_cve] = X[s_cve:f_cve].copy()

            # VCE
            Xt[ho_s_vce:ho_f_vce] = X[s_vce:f_vce].copy()

    else:
        if (X.shape[0] != (mr_adc.h_orth.dim)):
            raise Exception("Dimensions do not match when applying S_12")

        Xt = np.zeros(mr_adc.h0.dim + mr_adc.h1.dim)

        # C-C DEBUG
        Xt[s_c:f_c] = X[ho_s_c:ho_f_c].copy()

        # CCE
        Xt[s_cce:f_cce] = X[ho_s_cce:ho_f_cce].copy()

        if nval > 0:
            # CVE
            Xt[s_cve:f_cve] = X[ho_s_cve:ho_f_cve].copy()

            # VCE
            Xt[s_vce:f_vce] = X[ho_s_vce:ho_f_vce].copy()

    return Xt

def compute_sigma_vector_cce(mr_adc, M_00, M_01, M_11, Xt):

    # Einsum definition from kernel
    einsum = mr_adc.interface.einsum
    einsum_type = mr_adc.interface.einsum_type

    # Variables from kernel
    ncvs = mr_adc.ncvs
    nval = mr_adc.nval
    ncas = mr_adc.ncas
    nextern = mr_adc.nextern

    e_cvs = mr_adc.mo_energy.x
    e_core = mr_adc.mo_energy.c
    if nval > 0:
        e_val = mr_adc.mo_energy.v
    e_extern = mr_adc.mo_energy.e

    # One-electron integrals
    h_aa = mr_adc.h1eff.aa

    # Two-electrons integrals
    v_aaaa = mr_adc.v2e.aaaa

    # Reduced Density Matrices
    rdm_ca = mr_adc.rdm.ca
    rdm_ccaa = mr_adc.rdm.ccaa

    # Dimensions
    s_c = mr_adc.h0.s_c
    f_c = mr_adc.h0.f_c
    s_caa = mr_adc.h1.s_caa
    f_caa = mr_adc.h1.f_caa
    s_cce = mr_adc.h1.s_cce
    f_cce = mr_adc.h1.f_cce
    s_cae = mr_adc.h1.s_cae
    f_cae = mr_adc.h1.f_cae
    s_ace = mr_adc.h1.s_ace
    f_ace = mr_adc.h1.f_ace
    s_cca = mr_adc.h1.s_cca
    f_cca = mr_adc.h1.f_cca
    if nval > 0:
        s_cve = mr_adc.h1.s_cve
        f_cve = mr_adc.h1.f_cve
        s_vce = mr_adc.h1.s_vce
        f_vce = mr_adc.h1.f_vce

        s_cva = mr_adc.h1.s_cva
        f_cva = mr_adc.h1.f_cva
        s_vca = mr_adc.h1.s_vca
        f_vca = mr_adc.h1.f_vca

    cvs_tril_ind = np.tril_indices(ncvs, k=-1)

    # (CASCI + C) -> (CASCI + C)
    sigma = np.zeros_like(Xt)

    # h0-h0 contributions
    sigma[:mr_adc.h0.dim] = np.dot(M_00, Xt[:mr_adc.h0.dim])

    # h1-h1 contributions
    # CCE <- CCE
    X = Xt[s_cce:f_cce].reshape(ncvs, ncvs, nextern).copy()

    sigma_cce =- einsum('KLB,B->KLB', X, e_extern, optimize = einsum_type)
    sigma_cce += einsum('KLB,K->KLB', X, e_cvs, optimize = einsum_type)
    sigma_cce += einsum('KLB,L->KLB', X, e_cvs, optimize = einsum_type)

    sigma[s_cce:f_cce] += sigma_cce.reshape(-1).copy()

    #### TESTING M
    M11_cce_aaa =- 1/2 * einsum('A,AB,IK,JL->KLBIJA', e_extern, np.identity(nextern), np.identity(ncvs), np.identity(ncvs), optimize = einsum_type)
    M11_cce_aaa += 1/2 * einsum('A,AB,IL,JK->KLBIJA', e_extern, np.identity(nextern), np.identity(ncvs), np.identity(ncvs), optimize = einsum_type)
    M11_cce_aaa += 1/2 * einsum('I,AB,IK,JL->KLBIJA', e_cvs, np.identity(nextern), np.identity(ncvs), np.identity(ncvs), optimize = einsum_type)
    M11_cce_aaa -= 1/2 * einsum('I,AB,IL,JK->KLBIJA', e_cvs, np.identity(nextern), np.identity(ncvs), np.identity(ncvs), optimize = einsum_type)
    M11_cce_aaa += 1/2 * einsum('J,AB,IK,JL->KLBIJA', e_cvs, np.identity(nextern), np.identity(ncvs), np.identity(ncvs), optimize = einsum_type)
    M11_cce_aaa -= 1/2 * einsum('J,AB,IL,JK->KLBIJA', e_cvs, np.identity(nextern), np.identity(ncvs), np.identity(ncvs), optimize = einsum_type)

    M11_cce_abb =- 1/2 * einsum('A,AB,IK,JL->KLBIJA', e_extern, np.identity(nextern), np.identity(ncvs), np.identity(ncvs), optimize = einsum_type)
    M11_cce_abb += 1/2 * einsum('A,AB,IL,JK->KLBIJA', e_extern, np.identity(nextern), np.identity(ncvs), np.identity(ncvs), optimize = einsum_type)
    M11_cce_abb += 1/2 * einsum('I,AB,IK,JL->KLBIJA', e_cvs, np.identity(nextern), np.identity(ncvs), np.identity(ncvs), optimize = einsum_type)
    M11_cce_abb -= 1/2 * einsum('I,AB,IL,JK->KLBIJA', e_cvs, np.identity(nextern), np.identity(ncvs), np.identity(ncvs), optimize = einsum_type)
    M11_cce_abb += 1/2 * einsum('J,AB,IK,JL->KLBIJA', e_cvs, np.identity(nextern), np.identity(ncvs), np.identity(ncvs), optimize = einsum_type)
    M11_cce_abb -= 1/2 * einsum('J,AB,IL,JK->KLBIJA', e_cvs, np.identity(nextern), np.identity(ncvs), np.identity(ncvs), optimize = einsum_type)

    M11_cce_bab =- 1/2 * einsum('A,AB,IK,JL->KLBIJA', e_extern, np.identity(nextern), np.identity(ncvs), np.identity(ncvs), optimize = einsum_type)
    M11_cce_bab += 1/2 * einsum('A,AB,IL,JK->KLBIJA', e_extern, np.identity(nextern), np.identity(ncvs), np.identity(ncvs), optimize = einsum_type)
    M11_cce_bab += 1/2 * einsum('I,AB,IK,JL->KLBIJA', e_cvs, np.identity(nextern), np.identity(ncvs), np.identity(ncvs), optimize = einsum_type)
    M11_cce_bab -= 1/2 * einsum('I,AB,IL,JK->KLBIJA', e_cvs, np.identity(nextern), np.identity(ncvs), np.identity(ncvs), optimize = einsum_type)
    M11_cce_bab += 1/2 * einsum('J,AB,IK,JL->KLBIJA', e_cvs, np.identity(nextern), np.identity(ncvs), np.identity(ncvs), optimize = einsum_type)
    M11_cce_bab -= 1/2 * einsum('J,AB,IL,JK->KLBIJA', e_cvs, np.identity(nextern), np.identity(ncvs), np.identity(ncvs), optimize = einsum_type)
    print(">>> SA M11 CCE (aaa): {:}".format(np.linalg.norm(M11_cce_aaa)))
    print(">>> SA M11 CCE (abb): {:}".format(np.linalg.norm(M11_cce_abb)))
    print(">>> SA M11 CCE (bab): {:}".format(np.linalg.norm(M11_cce_bab)))
    print(">>> SA M11 CCE (aaa) == (abb)+(bab): {:}".format(np.allclose(M11_cce_aaa, M11_cce_abb + M11_cce_bab, atol=1e-12)))

    M11_cce_cce_aaa_aaa =- 1/2 * einsum('A,AB,IK,JL->KLBIJA', e_extern, np.identity(nextern), np.identity(ncvs), np.identity(ncvs), optimize = einsum_type)
    M11_cce_cce_aaa_aaa += 1/2 * einsum('A,AB,IL,JK->KLBIJA', e_extern, np.identity(nextern), np.identity(ncvs), np.identity(ncvs), optimize = einsum_type)
    M11_cce_cce_aaa_aaa += 1/2 * einsum('I,AB,IK,JL->KLBIJA', e_cvs, np.identity(nextern), np.identity(ncvs), np.identity(ncvs), optimize = einsum_type)
    M11_cce_cce_aaa_aaa -= 1/2 * einsum('I,AB,IL,JK->KLBIJA', e_cvs, np.identity(nextern), np.identity(ncvs), np.identity(ncvs), optimize = einsum_type)
    M11_cce_cce_aaa_aaa += 1/2 * einsum('J,AB,IK,JL->KLBIJA', e_cvs, np.identity(nextern), np.identity(ncvs), np.identity(ncvs), optimize = einsum_type)
    M11_cce_cce_aaa_aaa -= 1/2 * einsum('J,AB,IL,JK->KLBIJA', e_cvs, np.identity(nextern), np.identity(ncvs), np.identity(ncvs), optimize = einsum_type)

    M11_cce_cce_bba_abb =- einsum('A,AB,IK,JL->KLBIJA', e_extern, np.identity(nextern), np.identity(ncvs), np.identity(ncvs), optimize = einsum_type)
    M11_cce_cce_bba_abb += einsum('I,AB,IK,JL->KLBIJA', e_cvs, np.identity(nextern), np.identity(ncvs), np.identity(ncvs), optimize = einsum_type)
    M11_cce_cce_bba_abb += einsum('J,AB,IK,JL->KLBIJA', e_cvs, np.identity(nextern), np.identity(ncvs), np.identity(ncvs), optimize = einsum_type)

    M11_cce_cce_bba_bab  = einsum('A,AB,IL,JK->KLBIJA', e_extern, np.identity(nextern), np.identity(ncvs), np.identity(ncvs), optimize = einsum_type)
    M11_cce_cce_bba_bab -= einsum('I,AB,IL,JK->KLBIJA', e_cvs, np.identity(nextern), np.identity(ncvs), np.identity(ncvs), optimize = einsum_type)
    M11_cce_cce_bba_bab -= einsum('J,AB,IL,JK->KLBIJA', e_cvs, np.identity(nextern), np.identity(ncvs), np.identity(ncvs), optimize = einsum_type)
    print(">>> SA M11 CCE (aaa-aaa): {:}".format(np.linalg.norm(M11_cce_cce_aaa_aaa)))
    print(">>> SA M11 CCE (bba-abb): {:}".format(np.linalg.norm(M11_cce_cce_bba_abb)))
    print(">>> SA M11 CCE (bba-bab): {:}".format(np.linalg.norm(M11_cce_cce_bba_bab)))
    # print(">>> SA M11 CCE (bba-abb) == (bba-bab): {:}".format(np.allclose(M11_cce_cce_bba_abb, M11_cce_cce_bba_bab, atol=1e-12)))
    ####

    if nval > 0:
        # CVE <- CVE
        X = Xt[s_cve:f_cve].reshape(ncvs, nval, nextern).copy()

        sigma_cve =- einsum('KLB,B->KLB', X, e_extern, optimize = einsum_type)
        sigma_cve += einsum('KLB,K->KLB', X, e_cvs, optimize = einsum_type)
        sigma_cve += einsum('KLB,L->KLB', X, e_val, optimize = einsum_type)
        sigma[s_cve:f_cve] += sigma_cve.reshape(-1).copy()

        # VCE <- VCE
        X = Xt[s_vce:f_vce].reshape(nval, ncvs, nextern).copy()

        sigma_vce =- einsum('KLB,B->KLB', X, e_extern, optimize = einsum_type)
        sigma_vce += einsum('KLB,K->KLB', X, e_val, optimize = einsum_type)
        sigma_vce += einsum('KLB,L->KLB', X, e_cvs, optimize = einsum_type)
        sigma[s_vce:f_vce] += sigma_vce.reshape(-1).copy()

    return sigma

## CCE block (CCE + CVE + VCE): New approach
def compute_excitation_manifolds_cce(mr_adc):

    # MR-ADC(0) and MR-ADC(1)
    mr_adc.h0.n_c = mr_adc.ncvs
    mr_adc.h0.dim = mr_adc.h0.n_c # Total dimension of h0

    mr_adc.h0.s_c = 0
    mr_adc.h0.f_c = mr_adc.h0.s_c + mr_adc.h0.n_c

    print("Dimension of h0 excitation manifold:                       %d" % mr_adc.h0.dim)

    # MR-ADC(2)
    mr_adc.h1.dim = 0
    mr_adc.h_orth.dim = mr_adc.h0.dim

    if mr_adc.method in ("mr-adc(2)", "mr-adc(2)-x"):
        mr_adc.h1.n_caa = 0
        mr_adc.h1.n_cce = mr_adc.ncvs * mr_adc.ncvs * mr_adc.nextern
        mr_adc.h1.n_cae = 0
        mr_adc.h1.n_ace = 0
        mr_adc.h1.n_cca = 0
        if mr_adc.nval > 0:
            mr_adc.h1.n_cve = mr_adc.nextern * mr_adc.ncvs * mr_adc.nval
            mr_adc.h1.n_vce = mr_adc.nextern * mr_adc.ncvs * mr_adc.nval
            mr_adc.h1.n_cva = 0
            mr_adc.h1.n_vca = 0
            mr_adc.h1.dim = (mr_adc.h1.n_caa + 2 * mr_adc.h1.n_cce + mr_adc.h1.n_cve + mr_adc.h1.n_vce +
                             mr_adc.h1.n_cae + mr_adc.h1.n_ace + mr_adc.h1.n_cca + mr_adc.h1.n_cva + mr_adc.h1.n_vca)
        else:
            mr_adc.h1.dim = mr_adc.h1.n_caa + 2 * mr_adc.h1.n_cce + mr_adc.h1.n_cae + mr_adc.h1.n_cae + mr_adc.h1.n_cca

        if mr_adc.nval > 0:
            mr_adc.h1.s_caa = mr_adc.h0.f_c
            mr_adc.h1.f_caa = mr_adc.h1.s_caa + mr_adc.h1.n_caa
            mr_adc.h1.s_cce = mr_adc.h1.f_caa
            mr_adc.h1.f_cce = mr_adc.h1.s_cce + 2 * mr_adc.h1.n_cce
            mr_adc.h1.s_cve = mr_adc.h1.f_cce
            mr_adc.h1.f_cve = mr_adc.h1.s_cve + mr_adc.h1.n_cve
            mr_adc.h1.s_vce = mr_adc.h1.f_cve
            mr_adc.h1.f_vce = mr_adc.h1.s_vce + mr_adc.h1.n_vce
            mr_adc.h1.s_cae = mr_adc.h1.f_vce
            mr_adc.h1.f_cae = mr_adc.h1.s_cae + mr_adc.h1.n_cae
            mr_adc.h1.s_ace = mr_adc.h1.f_cae
            mr_adc.h1.f_ace = mr_adc.h1.s_ace + mr_adc.h1.n_ace
            mr_adc.h1.s_cca = mr_adc.h1.f_ace
            mr_adc.h1.f_cca = mr_adc.h1.s_cca + mr_adc.h1.n_cca
            mr_adc.h1.s_cva = mr_adc.h1.f_cca
            mr_adc.h1.f_cva = mr_adc.h1.s_cva + mr_adc.h1.n_cva
            mr_adc.h1.s_vca = mr_adc.h1.f_cva
            mr_adc.h1.f_vca = mr_adc.h1.s_vca + mr_adc.h1.n_vca
        else:
            mr_adc.h1.s_caa = mr_adc.h0.f_c
            mr_adc.h1.f_caa = mr_adc.h1.s_caa + mr_adc.h1.n_caa
            mr_adc.h1.s_cce = mr_adc.h1.f_caa
            mr_adc.h1.f_cce = mr_adc.h1.s_cce + 2 * mr_adc.h1.n_cce
            mr_adc.h1.s_cae = mr_adc.h1.f_cce
            mr_adc.h1.f_cae = mr_adc.h1.s_cae + mr_adc.h1.n_cae
            mr_adc.h1.s_ace = mr_adc.h1.f_cae
            mr_adc.h1.f_ace = mr_adc.h1.s_ace + mr_adc.h1.n_ace
            mr_adc.h1.s_cca = mr_adc.h1.f_ace
            mr_adc.h1.f_cca = mr_adc.h1.s_cca + mr_adc.h1.n_cca

        print("Dimension of h1 excitation manifold:                       %d" % mr_adc.h1.dim)

        # Overlap for c - caa
        mr_adc.S12.c_caa = mr_adc_overlap.compute_S12_0p_projector(mr_adc)
        mr_adc.S12.cae = mr_adc_overlap.compute_S12_m1(mr_adc)
        mr_adc.S12.cca = mr_adc_overlap.compute_S12_p1(mr_adc)

        # Determine dimensions of orthogonalized excitation spaces
        mr_adc.h_orth.n_c = mr_adc.ncvs
        mr_adc.h_orth.n_c_caa = 0
        mr_adc.h_orth.n_cce = 0
        mr_adc.h_orth.n_cce = mr_adc.h1.n_cce
        mr_adc.h_orth.n_cae = 0
        mr_adc.h_orth.n_ace = 0
        mr_adc.h_orth.n_cca = 0
        if mr_adc.nval > 0:
            mr_adc.h_orth.n_cve = mr_adc.h1.n_cve
            mr_adc.h_orth.n_vce = mr_adc.h1.n_vce
            mr_adc.h_orth.n_cva = 0
            mr_adc.h_orth.n_vca = 0
            mr_adc.h_orth.dim = (mr_adc.h_orth.n_c + mr_adc.h_orth.n_c_caa + 2 * mr_adc.h_orth.n_cce + mr_adc.h_orth.n_cve + mr_adc.h_orth.n_vce +
                                 mr_adc.h_orth.n_cae + mr_adc.h_orth.n_ace + mr_adc.h_orth.n_cca + mr_adc.h_orth.n_cva + mr_adc.h_orth.n_vca)
        else:
            mr_adc.h_orth.dim = mr_adc.h_orth.n_c + mr_adc.h_orth.n_c_caa + 2 * mr_adc.h_orth.n_cce + mr_adc.h_orth.n_cae + mr_adc.h_orth.n_ace + mr_adc.h_orth.n_cca

        if mr_adc.nval > 0:
            mr_adc.h_orth.s_c = 0
            mr_adc.h_orth.f_c = mr_adc.h_orth.n_c
            mr_adc.h_orth.s_c_caa = mr_adc.h_orth.f_c
            mr_adc.h_orth.f_c_caa = mr_adc.h_orth.s_c_caa + mr_adc.h_orth.n_c_caa
            mr_adc.h_orth.s_cce = mr_adc.h_orth.f_c_caa
            mr_adc.h_orth.f_cce = mr_adc.h_orth.s_cce + 2 * mr_adc.h_orth.n_cce
            mr_adc.h_orth.s_cve = mr_adc.h_orth.f_cce
            mr_adc.h_orth.f_cve = mr_adc.h_orth.s_cve + mr_adc.h_orth.n_cve
            mr_adc.h_orth.s_vce = mr_adc.h_orth.f_cve
            mr_adc.h_orth.f_vce = mr_adc.h_orth.s_vce + mr_adc.h_orth.n_vce
            mr_adc.h_orth.s_cae = mr_adc.h_orth.f_vce
            mr_adc.h_orth.f_cae = mr_adc.h_orth.s_cae + mr_adc.h_orth.n_cae
            mr_adc.h_orth.s_ace = mr_adc.h_orth.f_cae
            mr_adc.h_orth.f_ace = mr_adc.h_orth.s_ace + mr_adc.h_orth.n_ace
            mr_adc.h_orth.s_cca = mr_adc.h_orth.f_ace
            mr_adc.h_orth.f_cca = mr_adc.h_orth.s_cca + mr_adc.h_orth.n_cca
            mr_adc.h_orth.s_cva = mr_adc.h_orth.f_cca
            mr_adc.h_orth.f_cva = mr_adc.h_orth.s_cva + mr_adc.h_orth.n_cva
            mr_adc.h_orth.s_vca = mr_adc.h_orth.f_cva
            mr_adc.h_orth.f_vca = mr_adc.h_orth.s_vca + mr_adc.h_orth.n_vca
        else:
            mr_adc.h_orth.s_c = 0
            mr_adc.h_orth.f_c = mr_adc.h_orth.n_c
            mr_adc.h_orth.s_c_caa = mr_adc.h_orth.f_c
            mr_adc.h_orth.f_c_caa = mr_adc.h_orth.s_c_caa + mr_adc.h_orth.n_c_caa
            mr_adc.h_orth.s_cce = mr_adc.h_orth.f_c_caa
            mr_adc.h_orth.f_cce = mr_adc.h_orth.s_cce + 2 * mr_adc.h_orth.n_cce
            mr_adc.h_orth.s_cae = mr_adc.h_orth.f_cce
            mr_adc.h_orth.f_cae = mr_adc.h_orth.s_cae + mr_adc.h_orth.n_cae
            mr_adc.h_orth.s_ace = mr_adc.h_orth.f_cae
            mr_adc.h_orth.f_ace = mr_adc.h_orth.s_ace + mr_adc.h_orth.n_ace
            mr_adc.h_orth.s_cca = mr_adc.h_orth.f_ace
            mr_adc.h_orth.f_cca = mr_adc.h_orth.s_cca + mr_adc.h_orth.n_cca

    print("Total dimension of the excitation manifold:                %d" % (mr_adc.h0.dim + mr_adc.h1.dim))
    print("Dimension of the orthogonalized excitation manifold:       %d\n" % (mr_adc.h_orth.dim))
    sys.stdout.flush()

    if (mr_adc.h_orth.dim < mr_adc.nroots):
        mr_adc.nroots = mr_adc.h_orth.dim

    return mr_adc

def compute_preconditioner_cce(mr_adc, M_00):

    start_time = time.time()

    # Einsum definition from kernel
    einsum = mr_adc.interface.einsum
    einsum_type = mr_adc.interface.einsum_type

    if mr_adc.method in ("mr-adc(0)", "mr-adc(1)"):

        # Multiply by -1.0, since we are solving for -M C = -S C E
        return (-1.0 * np.diag(M_00))

    # Variables from kernel
    ncvs = mr_adc.ncvs
    nval = mr_adc.nval
    ncas = mr_adc.ncas
    nextern = mr_adc.nextern

    e_cvs = mr_adc.mo_energy.x
    if nval > 0:
        e_val = mr_adc.mo_energy.v
    e_extern = mr_adc.mo_energy.e

    # One-electron integrals
    h_aa = mr_adc.h1eff.aa

    # Two-electrons integrals
    v_aaaa = mr_adc.v2e.aaaa

    # Reduced Density Matrices
    rdm_ca = mr_adc.rdm.ca
    rdm_ccaa = mr_adc.rdm.ccaa

    # Overlap matrices
    S12_c_caa = mr_adc.S12.c_caa
    S12_cae = mr_adc.S12.cae
    S12_cca = mr_adc.S12.cca

    # Dimensions
    ho_s_c = mr_adc.h_orth.s_c
    ho_f_c = mr_adc.h_orth.f_c
    ho_s_c_caa = mr_adc.h_orth.s_c_caa
    ho_f_c_caa = mr_adc.h_orth.f_c_caa
    ho_s_cce = mr_adc.h_orth.s_cce
    ho_f_cce = mr_adc.h_orth.f_cce
    ho_s_cae = mr_adc.h_orth.s_cae
    ho_f_cae = mr_adc.h_orth.f_cae
    ho_s_ace = mr_adc.h_orth.s_ace
    ho_f_ace = mr_adc.h_orth.f_ace
    ho_s_cca = mr_adc.h_orth.s_cca
    ho_f_cca = mr_adc.h_orth.f_cca
    if nval > 0:
        ho_s_cve = mr_adc.h_orth.s_cve
        ho_f_cve = mr_adc.h_orth.f_cve
        ho_s_vce = mr_adc.h_orth.s_vce
        ho_f_vce = mr_adc.h_orth.f_vce

        ho_s_cva = mr_adc.h_orth.s_cva
        ho_f_cva = mr_adc.h_orth.f_cva
        ho_s_vca = mr_adc.h_orth.s_vca
        ho_f_vca = mr_adc.h_orth.f_vca

    s_c = mr_adc.h0.s_c
    f_c = mr_adc.h0.f_c

    n_cce = mr_adc.h_orth.n_cce

    cvs_tril_ind = np.tril_indices(ncvs, k=-1)
    # cas_ind = np.tril_indices(ncas, k=-1)

    # Build the preconditioner
    precond = np.zeros(mr_adc.h_orth.dim)

    # C-C debug
    precond[ho_s_c:ho_f_c] = np.diag(M_00[s_c:f_c, s_c:f_c]).copy()

    # CCE
    precond_cce =- einsum('A,AA,II,JJ->IJA', e_extern, np.identity(nextern), np.identity(ncvs), np.identity(ncvs), optimize = einsum_type)
    precond_cce += einsum('I,AA,II,JJ->IJA', e_cvs, np.identity(nextern), np.identity(ncvs), np.identity(ncvs), optimize = einsum_type)
    precond_cce += einsum('J,AA,II,JJ->IJA', e_cvs, np.identity(nextern), np.identity(ncvs), np.identity(ncvs), optimize = einsum_type)
    precond[ho_s_cce:ho_s_cce+n_cce] = precond_cce.reshape(-1).copy()
    precond[ho_s_cce+n_cce:ho_f_cce] = precond_cce.reshape(-1).copy()

    if nval > 0:
        # CVE
        precond_cve =- einsum('A,AA,II,JJ->IJA', e_extern, np.identity(nextern), np.identity(ncvs), np.identity(nval), optimize = einsum_type)
        precond_cve += einsum('I,AA,II,JJ->IJA', e_cvs, np.identity(nextern), np.identity(ncvs), np.identity(nval), optimize = einsum_type)
        precond_cve += einsum('J,AA,II,JJ->IJA', e_val, np.identity(nextern), np.identity(ncvs), np.identity(nval), optimize = einsum_type)
        precond[ho_s_cve:ho_f_cve] = precond_cve.reshape(-1).copy()

        # VCE
        precond_vce =- einsum('A,AA,II,JJ->IJA', e_extern, np.identity(nextern), np.identity(nval), np.identity(ncvs), optimize = einsum_type)
        precond_vce += einsum('I,AA,II,JJ->IJA', e_val, np.identity(nextern), np.identity(nval), np.identity(ncvs), optimize = einsum_type)
        precond_vce += einsum('J,AA,II,JJ->IJA', e_cvs, np.identity(nextern), np.identity(nval), np.identity(ncvs), optimize = einsum_type)
        precond[ho_s_vce:ho_f_vce] = precond_vce.reshape(-1).copy()

    # Multiply by -1.0, since we are solving for -M C = -S C E
    precond *= (-1.0)

    print ("Time for computing preconditioner:                %f sec\n" % (time.time() - start_time))
    sys.stdout.flush()

    return precond

def apply_S_12_cce(mr_adc, X, transpose = False):

    # Einsum definition from kernel
    einsum = mr_adc.interface.einsum
    einsum_type = mr_adc.interface.einsum_type

    # Dimensions
    nextern = mr_adc.nextern
    ncvs = mr_adc.ncvs
    nval = mr_adc.nval

    ho_s_c = mr_adc.h_orth.s_c
    ho_f_c = mr_adc.h_orth.f_c
    ho_s_c_caa = mr_adc.h_orth.s_c_caa
    ho_f_c_caa = mr_adc.h_orth.f_c_caa
    ho_s_cce = mr_adc.h_orth.s_cce
    ho_f_cce = mr_adc.h_orth.f_cce
    ho_s_cae = mr_adc.h_orth.s_cae
    ho_f_cae = mr_adc.h_orth.f_cae
    ho_s_ace = mr_adc.h_orth.s_ace
    ho_f_ace = mr_adc.h_orth.f_ace
    ho_s_cca = mr_adc.h_orth.s_cca
    ho_f_cca = mr_adc.h_orth.f_cca
    s_c = mr_adc.h0.s_c
    f_c = mr_adc.h0.f_c
    s_caa = mr_adc.h1.s_caa
    f_caa = mr_adc.h1.f_caa
    s_cce = mr_adc.h1.s_cce
    f_cce = mr_adc.h1.f_cce
    s_cae = mr_adc.h1.s_cae
    f_cae = mr_adc.h1.f_cae
    s_ace = mr_adc.h1.s_ace
    f_ace = mr_adc.h1.f_ace
    s_cca = mr_adc.h1.s_cca
    f_cca = mr_adc.h1.f_cca

    if nval > 0:
        ho_s_cve = mr_adc.h_orth.s_cve
        ho_f_cve = mr_adc.h_orth.f_cve
        ho_s_vce = mr_adc.h_orth.s_vce
        ho_f_vce = mr_adc.h_orth.f_vce

        ho_s_cva = mr_adc.h_orth.s_cva
        ho_f_cva = mr_adc.h_orth.f_cva
        ho_s_vca = mr_adc.h_orth.s_vca
        ho_f_vca = mr_adc.h_orth.f_vca

        s_cve = mr_adc.h1.s_cve
        f_cve = mr_adc.h1.f_cve
        s_vce = mr_adc.h1.s_vce
        f_vce = mr_adc.h1.f_vce

        s_cva = mr_adc.h1.s_cva
        f_cva = mr_adc.h1.f_cva
        s_vca = mr_adc.h1.s_vca
        f_vca = mr_adc.h1.f_vca

    # Overlap matrices
    S12_c_caa = mr_adc.S12.c_caa
    S12_cae = mr_adc.S12.cae
    S12_cca = mr_adc.S12.cca

    Xt = None

    if transpose:
        if (X.shape[0] != (mr_adc.h0.dim + mr_adc.h1.dim)):
            raise Exception("Dimensions do not match when applying S_12 transpose")

        Xt = np.zeros(mr_adc.h_orth.dim)


        # C-C DEBUG
        Xt[ho_s_c:ho_f_c] = X[s_c:f_c].copy()

        # CCE
        Xt[ho_s_cce:ho_f_cce] = X[s_cce:f_cce].copy()

        if nval > 0:
            # CVE
            Xt[ho_s_cve:ho_f_cve] = X[s_cve:f_cve].copy()

            # VCE
            Xt[ho_s_vce:ho_f_vce] = X[s_vce:f_vce].copy()

    else:
        if (X.shape[0] != (mr_adc.h_orth.dim)):
            raise Exception("Dimensions do not match when applying S_12")

        Xt = np.zeros(mr_adc.h0.dim + mr_adc.h1.dim)

        # C-C DEBUG
        Xt[s_c:f_c] = X[ho_s_c:ho_f_c].copy()

        # CCE
        Xt[s_cce:f_cce] = X[ho_s_cce:ho_f_cce].copy()

        if nval > 0:
            # CVE
            Xt[s_cve:f_cve] = X[ho_s_cve:ho_f_cve].copy()

            # VCE
            Xt[s_vce:f_vce] = X[ho_s_vce:ho_f_vce].copy()

    return Xt

def compute_sigma_vector_cce(mr_adc, M_00, M_01, M_11, Xt):

    # Einsum definition from kernel
    einsum = mr_adc.interface.einsum
    einsum_type = mr_adc.interface.einsum_type

    # Variables from kernel
    ncvs = mr_adc.ncvs
    nval = mr_adc.nval
    ncas = mr_adc.ncas
    nextern = mr_adc.nextern

    e_cvs = mr_adc.mo_energy.x
    e_core = mr_adc.mo_energy.c
    if nval > 0:
        e_val = mr_adc.mo_energy.v
    e_extern = mr_adc.mo_energy.e

    # One-electron integrals
    h_aa = mr_adc.h1eff.aa

    # Two-electrons integrals
    v_aaaa = mr_adc.v2e.aaaa

    # Reduced Density Matrices
    rdm_ca = mr_adc.rdm.ca
    rdm_ccaa = mr_adc.rdm.ccaa

    # Dimensions
    s_c = mr_adc.h0.s_c
    f_c = mr_adc.h0.f_c
    s_caa = mr_adc.h1.s_caa
    f_caa = mr_adc.h1.f_caa
    s_cce = mr_adc.h1.s_cce
    f_cce = mr_adc.h1.f_cce
    s_cae = mr_adc.h1.s_cae
    f_cae = mr_adc.h1.f_cae
    s_ace = mr_adc.h1.s_ace
    f_ace = mr_adc.h1.f_ace
    s_cca = mr_adc.h1.s_cca
    f_cca = mr_adc.h1.f_cca
    if nval > 0:
        s_cve = mr_adc.h1.s_cve
        f_cve = mr_adc.h1.f_cve
        s_vce = mr_adc.h1.s_vce
        f_vce = mr_adc.h1.f_vce

        s_cva = mr_adc.h1.s_cva
        f_cva = mr_adc.h1.f_cva
        s_vca = mr_adc.h1.s_vca
        f_vca = mr_adc.h1.f_vca

    n_cce = mr_adc.h1.n_cce

    cvs_tril_ind = np.tril_indices(ncvs, k=-1)

    # (CASCI + C) -> (CASCI + C)
    sigma = np.zeros_like(Xt)

    # h0-h0 contributions
    sigma[:mr_adc.h0.dim] = np.dot(M_00, Xt[:mr_adc.h0.dim])

    # h1-h1 contributions
    # CCE <- CCE
    X_aaa = Xt[s_cce:s_cce+n_cce].reshape(ncvs, ncvs, nextern).copy()
    X_abb = Xt[s_cce+n_cce:f_cce].reshape(ncvs, ncvs, nextern).copy()
    X_bab =- X_abb.transpose(1,0,2)

    sigma_cce =- 1/2 * einsum('KLB,B->KLB', X_aaa, e_extern, optimize = einsum_type)
    sigma_cce += 1/2 * einsum('KLB,K->KLB', X_aaa, e_cvs, optimize = einsum_type)
    sigma_cce += 1/2 * einsum('KLB,L->KLB', X_aaa, e_cvs, optimize = einsum_type)
    sigma_cce += 1/2 * einsum('LKB,B->KLB', X_aaa, e_extern, optimize = einsum_type)
    sigma_cce -= 1/2 * einsum('LKB,K->KLB', X_aaa, e_cvs, optimize = einsum_type)
    sigma_cce -= 1/2 * einsum('LKB,L->KLB', X_aaa, e_cvs, optimize = einsum_type)
    sigma[s_cce:s_cce+n_cce] += sigma_cce.reshape(-1).copy()

    sigma_cce =- 1/2 * einsum('KLB,B->KLB', X_abb, e_extern, optimize = einsum_type)
    sigma_cce += 1/2 * einsum('KLB,K->KLB', X_abb, e_cvs, optimize = einsum_type)
    sigma_cce += 1/2 * einsum('KLB,L->KLB', X_abb, e_cvs, optimize = einsum_type)
    sigma_cce += 1/2 * einsum('LKB,B->KLB', X_bab, e_extern, optimize = einsum_type)
    sigma_cce -= 1/2 * einsum('LKB,K->KLB', X_bab, e_cvs, optimize = einsum_type)
    sigma_cce -= 1/2 * einsum('LKB,L->KLB', X_bab, e_cvs, optimize = einsum_type)
    sigma[s_cce+n_cce:f_cce] += sigma_cce.reshape(-1).copy()

    if nval > 0:
        # CVE <- CVE
        X = Xt[s_cve:f_cve].reshape(ncvs, nval, nextern).copy()

        sigma_cve =- einsum('KLB,B->KLB', X, e_extern, optimize = einsum_type)
        sigma_cve += einsum('KLB,K->KLB', X, e_cvs, optimize = einsum_type)
        sigma_cve += einsum('KLB,L->KLB', X, e_val, optimize = einsum_type)
        sigma[s_cve:f_cve] += sigma_cve.reshape(-1).copy()

        # VCE <- VCE
        X = Xt[s_vce:f_vce].reshape(nval, ncvs, nextern).copy()

        sigma_vce =- einsum('KLB,B->KLB', X, e_extern, optimize = einsum_type)
        sigma_vce += einsum('KLB,K->KLB', X, e_val, optimize = einsum_type)
        sigma_vce += einsum('KLB,L->KLB', X, e_cvs, optimize = einsum_type)
        sigma[s_vce:f_vce] += sigma_vce.reshape(-1).copy()

    return sigma

## CCA block (CCA + CVA + VCA)
def compute_excitation_manifolds_cca(mr_adc):

    # MR-ADC(0) and MR-ADC(1)
    mr_adc.h0.n_c = mr_adc.ncvs
    mr_adc.h0.dim = mr_adc.h0.n_c # Total dimension of h0

    mr_adc.h0.s_c = 0
    mr_adc.h0.f_c = mr_adc.h0.s_c + mr_adc.h0.n_c

    print("Dimension of h0 excitation manifold:                       %d" % mr_adc.h0.dim)

    # MR-ADC(2)
    mr_adc.h1.dim = 0
    mr_adc.h_orth.dim = mr_adc.h0.dim

    if mr_adc.method in ("mr-adc(2)", "mr-adc(2)-x"):
        mr_adc.h1.n_caa = 0
        mr_adc.h1.n_cce = 0
        mr_adc.h1.n_cae = 0
        mr_adc.h1.n_ace = 0
        mr_adc.h1.n_cca = mr_adc.ncas * mr_adc.ncvs * mr_adc.ncvs
        if mr_adc.nval > 0:
            mr_adc.h1.n_cve = 0
            mr_adc.h1.n_vce = 0
            mr_adc.h1.n_cva = mr_adc.ncas * mr_adc.ncvs * mr_adc.nval
            mr_adc.h1.n_vca = mr_adc.ncas * mr_adc.ncvs * mr_adc.nval
            mr_adc.h1.dim = (mr_adc.h1.n_caa + mr_adc.h1.n_cce + mr_adc.h1.n_cve + mr_adc.h1.n_vce +
                             mr_adc.h1.n_cae + mr_adc.h1.n_ace + mr_adc.h1.n_cca + mr_adc.h1.n_cva + mr_adc.h1.n_vca)
        else:
            mr_adc.h1.dim = mr_adc.h1.n_caa + mr_adc.h1.n_cce + mr_adc.h1.n_cae + mr_adc.h1.n_cae + mr_adc.h1.n_cca

        if mr_adc.nval > 0:
            mr_adc.h1.s_caa = mr_adc.h0.f_c
            mr_adc.h1.f_caa = mr_adc.h1.s_caa + mr_adc.h1.n_caa
            mr_adc.h1.s_cce = mr_adc.h1.f_caa
            mr_adc.h1.f_cce = mr_adc.h1.s_cce + mr_adc.h1.n_cce
            mr_adc.h1.s_cve = mr_adc.h1.f_cce
            mr_adc.h1.f_cve = mr_adc.h1.s_cve + mr_adc.h1.n_cve
            mr_adc.h1.s_vce = mr_adc.h1.f_cve
            mr_adc.h1.f_vce = mr_adc.h1.s_vce + mr_adc.h1.n_vce
            mr_adc.h1.s_cae = mr_adc.h1.f_vce
            mr_adc.h1.f_cae = mr_adc.h1.s_cae + mr_adc.h1.n_cae
            mr_adc.h1.s_ace = mr_adc.h1.f_cae
            mr_adc.h1.f_ace = mr_adc.h1.s_ace + mr_adc.h1.n_ace
            mr_adc.h1.s_cca = mr_adc.h1.f_ace
            mr_adc.h1.f_cca = mr_adc.h1.s_cca + mr_adc.h1.n_cca
            mr_adc.h1.s_cva = mr_adc.h1.f_cca
            mr_adc.h1.f_cva = mr_adc.h1.s_cva + mr_adc.h1.n_cva
            mr_adc.h1.s_vca = mr_adc.h1.f_cva
            mr_adc.h1.f_vca = mr_adc.h1.s_vca + mr_adc.h1.n_vca
        else:
            mr_adc.h1.s_caa = mr_adc.h0.f_c
            mr_adc.h1.f_caa = mr_adc.h1.s_caa + mr_adc.h1.n_caa
            mr_adc.h1.s_cce = mr_adc.h1.f_caa
            mr_adc.h1.f_cce = mr_adc.h1.s_cce + mr_adc.h1.n_cce
            mr_adc.h1.s_cae = mr_adc.h1.f_cce
            mr_adc.h1.f_cae = mr_adc.h1.s_cae + mr_adc.h1.n_cae
            mr_adc.h1.s_ace = mr_adc.h1.f_cae
            mr_adc.h1.f_ace = mr_adc.h1.s_ace + mr_adc.h1.n_ace
            mr_adc.h1.s_cca = mr_adc.h1.f_ace
            mr_adc.h1.f_cca = mr_adc.h1.s_cca + mr_adc.h1.n_cca

        print("Dimension of h1 excitation manifold:                       %d" % mr_adc.h1.dim)

        # Overlap for c - caa
        mr_adc.S12.c_caa = mr_adc_overlap.compute_S12_0p_projector(mr_adc)
        mr_adc.S12.cae = mr_adc_overlap.compute_S12_m1(mr_adc)
        mr_adc.S12.cca = mr_adc_overlap.compute_S12_p1(mr_adc)

        # Determine dimensions of orthogonalized excitation spaces
        mr_adc.h_orth.n_c = mr_adc.ncvs
        mr_adc.h_orth.n_c_caa = 0
        mr_adc.h_orth.n_cce = 0
        mr_adc.h_orth.n_cce = 0
        mr_adc.h_orth.n_cae = 0
        mr_adc.h_orth.n_ace = 0
        mr_adc.h_orth.n_cca = mr_adc.S12.cca.shape[1] * mr_adc.ncvs * mr_adc.ncvs
        if mr_adc.nval > 0:
            mr_adc.h_orth.n_cve = 0
            mr_adc.h_orth.n_vce = 0
            mr_adc.h_orth.n_cva = mr_adc.S12.cca.shape[1] * mr_adc.ncvs * mr_adc.nval
            mr_adc.h_orth.n_vca = mr_adc.S12.cca.shape[1] * mr_adc.ncvs * mr_adc.nval
            mr_adc.h_orth.dim = (mr_adc.h_orth.n_c + mr_adc.h_orth.n_c_caa + mr_adc.h_orth.n_cce + mr_adc.h_orth.n_cve + mr_adc.h_orth.n_vce +
                                 mr_adc.h_orth.n_cae + mr_adc.h_orth.n_ace + mr_adc.h_orth.n_cca + mr_adc.h_orth.n_cva + mr_adc.h_orth.n_vca)
        else:
            mr_adc.h_orth.dim = mr_adc.h_orth.n_c + mr_adc.h_orth.n_c_caa + mr_adc.h_orth.n_cce + mr_adc.h_orth.n_cae + mr_adc.h_orth.n_ace + mr_adc.h_orth.n_cca

        if mr_adc.nval > 0:
            mr_adc.h_orth.s_c = 0
            mr_adc.h_orth.f_c = mr_adc.h_orth.n_c
            mr_adc.h_orth.s_c_caa = mr_adc.h_orth.f_c
            mr_adc.h_orth.f_c_caa = mr_adc.h_orth.s_c_caa + mr_adc.h_orth.n_c_caa
            mr_adc.h_orth.s_cce = mr_adc.h_orth.f_c_caa
            mr_adc.h_orth.f_cce = mr_adc.h_orth.s_cce + mr_adc.h_orth.n_cce
            mr_adc.h_orth.s_cve = mr_adc.h_orth.f_cce
            mr_adc.h_orth.f_cve = mr_adc.h_orth.s_cve + mr_adc.h_orth.n_cve
            mr_adc.h_orth.s_vce = mr_adc.h_orth.f_cve
            mr_adc.h_orth.f_vce = mr_adc.h_orth.s_vce + mr_adc.h_orth.n_vce
            mr_adc.h_orth.s_cae = mr_adc.h_orth.f_vce
            mr_adc.h_orth.f_cae = mr_adc.h_orth.s_cae + mr_adc.h_orth.n_cae
            mr_adc.h_orth.s_ace = mr_adc.h_orth.f_cae
            mr_adc.h_orth.f_ace = mr_adc.h_orth.s_ace + mr_adc.h_orth.n_ace
            mr_adc.h_orth.s_cca = mr_adc.h_orth.f_ace
            mr_adc.h_orth.f_cca = mr_adc.h_orth.s_cca + mr_adc.h_orth.n_cca
            mr_adc.h_orth.s_cva = mr_adc.h_orth.f_cca
            mr_adc.h_orth.f_cva = mr_adc.h_orth.s_cva + mr_adc.h_orth.n_cva
            mr_adc.h_orth.s_vca = mr_adc.h_orth.f_cva
            mr_adc.h_orth.f_vca = mr_adc.h_orth.s_vca + mr_adc.h_orth.n_vca
        else:
            mr_adc.h_orth.s_c = 0
            mr_adc.h_orth.f_c = mr_adc.h_orth.n_c
            mr_adc.h_orth.s_cce = mr_adc.h_orth.f_c_caa
            mr_adc.h_orth.f_cce = mr_adc.h_orth.s_cce + mr_adc.h_orth.n_cce
            mr_adc.h_orth.s_cae = mr_adc.h_orth.f_cce
            mr_adc.h_orth.f_cae = mr_adc.h_orth.s_cae + mr_adc.h_orth.n_cae
            mr_adc.h_orth.s_ace = mr_adc.h_orth.f_ace
            mr_adc.h_orth.f_ace = mr_adc.h_orth.s_ace + mr_adc.h_orth.n_ace
            mr_adc.h_orth.s_cca = mr_adc.h_orth.f_ace
            mr_adc.h_orth.f_cca = mr_adc.h_orth.s_cca + mr_adc.h_orth.n_cca

    print("Total dimension of the excitation manifold:                %d" % (mr_adc.h0.dim + mr_adc.h1.dim))
    print("Dimension of the orthogonalized excitation manifold:       %d\n" % (mr_adc.h_orth.dim))
    sys.stdout.flush()

    if (mr_adc.h_orth.dim < mr_adc.nroots):
        mr_adc.nroots = mr_adc.h_orth.dim

    return mr_adc

def compute_preconditioner_cca(mr_adc, M_00):

    start_time = time.time()

    # Einsum definition from kernel
    einsum = mr_adc.interface.einsum
    einsum_type = mr_adc.interface.einsum_type

    if mr_adc.method in ("mr-adc(0)", "mr-adc(1)"):

        # Multiply by -1.0, since we are solving for -M C = -S C E
        return (-1.0 * np.diag(M_00))

    # Variables from kernel
    ncvs = mr_adc.ncvs
    nval = mr_adc.nval
    ncas = mr_adc.ncas
    nextern = mr_adc.nextern

    e_cvs = mr_adc.mo_energy.x
    e_val = mr_adc.mo_energy.v
    e_extern = mr_adc.mo_energy.e

    # One-electron integrals
    h_aa = mr_adc.h1eff.aa

    # Two-electrons integrals
    v_aaaa = mr_adc.v2e.aaaa

    # Reduced Density Matrices
    rdm_ca = mr_adc.rdm.ca
    rdm_ccaa = mr_adc.rdm.ccaa

    # Overlap matrices
    S12_c_caa = mr_adc.S12.c_caa
    S12_cae = mr_adc.S12.cae
    S12_cca = mr_adc.S12.cca

    # Dimensions
    ho_s_c = mr_adc.h_orth.s_c
    ho_f_c = mr_adc.h_orth.f_c
    ho_s_c_caa = mr_adc.h_orth.s_c_caa
    ho_f_c_caa = mr_adc.h_orth.f_c_caa
    ho_s_cce = mr_adc.h_orth.s_cce
    ho_f_cce = mr_adc.h_orth.f_cce
    ho_s_cae = mr_adc.h_orth.s_cae
    ho_f_cae = mr_adc.h_orth.f_cae
    ho_s_ace = mr_adc.h_orth.s_ace
    ho_f_ace = mr_adc.h_orth.f_ace
    ho_s_cca = mr_adc.h_orth.s_cca
    ho_f_cca = mr_adc.h_orth.f_cca
    if nval > 0:
        ho_s_cve = mr_adc.h_orth.s_cve
        ho_f_cve = mr_adc.h_orth.f_cve
        ho_s_vce = mr_adc.h_orth.s_vce
        ho_f_vce = mr_adc.h_orth.f_vce

        ho_s_cva = mr_adc.h_orth.s_cva
        ho_f_cva = mr_adc.h_orth.f_cva
        ho_s_vca = mr_adc.h_orth.s_vca
        ho_f_vca = mr_adc.h_orth.f_vca

    s_c = mr_adc.h0.s_c
    f_c = mr_adc.h0.f_c

    cvs_tril_ind = np.tril_indices(ncvs, k=-1)
    # cas_ind = np.tril_indices(ncas, k=-1)

    # Build the preconditioner
    precond = np.zeros(mr_adc.h_orth.dim)

    # C-C debug
    precond[ho_s_c:ho_f_c] = np.diag(M_00[s_c:f_c, s_c:f_c]).copy()

    # CCA
    precond_cca =- einsum('XY,II,JJ->IJXY', h_aa, np.identity(ncvs), np.identity(ncvs), optimize = einsum_type)
    precond_cca += einsum('I,II,JJ,XY->IJXY', e_cvs, np.identity(ncvs), np.identity(ncvs), np.identity(ncas), optimize = einsum_type)
    precond_cca += einsum('J,II,JJ,XY->IJXY', e_cvs, np.identity(ncvs), np.identity(ncvs), np.identity(ncas), optimize = einsum_type)
    precond_cca -= 1/2 * einsum('I,II,JJ,YX->IJXY', e_cvs, np.identity(ncvs), np.identity(ncvs), rdm_ca, optimize = einsum_type)
    precond_cca -= 1/2 * einsum('J,II,JJ,YX->IJXY', e_cvs, np.identity(ncvs), np.identity(ncvs), rdm_ca, optimize = einsum_type)
    precond_cca += 1/2 * einsum('Xx,II,JJ,Yx->IJXY', h_aa, np.identity(ncvs), np.identity(ncvs), rdm_ca, optimize = einsum_type)
    precond_cca -= einsum('XxYy,II,JJ,xy->IJXY', v_aaaa, np.identity(ncvs), np.identity(ncvs), rdm_ca, optimize = einsum_type)
    precond_cca += 1/2 * einsum('XxyY,II,JJ,xy->IJXY', v_aaaa, np.identity(ncvs), np.identity(ncvs), rdm_ca, optimize = einsum_type)
    precond_cca += 1/2 * einsum('Xxyz,II,JJ,Yxyz->IJXY', v_aaaa, np.identity(ncvs), np.identity(ncvs), rdm_ccaa, optimize = einsum_type)

    precond_cca = einsum("IJXY,XP,YP->IJP", precond_cca, S12_cca, S12_cca, optimize = einsum_type)
    precond[ho_s_cca:ho_f_cca] = precond_cca.reshape(-1).copy()

    if nval > 0:
        # CVA
        precond_cva =- einsum('XY,II,JJ->IJXY', h_aa, np.identity(ncvs), np.identity(nval), optimize = einsum_type)
        precond_cva += einsum('I,II,JJ,XY->IJXY', e_cvs, np.identity(ncvs), np.identity(nval), np.identity(ncas), optimize = einsum_type)
        precond_cva += einsum('J,II,JJ,XY->IJXY', e_val, np.identity(ncvs), np.identity(nval), np.identity(ncas), optimize = einsum_type)
        precond_cva -= 1/2 * einsum('I,II,JJ,YX->IJXY', e_cvs, np.identity(ncvs), np.identity(nval), rdm_ca, optimize = einsum_type)
        precond_cva -= 1/2 * einsum('J,II,JJ,YX->IJXY', e_val, np.identity(ncvs), np.identity(nval), rdm_ca, optimize = einsum_type)
        precond_cva += 1/2 * einsum('Xx,II,JJ,Yx->IJXY', h_aa, np.identity(ncvs), np.identity(nval), rdm_ca, optimize = einsum_type)
        precond_cva -= einsum('XxYy,II,JJ,xy->IJXY', v_aaaa, np.identity(ncvs), np.identity(nval), rdm_ca, optimize = einsum_type)
        precond_cva += 1/2 * einsum('XxyY,II,JJ,xy->IJXY', v_aaaa, np.identity(ncvs), np.identity(nval), rdm_ca, optimize = einsum_type)
        precond_cva += 1/2 * einsum('Xxyz,II,JJ,Yxyz->IJXY', v_aaaa, np.identity(ncvs), np.identity(nval), rdm_ccaa, optimize = einsum_type)
        precond_cva = einsum("IJXY,XP,YP->IJP", precond_cva, S12_cca, S12_cca, optimize = einsum_type)
        precond[ho_s_cva:ho_f_cva] = precond_cva.reshape(-1).copy()

        precond_vca =- einsum('XY,II,JJ->IJXY', h_aa, np.identity(nval), np.identity(ncvs), optimize = einsum_type)
        precond_vca += einsum('I,II,JJ,XY->IJXY', e_val, np.identity(nval), np.identity(ncvs), np.identity(ncas), optimize = einsum_type)
        precond_vca += einsum('J,II,JJ,XY->IJXY', e_cvs, np.identity(nval), np.identity(ncvs), np.identity(ncas), optimize = einsum_type)
        precond_vca -= 1/2 * einsum('I,II,JJ,YX->IJXY', e_val, np.identity(nval), np.identity(ncvs), rdm_ca, optimize = einsum_type)
        precond_vca -= 1/2 * einsum('J,II,JJ,YX->IJXY', e_cvs, np.identity(nval), np.identity(ncvs), rdm_ca, optimize = einsum_type)
        precond_vca += 1/2 * einsum('Xx,II,JJ,Yx->IJXY', h_aa, np.identity(nval), np.identity(ncvs), rdm_ca, optimize = einsum_type)
        precond_vca -= einsum('XxYy,II,JJ,xy->IJXY', v_aaaa, np.identity(nval), np.identity(ncvs), rdm_ca, optimize = einsum_type)
        precond_vca += 1/2 * einsum('XxyY,II,JJ,xy->IJXY', v_aaaa, np.identity(nval), np.identity(ncvs), rdm_ca, optimize = einsum_type)
        precond_vca += 1/2 * einsum('Xxyz,II,JJ,Yxyz->IJXY', v_aaaa, np.identity(nval), np.identity(ncvs), rdm_ccaa, optimize = einsum_type)
        precond_vca = einsum("IJXY,XP,YP->IJP", precond_vca, S12_cca, S12_cca, optimize = einsum_type)
        precond[ho_s_vca:ho_f_vca] = precond_vca.reshape(-1).copy()

    # Multiply by -1.0, since we are solving for -M C = -S C E
    precond *= (-1.0)

    print ("Time for computing preconditioner:                %f sec\n" % (time.time() - start_time))
    sys.stdout.flush()

    return precond

def apply_S_12_cca(mr_adc, X, transpose = False):

    # Einsum definition from kernel
    einsum = mr_adc.interface.einsum
    einsum_type = mr_adc.interface.einsum_type

    # Dimensions
    nextern = mr_adc.nextern
    ncvs = mr_adc.ncvs
    nval = mr_adc.nval

    ho_s_c = mr_adc.h_orth.s_c
    ho_f_c = mr_adc.h_orth.f_c
    ho_s_c_caa = mr_adc.h_orth.s_c_caa
    ho_f_c_caa = mr_adc.h_orth.f_c_caa
    ho_s_cce = mr_adc.h_orth.s_cce
    ho_f_cce = mr_adc.h_orth.f_cce
    ho_s_cae = mr_adc.h_orth.s_cae
    ho_f_cae = mr_adc.h_orth.f_cae
    ho_s_ace = mr_adc.h_orth.s_ace
    ho_f_ace = mr_adc.h_orth.f_ace
    ho_s_cca = mr_adc.h_orth.s_cca
    ho_f_cca = mr_adc.h_orth.f_cca
    s_c = mr_adc.h0.s_c
    f_c = mr_adc.h0.f_c
    s_caa = mr_adc.h1.s_caa
    f_caa = mr_adc.h1.f_caa
    s_cce = mr_adc.h1.s_cce
    f_cce = mr_adc.h1.f_cce
    s_cae = mr_adc.h1.s_cae
    f_cae = mr_adc.h1.f_cae
    s_ace = mr_adc.h1.s_ace
    f_ace = mr_adc.h1.f_ace
    s_cca = mr_adc.h1.s_cca
    f_cca = mr_adc.h1.f_cca

    if nval > 0:
        ho_s_cve = mr_adc.h_orth.s_cve
        ho_f_cve = mr_adc.h_orth.f_cve
        ho_s_vce = mr_adc.h_orth.s_vce
        ho_f_vce = mr_adc.h_orth.f_vce

        ho_s_cva = mr_adc.h_orth.s_cva
        ho_f_cva = mr_adc.h_orth.f_cva
        ho_s_vca = mr_adc.h_orth.s_vca
        ho_f_vca = mr_adc.h_orth.f_vca

        s_cve = mr_adc.h1.s_cve
        f_cve = mr_adc.h1.f_cve
        s_vce = mr_adc.h1.s_vce
        f_vce = mr_adc.h1.f_vce

        s_cva = mr_adc.h1.s_cva
        f_cva = mr_adc.h1.f_cva
        s_vca = mr_adc.h1.s_vca
        f_vca = mr_adc.h1.f_vca

    # Overlap matrices
    S12_c_caa = mr_adc.S12.c_caa
    S12_cae = mr_adc.S12.cae
    S12_cca = mr_adc.S12.cca

    Xt = None

    if transpose:
        if (X.shape[0] != (mr_adc.h0.dim + mr_adc.h1.dim)):
            raise Exception("Dimensions do not match when applying S_12 transpose")

        Xt = np.zeros(mr_adc.h_orth.dim)


        # C-C DEBUG
        Xt[ho_s_c:ho_f_c] = X[s_c:f_c].copy()

        # CCA
        n_cc = ncvs * ncvs
        temp = X[s_cca:f_cca].reshape(n_cc, S12_cca.shape[0]).copy()
        Xt[ho_s_cca:ho_f_cca] = einsum("IX,XP->IP", temp, S12_cca).reshape(-1).copy()

        if nval > 0:
            # CVA
            n_cv = ncvs * nval
            temp = X[s_cva:f_cva].reshape(n_cv, S12_cca.shape[0]).copy()
            Xt[ho_s_cva:ho_f_cva] = einsum("IX,XP->IP", temp, S12_cca).reshape(-1).copy()

            # VCA
            temp = X[s_vca:f_vca].reshape(n_cv, S12_cca.shape[0]).copy()
            Xt[ho_s_vca:ho_f_vca] = einsum("IX,XP->IP", temp, S12_cca).reshape(-1).copy()

    else:
        if (X.shape[0] != (mr_adc.h_orth.dim)):
            raise Exception("Dimensions do not match when applying S_12")

        Xt = np.zeros(mr_adc.h0.dim + mr_adc.h1.dim)

        # C-C DEBUG
        Xt[s_c:f_c] = X[ho_s_c:ho_f_c].copy()

        # CCA
        n_cc = ncvs * ncvs
        temp = X[ho_s_cca:ho_f_cca].reshape(n_cc, S12_cca.shape[1]).copy()
        Xt[s_cca:f_cca] = einsum("IP,XP->IX", temp, S12_cca).reshape(-1).copy()

        if nval > 0:
            # CVA
            n_cv = ncvs * nval
            temp = X[ho_s_cva:ho_f_cva].reshape(n_cv, S12_cca.shape[1]).copy()
            Xt[s_cva:f_cva] = einsum("IP,XP->IX", temp, S12_cca).reshape(-1).copy()

            # VCA
            temp = X[ho_s_vca:ho_f_vca].reshape(n_cv, S12_cca.shape[1]).copy()
            Xt[s_vca:f_vca] = einsum("IP,XP->IX", temp, S12_cca).reshape(-1).copy()

    return Xt

def compute_sigma_vector_cca(mr_adc, M_00, M_01, M_11, Xt):

    # Einsum definition from kernel
    einsum = mr_adc.interface.einsum
    einsum_type = mr_adc.interface.einsum_type

    # Variables from kernel
    ncvs = mr_adc.ncvs
    nval = mr_adc.nval
    ncas = mr_adc.ncas
    nextern = mr_adc.nextern

    e_cvs = mr_adc.mo_energy.x
    e_core = mr_adc.mo_energy.c
    e_val = mr_adc.mo_energy.v
    e_extern = mr_adc.mo_energy.e

    # One-electron integrals
    h_aa = mr_adc.h1eff.aa

    # Two-electrons integrals
    v_aaaa = mr_adc.v2e.aaaa

    # Reduced Density Matrices
    rdm_ca = mr_adc.rdm.ca
    rdm_ccaa = mr_adc.rdm.ccaa

    # Dimensions
    s_c = mr_adc.h0.s_c
    f_c = mr_adc.h0.f_c
    s_caa = mr_adc.h1.s_caa
    f_caa = mr_adc.h1.f_caa
    s_cce = mr_adc.h1.s_cce
    f_cce = mr_adc.h1.f_cce
    s_cae = mr_adc.h1.s_cae
    f_cae = mr_adc.h1.f_cae
    s_ace = mr_adc.h1.s_ace
    f_ace = mr_adc.h1.f_ace
    s_cca = mr_adc.h1.s_cca
    f_cca = mr_adc.h1.f_cca
    if nval > 0:
        s_cve = mr_adc.h1.s_cve
        f_cve = mr_adc.h1.f_cve
        s_vce = mr_adc.h1.s_vce
        f_vce = mr_adc.h1.f_vce

        s_cva = mr_adc.h1.s_cva
        f_cva = mr_adc.h1.f_cva
        s_vca = mr_adc.h1.s_vca
        f_vca = mr_adc.h1.f_vca

    cvs_tril_ind = np.tril_indices(ncvs, k=-1)

    # (CASCI + C) -> (CASCI + C)
    sigma = np.zeros_like(Xt)

    # h0-h0 contributions
    sigma[:mr_adc.h0.dim] = np.dot(M_00, Xt[:mr_adc.h0.dim])

    # h1-h1 contributions
    # CCA <- CCA
    X = Xt[s_cca:f_cca].reshape(ncvs, ncvs, ncas).copy()

    sigma_cca  = einsum('KLY,K->KLY', X, e_cvs, optimize = einsum_type)
    sigma_cca += einsum('KLY,L->KLY', X, e_cvs, optimize = einsum_type)
    sigma_cca -= einsum('KLx,Yx->KLY', X, h_aa, optimize = einsum_type)
    sigma_cca -= 1/2 * einsum('KLx,K,Yx->KLY', X, e_cvs, rdm_ca, optimize = einsum_type)
    sigma_cca -= 1/2 * einsum('KLx,L,Yx->KLY', X, e_cvs, rdm_ca, optimize = einsum_type)
    sigma_cca += 1/2 * einsum('KLx,xy,Yy->KLY', X, h_aa, rdm_ca, optimize = einsum_type)
    sigma_cca -= einsum('KLx,Yyxz,zy->KLY', X, v_aaaa, rdm_ca, optimize = einsum_type)
    sigma_cca += 1/2 * einsum('KLx,Yyzx,zy->KLY', X, v_aaaa, rdm_ca, optimize = einsum_type)
    sigma_cca += 1/2 * einsum('KLx,xyzw,Yyzw->KLY', X, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma[s_cca:f_cca] += sigma_cca.reshape(-1).copy()

    if nval > 0:
        # CVA <- CVA
        X = Xt[s_cva:f_cva].reshape(ncvs, nval, ncas).copy()

        sigma_cva  = einsum('KLY,K->KLY', X, e_cvs, optimize = einsum_type)
        sigma_cva += einsum('KLY,L->KLY', X, e_val, optimize = einsum_type)
        sigma_cva -= einsum('KLx,Yx->KLY', X, h_aa, optimize = einsum_type)
        sigma_cva -= 1/2 * einsum('KLx,K,Yx->KLY', X, e_cvs, rdm_ca, optimize = einsum_type)
        sigma_cva -= 1/2 * einsum('KLx,L,Yx->KLY', X, e_val, rdm_ca, optimize = einsum_type)
        sigma_cva += 1/2 * einsum('KLx,xy,Yy->KLY', X, h_aa, rdm_ca, optimize = einsum_type)
        sigma_cva -= einsum('KLx,Yyxz,zy->KLY', X, v_aaaa, rdm_ca, optimize = einsum_type)
        sigma_cva += 1/2 * einsum('KLx,Yyzx,zy->KLY', X, v_aaaa, rdm_ca, optimize = einsum_type)
        sigma_cva += 1/2 * einsum('KLx,xyzw,Yyzw->KLY', X, v_aaaa, rdm_ccaa, optimize = einsum_type)
        sigma[s_cva:f_cva] += sigma_cva.reshape(-1).copy()

        # VCA <- VCA
        X = Xt[s_vca:f_vca].reshape(nval, ncvs, ncas).copy()

        sigma_vca  = einsum('KLY,K->KLY', X, e_val, optimize = einsum_type)
        sigma_vca += einsum('KLY,L->KLY', X, e_cvs, optimize = einsum_type)
        sigma_vca -= einsum('KLx,Yx->KLY', X, h_aa, optimize = einsum_type)
        sigma_vca -= 1/2 * einsum('KLx,K,Yx->KLY', X, e_val, rdm_ca, optimize = einsum_type)
        sigma_vca -= 1/2 * einsum('KLx,L,Yx->KLY', X, e_cvs, rdm_ca, optimize = einsum_type)
        sigma_vca += 1/2 * einsum('KLx,xy,Yy->KLY', X, h_aa, rdm_ca, optimize = einsum_type)
        sigma_vca -= einsum('KLx,Yyxz,zy->KLY', X, v_aaaa, rdm_ca, optimize = einsum_type)
        sigma_vca += 1/2 * einsum('KLx,Yyzx,zy->KLY', X, v_aaaa, rdm_ca, optimize = einsum_type)
        sigma_vca += 1/2 * einsum('KLx,xyzw,Yyzw->KLY', X, v_aaaa, rdm_ccaa, optimize = einsum_type)
        sigma[s_vca:f_vca] += sigma_vca.reshape(-1).copy()

    return sigma

## CAE block
def compute_excitation_manifolds_cae(mr_adc):

    # MR-ADC(0) and MR-ADC(1)
    mr_adc.h0.n_c = mr_adc.ncvs
    mr_adc.h0.dim = mr_adc.h0.n_c # Total dimension of h0

    mr_adc.h0.s_c = 0
    mr_adc.h0.f_c = mr_adc.h0.s_c + mr_adc.h0.n_c

    print("Dimension of h0 excitation manifold:                       %d" % mr_adc.h0.dim)

    # MR-ADC(2)
    mr_adc.h1.dim = 0
    mr_adc.h_orth.dim = mr_adc.h0.dim

    if mr_adc.method in ("mr-adc(2)", "mr-adc(2)-x"):
        mr_adc.h1.n_caa = 0
        mr_adc.h1.n_cce = 0
        mr_adc.h1.n_cae = mr_adc.nextern * mr_adc.ncas * mr_adc.ncvs
        mr_adc.h1.n_ace = mr_adc.nextern * mr_adc.ncas * mr_adc.ncvs
        mr_adc.h1.n_cca = 0
        if mr_adc.nval > 0:
            mr_adc.h1.n_cve = 0
            mr_adc.h1.n_vce = 0
            mr_adc.h1.n_cva = mr_adc.ncas * mr_adc.ncvs * mr_adc.nval
            mr_adc.h1.n_vca = mr_adc.ncas * mr_adc.ncvs * mr_adc.nval
            mr_adc.h1.dim = (mr_adc.h1.n_caa + mr_adc.h1.n_cce + mr_adc.h1.n_cve + mr_adc.h1.n_vce +
                             mr_adc.h1.n_cae + mr_adc.h1.n_ace + mr_adc.h1.n_cca + mr_adc.h1.n_cva + mr_adc.h1.n_vca)
        else:
            mr_adc.h1.dim = mr_adc.h1.n_caa + mr_adc.h1.n_cce + mr_adc.h1.n_cae + mr_adc.h1.n_cae + mr_adc.h1.n_cca

        if mr_adc.nval > 0:
            mr_adc.h1.s_caa = mr_adc.h0.f_c
            mr_adc.h1.f_caa = mr_adc.h1.s_caa + mr_adc.h1.n_caa
            mr_adc.h1.s_cce = mr_adc.h1.f_caa
            mr_adc.h1.f_cce = mr_adc.h1.s_cce + mr_adc.h1.n_cce
            mr_adc.h1.s_cve = mr_adc.h1.f_cce
            mr_adc.h1.f_cve = mr_adc.h1.s_cve + mr_adc.h1.n_cve
            mr_adc.h1.s_vce = mr_adc.h1.f_cve
            mr_adc.h1.f_vce = mr_adc.h1.s_vce + mr_adc.h1.n_vce
            mr_adc.h1.s_cae = mr_adc.h1.f_vce
            mr_adc.h1.f_cae = mr_adc.h1.s_cae + mr_adc.h1.n_cae
            mr_adc.h1.s_ace = mr_adc.h1.f_cae
            mr_adc.h1.f_ace = mr_adc.h1.s_ace + mr_adc.h1.n_ace
            mr_adc.h1.s_cca = mr_adc.h1.f_ace
            mr_adc.h1.f_cca = mr_adc.h1.s_cca + mr_adc.h1.n_cca
            mr_adc.h1.s_cva = mr_adc.h1.f_cca
            mr_adc.h1.f_cva = mr_adc.h1.s_cva + mr_adc.h1.n_cva
            mr_adc.h1.s_vca = mr_adc.h1.f_cva
            mr_adc.h1.f_vca = mr_adc.h1.s_vca + mr_adc.h1.n_vca
        else:
            mr_adc.h1.s_caa = mr_adc.h0.f_c
            mr_adc.h1.f_caa = mr_adc.h1.s_caa + mr_adc.h1.n_caa
            mr_adc.h1.s_cce = mr_adc.h1.f_caa
            mr_adc.h1.f_cce = mr_adc.h1.s_cce + mr_adc.h1.n_cce
            mr_adc.h1.s_cae = mr_adc.h1.f_cce
            mr_adc.h1.f_cae = mr_adc.h1.s_cae + mr_adc.h1.n_cae
            mr_adc.h1.s_ace = mr_adc.h1.f_cae
            mr_adc.h1.f_ace = mr_adc.h1.s_ace + mr_adc.h1.n_ace
            mr_adc.h1.s_cca = mr_adc.h1.f_ace
            mr_adc.h1.f_cca = mr_adc.h1.s_cca + mr_adc.h1.n_cca

        print("Dimension of h1 excitation manifold:                       %d" % mr_adc.h1.dim)

        # Overlap for c - caa
        mr_adc.S12.c_caa = mr_adc_overlap.compute_S12_0p_projector(mr_adc)
        mr_adc.S12.cae = mr_adc_overlap.compute_S12_m1(mr_adc)
        mr_adc.S12.cca = mr_adc_overlap.compute_S12_p1(mr_adc)

        # Determine dimensions of orthogonalized excitation spaces
        mr_adc.h_orth.n_c = mr_adc.ncvs
        mr_adc.h_orth.n_c_caa = 0
        mr_adc.h_orth.n_cce = 0
        mr_adc.h_orth.n_cce = 0
        mr_adc.h_orth.n_cae = mr_adc.nextern * mr_adc.ncvs * mr_adc.S12.cae.shape[1]
        mr_adc.h_orth.n_ace = mr_adc.nextern * mr_adc.ncvs * mr_adc.S12.cae.shape[1]
        mr_adc.h_orth.n_cca = 0
        if mr_adc.nval > 0:
            mr_adc.h_orth.n_cve = 0
            mr_adc.h_orth.n_vce = 0
            mr_adc.h_orth.n_cva = 0
            mr_adc.h_orth.n_vca = 0
            mr_adc.h_orth.dim = (mr_adc.h_orth.n_c + mr_adc.h_orth.n_c_caa + mr_adc.h_orth.n_cce + mr_adc.h_orth.n_cve + mr_adc.h_orth.n_vce +
                                 mr_adc.h_orth.n_cae + mr_adc.h_orth.n_ace + mr_adc.h_orth.n_cca + mr_adc.h_orth.n_cva + mr_adc.h_orth.n_vca)
        else:
            mr_adc.h_orth.dim = mr_adc.h_orth.n_c + mr_adc.h_orth.n_c_caa + mr_adc.h_orth.n_cce + mr_adc.h_orth.n_cae + mr_adc.h_orth.n_ace + mr_adc.h_orth.n_cca

        if mr_adc.nval > 0:
            mr_adc.h_orth.s_c = 0
            mr_adc.h_orth.f_c = mr_adc.h_orth.n_c
            mr_adc.h_orth.s_c_caa = mr_adc.h_orth.f_c
            mr_adc.h_orth.f_c_caa = mr_adc.h_orth.s_c_caa + mr_adc.h_orth.n_c_caa
            mr_adc.h_orth.s_cce = mr_adc.h_orth.f_c_caa
            mr_adc.h_orth.f_cce = mr_adc.h_orth.s_cce + mr_adc.h_orth.n_cce
            mr_adc.h_orth.s_cve = mr_adc.h_orth.f_cce
            mr_adc.h_orth.f_cve = mr_adc.h_orth.s_cve + mr_adc.h_orth.n_cve
            mr_adc.h_orth.s_vce = mr_adc.h_orth.f_cve
            mr_adc.h_orth.f_vce = mr_adc.h_orth.s_vce + mr_adc.h_orth.n_vce
            mr_adc.h_orth.s_cae = mr_adc.h_orth.f_vce
            mr_adc.h_orth.f_cae = mr_adc.h_orth.s_cae + mr_adc.h_orth.n_cae
            mr_adc.h_orth.s_ace = mr_adc.h_orth.f_cae
            mr_adc.h_orth.f_ace = mr_adc.h_orth.s_ace + mr_adc.h_orth.n_ace
            mr_adc.h_orth.s_cca = mr_adc.h_orth.f_ace
            mr_adc.h_orth.f_cca = mr_adc.h_orth.s_cca + mr_adc.h_orth.n_cca
            mr_adc.h_orth.s_cva = mr_adc.h_orth.f_cca
            mr_adc.h_orth.f_cva = mr_adc.h_orth.s_cva + mr_adc.h_orth.n_cva
            mr_adc.h_orth.s_vca = mr_adc.h_orth.f_cva
            mr_adc.h_orth.f_vca = mr_adc.h_orth.s_vca + mr_adc.h_orth.n_vca
        else:
            mr_adc.h_orth.s_c = 0
            mr_adc.h_orth.f_c = mr_adc.h_orth.n_c
            mr_adc.h_orth.s_cce = mr_adc.h_orth.f_c_caa
            mr_adc.h_orth.f_cce = mr_adc.h_orth.s_cce + mr_adc.h_orth.n_cce
            mr_adc.h_orth.s_cae = mr_adc.h_orth.f_cce
            mr_adc.h_orth.f_cae = mr_adc.h_orth.s_cae + mr_adc.h_orth.n_cae
            mr_adc.h_orth.s_ace = mr_adc.h_orth.f_ace
            mr_adc.h_orth.f_ace = mr_adc.h_orth.s_ace + mr_adc.h_orth.n_ace
            mr_adc.h_orth.s_cca = mr_adc.h_orth.f_ace
            mr_adc.h_orth.f_cca = mr_adc.h_orth.s_cca + mr_adc.h_orth.n_cca

    print("Total dimension of the excitation manifold:                %d" % (mr_adc.h0.dim + mr_adc.h1.dim))
    print("Dimension of the orthogonalized excitation manifold:       %d\n" % (mr_adc.h_orth.dim))
    sys.stdout.flush()

    if (mr_adc.h_orth.dim < mr_adc.nroots):
        mr_adc.nroots = mr_adc.h_orth.dim

    return mr_adc

def compute_preconditioner_cae(mr_adc, M_00):

    start_time = time.time()

    # Einsum definition from kernel
    einsum = mr_adc.interface.einsum
    einsum_type = mr_adc.interface.einsum_type

    if mr_adc.method in ("mr-adc(0)", "mr-adc(1)"):

        # Multiply by -1.0, since we are solving for -M C = -S C E
        return (-1.0 * np.diag(M_00))

    # Variables from kernel
    ncvs = mr_adc.ncvs
    nval = mr_adc.nval
    ncas = mr_adc.ncas
    nextern = mr_adc.nextern

    e_cvs = mr_adc.mo_energy.x
    e_val = mr_adc.mo_energy.v
    e_extern = mr_adc.mo_energy.e

    # One-electron integrals
    h_aa = mr_adc.h1eff.aa

    # Two-electrons integrals
    v_aaaa = mr_adc.v2e.aaaa

    # Reduced Density Matrices
    rdm_ca = mr_adc.rdm.ca
    rdm_ccaa = mr_adc.rdm.ccaa

    # Overlap matrices
    S12_c_caa = mr_adc.S12.c_caa
    S12_cae = mr_adc.S12.cae
    S12_cca = mr_adc.S12.cca

    # Dimensions
    ho_s_c = mr_adc.h_orth.s_c
    ho_f_c = mr_adc.h_orth.f_c
    ho_s_c_caa = mr_adc.h_orth.s_c_caa
    ho_f_c_caa = mr_adc.h_orth.f_c_caa
    ho_s_cce = mr_adc.h_orth.s_cce
    ho_f_cce = mr_adc.h_orth.f_cce
    ho_s_cae = mr_adc.h_orth.s_cae
    ho_f_cae = mr_adc.h_orth.f_cae
    ho_s_ace = mr_adc.h_orth.s_ace
    ho_f_ace = mr_adc.h_orth.f_ace
    ho_s_cca = mr_adc.h_orth.s_cca
    ho_f_cca = mr_adc.h_orth.f_cca
    if nval > 0:
        ho_s_cve = mr_adc.h_orth.s_cve
        ho_f_cve = mr_adc.h_orth.f_cve
        ho_s_vce = mr_adc.h_orth.s_vce
        ho_f_vce = mr_adc.h_orth.f_vce

        ho_s_cva = mr_adc.h_orth.s_cva
        ho_f_cva = mr_adc.h_orth.f_cva
        ho_s_vca = mr_adc.h_orth.s_vca
        ho_f_vca = mr_adc.h_orth.f_vca

    s_c = mr_adc.h0.s_c
    f_c = mr_adc.h0.f_c

    cvs_tril_ind = np.tril_indices(ncvs, k=-1)
    # cas_ind = np.tril_indices(ncas, k=-1)

    # Build the preconditioner
    precond = np.zeros(mr_adc.h_orth.dim)

    # C-C debug
    precond[ho_s_c:ho_f_c] = np.diag(M_00[s_c:f_c, s_c:f_c]).copy()

    # CAE
    precond_cae =- 1/2 * einsum('A,AA,II,XY->IAXY', e_extern, np.identity(nextern), np.identity(ncvs), rdm_ca, optimize = einsum_type)
    precond_cae += 1/2 * einsum('I,AA,II,XY->IAXY', e_cvs, np.identity(nextern), np.identity(ncvs), rdm_ca, optimize = einsum_type)
    precond_cae += 1/2 * einsum('Xx,AA,II,xY->IAXY', h_aa, np.identity(nextern), np.identity(ncvs), rdm_ca, optimize = einsum_type)
    precond_cae += 1/2 * einsum('Xxyz,AA,II,Yxyz->IAXY', v_aaaa, np.identity(nextern), np.identity(ncvs), rdm_ccaa, optimize = einsum_type)

    precond_cae = einsum("IAXY,XP,YP->IPA", precond_cae, S12_cae, S12_cae, optimize = einsum_type)
    precond[ho_s_cae:ho_f_cae] = precond_cae.reshape(-1).copy()

    # ACE
    precond_ace =- 1/2 * einsum('A,AA,II,XY->XYIA', e_extern, np.identity(nextern), np.identity(ncvs), rdm_ca, optimize = einsum_type)
    precond_ace += 1/2 * einsum('I,AA,II,XY->XYIA', e_cvs, np.identity(nextern), np.identity(ncvs), rdm_ca, optimize = einsum_type)
    precond_ace += 1/2 * einsum('Xx,AA,II,xY->XYIA', h_aa, np.identity(nextern), np.identity(ncvs), rdm_ca, optimize = einsum_type)
    precond_ace += 1/2 * einsum('Xxyz,AA,II,Yxyz->XYIA', v_aaaa, np.identity(nextern), np.identity(ncvs), rdm_ccaa, optimize = einsum_type)

    precond_ace = einsum("XYIA,XP,YP->PIA", precond_ace, S12_cae, S12_cae, optimize = einsum_type)
    precond[ho_s_ace:ho_f_ace] = precond_ace.reshape(-1).copy()

    # Multiply by -1.0, since we are solving for -M C = -S C E
    precond *= (-1.0)

    print ("Time for computing preconditioner:                %f sec\n" % (time.time() - start_time))
    sys.stdout.flush()

    return precond

def apply_S_12_cae(mr_adc, X, transpose = False):

    # Einsum definition from kernel
    einsum = mr_adc.interface.einsum
    einsum_type = mr_adc.interface.einsum_type

    # Dimensions
    nextern = mr_adc.nextern
    ncvs = mr_adc.ncvs
    nval = mr_adc.nval

    ho_s_c = mr_adc.h_orth.s_c
    ho_f_c = mr_adc.h_orth.f_c
    ho_s_c_caa = mr_adc.h_orth.s_c_caa
    ho_f_c_caa = mr_adc.h_orth.f_c_caa
    ho_s_cce = mr_adc.h_orth.s_cce
    ho_f_cce = mr_adc.h_orth.f_cce
    ho_s_cae = mr_adc.h_orth.s_cae
    ho_f_cae = mr_adc.h_orth.f_cae
    ho_s_ace = mr_adc.h_orth.s_ace
    ho_f_ace = mr_adc.h_orth.f_ace
    ho_s_cca = mr_adc.h_orth.s_cca
    ho_f_cca = mr_adc.h_orth.f_cca
    s_c = mr_adc.h0.s_c
    f_c = mr_adc.h0.f_c
    s_caa = mr_adc.h1.s_caa
    f_caa = mr_adc.h1.f_caa
    s_cce = mr_adc.h1.s_cce
    f_cce = mr_adc.h1.f_cce
    s_cae = mr_adc.h1.s_cae
    f_cae = mr_adc.h1.f_cae
    s_ace = mr_adc.h1.s_ace
    f_ace = mr_adc.h1.f_ace
    s_cca = mr_adc.h1.s_cca
    f_cca = mr_adc.h1.f_cca

    if nval > 0:
        ho_s_cve = mr_adc.h_orth.s_cve
        ho_f_cve = mr_adc.h_orth.f_cve
        ho_s_vce = mr_adc.h_orth.s_vce
        ho_f_vce = mr_adc.h_orth.f_vce

        ho_s_cva = mr_adc.h_orth.s_cva
        ho_f_cva = mr_adc.h_orth.f_cva
        ho_s_vca = mr_adc.h_orth.s_vca
        ho_f_vca = mr_adc.h_orth.f_vca

        s_cve = mr_adc.h1.s_cve
        f_cve = mr_adc.h1.f_cve
        s_vce = mr_adc.h1.s_vce
        f_vce = mr_adc.h1.f_vce

        s_cva = mr_adc.h1.s_cva
        f_cva = mr_adc.h1.f_cva
        s_vca = mr_adc.h1.s_vca
        f_vca = mr_adc.h1.f_vca

    # Overlap matrices
    S12_c_caa = mr_adc.S12.c_caa
    S12_cae = mr_adc.S12.cae
    S12_cca = mr_adc.S12.cca

    Xt = None

    if transpose:
        if (X.shape[0] != (mr_adc.h0.dim + mr_adc.h1.dim)):
            raise Exception("Dimensions do not match when applying S_12 transpose")

        Xt = np.zeros(mr_adc.h_orth.dim)


        # C-C DEBUG
        Xt[ho_s_c:ho_f_c] = X[s_c:f_c].copy()

        # CAE
        temp = X[s_cae:f_cae].reshape(ncvs, S12_cae.shape[0], nextern).copy()
        Xt[ho_s_cae:ho_f_cae] = einsum("IXA,XP->IPA", temp, S12_cae).reshape(-1).copy()

        # ACE
        temp = X[s_ace:f_ace].reshape(S12_cae.shape[0], ncvs, nextern).copy()
        Xt[ho_s_ace:ho_f_ace] = einsum("XIA,XP->PIA", temp, S12_cae).reshape(-1).copy()

    else:
        if (X.shape[0] != (mr_adc.h_orth.dim)):
            raise Exception("Dimensions do not match when applying S_12")

        Xt = np.zeros(mr_adc.h0.dim + mr_adc.h1.dim)

        # C-C DEBUG
        Xt[s_c:f_c] = X[ho_s_c:ho_f_c].copy()

        # CAE
        temp = X[ho_s_cae:ho_f_cae].reshape(ncvs, S12_cae.shape[1], nextern).copy()
        Xt[s_cae:f_cae] = einsum("IPA,XP->IXA", temp, S12_cae).reshape(-1).copy()

        # ACE
        temp = X[ho_s_ace:ho_f_ace].reshape(S12_cae.shape[1], ncvs, nextern).copy()
        Xt[s_ace:f_ace] = einsum("PIA,XP->XIA", temp, S12_cae).reshape(-1).copy()

    return Xt

def compute_sigma_vector_cae(mr_adc, M_00, M_01, M_11, Xt):

    # Einsum definition from kernel
    einsum = mr_adc.interface.einsum
    einsum_type = mr_adc.interface.einsum_type

    # Variables from kernel
    ncvs = mr_adc.ncvs
    nval = mr_adc.nval
    ncas = mr_adc.ncas
    nextern = mr_adc.nextern

    e_cvs = mr_adc.mo_energy.x
    e_core = mr_adc.mo_energy.c
    e_val = mr_adc.mo_energy.v
    e_extern = mr_adc.mo_energy.e

    # One-electron integrals
    h_aa = mr_adc.h1eff.aa

    # Two-electrons integrals
    v_aaaa = mr_adc.v2e.aaaa

    # Reduced Density Matrices
    rdm_ca = mr_adc.rdm.ca
    rdm_ccaa = mr_adc.rdm.ccaa

    # Dimensions
    s_c = mr_adc.h0.s_c
    f_c = mr_adc.h0.f_c
    s_caa = mr_adc.h1.s_caa
    f_caa = mr_adc.h1.f_caa
    s_cce = mr_adc.h1.s_cce
    f_cce = mr_adc.h1.f_cce
    s_cae = mr_adc.h1.s_cae
    f_cae = mr_adc.h1.f_cae
    s_ace = mr_adc.h1.s_ace
    f_ace = mr_adc.h1.f_ace
    s_cca = mr_adc.h1.s_cca
    f_cca = mr_adc.h1.f_cca
    if nval > 0:
        s_cve = mr_adc.h1.s_cve
        f_cve = mr_adc.h1.f_cve
        s_vce = mr_adc.h1.s_vce
        f_vce = mr_adc.h1.f_vce

        s_cva = mr_adc.h1.s_cva
        f_cva = mr_adc.h1.f_cva
        s_vca = mr_adc.h1.s_vca
        f_vca = mr_adc.h1.f_vca

    cvs_tril_ind = np.tril_indices(ncvs, k=-1)

    # (CASCI + C) -> (CASCI + C)
    sigma = np.zeros_like(Xt)

    # h0-h0 contributions
    sigma[:mr_adc.h0.dim] = np.dot(M_00, Xt[:mr_adc.h0.dim])

    # h1-h1 contributions
    # CAE <- CAE
    X = Xt[s_cae:f_cae].reshape(ncvs, ncas, nextern).copy()

    sigma_cae =- 1/2 * einsum('KxB,B,xY->KYB', X, e_extern, rdm_ca, optimize = einsum_type)
    sigma_cae += 1/2 * einsum('KxB,K,xY->KYB', X, e_cvs, rdm_ca, optimize = einsum_type)
    sigma_cae += 1/2 * einsum('KxB,xy,yY->KYB', X, h_aa, rdm_ca, optimize = einsum_type)
    sigma_cae += 1/2 * einsum('KxB,xyzw,Yyzw->KYB', X, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma[s_cae:f_cae] += sigma_cae.reshape(-1).copy()

    # ACE <- ACE
    X = Xt[s_ace:f_ace].reshape(ncas, ncvs, nextern).copy()

    sigma_ace =- 1/2 * einsum('xKB,B,xY->YKB', X, e_extern, rdm_ca, optimize = einsum_type)
    sigma_ace += 1/2 * einsum('xKB,K,xY->YKB', X, e_cvs, rdm_ca, optimize = einsum_type)
    sigma_ace += 1/2 * einsum('xKB,xy,yY->YKB', X, h_aa, rdm_ca, optimize = einsum_type)
    sigma_ace += 1/2 * einsum('xKB,xyzw,Yyzw->YKB', X, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma[s_ace:f_ace] += sigma_ace.reshape(-1).copy()

    return sigma

## CAA block (v1)
def compute_excitation_manifolds_caa(mr_adc):

    # MR-ADC(0) and MR-ADC(1)
    mr_adc.h0.n_c = mr_adc.ncvs
    mr_adc.h0.dim = mr_adc.h0.n_c # Total dimension of h0

    mr_adc.h0.s_c = 0
    mr_adc.h0.f_c = mr_adc.h0.s_c + mr_adc.h0.n_c

    print("Dimension of h0 excitation manifold:                       %d" % mr_adc.h0.dim)

    # MR-ADC(2)
    mr_adc.h1.dim = 0
    mr_adc.h_orth.dim = mr_adc.h0.dim

    if mr_adc.method in ("mr-adc(2)", "mr-adc(2)-x"):
        # mr_adc.h1.n_caa = 2 * mr_adc.ncas * mr_adc.ncas * mr_adc.ncvs
        mr_adc.h1.n_caa = 3 * mr_adc.ncas * mr_adc.ncas * mr_adc.ncvs
        mr_adc.h1.n_cce = 0
        mr_adc.h1.n_cae = 0
        mr_adc.h1.n_ace = 0
        mr_adc.h1.n_cca = 0
        if mr_adc.nval > 0:
            mr_adc.h1.n_cve = 0
            mr_adc.h1.n_vce = 0
            mr_adc.h1.n_cva = 0
            mr_adc.h1.n_vca = 0
            mr_adc.h1.dim = (mr_adc.h1.n_caa + mr_adc.h1.n_cce + mr_adc.h1.n_cve + mr_adc.h1.n_vce +
                             mr_adc.h1.n_cae + mr_adc.h1.n_ace + mr_adc.h1.n_cca + mr_adc.h1.n_cva + mr_adc.h1.n_vca)
        else:
            mr_adc.h1.dim = mr_adc.h1.n_caa + mr_adc.h1.n_cce + mr_adc.h1.n_cae + mr_adc.h1.n_cae + mr_adc.h1.n_cca

        if mr_adc.nval > 0:
            mr_adc.h1.s_caa = mr_adc.h0.f_c
            mr_adc.h1.f_caa = mr_adc.h1.s_caa + mr_adc.h1.n_caa
            mr_adc.h1.s_cce = mr_adc.h1.f_caa
            mr_adc.h1.f_cce = mr_adc.h1.s_cce + mr_adc.h1.n_cce
            mr_adc.h1.s_cve = mr_adc.h1.f_cce
            mr_adc.h1.f_cve = mr_adc.h1.s_cve + mr_adc.h1.n_cve
            mr_adc.h1.s_vce = mr_adc.h1.f_cve
            mr_adc.h1.f_vce = mr_adc.h1.s_vce + mr_adc.h1.n_vce
            mr_adc.h1.s_cae = mr_adc.h1.f_vce
            mr_adc.h1.f_cae = mr_adc.h1.s_cae + mr_adc.h1.n_cae
            mr_adc.h1.s_ace = mr_adc.h1.f_cae
            mr_adc.h1.f_ace = mr_adc.h1.s_ace + mr_adc.h1.n_ace
            mr_adc.h1.s_cca = mr_adc.h1.f_ace
            mr_adc.h1.f_cca = mr_adc.h1.s_cca + mr_adc.h1.n_cca
            mr_adc.h1.s_cva = mr_adc.h1.f_cca
            mr_adc.h1.f_cva = mr_adc.h1.s_cva + mr_adc.h1.n_cva
            mr_adc.h1.s_vca = mr_adc.h1.f_cva
            mr_adc.h1.f_vca = mr_adc.h1.s_vca + mr_adc.h1.n_vca
        else:
            mr_adc.h1.s_caa = mr_adc.h0.f_c
            mr_adc.h1.f_caa = mr_adc.h1.s_caa + mr_adc.h1.n_caa
            mr_adc.h1.s_cce = mr_adc.h1.f_caa
            mr_adc.h1.f_cce = mr_adc.h1.s_cce + mr_adc.h1.n_cce
            mr_adc.h1.s_cae = mr_adc.h1.f_cce
            mr_adc.h1.f_cae = mr_adc.h1.s_cae + mr_adc.h1.n_cae
            mr_adc.h1.s_ace = mr_adc.h1.f_cae
            mr_adc.h1.f_ace = mr_adc.h1.s_ace + mr_adc.h1.n_ace
            mr_adc.h1.s_cca = mr_adc.h1.f_ace
            mr_adc.h1.f_cca = mr_adc.h1.s_cca + mr_adc.h1.n_cca

        print("Dimension of h1 excitation manifold:                       %d" % mr_adc.h1.dim)

        # Overlap for c - caa
        mr_adc.S12.c_caa = mr_adc_overlap.compute_S12_0p_projector(mr_adc)
        mr_adc.S12.cae = mr_adc_overlap.compute_S12_m1(mr_adc)
        mr_adc.S12.cca = mr_adc_overlap.compute_S12_p1(mr_adc)

        # Determine dimensions of orthogonalized excitation spaces
        mr_adc.h_orth.n_c = 0
        mr_adc.h_orth.n_c_caa = mr_adc.ncvs * mr_adc.S12.c_caa.shape[1]
        mr_adc.h_orth.n_cce = 0
        mr_adc.h_orth.n_cce = 0
        mr_adc.h_orth.n_cae = 0
        mr_adc.h_orth.n_ace = 0
        mr_adc.h_orth.n_cca = 0
        if mr_adc.nval > 0:
            mr_adc.h_orth.n_cve = 0
            mr_adc.h_orth.n_vce = 0
            mr_adc.h_orth.n_cva = 0
            mr_adc.h_orth.n_vca = 0
            mr_adc.h_orth.dim = (mr_adc.h_orth.n_c + mr_adc.h_orth.n_c_caa + mr_adc.h_orth.n_cce + mr_adc.h_orth.n_cve + mr_adc.h_orth.n_vce +
                                 mr_adc.h_orth.n_cae + mr_adc.h_orth.n_ace + mr_adc.h_orth.n_cca + mr_adc.h_orth.n_cva + mr_adc.h_orth.n_vca)
        else:
            mr_adc.h_orth.dim = mr_adc.h_orth.n_c + mr_adc.h_orth.n_c_caa + mr_adc.h_orth.n_cce + mr_adc.h_orth.n_cae + mr_adc.h_orth.n_ace + mr_adc.h_orth.n_cca

        if mr_adc.nval > 0:
            mr_adc.h_orth.s_c = 0
            mr_adc.h_orth.f_c = mr_adc.h_orth.n_c
            mr_adc.h_orth.s_c_caa = mr_adc.h_orth.f_c
            mr_adc.h_orth.f_c_caa = mr_adc.h_orth.s_c_caa + mr_adc.h_orth.n_c_caa
            mr_adc.h_orth.s_cce = mr_adc.h_orth.f_c_caa
            mr_adc.h_orth.f_cce = mr_adc.h_orth.s_cce + mr_adc.h_orth.n_cce
            mr_adc.h_orth.s_cve = mr_adc.h_orth.f_cce
            mr_adc.h_orth.f_cve = mr_adc.h_orth.s_cve + mr_adc.h_orth.n_cve
            mr_adc.h_orth.s_vce = mr_adc.h_orth.f_cve
            mr_adc.h_orth.f_vce = mr_adc.h_orth.s_vce + mr_adc.h_orth.n_vce
            mr_adc.h_orth.s_cae = mr_adc.h_orth.f_vce
            mr_adc.h_orth.f_cae = mr_adc.h_orth.s_cae + mr_adc.h_orth.n_cae
            mr_adc.h_orth.s_ace = mr_adc.h_orth.f_cae
            mr_adc.h_orth.f_ace = mr_adc.h_orth.s_ace + mr_adc.h_orth.n_ace
            mr_adc.h_orth.s_cca = mr_adc.h_orth.f_ace
            mr_adc.h_orth.f_cca = mr_adc.h_orth.s_cca + mr_adc.h_orth.n_cca
            mr_adc.h_orth.s_cva = mr_adc.h_orth.f_cca
            mr_adc.h_orth.f_cva = mr_adc.h_orth.s_cva + mr_adc.h_orth.n_cva
            mr_adc.h_orth.s_vca = mr_adc.h_orth.f_cva
            mr_adc.h_orth.f_vca = mr_adc.h_orth.s_vca + mr_adc.h_orth.n_vca
        else:
            mr_adc.h_orth.s_c = 0
            mr_adc.h_orth.f_c = mr_adc.h_orth.n_c
            mr_adc.h_orth.s_c_caa = mr_adc.h_orth.f_c
            mr_adc.h_orth.f_c_caa = mr_adc.h_orth.s_c_caa + mr_adc.h_orth.n_c_caa
            mr_adc.h_orth.s_cce = mr_adc.h_orth.f_c_caa
            mr_adc.h_orth.f_cce = mr_adc.h_orth.s_cce + mr_adc.h_orth.n_cce
            mr_adc.h_orth.s_cae = mr_adc.h_orth.f_cce
            mr_adc.h_orth.f_cae = mr_adc.h_orth.s_cae + mr_adc.h_orth.n_cae
            mr_adc.h_orth.s_ace = mr_adc.h_orth.f_cae
            mr_adc.h_orth.f_ace = mr_adc.h_orth.s_ace + mr_adc.h_orth.n_ace
            mr_adc.h_orth.s_cca = mr_adc.h_orth.f_ace
            mr_adc.h_orth.f_cca = mr_adc.h_orth.s_cca + mr_adc.h_orth.n_cca

    print("Total dimension of the excitation manifold:                %d" % (mr_adc.h0.dim + mr_adc.h1.dim))
    print("Dimension of the orthogonalized excitation manifold:       %d\n" % (mr_adc.h_orth.dim))
    sys.stdout.flush()

    if (mr_adc.h_orth.dim < mr_adc.nroots):
        mr_adc.nroots = mr_adc.h_orth.dim

    return mr_adc

def compute_preconditioner_caa(mr_adc, M_00):

    start_time = time.time()

    # Einsum definition from kernel
    einsum = mr_adc.interface.einsum
    einsum_type = mr_adc.interface.einsum_type

    if mr_adc.method in ("mr-adc(0)", "mr-adc(1)"):

        # Multiply by -1.0, since we are solving for -M C = -S C E
        return (-1.0 * np.diag(M_00))

    # Variables from kernel
    ncvs = mr_adc.ncvs
    nval = mr_adc.nval
    ncas = mr_adc.ncas
    nextern = mr_adc.nextern

    e_cvs = mr_adc.mo_energy.x
    if nval > 0:
        e_val = mr_adc.mo_energy.v
    e_extern = mr_adc.mo_energy.e

    # One-electron integrals
    h_aa = mr_adc.h1eff.aa

    # Two-electrons integrals
    v_aaaa = mr_adc.v2e.aaaa

    v_xaxa = mr_adc.v2e.xaxa
    v_xaax = mr_adc.v2e.xaax

    # Reduced Density Matrices
    rdm_ca = mr_adc.rdm.ca
    rdm_ccaa = mr_adc.rdm.ccaa
    rdm_cccaaa = mr_adc.rdm.cccaaa

    # Overlap matrices
    S12_c_caa = mr_adc.S12.c_caa
    S12_cae = mr_adc.S12.cae
    S12_cca = mr_adc.S12.cca

    # Dimensions
    ho_s_c = mr_adc.h_orth.s_c
    ho_f_c = mr_adc.h_orth.f_c
    ho_s_c_caa = mr_adc.h_orth.s_c_caa
    ho_f_c_caa = mr_adc.h_orth.f_c_caa
    ho_s_cce = mr_adc.h_orth.s_cce
    ho_f_cce = mr_adc.h_orth.f_cce
    ho_s_cae = mr_adc.h_orth.s_cae
    ho_f_cae = mr_adc.h_orth.f_cae
    ho_s_ace = mr_adc.h_orth.s_ace
    ho_f_ace = mr_adc.h_orth.f_ace
    ho_s_cca = mr_adc.h_orth.s_cca
    ho_f_cca = mr_adc.h_orth.f_cca
    if nval > 0:
        ho_s_cve = mr_adc.h_orth.s_cve
        ho_f_cve = mr_adc.h_orth.f_cve
        ho_s_vce = mr_adc.h_orth.s_vce
        ho_f_vce = mr_adc.h_orth.f_vce

        ho_s_cva = mr_adc.h_orth.s_cva
        ho_f_cva = mr_adc.h_orth.f_cva
        ho_s_vca = mr_adc.h_orth.s_vca
        ho_f_vca = mr_adc.h_orth.f_vca

    s_c = mr_adc.h0.s_c
    f_c = mr_adc.h0.f_c

    cvs_tril_ind = np.tril_indices(ncvs, k=-1)
    # cas_ind = np.tril_indices(ncas, k=-1)

    # Build the preconditioner
    precond = np.zeros(mr_adc.h_orth.dim)

    # C and CAA
    # 0th-order
    precond_c_caa_a_aaa  = 1/2 * einsum('I,II,XY->IXY', e_cvs, np.identity(ncvs), rdm_ca, optimize = einsum_type)
    # precond_c_caa_a_aaa += 1/2 * einsum('Xx,II,xY->IXY', h_aa, np.identity(ncvs), rdm_ca, optimize = einsum_type)
    # precond_c_caa_a_aaa -= 1/2 * einsum('Yx,II,Xx->IXY', h_aa, np.identity(ncvs), rdm_ca, optimize = einsum_type)
    # precond_c_caa_a_aaa += 1/2 * einsum('II,Xxyz,Yxyz->IXY', np.identity(ncvs), v_aaaa, rdm_ccaa, optimize = einsum_type)
    # precond_c_caa_a_aaa -= 1/2 * einsum('II,Yxyz,Xxyz->IXY', np.identity(ncvs), v_aaaa, rdm_ccaa, optimize = einsum_type)

    precond_c_caa_a_abb  = 1/2 * einsum('I,II,XY->IXY', e_cvs, np.identity(ncvs), rdm_ca, optimize = einsum_type)
    # precond_c_caa_a_abb += 1/2 * einsum('Xx,II,xY->IXY', h_aa, np.identity(ncvs), rdm_ca, optimize = einsum_type)
    # precond_c_caa_a_abb -= 1/2 * einsum('Yx,II,Xx->IXY', h_aa, np.identity(ncvs), rdm_ca, optimize = einsum_type)
    # precond_c_caa_a_abb += 1/2 * einsum('II,Xxyz,Yxyz->IXY', np.identity(ncvs), v_aaaa, rdm_ccaa, optimize = einsum_type)
    # precond_c_caa_a_abb -= 1/2 * einsum('II,Yxyz,Xxyz->IXY', np.identity(ncvs), v_aaaa, rdm_ccaa, optimize = einsum_type)

    # 1st-order
    precond_c_caa_a_aaa += 1/2 * einsum('IxIY,Xx->IXY', v_xaxa, rdm_ca, optimize = einsum_type)
    precond_c_caa_a_aaa += 1/2 * einsum('IxIy,XyYx->IXY', v_xaxa, rdm_ccaa, optimize = einsum_type)
    precond_c_caa_a_aaa -= 1/2 * einsum('IxYI,Xx->IXY', v_xaax, rdm_ca, optimize = einsum_type)
    precond_c_caa_a_aaa -= 1/6 * einsum('IxyI,XyYx->IXY', v_xaax, rdm_ccaa, optimize = einsum_type)
    precond_c_caa_a_aaa += 1/6 * einsum('IxyI,XyxY->IXY', v_xaax, rdm_ccaa, optimize = einsum_type)
    precond_c_caa_a_aaa -= 1/2 * einsum('IxIy,xy,XY->IXY', v_xaxa, rdm_ca, rdm_ca, optimize = einsum_type)
    precond_c_caa_a_aaa += 1/4 * einsum('IxyI,xy,XY->IXY', v_xaax, rdm_ca, rdm_ca, optimize = einsum_type)

    precond_c_caa_a_abb += 1/2 * einsum('IxIY,Xx->IXY', v_xaxa, rdm_ca, optimize = einsum_type)
    precond_c_caa_a_abb += 1/2 * einsum('IxIy,XyYx->IXY', v_xaxa, rdm_ccaa, optimize = einsum_type)
    precond_c_caa_a_abb -= 1/3 * einsum('IxyI,XyYx->IXY', v_xaax, rdm_ccaa, optimize = einsum_type)
    precond_c_caa_a_abb -= 1/6 * einsum('IxyI,XyxY->IXY', v_xaax, rdm_ccaa, optimize = einsum_type)
    precond_c_caa_a_abb -= 1/2 * einsum('IxIy,xy,XY->IXY', v_xaxa, rdm_ca, rdm_ca, optimize = einsum_type)
    precond_c_caa_a_abb += 1/4 * einsum('IxyI,xy,XY->IXY', v_xaax, rdm_ca, rdm_ca, optimize = einsum_type)

    # precond_caa_caa_aaa_aaa =- 1/6 * einsum('I,II,WYXZ->IWZXY', e_cvs, np.identity(ncvs), rdm_ccaa, optimize = einsum_type)
    # precond_caa_caa_aaa_aaa += 1/6 * einsum('I,II,WYZX->IWZXY', e_cvs, np.identity(ncvs), rdm_ccaa, optimize = einsum_type)
    # precond_caa_caa_aaa_aaa += 1/6 * einsum('Xx,II,WYZx->IWZXY', h_aa, np.identity(ncvs), rdm_ccaa, optimize = einsum_type)
    # precond_caa_caa_aaa_aaa -= 1/6 * einsum('Xx,II,WYxZ->IWZXY', h_aa, np.identity(ncvs), rdm_ccaa, optimize = einsum_type)
    # precond_caa_caa_aaa_aaa -= 1/2 * einsum('YZ,II,XW->IWZXY', h_aa, np.identity(ncvs), rdm_ca, optimize = einsum_type)
    # precond_caa_caa_aaa_aaa += 1/6 * einsum('Yx,II,WxXZ->IWZXY', h_aa, np.identity(ncvs), rdm_ccaa, optimize = einsum_type)
    # precond_caa_caa_aaa_aaa -= 1/6 * einsum('Yx,II,WxZX->IWZXY', h_aa, np.identity(ncvs), rdm_ccaa, optimize = einsum_type)
    # precond_caa_caa_aaa_aaa -= 1/6 * einsum('II,XZxy,WYxy->IWZXY', np.identity(ncvs), v_aaaa, rdm_ccaa, optimize = einsum_type)
    # precond_caa_caa_aaa_aaa += 1/6 * einsum('II,XZxy,WYyx->IWZXY', np.identity(ncvs), v_aaaa, rdm_ccaa, optimize = einsum_type)
    # precond_caa_caa_aaa_aaa += 1/6 * einsum('II,Xxyz,ZyzWYx->IWZXY', np.identity(ncvs), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    # precond_caa_caa_aaa_aaa -= 1/6 * einsum('II,Xxyz,ZyzYWx->IWZXY', np.identity(ncvs), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    # precond_caa_caa_aaa_aaa -= 1/2 * einsum('II,YxZy,WyXx->IWZXY', np.identity(ncvs), v_aaaa, rdm_ccaa, optimize = einsum_type)
    # precond_caa_caa_aaa_aaa += 1/6 * einsum('II,YxyZ,WyXx->IWZXY', np.identity(ncvs), v_aaaa, rdm_ccaa, optimize = einsum_type)
    # precond_caa_caa_aaa_aaa -= 1/6 * einsum('II,YxyZ,WyxX->IWZXY', np.identity(ncvs), v_aaaa, rdm_ccaa, optimize = einsum_type)
    # precond_caa_caa_aaa_aaa += 1/6 * einsum('II,Yxyz,XZxWyz->IWZXY', np.identity(ncvs), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    # precond_caa_caa_aaa_aaa -= 1/6 * einsum('II,Yxyz,XZxyWz->IWZXY', np.identity(ncvs), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    # precond_caa_caa_aaa_aaa += 1/2 * einsum('I,II,YZ,XW->IWZXY', e_cvs, np.identity(ncvs), np.identity(ncas), rdm_ca, optimize = einsum_type)
    # precond_caa_caa_aaa_aaa += 1/2 * einsum('Xx,II,YZ,xW->IWZXY', h_aa, np.identity(ncvs), np.identity(ncas), rdm_ca, optimize = einsum_type)
    # precond_caa_caa_aaa_aaa += 1/2 * einsum('Xxyz,II,YZ,Wxyz->IWZXY', v_aaaa, np.identity(ncvs), np.identity(ncas), rdm_ccaa, optimize = einsum_type)

    # precond_caa_caa_abb_abb =- 1/6 * einsum('I,II,WYXZ->IWZXY', e_cvs, np.identity(ncvs), rdm_ccaa, optimize = einsum_type)
    # precond_caa_caa_abb_abb += 1/6 * einsum('I,II,WYZX->IWZXY', e_cvs, np.identity(ncvs), rdm_ccaa, optimize = einsum_type)
    # precond_caa_caa_abb_abb += 1/6 * einsum('Xx,II,WYZx->IWZXY', h_aa, np.identity(ncvs), rdm_ccaa, optimize = einsum_type)
    # precond_caa_caa_abb_abb -= 1/6 * einsum('Xx,II,WYxZ->IWZXY', h_aa, np.identity(ncvs), rdm_ccaa, optimize = einsum_type)
    # precond_caa_caa_abb_abb -= 1/2 * einsum('YZ,II,XW->IWZXY', h_aa, np.identity(ncvs), rdm_ca, optimize = einsum_type)
    # precond_caa_caa_abb_abb += 1/6 * einsum('Yx,II,WxXZ->IWZXY', h_aa, np.identity(ncvs), rdm_ccaa, optimize = einsum_type)
    # precond_caa_caa_abb_abb -= 1/6 * einsum('Yx,II,WxZX->IWZXY', h_aa, np.identity(ncvs), rdm_ccaa, optimize = einsum_type)
    # precond_caa_caa_abb_abb -= 1/6 * einsum('II,XZxy,WYxy->IWZXY', np.identity(ncvs), v_aaaa, rdm_ccaa, optimize = einsum_type)
    # precond_caa_caa_abb_abb += 1/6 * einsum('II,XZxy,WYyx->IWZXY', np.identity(ncvs), v_aaaa, rdm_ccaa, optimize = einsum_type)
    # precond_caa_caa_abb_abb += 1/6 * einsum('II,Xxyz,ZyzWYx->IWZXY', np.identity(ncvs), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    # precond_caa_caa_abb_abb -= 1/6 * einsum('II,Xxyz,ZyzYWx->IWZXY', np.identity(ncvs), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    # precond_caa_caa_abb_abb -= 1/2 * einsum('II,YxZy,WyXx->IWZXY', np.identity(ncvs), v_aaaa, rdm_ccaa, optimize = einsum_type)
    # precond_caa_caa_abb_abb += 1/6 * einsum('II,YxyZ,WyXx->IWZXY', np.identity(ncvs), v_aaaa, rdm_ccaa, optimize = einsum_type)
    # precond_caa_caa_abb_abb -= 1/6 * einsum('II,YxyZ,WyxX->IWZXY', np.identity(ncvs), v_aaaa, rdm_ccaa, optimize = einsum_type)
    # precond_caa_caa_abb_abb += 1/6 * einsum('II,Yxyz,XZxWyz->IWZXY', np.identity(ncvs), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    # precond_caa_caa_abb_abb -= 1/6 * einsum('II,Yxyz,XZxyWz->IWZXY', np.identity(ncvs), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    # precond_caa_caa_abb_abb += 1/2 * einsum('I,II,YZ,XW->IWZXY', e_cvs, np.identity(ncvs), np.identity(ncas), rdm_ca, optimize = einsum_type)
    # precond_caa_caa_abb_abb += 1/2 * einsum('Xx,II,YZ,xW->IWZXY', h_aa, np.identity(ncvs), np.identity(ncas), rdm_ca, optimize = einsum_type)
    # precond_caa_caa_abb_abb += 1/2 * einsum('Xxyz,II,YZ,Wxyz->IWZXY', v_aaaa, np.identity(ncvs), np.identity(ncas), rdm_ccaa, optimize = einsum_type)

    # precond_caa_caa_bab_bab =- 1/3 * einsum('I,II,WYXZ->IWZXY', e_cvs, np.identity(ncvs), rdm_ccaa, optimize = einsum_type)
    # precond_caa_caa_bab_bab -= 1/6 * einsum('I,II,WYZX->IWZXY', e_cvs, np.identity(ncvs), rdm_ccaa, optimize = einsum_type)
    # precond_caa_caa_bab_bab -= 1/6 * einsum('Xx,II,WYZx->IWZXY', h_aa, np.identity(ncvs), rdm_ccaa, optimize = einsum_type)
    # precond_caa_caa_bab_bab -= 1/3 * einsum('Xx,II,WYxZ->IWZXY', h_aa, np.identity(ncvs), rdm_ccaa, optimize = einsum_type)
    # precond_caa_caa_bab_bab -= 1/2 * einsum('YZ,II,XW->IWZXY', h_aa, np.identity(ncvs), rdm_ca, optimize = einsum_type)
    # precond_caa_caa_bab_bab += 1/3 * einsum('Yx,II,WxXZ->IWZXY', h_aa, np.identity(ncvs), rdm_ccaa, optimize = einsum_type)
    # precond_caa_caa_bab_bab += 1/6 * einsum('Yx,II,WxZX->IWZXY', h_aa, np.identity(ncvs), rdm_ccaa, optimize = einsum_type)
    # precond_caa_caa_bab_bab -= 1/3 * einsum('II,XZxy,WYxy->IWZXY', np.identity(ncvs), v_aaaa, rdm_ccaa, optimize = einsum_type)
    # precond_caa_caa_bab_bab -= 1/6 * einsum('II,XZxy,WYyx->IWZXY', np.identity(ncvs), v_aaaa, rdm_ccaa, optimize = einsum_type)
    # precond_caa_caa_bab_bab -= 1/12 * einsum('II,Xxyz,ZyzWYx->IWZXY', np.identity(ncvs), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    # precond_caa_caa_bab_bab += 1/12 * einsum('II,Xxyz,ZyzWxY->IWZXY', np.identity(ncvs), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    # precond_caa_caa_bab_bab -= 1/4 * einsum('II,Xxyz,ZyzYWx->IWZXY', np.identity(ncvs), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    # precond_caa_caa_bab_bab += 1/12 * einsum('II,Xxyz,ZyzYxW->IWZXY', np.identity(ncvs), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    # precond_caa_caa_bab_bab += 1/12 * einsum('II,Xxyz,ZyzxWY->IWZXY', np.identity(ncvs), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    # precond_caa_caa_bab_bab += 1/12 * einsum('II,Xxyz,ZyzxYW->IWZXY', np.identity(ncvs), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    # precond_caa_caa_bab_bab -= 1/2 * einsum('II,YxZy,WyXx->IWZXY', np.identity(ncvs), v_aaaa, rdm_ccaa, optimize = einsum_type)
    # precond_caa_caa_bab_bab += 1/3 * einsum('II,YxyZ,WyXx->IWZXY', np.identity(ncvs), v_aaaa, rdm_ccaa, optimize = einsum_type)
    # precond_caa_caa_bab_bab += 1/6 * einsum('II,YxyZ,WyxX->IWZXY', np.identity(ncvs), v_aaaa, rdm_ccaa, optimize = einsum_type)
    # precond_caa_caa_bab_bab += 1/4 * einsum('II,Yxyz,XZxWyz->IWZXY', np.identity(ncvs), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    # precond_caa_caa_bab_bab -= 1/12 * einsum('II,Yxyz,XZxWzy->IWZXY', np.identity(ncvs), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    # precond_caa_caa_bab_bab += 1/12 * einsum('II,Yxyz,XZxyWz->IWZXY', np.identity(ncvs), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    # precond_caa_caa_bab_bab -= 1/12 * einsum('II,Yxyz,XZxyzW->IWZXY', np.identity(ncvs), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    # precond_caa_caa_bab_bab -= 1/12 * einsum('II,Yxyz,XZxzWy->IWZXY', np.identity(ncvs), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    # precond_caa_caa_bab_bab -= 1/12 * einsum('II,Yxyz,XZxzyW->IWZXY', np.identity(ncvs), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    # precond_caa_caa_bab_bab += 1/2 * einsum('I,II,YZ,XW->IWZXY', e_cvs, np.identity(ncvs), np.identity(ncas), rdm_ca, optimize = einsum_type)
    # precond_caa_caa_bab_bab += 1/2 * einsum('Xx,II,YZ,xW->IWZXY', h_aa, np.identity(ncvs), np.identity(ncas), rdm_ca, optimize = einsum_type)
    # precond_caa_caa_bab_bab += 1/2 * einsum('Xxyz,II,YZ,Wxyz->IWZXY', v_aaaa, np.identity(ncvs), np.identity(ncas), rdm_ccaa, optimize = einsum_type)

    # precond_caa_caa_aaa_abb  = 1/6 * einsum('I,II,WYXZ->IWZXY', e_cvs, np.identity(ncvs), rdm_ccaa, optimize = einsum_type)
    # precond_caa_caa_aaa_abb += 1/3 * einsum('I,II,WYZX->IWZXY', e_cvs, np.identity(ncvs), rdm_ccaa, optimize = einsum_type)
    # precond_caa_caa_aaa_abb += 1/3 * einsum('Xx,II,WYZx->IWZXY', h_aa, np.identity(ncvs), rdm_ccaa, optimize = einsum_type)
    # precond_caa_caa_aaa_abb += 1/6 * einsum('Xx,II,WYxZ->IWZXY', h_aa, np.identity(ncvs), rdm_ccaa, optimize = einsum_type)
    # precond_caa_caa_aaa_abb -= 1/6 * einsum('Yx,II,WxXZ->IWZXY', h_aa, np.identity(ncvs), rdm_ccaa, optimize = einsum_type)
    # precond_caa_caa_aaa_abb -= 1/3 * einsum('Yx,II,WxZX->IWZXY', h_aa, np.identity(ncvs), rdm_ccaa, optimize = einsum_type)
    # precond_caa_caa_aaa_abb += 1/6 * einsum('II,XZxy,WYxy->IWZXY', np.identity(ncvs), v_aaaa, rdm_ccaa, optimize = einsum_type)
    # precond_caa_caa_aaa_abb += 1/3 * einsum('II,XZxy,WYyx->IWZXY', np.identity(ncvs), v_aaaa, rdm_ccaa, optimize = einsum_type)
    # precond_caa_caa_aaa_abb += 1/4 * einsum('II,Xxyz,ZyzWYx->IWZXY', np.identity(ncvs), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    # precond_caa_caa_aaa_abb -= 1/12 * einsum('II,Xxyz,ZyzWxY->IWZXY', np.identity(ncvs), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    # precond_caa_caa_aaa_abb += 1/12 * einsum('II,Xxyz,ZyzYWx->IWZXY', np.identity(ncvs), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    # precond_caa_caa_aaa_abb -= 1/12 * einsum('II,Xxyz,ZyzYxW->IWZXY', np.identity(ncvs), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    # precond_caa_caa_aaa_abb -= 1/12 * einsum('II,Xxyz,ZyzxWY->IWZXY', np.identity(ncvs), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    # precond_caa_caa_aaa_abb -= 1/12 * einsum('II,Xxyz,ZyzxYW->IWZXY', np.identity(ncvs), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    # precond_caa_caa_aaa_abb -= 1/6 * einsum('II,YxyZ,WyXx->IWZXY', np.identity(ncvs), v_aaaa, rdm_ccaa, optimize = einsum_type)
    # precond_caa_caa_aaa_abb -= 1/3 * einsum('II,YxyZ,WyxX->IWZXY', np.identity(ncvs), v_aaaa, rdm_ccaa, optimize = einsum_type)
    # precond_caa_caa_aaa_abb -= 1/12 * einsum('II,Yxyz,XZxWyz->IWZXY', np.identity(ncvs), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    # precond_caa_caa_aaa_abb += 1/12 * einsum('II,Yxyz,XZxWzy->IWZXY', np.identity(ncvs), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    # precond_caa_caa_aaa_abb -= 1/4 * einsum('II,Yxyz,XZxyWz->IWZXY', np.identity(ncvs), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    # precond_caa_caa_aaa_abb += 1/12 * einsum('II,Yxyz,XZxyzW->IWZXY', np.identity(ncvs), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    # precond_caa_caa_aaa_abb += 1/12 * einsum('II,Yxyz,XZxzWy->IWZXY', np.identity(ncvs), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    # precond_caa_caa_aaa_abb += 1/12 * einsum('II,Yxyz,XZxzyW->IWZXY', np.identity(ncvs), v_aaaa, rdm_cccaaa, optimize = einsum_type)

    # precond_caa_caa_abb_aaa  = 1/6 * einsum('I,II,WYXZ->IWZXY', e_cvs, np.identity(ncvs), rdm_ccaa, optimize = einsum_type)
    # precond_caa_caa_abb_aaa += 1/3 * einsum('I,II,WYZX->IWZXY', e_cvs, np.identity(ncvs), rdm_ccaa, optimize = einsum_type)
    # precond_caa_caa_abb_aaa += 1/3 * einsum('Xx,II,WYZx->IWZXY', h_aa, np.identity(ncvs), rdm_ccaa, optimize = einsum_type)
    # precond_caa_caa_abb_aaa += 1/6 * einsum('Xx,II,WYxZ->IWZXY', h_aa, np.identity(ncvs), rdm_ccaa, optimize = einsum_type)
    # precond_caa_caa_abb_aaa -= 1/6 * einsum('Yx,II,WxXZ->IWZXY', h_aa, np.identity(ncvs), rdm_ccaa, optimize = einsum_type)
    # precond_caa_caa_abb_aaa -= 1/3 * einsum('Yx,II,WxZX->IWZXY', h_aa, np.identity(ncvs), rdm_ccaa, optimize = einsum_type)
    # precond_caa_caa_abb_aaa += 1/6 * einsum('II,XZxy,WYxy->IWZXY', np.identity(ncvs), v_aaaa, rdm_ccaa, optimize = einsum_type)
    # precond_caa_caa_abb_aaa += 1/3 * einsum('II,XZxy,WYyx->IWZXY', np.identity(ncvs), v_aaaa, rdm_ccaa, optimize = einsum_type)
    # precond_caa_caa_abb_aaa += 1/4 * einsum('II,Xxyz,ZyzWYx->IWZXY', np.identity(ncvs), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    # precond_caa_caa_abb_aaa -= 1/12 * einsum('II,Xxyz,ZyzWxY->IWZXY', np.identity(ncvs), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    # precond_caa_caa_abb_aaa += 1/12 * einsum('II,Xxyz,ZyzYWx->IWZXY', np.identity(ncvs), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    # precond_caa_caa_abb_aaa -= 1/12 * einsum('II,Xxyz,ZyzYxW->IWZXY', np.identity(ncvs), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    # precond_caa_caa_abb_aaa -= 1/12 * einsum('II,Xxyz,ZyzxWY->IWZXY', np.identity(ncvs), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    # precond_caa_caa_abb_aaa -= 1/12 * einsum('II,Xxyz,ZyzxYW->IWZXY', np.identity(ncvs), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    # precond_caa_caa_abb_aaa -= 1/6 * einsum('II,YxyZ,WyXx->IWZXY', np.identity(ncvs), v_aaaa, rdm_ccaa, optimize = einsum_type)
    # precond_caa_caa_abb_aaa -= 1/3 * einsum('II,YxyZ,WyxX->IWZXY', np.identity(ncvs), v_aaaa, rdm_ccaa, optimize = einsum_type)
    # precond_caa_caa_abb_aaa -= 1/12 * einsum('II,Yxyz,XZxWyz->IWZXY', np.identity(ncvs), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    # precond_caa_caa_abb_aaa += 1/12 * einsum('II,Yxyz,XZxWzy->IWZXY', np.identity(ncvs), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    # precond_caa_caa_abb_aaa -= 1/4 * einsum('II,Yxyz,XZxyWz->IWZXY', np.identity(ncvs), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    # precond_caa_caa_abb_aaa += 1/12 * einsum('II,Yxyz,XZxyzW->IWZXY', np.identity(ncvs), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    # precond_caa_caa_abb_aaa += 1/12 * einsum('II,Yxyz,XZxzWy->IWZXY', np.identity(ncvs), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    # precond_caa_caa_abb_aaa += 1/12 * einsum('II,Yxyz,XZxzyW->IWZXY', np.identity(ncvs), v_aaaa, rdm_cccaaa, optimize = einsum_type)

    precond_caa_aaa =- 1/6 * einsum('I,II,WYXZ->IWZXY', e_cvs, np.identity(ncvs), rdm_ccaa, optimize = einsum_type)
    precond_caa_aaa += 1/6 * einsum('I,II,WYZX->IWZXY', e_cvs, np.identity(ncvs), rdm_ccaa, optimize = einsum_type)
    precond_caa_aaa += 1/6 * einsum('Xx,II,WYZx->IWZXY', h_aa, np.identity(ncvs), rdm_ccaa, optimize = einsum_type)
    precond_caa_aaa -= 1/6 * einsum('Xx,II,WYxZ->IWZXY', h_aa, np.identity(ncvs), rdm_ccaa, optimize = einsum_type)
    precond_caa_aaa -= 1/2 * einsum('YZ,II,XW->IWZXY', h_aa, np.identity(ncvs), rdm_ca, optimize = einsum_type)
    precond_caa_aaa += 1/6 * einsum('Yx,II,WxXZ->IWZXY', h_aa, np.identity(ncvs), rdm_ccaa, optimize = einsum_type)
    precond_caa_aaa -= 1/6 * einsum('Yx,II,WxZX->IWZXY', h_aa, np.identity(ncvs), rdm_ccaa, optimize = einsum_type)
    precond_caa_aaa -= 1/6 * einsum('II,XZxy,WYxy->IWZXY', np.identity(ncvs), v_aaaa, rdm_ccaa, optimize = einsum_type)
    precond_caa_aaa += 1/6 * einsum('II,XZxy,WYyx->IWZXY', np.identity(ncvs), v_aaaa, rdm_ccaa, optimize = einsum_type)
    precond_caa_aaa += 1/6 * einsum('II,Xxyz,ZyzWYx->IWZXY', np.identity(ncvs), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    precond_caa_aaa -= 1/6 * einsum('II,Xxyz,ZyzYWx->IWZXY', np.identity(ncvs), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    precond_caa_aaa -= 1/2 * einsum('II,YxZy,WyXx->IWZXY', np.identity(ncvs), v_aaaa, rdm_ccaa, optimize = einsum_type)
    precond_caa_aaa += 1/6 * einsum('II,YxyZ,WyXx->IWZXY', np.identity(ncvs), v_aaaa, rdm_ccaa, optimize = einsum_type)
    precond_caa_aaa -= 1/6 * einsum('II,YxyZ,WyxX->IWZXY', np.identity(ncvs), v_aaaa, rdm_ccaa, optimize = einsum_type)
    precond_caa_aaa += 1/6 * einsum('II,Yxyz,XZxWyz->IWZXY', np.identity(ncvs), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    precond_caa_aaa -= 1/6 * einsum('II,Yxyz,XZxyWz->IWZXY', np.identity(ncvs), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    precond_caa_aaa += 1/2 * einsum('I,II,YZ,XW->IWZXY', e_cvs, np.identity(ncvs), np.identity(ncas), rdm_ca, optimize = einsum_type)
    precond_caa_aaa += 1/2 * einsum('Xx,II,YZ,xW->IWZXY', h_aa, np.identity(ncvs), np.identity(ncas), rdm_ca, optimize = einsum_type)
    precond_caa_aaa += 1/2 * einsum('Xxyz,II,YZ,Wxyz->IWZXY', v_aaaa, np.identity(ncvs), np.identity(ncas), rdm_ccaa, optimize = einsum_type)

    precond_caa_abb =- 1/6 * einsum('I,II,WYXZ->IWZXY', e_cvs, np.identity(ncvs), rdm_ccaa, optimize = einsum_type)
    precond_caa_abb += 1/6 * einsum('I,II,WYZX->IWZXY', e_cvs, np.identity(ncvs), rdm_ccaa, optimize = einsum_type)
    precond_caa_abb += 1/6 * einsum('Xx,II,WYZx->IWZXY', h_aa, np.identity(ncvs), rdm_ccaa, optimize = einsum_type)
    precond_caa_abb -= 1/6 * einsum('Xx,II,WYxZ->IWZXY', h_aa, np.identity(ncvs), rdm_ccaa, optimize = einsum_type)
    precond_caa_abb -= 1/2 * einsum('YZ,II,XW->IWZXY', h_aa, np.identity(ncvs), rdm_ca, optimize = einsum_type)
    precond_caa_abb += 1/6 * einsum('Yx,II,WxXZ->IWZXY', h_aa, np.identity(ncvs), rdm_ccaa, optimize = einsum_type)
    precond_caa_abb -= 1/6 * einsum('Yx,II,WxZX->IWZXY', h_aa, np.identity(ncvs), rdm_ccaa, optimize = einsum_type)
    precond_caa_abb -= 1/6 * einsum('II,XZxy,WYxy->IWZXY', np.identity(ncvs), v_aaaa, rdm_ccaa, optimize = einsum_type)
    precond_caa_abb += 1/6 * einsum('II,XZxy,WYyx->IWZXY', np.identity(ncvs), v_aaaa, rdm_ccaa, optimize = einsum_type)
    precond_caa_abb += 1/6 * einsum('II,Xxyz,ZyzWYx->IWZXY', np.identity(ncvs), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    precond_caa_abb -= 1/6 * einsum('II,Xxyz,ZyzYWx->IWZXY', np.identity(ncvs), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    precond_caa_abb -= 1/2 * einsum('II,YxZy,WyXx->IWZXY', np.identity(ncvs), v_aaaa, rdm_ccaa, optimize = einsum_type)
    precond_caa_abb += 1/6 * einsum('II,YxyZ,WyXx->IWZXY', np.identity(ncvs), v_aaaa, rdm_ccaa, optimize = einsum_type)
    precond_caa_abb -= 1/6 * einsum('II,YxyZ,WyxX->IWZXY', np.identity(ncvs), v_aaaa, rdm_ccaa, optimize = einsum_type)
    precond_caa_abb += 1/6 * einsum('II,Yxyz,XZxWyz->IWZXY', np.identity(ncvs), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    precond_caa_abb -= 1/6 * einsum('II,Yxyz,XZxyWz->IWZXY', np.identity(ncvs), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    precond_caa_abb += 1/2 * einsum('I,II,YZ,XW->IWZXY', e_cvs, np.identity(ncvs), np.identity(ncas), rdm_ca, optimize = einsum_type)
    precond_caa_abb += 1/2 * einsum('Xx,II,YZ,xW->IWZXY', h_aa, np.identity(ncvs), np.identity(ncas), rdm_ca, optimize = einsum_type)
    precond_caa_abb += 1/2 * einsum('Xxyz,II,YZ,Wxyz->IWZXY', v_aaaa, np.identity(ncvs), np.identity(ncas), rdm_ccaa, optimize = einsum_type)

    precond_caa_bab =- 1/3 * einsum('I,II,WYXZ->IWZXY', e_cvs, np.identity(ncvs), rdm_ccaa, optimize = einsum_type)
    precond_caa_bab -= 1/6 * einsum('I,II,WYZX->IWZXY', e_cvs, np.identity(ncvs), rdm_ccaa, optimize = einsum_type)
    precond_caa_bab -= 1/6 * einsum('Xx,II,WYZx->IWZXY', h_aa, np.identity(ncvs), rdm_ccaa, optimize = einsum_type)
    precond_caa_bab -= 1/3 * einsum('Xx,II,WYxZ->IWZXY', h_aa, np.identity(ncvs), rdm_ccaa, optimize = einsum_type)
    precond_caa_bab -= 1/2 * einsum('YZ,II,XW->IWZXY', h_aa, np.identity(ncvs), rdm_ca, optimize = einsum_type)
    precond_caa_bab += 1/3 * einsum('Yx,II,WxXZ->IWZXY', h_aa, np.identity(ncvs), rdm_ccaa, optimize = einsum_type)
    precond_caa_bab += 1/6 * einsum('Yx,II,WxZX->IWZXY', h_aa, np.identity(ncvs), rdm_ccaa, optimize = einsum_type)
    precond_caa_bab -= 1/3 * einsum('II,XZxy,WYxy->IWZXY', np.identity(ncvs), v_aaaa, rdm_ccaa, optimize = einsum_type)
    precond_caa_bab -= 1/6 * einsum('II,XZxy,WYyx->IWZXY', np.identity(ncvs), v_aaaa, rdm_ccaa, optimize = einsum_type)
    precond_caa_bab -= 1/12 * einsum('II,Xxyz,ZyzWYx->IWZXY', np.identity(ncvs), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    precond_caa_bab += 1/12 * einsum('II,Xxyz,ZyzWxY->IWZXY', np.identity(ncvs), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    precond_caa_bab -= 1/4 * einsum('II,Xxyz,ZyzYWx->IWZXY', np.identity(ncvs), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    precond_caa_bab += 1/12 * einsum('II,Xxyz,ZyzYxW->IWZXY', np.identity(ncvs), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    precond_caa_bab += 1/12 * einsum('II,Xxyz,ZyzxWY->IWZXY', np.identity(ncvs), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    precond_caa_bab += 1/12 * einsum('II,Xxyz,ZyzxYW->IWZXY', np.identity(ncvs), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    precond_caa_bab -= 1/2 * einsum('II,YxZy,WyXx->IWZXY', np.identity(ncvs), v_aaaa, rdm_ccaa, optimize = einsum_type)
    precond_caa_bab += 1/3 * einsum('II,YxyZ,WyXx->IWZXY', np.identity(ncvs), v_aaaa, rdm_ccaa, optimize = einsum_type)
    precond_caa_bab += 1/6 * einsum('II,YxyZ,WyxX->IWZXY', np.identity(ncvs), v_aaaa, rdm_ccaa, optimize = einsum_type)
    precond_caa_bab += 1/4 * einsum('II,Yxyz,XZxWyz->IWZXY', np.identity(ncvs), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    precond_caa_bab -= 1/12 * einsum('II,Yxyz,XZxWzy->IWZXY', np.identity(ncvs), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    precond_caa_bab += 1/12 * einsum('II,Yxyz,XZxyWz->IWZXY', np.identity(ncvs), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    precond_caa_bab -= 1/12 * einsum('II,Yxyz,XZxyzW->IWZXY', np.identity(ncvs), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    precond_caa_bab -= 1/12 * einsum('II,Yxyz,XZxzWy->IWZXY', np.identity(ncvs), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    precond_caa_bab -= 1/12 * einsum('II,Yxyz,XZxzyW->IWZXY', np.identity(ncvs), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    precond_caa_bab += 1/2 * einsum('I,II,YZ,XW->IWZXY', e_cvs, np.identity(ncvs), np.identity(ncas), rdm_ca, optimize = einsum_type)
    precond_caa_bab += 1/2 * einsum('Xx,II,YZ,xW->IWZXY', h_aa, np.identity(ncvs), np.identity(ncas), rdm_ca, optimize = einsum_type)
    precond_caa_bab += 1/2 * einsum('Xxyz,II,YZ,Wxyz->IWZXY', v_aaaa, np.identity(ncvs), np.identity(ncas), rdm_ccaa, optimize = einsum_type)

    # Off-diagonal terms
    precond_caa_aaa_abb  = 1/6 * einsum('I,II,WYXZ->IWZXY', e_cvs, np.identity(ncvs), rdm_ccaa, optimize = einsum_type)
    precond_caa_aaa_abb += 1/3 * einsum('I,II,WYZX->IWZXY', e_cvs, np.identity(ncvs), rdm_ccaa, optimize = einsum_type)
    precond_caa_aaa_abb += 1/3 * einsum('Xx,II,WYZx->IWZXY', h_aa, np.identity(ncvs), rdm_ccaa, optimize = einsum_type)
    precond_caa_aaa_abb += 1/6 * einsum('Xx,II,WYxZ->IWZXY', h_aa, np.identity(ncvs), rdm_ccaa, optimize = einsum_type)
    precond_caa_aaa_abb -= 1/6 * einsum('Yx,II,WxXZ->IWZXY', h_aa, np.identity(ncvs), rdm_ccaa, optimize = einsum_type)
    precond_caa_aaa_abb -= 1/3 * einsum('Yx,II,WxZX->IWZXY', h_aa, np.identity(ncvs), rdm_ccaa, optimize = einsum_type)
    precond_caa_aaa_abb += 1/6 * einsum('II,XZxy,WYxy->IWZXY', np.identity(ncvs), v_aaaa, rdm_ccaa, optimize = einsum_type)
    precond_caa_aaa_abb += 1/3 * einsum('II,XZxy,WYyx->IWZXY', np.identity(ncvs), v_aaaa, rdm_ccaa, optimize = einsum_type)
    precond_caa_aaa_abb += 1/4 * einsum('II,Xxyz,ZyzWYx->IWZXY', np.identity(ncvs), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    precond_caa_aaa_abb -= 1/12 * einsum('II,Xxyz,ZyzWxY->IWZXY', np.identity(ncvs), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    precond_caa_aaa_abb += 1/12 * einsum('II,Xxyz,ZyzYWx->IWZXY', np.identity(ncvs), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    precond_caa_aaa_abb -= 1/12 * einsum('II,Xxyz,ZyzYxW->IWZXY', np.identity(ncvs), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    precond_caa_aaa_abb -= 1/12 * einsum('II,Xxyz,ZyzxWY->IWZXY', np.identity(ncvs), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    precond_caa_aaa_abb -= 1/12 * einsum('II,Xxyz,ZyzxYW->IWZXY', np.identity(ncvs), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    precond_caa_aaa_abb -= 1/6 * einsum('II,YxyZ,WyXx->IWZXY', np.identity(ncvs), v_aaaa, rdm_ccaa, optimize = einsum_type)
    precond_caa_aaa_abb -= 1/3 * einsum('II,YxyZ,WyxX->IWZXY', np.identity(ncvs), v_aaaa, rdm_ccaa, optimize = einsum_type)
    precond_caa_aaa_abb -= 1/12 * einsum('II,Yxyz,XZxWyz->IWZXY', np.identity(ncvs), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    precond_caa_aaa_abb += 1/12 * einsum('II,Yxyz,XZxWzy->IWZXY', np.identity(ncvs), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    precond_caa_aaa_abb -= 1/4 * einsum('II,Yxyz,XZxyWz->IWZXY', np.identity(ncvs), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    precond_caa_aaa_abb += 1/12 * einsum('II,Yxyz,XZxyzW->IWZXY', np.identity(ncvs), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    precond_caa_aaa_abb += 1/12 * einsum('II,Yxyz,XZxzWy->IWZXY', np.identity(ncvs), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    precond_caa_aaa_abb += 1/12 * einsum('II,Yxyz,XZxzyW->IWZXY', np.identity(ncvs), v_aaaa, rdm_cccaaa, optimize = einsum_type)

    precond_caa_abb_aaa  = 1/6 * einsum('I,II,WYXZ->IWZXY', e_cvs, np.identity(ncvs), rdm_ccaa, optimize = einsum_type)
    precond_caa_abb_aaa += 1/3 * einsum('I,II,WYZX->IWZXY', e_cvs, np.identity(ncvs), rdm_ccaa, optimize = einsum_type)
    precond_caa_abb_aaa += 1/3 * einsum('Xx,II,WYZx->IWZXY', h_aa, np.identity(ncvs), rdm_ccaa, optimize = einsum_type)
    precond_caa_abb_aaa += 1/6 * einsum('Xx,II,WYxZ->IWZXY', h_aa, np.identity(ncvs), rdm_ccaa, optimize = einsum_type)
    precond_caa_abb_aaa -= 1/6 * einsum('Yx,II,WxXZ->IWZXY', h_aa, np.identity(ncvs), rdm_ccaa, optimize = einsum_type)
    precond_caa_abb_aaa -= 1/3 * einsum('Yx,II,WxZX->IWZXY', h_aa, np.identity(ncvs), rdm_ccaa, optimize = einsum_type)
    precond_caa_abb_aaa += 1/6 * einsum('II,XZxy,WYxy->IWZXY', np.identity(ncvs), v_aaaa, rdm_ccaa, optimize = einsum_type)
    precond_caa_abb_aaa += 1/3 * einsum('II,XZxy,WYyx->IWZXY', np.identity(ncvs), v_aaaa, rdm_ccaa, optimize = einsum_type)
    precond_caa_abb_aaa += 1/4 * einsum('II,Xxyz,ZyzWYx->IWZXY', np.identity(ncvs), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    precond_caa_abb_aaa -= 1/12 * einsum('II,Xxyz,ZyzWxY->IWZXY', np.identity(ncvs), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    precond_caa_abb_aaa += 1/12 * einsum('II,Xxyz,ZyzYWx->IWZXY', np.identity(ncvs), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    precond_caa_abb_aaa -= 1/12 * einsum('II,Xxyz,ZyzYxW->IWZXY', np.identity(ncvs), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    precond_caa_abb_aaa -= 1/12 * einsum('II,Xxyz,ZyzxWY->IWZXY', np.identity(ncvs), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    precond_caa_abb_aaa -= 1/12 * einsum('II,Xxyz,ZyzxYW->IWZXY', np.identity(ncvs), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    precond_caa_abb_aaa -= 1/6 * einsum('II,YxyZ,WyXx->IWZXY', np.identity(ncvs), v_aaaa, rdm_ccaa, optimize = einsum_type)
    precond_caa_abb_aaa -= 1/3 * einsum('II,YxyZ,WyxX->IWZXY', np.identity(ncvs), v_aaaa, rdm_ccaa, optimize = einsum_type)
    precond_caa_abb_aaa -= 1/12 * einsum('II,Yxyz,XZxWyz->IWZXY', np.identity(ncvs), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    precond_caa_abb_aaa += 1/12 * einsum('II,Yxyz,XZxWzy->IWZXY', np.identity(ncvs), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    precond_caa_abb_aaa -= 1/4 * einsum('II,Yxyz,XZxyWz->IWZXY', np.identity(ncvs), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    precond_caa_abb_aaa += 1/12 * einsum('II,Yxyz,XZxyzW->IWZXY', np.identity(ncvs), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    precond_caa_abb_aaa += 1/12 * einsum('II,Yxyz,XZxzWy->IWZXY', np.identity(ncvs), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    precond_caa_abb_aaa += 1/12 * einsum('II,Yxyz,XZxzyW->IWZXY', np.identity(ncvs), v_aaaa, rdm_cccaaa, optimize = einsum_type)

    # print(">>> SA precond C-CAA (a-aaa) norm: {:}".format(np.linalg.norm(precond_c_caa_a_aaa)))
    # print(">>> SA precond C-CAA (a-abb) norm: {:}".format(np.linalg.norm(precond_c_caa_a_abb)))
    # print(">>> SA precond C-CAA (b-bab) norm: {:}".format(np.linalg.norm(precond_c_caa_b_bab)))

    # print(">>> SA precond CAA-CAA (aaa-aaa) norm: {:}".format(np.linalg.norm(precond_caa_caa_aaa_aaa)))
    # print(">>> SA precond CAA-CAA (abb-abb) norm: {:}".format(np.linalg.norm(precond_caa_caa_abb_abb)))
    # print(">>> SA precond CAA-CAA (bab-bab) norm: {:}".format(np.linalg.norm(precond_caa_caa_bab_bab)))

    print(">>> SA precond CAA-CAA (aaa-abb) norm: {:}".format(np.linalg.norm(precond_caa_aaa_abb)))
    # print(">>> SA precond CAA-CAA (abb-aaa) norm: {:}".format(np.linalg.norm(precond_caa_caa_abb_aaa)))

    ## Building C-CAA matrix
    dim_XY = ncas * ncas
    dim_c_caa = 3 * dim_XY

    precond_aa_i = 1
    precond_aa_f = precond_aa_i + dim_XY
    precond_bb_i = precond_aa_f
    precond_bb_f = precond_bb_i + dim_XY
    precond_ab_i = precond_bb_f
    precond_ab_f = precond_ab_i + dim_XY

    precond_temp = np.zeros((ncvs, (1 + dim_c_caa), (1 + dim_c_caa)))
    precond_temp[:, 0, 0] = np.diag(M_00[s_c:f_c, s_c:f_c]).copy()

    # precond_temp[:, 0, precond_aa_i:precond_aa_f] = precond_c_caa_a_aaa.reshape(ncvs, ncas * ncas).copy()
    # precond_temp[:, 0, precond_bb_i:precond_bb_f] = precond_c_caa_a_abb.reshape(ncvs, ncas * ncas).copy()
    # # precond_temp[:, 0, precond_ab_i:precond_ab_f] = precond_c_caa_b_bab.reshape(ncvs, ncas * ncas).copy()
    # precond_temp[:, precond_aa_i:precond_ab_f, 0] = precond_temp[:, 0, precond_aa_i:precond_ab_f].copy()

    precond_temp[:, precond_aa_i:precond_aa_f, precond_aa_i:precond_aa_f] = precond_caa_aaa.reshape(ncvs, ncas * ncas, ncas * ncas).copy()
    precond_temp[:, precond_aa_i:precond_aa_f, precond_bb_i:precond_bb_f] = precond_caa_aaa_abb.reshape(ncvs, ncas * ncas, ncas * ncas).copy()

    precond_temp[:, precond_bb_i:precond_bb_f, precond_bb_i:precond_bb_f] = precond_caa_abb.reshape(ncvs, ncas * ncas, ncas * ncas).copy()
    precond_temp[:, precond_bb_i:precond_bb_f, precond_aa_i:precond_aa_f] = precond_caa_abb_aaa.reshape(ncvs, ncas * ncas, ncas * ncas).copy()

    precond_temp[:, precond_ab_i:precond_ab_f, precond_ab_i:precond_ab_f] = precond_caa_bab.reshape(ncvs, ncas * ncas, ncas * ncas).copy()

    np.save('precond_caa_aaa_sa', precond_caa_aaa)
    np.save('precond_temp_sa', precond_temp)
    precond_temp = einsum('IXY,XP,YP->IP', precond_temp, S12_c_caa, S12_c_caa, optimize = einsum_type)

    print(">>> SA S12_c_caa: {:}".format(S12_c_caa.shape))
    np.save('precond_sa', precond_temp)
    precond_temp_out = np.around(precond_temp, decimals=6)
    np.savetxt('precond_sa_full.txt', precond_temp_out, fmt='%.6f')
    precond_temp_out = np.unique(precond_temp_out)
    np.savetxt('precond_sa.txt', precond_temp_out, fmt='%.6f')

    precond[ho_s_c_caa:ho_f_c_caa] = precond_temp.reshape(-1)

    ## Building C-CAA matrix
    # dim_XY = ncas * ncas
    # dim_c_caa = 2 * dim_XY

    # precond_aa_i = 1
    # precond_aa_f = precond_aa_i + dim_XY
    # precond_bb_i = precond_aa_f
    # precond_bb_f = precond_bb_i + dim_XY

    # precond_temp = np.zeros((ncvs, (1 + dim_c_caa), (1 + dim_c_caa)))
    # precond_temp[:, 0, 0] = np.diag(M_00[s_c:f_c, s_c:f_c]).copy()

    # precond_temp[:, 0, precond_aa_i:precond_aa_f] = precond_c_caa_a_aaa.reshape(ncvs, ncas * ncas).copy()
    # precond_temp[:, 0, precond_bb_i:precond_bb_f] = precond_c_caa_a_abb.reshape(ncvs, ncas * ncas).copy()
    # precond_temp[:, precond_aa_i:precond_bb_f, 0] = precond_temp[:, 0, precond_aa_i:precond_bb_f].copy()

    # precond_temp[:, precond_aa_i:precond_aa_f, precond_aa_i:precond_aa_f] = precond_caa_caa_aaa_aaa.reshape(ncvs, ncas * ncas, ncas * ncas).copy()
    # precond_temp[:, precond_aa_i:precond_aa_f, precond_bb_i:precond_bb_f] = precond_caa_caa_aaa_abb.reshape(ncvs, ncas * ncas, ncas * ncas).copy()

    # precond_temp[:, precond_bb_i:precond_bb_f, precond_bb_i:precond_bb_f] = precond_caa_caa_abb_abb.reshape(ncvs, ncas * ncas, ncas * ncas).copy()
    # precond_temp[:, precond_bb_i:precond_bb_f, precond_aa_i:precond_aa_f] = precond_caa_caa_abb_aaa.reshape(ncvs, ncas * ncas, ncas * ncas).copy()

    # precond_temp = einsum('IXY,XP,YP->IP', precond_temp, S12_c_caa, S12_c_caa, optimize = einsum_type)

    # np.savetxt('S12_c_caa_sa.txt', S12_c_caa, fmt='%.6f')
    # np.savetxt('precond_c_caa__a_aaa_sa.txt', precond_c_caa_a_aaa.reshape(ncvs, ncas * ncas), fmt='%.6f')
    # np.savetxt('precond_c_caa__a_abb_sa.txt', precond_c_caa_a_abb.reshape(ncvs, ncas * ncas), fmt='%.6f')

    # np.savetxt('precond_caa_caa__aaa_aaa_sa.txt', precond_caa_caa_aaa_aaa.reshape(ncvs, ncas * ncas, ncas * ncas)[0], fmt='%.6f')
    # np.savetxt('precond_caa_caa__aaa_abb_sa.txt', precond_caa_caa_aaa_abb.reshape(ncvs, ncas * ncas, ncas * ncas)[0], fmt='%.6f')
    # np.savetxt('precond_caa_caa__abb_aaa_sa.txt', precond_caa_caa_abb_aaa.reshape(ncvs, ncas * ncas, ncas * ncas)[0], fmt='%.6f')
    # np.savetxt('precond_caa_caa__abb_abb_sa.txt', precond_caa_caa_abb_abb.reshape(ncvs, ncas * ncas, ncas * ncas)[0], fmt='%.6f')
    # np.savetxt('precond_caa_caa__bab_bab_sa.txt', precond_caa_caa_bab_bab.reshape(ncvs, ncas * ncas, ncas * ncas)[0], fmt='%.6f')

    # precond_temp_out = np.around(precond_temp, decimals=6)
    # np.savetxt('precond_sa_full.txt', precond_temp_out.reshape(-1), fmt='%.6f')
    # precond_temp_out = np.unique(precond_temp_out)
    # np.savetxt('precond_sa.txt', precond_temp_out.reshape(-1), fmt='%.6f')
    # precond[ho_s_c_caa:ho_f_c_caa] = precond_temp.reshape(-1)

    # Multiply by -1.0, since we are solving for -M C = -S C E
    precond *= (-1.0)

    print ("Time for computing preconditioner:                %f sec\n" % (time.time() - start_time))
    sys.stdout.flush()

    return precond

def compute_M_01_caa(mr_adc):

    start_time = time.time()

    print ("Computing M(h0-h1) blocks...")
    sys.stdout.flush()

    shift = 100000.0
    M_C_CAA = shift
    M_C_CCE = shift
    M_C_CAE = shift
    M_C_CCA = shift

    nval = mr_adc.nval
    if nval > 0:
        M_C_CVE = shift
        M_C_CVA = shift

    print ("Time for computing M(h0-h1) blocks:               %f sec\n" % (time.time() - start_time))
    sys.stdout.flush()

    if nval > 0:
        return M_C_CAA, M_C_CCE, M_C_CVE, M_C_CAE, M_C_CCA, M_C_CVA
    else:
        return M_C_CAA, M_C_CCE, M_C_CAE, M_C_CCA

def apply_S_12_caa(mr_adc, X, transpose = False):

    # Einsum definition from kernel
    einsum = mr_adc.interface.einsum
    einsum_type = mr_adc.interface.einsum_type

    # Dimensions
    nextern = mr_adc.nextern
    ncvs = mr_adc.ncvs
    nval = mr_adc.nval

    ho_s_c = mr_adc.h_orth.s_c
    ho_f_c = mr_adc.h_orth.f_c
    ho_s_c_caa = mr_adc.h_orth.s_c_caa
    ho_f_c_caa = mr_adc.h_orth.f_c_caa
    ho_s_cce = mr_adc.h_orth.s_cce
    ho_f_cce = mr_adc.h_orth.f_cce
    ho_s_cae = mr_adc.h_orth.s_cae
    ho_f_cae = mr_adc.h_orth.f_cae
    ho_s_ace = mr_adc.h_orth.s_ace
    ho_f_ace = mr_adc.h_orth.f_ace
    ho_s_cca = mr_adc.h_orth.s_cca
    ho_f_cca = mr_adc.h_orth.f_cca
    s_c = mr_adc.h0.s_c
    f_c = mr_adc.h0.f_c
    s_caa = mr_adc.h1.s_caa
    f_caa = mr_adc.h1.f_caa
    s_cce = mr_adc.h1.s_cce
    f_cce = mr_adc.h1.f_cce
    s_cae = mr_adc.h1.s_cae
    f_cae = mr_adc.h1.f_cae
    s_ace = mr_adc.h1.s_ace
    f_ace = mr_adc.h1.f_ace
    s_cca = mr_adc.h1.s_cca
    f_cca = mr_adc.h1.f_cca

    if nval > 0:
        ho_s_cve = mr_adc.h_orth.s_cve
        ho_f_cve = mr_adc.h_orth.f_cve
        ho_s_vce = mr_adc.h_orth.s_vce
        ho_f_vce = mr_adc.h_orth.f_vce

        ho_s_cva = mr_adc.h_orth.s_cva
        ho_f_cva = mr_adc.h_orth.f_cva
        ho_s_vca = mr_adc.h_orth.s_vca
        ho_f_vca = mr_adc.h_orth.f_vca

        s_cve = mr_adc.h1.s_cve
        f_cve = mr_adc.h1.f_cve
        s_vce = mr_adc.h1.s_vce
        f_vce = mr_adc.h1.f_vce

        s_cva = mr_adc.h1.s_cva
        f_cva = mr_adc.h1.f_cva
        s_vca = mr_adc.h1.s_vca
        f_vca = mr_adc.h1.f_vca

    # Overlap matrices
    S12_c_caa = mr_adc.S12.c_caa
    S12_cae = mr_adc.S12.cae
    S12_cca = mr_adc.S12.cca

    Xt = None

    if transpose:
        if (X.shape[0] != (mr_adc.h0.dim + mr_adc.h1.dim)):
            raise Exception("Dimensions do not match when applying S_12 transpose")

        Xt = np.zeros(mr_adc.h_orth.dim)

        # C and CAA -> C_CAA
        # temp = np.zeros((ncvs, S12_c_caa.shape[0]))
        # temp[:,0] = X[s_c:f_c].copy()
        # temp[:,1:] = X[s_caa:f_caa].reshape(ncvs, -1).copy()
        # Xt[ho_s_c_caa:ho_f_c_caa] = np.dot(temp, S12_c_caa).reshape(-1).copy()

        ncas = mr_adc.ncas
        n_caa = ncvs * ncas * ncas
        n_aa = ncas * ncas
        s_caa_aaa = s_caa
        f_caa_aaa = s_caa_aaa + n_caa
        s_caa_abb = f_caa_aaa
        f_caa_abb = s_caa_abb + n_caa
        s_caa_bab = f_caa_abb
        f_caa_bab = s_caa_bab + n_caa

        temp = np.zeros((ncvs, S12_c_caa.shape[0]))
        temp[:,0] = X[s_c:f_c].copy()
        temp[:,1:n_aa+1] = X[s_caa_aaa:f_caa_aaa].reshape(ncvs, -1).copy()
        temp[:,n_aa+1:(2*n_aa)+1] = X[s_caa_abb:f_caa_abb].reshape(ncvs, -1).copy()
        temp[:,(2*n_aa)+1:] = X[s_caa_bab:f_caa_bab].reshape(ncvs, -1).copy()
        Xt[ho_s_c_caa:ho_f_c_caa] = np.dot(temp, S12_c_caa).reshape(-1).copy()

    else:
        if (X.shape[0] != (mr_adc.h_orth.dim)):
            raise Exception("Dimensions do not match when applying S_12")

        Xt = np.zeros(mr_adc.h0.dim + mr_adc.h1.dim)

        # C_CAA -> C and CAA
        # temp = X[ho_s_c_caa:ho_f_c_caa].reshape(ncvs, -1).copy()
        # temp = np.dot(temp, S12_c_caa.T)
        # Xt[s_c:f_c] = temp[:,0].copy()
        # Xt[s_caa:f_caa] = temp[:,1:].reshape(-1).copy()

        temp = X[ho_s_c_caa:ho_f_c_caa].reshape(ncvs, -1).copy()
        temp = np.dot(temp, S12_c_caa.T)
        Xt[s_c:f_c] = temp[:,0].copy()

        ncas = mr_adc.ncas
        n_caa = ncvs * ncas * ncas
        n_aa = ncas * ncas
        s_caa_aaa = s_caa
        f_caa_aaa = s_caa_aaa + n_caa
        s_caa_abb = f_caa_aaa
        f_caa_abb = s_caa_abb + n_caa
        s_caa_bab = f_caa_abb
        f_caa_bab = s_caa_bab + n_caa

        Xt[s_caa_aaa:f_caa_aaa] = temp[:,1:(n_aa)+1].reshape(-1).copy()
        Xt[s_caa_abb:f_caa_abb] = temp[:,(n_aa)+1:(2*n_aa)+1].reshape(-1).copy()
        Xt[s_caa_bab:f_caa_bab] = temp[:,(2*n_aa)+1:].reshape(-1).copy()

    return Xt

def compute_sigma_vector_caa(mr_adc, M_00, M_01, M_11, Xt):

    # Einsum definition from kernel
    einsum = mr_adc.interface.einsum
    einsum_type = mr_adc.interface.einsum_type

    # Variables from kernel
    ncvs = mr_adc.ncvs
    nval = mr_adc.nval
    ncas = mr_adc.ncas
    nextern = mr_adc.nextern

    e_cvs = mr_adc.mo_energy.x
    e_core = mr_adc.mo_energy.c
    if nval > 0:
        e_val = mr_adc.mo_energy.v
    e_extern = mr_adc.mo_energy.e

    # One-electron integrals
    h_aa = mr_adc.h1eff.aa

    # Two-electrons integrals
    v_aaaa = mr_adc.v2e.aaaa

    # Reduced Density Matrices
    rdm_ca = mr_adc.rdm.ca
    rdm_ccaa = mr_adc.rdm.ccaa
    rdm_cccaaa = mr_adc.rdm.cccaaa

    # Dimensions
    s_c = mr_adc.h0.s_c
    f_c = mr_adc.h0.f_c
    s_caa = mr_adc.h1.s_caa
    f_caa = mr_adc.h1.f_caa
    s_cce = mr_adc.h1.s_cce
    f_cce = mr_adc.h1.f_cce
    s_cae = mr_adc.h1.s_cae
    f_cae = mr_adc.h1.f_cae
    s_ace = mr_adc.h1.s_ace
    f_ace = mr_adc.h1.f_ace
    s_cca = mr_adc.h1.s_cca
    f_cca = mr_adc.h1.f_cca
    if nval > 0:
        s_cve = mr_adc.h1.s_cve
        f_cve = mr_adc.h1.f_cve
        s_vce = mr_adc.h1.s_vce
        f_vce = mr_adc.h1.f_vce

        s_cva = mr_adc.h1.s_cva
        f_cva = mr_adc.h1.f_cva
        s_vca = mr_adc.h1.s_vca
        f_vca = mr_adc.h1.f_vca

    cvs_tril_ind = np.tril_indices(ncvs, k=-1)

    # (CASCI + C) -> (CASCI + C)
    sigma = np.zeros_like(Xt)

    # h0-h0 contributions
    sigma[:mr_adc.h0.dim] = np.dot(M_00, Xt[:mr_adc.h0.dim])

    # h1-h1 contributions
    # CAA <- CAA
    n_caa = ncas * ncas * ncvs
    s_caa_aaa = mr_adc.h1.s_caa
    f_caa_aaa = s_caa_aaa + n_caa
    s_caa_abb = f_caa_aaa
    f_caa_abb = s_caa_abb + n_caa
    s_caa_bab = f_caa_abb
    f_caa_bab = s_caa_bab + n_caa

    X_aaa = Xt[s_caa_aaa:f_caa_aaa].reshape(ncvs, ncas, ncas).copy()
    X_abb = Xt[s_caa_abb:f_caa_abb].reshape(ncvs, ncas, ncas).copy()
    X_bab = Xt[s_caa_bab:f_caa_bab].reshape(ncvs, ncas, ncas).copy()

    sigma_caa_aaa  = 1/2 * einsum('KxZ,K,xW->KWZ', X_aaa, e_cvs, rdm_ca, optimize = einsum_type)
    sigma_caa_aaa += 1/6 * einsum('Kxy,K,WyZx->KWZ', X_aaa, e_cvs, rdm_ccaa, optimize = einsum_type)
    sigma_caa_aaa -= 1/6 * einsum('Kxy,K,WyxZ->KWZ', X_aaa, e_cvs, rdm_ccaa, optimize = einsum_type)
    sigma_caa_aaa += 1/2 * einsum('KxZ,xy,yW->KWZ', X_aaa, h_aa, rdm_ca, optimize = einsum_type)
    sigma_caa_aaa -= 1/2 * einsum('Kxy,Zy,xW->KWZ', X_aaa, h_aa, rdm_ca, optimize = einsum_type)
    sigma_caa_aaa += 1/6 * einsum('Kxy,xz,WyZz->KWZ', X_aaa, h_aa, rdm_ccaa, optimize = einsum_type)
    sigma_caa_aaa -= 1/6 * einsum('Kxy,xz,WyzZ->KWZ', X_aaa, h_aa, rdm_ccaa, optimize = einsum_type)
    sigma_caa_aaa -= 1/6 * einsum('Kxy,yz,WzZx->KWZ', X_aaa, h_aa, rdm_ccaa, optimize = einsum_type)
    sigma_caa_aaa += 1/6 * einsum('Kxy,yz,WzxZ->KWZ', X_aaa, h_aa, rdm_ccaa, optimize = einsum_type)
    sigma_caa_aaa += 1/2 * einsum('KxZ,xyzw,Wyzw->KWZ', X_aaa, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_caa_aaa += 1/6 * einsum('Kxy,Zxzw,Wyzw->KWZ', X_aaa, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_caa_aaa -= 1/6 * einsum('Kxy,Zxzw,Wywz->KWZ', X_aaa, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_caa_aaa -= 1/2 * einsum('Kxy,Zzyw,Wzxw->KWZ', X_aaa, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_caa_aaa += 1/6 * einsum('Kxy,Zzwy,Wzxw->KWZ', X_aaa, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_caa_aaa -= 1/6 * einsum('Kxy,Zzwy,Wzwx->KWZ', X_aaa, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_caa_aaa += 1/6 * einsum('Kxy,xzwu,ZwuWyz->KWZ', X_aaa, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_caa_aaa -= 1/6 * einsum('Kxy,xzwu,ZwuyWz->KWZ', X_aaa, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_caa_aaa -= 1/6 * einsum('Kxy,yzwu,ZxzWwu->KWZ', X_aaa, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_caa_aaa += 1/6 * einsum('Kxy,yzwu,ZxzwWu->KWZ', X_aaa, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_caa_aaa += 1/3 * einsum('Kxy,K,WyZx->KWZ', X_abb, e_cvs, rdm_ccaa, optimize = einsum_type)
    sigma_caa_aaa += 1/6 * einsum('Kxy,K,WyxZ->KWZ', X_abb, e_cvs, rdm_ccaa, optimize = einsum_type)
    sigma_caa_aaa += 1/3 * einsum('Kxy,xz,WyZz->KWZ', X_abb, h_aa, rdm_ccaa, optimize = einsum_type)
    sigma_caa_aaa += 1/6 * einsum('Kxy,xz,WyzZ->KWZ', X_abb, h_aa, rdm_ccaa, optimize = einsum_type)
    sigma_caa_aaa -= 1/3 * einsum('Kxy,yz,WzZx->KWZ', X_abb, h_aa, rdm_ccaa, optimize = einsum_type)
    sigma_caa_aaa -= 1/6 * einsum('Kxy,yz,WzxZ->KWZ', X_abb, h_aa, rdm_ccaa, optimize = einsum_type)
    sigma_caa_aaa += 1/3 * einsum('Kxy,Zxzw,Wyzw->KWZ', X_abb, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_caa_aaa += 1/6 * einsum('Kxy,Zxzw,Wywz->KWZ', X_abb, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_caa_aaa -= 1/6 * einsum('Kxy,Zzwy,Wzxw->KWZ', X_abb, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_caa_aaa -= 1/3 * einsum('Kxy,Zzwy,Wzwx->KWZ', X_abb, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_caa_aaa += 1/4 * einsum('Kxy,xzwu,ZwuWyz->KWZ', X_abb, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_caa_aaa -= 1/12 * einsum('Kxy,xzwu,ZwuWzy->KWZ', X_abb, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_caa_aaa += 1/12 * einsum('Kxy,xzwu,ZwuyWz->KWZ', X_abb, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_caa_aaa -= 1/12 * einsum('Kxy,xzwu,ZwuyzW->KWZ', X_abb, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_caa_aaa -= 1/12 * einsum('Kxy,xzwu,ZwuzWy->KWZ', X_abb, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_caa_aaa -= 1/12 * einsum('Kxy,xzwu,ZwuzyW->KWZ', X_abb, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_caa_aaa -= 1/4 * einsum('Kxy,yzwu,ZxzWwu->KWZ', X_abb, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_caa_aaa += 1/12 * einsum('Kxy,yzwu,ZxzWuw->KWZ', X_abb, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_caa_aaa -= 1/12 * einsum('Kxy,yzwu,ZxzwWu->KWZ', X_abb, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_caa_aaa += 1/12 * einsum('Kxy,yzwu,ZxzwuW->KWZ', X_abb, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_caa_aaa += 1/12 * einsum('Kxy,yzwu,ZxzuWw->KWZ', X_abb, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_caa_aaa += 1/12 * einsum('Kxy,yzwu,ZxzuwW->KWZ', X_abb, v_aaaa, rdm_cccaaa, optimize = einsum_type)

    sigma_caa_abb  = 1/3 * einsum('Kxy,K,WyZx->KWZ', X_aaa, e_cvs, rdm_ccaa, optimize = einsum_type)
    sigma_caa_abb += 1/6 * einsum('Kxy,K,WyxZ->KWZ', X_aaa, e_cvs, rdm_ccaa, optimize = einsum_type)
    sigma_caa_abb += 1/3 * einsum('Kxy,xz,WyZz->KWZ', X_aaa, h_aa, rdm_ccaa, optimize = einsum_type)
    sigma_caa_abb += 1/6 * einsum('Kxy,xz,WyzZ->KWZ', X_aaa, h_aa, rdm_ccaa, optimize = einsum_type)
    sigma_caa_abb -= 1/3 * einsum('Kxy,yz,WzZx->KWZ', X_aaa, h_aa, rdm_ccaa, optimize = einsum_type)
    sigma_caa_abb -= 1/6 * einsum('Kxy,yz,WzxZ->KWZ', X_aaa, h_aa, rdm_ccaa, optimize = einsum_type)
    sigma_caa_abb += 1/3 * einsum('Kxy,Zxzw,Wyzw->KWZ', X_aaa, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_caa_abb += 1/6 * einsum('Kxy,Zxzw,Wywz->KWZ', X_aaa, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_caa_abb -= 1/6 * einsum('Kxy,Zzwy,Wzxw->KWZ', X_aaa, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_caa_abb -= 1/3 * einsum('Kxy,Zzwy,Wzwx->KWZ', X_aaa, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_caa_abb += 1/4 * einsum('Kxy,xzwu,ZwuWyz->KWZ', X_aaa, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_caa_abb -= 1/12 * einsum('Kxy,xzwu,ZwuWzy->KWZ', X_aaa, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_caa_abb += 1/12 * einsum('Kxy,xzwu,ZwuyWz->KWZ', X_aaa, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_caa_abb -= 1/12 * einsum('Kxy,xzwu,ZwuyzW->KWZ', X_aaa, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_caa_abb -= 1/12 * einsum('Kxy,xzwu,ZwuzWy->KWZ', X_aaa, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_caa_abb -= 1/12 * einsum('Kxy,xzwu,ZwuzyW->KWZ', X_aaa, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_caa_abb -= 1/4 * einsum('Kxy,yzwu,ZxzWwu->KWZ', X_aaa, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_caa_abb += 1/12 * einsum('Kxy,yzwu,ZxzWuw->KWZ', X_aaa, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_caa_abb -= 1/12 * einsum('Kxy,yzwu,ZxzwWu->KWZ', X_aaa, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_caa_abb += 1/12 * einsum('Kxy,yzwu,ZxzwuW->KWZ', X_aaa, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_caa_abb += 1/12 * einsum('Kxy,yzwu,ZxzuWw->KWZ', X_aaa, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_caa_abb += 1/12 * einsum('Kxy,yzwu,ZxzuwW->KWZ', X_aaa, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_caa_abb += 1/2 * einsum('KxZ,K,xW->KWZ', X_abb, e_cvs, rdm_ca, optimize = einsum_type)
    sigma_caa_abb += 1/6 * einsum('Kxy,K,WyZx->KWZ', X_abb, e_cvs, rdm_ccaa, optimize = einsum_type)
    sigma_caa_abb -= 1/6 * einsum('Kxy,K,WyxZ->KWZ', X_abb, e_cvs, rdm_ccaa, optimize = einsum_type)
    sigma_caa_abb += 1/2 * einsum('KxZ,xy,yW->KWZ', X_abb, h_aa, rdm_ca, optimize = einsum_type)
    sigma_caa_abb -= 1/2 * einsum('Kxy,Zy,xW->KWZ', X_abb, h_aa, rdm_ca, optimize = einsum_type)
    sigma_caa_abb += 1/6 * einsum('Kxy,xz,WyZz->KWZ', X_abb, h_aa, rdm_ccaa, optimize = einsum_type)
    sigma_caa_abb -= 1/6 * einsum('Kxy,xz,WyzZ->KWZ', X_abb, h_aa, rdm_ccaa, optimize = einsum_type)
    sigma_caa_abb -= 1/6 * einsum('Kxy,yz,WzZx->KWZ', X_abb, h_aa, rdm_ccaa, optimize = einsum_type)
    sigma_caa_abb += 1/6 * einsum('Kxy,yz,WzxZ->KWZ', X_abb, h_aa, rdm_ccaa, optimize = einsum_type)
    sigma_caa_abb += 1/2 * einsum('KxZ,xyzw,Wyzw->KWZ', X_abb, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_caa_abb += 1/6 * einsum('Kxy,Zxzw,Wyzw->KWZ', X_abb, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_caa_abb -= 1/6 * einsum('Kxy,Zxzw,Wywz->KWZ', X_abb, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_caa_abb -= 1/2 * einsum('Kxy,Zzyw,Wzxw->KWZ', X_abb, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_caa_abb += 1/6 * einsum('Kxy,Zzwy,Wzxw->KWZ', X_abb, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_caa_abb -= 1/6 * einsum('Kxy,Zzwy,Wzwx->KWZ', X_abb, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_caa_abb += 1/6 * einsum('Kxy,xzwu,ZwuWyz->KWZ', X_abb, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_caa_abb -= 1/6 * einsum('Kxy,xzwu,ZwuyWz->KWZ', X_abb, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_caa_abb -= 1/6 * einsum('Kxy,yzwu,ZxzWwu->KWZ', X_abb, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_caa_abb += 1/6 * einsum('Kxy,yzwu,ZxzwWu->KWZ', X_abb, v_aaaa, rdm_cccaaa, optimize = einsum_type)

    sigma_caa_bab  = 1/2 * einsum('KxZ,K,xW->KWZ', X_bab, e_cvs, rdm_ca, optimize = einsum_type)
    sigma_caa_bab -= 1/6 * einsum('Kxy,K,WyZx->KWZ', X_bab, e_cvs, rdm_ccaa, optimize = einsum_type)
    sigma_caa_bab -= 1/3 * einsum('Kxy,K,WyxZ->KWZ', X_bab, e_cvs, rdm_ccaa, optimize = einsum_type)
    sigma_caa_bab += 1/2 * einsum('KxZ,xy,yW->KWZ', X_bab, h_aa, rdm_ca, optimize = einsum_type)
    sigma_caa_bab -= 1/2 * einsum('Kxy,Zy,xW->KWZ', X_bab, h_aa, rdm_ca, optimize = einsum_type)
    sigma_caa_bab -= 1/6 * einsum('Kxy,xz,WyZz->KWZ', X_bab, h_aa, rdm_ccaa, optimize = einsum_type)
    sigma_caa_bab -= 1/3 * einsum('Kxy,xz,WyzZ->KWZ', X_bab, h_aa, rdm_ccaa, optimize = einsum_type)
    sigma_caa_bab += 1/6 * einsum('Kxy,yz,WzZx->KWZ', X_bab, h_aa, rdm_ccaa, optimize = einsum_type)
    sigma_caa_bab += 1/3 * einsum('Kxy,yz,WzxZ->KWZ', X_bab, h_aa, rdm_ccaa, optimize = einsum_type)
    sigma_caa_bab += 1/2 * einsum('KxZ,xyzw,Wyzw->KWZ', X_bab, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_caa_bab -= 1/6 * einsum('Kxy,Zxzw,Wyzw->KWZ', X_bab, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_caa_bab -= 1/3 * einsum('Kxy,Zxzw,Wywz->KWZ', X_bab, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_caa_bab -= 1/2 * einsum('Kxy,Zzyw,Wzxw->KWZ', X_bab, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_caa_bab += 1/3 * einsum('Kxy,Zzwy,Wzxw->KWZ', X_bab, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_caa_bab += 1/6 * einsum('Kxy,Zzwy,Wzwx->KWZ', X_bab, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_caa_bab -= 1/12 * einsum('Kxy,xzwu,ZwuWyz->KWZ', X_bab, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_caa_bab += 1/12 * einsum('Kxy,xzwu,ZwuWzy->KWZ', X_bab, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_caa_bab -= 1/4 * einsum('Kxy,xzwu,ZwuyWz->KWZ', X_bab, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_caa_bab += 1/12 * einsum('Kxy,xzwu,ZwuyzW->KWZ', X_bab, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_caa_bab += 1/12 * einsum('Kxy,xzwu,ZwuzWy->KWZ', X_bab, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_caa_bab += 1/12 * einsum('Kxy,xzwu,ZwuzyW->KWZ', X_bab, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_caa_bab += 1/12 * einsum('Kxy,yzwu,ZxzWwu->KWZ', X_bab, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_caa_bab -= 1/12 * einsum('Kxy,yzwu,ZxzWuw->KWZ', X_bab, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_caa_bab += 1/4 * einsum('Kxy,yzwu,ZxzwWu->KWZ', X_bab, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_caa_bab -= 1/12 * einsum('Kxy,yzwu,ZxzwuW->KWZ', X_bab, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_caa_bab -= 1/12 * einsum('Kxy,yzwu,ZxzuWw->KWZ', X_bab, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_caa_bab -= 1/12 * einsum('Kxy,yzwu,ZxzuwW->KWZ', X_bab, v_aaaa, rdm_cccaaa, optimize = einsum_type)

    sigma[s_caa_aaa:f_caa_aaa] += sigma_caa_aaa.reshape(-1).copy()
    sigma[s_caa_abb:f_caa_abb] += sigma_caa_abb.reshape(-1).copy()
    sigma[s_caa_bab:f_caa_bab] += sigma_caa_bab.reshape(-1).copy()

    return sigma

## CAA block (v2)
def compute_excitation_manifolds_caa(mr_adc):

    # MR-ADC(0) and MR-ADC(1)
    mr_adc.h0.n_c = mr_adc.ncvs
    mr_adc.h0.dim = mr_adc.h0.n_c # Total dimension of h0

    mr_adc.h0.s_c = 0
    mr_adc.h0.f_c = mr_adc.h0.s_c + mr_adc.h0.n_c

    print("Dimension of h0 excitation manifold:                       %d" % mr_adc.h0.dim)

    # MR-ADC(2)
    mr_adc.h1.dim = 0
    mr_adc.h_orth.dim = mr_adc.h0.dim

    if mr_adc.method in ("mr-adc(2)", "mr-adc(2)-x"):
        mr_adc.h1.n_caa = 2 * mr_adc.ncas * mr_adc.ncas * mr_adc.ncvs
        mr_adc.h1.n_cce = 0
        mr_adc.h1.n_cae = 0
        mr_adc.h1.n_ace = 0
        mr_adc.h1.n_cca = 0
        if mr_adc.nval > 0:
            mr_adc.h1.n_cve = 0
            mr_adc.h1.n_vce = 0
            mr_adc.h1.n_cva = 0
            mr_adc.h1.n_vca = 0
            mr_adc.h1.dim = (mr_adc.h1.n_caa + mr_adc.h1.n_cce + mr_adc.h1.n_cve + mr_adc.h1.n_vce +
                             mr_adc.h1.n_cae + mr_adc.h1.n_ace + mr_adc.h1.n_cca + mr_adc.h1.n_cva + mr_adc.h1.n_vca)
        else:
            mr_adc.h1.dim = mr_adc.h1.n_caa + mr_adc.h1.n_cce + mr_adc.h1.n_cae + mr_adc.h1.n_cae + mr_adc.h1.n_cca

        if mr_adc.nval > 0:
            mr_adc.h1.s_caa = mr_adc.h0.f_c
            mr_adc.h1.f_caa = mr_adc.h1.s_caa + mr_adc.h1.n_caa
            mr_adc.h1.s_cce = mr_adc.h1.f_caa
            mr_adc.h1.f_cce = mr_adc.h1.s_cce + mr_adc.h1.n_cce
            mr_adc.h1.s_cve = mr_adc.h1.f_cce
            mr_adc.h1.f_cve = mr_adc.h1.s_cve + mr_adc.h1.n_cve
            mr_adc.h1.s_vce = mr_adc.h1.f_cve
            mr_adc.h1.f_vce = mr_adc.h1.s_vce + mr_adc.h1.n_vce
            mr_adc.h1.s_cae = mr_adc.h1.f_vce
            mr_adc.h1.f_cae = mr_adc.h1.s_cae + mr_adc.h1.n_cae
            mr_adc.h1.s_ace = mr_adc.h1.f_cae
            mr_adc.h1.f_ace = mr_adc.h1.s_ace + mr_adc.h1.n_ace
            mr_adc.h1.s_cca = mr_adc.h1.f_ace
            mr_adc.h1.f_cca = mr_adc.h1.s_cca + mr_adc.h1.n_cca
            mr_adc.h1.s_cva = mr_adc.h1.f_cca
            mr_adc.h1.f_cva = mr_adc.h1.s_cva + mr_adc.h1.n_cva
            mr_adc.h1.s_vca = mr_adc.h1.f_cva
            mr_adc.h1.f_vca = mr_adc.h1.s_vca + mr_adc.h1.n_vca
        else:
            mr_adc.h1.s_caa = mr_adc.h0.f_c
            mr_adc.h1.f_caa = mr_adc.h1.s_caa + mr_adc.h1.n_caa
            mr_adc.h1.s_cce = mr_adc.h1.f_caa
            mr_adc.h1.f_cce = mr_adc.h1.s_cce + mr_adc.h1.n_cce
            mr_adc.h1.s_cae = mr_adc.h1.f_cce
            mr_adc.h1.f_cae = mr_adc.h1.s_cae + mr_adc.h1.n_cae
            mr_adc.h1.s_ace = mr_adc.h1.f_cae
            mr_adc.h1.f_ace = mr_adc.h1.s_ace + mr_adc.h1.n_ace
            mr_adc.h1.s_cca = mr_adc.h1.f_ace
            mr_adc.h1.f_cca = mr_adc.h1.s_cca + mr_adc.h1.n_cca

        print("Dimension of h1 excitation manifold:                       %d" % mr_adc.h1.dim)

        # Overlap for c - caa
        mr_adc.S12.c_caa = mr_adc_overlap.compute_S12_0p_projector(mr_adc)
        mr_adc.S12.cae = mr_adc_overlap.compute_S12_m1(mr_adc)
        mr_adc.S12.cca = mr_adc_overlap.compute_S12_p1(mr_adc)

        # Determine dimensions of orthogonalized excitation spaces
        mr_adc.h_orth.n_c = 0
        mr_adc.h_orth.n_c_caa = mr_adc.ncvs * mr_adc.S12.c_caa.shape[1]
        mr_adc.h_orth.n_cce = 0
        mr_adc.h_orth.n_cce = 0
        mr_adc.h_orth.n_cae = 0
        mr_adc.h_orth.n_ace = 0
        mr_adc.h_orth.n_cca = 0
        if mr_adc.nval > 0:
            mr_adc.h_orth.n_cve = 0
            mr_adc.h_orth.n_vce = 0
            mr_adc.h_orth.n_cva = 0
            mr_adc.h_orth.n_vca = 0
            mr_adc.h_orth.dim = (mr_adc.h_orth.n_c + mr_adc.h_orth.n_c_caa + mr_adc.h_orth.n_cce + mr_adc.h_orth.n_cve + mr_adc.h_orth.n_vce +
                                 mr_adc.h_orth.n_cae + mr_adc.h_orth.n_ace + mr_adc.h_orth.n_cca + mr_adc.h_orth.n_cva + mr_adc.h_orth.n_vca)
        else:
            mr_adc.h_orth.dim = mr_adc.h_orth.n_c + mr_adc.h_orth.n_c_caa + mr_adc.h_orth.n_cce + mr_adc.h_orth.n_cae + mr_adc.h_orth.n_ace + mr_adc.h_orth.n_cca

        if mr_adc.nval > 0:
            mr_adc.h_orth.s_c = 0
            mr_adc.h_orth.f_c = mr_adc.h_orth.n_c
            mr_adc.h_orth.s_c_caa = mr_adc.h_orth.f_c
            mr_adc.h_orth.f_c_caa = mr_adc.h_orth.s_c_caa + mr_adc.h_orth.n_c_caa
            mr_adc.h_orth.s_cce = mr_adc.h_orth.f_c_caa
            mr_adc.h_orth.f_cce = mr_adc.h_orth.s_cce + mr_adc.h_orth.n_cce
            mr_adc.h_orth.s_cve = mr_adc.h_orth.f_cce
            mr_adc.h_orth.f_cve = mr_adc.h_orth.s_cve + mr_adc.h_orth.n_cve
            mr_adc.h_orth.s_vce = mr_adc.h_orth.f_cve
            mr_adc.h_orth.f_vce = mr_adc.h_orth.s_vce + mr_adc.h_orth.n_vce
            mr_adc.h_orth.s_cae = mr_adc.h_orth.f_vce
            mr_adc.h_orth.f_cae = mr_adc.h_orth.s_cae + mr_adc.h_orth.n_cae
            mr_adc.h_orth.s_ace = mr_adc.h_orth.f_cae
            mr_adc.h_orth.f_ace = mr_adc.h_orth.s_ace + mr_adc.h_orth.n_ace
            mr_adc.h_orth.s_cca = mr_adc.h_orth.f_ace
            mr_adc.h_orth.f_cca = mr_adc.h_orth.s_cca + mr_adc.h_orth.n_cca
            mr_adc.h_orth.s_cva = mr_adc.h_orth.f_cca
            mr_adc.h_orth.f_cva = mr_adc.h_orth.s_cva + mr_adc.h_orth.n_cva
            mr_adc.h_orth.s_vca = mr_adc.h_orth.f_cva
            mr_adc.h_orth.f_vca = mr_adc.h_orth.s_vca + mr_adc.h_orth.n_vca
        else:
            mr_adc.h_orth.s_c = 0
            mr_adc.h_orth.f_c = mr_adc.h_orth.n_c
            mr_adc.h_orth.s_c_caa = mr_adc.h_orth.f_c
            mr_adc.h_orth.f_c_caa = mr_adc.h_orth.s_c_caa + mr_adc.h_orth.n_c_caa
            mr_adc.h_orth.s_cce = mr_adc.h_orth.f_c_caa
            mr_adc.h_orth.f_cce = mr_adc.h_orth.s_cce + mr_adc.h_orth.n_cce
            mr_adc.h_orth.s_cae = mr_adc.h_orth.f_cce
            mr_adc.h_orth.f_cae = mr_adc.h_orth.s_cae + mr_adc.h_orth.n_cae
            mr_adc.h_orth.s_ace = mr_adc.h_orth.f_ace
            mr_adc.h_orth.f_ace = mr_adc.h_orth.s_ace + mr_adc.h_orth.n_ace
            mr_adc.h_orth.s_cca = mr_adc.h_orth.f_ace
            mr_adc.h_orth.f_cca = mr_adc.h_orth.s_cca + mr_adc.h_orth.n_cca

    print("Total dimension of the excitation manifold:                %d" % (mr_adc.h0.dim + mr_adc.h1.dim))
    print("Dimension of the orthogonalized excitation manifold:       %d\n" % (mr_adc.h_orth.dim))
    sys.stdout.flush()

    if (mr_adc.h_orth.dim < mr_adc.nroots):
        mr_adc.nroots = mr_adc.h_orth.dim

    return mr_adc

def compute_preconditioner_caa(mr_adc, M_00):

    start_time = time.time()

    # Einsum definition from kernel
    einsum = mr_adc.interface.einsum
    einsum_type = mr_adc.interface.einsum_type

    if mr_adc.method in ("mr-adc(0)", "mr-adc(1)"):

        # Multiply by -1.0, since we are solving for -M C = -S C E
        return (-1.0 * np.diag(M_00))

    # Variables from kernel
    ncvs = mr_adc.ncvs
    nval = mr_adc.nval
    ncas = mr_adc.ncas
    nextern = mr_adc.nextern

    e_cvs = mr_adc.mo_energy.x
    e_val = mr_adc.mo_energy.v
    e_extern = mr_adc.mo_energy.e

    # One-electron integrals
    h_aa = mr_adc.h1eff.aa

    # Two-electrons integrals
    v_aaaa = mr_adc.v2e.aaaa

    v_xaxa = mr_adc.v2e.xaxa
    v_xaax = mr_adc.v2e.xaax

    # Reduced Density Matrices
    rdm_ca = mr_adc.rdm.ca
    rdm_ccaa = mr_adc.rdm.ccaa
    rdm_cccaaa = mr_adc.rdm.cccaaa

    # Overlap matrices
    S12_c_caa = mr_adc.S12.c_caa
    S12_cae = mr_adc.S12.cae
    S12_cca = mr_adc.S12.cca

    # Dimensions
    ho_s_c = mr_adc.h_orth.s_c
    ho_f_c = mr_adc.h_orth.f_c
    ho_s_c_caa = mr_adc.h_orth.s_c_caa
    ho_f_c_caa = mr_adc.h_orth.f_c_caa
    ho_s_cce = mr_adc.h_orth.s_cce
    ho_f_cce = mr_adc.h_orth.f_cce
    ho_s_cae = mr_adc.h_orth.s_cae
    ho_f_cae = mr_adc.h_orth.f_cae
    ho_s_ace = mr_adc.h_orth.s_ace
    ho_f_ace = mr_adc.h_orth.f_ace
    ho_s_cca = mr_adc.h_orth.s_cca
    ho_f_cca = mr_adc.h_orth.f_cca
    if nval > 0:
        ho_s_cve = mr_adc.h_orth.s_cve
        ho_f_cve = mr_adc.h_orth.f_cve
        ho_s_vce = mr_adc.h_orth.s_vce
        ho_f_vce = mr_adc.h_orth.f_vce

        ho_s_cva = mr_adc.h_orth.s_cva
        ho_f_cva = mr_adc.h_orth.f_cva
        ho_s_vca = mr_adc.h_orth.s_vca
        ho_f_vca = mr_adc.h_orth.f_vca

    s_c = mr_adc.h0.s_c
    f_c = mr_adc.h0.f_c

    cvs_tril_ind = np.tril_indices(ncvs, k=-1)
    # cas_ind = np.tril_indices(ncas, k=-1)

    # Build the preconditioner
    precond = np.zeros(mr_adc.h_orth.dim)

    # C and CAA
    # 0th-order
    precond_c_caa_a_aaa  = 1/2 * einsum('I,II,XY->IXY', e_cvs, np.identity(ncvs), rdm_ca, optimize = einsum_type)
    precond_c_caa_a_abb  = 1/2 * einsum('I,II,XY->IXY', e_cvs, np.identity(ncvs), rdm_ca, optimize = einsum_type)

    # 1st-order
    precond_c_caa_a_aaa += 1/2 * einsum('IxIY,Xx->IXY', v_xaxa, rdm_ca, optimize = einsum_type)
    precond_c_caa_a_aaa += 1/2 * einsum('IxIy,XyYx->IXY', v_xaxa, rdm_ccaa, optimize = einsum_type)
    precond_c_caa_a_aaa -= 1/2 * einsum('IxYI,Xx->IXY', v_xaax, rdm_ca, optimize = einsum_type)
    precond_c_caa_a_aaa -= 1/6 * einsum('IxyI,XyYx->IXY', v_xaax, rdm_ccaa, optimize = einsum_type)
    precond_c_caa_a_aaa += 1/6 * einsum('IxyI,XyxY->IXY', v_xaax, rdm_ccaa, optimize = einsum_type)
    precond_c_caa_a_aaa -= 1/2 * einsum('IxIy,xy,XY->IXY', v_xaxa, rdm_ca, rdm_ca, optimize = einsum_type)
    precond_c_caa_a_aaa += 1/4 * einsum('IxyI,xy,XY->IXY', v_xaax, rdm_ca, rdm_ca, optimize = einsum_type)

    precond_c_caa_a_abb += 1/2 * einsum('IxIY,Xx->IXY', v_xaxa, rdm_ca, optimize = einsum_type)
    precond_c_caa_a_abb += 1/2 * einsum('IxIy,XyYx->IXY', v_xaxa, rdm_ccaa, optimize = einsum_type)
    precond_c_caa_a_abb -= 1/3 * einsum('IxyI,XyYx->IXY', v_xaax, rdm_ccaa, optimize = einsum_type)
    precond_c_caa_a_abb -= 1/6 * einsum('IxyI,XyxY->IXY', v_xaax, rdm_ccaa, optimize = einsum_type)
    precond_c_caa_a_abb -= 1/2 * einsum('IxIy,xy,XY->IXY', v_xaxa, rdm_ca, rdm_ca, optimize = einsum_type)
    precond_c_caa_a_abb += 1/4 * einsum('IxyI,xy,XY->IXY', v_xaax, rdm_ca, rdm_ca, optimize = einsum_type)

    precond_caa_caa_aaa_aaa =- 1/6 * einsum('I,II,WYXZ->IWZXY', e_cvs, np.identity(ncvs), rdm_ccaa, optimize = einsum_type)
    precond_caa_caa_aaa_aaa += 1/6 * einsum('I,II,WYZX->IWZXY', e_cvs, np.identity(ncvs), rdm_ccaa, optimize = einsum_type)
    precond_caa_caa_aaa_aaa += 1/6 * einsum('Xx,II,WYZx->IWZXY', h_aa, np.identity(ncvs), rdm_ccaa, optimize = einsum_type)
    precond_caa_caa_aaa_aaa -= 1/6 * einsum('Xx,II,WYxZ->IWZXY', h_aa, np.identity(ncvs), rdm_ccaa, optimize = einsum_type)
    precond_caa_caa_aaa_aaa -= 1/2 * einsum('YZ,II,XW->IWZXY', h_aa, np.identity(ncvs), rdm_ca, optimize = einsum_type)
    precond_caa_caa_aaa_aaa += 1/6 * einsum('Yx,II,WxXZ->IWZXY', h_aa, np.identity(ncvs), rdm_ccaa, optimize = einsum_type)
    precond_caa_caa_aaa_aaa -= 1/6 * einsum('Yx,II,WxZX->IWZXY', h_aa, np.identity(ncvs), rdm_ccaa, optimize = einsum_type)
    precond_caa_caa_aaa_aaa -= 1/6 * einsum('II,XZxy,WYxy->IWZXY', np.identity(ncvs), v_aaaa, rdm_ccaa, optimize = einsum_type)
    precond_caa_caa_aaa_aaa += 1/6 * einsum('II,XZxy,WYyx->IWZXY', np.identity(ncvs), v_aaaa, rdm_ccaa, optimize = einsum_type)
    precond_caa_caa_aaa_aaa += 1/6 * einsum('II,Xxyz,ZyzWYx->IWZXY', np.identity(ncvs), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    precond_caa_caa_aaa_aaa -= 1/6 * einsum('II,Xxyz,ZyzYWx->IWZXY', np.identity(ncvs), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    precond_caa_caa_aaa_aaa -= 1/2 * einsum('II,YxZy,WyXx->IWZXY', np.identity(ncvs), v_aaaa, rdm_ccaa, optimize = einsum_type)
    precond_caa_caa_aaa_aaa += 1/6 * einsum('II,YxyZ,WyXx->IWZXY', np.identity(ncvs), v_aaaa, rdm_ccaa, optimize = einsum_type)
    precond_caa_caa_aaa_aaa -= 1/6 * einsum('II,YxyZ,WyxX->IWZXY', np.identity(ncvs), v_aaaa, rdm_ccaa, optimize = einsum_type)
    precond_caa_caa_aaa_aaa += 1/6 * einsum('II,Yxyz,XZxWyz->IWZXY', np.identity(ncvs), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    precond_caa_caa_aaa_aaa -= 1/6 * einsum('II,Yxyz,XZxyWz->IWZXY', np.identity(ncvs), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    precond_caa_caa_aaa_aaa += 1/2 * einsum('I,II,YZ,XW->IWZXY', e_cvs, np.identity(ncvs), np.identity(ncas), rdm_ca, optimize = einsum_type)
    precond_caa_caa_aaa_aaa += 1/2 * einsum('Xx,II,YZ,xW->IWZXY', h_aa, np.identity(ncvs), np.identity(ncas), rdm_ca, optimize = einsum_type)
    precond_caa_caa_aaa_aaa += 1/2 * einsum('Xxyz,II,YZ,Wxyz->IWZXY', v_aaaa, np.identity(ncvs), np.identity(ncas), rdm_ccaa, optimize = einsum_type)

    precond_caa_caa_abb_abb =- 1/6 * einsum('I,II,WYXZ->IWZXY', e_cvs, np.identity(ncvs), rdm_ccaa, optimize = einsum_type)
    precond_caa_caa_abb_abb += 1/6 * einsum('I,II,WYZX->IWZXY', e_cvs, np.identity(ncvs), rdm_ccaa, optimize = einsum_type)
    precond_caa_caa_abb_abb += 1/6 * einsum('Xx,II,WYZx->IWZXY', h_aa, np.identity(ncvs), rdm_ccaa, optimize = einsum_type)
    precond_caa_caa_abb_abb -= 1/6 * einsum('Xx,II,WYxZ->IWZXY', h_aa, np.identity(ncvs), rdm_ccaa, optimize = einsum_type)
    precond_caa_caa_abb_abb -= 1/2 * einsum('YZ,II,XW->IWZXY', h_aa, np.identity(ncvs), rdm_ca, optimize = einsum_type)
    precond_caa_caa_abb_abb += 1/6 * einsum('Yx,II,WxXZ->IWZXY', h_aa, np.identity(ncvs), rdm_ccaa, optimize = einsum_type)
    precond_caa_caa_abb_abb -= 1/6 * einsum('Yx,II,WxZX->IWZXY', h_aa, np.identity(ncvs), rdm_ccaa, optimize = einsum_type)
    precond_caa_caa_abb_abb -= 1/6 * einsum('II,XZxy,WYxy->IWZXY', np.identity(ncvs), v_aaaa, rdm_ccaa, optimize = einsum_type)
    precond_caa_caa_abb_abb += 1/6 * einsum('II,XZxy,WYyx->IWZXY', np.identity(ncvs), v_aaaa, rdm_ccaa, optimize = einsum_type)
    precond_caa_caa_abb_abb += 1/6 * einsum('II,Xxyz,ZyzWYx->IWZXY', np.identity(ncvs), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    precond_caa_caa_abb_abb -= 1/6 * einsum('II,Xxyz,ZyzYWx->IWZXY', np.identity(ncvs), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    precond_caa_caa_abb_abb -= 1/2 * einsum('II,YxZy,WyXx->IWZXY', np.identity(ncvs), v_aaaa, rdm_ccaa, optimize = einsum_type)
    precond_caa_caa_abb_abb += 1/6 * einsum('II,YxyZ,WyXx->IWZXY', np.identity(ncvs), v_aaaa, rdm_ccaa, optimize = einsum_type)
    precond_caa_caa_abb_abb -= 1/6 * einsum('II,YxyZ,WyxX->IWZXY', np.identity(ncvs), v_aaaa, rdm_ccaa, optimize = einsum_type)
    precond_caa_caa_abb_abb += 1/6 * einsum('II,Yxyz,XZxWyz->IWZXY', np.identity(ncvs), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    precond_caa_caa_abb_abb -= 1/6 * einsum('II,Yxyz,XZxyWz->IWZXY', np.identity(ncvs), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    precond_caa_caa_abb_abb += 1/2 * einsum('I,II,YZ,XW->IWZXY', e_cvs, np.identity(ncvs), np.identity(ncas), rdm_ca, optimize = einsum_type)
    precond_caa_caa_abb_abb += 1/2 * einsum('Xx,II,YZ,xW->IWZXY', h_aa, np.identity(ncvs), np.identity(ncas), rdm_ca, optimize = einsum_type)
    precond_caa_caa_abb_abb += 1/2 * einsum('Xxyz,II,YZ,Wxyz->IWZXY', v_aaaa, np.identity(ncvs), np.identity(ncas), rdm_ccaa, optimize = einsum_type)

    precond_caa_caa_aaa_abb  = 1/6 * einsum('I,II,WYXZ->IWZXY', e_cvs, np.identity(ncvs), rdm_ccaa, optimize = einsum_type)
    precond_caa_caa_aaa_abb += 1/3 * einsum('I,II,WYZX->IWZXY', e_cvs, np.identity(ncvs), rdm_ccaa, optimize = einsum_type)
    precond_caa_caa_aaa_abb += 1/3 * einsum('Xx,II,WYZx->IWZXY', h_aa, np.identity(ncvs), rdm_ccaa, optimize = einsum_type)
    precond_caa_caa_aaa_abb += 1/6 * einsum('Xx,II,WYxZ->IWZXY', h_aa, np.identity(ncvs), rdm_ccaa, optimize = einsum_type)
    precond_caa_caa_aaa_abb -= 1/6 * einsum('Yx,II,WxXZ->IWZXY', h_aa, np.identity(ncvs), rdm_ccaa, optimize = einsum_type)
    precond_caa_caa_aaa_abb -= 1/3 * einsum('Yx,II,WxZX->IWZXY', h_aa, np.identity(ncvs), rdm_ccaa, optimize = einsum_type)
    precond_caa_caa_aaa_abb += 1/6 * einsum('II,XZxy,WYxy->IWZXY', np.identity(ncvs), v_aaaa, rdm_ccaa, optimize = einsum_type)
    precond_caa_caa_aaa_abb += 1/3 * einsum('II,XZxy,WYyx->IWZXY', np.identity(ncvs), v_aaaa, rdm_ccaa, optimize = einsum_type)
    precond_caa_caa_aaa_abb += 1/4 * einsum('II,Xxyz,ZyzWYx->IWZXY', np.identity(ncvs), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    precond_caa_caa_aaa_abb -= 1/12 * einsum('II,Xxyz,ZyzWxY->IWZXY', np.identity(ncvs), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    precond_caa_caa_aaa_abb += 1/12 * einsum('II,Xxyz,ZyzYWx->IWZXY', np.identity(ncvs), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    precond_caa_caa_aaa_abb -= 1/12 * einsum('II,Xxyz,ZyzYxW->IWZXY', np.identity(ncvs), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    precond_caa_caa_aaa_abb -= 1/12 * einsum('II,Xxyz,ZyzxWY->IWZXY', np.identity(ncvs), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    precond_caa_caa_aaa_abb -= 1/12 * einsum('II,Xxyz,ZyzxYW->IWZXY', np.identity(ncvs), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    precond_caa_caa_aaa_abb -= 1/6 * einsum('II,YxyZ,WyXx->IWZXY', np.identity(ncvs), v_aaaa, rdm_ccaa, optimize = einsum_type)
    precond_caa_caa_aaa_abb -= 1/3 * einsum('II,YxyZ,WyxX->IWZXY', np.identity(ncvs), v_aaaa, rdm_ccaa, optimize = einsum_type)
    precond_caa_caa_aaa_abb -= 1/12 * einsum('II,Yxyz,XZxWyz->IWZXY', np.identity(ncvs), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    precond_caa_caa_aaa_abb += 1/12 * einsum('II,Yxyz,XZxWzy->IWZXY', np.identity(ncvs), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    precond_caa_caa_aaa_abb -= 1/4 * einsum('II,Yxyz,XZxyWz->IWZXY', np.identity(ncvs), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    precond_caa_caa_aaa_abb += 1/12 * einsum('II,Yxyz,XZxyzW->IWZXY', np.identity(ncvs), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    precond_caa_caa_aaa_abb += 1/12 * einsum('II,Yxyz,XZxzWy->IWZXY', np.identity(ncvs), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    precond_caa_caa_aaa_abb += 1/12 * einsum('II,Yxyz,XZxzyW->IWZXY', np.identity(ncvs), v_aaaa, rdm_cccaaa, optimize = einsum_type)

    precond_caa_caa_abb_aaa  = 1/6 * einsum('I,II,WYXZ->IWZXY', e_cvs, np.identity(ncvs), rdm_ccaa, optimize = einsum_type)
    precond_caa_caa_abb_aaa += 1/3 * einsum('I,II,WYZX->IWZXY', e_cvs, np.identity(ncvs), rdm_ccaa, optimize = einsum_type)
    precond_caa_caa_abb_aaa += 1/3 * einsum('Xx,II,WYZx->IWZXY', h_aa, np.identity(ncvs), rdm_ccaa, optimize = einsum_type)
    precond_caa_caa_abb_aaa += 1/6 * einsum('Xx,II,WYxZ->IWZXY', h_aa, np.identity(ncvs), rdm_ccaa, optimize = einsum_type)
    precond_caa_caa_abb_aaa -= 1/6 * einsum('Yx,II,WxXZ->IWZXY', h_aa, np.identity(ncvs), rdm_ccaa, optimize = einsum_type)
    precond_caa_caa_abb_aaa -= 1/3 * einsum('Yx,II,WxZX->IWZXY', h_aa, np.identity(ncvs), rdm_ccaa, optimize = einsum_type)
    precond_caa_caa_abb_aaa += 1/6 * einsum('II,XZxy,WYxy->IWZXY', np.identity(ncvs), v_aaaa, rdm_ccaa, optimize = einsum_type)
    precond_caa_caa_abb_aaa += 1/3 * einsum('II,XZxy,WYyx->IWZXY', np.identity(ncvs), v_aaaa, rdm_ccaa, optimize = einsum_type)
    precond_caa_caa_abb_aaa += 1/4 * einsum('II,Xxyz,ZyzWYx->IWZXY', np.identity(ncvs), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    precond_caa_caa_abb_aaa -= 1/12 * einsum('II,Xxyz,ZyzWxY->IWZXY', np.identity(ncvs), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    precond_caa_caa_abb_aaa += 1/12 * einsum('II,Xxyz,ZyzYWx->IWZXY', np.identity(ncvs), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    precond_caa_caa_abb_aaa -= 1/12 * einsum('II,Xxyz,ZyzYxW->IWZXY', np.identity(ncvs), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    precond_caa_caa_abb_aaa -= 1/12 * einsum('II,Xxyz,ZyzxWY->IWZXY', np.identity(ncvs), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    precond_caa_caa_abb_aaa -= 1/12 * einsum('II,Xxyz,ZyzxYW->IWZXY', np.identity(ncvs), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    precond_caa_caa_abb_aaa -= 1/6 * einsum('II,YxyZ,WyXx->IWZXY', np.identity(ncvs), v_aaaa, rdm_ccaa, optimize = einsum_type)
    precond_caa_caa_abb_aaa -= 1/3 * einsum('II,YxyZ,WyxX->IWZXY', np.identity(ncvs), v_aaaa, rdm_ccaa, optimize = einsum_type)
    precond_caa_caa_abb_aaa -= 1/12 * einsum('II,Yxyz,XZxWyz->IWZXY', np.identity(ncvs), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    precond_caa_caa_abb_aaa += 1/12 * einsum('II,Yxyz,XZxWzy->IWZXY', np.identity(ncvs), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    precond_caa_caa_abb_aaa -= 1/4 * einsum('II,Yxyz,XZxyWz->IWZXY', np.identity(ncvs), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    precond_caa_caa_abb_aaa += 1/12 * einsum('II,Yxyz,XZxyzW->IWZXY', np.identity(ncvs), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    precond_caa_caa_abb_aaa += 1/12 * einsum('II,Yxyz,XZxzWy->IWZXY', np.identity(ncvs), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    precond_caa_caa_abb_aaa += 1/12 * einsum('II,Yxyz,XZxzyW->IWZXY', np.identity(ncvs), v_aaaa, rdm_cccaaa, optimize = einsum_type)

    ## Building C-CAA matrix
    dim_XY = ncas * ncas
    dim_c_caa = 2 * dim_XY

    precond_aa_i = 1
    precond_aa_f = precond_aa_i + dim_XY
    precond_bb_i = precond_aa_f
    precond_bb_f = precond_bb_i + dim_XY

    precond_temp = np.zeros((ncvs, (1 + dim_c_caa), (1 + dim_c_caa)))
    precond_temp[:, 0, 0] = np.diag(M_00[s_c:f_c, s_c:f_c]).copy()

    precond_temp[:, 0, precond_aa_i:precond_aa_f] = precond_c_caa_a_aaa.reshape(ncvs, ncas * ncas).copy()
    precond_temp[:, 0, precond_bb_i:precond_bb_f] = precond_c_caa_a_abb.reshape(ncvs, ncas * ncas).copy()

    precond_temp[:, precond_aa_i:precond_aa_f, precond_aa_i:precond_aa_f] = precond_caa_caa_aaa_aaa.reshape(ncvs, ncas * ncas, ncas * ncas).copy()
    precond_temp[:, precond_aa_i:precond_aa_f, precond_bb_i:precond_bb_f] = precond_caa_caa_aaa_abb.reshape(ncvs, ncas * ncas, ncas * ncas).copy()

    precond_temp[:, precond_bb_i:precond_bb_f, precond_bb_i:precond_bb_f] = precond_caa_caa_abb_abb.reshape(ncvs, ncas * ncas, ncas * ncas).copy()
    precond_temp[:, precond_bb_i:precond_bb_f, precond_aa_i:precond_aa_f] = precond_caa_caa_abb_aaa.reshape(ncvs, ncas * ncas, ncas * ncas).copy()

    precond_temp = einsum('IXY,XP,YP->IP', precond_temp, S12_c_caa, S12_c_caa, optimize = einsum_type)

    precond_temp_out = np.around(precond_temp, decimals=6)
    # np.savetxt('precond_so_full.txt', precond_temp_out, fmt='%.6f')
    precond_temp_out = np.unique(precond_temp_out)
    np.savetxt('precond_sa.txt', precond_temp_out, fmt='%.6f')

    precond[ho_s_c_caa:ho_f_c_caa] = precond_temp.reshape(-1)

    # Multiply by -1.0, since we are solving for -M C = -S C E
    precond *= (-1.0)

    print ("Time for computing preconditioner:                %f sec\n" % (time.time() - start_time))
    sys.stdout.flush()

    return precond

def compute_M_01_caa(mr_adc):

    start_time = time.time()

    print ("Computing M(h0-h1) blocks...")
    sys.stdout.flush()

    shift = 100000.0
    M_C_CAA = shift
    M_C_CCE = shift
    M_C_CAE = shift
    M_C_CCA = shift

    nval = mr_adc.nval
    if nval > 0:
        M_C_CVE = shift
        M_C_CVA = shift

    print ("Time for computing M(h0-h1) blocks:               %f sec\n" % (time.time() - start_time))
    sys.stdout.flush()

    if nval > 0:
        return M_C_CAA, M_C_CCE, M_C_CVE, M_C_CAE, M_C_CCA, M_C_CVA
    else:
        return M_C_CAA, M_C_CCE, M_C_CAE, M_C_CCA

def apply_S_12_caa(mr_adc, X, transpose = False):

    # Einsum definition from kernel
    einsum = mr_adc.interface.einsum
    einsum_type = mr_adc.interface.einsum_type

    # Dimensions
    nextern = mr_adc.nextern
    ncvs = mr_adc.ncvs
    nval = mr_adc.nval

    ho_s_c = mr_adc.h_orth.s_c
    ho_f_c = mr_adc.h_orth.f_c
    ho_s_c_caa = mr_adc.h_orth.s_c_caa
    ho_f_c_caa = mr_adc.h_orth.f_c_caa
    ho_s_cce = mr_adc.h_orth.s_cce
    ho_f_cce = mr_adc.h_orth.f_cce
    ho_s_cae = mr_adc.h_orth.s_cae
    ho_f_cae = mr_adc.h_orth.f_cae
    ho_s_ace = mr_adc.h_orth.s_ace
    ho_f_ace = mr_adc.h_orth.f_ace
    ho_s_cca = mr_adc.h_orth.s_cca
    ho_f_cca = mr_adc.h_orth.f_cca
    s_c = mr_adc.h0.s_c
    f_c = mr_adc.h0.f_c
    s_caa = mr_adc.h1.s_caa
    f_caa = mr_adc.h1.f_caa
    s_cce = mr_adc.h1.s_cce
    f_cce = mr_adc.h1.f_cce
    s_cae = mr_adc.h1.s_cae
    f_cae = mr_adc.h1.f_cae
    s_ace = mr_adc.h1.s_ace
    f_ace = mr_adc.h1.f_ace
    s_cca = mr_adc.h1.s_cca
    f_cca = mr_adc.h1.f_cca

    if nval > 0:
        ho_s_cve = mr_adc.h_orth.s_cve
        ho_f_cve = mr_adc.h_orth.f_cve
        ho_s_vce = mr_adc.h_orth.s_vce
        ho_f_vce = mr_adc.h_orth.f_vce

        ho_s_cva = mr_adc.h_orth.s_cva
        ho_f_cva = mr_adc.h_orth.f_cva
        ho_s_vca = mr_adc.h_orth.s_vca
        ho_f_vca = mr_adc.h_orth.f_vca

        s_cve = mr_adc.h1.s_cve
        f_cve = mr_adc.h1.f_cve
        s_vce = mr_adc.h1.s_vce
        f_vce = mr_adc.h1.f_vce

        s_cva = mr_adc.h1.s_cva
        f_cva = mr_adc.h1.f_cva
        s_vca = mr_adc.h1.s_vca
        f_vca = mr_adc.h1.f_vca

    # Overlap matrices
    S12_c_caa = mr_adc.S12.c_caa
    S12_cae = mr_adc.S12.cae
    S12_cca = mr_adc.S12.cca

    Xt = None

    if transpose:
        if (X.shape[0] != (mr_adc.h0.dim + mr_adc.h1.dim)):
            raise Exception("Dimensions do not match when applying S_12 transpose")

        Xt = np.zeros(mr_adc.h_orth.dim)

        # C and CAA -> C_CAA
        temp = np.zeros((ncvs, S12_c_caa.shape[0]))
        temp[:,0] = X[s_c:f_c].copy()
        temp[:,1:] = X[s_caa:f_caa].reshape(ncvs, -1).copy()
        Xt[ho_s_c_caa:ho_f_c_caa] = np.dot(temp, S12_c_caa).reshape(-1).copy()

    else:
        if (X.shape[0] != (mr_adc.h_orth.dim)):
            raise Exception("Dimensions do not match when applying S_12")

        Xt = np.zeros(mr_adc.h0.dim + mr_adc.h1.dim)

        # C_CAA -> C and CAA
        temp = X[ho_s_c_caa:ho_f_c_caa].reshape(ncvs, -1).copy()
        temp = np.dot(temp, S12_c_caa.T)
        Xt[s_c:f_c] = temp[:,0].copy()
        Xt[s_caa:f_caa] = temp[:,1:].reshape(-1).copy()

    return Xt

def compute_sigma_vector_caa(mr_adc, M_00, M_01, M_11, Xt):

    # Einsum definition from kernel
    einsum = mr_adc.interface.einsum
    einsum_type = mr_adc.interface.einsum_type

    # Variables from kernel
    ncvs = mr_adc.ncvs
    nval = mr_adc.nval
    ncas = mr_adc.ncas
    nextern = mr_adc.nextern

    e_cvs = mr_adc.mo_energy.x
    e_core = mr_adc.mo_energy.c
    e_val = mr_adc.mo_energy.v
    e_extern = mr_adc.mo_energy.e

    # One-electron integrals
    h_aa = mr_adc.h1eff.aa

    # Two-electrons integrals
    v_aaaa = mr_adc.v2e.aaaa

    # Reduced Density Matrices
    rdm_ca = mr_adc.rdm.ca
    rdm_ccaa = mr_adc.rdm.ccaa
    rdm_cccaaa = mr_adc.rdm.cccaaa

    # Dimensions
    s_c = mr_adc.h0.s_c
    f_c = mr_adc.h0.f_c
    s_caa = mr_adc.h1.s_caa
    f_caa = mr_adc.h1.f_caa
    s_cce = mr_adc.h1.s_cce
    f_cce = mr_adc.h1.f_cce
    s_cae = mr_adc.h1.s_cae
    f_cae = mr_adc.h1.f_cae
    s_ace = mr_adc.h1.s_ace
    f_ace = mr_adc.h1.f_ace
    s_cca = mr_adc.h1.s_cca
    f_cca = mr_adc.h1.f_cca
    if nval > 0:
        s_cve = mr_adc.h1.s_cve
        f_cve = mr_adc.h1.f_cve
        s_vce = mr_adc.h1.s_vce
        f_vce = mr_adc.h1.f_vce

        s_cva = mr_adc.h1.s_cva
        f_cva = mr_adc.h1.f_cva
        s_vca = mr_adc.h1.s_vca
        f_vca = mr_adc.h1.f_vca

    cvs_tril_ind = np.tril_indices(ncvs, k=-1)

    # (CASCI + C) -> (CASCI + C)
    sigma = np.zeros_like(Xt)

    # h0-h0 contributions
    sigma[:mr_adc.h0.dim] = np.dot(M_00, Xt[:mr_adc.h0.dim])

    # h1-h1 contributions
    # CAA <- CAA
    X_caa = Xt[s_caa:f_caa].reshape(-1).copy()

    dim_WZ = ncas * ncas
    dim_c_caa = ncvs * dim_WZ

    sigma_aaa_i = 0
    sigma_aaa_f = sigma_aaa_i + dim_c_caa
    sigma_abb_i = sigma_aaa_f
    sigma_abb_f = sigma_abb_i + dim_c_caa

    X_aaa = X_caa[sigma_aaa_i:sigma_aaa_f].reshape(ncvs, ncas, ncas).copy()
    X_abb = X_caa[sigma_abb_i:sigma_abb_f].reshape(ncvs, ncas, ncas).copy()

    sigma_caa_aaa  = 1/2 * einsum('KxZ,K,xW->KWZ', X_aaa, e_cvs, rdm_ca, optimize = einsum_type)
    sigma_caa_aaa += 1/6 * einsum('Kxy,K,WyZx->KWZ', X_aaa, e_cvs, rdm_ccaa, optimize = einsum_type)
    sigma_caa_aaa -= 1/6 * einsum('Kxy,K,WyxZ->KWZ', X_aaa, e_cvs, rdm_ccaa, optimize = einsum_type)
    sigma_caa_aaa += 1/2 * einsum('KxZ,xy,yW->KWZ', X_aaa, h_aa, rdm_ca, optimize = einsum_type)
    sigma_caa_aaa -= 1/2 * einsum('Kxy,Zy,xW->KWZ', X_aaa, h_aa, rdm_ca, optimize = einsum_type)
    sigma_caa_aaa += 1/6 * einsum('Kxy,xz,WyZz->KWZ', X_aaa, h_aa, rdm_ccaa, optimize = einsum_type)
    sigma_caa_aaa -= 1/6 * einsum('Kxy,xz,WyzZ->KWZ', X_aaa, h_aa, rdm_ccaa, optimize = einsum_type)
    sigma_caa_aaa -= 1/6 * einsum('Kxy,yz,WzZx->KWZ', X_aaa, h_aa, rdm_ccaa, optimize = einsum_type)
    sigma_caa_aaa += 1/6 * einsum('Kxy,yz,WzxZ->KWZ', X_aaa, h_aa, rdm_ccaa, optimize = einsum_type)
    sigma_caa_aaa += 1/2 * einsum('KxZ,xyzw,Wyzw->KWZ', X_aaa, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_caa_aaa += 1/6 * einsum('Kxy,Zxzw,Wyzw->KWZ', X_aaa, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_caa_aaa -= 1/6 * einsum('Kxy,Zxzw,Wywz->KWZ', X_aaa, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_caa_aaa -= 1/2 * einsum('Kxy,Zzyw,Wzxw->KWZ', X_aaa, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_caa_aaa += 1/6 * einsum('Kxy,Zzwy,Wzxw->KWZ', X_aaa, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_caa_aaa -= 1/6 * einsum('Kxy,Zzwy,Wzwx->KWZ', X_aaa, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_caa_aaa += 1/6 * einsum('Kxy,xzwu,ZwuWyz->KWZ', X_aaa, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_caa_aaa -= 1/6 * einsum('Kxy,xzwu,ZwuyWz->KWZ', X_aaa, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_caa_aaa -= 1/6 * einsum('Kxy,yzwu,ZxzWwu->KWZ', X_aaa, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_caa_aaa += 1/6 * einsum('Kxy,yzwu,ZxzwWu->KWZ', X_aaa, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_caa_aaa += 1/3 * einsum('Kxy,K,WyZx->KWZ', X_abb, e_cvs, rdm_ccaa, optimize = einsum_type)
    sigma_caa_aaa += 1/6 * einsum('Kxy,K,WyxZ->KWZ', X_abb, e_cvs, rdm_ccaa, optimize = einsum_type)
    sigma_caa_aaa += 1/3 * einsum('Kxy,xz,WyZz->KWZ', X_abb, h_aa, rdm_ccaa, optimize = einsum_type)
    sigma_caa_aaa += 1/6 * einsum('Kxy,xz,WyzZ->KWZ', X_abb, h_aa, rdm_ccaa, optimize = einsum_type)
    sigma_caa_aaa -= 1/3 * einsum('Kxy,yz,WzZx->KWZ', X_abb, h_aa, rdm_ccaa, optimize = einsum_type)
    sigma_caa_aaa -= 1/6 * einsum('Kxy,yz,WzxZ->KWZ', X_abb, h_aa, rdm_ccaa, optimize = einsum_type)
    sigma_caa_aaa += 1/3 * einsum('Kxy,Zxzw,Wyzw->KWZ', X_abb, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_caa_aaa += 1/6 * einsum('Kxy,Zxzw,Wywz->KWZ', X_abb, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_caa_aaa -= 1/6 * einsum('Kxy,Zzwy,Wzxw->KWZ', X_abb, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_caa_aaa -= 1/3 * einsum('Kxy,Zzwy,Wzwx->KWZ', X_abb, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_caa_aaa += 1/4 * einsum('Kxy,xzwu,ZwuWyz->KWZ', X_abb, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_caa_aaa -= 1/12 * einsum('Kxy,xzwu,ZwuWzy->KWZ', X_abb, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_caa_aaa += 1/12 * einsum('Kxy,xzwu,ZwuyWz->KWZ', X_abb, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_caa_aaa -= 1/12 * einsum('Kxy,xzwu,ZwuyzW->KWZ', X_abb, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_caa_aaa -= 1/12 * einsum('Kxy,xzwu,ZwuzWy->KWZ', X_abb, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_caa_aaa -= 1/12 * einsum('Kxy,xzwu,ZwuzyW->KWZ', X_abb, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_caa_aaa -= 1/4 * einsum('Kxy,yzwu,ZxzWwu->KWZ', X_abb, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_caa_aaa += 1/12 * einsum('Kxy,yzwu,ZxzWuw->KWZ', X_abb, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_caa_aaa -= 1/12 * einsum('Kxy,yzwu,ZxzwWu->KWZ', X_abb, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_caa_aaa += 1/12 * einsum('Kxy,yzwu,ZxzwuW->KWZ', X_abb, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_caa_aaa += 1/12 * einsum('Kxy,yzwu,ZxzuWw->KWZ', X_abb, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_caa_aaa += 1/12 * einsum('Kxy,yzwu,ZxzuwW->KWZ', X_abb, v_aaaa, rdm_cccaaa, optimize = einsum_type)

    sigma_caa_abb  = 1/3 * einsum('Kxy,K,WyZx->KWZ', X_aaa, e_cvs, rdm_ccaa, optimize = einsum_type)
    sigma_caa_abb += 1/6 * einsum('Kxy,K,WyxZ->KWZ', X_aaa, e_cvs, rdm_ccaa, optimize = einsum_type)
    sigma_caa_abb += 1/3 * einsum('Kxy,xz,WyZz->KWZ', X_aaa, h_aa, rdm_ccaa, optimize = einsum_type)
    sigma_caa_abb += 1/6 * einsum('Kxy,xz,WyzZ->KWZ', X_aaa, h_aa, rdm_ccaa, optimize = einsum_type)
    sigma_caa_abb -= 1/3 * einsum('Kxy,yz,WzZx->KWZ', X_aaa, h_aa, rdm_ccaa, optimize = einsum_type)
    sigma_caa_abb -= 1/6 * einsum('Kxy,yz,WzxZ->KWZ', X_aaa, h_aa, rdm_ccaa, optimize = einsum_type)
    sigma_caa_abb += 1/3 * einsum('Kxy,Zxzw,Wyzw->KWZ', X_aaa, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_caa_abb += 1/6 * einsum('Kxy,Zxzw,Wywz->KWZ', X_aaa, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_caa_abb -= 1/6 * einsum('Kxy,Zzwy,Wzxw->KWZ', X_aaa, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_caa_abb -= 1/3 * einsum('Kxy,Zzwy,Wzwx->KWZ', X_aaa, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_caa_abb += 1/4 * einsum('Kxy,xzwu,ZwuWyz->KWZ', X_aaa, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_caa_abb -= 1/12 * einsum('Kxy,xzwu,ZwuWzy->KWZ', X_aaa, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_caa_abb += 1/12 * einsum('Kxy,xzwu,ZwuyWz->KWZ', X_aaa, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_caa_abb -= 1/12 * einsum('Kxy,xzwu,ZwuyzW->KWZ', X_aaa, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_caa_abb -= 1/12 * einsum('Kxy,xzwu,ZwuzWy->KWZ', X_aaa, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_caa_abb -= 1/12 * einsum('Kxy,xzwu,ZwuzyW->KWZ', X_aaa, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_caa_abb -= 1/4 * einsum('Kxy,yzwu,ZxzWwu->KWZ', X_aaa, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_caa_abb += 1/12 * einsum('Kxy,yzwu,ZxzWuw->KWZ', X_aaa, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_caa_abb -= 1/12 * einsum('Kxy,yzwu,ZxzwWu->KWZ', X_aaa, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_caa_abb += 1/12 * einsum('Kxy,yzwu,ZxzwuW->KWZ', X_aaa, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_caa_abb += 1/12 * einsum('Kxy,yzwu,ZxzuWw->KWZ', X_aaa, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_caa_abb += 1/12 * einsum('Kxy,yzwu,ZxzuwW->KWZ', X_aaa, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_caa_abb += 1/2 * einsum('KxZ,K,xW->KWZ', X_abb, e_cvs, rdm_ca, optimize = einsum_type)
    sigma_caa_abb += 1/6 * einsum('Kxy,K,WyZx->KWZ', X_abb, e_cvs, rdm_ccaa, optimize = einsum_type)
    sigma_caa_abb -= 1/6 * einsum('Kxy,K,WyxZ->KWZ', X_abb, e_cvs, rdm_ccaa, optimize = einsum_type)
    sigma_caa_abb += 1/2 * einsum('KxZ,xy,yW->KWZ', X_abb, h_aa, rdm_ca, optimize = einsum_type)
    sigma_caa_abb -= 1/2 * einsum('Kxy,Zy,xW->KWZ', X_abb, h_aa, rdm_ca, optimize = einsum_type)
    sigma_caa_abb += 1/6 * einsum('Kxy,xz,WyZz->KWZ', X_abb, h_aa, rdm_ccaa, optimize = einsum_type)
    sigma_caa_abb -= 1/6 * einsum('Kxy,xz,WyzZ->KWZ', X_abb, h_aa, rdm_ccaa, optimize = einsum_type)
    sigma_caa_abb -= 1/6 * einsum('Kxy,yz,WzZx->KWZ', X_abb, h_aa, rdm_ccaa, optimize = einsum_type)
    sigma_caa_abb += 1/6 * einsum('Kxy,yz,WzxZ->KWZ', X_abb, h_aa, rdm_ccaa, optimize = einsum_type)
    sigma_caa_abb += 1/2 * einsum('KxZ,xyzw,Wyzw->KWZ', X_abb, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_caa_abb += 1/6 * einsum('Kxy,Zxzw,Wyzw->KWZ', X_abb, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_caa_abb -= 1/6 * einsum('Kxy,Zxzw,Wywz->KWZ', X_abb, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_caa_abb -= 1/2 * einsum('Kxy,Zzyw,Wzxw->KWZ', X_abb, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_caa_abb += 1/6 * einsum('Kxy,Zzwy,Wzxw->KWZ', X_abb, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_caa_abb -= 1/6 * einsum('Kxy,Zzwy,Wzwx->KWZ', X_abb, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_caa_abb += 1/6 * einsum('Kxy,xzwu,ZwuWyz->KWZ', X_abb, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_caa_abb -= 1/6 * einsum('Kxy,xzwu,ZwuyWz->KWZ', X_abb, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_caa_abb -= 1/6 * einsum('Kxy,yzwu,ZxzWwu->KWZ', X_abb, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_caa_abb += 1/6 * einsum('Kxy,yzwu,ZxzwWu->KWZ', X_abb, v_aaaa, rdm_cccaaa, optimize = einsum_type)

    ## Building C-CAA matrix
    dim_caa = ncvs * ncas * ncas

    sigma_aaa_i = 0
    sigma_aaa_f = sigma_aaa_i + dim_caa
    sigma_abb_i = sigma_aaa_f
    sigma_abb_f = sigma_abb_i + dim_caa

    sigma_caa = np.zeros((2 * dim_caa))
    sigma_caa[sigma_aaa_i:sigma_aaa_f] = sigma_caa_aaa.reshape(-1).copy()
    sigma_caa[sigma_abb_i:sigma_abb_f] = sigma_caa_abb.reshape(-1).copy()

    sigma[s_caa:f_caa] += sigma_caa.reshape(-1).copy()

    return sigma

## Diagonal block
def compute_excitation_manifolds_diag(mr_adc):

    # MR-ADC(0) and MR-ADC(1)
    mr_adc.h0.n_c = mr_adc.ncvs
    mr_adc.h0.dim = mr_adc.h0.n_c # Total dimension of h0

    mr_adc.h0.s_c = 0
    mr_adc.h0.f_c = mr_adc.h0.s_c + mr_adc.h0.n_c

    print("Dimension of h0 excitation manifold:                       %d" % mr_adc.h0.dim)

    # MR-ADC(2)
    mr_adc.h1.dim = 0
    mr_adc.h_orth.dim = mr_adc.h0.dim

    if mr_adc.method in ("mr-adc(2)", "mr-adc(2)-x"):
        mr_adc.h1.n_caa = 3 * mr_adc.ncas * mr_adc.ncas * mr_adc.ncvs
        mr_adc.h1.n_cce = mr_adc.ncvs * mr_adc.ncvs * mr_adc.nextern
        mr_adc.h1.n_cae = mr_adc.ncvs * mr_adc.ncas * mr_adc.nextern
        mr_adc.h1.n_ace = mr_adc.h1.n_cae
        mr_adc.h1.n_cca = mr_adc.ncvs * mr_adc.ncvs * mr_adc.ncas
        if mr_adc.nval > 0:
            mr_adc.h1.n_cve = mr_adc.ncvs * mr_adc.nval * mr_adc.nextern
            mr_adc.h1.n_vce = mr_adc.h1.n_cve
            mr_adc.h1.n_cva = mr_adc.ncvs * mr_adc.nval * mr_adc.ncas
            mr_adc.h1.n_vca = mr_adc.h1.n_cva
            mr_adc.h1.dim = (mr_adc.h1.n_caa + mr_adc.h1.n_cce + mr_adc.h1.n_cve + mr_adc.h1.n_vce +
                             mr_adc.h1.n_cae + mr_adc.h1.n_ace + mr_adc.h1.n_cca + mr_adc.h1.n_cva + mr_adc.h1.n_vca)
        else:
            mr_adc.h1.dim = mr_adc.h1.n_caa + mr_adc.h1.n_cce + mr_adc.h1.n_cae + mr_adc.h1.n_cae + mr_adc.h1.n_cca

        if mr_adc.nval > 0:
            mr_adc.h1.s_caa = mr_adc.h0.f_c
            mr_adc.h1.f_caa = mr_adc.h1.s_caa + mr_adc.h1.n_caa
            mr_adc.h1.s_cce = mr_adc.h1.f_caa
            mr_adc.h1.f_cce = mr_adc.h1.s_cce + mr_adc.h1.n_cce
            mr_adc.h1.s_cve = mr_adc.h1.f_cce
            mr_adc.h1.f_cve = mr_adc.h1.s_cve + mr_adc.h1.n_cve
            mr_adc.h1.s_vce = mr_adc.h1.f_cve
            mr_adc.h1.f_vce = mr_adc.h1.s_vce + mr_adc.h1.n_vce
            mr_adc.h1.s_cae = mr_adc.h1.f_vce
            mr_adc.h1.f_cae = mr_adc.h1.s_cae + mr_adc.h1.n_cae
            mr_adc.h1.s_ace = mr_adc.h1.f_cae
            mr_adc.h1.f_ace = mr_adc.h1.s_ace + mr_adc.h1.n_ace
            mr_adc.h1.s_cca = mr_adc.h1.f_ace
            mr_adc.h1.f_cca = mr_adc.h1.s_cca + mr_adc.h1.n_cca
            mr_adc.h1.s_cva = mr_adc.h1.f_cca
            mr_adc.h1.f_cva = mr_adc.h1.s_cva + mr_adc.h1.n_cva
            mr_adc.h1.s_vca = mr_adc.h1.f_cva
            mr_adc.h1.f_vca = mr_adc.h1.s_vca + mr_adc.h1.n_vca
        else:
            mr_adc.h1.s_caa = mr_adc.h0.f_c
            mr_adc.h1.f_caa = mr_adc.h1.s_caa + mr_adc.h1.n_caa
            mr_adc.h1.s_cce = mr_adc.h1.f_caa
            mr_adc.h1.f_cce = mr_adc.h1.s_cce + mr_adc.h1.n_cce
            mr_adc.h1.s_cae = mr_adc.h1.f_cce
            mr_adc.h1.f_cae = mr_adc.h1.s_cae + mr_adc.h1.n_cae
            mr_adc.h1.s_ace = mr_adc.h1.f_cae
            mr_adc.h1.f_ace = mr_adc.h1.s_ace + mr_adc.h1.n_ace
            mr_adc.h1.s_cca = mr_adc.h1.f_ace
            mr_adc.h1.f_cca = mr_adc.h1.s_cca + mr_adc.h1.n_cca

        print("Dimension of h1 excitation manifold:                       %d" % mr_adc.h1.dim)

        # Overlap for c - caa
        mr_adc.S12.c_caa = mr_adc_overlap.compute_S12_0p_projector(mr_adc)
        mr_adc.S12.cae = mr_adc_overlap.compute_S12_m1(mr_adc)
        mr_adc.S12.cca = mr_adc_overlap.compute_S12_p1(mr_adc)

        # Determine dimensions of orthogonalized excitation spaces
        mr_adc.h_orth.n_c_caa = mr_adc.ncvs * mr_adc.S12.c_caa.shape[1]
        mr_adc.h_orth.n_cce = mr_adc.h1.n_cce
        mr_adc.h_orth.n_cae = mr_adc.ncvs * mr_adc.S12.cae.shape[1] * mr_adc.nextern
        mr_adc.h_orth.n_ace = mr_adc.h_orth.n_cae
        mr_adc.h_orth.n_cca = mr_adc.ncvs * mr_adc.ncvs * mr_adc.S12.cca.shape[1]
        if mr_adc.nval > 0:
            mr_adc.h_orth.n_cve = mr_adc.h1.n_cve
            mr_adc.h_orth.n_vce = mr_adc.h1.n_vce
            mr_adc.h_orth.n_cva = mr_adc.ncvs * mr_adc.nval * mr_adc.S12.cca.shape[1]
            mr_adc.h_orth.n_vca = mr_adc.h_orth.n_cva
            mr_adc.h_orth.dim = (mr_adc.h_orth.n_c_caa + mr_adc.h_orth.n_cce + mr_adc.h_orth.n_cve + mr_adc.h_orth.n_vce +
                                 mr_adc.h_orth.n_cae + mr_adc.h_orth.n_ace + mr_adc.h_orth.n_cca + mr_adc.h_orth.n_cva + mr_adc.h_orth.n_vca)
        else:
            mr_adc.h_orth.dim = mr_adc.h_orth.n_c_caa + mr_adc.h_orth.n_cce + mr_adc.h_orth.n_cae + mr_adc.h_orth.n_ace + mr_adc.h_orth.n_cca

        if mr_adc.nval > 0:
            mr_adc.h_orth.s_c_caa = 0
            mr_adc.h_orth.f_c_caa = mr_adc.h_orth.s_c_caa + mr_adc.h_orth.n_c_caa
            mr_adc.h_orth.s_cce = mr_adc.h_orth.f_c_caa
            mr_adc.h_orth.f_cce = mr_adc.h_orth.s_cce + mr_adc.h_orth.n_cce
            mr_adc.h_orth.s_cve = mr_adc.h_orth.f_cce
            mr_adc.h_orth.f_cve = mr_adc.h_orth.s_cve + mr_adc.h_orth.n_cve
            mr_adc.h_orth.s_vce = mr_adc.h_orth.f_cve
            mr_adc.h_orth.f_vce = mr_adc.h_orth.s_vce + mr_adc.h_orth.n_vce
            mr_adc.h_orth.s_cae = mr_adc.h_orth.f_vce
            mr_adc.h_orth.f_cae = mr_adc.h_orth.s_cae + mr_adc.h_orth.n_cae
            mr_adc.h_orth.s_ace = mr_adc.h_orth.f_cae
            mr_adc.h_orth.f_ace = mr_adc.h_orth.s_ace + mr_adc.h_orth.n_ace
            mr_adc.h_orth.s_cca = mr_adc.h_orth.f_ace
            mr_adc.h_orth.f_cca = mr_adc.h_orth.s_cca + mr_adc.h_orth.n_cca
            mr_adc.h_orth.s_cva = mr_adc.h_orth.f_cca
            mr_adc.h_orth.f_cva = mr_adc.h_orth.s_cva + mr_adc.h_orth.n_cva
            mr_adc.h_orth.s_vca = mr_adc.h_orth.f_cva
            mr_adc.h_orth.f_vca = mr_adc.h_orth.s_vca + mr_adc.h_orth.n_vca
        else:
            mr_adc.h_orth.s_c_caa = 0
            mr_adc.h_orth.f_c_caa = mr_adc.h_orth.s_c_caa + mr_adc.h_orth.n_c_caa
            mr_adc.h_orth.s_cce = mr_adc.h_orth.f_c_caa
            mr_adc.h_orth.f_cce = mr_adc.h_orth.s_cce + mr_adc.h_orth.n_cce
            mr_adc.h_orth.s_cae = mr_adc.h_orth.f_cce
            mr_adc.h_orth.f_cae = mr_adc.h_orth.s_cae + mr_adc.h_orth.n_cae
            mr_adc.h_orth.s_ace = mr_adc.h_orth.f_ace
            mr_adc.h_orth.f_ace = mr_adc.h_orth.s_ace + mr_adc.h_orth.n_ace
            mr_adc.h_orth.s_cca = mr_adc.h_orth.f_ace
            mr_adc.h_orth.f_cca = mr_adc.h_orth.s_cca + mr_adc.h_orth.n_cca

    print("Total dimension of the excitation manifold:                %d" % (mr_adc.h0.dim + mr_adc.h1.dim))
    print("Dimension of the orthogonalized excitation manifold:       %d\n" % (mr_adc.h_orth.dim))
    sys.stdout.flush()

    if (mr_adc.h_orth.dim < mr_adc.nroots):
        mr_adc.nroots = mr_adc.h_orth.dim

    return mr_adc

def compute_preconditioner_diag(mr_adc, M_00):

    start_time = time.time()

    # Einsum definition from kernel
    einsum = mr_adc.interface.einsum
    einsum_type = mr_adc.interface.einsum_type

    if mr_adc.method in ("mr-adc(0)", "mr-adc(1)"):

        # Multiply by -1.0, since we are solving for -M C = -S C E
        return (-1.0 * np.diag(M_00))

    # Variables from kernel
    ncvs = mr_adc.ncvs
    nval = mr_adc.nval
    ncas = mr_adc.ncas
    nextern = mr_adc.nextern

    e_cvs = mr_adc.mo_energy.x
    e_val = mr_adc.mo_energy.v
    e_extern = mr_adc.mo_energy.e

    # One-electron integrals
    h_aa = mr_adc.h1eff.aa

    # Two-electrons integrals
    v_aaaa = mr_adc.v2e.aaaa

    v_xaxa = mr_adc.v2e.xaxa
    v_xaax = mr_adc.v2e.xaax

    # Reduced Density Matrices
    rdm_ca = mr_adc.rdm.ca
    rdm_ccaa = mr_adc.rdm.ccaa
    rdm_cccaaa = mr_adc.rdm.cccaaa

    # Overlap matrices
    S12_c_caa = mr_adc.S12.c_caa
    S12_cae = mr_adc.S12.cae
    S12_cca = mr_adc.S12.cca

    # Dimensions
    ho_s_c_caa = mr_adc.h_orth.s_c_caa
    ho_f_c_caa = mr_adc.h_orth.f_c_caa
    ho_s_cce = mr_adc.h_orth.s_cce
    ho_f_cce = mr_adc.h_orth.f_cce
    ho_s_cae = mr_adc.h_orth.s_cae
    ho_f_cae = mr_adc.h_orth.f_cae
    ho_s_ace = mr_adc.h_orth.s_ace
    ho_f_ace = mr_adc.h_orth.f_ace
    ho_s_cca = mr_adc.h_orth.s_cca
    ho_f_cca = mr_adc.h_orth.f_cca
    if nval > 0:
        ho_s_cve = mr_adc.h_orth.s_cve
        ho_f_cve = mr_adc.h_orth.f_cve
        ho_s_vce = mr_adc.h_orth.s_vce
        ho_f_vce = mr_adc.h_orth.f_vce

        ho_s_cva = mr_adc.h_orth.s_cva
        ho_f_cva = mr_adc.h_orth.f_cva
        ho_s_vca = mr_adc.h_orth.s_vca
        ho_f_vca = mr_adc.h_orth.f_vca

    s_c = mr_adc.h0.s_c
    f_c = mr_adc.h0.f_c

    # Build the preconditioner
    precond = np.zeros(mr_adc.h_orth.dim)

    # C and CAA
    # 0th-order
    precond_c_caa_a_aaa  = 1/2 * einsum('I,II,XY->IXY', e_cvs, np.identity(ncvs), rdm_ca, optimize = einsum_type)

    precond_c_caa_a_abb  = 1/2 * einsum('I,II,XY->IXY', e_cvs, np.identity(ncvs), rdm_ca, optimize = einsum_type)

    # 1st-order
    precond_c_caa_a_aaa += 1/2 * einsum('IxIY,Xx->IXY', v_xaxa, rdm_ca, optimize = einsum_type)
    precond_c_caa_a_aaa += 1/2 * einsum('IxIy,XyYx->IXY', v_xaxa, rdm_ccaa, optimize = einsum_type)
    precond_c_caa_a_aaa -= 1/2 * einsum('IxYI,Xx->IXY', v_xaax, rdm_ca, optimize = einsum_type)
    precond_c_caa_a_aaa -= 1/6 * einsum('IxyI,XyYx->IXY', v_xaax, rdm_ccaa, optimize = einsum_type)
    precond_c_caa_a_aaa += 1/6 * einsum('IxyI,XyxY->IXY', v_xaax, rdm_ccaa, optimize = einsum_type)
    precond_c_caa_a_aaa -= 1/2 * einsum('IxIy,xy,XY->IXY', v_xaxa, rdm_ca, rdm_ca, optimize = einsum_type)
    precond_c_caa_a_aaa += 1/4 * einsum('IxyI,xy,XY->IXY', v_xaax, rdm_ca, rdm_ca, optimize = einsum_type)

    precond_c_caa_a_abb += 1/2 * einsum('IxIY,Xx->IXY', v_xaxa, rdm_ca, optimize = einsum_type)
    precond_c_caa_a_abb += 1/2 * einsum('IxIy,XyYx->IXY', v_xaxa, rdm_ccaa, optimize = einsum_type)
    precond_c_caa_a_abb -= 1/3 * einsum('IxyI,XyYx->IXY', v_xaax, rdm_ccaa, optimize = einsum_type)
    precond_c_caa_a_abb -= 1/6 * einsum('IxyI,XyxY->IXY', v_xaax, rdm_ccaa, optimize = einsum_type)
    precond_c_caa_a_abb -= 1/2 * einsum('IxIy,xy,XY->IXY', v_xaxa, rdm_ca, rdm_ca, optimize = einsum_type)
    precond_c_caa_a_abb += 1/4 * einsum('IxyI,xy,XY->IXY', v_xaax, rdm_ca, rdm_ca, optimize = einsum_type)

    precond_caa_caa_aaa_aaa =- 1/6 * einsum('I,II,WYXZ->IWZXY', e_cvs, np.identity(ncvs), rdm_ccaa, optimize = einsum_type)
    precond_caa_caa_aaa_aaa += 1/6 * einsum('I,II,WYZX->IWZXY', e_cvs, np.identity(ncvs), rdm_ccaa, optimize = einsum_type)
    precond_caa_caa_aaa_aaa += 1/6 * einsum('Xx,II,WYZx->IWZXY', h_aa, np.identity(ncvs), rdm_ccaa, optimize = einsum_type)
    precond_caa_caa_aaa_aaa -= 1/6 * einsum('Xx,II,WYxZ->IWZXY', h_aa, np.identity(ncvs), rdm_ccaa, optimize = einsum_type)
    precond_caa_caa_aaa_aaa -= 1/2 * einsum('YZ,II,XW->IWZXY', h_aa, np.identity(ncvs), rdm_ca, optimize = einsum_type)
    precond_caa_caa_aaa_aaa += 1/6 * einsum('Yx,II,WxXZ->IWZXY', h_aa, np.identity(ncvs), rdm_ccaa, optimize = einsum_type)
    precond_caa_caa_aaa_aaa -= 1/6 * einsum('Yx,II,WxZX->IWZXY', h_aa, np.identity(ncvs), rdm_ccaa, optimize = einsum_type)
    precond_caa_caa_aaa_aaa -= 1/6 * einsum('II,XZxy,WYxy->IWZXY', np.identity(ncvs), v_aaaa, rdm_ccaa, optimize = einsum_type)
    precond_caa_caa_aaa_aaa += 1/6 * einsum('II,XZxy,WYyx->IWZXY', np.identity(ncvs), v_aaaa, rdm_ccaa, optimize = einsum_type)
    precond_caa_caa_aaa_aaa += 1/6 * einsum('II,Xxyz,ZyzWYx->IWZXY', np.identity(ncvs), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    precond_caa_caa_aaa_aaa -= 1/6 * einsum('II,Xxyz,ZyzYWx->IWZXY', np.identity(ncvs), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    precond_caa_caa_aaa_aaa -= 1/2 * einsum('II,YxZy,WyXx->IWZXY', np.identity(ncvs), v_aaaa, rdm_ccaa, optimize = einsum_type)
    precond_caa_caa_aaa_aaa += 1/6 * einsum('II,YxyZ,WyXx->IWZXY', np.identity(ncvs), v_aaaa, rdm_ccaa, optimize = einsum_type)
    precond_caa_caa_aaa_aaa -= 1/6 * einsum('II,YxyZ,WyxX->IWZXY', np.identity(ncvs), v_aaaa, rdm_ccaa, optimize = einsum_type)
    precond_caa_caa_aaa_aaa += 1/6 * einsum('II,Yxyz,XZxWyz->IWZXY', np.identity(ncvs), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    precond_caa_caa_aaa_aaa -= 1/6 * einsum('II,Yxyz,XZxyWz->IWZXY', np.identity(ncvs), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    precond_caa_caa_aaa_aaa += 1/2 * einsum('I,II,YZ,XW->IWZXY', e_cvs, np.identity(ncvs), np.identity(ncas), rdm_ca, optimize = einsum_type)
    precond_caa_caa_aaa_aaa += 1/2 * einsum('Xx,II,YZ,xW->IWZXY', h_aa, np.identity(ncvs), np.identity(ncas), rdm_ca, optimize = einsum_type)
    precond_caa_caa_aaa_aaa += 1/2 * einsum('Xxyz,II,YZ,Wxyz->IWZXY', v_aaaa, np.identity(ncvs), np.identity(ncas), rdm_ccaa, optimize = einsum_type)

    precond_caa_caa_abb_abb =- 1/6 * einsum('I,II,WYXZ->IWZXY', e_cvs, np.identity(ncvs), rdm_ccaa, optimize = einsum_type)
    precond_caa_caa_abb_abb += 1/6 * einsum('I,II,WYZX->IWZXY', e_cvs, np.identity(ncvs), rdm_ccaa, optimize = einsum_type)
    precond_caa_caa_abb_abb += 1/6 * einsum('Xx,II,WYZx->IWZXY', h_aa, np.identity(ncvs), rdm_ccaa, optimize = einsum_type)
    precond_caa_caa_abb_abb -= 1/6 * einsum('Xx,II,WYxZ->IWZXY', h_aa, np.identity(ncvs), rdm_ccaa, optimize = einsum_type)
    precond_caa_caa_abb_abb -= 1/2 * einsum('YZ,II,XW->IWZXY', h_aa, np.identity(ncvs), rdm_ca, optimize = einsum_type)
    precond_caa_caa_abb_abb += 1/6 * einsum('Yx,II,WxXZ->IWZXY', h_aa, np.identity(ncvs), rdm_ccaa, optimize = einsum_type)
    precond_caa_caa_abb_abb -= 1/6 * einsum('Yx,II,WxZX->IWZXY', h_aa, np.identity(ncvs), rdm_ccaa, optimize = einsum_type)
    precond_caa_caa_abb_abb -= 1/6 * einsum('II,XZxy,WYxy->IWZXY', np.identity(ncvs), v_aaaa, rdm_ccaa, optimize = einsum_type)
    precond_caa_caa_abb_abb += 1/6 * einsum('II,XZxy,WYyx->IWZXY', np.identity(ncvs), v_aaaa, rdm_ccaa, optimize = einsum_type)
    precond_caa_caa_abb_abb += 1/6 * einsum('II,Xxyz,ZyzWYx->IWZXY', np.identity(ncvs), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    precond_caa_caa_abb_abb -= 1/6 * einsum('II,Xxyz,ZyzYWx->IWZXY', np.identity(ncvs), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    precond_caa_caa_abb_abb -= 1/2 * einsum('II,YxZy,WyXx->IWZXY', np.identity(ncvs), v_aaaa, rdm_ccaa, optimize = einsum_type)
    precond_caa_caa_abb_abb += 1/6 * einsum('II,YxyZ,WyXx->IWZXY', np.identity(ncvs), v_aaaa, rdm_ccaa, optimize = einsum_type)
    precond_caa_caa_abb_abb -= 1/6 * einsum('II,YxyZ,WyxX->IWZXY', np.identity(ncvs), v_aaaa, rdm_ccaa, optimize = einsum_type)
    precond_caa_caa_abb_abb += 1/6 * einsum('II,Yxyz,XZxWyz->IWZXY', np.identity(ncvs), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    precond_caa_caa_abb_abb -= 1/6 * einsum('II,Yxyz,XZxyWz->IWZXY', np.identity(ncvs), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    precond_caa_caa_abb_abb += 1/2 * einsum('I,II,YZ,XW->IWZXY', e_cvs, np.identity(ncvs), np.identity(ncas), rdm_ca, optimize = einsum_type)
    precond_caa_caa_abb_abb += 1/2 * einsum('Xx,II,YZ,xW->IWZXY', h_aa, np.identity(ncvs), np.identity(ncas), rdm_ca, optimize = einsum_type)
    precond_caa_caa_abb_abb += 1/2 * einsum('Xxyz,II,YZ,Wxyz->IWZXY', v_aaaa, np.identity(ncvs), np.identity(ncas), rdm_ccaa, optimize = einsum_type)

    precond_caa_caa_bab_bab =- 1/3 * einsum('I,II,WYXZ->IWZXY', e_cvs, np.identity(ncvs), rdm_ccaa, optimize = einsum_type)
    precond_caa_caa_bab_bab -= 1/6 * einsum('I,II,WYZX->IWZXY', e_cvs, np.identity(ncvs), rdm_ccaa, optimize = einsum_type)
    precond_caa_caa_bab_bab -= 1/6 * einsum('Xx,II,WYZx->IWZXY', h_aa, np.identity(ncvs), rdm_ccaa, optimize = einsum_type)
    precond_caa_caa_bab_bab -= 1/3 * einsum('Xx,II,WYxZ->IWZXY', h_aa, np.identity(ncvs), rdm_ccaa, optimize = einsum_type)
    precond_caa_caa_bab_bab -= 1/2 * einsum('YZ,II,XW->IWZXY', h_aa, np.identity(ncvs), rdm_ca, optimize = einsum_type)
    precond_caa_caa_bab_bab += 1/3 * einsum('Yx,II,WxXZ->IWZXY', h_aa, np.identity(ncvs), rdm_ccaa, optimize = einsum_type)
    precond_caa_caa_bab_bab += 1/6 * einsum('Yx,II,WxZX->IWZXY', h_aa, np.identity(ncvs), rdm_ccaa, optimize = einsum_type)
    precond_caa_caa_bab_bab -= 1/3 * einsum('II,XZxy,WYxy->IWZXY', np.identity(ncvs), v_aaaa, rdm_ccaa, optimize = einsum_type)
    precond_caa_caa_bab_bab -= 1/6 * einsum('II,XZxy,WYyx->IWZXY', np.identity(ncvs), v_aaaa, rdm_ccaa, optimize = einsum_type)
    precond_caa_caa_bab_bab -= 1/12 * einsum('II,Xxyz,ZyzWYx->IWZXY', np.identity(ncvs), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    precond_caa_caa_bab_bab += 1/12 * einsum('II,Xxyz,ZyzWxY->IWZXY', np.identity(ncvs), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    precond_caa_caa_bab_bab -= 1/4 * einsum('II,Xxyz,ZyzYWx->IWZXY', np.identity(ncvs), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    precond_caa_caa_bab_bab += 1/12 * einsum('II,Xxyz,ZyzYxW->IWZXY', np.identity(ncvs), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    precond_caa_caa_bab_bab += 1/12 * einsum('II,Xxyz,ZyzxWY->IWZXY', np.identity(ncvs), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    precond_caa_caa_bab_bab += 1/12 * einsum('II,Xxyz,ZyzxYW->IWZXY', np.identity(ncvs), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    precond_caa_caa_bab_bab -= 1/2 * einsum('II,YxZy,WyXx->IWZXY', np.identity(ncvs), v_aaaa, rdm_ccaa, optimize = einsum_type)
    precond_caa_caa_bab_bab += 1/3 * einsum('II,YxyZ,WyXx->IWZXY', np.identity(ncvs), v_aaaa, rdm_ccaa, optimize = einsum_type)
    precond_caa_caa_bab_bab += 1/6 * einsum('II,YxyZ,WyxX->IWZXY', np.identity(ncvs), v_aaaa, rdm_ccaa, optimize = einsum_type)
    precond_caa_caa_bab_bab += 1/4 * einsum('II,Yxyz,XZxWyz->IWZXY', np.identity(ncvs), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    precond_caa_caa_bab_bab -= 1/12 * einsum('II,Yxyz,XZxWzy->IWZXY', np.identity(ncvs), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    precond_caa_caa_bab_bab += 1/12 * einsum('II,Yxyz,XZxyWz->IWZXY', np.identity(ncvs), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    precond_caa_caa_bab_bab -= 1/12 * einsum('II,Yxyz,XZxyzW->IWZXY', np.identity(ncvs), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    precond_caa_caa_bab_bab -= 1/12 * einsum('II,Yxyz,XZxzWy->IWZXY', np.identity(ncvs), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    precond_caa_caa_bab_bab -= 1/12 * einsum('II,Yxyz,XZxzyW->IWZXY', np.identity(ncvs), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    precond_caa_caa_bab_bab += 1/2 * einsum('I,II,YZ,XW->IWZXY', e_cvs, np.identity(ncvs), np.identity(ncas), rdm_ca, optimize = einsum_type)
    precond_caa_caa_bab_bab += 1/2 * einsum('Xx,II,YZ,xW->IWZXY', h_aa, np.identity(ncvs), np.identity(ncas), rdm_ca, optimize = einsum_type)
    precond_caa_caa_bab_bab += 1/2 * einsum('Xxyz,II,YZ,Wxyz->IWZXY', v_aaaa, np.identity(ncvs), np.identity(ncas), rdm_ccaa, optimize = einsum_type)

    precond_caa_caa_aaa_abb  = 1/6 * einsum('I,II,WYXZ->IWZXY', e_cvs, np.identity(ncvs), rdm_ccaa, optimize = einsum_type)
    precond_caa_caa_aaa_abb += 1/3 * einsum('I,II,WYZX->IWZXY', e_cvs, np.identity(ncvs), rdm_ccaa, optimize = einsum_type)
    precond_caa_caa_aaa_abb += 1/3 * einsum('Xx,II,WYZx->IWZXY', h_aa, np.identity(ncvs), rdm_ccaa, optimize = einsum_type)
    precond_caa_caa_aaa_abb += 1/6 * einsum('Xx,II,WYxZ->IWZXY', h_aa, np.identity(ncvs), rdm_ccaa, optimize = einsum_type)
    precond_caa_caa_aaa_abb -= 1/6 * einsum('Yx,II,WxXZ->IWZXY', h_aa, np.identity(ncvs), rdm_ccaa, optimize = einsum_type)
    precond_caa_caa_aaa_abb -= 1/3 * einsum('Yx,II,WxZX->IWZXY', h_aa, np.identity(ncvs), rdm_ccaa, optimize = einsum_type)
    precond_caa_caa_aaa_abb += 1/6 * einsum('II,XZxy,WYxy->IWZXY', np.identity(ncvs), v_aaaa, rdm_ccaa, optimize = einsum_type)
    precond_caa_caa_aaa_abb += 1/3 * einsum('II,XZxy,WYyx->IWZXY', np.identity(ncvs), v_aaaa, rdm_ccaa, optimize = einsum_type)
    precond_caa_caa_aaa_abb += 1/4 * einsum('II,Xxyz,ZyzWYx->IWZXY', np.identity(ncvs), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    precond_caa_caa_aaa_abb -= 1/12 * einsum('II,Xxyz,ZyzWxY->IWZXY', np.identity(ncvs), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    precond_caa_caa_aaa_abb += 1/12 * einsum('II,Xxyz,ZyzYWx->IWZXY', np.identity(ncvs), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    precond_caa_caa_aaa_abb -= 1/12 * einsum('II,Xxyz,ZyzYxW->IWZXY', np.identity(ncvs), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    precond_caa_caa_aaa_abb -= 1/12 * einsum('II,Xxyz,ZyzxWY->IWZXY', np.identity(ncvs), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    precond_caa_caa_aaa_abb -= 1/12 * einsum('II,Xxyz,ZyzxYW->IWZXY', np.identity(ncvs), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    precond_caa_caa_aaa_abb -= 1/6 * einsum('II,YxyZ,WyXx->IWZXY', np.identity(ncvs), v_aaaa, rdm_ccaa, optimize = einsum_type)
    precond_caa_caa_aaa_abb -= 1/3 * einsum('II,YxyZ,WyxX->IWZXY', np.identity(ncvs), v_aaaa, rdm_ccaa, optimize = einsum_type)
    precond_caa_caa_aaa_abb -= 1/12 * einsum('II,Yxyz,XZxWyz->IWZXY', np.identity(ncvs), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    precond_caa_caa_aaa_abb += 1/12 * einsum('II,Yxyz,XZxWzy->IWZXY', np.identity(ncvs), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    precond_caa_caa_aaa_abb -= 1/4 * einsum('II,Yxyz,XZxyWz->IWZXY', np.identity(ncvs), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    precond_caa_caa_aaa_abb += 1/12 * einsum('II,Yxyz,XZxyzW->IWZXY', np.identity(ncvs), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    precond_caa_caa_aaa_abb += 1/12 * einsum('II,Yxyz,XZxzWy->IWZXY', np.identity(ncvs), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    precond_caa_caa_aaa_abb += 1/12 * einsum('II,Yxyz,XZxzyW->IWZXY', np.identity(ncvs), v_aaaa, rdm_cccaaa, optimize = einsum_type)

    precond_caa_caa_abb_aaa  = 1/6 * einsum('I,II,WYXZ->IWZXY', e_cvs, np.identity(ncvs), rdm_ccaa, optimize = einsum_type)
    precond_caa_caa_abb_aaa += 1/3 * einsum('I,II,WYZX->IWZXY', e_cvs, np.identity(ncvs), rdm_ccaa, optimize = einsum_type)
    precond_caa_caa_abb_aaa += 1/3 * einsum('Xx,II,WYZx->IWZXY', h_aa, np.identity(ncvs), rdm_ccaa, optimize = einsum_type)
    precond_caa_caa_abb_aaa += 1/6 * einsum('Xx,II,WYxZ->IWZXY', h_aa, np.identity(ncvs), rdm_ccaa, optimize = einsum_type)
    precond_caa_caa_abb_aaa -= 1/6 * einsum('Yx,II,WxXZ->IWZXY', h_aa, np.identity(ncvs), rdm_ccaa, optimize = einsum_type)
    precond_caa_caa_abb_aaa -= 1/3 * einsum('Yx,II,WxZX->IWZXY', h_aa, np.identity(ncvs), rdm_ccaa, optimize = einsum_type)
    precond_caa_caa_abb_aaa += 1/6 * einsum('II,XZxy,WYxy->IWZXY', np.identity(ncvs), v_aaaa, rdm_ccaa, optimize = einsum_type)
    precond_caa_caa_abb_aaa += 1/3 * einsum('II,XZxy,WYyx->IWZXY', np.identity(ncvs), v_aaaa, rdm_ccaa, optimize = einsum_type)
    precond_caa_caa_abb_aaa += 1/4 * einsum('II,Xxyz,ZyzWYx->IWZXY', np.identity(ncvs), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    precond_caa_caa_abb_aaa -= 1/12 * einsum('II,Xxyz,ZyzWxY->IWZXY', np.identity(ncvs), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    precond_caa_caa_abb_aaa += 1/12 * einsum('II,Xxyz,ZyzYWx->IWZXY', np.identity(ncvs), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    precond_caa_caa_abb_aaa -= 1/12 * einsum('II,Xxyz,ZyzYxW->IWZXY', np.identity(ncvs), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    precond_caa_caa_abb_aaa -= 1/12 * einsum('II,Xxyz,ZyzxWY->IWZXY', np.identity(ncvs), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    precond_caa_caa_abb_aaa -= 1/12 * einsum('II,Xxyz,ZyzxYW->IWZXY', np.identity(ncvs), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    precond_caa_caa_abb_aaa -= 1/6 * einsum('II,YxyZ,WyXx->IWZXY', np.identity(ncvs), v_aaaa, rdm_ccaa, optimize = einsum_type)
    precond_caa_caa_abb_aaa -= 1/3 * einsum('II,YxyZ,WyxX->IWZXY', np.identity(ncvs), v_aaaa, rdm_ccaa, optimize = einsum_type)
    precond_caa_caa_abb_aaa -= 1/12 * einsum('II,Yxyz,XZxWyz->IWZXY', np.identity(ncvs), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    precond_caa_caa_abb_aaa += 1/12 * einsum('II,Yxyz,XZxWzy->IWZXY', np.identity(ncvs), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    precond_caa_caa_abb_aaa -= 1/4 * einsum('II,Yxyz,XZxyWz->IWZXY', np.identity(ncvs), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    precond_caa_caa_abb_aaa += 1/12 * einsum('II,Yxyz,XZxyzW->IWZXY', np.identity(ncvs), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    precond_caa_caa_abb_aaa += 1/12 * einsum('II,Yxyz,XZxzWy->IWZXY', np.identity(ncvs), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    precond_caa_caa_abb_aaa += 1/12 * einsum('II,Yxyz,XZxzyW->IWZXY', np.identity(ncvs), v_aaaa, rdm_cccaaa, optimize = einsum_type)

    ## Building C-CAA matrix
    dim_XY = ncas * ncas
    dim_c_caa = 3 * dim_XY

    precond_aa_i = 1
    precond_aa_f = precond_aa_i + dim_XY
    precond_bb_i = precond_aa_f
    precond_bb_f = precond_bb_i + dim_XY
    precond_ab_i = precond_bb_f
    precond_ab_f = precond_ab_i + dim_XY

    precond_temp = np.zeros((ncvs, (1 + dim_c_caa), (1 + dim_c_caa)))
    precond_temp[:, 0, 0] = np.diag(M_00[s_c:f_c, s_c:f_c]).copy()

    precond_temp[:, 0, precond_aa_i:precond_aa_f] = precond_c_caa_a_aaa.reshape(ncvs, ncas * ncas).copy()
    precond_temp[:, 0, precond_bb_i:precond_bb_f] = precond_c_caa_a_abb.reshape(ncvs, ncas * ncas).copy()
    precond_temp[:, precond_aa_i:precond_ab_f, 0] = precond_temp[:, 0, precond_aa_i:precond_ab_f].copy()

    precond_temp[:, precond_aa_i:precond_aa_f, precond_aa_i:precond_aa_f] = precond_caa_caa_aaa_aaa.reshape(ncvs, ncas * ncas, ncas * ncas).copy()
    precond_temp[:, precond_aa_i:precond_aa_f, precond_bb_i:precond_bb_f] = precond_caa_caa_aaa_abb.reshape(ncvs, ncas * ncas, ncas * ncas).copy()

    precond_temp[:, precond_bb_i:precond_bb_f, precond_bb_i:precond_bb_f] = precond_caa_caa_abb_abb.reshape(ncvs, ncas * ncas, ncas * ncas).copy()
    precond_temp[:, precond_bb_i:precond_bb_f, precond_aa_i:precond_aa_f] = precond_caa_caa_abb_aaa.reshape(ncvs, ncas * ncas, ncas * ncas).copy()

    precond_temp[:, precond_ab_i:precond_ab_f, precond_ab_i:precond_ab_f] = precond_caa_caa_bab_bab.reshape(ncvs, ncas * ncas, ncas * ncas).copy()

    precond_temp = einsum('IXY,XP,YP->IP', precond_temp, S12_c_caa, S12_c_caa, optimize = einsum_type)

    precond[ho_s_c_caa:ho_f_c_caa] = precond_temp.reshape(-1)

    # CCE
    precond_cce =- einsum('A,AA,II,JJ->IJA', e_extern, np.identity(nextern), np.identity(ncvs), np.identity(ncvs), optimize = einsum_type)
    precond_cce += einsum('I,AA,II,JJ->IJA', e_cvs, np.identity(nextern), np.identity(ncvs), np.identity(ncvs), optimize = einsum_type)
    precond_cce += einsum('J,AA,II,JJ->IJA', e_cvs, np.identity(nextern), np.identity(ncvs), np.identity(ncvs), optimize = einsum_type)
    precond[ho_s_cce:ho_f_cce] = precond_cce.reshape(-1).copy()

    if nval > 0:
        # CVE
        precond_cve =- einsum('A,AA,II,JJ->IJA', e_extern, np.identity(nextern), np.identity(ncvs), np.identity(nval), optimize = einsum_type)
        precond_cve += einsum('I,AA,II,JJ->IJA', e_cvs, np.identity(nextern), np.identity(ncvs), np.identity(nval), optimize = einsum_type)
        precond_cve += einsum('J,AA,II,JJ->IJA', e_val, np.identity(nextern), np.identity(ncvs), np.identity(nval), optimize = einsum_type)
        precond[ho_s_cve:ho_f_cve] = precond_cve.reshape(-1).copy()

        # VCE
        precond_vce =- einsum('A,AA,II,JJ->IJA', e_extern, np.identity(nextern), np.identity(nval), np.identity(ncvs), optimize = einsum_type)
        precond_vce += einsum('I,AA,II,JJ->IJA', e_val, np.identity(nextern), np.identity(nval), np.identity(ncvs), optimize = einsum_type)
        precond_vce += einsum('J,AA,II,JJ->IJA', e_cvs, np.identity(nextern), np.identity(nval), np.identity(ncvs), optimize = einsum_type)
        precond[ho_s_vce:ho_f_vce] = precond_vce.reshape(-1).copy()

    # CAE
    precond_cae =- 1/2 * einsum('A,AA,II,XY->IAXY', e_extern, np.identity(nextern), np.identity(ncvs), rdm_ca, optimize = einsum_type)
    precond_cae += 1/2 * einsum('I,AA,II,XY->IAXY', e_cvs, np.identity(nextern), np.identity(ncvs), rdm_ca, optimize = einsum_type)
    precond_cae += 1/2 * einsum('Xx,AA,II,xY->IAXY', h_aa, np.identity(nextern), np.identity(ncvs), rdm_ca, optimize = einsum_type)
    precond_cae += 1/2 * einsum('Xxyz,AA,II,Yxyz->IAXY', v_aaaa, np.identity(nextern), np.identity(ncvs), rdm_ccaa, optimize = einsum_type)

    precond_cae = einsum("IAXY,XP,YP->IPA", precond_cae, S12_cae, S12_cae, optimize = einsum_type)
    precond[ho_s_cae:ho_f_cae] = precond_cae.reshape(-1).copy()

    # ACE
    precond_ace =- 1/2 * einsum('A,AA,II,XY->XYIA', e_extern, np.identity(nextern), np.identity(ncvs), rdm_ca, optimize = einsum_type)
    precond_ace += 1/2 * einsum('I,AA,II,XY->XYIA', e_cvs, np.identity(nextern), np.identity(ncvs), rdm_ca, optimize = einsum_type)
    precond_ace += 1/2 * einsum('Xx,AA,II,xY->XYIA', h_aa, np.identity(nextern), np.identity(ncvs), rdm_ca, optimize = einsum_type)
    precond_ace += 1/2 * einsum('Xxyz,AA,II,Yxyz->XYIA', v_aaaa, np.identity(nextern), np.identity(ncvs), rdm_ccaa, optimize = einsum_type)

    precond_ace = einsum("XYIA,XP,YP->PIA", precond_ace, S12_cae, S12_cae, optimize = einsum_type)
    precond[ho_s_ace:ho_f_ace] = precond_ace.reshape(-1).copy()

    # CCA
    precond_cca =- einsum('XY,II,JJ->IJXY', h_aa, np.identity(ncvs), np.identity(ncvs), optimize = einsum_type)
    precond_cca += einsum('I,II,JJ,XY->IJXY', e_cvs, np.identity(ncvs), np.identity(ncvs), np.identity(ncas), optimize = einsum_type)
    precond_cca += einsum('J,II,JJ,XY->IJXY', e_cvs, np.identity(ncvs), np.identity(ncvs), np.identity(ncas), optimize = einsum_type)
    precond_cca -= 1/2 * einsum('I,II,JJ,YX->IJXY', e_cvs, np.identity(ncvs), np.identity(ncvs), rdm_ca, optimize = einsum_type)
    precond_cca -= 1/2 * einsum('J,II,JJ,YX->IJXY', e_cvs, np.identity(ncvs), np.identity(ncvs), rdm_ca, optimize = einsum_type)
    precond_cca += 1/2 * einsum('Xx,II,JJ,Yx->IJXY', h_aa, np.identity(ncvs), np.identity(ncvs), rdm_ca, optimize = einsum_type)
    precond_cca -= einsum('XxYy,II,JJ,xy->IJXY', v_aaaa, np.identity(ncvs), np.identity(ncvs), rdm_ca, optimize = einsum_type)
    precond_cca += 1/2 * einsum('XxyY,II,JJ,xy->IJXY', v_aaaa, np.identity(ncvs), np.identity(ncvs), rdm_ca, optimize = einsum_type)
    precond_cca += 1/2 * einsum('Xxyz,II,JJ,Yxyz->IJXY', v_aaaa, np.identity(ncvs), np.identity(ncvs), rdm_ccaa, optimize = einsum_type)

    precond_cca = einsum("IJXY,XP,YP->IJP", precond_cca, S12_cca, S12_cca, optimize = einsum_type)
    precond[ho_s_cca:ho_f_cca] = precond_cca.reshape(-1).copy()

    if nval > 0:
        # CVA
        precond_cva =- einsum('XY,II,JJ->IJXY', h_aa, np.identity(ncvs), np.identity(nval), optimize = einsum_type)
        precond_cva += einsum('I,II,JJ,XY->IJXY', e_cvs, np.identity(ncvs), np.identity(nval), np.identity(ncas), optimize = einsum_type)
        precond_cva += einsum('J,II,JJ,XY->IJXY', e_val, np.identity(ncvs), np.identity(nval), np.identity(ncas), optimize = einsum_type)
        precond_cva -= 1/2 * einsum('I,II,JJ,YX->IJXY', e_cvs, np.identity(ncvs), np.identity(nval), rdm_ca, optimize = einsum_type)
        precond_cva -= 1/2 * einsum('J,II,JJ,YX->IJXY', e_val, np.identity(ncvs), np.identity(nval), rdm_ca, optimize = einsum_type)
        precond_cva += 1/2 * einsum('Xx,II,JJ,Yx->IJXY', h_aa, np.identity(ncvs), np.identity(nval), rdm_ca, optimize = einsum_type)
        precond_cva -= einsum('XxYy,II,JJ,xy->IJXY', v_aaaa, np.identity(ncvs), np.identity(nval), rdm_ca, optimize = einsum_type)
        precond_cva += 1/2 * einsum('XxyY,II,JJ,xy->IJXY', v_aaaa, np.identity(ncvs), np.identity(nval), rdm_ca, optimize = einsum_type)
        precond_cva += 1/2 * einsum('Xxyz,II,JJ,Yxyz->IJXY', v_aaaa, np.identity(ncvs), np.identity(nval), rdm_ccaa, optimize = einsum_type)
        precond_cva = einsum("IJXY,XP,YP->IJP", precond_cva, S12_cca, S12_cca, optimize = einsum_type)
        precond[ho_s_cva:ho_f_cva] = precond_cva.reshape(-1).copy()

        precond_vca =- einsum('XY,II,JJ->IJXY', h_aa, np.identity(nval), np.identity(ncvs), optimize = einsum_type)
        precond_vca += einsum('I,II,JJ,XY->IJXY', e_val, np.identity(nval), np.identity(ncvs), np.identity(ncas), optimize = einsum_type)
        precond_vca += einsum('J,II,JJ,XY->IJXY', e_cvs, np.identity(nval), np.identity(ncvs), np.identity(ncas), optimize = einsum_type)
        precond_vca -= 1/2 * einsum('I,II,JJ,YX->IJXY', e_val, np.identity(nval), np.identity(ncvs), rdm_ca, optimize = einsum_type)
        precond_vca -= 1/2 * einsum('J,II,JJ,YX->IJXY', e_cvs, np.identity(nval), np.identity(ncvs), rdm_ca, optimize = einsum_type)
        precond_vca += 1/2 * einsum('Xx,II,JJ,Yx->IJXY', h_aa, np.identity(nval), np.identity(ncvs), rdm_ca, optimize = einsum_type)
        precond_vca -= einsum('XxYy,II,JJ,xy->IJXY', v_aaaa, np.identity(nval), np.identity(ncvs), rdm_ca, optimize = einsum_type)
        precond_vca += 1/2 * einsum('XxyY,II,JJ,xy->IJXY', v_aaaa, np.identity(nval), np.identity(ncvs), rdm_ca, optimize = einsum_type)
        precond_vca += 1/2 * einsum('Xxyz,II,JJ,Yxyz->IJXY', v_aaaa, np.identity(nval), np.identity(ncvs), rdm_ccaa, optimize = einsum_type)
        precond_vca = einsum("IJXY,XP,YP->IJP", precond_vca, S12_cca, S12_cca, optimize = einsum_type)
        precond[ho_s_vca:ho_f_vca] = precond_vca.reshape(-1).copy()

    # Multiply by -1.0, since we are solving for -M C = -S C E
    precond *= (-1.0)

    print ("Time for computing preconditioner:                %f sec\n" % (time.time() - start_time))
    sys.stdout.flush()

    return precond

def apply_S_12_diag(mr_adc, X, transpose = False):

    # Einsum definition from kernel
    einsum = mr_adc.interface.einsum
    einsum_type = mr_adc.interface.einsum_type

    # Dimensions
    nextern = mr_adc.nextern
    ncvs = mr_adc.ncvs
    nval = mr_adc.nval

    ho_s_c_caa = mr_adc.h_orth.s_c_caa
    ho_f_c_caa = mr_adc.h_orth.f_c_caa
    ho_s_cce = mr_adc.h_orth.s_cce
    ho_f_cce = mr_adc.h_orth.f_cce
    ho_s_cae = mr_adc.h_orth.s_cae
    ho_f_cae = mr_adc.h_orth.f_cae
    ho_s_ace = mr_adc.h_orth.s_ace
    ho_f_ace = mr_adc.h_orth.f_ace
    ho_s_cca = mr_adc.h_orth.s_cca
    ho_f_cca = mr_adc.h_orth.f_cca
    s_c = mr_adc.h0.s_c
    f_c = mr_adc.h0.f_c
    s_caa = mr_adc.h1.s_caa
    f_caa = mr_adc.h1.f_caa
    s_cce = mr_adc.h1.s_cce
    f_cce = mr_adc.h1.f_cce
    s_cae = mr_adc.h1.s_cae
    f_cae = mr_adc.h1.f_cae
    s_ace = mr_adc.h1.s_ace
    f_ace = mr_adc.h1.f_ace
    s_cca = mr_adc.h1.s_cca
    f_cca = mr_adc.h1.f_cca

    if nval > 0:
        ho_s_cve = mr_adc.h_orth.s_cve
        ho_f_cve = mr_adc.h_orth.f_cve
        ho_s_vce = mr_adc.h_orth.s_vce
        ho_f_vce = mr_adc.h_orth.f_vce

        ho_s_cva = mr_adc.h_orth.s_cva
        ho_f_cva = mr_adc.h_orth.f_cva
        ho_s_vca = mr_adc.h_orth.s_vca
        ho_f_vca = mr_adc.h_orth.f_vca

        s_cve = mr_adc.h1.s_cve
        f_cve = mr_adc.h1.f_cve
        s_vce = mr_adc.h1.s_vce
        f_vce = mr_adc.h1.f_vce

        s_cva = mr_adc.h1.s_cva
        f_cva = mr_adc.h1.f_cva
        s_vca = mr_adc.h1.s_vca
        f_vca = mr_adc.h1.f_vca

    # Overlap matrices
    S12_c_caa = mr_adc.S12.c_caa
    S12_cae = mr_adc.S12.cae
    S12_cca = mr_adc.S12.cca

    Xt = None

    if transpose:
        if (X.shape[0] != (mr_adc.h0.dim + mr_adc.h1.dim)):
            raise Exception("Dimensions do not match when applying S_12 transpose")

        Xt = np.zeros(mr_adc.h_orth.dim)

        # C and CAA -> C_CAA
        temp = np.zeros((ncvs, S12_c_caa.shape[0]))
        temp[:,0] = X[s_c:f_c].copy()
        temp[:,1:] = X[s_caa:f_caa].reshape(ncvs, -1).copy()
        Xt[ho_s_c_caa:ho_f_c_caa] = np.dot(temp, S12_c_caa).reshape(-1).copy()

        # CCE
        Xt[ho_s_cce:ho_f_cce] = X[s_cce:f_cce].copy()

        if nval > 0:
            # CVE
            Xt[ho_s_cve:ho_f_cve] = X[s_cve:f_cve].copy()

            # VCE
            Xt[ho_s_vce:ho_f_vce] = X[s_vce:f_vce].copy()

        # CAE
        temp = X[s_cae:f_cae].reshape(ncvs, S12_cae.shape[0], nextern).copy()
        Xt[ho_s_cae:ho_f_cae] = einsum("IXA,XP->IPA", temp, S12_cae).reshape(-1).copy()

        # ACE
        temp = X[s_ace:f_ace].reshape(S12_cae.shape[0], ncvs, nextern).copy()
        Xt[ho_s_ace:ho_f_ace] = einsum("XIA,XP->PIA", temp, S12_cae).reshape(-1).copy()

        # CCA
        n_cc = ncvs * ncvs
        temp = X[s_cca:f_cca].reshape(n_cc, S12_cca.shape[0]).copy()
        Xt[ho_s_cca:ho_f_cca] = einsum("IX,XP->IP", temp, S12_cca).reshape(-1).copy()

        if nval > 0:
            # CVA
            n_cv = ncvs * nval
            temp = X[s_cva:f_cva].reshape(n_cv, S12_cca.shape[0]).copy()
            Xt[ho_s_cva:ho_f_cva] = einsum("IX,XP->IP", temp, S12_cca).reshape(-1).copy()

            # VCA
            temp = X[s_vca:f_vca].reshape(n_cv, S12_cca.shape[0]).copy()
            Xt[ho_s_vca:ho_f_vca] = einsum("IX,XP->IP", temp, S12_cca).reshape(-1).copy()

    else:
        if (X.shape[0] != (mr_adc.h_orth.dim)):
            raise Exception("Dimensions do not match when applying S_12")

        Xt = np.zeros(mr_adc.h0.dim + mr_adc.h1.dim)

        # C_CAA -> C and CAA
        temp = X[ho_s_c_caa:ho_f_c_caa].reshape(ncvs, -1).copy()
        temp = np.dot(temp, S12_c_caa.T)
        Xt[s_c:f_c] = temp[:,0].copy()
        Xt[s_caa:f_caa] = temp[:,1:].reshape(-1).copy()

        # CCE
        Xt[s_cce:f_cce] = X[ho_s_cce:ho_f_cce].copy()

        if nval > 0:
            # CVE
            Xt[s_cve:f_cve] = X[ho_s_cve:ho_f_cve].copy()

            # VCE
            Xt[s_vce:f_vce] = X[ho_s_vce:ho_f_vce].copy()

        # CAE
        temp = X[ho_s_cae:ho_f_cae].reshape(ncvs, S12_cae.shape[1], nextern).copy()
        Xt[s_cae:f_cae] = einsum("IPA,XP->IXA", temp, S12_cae).reshape(-1).copy()

        # ACE
        temp = X[ho_s_ace:ho_f_ace].reshape(S12_cae.shape[1], ncvs, nextern).copy()
        Xt[s_ace:f_ace] = einsum("PIA,XP->XIA", temp, S12_cae).reshape(-1).copy()


        # CCA
        n_cc = ncvs * ncvs
        temp = X[ho_s_cca:ho_f_cca].reshape(n_cc, S12_cca.shape[1]).copy()
        Xt[s_cca:f_cca] = einsum("IP,XP->IX", temp, S12_cca).reshape(-1).copy()

        if nval > 0:
            # CVA
            n_cv = ncvs * nval
            temp = X[ho_s_cva:ho_f_cva].reshape(n_cv, S12_cca.shape[1]).copy()
            Xt[s_cva:f_cva] = einsum("IP,XP->IX", temp, S12_cca).reshape(-1).copy()

            # VCA
            temp = X[ho_s_vca:ho_f_vca].reshape(n_cv, S12_cca.shape[1]).copy()
            Xt[s_vca:f_vca] = einsum("IP,XP->IX", temp, S12_cca).reshape(-1).copy()

    return Xt

def compute_sigma_vector_diag(mr_adc, M_00, M_01, M_11, Xt):

    # Einsum definition from kernel
    einsum = mr_adc.interface.einsum
    einsum_type = mr_adc.interface.einsum_type

    # Variables from kernel
    ncvs = mr_adc.ncvs
    nval = mr_adc.nval
    ncas = mr_adc.ncas
    nextern = mr_adc.nextern

    e_cvs = mr_adc.mo_energy.x
    e_core = mr_adc.mo_energy.c
    e_val = mr_adc.mo_energy.v
    e_extern = mr_adc.mo_energy.e

    # One-electron integrals
    h_aa = mr_adc.h1eff.aa

    # Two-electrons integrals
    v_aaaa = mr_adc.v2e.aaaa

    # Reduced Density Matrices
    rdm_ca = mr_adc.rdm.ca
    rdm_ccaa = mr_adc.rdm.ccaa
    rdm_cccaaa = mr_adc.rdm.cccaaa

    # Dimensions
    s_caa = mr_adc.h1.s_caa
    f_caa = mr_adc.h1.f_caa
    s_cce = mr_adc.h1.s_cce
    f_cce = mr_adc.h1.f_cce
    s_cae = mr_adc.h1.s_cae
    f_cae = mr_adc.h1.f_cae
    s_ace = mr_adc.h1.s_ace
    f_ace = mr_adc.h1.f_ace
    s_cca = mr_adc.h1.s_cca
    f_cca = mr_adc.h1.f_cca
    if nval > 0:
        s_cve = mr_adc.h1.s_cve
        f_cve = mr_adc.h1.f_cve
        s_vce = mr_adc.h1.s_vce
        f_vce = mr_adc.h1.f_vce

        s_cva = mr_adc.h1.s_cva
        f_cva = mr_adc.h1.f_cva
        s_vca = mr_adc.h1.s_vca
        f_vca = mr_adc.h1.f_vca

    cvs_tril_ind = np.tril_indices(ncvs, k=-1)

    # (CASCI + C) -> (CASCI + C)
    sigma = np.zeros_like(Xt)

    # h0-h0 contributions
    sigma[:mr_adc.h0.dim] = np.dot(M_00, Xt[:mr_adc.h0.dim])

    # h1-h1 contributions
    # CAA <- CAA
    X_caa = Xt[s_caa:f_caa].reshape(-1).copy()

    dim_WZ = ncas * ncas
    dim_c_caa = ncvs * dim_WZ

    sigma_aaa_i = 0
    sigma_aaa_f = sigma_aaa_i + dim_c_caa
    sigma_abb_i = sigma_aaa_f
    sigma_abb_f = sigma_abb_i + dim_c_caa
    sigma_bab_i = sigma_abb_f
    sigma_bab_f = sigma_bab_i + dim_c_caa

    X_aaa = X_caa[sigma_aaa_i:sigma_aaa_f].reshape(ncvs, ncas, ncas).copy()
    X_abb = X_caa[sigma_abb_i:sigma_abb_f].reshape(ncvs, ncas, ncas).copy()
    X_bab = X_caa[sigma_bab_i:sigma_bab_f].reshape(ncvs, ncas, ncas).copy()

    sigma_caa_aaa  = 1/2 * einsum('KxZ,K,xW->KWZ', X_aaa, e_cvs, rdm_ca, optimize = einsum_type)
    sigma_caa_aaa += 1/6 * einsum('Kxy,K,WyZx->KWZ', X_aaa, e_cvs, rdm_ccaa, optimize = einsum_type)
    sigma_caa_aaa -= 1/6 * einsum('Kxy,K,WyxZ->KWZ', X_aaa, e_cvs, rdm_ccaa, optimize = einsum_type)
    sigma_caa_aaa += 1/2 * einsum('KxZ,xy,yW->KWZ', X_aaa, h_aa, rdm_ca, optimize = einsum_type)
    sigma_caa_aaa -= 1/2 * einsum('Kxy,Zy,xW->KWZ', X_aaa, h_aa, rdm_ca, optimize = einsum_type)
    sigma_caa_aaa += 1/6 * einsum('Kxy,xz,WyZz->KWZ', X_aaa, h_aa, rdm_ccaa, optimize = einsum_type)
    sigma_caa_aaa -= 1/6 * einsum('Kxy,xz,WyzZ->KWZ', X_aaa, h_aa, rdm_ccaa, optimize = einsum_type)
    sigma_caa_aaa -= 1/6 * einsum('Kxy,yz,WzZx->KWZ', X_aaa, h_aa, rdm_ccaa, optimize = einsum_type)
    sigma_caa_aaa += 1/6 * einsum('Kxy,yz,WzxZ->KWZ', X_aaa, h_aa, rdm_ccaa, optimize = einsum_type)
    sigma_caa_aaa += 1/2 * einsum('KxZ,xyzw,Wyzw->KWZ', X_aaa, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_caa_aaa += 1/6 * einsum('Kxy,Zxzw,Wyzw->KWZ', X_aaa, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_caa_aaa -= 1/6 * einsum('Kxy,Zxzw,Wywz->KWZ', X_aaa, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_caa_aaa -= 1/2 * einsum('Kxy,Zzyw,Wzxw->KWZ', X_aaa, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_caa_aaa += 1/6 * einsum('Kxy,Zzwy,Wzxw->KWZ', X_aaa, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_caa_aaa -= 1/6 * einsum('Kxy,Zzwy,Wzwx->KWZ', X_aaa, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_caa_aaa += 1/6 * einsum('Kxy,xzwu,ZwuWyz->KWZ', X_aaa, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_caa_aaa -= 1/6 * einsum('Kxy,xzwu,ZwuyWz->KWZ', X_aaa, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_caa_aaa -= 1/6 * einsum('Kxy,yzwu,ZxzWwu->KWZ', X_aaa, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_caa_aaa += 1/6 * einsum('Kxy,yzwu,ZxzwWu->KWZ', X_aaa, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_caa_aaa += 1/3 * einsum('Kxy,K,WyZx->KWZ', X_abb, e_cvs, rdm_ccaa, optimize = einsum_type)
    sigma_caa_aaa += 1/6 * einsum('Kxy,K,WyxZ->KWZ', X_abb, e_cvs, rdm_ccaa, optimize = einsum_type)
    sigma_caa_aaa += 1/3 * einsum('Kxy,xz,WyZz->KWZ', X_abb, h_aa, rdm_ccaa, optimize = einsum_type)
    sigma_caa_aaa += 1/6 * einsum('Kxy,xz,WyzZ->KWZ', X_abb, h_aa, rdm_ccaa, optimize = einsum_type)
    sigma_caa_aaa -= 1/3 * einsum('Kxy,yz,WzZx->KWZ', X_abb, h_aa, rdm_ccaa, optimize = einsum_type)
    sigma_caa_aaa -= 1/6 * einsum('Kxy,yz,WzxZ->KWZ', X_abb, h_aa, rdm_ccaa, optimize = einsum_type)
    sigma_caa_aaa += 1/3 * einsum('Kxy,Zxzw,Wyzw->KWZ', X_abb, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_caa_aaa += 1/6 * einsum('Kxy,Zxzw,Wywz->KWZ', X_abb, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_caa_aaa -= 1/6 * einsum('Kxy,Zzwy,Wzxw->KWZ', X_abb, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_caa_aaa -= 1/3 * einsum('Kxy,Zzwy,Wzwx->KWZ', X_abb, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_caa_aaa += 1/4 * einsum('Kxy,xzwu,ZwuWyz->KWZ', X_abb, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_caa_aaa -= 1/12 * einsum('Kxy,xzwu,ZwuWzy->KWZ', X_abb, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_caa_aaa += 1/12 * einsum('Kxy,xzwu,ZwuyWz->KWZ', X_abb, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_caa_aaa -= 1/12 * einsum('Kxy,xzwu,ZwuyzW->KWZ', X_abb, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_caa_aaa -= 1/12 * einsum('Kxy,xzwu,ZwuzWy->KWZ', X_abb, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_caa_aaa -= 1/12 * einsum('Kxy,xzwu,ZwuzyW->KWZ', X_abb, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_caa_aaa -= 1/4 * einsum('Kxy,yzwu,ZxzWwu->KWZ', X_abb, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_caa_aaa += 1/12 * einsum('Kxy,yzwu,ZxzWuw->KWZ', X_abb, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_caa_aaa -= 1/12 * einsum('Kxy,yzwu,ZxzwWu->KWZ', X_abb, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_caa_aaa += 1/12 * einsum('Kxy,yzwu,ZxzwuW->KWZ', X_abb, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_caa_aaa += 1/12 * einsum('Kxy,yzwu,ZxzuWw->KWZ', X_abb, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_caa_aaa += 1/12 * einsum('Kxy,yzwu,ZxzuwW->KWZ', X_abb, v_aaaa, rdm_cccaaa, optimize = einsum_type)

    sigma_caa_abb  = 1/3 * einsum('Kxy,K,WyZx->KWZ', X_aaa, e_cvs, rdm_ccaa, optimize = einsum_type)
    sigma_caa_abb += 1/6 * einsum('Kxy,K,WyxZ->KWZ', X_aaa, e_cvs, rdm_ccaa, optimize = einsum_type)
    sigma_caa_abb += 1/3 * einsum('Kxy,xz,WyZz->KWZ', X_aaa, h_aa, rdm_ccaa, optimize = einsum_type)
    sigma_caa_abb += 1/6 * einsum('Kxy,xz,WyzZ->KWZ', X_aaa, h_aa, rdm_ccaa, optimize = einsum_type)
    sigma_caa_abb -= 1/3 * einsum('Kxy,yz,WzZx->KWZ', X_aaa, h_aa, rdm_ccaa, optimize = einsum_type)
    sigma_caa_abb -= 1/6 * einsum('Kxy,yz,WzxZ->KWZ', X_aaa, h_aa, rdm_ccaa, optimize = einsum_type)
    sigma_caa_abb += 1/3 * einsum('Kxy,Zxzw,Wyzw->KWZ', X_aaa, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_caa_abb += 1/6 * einsum('Kxy,Zxzw,Wywz->KWZ', X_aaa, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_caa_abb -= 1/6 * einsum('Kxy,Zzwy,Wzxw->KWZ', X_aaa, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_caa_abb -= 1/3 * einsum('Kxy,Zzwy,Wzwx->KWZ', X_aaa, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_caa_abb += 1/4 * einsum('Kxy,xzwu,ZwuWyz->KWZ', X_aaa, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_caa_abb -= 1/12 * einsum('Kxy,xzwu,ZwuWzy->KWZ', X_aaa, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_caa_abb += 1/12 * einsum('Kxy,xzwu,ZwuyWz->KWZ', X_aaa, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_caa_abb -= 1/12 * einsum('Kxy,xzwu,ZwuyzW->KWZ', X_aaa, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_caa_abb -= 1/12 * einsum('Kxy,xzwu,ZwuzWy->KWZ', X_aaa, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_caa_abb -= 1/12 * einsum('Kxy,xzwu,ZwuzyW->KWZ', X_aaa, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_caa_abb -= 1/4 * einsum('Kxy,yzwu,ZxzWwu->KWZ', X_aaa, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_caa_abb += 1/12 * einsum('Kxy,yzwu,ZxzWuw->KWZ', X_aaa, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_caa_abb -= 1/12 * einsum('Kxy,yzwu,ZxzwWu->KWZ', X_aaa, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_caa_abb += 1/12 * einsum('Kxy,yzwu,ZxzwuW->KWZ', X_aaa, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_caa_abb += 1/12 * einsum('Kxy,yzwu,ZxzuWw->KWZ', X_aaa, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_caa_abb += 1/12 * einsum('Kxy,yzwu,ZxzuwW->KWZ', X_aaa, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_caa_abb += 1/2 * einsum('KxZ,K,xW->KWZ', X_abb, e_cvs, rdm_ca, optimize = einsum_type)
    sigma_caa_abb += 1/6 * einsum('Kxy,K,WyZx->KWZ', X_abb, e_cvs, rdm_ccaa, optimize = einsum_type)
    sigma_caa_abb -= 1/6 * einsum('Kxy,K,WyxZ->KWZ', X_abb, e_cvs, rdm_ccaa, optimize = einsum_type)
    sigma_caa_abb += 1/2 * einsum('KxZ,xy,yW->KWZ', X_abb, h_aa, rdm_ca, optimize = einsum_type)
    sigma_caa_abb -= 1/2 * einsum('Kxy,Zy,xW->KWZ', X_abb, h_aa, rdm_ca, optimize = einsum_type)
    sigma_caa_abb += 1/6 * einsum('Kxy,xz,WyZz->KWZ', X_abb, h_aa, rdm_ccaa, optimize = einsum_type)
    sigma_caa_abb -= 1/6 * einsum('Kxy,xz,WyzZ->KWZ', X_abb, h_aa, rdm_ccaa, optimize = einsum_type)
    sigma_caa_abb -= 1/6 * einsum('Kxy,yz,WzZx->KWZ', X_abb, h_aa, rdm_ccaa, optimize = einsum_type)
    sigma_caa_abb += 1/6 * einsum('Kxy,yz,WzxZ->KWZ', X_abb, h_aa, rdm_ccaa, optimize = einsum_type)
    sigma_caa_abb += 1/2 * einsum('KxZ,xyzw,Wyzw->KWZ', X_abb, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_caa_abb += 1/6 * einsum('Kxy,Zxzw,Wyzw->KWZ', X_abb, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_caa_abb -= 1/6 * einsum('Kxy,Zxzw,Wywz->KWZ', X_abb, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_caa_abb -= 1/2 * einsum('Kxy,Zzyw,Wzxw->KWZ', X_abb, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_caa_abb += 1/6 * einsum('Kxy,Zzwy,Wzxw->KWZ', X_abb, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_caa_abb -= 1/6 * einsum('Kxy,Zzwy,Wzwx->KWZ', X_abb, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_caa_abb += 1/6 * einsum('Kxy,xzwu,ZwuWyz->KWZ', X_abb, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_caa_abb -= 1/6 * einsum('Kxy,xzwu,ZwuyWz->KWZ', X_abb, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_caa_abb -= 1/6 * einsum('Kxy,yzwu,ZxzWwu->KWZ', X_abb, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_caa_abb += 1/6 * einsum('Kxy,yzwu,ZxzwWu->KWZ', X_abb, v_aaaa, rdm_cccaaa, optimize = einsum_type)

    sigma_caa_bab  = 1/2 * einsum('KxZ,K,xW->KWZ', X_bab, e_cvs, rdm_ca, optimize = einsum_type)
    sigma_caa_bab -= 1/6 * einsum('Kxy,K,WyZx->KWZ', X_bab, e_cvs, rdm_ccaa, optimize = einsum_type)
    sigma_caa_bab -= 1/3 * einsum('Kxy,K,WyxZ->KWZ', X_bab, e_cvs, rdm_ccaa, optimize = einsum_type)
    sigma_caa_bab += 1/2 * einsum('KxZ,xy,yW->KWZ', X_bab, h_aa, rdm_ca, optimize = einsum_type)
    sigma_caa_bab -= 1/2 * einsum('Kxy,Zy,xW->KWZ', X_bab, h_aa, rdm_ca, optimize = einsum_type)
    sigma_caa_bab -= 1/6 * einsum('Kxy,xz,WyZz->KWZ', X_bab, h_aa, rdm_ccaa, optimize = einsum_type)
    sigma_caa_bab -= 1/3 * einsum('Kxy,xz,WyzZ->KWZ', X_bab, h_aa, rdm_ccaa, optimize = einsum_type)
    sigma_caa_bab += 1/6 * einsum('Kxy,yz,WzZx->KWZ', X_bab, h_aa, rdm_ccaa, optimize = einsum_type)
    sigma_caa_bab += 1/3 * einsum('Kxy,yz,WzxZ->KWZ', X_bab, h_aa, rdm_ccaa, optimize = einsum_type)
    sigma_caa_bab += 1/2 * einsum('KxZ,xyzw,Wyzw->KWZ', X_bab, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_caa_bab -= 1/6 * einsum('Kxy,Zxzw,Wyzw->KWZ', X_bab, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_caa_bab -= 1/3 * einsum('Kxy,Zxzw,Wywz->KWZ', X_bab, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_caa_bab -= 1/2 * einsum('Kxy,Zzyw,Wzxw->KWZ', X_bab, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_caa_bab += 1/3 * einsum('Kxy,Zzwy,Wzxw->KWZ', X_bab, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_caa_bab += 1/6 * einsum('Kxy,Zzwy,Wzwx->KWZ', X_bab, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_caa_bab -= 1/12 * einsum('Kxy,xzwu,ZwuWyz->KWZ', X_bab, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_caa_bab += 1/12 * einsum('Kxy,xzwu,ZwuWzy->KWZ', X_bab, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_caa_bab -= 1/4 * einsum('Kxy,xzwu,ZwuyWz->KWZ', X_bab, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_caa_bab += 1/12 * einsum('Kxy,xzwu,ZwuyzW->KWZ', X_bab, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_caa_bab += 1/12 * einsum('Kxy,xzwu,ZwuzWy->KWZ', X_bab, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_caa_bab += 1/12 * einsum('Kxy,xzwu,ZwuzyW->KWZ', X_bab, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_caa_bab += 1/12 * einsum('Kxy,yzwu,ZxzWwu->KWZ', X_bab, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_caa_bab -= 1/12 * einsum('Kxy,yzwu,ZxzWuw->KWZ', X_bab, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_caa_bab += 1/4 * einsum('Kxy,yzwu,ZxzwWu->KWZ', X_bab, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_caa_bab -= 1/12 * einsum('Kxy,yzwu,ZxzwuW->KWZ', X_bab, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_caa_bab -= 1/12 * einsum('Kxy,yzwu,ZxzuWw->KWZ', X_bab, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_caa_bab -= 1/12 * einsum('Kxy,yzwu,ZxzuwW->KWZ', X_bab, v_aaaa, rdm_cccaaa, optimize = einsum_type)

    ## Building C-CAA matrix
    dim_caa = ncvs * ncas * ncas

    sigma_aaa_i = s_caa
    sigma_aaa_f = sigma_aaa_i + dim_caa
    sigma_abb_i = sigma_aaa_f
    sigma_abb_f = sigma_abb_i + dim_caa
    sigma_bab_i = sigma_abb_f
    sigma_bab_f = sigma_bab_i + dim_caa

    sigma[sigma_aaa_i:sigma_aaa_f] += sigma_caa_aaa.reshape(-1).copy()
    sigma[sigma_abb_i:sigma_abb_f] += sigma_caa_abb.reshape(-1).copy()
    sigma[sigma_bab_i:sigma_bab_f] += sigma_caa_bab.reshape(-1).copy()

    # CCE <- CCE
    X = Xt[s_cce:f_cce].reshape(ncvs, ncvs, nextern).copy()

    sigma_cce =- einsum('KLB,B->KLB', X, e_extern, optimize = einsum_type)
    sigma_cce += einsum('KLB,K->KLB', X, e_cvs, optimize = einsum_type)
    sigma_cce += einsum('KLB,L->KLB', X, e_cvs, optimize = einsum_type)
    sigma[s_cce:f_cce] += sigma_cce.reshape(-1).copy()

    if nval > 0:
        # CVE <- CVE
        X = Xt[s_cve:f_cve].reshape(ncvs, nval, nextern).copy()

        sigma_cve =- einsum('KLB,B->KLB', X, e_extern, optimize = einsum_type)
        sigma_cve += einsum('KLB,K->KLB', X, e_cvs, optimize = einsum_type)
        sigma_cve += einsum('KLB,L->KLB', X, e_val, optimize = einsum_type)
        sigma[s_cve:f_cve] += sigma_cve.reshape(-1).copy()

        # VCE <- VCE
        X = Xt[s_vce:f_vce].reshape(nval, ncvs, nextern).copy()

        sigma_vce =- einsum('KLB,B->KLB', X, e_extern, optimize = einsum_type)
        sigma_vce += einsum('KLB,K->KLB', X, e_val, optimize = einsum_type)
        sigma_vce += einsum('KLB,L->KLB', X, e_cvs, optimize = einsum_type)
        sigma[s_vce:f_vce] += sigma_vce.reshape(-1).copy()

    # CAE <- CAE
    X = Xt[s_cae:f_cae].reshape(ncvs, ncas, nextern).copy()

    sigma_cae =- 1/2 * einsum('KxB,B,xY->KYB', X, e_extern, rdm_ca, optimize = einsum_type)
    sigma_cae += 1/2 * einsum('KxB,K,xY->KYB', X, e_cvs, rdm_ca, optimize = einsum_type)
    sigma_cae += 1/2 * einsum('KxB,xy,yY->KYB', X, h_aa, rdm_ca, optimize = einsum_type)
    sigma_cae += 1/2 * einsum('KxB,xyzw,Yyzw->KYB', X, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma[s_cae:f_cae] += sigma_cae.reshape(-1).copy()

    # ACE <- ACE
    X = Xt[s_ace:f_ace].reshape(ncas, ncvs, nextern).copy()

    sigma_ace =- 1/2 * einsum('xKB,B,xY->YKB', X, e_extern, rdm_ca, optimize = einsum_type)
    sigma_ace += 1/2 * einsum('xKB,K,xY->YKB', X, e_cvs, rdm_ca, optimize = einsum_type)
    sigma_ace += 1/2 * einsum('xKB,xy,yY->YKB', X, h_aa, rdm_ca, optimize = einsum_type)
    sigma_ace += 1/2 * einsum('xKB,xyzw,Yyzw->YKB', X, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma[s_ace:f_ace] += sigma_ace.reshape(-1).copy()

    # CCA <- CCA
    X = Xt[s_cca:f_cca].reshape(ncvs, ncvs, ncas).copy()

    sigma_cca  = einsum('KLY,K->KLY', X, e_cvs, optimize = einsum_type)
    sigma_cca += einsum('KLY,L->KLY', X, e_cvs, optimize = einsum_type)
    sigma_cca -= einsum('KLx,Yx->KLY', X, h_aa, optimize = einsum_type)
    sigma_cca -= 1/2 * einsum('KLx,K,Yx->KLY', X, e_cvs, rdm_ca, optimize = einsum_type)
    sigma_cca -= 1/2 * einsum('KLx,L,Yx->KLY', X, e_cvs, rdm_ca, optimize = einsum_type)
    sigma_cca += 1/2 * einsum('KLx,xy,Yy->KLY', X, h_aa, rdm_ca, optimize = einsum_type)
    sigma_cca -= einsum('KLx,Yyxz,zy->KLY', X, v_aaaa, rdm_ca, optimize = einsum_type)
    sigma_cca += 1/2 * einsum('KLx,Yyzx,zy->KLY', X, v_aaaa, rdm_ca, optimize = einsum_type)
    sigma_cca += 1/2 * einsum('KLx,xyzw,Yyzw->KLY', X, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma[s_cca:f_cca] += sigma_cca.reshape(-1).copy()

    if nval > 0:
        # CVA <- CVA
        X = Xt[s_cva:f_cva].reshape(ncvs, nval, ncas).copy()

        sigma_cva  = einsum('KLY,K->KLY', X, e_cvs, optimize = einsum_type)
        sigma_cva += einsum('KLY,L->KLY', X, e_val, optimize = einsum_type)
        sigma_cva -= einsum('KLx,Yx->KLY', X, h_aa, optimize = einsum_type)
        sigma_cva -= 1/2 * einsum('KLx,K,Yx->KLY', X, e_cvs, rdm_ca, optimize = einsum_type)
        sigma_cva -= 1/2 * einsum('KLx,L,Yx->KLY', X, e_val, rdm_ca, optimize = einsum_type)
        sigma_cva += 1/2 * einsum('KLx,xy,Yy->KLY', X, h_aa, rdm_ca, optimize = einsum_type)
        sigma_cva -= einsum('KLx,Yyxz,zy->KLY', X, v_aaaa, rdm_ca, optimize = einsum_type)
        sigma_cva += 1/2 * einsum('KLx,Yyzx,zy->KLY', X, v_aaaa, rdm_ca, optimize = einsum_type)
        sigma_cva += 1/2 * einsum('KLx,xyzw,Yyzw->KLY', X, v_aaaa, rdm_ccaa, optimize = einsum_type)
        sigma[s_cva:f_cva] += sigma_cva.reshape(-1).copy()

        # VCA <- VCA
        X = Xt[s_vca:f_vca].reshape(nval, ncvs, ncas).copy()

        sigma_vca  = einsum('KLY,K->KLY', X, e_val, optimize = einsum_type)
        sigma_vca += einsum('KLY,L->KLY', X, e_cvs, optimize = einsum_type)
        sigma_vca -= einsum('KLx,Yx->KLY', X, h_aa, optimize = einsum_type)
        sigma_vca -= 1/2 * einsum('KLx,K,Yx->KLY', X, e_val, rdm_ca, optimize = einsum_type)
        sigma_vca -= 1/2 * einsum('KLx,L,Yx->KLY', X, e_cvs, rdm_ca, optimize = einsum_type)
        sigma_vca += 1/2 * einsum('KLx,xy,Yy->KLY', X, h_aa, rdm_ca, optimize = einsum_type)
        sigma_vca -= einsum('KLx,Yyxz,zy->KLY', X, v_aaaa, rdm_ca, optimize = einsum_type)
        sigma_vca += 1/2 * einsum('KLx,Yyzx,zy->KLY', X, v_aaaa, rdm_ca, optimize = einsum_type)
        sigma_vca += 1/2 * einsum('KLx,xyzw,Yyzw->KLY', X, v_aaaa, rdm_ccaa, optimize = einsum_type)
        sigma[s_vca:f_vca] += sigma_vca.reshape(-1).copy()

    return sigma

## CCE block (CCE + CVE + VCE): Diagonal + M01
def compute_excitation_manifolds_cce(mr_adc):

    # MR-ADC(0) and MR-ADC(1)
    mr_adc.h0.n_c = mr_adc.ncvs
    mr_adc.h0.dim = mr_adc.h0.n_c # Total dimension of h0

    mr_adc.h0.s_c = 0
    mr_adc.h0.f_c = mr_adc.h0.s_c + mr_adc.h0.n_c

    print("Dimension of h0 excitation manifold:                       %d" % mr_adc.h0.dim)

    # MR-ADC(2)
    mr_adc.h1.dim = 0
    mr_adc.h_orth.dim = mr_adc.h0.dim

    if mr_adc.method in ("mr-adc(2)", "mr-adc(2)-x"):
        mr_adc.h1.n_caa = 0
        mr_adc.h1.n_cce_aaa = mr_adc.nextern * mr_adc.ncvs * (mr_adc.ncvs - 1) //2
        mr_adc.h1.n_cce_abb = mr_adc.nextern * mr_adc.ncvs * mr_adc.ncvs
        mr_adc.h1.n_cce = mr_adc.h1.n_cce_aaa + mr_adc.h1.n_cce_abb
        mr_adc.h1.n_cae = 0
        mr_adc.h1.n_ace = 0
        mr_adc.h1.n_cca = 0
        if mr_adc.nval > 0:
            mr_adc.h1.n_cve = mr_adc.nextern * mr_adc.ncvs * mr_adc.nval
            mr_adc.h1.n_cva = 0
            mr_adc.h1.n_vca = 0
            mr_adc.h1.dim = (mr_adc.h1.n_caa + mr_adc.h1.n_cce + 3 * mr_adc.h1.n_cve +
                             mr_adc.h1.n_cae + mr_adc.h1.n_ace + mr_adc.h1.n_cca + mr_adc.h1.n_cva + mr_adc.h1.n_vca)
        else:
            mr_adc.h1.dim = mr_adc.h1.n_caa + mr_adc.h1.n_cce + mr_adc.h1.n_cae + mr_adc.h1.n_cae + mr_adc.h1.n_cca

        if mr_adc.nval > 0:
            mr_adc.h1.s_caa = mr_adc.h0.f_c
            mr_adc.h1.f_caa = mr_adc.h1.s_caa + mr_adc.h1.n_caa
            mr_adc.h1.s_cce_aaa = mr_adc.h1.f_caa
            mr_adc.h1.f_cce_aaa = mr_adc.h1.s_cce_aaa + mr_adc.h1.n_cce_aaa
            mr_adc.h1.s_cce_abb = mr_adc.h1.f_cce_aaa
            mr_adc.h1.f_cce_abb = mr_adc.h1.s_cce_abb + mr_adc.h1.n_cce_abb
            mr_adc.h1.s_cve_aaa = mr_adc.h1.f_cce_abb
            mr_adc.h1.f_cve_aaa = mr_adc.h1.s_cve_aaa + mr_adc.h1.n_cve
            mr_adc.h1.s_cve_abb = mr_adc.h1.f_cve_aaa
            mr_adc.h1.f_cve_abb = mr_adc.h1.s_cve_abb + mr_adc.h1.n_cve
            mr_adc.h1.s_cve_bab = mr_adc.h1.f_cve_abb
            mr_adc.h1.f_cve_bab = mr_adc.h1.s_cve_bab + mr_adc.h1.n_cve
            mr_adc.h1.s_cae = mr_adc.h1.f_cve_bab
            mr_adc.h1.f_cae = mr_adc.h1.s_cae + mr_adc.h1.n_cae
            mr_adc.h1.s_ace = mr_adc.h1.f_cae
            mr_adc.h1.f_ace = mr_adc.h1.s_ace + mr_adc.h1.n_ace
            mr_adc.h1.s_cca = mr_adc.h1.f_ace
            mr_adc.h1.f_cca = mr_adc.h1.s_cca + mr_adc.h1.n_cca
            mr_adc.h1.s_cva = mr_adc.h1.f_cca
            mr_adc.h1.f_cva = mr_adc.h1.s_cva + mr_adc.h1.n_cva
            mr_adc.h1.s_vca = mr_adc.h1.f_cva
            mr_adc.h1.f_vca = mr_adc.h1.s_vca + mr_adc.h1.n_vca
        else:
            mr_adc.h1.s_caa = mr_adc.h0.f_c
            mr_adc.h1.f_caa = mr_adc.h1.s_caa + mr_adc.h1.n_caa
            mr_adc.h1.s_cce_aaa = mr_adc.h1.f_caa
            mr_adc.h1.f_cce_aaa = mr_adc.h1.s_cce_aaa + mr_adc.h1.n_cce_aaa
            mr_adc.h1.s_cce_abb = mr_adc.h1.f_cce_aaa
            mr_adc.h1.f_cce_abb = mr_adc.h1.s_cce_abb + mr_adc.h1.n_cce_abb
            mr_adc.h1.s_cae = mr_adc.h1.f_cce_abb
            mr_adc.h1.f_cae = mr_adc.h1.s_cae + mr_adc.h1.n_cae
            mr_adc.h1.s_ace = mr_adc.h1.f_cae
            mr_adc.h1.f_ace = mr_adc.h1.s_ace + mr_adc.h1.n_ace
            mr_adc.h1.s_cca = mr_adc.h1.f_ace
            mr_adc.h1.f_cca = mr_adc.h1.s_cca + mr_adc.h1.n_cca

        print("Dimension of h1 excitation manifold:                       %d" % mr_adc.h1.dim)

        # Overlap for c - caa
        mr_adc.S12.c_caa = mr_adc_overlap.compute_S12_0p_projector(mr_adc)
        mr_adc.S12.cae = mr_adc_overlap.compute_S12_m1(mr_adc)
        mr_adc.S12.cca = mr_adc_overlap.compute_S12_p1(mr_adc)

        # Determine dimensions of orthogonalized excitation spaces
        mr_adc.h_orth.n_c = mr_adc.ncvs
        mr_adc.h_orth.n_c_caa = 0
        mr_adc.h_orth.n_cce_aaa = mr_adc.h1.n_cce_aaa
        mr_adc.h_orth.n_cce_abb = mr_adc.h1.n_cce_abb
        mr_adc.h_orth.n_cce = mr_adc.h1.n_cce
        mr_adc.h_orth.n_cae = 0
        mr_adc.h_orth.n_ace = 0
        mr_adc.h_orth.n_cca = 0
        if mr_adc.nval > 0:
            mr_adc.h_orth.n_cve = mr_adc.h1.n_cve
            mr_adc.h_orth.n_cva = 0
            mr_adc.h_orth.n_vca = 0
            mr_adc.h_orth.dim = (mr_adc.h_orth.n_c + mr_adc.h_orth.n_c_caa + mr_adc.h_orth.n_cce + 3 * mr_adc.h_orth.n_cve +
                                 mr_adc.h_orth.n_cae + mr_adc.h_orth.n_ace + mr_adc.h_orth.n_cca + mr_adc.h_orth.n_cva + mr_adc.h_orth.n_vca)
        else:
            mr_adc.h_orth.dim = mr_adc.h_orth.n_c + mr_adc.h_orth.n_c_caa + mr_adc.h_orth.n_cce + mr_adc.h_orth.n_cae + mr_adc.h_orth.n_ace + mr_adc.h_orth.n_cca

        if mr_adc.nval > 0:
            mr_adc.h_orth.s_c = 0
            mr_adc.h_orth.f_c = mr_adc.h_orth.n_c
            mr_adc.h_orth.s_c_caa = mr_adc.h_orth.f_c
            mr_adc.h_orth.f_c_caa = mr_adc.h_orth.s_c_caa + mr_adc.h_orth.n_c_caa
            mr_adc.h_orth.s_cce_aaa = mr_adc.h_orth.f_c_caa
            mr_adc.h_orth.f_cce_aaa = mr_adc.h_orth.s_cce_aaa + mr_adc.h_orth.n_cce_aaa
            mr_adc.h_orth.s_cce_abb = mr_adc.h_orth.f_cce_aaa
            mr_adc.h_orth.f_cce_abb = mr_adc.h_orth.s_cce_abb + mr_adc.h_orth.n_cce_abb
            mr_adc.h_orth.s_cve_aaa = mr_adc.h_orth.f_cce_abb
            mr_adc.h_orth.f_cve_aaa = mr_adc.h_orth.s_cve_aaa + mr_adc.h_orth.n_cve
            mr_adc.h_orth.s_cve_abb = mr_adc.h_orth.f_cve_aaa
            mr_adc.h_orth.f_cve_abb = mr_adc.h_orth.s_cve_abb + mr_adc.h_orth.n_cve
            mr_adc.h_orth.s_cve_bab = mr_adc.h_orth.f_cve_abb
            mr_adc.h_orth.f_cve_bab = mr_adc.h_orth.s_cve_bab + mr_adc.h_orth.n_cve
            mr_adc.h_orth.s_cae = mr_adc.h_orth.f_cve_bab
            mr_adc.h_orth.f_cae = mr_adc.h_orth.s_cae + mr_adc.h_orth.n_cae
            mr_adc.h_orth.s_ace = mr_adc.h_orth.f_cae
            mr_adc.h_orth.f_ace = mr_adc.h_orth.s_ace + mr_adc.h_orth.n_ace
            mr_adc.h_orth.s_cca = mr_adc.h_orth.f_ace
            mr_adc.h_orth.f_cca = mr_adc.h_orth.s_cca + mr_adc.h_orth.n_cca
            mr_adc.h_orth.s_cva = mr_adc.h_orth.f_cca
            mr_adc.h_orth.f_cva = mr_adc.h_orth.s_cva + mr_adc.h_orth.n_cva
            mr_adc.h_orth.s_vca = mr_adc.h_orth.f_cva
            mr_adc.h_orth.f_vca = mr_adc.h_orth.s_vca + mr_adc.h_orth.n_vca
        else:
            mr_adc.h_orth.s_c = 0
            mr_adc.h_orth.f_c = mr_adc.h_orth.n_c
            mr_adc.h_orth.s_c_caa = mr_adc.h_orth.f_c
            mr_adc.h_orth.f_c_caa = mr_adc.h_orth.s_c_caa + mr_adc.h_orth.n_c_caa
            mr_adc.h_orth.s_cce_aaa = mr_adc.h_orth.f_c_caa
            mr_adc.h_orth.f_cce_aaa = mr_adc.h_orth.s_cce_aaa + mr_adc.h_orth.n_cce_aaa
            mr_adc.h_orth.s_cce_abb = mr_adc.h_orth.f_cce_aaa
            mr_adc.h_orth.f_cce_abb = mr_adc.h_orth.s_cce_abb + mr_adc.h_orth.n_cce_abb
            mr_adc.h_orth.s_cae = mr_adc.h_orth.f_cce_abb
            mr_adc.h_orth.f_cae = mr_adc.h_orth.s_cae + mr_adc.h_orth.n_cae
            mr_adc.h_orth.s_ace = mr_adc.h_orth.f_cae
            mr_adc.h_orth.f_ace = mr_adc.h_orth.s_ace + mr_adc.h_orth.n_ace
            mr_adc.h_orth.s_cca = mr_adc.h_orth.f_ace
            mr_adc.h_orth.f_cca = mr_adc.h_orth.s_cca + mr_adc.h_orth.n_cca

    print("Total dimension of the excitation manifold:                %d" % (mr_adc.h0.dim + mr_adc.h1.dim))
    print("Dimension of the orthogonalized excitation manifold:       %d\n" % (mr_adc.h_orth.dim))
    sys.stdout.flush()

    if (mr_adc.h_orth.dim < mr_adc.nroots):
        mr_adc.nroots = mr_adc.h_orth.dim

    return mr_adc

def compute_M_01_cce(mr_adc):

    start_time = time.time()

    print ("Computing M(h0-h1) blocks...")
    sys.stdout.flush()

    # Einsum definition from kernel
    einsum = mr_adc.interface.einsum
    einsum_type = mr_adc.interface.einsum_type

    # Dimensions
    ncore = mr_adc.ncore
    ncas = mr_adc.ncas
    nocc = mr_adc.nocc
    nextern = mr_adc.nextern

    ncvs = mr_adc.ncvs
    nval = mr_adc.nval

    # cvs_tril_ind = np.tril_indices(ncvs, k=-1)

    # MOs Energy
    e_cvs = mr_adc.mo_energy.x
    if nval > 0:
        e_val = mr_adc.mo_energy.v
    e_core = mr_adc.mo_energy.c
    e_extern = mr_adc.mo_energy.e

    # Amplitudes
    t1_ce = mr_adc.t1.ce
    t1_ca = mr_adc.t1.ca
    t1_ae = mr_adc.t1.ae
    t1_caea = mr_adc.t1.caea
    t1_caae = mr_adc.t1.caae
    t1_caaa = mr_adc.t1.caaa
    t1_aaea = mr_adc.t1.aaea

    t1_xe = mr_adc.t1.xe
    t1_xaea = mr_adc.t1.xaea
    t1_xaae = mr_adc.t1.xaae

    if nval > 0:
        t1_ve = mr_adc.t1.ve
        t1_vaea = mr_adc.t1.vaea
        t1_vaae = mr_adc.t1.vaae

    # One-electron integrals
    h_aa = mr_adc.h1eff.aa

    h_xe = mr_adc.h1eff.xe

    if nval > 0:
        h_ve = mr_adc.h1eff.ve

    # Two-electrons integrals
    v_aaaa = mr_adc.v2e.aaaa

    v_xaxa = mr_adc.v2e.xaxa
    v_xaax = mr_adc.v2e.xaax

    if nval > 0:
        v_vxxe = mr_adc.v2e.vxxe
        v_xvxe = mr_adc.v2e.xvxe

    v_xaea = mr_adc.v2e.xaea
    v_xaae = mr_adc.v2e.xaae
    v_xxxe = mr_adc.v2e.xxxe

    if nval > 0:
        v_vaea = mr_adc.v2e.vaea
        v_vaae = mr_adc.v2e.vaae

    # Reduced Density Matrices
    rdm_ca = mr_adc.rdm.ca
    rdm_ccaa = mr_adc.rdm.ccaa
    rdm_cccaaa = mr_adc.rdm.cccaaa

    # Indices
    cvs_tril_ind = np.tril_indices(ncvs, k=-1)

    # C - CCE
    M_C_CCE_a_abb  = einsum('KLIB->IKLB', v_xxxe, optimize = einsum_type).copy()
    M_C_CCE_a_abb -= einsum('LB,IK->IKLB', h_xe, np.identity(ncvs), optimize = einsum_type)
    M_C_CCE_a_abb -= einsum('B,IK,LB->IKLB', e_extern, np.identity(ncvs), t1_xe, optimize = einsum_type)
    M_C_CCE_a_abb += einsum('L,IK,LB->IKLB', e_cvs, np.identity(ncvs), t1_xe, optimize = einsum_type)
    M_C_CCE_a_abb -= einsum('IK,LxBy,yx->IKLB', np.identity(ncvs), v_xaea, rdm_ca, optimize = einsum_type)
    M_C_CCE_a_abb += 1/2 * einsum('IK,LxyB,yx->IKLB', np.identity(ncvs), v_xaae, rdm_ca, optimize = einsum_type)
    M_C_CCE_a_abb -= einsum('B,IK,LxBy,yx->IKLB', e_extern, np.identity(ncvs), t1_xaea, rdm_ca, optimize = einsum_type)
    M_C_CCE_a_abb += 1/2 * einsum('B,IK,LxyB,yx->IKLB', e_extern, np.identity(ncvs), t1_xaae, rdm_ca, optimize = einsum_type)
    M_C_CCE_a_abb += einsum('L,IK,LxBy,yx->IKLB', e_cvs, np.identity(ncvs), t1_xaea, rdm_ca, optimize = einsum_type)
    M_C_CCE_a_abb -= 1/2 * einsum('L,IK,LxyB,yx->IKLB', e_cvs, np.identity(ncvs), t1_xaae, rdm_ca, optimize = einsum_type)
    M_C_CCE_a_abb += einsum('xy,IK,LxBz,zy->IKLB', h_aa, np.identity(ncvs), t1_xaea, rdm_ca, optimize = einsum_type)
    M_C_CCE_a_abb -= 1/2 * einsum('xy,IK,LxzB,zy->IKLB', h_aa, np.identity(ncvs), t1_xaae, rdm_ca, optimize = einsum_type)
    M_C_CCE_a_abb -= einsum('xy,IK,LzBx,yz->IKLB', h_aa, np.identity(ncvs), t1_xaea, rdm_ca, optimize = einsum_type)
    M_C_CCE_a_abb += 1/2 * einsum('xy,IK,LzxB,yz->IKLB', h_aa, np.identity(ncvs), t1_xaae, rdm_ca, optimize = einsum_type)
    M_C_CCE_a_abb += einsum('IK,LxBy,xzwu,yzwu->IKLB', np.identity(ncvs), t1_xaea, v_aaaa, rdm_ccaa, optimize = einsum_type)
    M_C_CCE_a_abb -= einsum('IK,LxBy,yzwu,xzwu->IKLB', np.identity(ncvs), t1_xaea, v_aaaa, rdm_ccaa, optimize = einsum_type)
    M_C_CCE_a_abb -= 1/2 * einsum('IK,LxyB,xzwu,yzwu->IKLB', np.identity(ncvs), t1_xaae, v_aaaa, rdm_ccaa, optimize = einsum_type)
    M_C_CCE_a_abb += 1/2 * einsum('IK,LxyB,yzwu,xzwu->IKLB', np.identity(ncvs), t1_xaae, v_aaaa, rdm_ccaa, optimize = einsum_type)

    M_C_CCE_a_aaa = M_C_CCE_a_abb - M_C_CCE_a_abb.transpose(0,2,1,3)

    ## Reshape tensors to matrix form
    M_C_CCE_a_aaa = M_C_CCE_a_aaa[:, cvs_tril_ind[0], cvs_tril_ind[1]]

    M_C_CCE_a_aaa = M_C_CCE_a_aaa.reshape(ncvs, -1)
    M_C_CCE_a_abb = M_C_CCE_a_abb.reshape(ncvs, -1)

    ## Building C-CCE matrix
    dim_cce = ncvs * ncvs * nextern
    dim_tril_cce = ncvs * (ncvs - 1) * nextern // 2
    dim_c_cce = dim_tril_cce + dim_cce

    m_c_cce_aaa_i = 0
    m_c_cce_aaa_f = m_c_cce_aaa_i + dim_tril_cce
    m_c_cce_abb_i = m_c_cce_aaa_f
    m_c_cce_abb_f = m_c_cce_abb_i + dim_cce

    M_C_CCE = np.zeros((ncvs, dim_c_cce))
    M_C_CCE[:, m_c_cce_aaa_i:m_c_cce_aaa_f] = M_C_CCE_a_aaa.copy()
    M_C_CCE[:, m_c_cce_abb_i:m_c_cce_abb_f] = M_C_CCE_a_abb.copy()

    if nval > 0:
        # C - CVE
        M_C_CVE_a_abb  = einsum('KLIB->IKLB', v_xvxe, optimize = einsum_type).copy()
        M_C_CVE_a_abb -= einsum('LB,IK->IKLB', h_ve, np.identity(ncvs), optimize = einsum_type)
        M_C_CVE_a_abb -= einsum('B,IK,LB->IKLB', e_extern, np.identity(ncvs), t1_ve, optimize = einsum_type)
        M_C_CVE_a_abb += einsum('L,IK,LB->IKLB', e_val, np.identity(ncvs), t1_ve, optimize = einsum_type)
        M_C_CVE_a_abb -= einsum('IK,LxBy,yx->IKLB', np.identity(ncvs), v_vaea, rdm_ca, optimize = einsum_type)
        M_C_CVE_a_abb += 1/2 * einsum('IK,LxyB,yx->IKLB', np.identity(ncvs), v_vaae, rdm_ca, optimize = einsum_type)
        M_C_CVE_a_abb -= einsum('B,IK,LxBy,yx->IKLB', e_extern, np.identity(ncvs), t1_vaea, rdm_ca, optimize = einsum_type)
        M_C_CVE_a_abb += 1/2 * einsum('B,IK,LxyB,yx->IKLB', e_extern, np.identity(ncvs), t1_vaae, rdm_ca, optimize = einsum_type)
        M_C_CVE_a_abb += einsum('L,IK,LxBy,yx->IKLB', e_val, np.identity(ncvs), t1_vaea, rdm_ca, optimize = einsum_type)
        M_C_CVE_a_abb -= 1/2 * einsum('L,IK,LxyB,yx->IKLB', e_val, np.identity(ncvs), t1_vaae, rdm_ca, optimize = einsum_type)
        M_C_CVE_a_abb += einsum('xy,IK,LxBz,zy->IKLB', h_aa, np.identity(ncvs), t1_vaea, rdm_ca, optimize = einsum_type)
        M_C_CVE_a_abb -= 1/2 * einsum('xy,IK,LxzB,zy->IKLB', h_aa, np.identity(ncvs), t1_vaae, rdm_ca, optimize = einsum_type)
        M_C_CVE_a_abb -= einsum('xy,IK,LzBx,yz->IKLB', h_aa, np.identity(ncvs), t1_vaea, rdm_ca, optimize = einsum_type)
        M_C_CVE_a_abb += 1/2 * einsum('xy,IK,LzxB,yz->IKLB', h_aa, np.identity(ncvs), t1_vaae, rdm_ca, optimize = einsum_type)
        M_C_CVE_a_abb += einsum('IK,LxBy,xzwu,yzwu->IKLB', np.identity(ncvs), t1_vaea, v_aaaa, rdm_ccaa, optimize = einsum_type)
        M_C_CVE_a_abb -= einsum('IK,LxBy,yzwu,xzwu->IKLB', np.identity(ncvs), t1_vaea, v_aaaa, rdm_ccaa, optimize = einsum_type)
        M_C_CVE_a_abb -= 1/2 * einsum('IK,LxyB,xzwu,yzwu->IKLB', np.identity(ncvs), t1_vaae, v_aaaa, rdm_ccaa, optimize = einsum_type)
        M_C_CVE_a_abb += 1/2 * einsum('IK,LxyB,yzwu,xzwu->IKLB', np.identity(ncvs), t1_vaae, v_aaaa, rdm_ccaa, optimize = einsum_type)

        M_C_CVE_a_bab =- einsum('LKIB->IKLB', v_vxxe, optimize = einsum_type).copy()

        M_C_CVE_a_aaa = np.ascontiguousarray(M_C_CVE_a_abb + M_C_CVE_a_bab)

        ## Reshape tensors to matrix form
        M_C_CVE_a_aaa = M_C_CVE_a_aaa.reshape(ncvs, -1)
        M_C_CVE_a_abb = M_C_CVE_a_abb.reshape(ncvs, -1)
        M_C_CVE_a_bab = M_C_CVE_a_bab.reshape(ncvs, -1)

        ## Building C-CVE matrix
        dim_cve = ncvs * nval * nextern
        dim_c_cve = 3 * dim_cve

        m_c_cve_aaa_i = 0
        m_c_cve_aaa_f = m_c_cve_aaa_i + dim_cve
        m_c_cve_abb_i = m_c_cve_aaa_f
        m_c_cve_abb_f = m_c_cve_abb_i + dim_cve
        m_c_cve_bab_i = m_c_cve_abb_f
        m_c_cve_bab_f = m_c_cve_bab_i + dim_cve

        M_C_CVE = np.zeros((ncvs, dim_c_cve))
        M_C_CVE[:, m_c_cve_aaa_i:m_c_cve_aaa_f] = M_C_CVE_a_aaa.copy()
        M_C_CVE[:, m_c_cve_abb_i:m_c_cve_abb_f] = M_C_CVE_a_abb.copy()
        M_C_CVE[:, m_c_cve_bab_i:m_c_cve_bab_f] = M_C_CVE_a_bab.copy()

    print("Time for computing M(h0-h1) blocks:               %f sec\n" % (time.time() - start_time))
    sys.stdout.flush()

    shift = 100000.0
    M_C_CAA = shift
    M_C_CAE = shift
    M_C_ACE = shift
    M_C_CCA = shift

    nval = mr_adc.nval
    if nval > 0:
        M_C_CVA = shift
        M_C_VCA = shift

    if nval > 0:
        M_01 = (M_C_CAA, M_C_CCE, M_C_CVE, M_C_CAE, M_C_ACE, M_C_CCA, M_C_CVA, M_C_VCA)
    else:
        M_01 = (M_C_CAA, M_C_CCE, M_C_CAE, M_C_ACE, M_C_CCA)

    return M_01

def compute_preconditioner_cce(mr_adc, M_00):

    start_time = time.time()

    # Einsum definition from kernel
    einsum = mr_adc.interface.einsum
    einsum_type = mr_adc.interface.einsum_type

    if mr_adc.method in ("mr-adc(0)", "mr-adc(1)"):

        # Multiply by -1.0, since we are solving for -M C = -S C E
        return (-1.0 * np.diag(M_00))

    # Variables from kernel
    ncvs = mr_adc.ncvs
    nval = mr_adc.nval
    ncas = mr_adc.ncas
    nextern = mr_adc.nextern

    e_cvs = mr_adc.mo_energy.x
    if nval > 0:
        e_val = mr_adc.mo_energy.v
    e_extern = mr_adc.mo_energy.e

    # One-electron integrals
    h_aa = mr_adc.h1eff.aa

    # Two-electrons integrals
    v_aaaa = mr_adc.v2e.aaaa

    # Reduced Density Matrices
    rdm_ca = mr_adc.rdm.ca
    rdm_ccaa = mr_adc.rdm.ccaa

    # Overlap matrices
    S12_c_caa = mr_adc.S12.c_caa
    S12_cae = mr_adc.S12.cae
    S12_cca = mr_adc.S12.cca

    # Dimensions
    ho_s_c = mr_adc.h_orth.s_c
    ho_f_c = mr_adc.h_orth.f_c
    ho_s_c_caa = mr_adc.h_orth.s_c_caa
    ho_f_c_caa = mr_adc.h_orth.f_c_caa
    ho_s_cce_aaa = mr_adc.h_orth.s_cce_aaa
    ho_f_cce_aaa = mr_adc.h_orth.f_cce_aaa
    ho_s_cce_abb = mr_adc.h_orth.s_cce_abb
    ho_f_cce_abb = mr_adc.h_orth.f_cce_abb
    ho_s_cae = mr_adc.h_orth.s_cae
    ho_f_cae = mr_adc.h_orth.f_cae
    ho_s_ace = mr_adc.h_orth.s_ace
    ho_f_ace = mr_adc.h_orth.f_ace
    ho_s_cca = mr_adc.h_orth.s_cca
    ho_f_cca = mr_adc.h_orth.f_cca
    if nval > 0:
        ho_s_cve_aaa = mr_adc.h_orth.s_cve_aaa
        ho_f_cve_aaa = mr_adc.h_orth.f_cve_aaa
        ho_s_cve_abb = mr_adc.h_orth.s_cve_abb
        ho_f_cve_abb = mr_adc.h_orth.f_cve_abb
        ho_s_cve_bab = mr_adc.h_orth.s_cve_bab
        ho_f_cve_bab = mr_adc.h_orth.f_cve_bab

        ho_s_cva = mr_adc.h_orth.s_cva
        ho_f_cva = mr_adc.h_orth.f_cva
        ho_s_vca = mr_adc.h_orth.s_vca
        ho_f_vca = mr_adc.h_orth.f_vca

    s_c = mr_adc.h0.s_c
    f_c = mr_adc.h0.f_c

    cvs_tril_ind = np.tril_indices(ncvs, k=-1)
    # cas_ind = np.tril_indices(ncas, k=-1)

    # Build the preconditioner
    precond = np.zeros(mr_adc.h_orth.dim)

    # C-C debug
    precond[ho_s_c:ho_f_c] = np.diag(M_00[s_c:f_c, s_c:f_c]).copy()

    # CCE
    precond_cce =- einsum('A,AA,II,JJ->IJA', e_extern, np.identity(nextern), np.identity(ncvs), np.identity(ncvs), optimize = einsum_type)
    precond_cce += einsum('I,AA,II,JJ->IJA', e_cvs, np.identity(nextern), np.identity(ncvs), np.identity(ncvs), optimize = einsum_type)
    precond_cce += einsum('J,AA,II,JJ->IJA', e_cvs, np.identity(nextern), np.identity(ncvs), np.identity(ncvs), optimize = einsum_type)
    precond[ho_s_cce_aaa:ho_f_cce_aaa] = precond_cce[cvs_tril_ind[0], cvs_tril_ind[1]].reshape(-1).copy()
    precond[ho_s_cce_abb:ho_f_cce_abb] = precond_cce.reshape(-1).copy()

    if nval > 0:
        # CVE
        precond_cve =- einsum('A,AA,II,JJ->IJA', e_extern, np.identity(nextern), np.identity(ncvs), np.identity(nval), optimize = einsum_type)
        precond_cve += einsum('I,AA,II,JJ->IJA', e_cvs, np.identity(nextern), np.identity(ncvs), np.identity(nval), optimize = einsum_type)
        precond_cve += einsum('J,AA,II,JJ->IJA', e_val, np.identity(nextern), np.identity(ncvs), np.identity(nval), optimize = einsum_type)

        precond[ho_s_cve_aaa:ho_f_cve_aaa] = precond_cve.reshape(-1).copy()
        precond[ho_s_cve_abb:ho_f_cve_abb] = precond_cve.reshape(-1).copy()
        precond[ho_s_cve_bab:ho_f_cve_bab] = precond_cve.reshape(-1).copy()

    # Multiply by -1.0, since we are solving for -M C = -S C E
    precond *= (-1.0)

    print ("Time for computing preconditioner:                %f sec\n" % (time.time() - start_time))
    sys.stdout.flush()

    return precond

def apply_S_12_cce(mr_adc, X, transpose = False):

    # Einsum definition from kernel
    einsum = mr_adc.interface.einsum
    einsum_type = mr_adc.interface.einsum_type

    # Dimensions
    nextern = mr_adc.nextern
    ncvs = mr_adc.ncvs
    nval = mr_adc.nval

    ho_s_c = mr_adc.h_orth.s_c
    ho_f_c = mr_adc.h_orth.f_c
    ho_s_c_caa = mr_adc.h_orth.s_c_caa
    ho_f_c_caa = mr_adc.h_orth.f_c_caa
    ho_s_cce = mr_adc.h_orth.s_cce_aaa
    ho_f_cce = mr_adc.h_orth.f_cce_abb
    ho_s_cae = mr_adc.h_orth.s_cae
    ho_f_cae = mr_adc.h_orth.f_cae
    ho_s_ace = mr_adc.h_orth.s_ace
    ho_f_ace = mr_adc.h_orth.f_ace
    ho_s_cca = mr_adc.h_orth.s_cca
    ho_f_cca = mr_adc.h_orth.f_cca
    s_c = mr_adc.h0.s_c
    f_c = mr_adc.h0.f_c
    s_caa = mr_adc.h1.s_caa
    f_caa = mr_adc.h1.f_caa
    s_cce = mr_adc.h1.s_cce_aaa
    f_cce = mr_adc.h1.f_cce_abb
    s_cae = mr_adc.h1.s_cae
    f_cae = mr_adc.h1.f_cae
    s_ace = mr_adc.h1.s_ace
    f_ace = mr_adc.h1.f_ace
    s_cca = mr_adc.h1.s_cca
    f_cca = mr_adc.h1.f_cca

    if nval > 0:
        ho_s_cve = mr_adc.h_orth.s_cve_aaa
        ho_f_cve = mr_adc.h_orth.f_cve_bab

        ho_s_cva = mr_adc.h_orth.s_cva
        ho_f_cva = mr_adc.h_orth.f_cva
        ho_s_vca = mr_adc.h_orth.s_vca
        ho_f_vca = mr_adc.h_orth.f_vca

        s_cve = mr_adc.h1.s_cve_aaa
        f_cve = mr_adc.h1.f_cve_bab

        s_cva = mr_adc.h1.s_cva
        f_cva = mr_adc.h1.f_cva
        s_vca = mr_adc.h1.s_vca
        f_vca = mr_adc.h1.f_vca

    # Overlap matrices
    S12_c_caa = mr_adc.S12.c_caa
    S12_cae = mr_adc.S12.cae
    S12_cca = mr_adc.S12.cca

    Xt = None

    if transpose:
        if (X.shape[0] != (mr_adc.h0.dim + mr_adc.h1.dim)):
            raise Exception("Dimensions do not match when applying S_12 transpose")

        Xt = np.zeros(mr_adc.h_orth.dim)

        # C-C DEBUG
        Xt[ho_s_c:ho_f_c] = X[s_c:f_c].copy()

        # CCE
        Xt[ho_s_cce:ho_f_cce] = X[s_cce:f_cce].copy()

        if nval > 0:
            # CVE
            Xt[ho_s_cve:ho_f_cve] = X[s_cve:f_cve].copy()

    else:
        if (X.shape[0] != (mr_adc.h_orth.dim)):
            raise Exception("Dimensions do not match when applying S_12")

        Xt = np.zeros(mr_adc.h0.dim + mr_adc.h1.dim)

        # C-C DEBUG
        Xt[s_c:f_c] = X[ho_s_c:ho_f_c].copy()

        # CCE
        Xt[s_cce:f_cce] = X[ho_s_cce:ho_f_cce].copy()

        if nval > 0:
            # CVE
            Xt[s_cve:f_cve] = X[ho_s_cve:ho_f_cve].copy()

    return Xt

def compute_sigma_vector_cce(mr_adc, M_00, M_01, M_11, Xt):

    # Einsum definition from kernel
    einsum = mr_adc.interface.einsum
    einsum_type = mr_adc.interface.einsum_type

    # Variables from kernel
    ncvs = mr_adc.ncvs
    nval = mr_adc.nval
    ncas = mr_adc.ncas
    nextern = mr_adc.nextern

    e_cvs = mr_adc.mo_energy.x
    e_core = mr_adc.mo_energy.c
    if nval > 0:
        e_val = mr_adc.mo_energy.v
    e_extern = mr_adc.mo_energy.e

    # One-electron integrals
    h_aa = mr_adc.h1eff.aa

    # Two-electrons integrals
    v_aaaa = mr_adc.v2e.aaaa

    # Reduced Density Matrices
    rdm_ca = mr_adc.rdm.ca
    rdm_ccaa = mr_adc.rdm.ccaa

    # Dimensions
    s_c = mr_adc.h0.s_c
    f_c = mr_adc.h0.f_c
    s_caa = mr_adc.h1.s_caa
    f_caa = mr_adc.h1.f_caa
    s_cce = mr_adc.h1.s_cce_aaa
    f_cce = mr_adc.h1.f_cce_abb
    s_cce_aaa = mr_adc.h1.s_cce_aaa
    f_cce_aaa = mr_adc.h1.f_cce_aaa
    s_cce_abb = mr_adc.h1.s_cce_abb
    f_cce_abb = mr_adc.h1.f_cce_abb
    s_cae = mr_adc.h1.s_cae
    f_cae = mr_adc.h1.f_cae
    s_ace = mr_adc.h1.s_ace
    f_ace = mr_adc.h1.f_ace
    s_cca = mr_adc.h1.s_cca
    f_cca = mr_adc.h1.f_cca
    if nval > 0:
        s_cve = mr_adc.h1.s_cve_aaa
        f_cve = mr_adc.h1.f_cve_bab

        s_cve_aaa = mr_adc.h1.s_cve_aaa
        f_cve_aaa = mr_adc.h1.f_cve_aaa
        s_cve_abb = mr_adc.h1.s_cve_abb
        f_cve_abb = mr_adc.h1.f_cve_abb
        s_cve_bab = mr_adc.h1.s_cve_bab
        f_cve_bab = mr_adc.h1.f_cve_bab

        s_cva = mr_adc.h1.s_cva
        f_cva = mr_adc.h1.f_cva
        s_vca = mr_adc.h1.s_vca
        f_vca = mr_adc.h1.f_vca

    cvs_tril_ind = np.tril_indices(ncvs, k=-1)

    # (CASCI + C) -> (CASCI + C)
    sigma = np.zeros_like(Xt)

    # h0-h0 contributions
    sigma[:mr_adc.h0.dim] = np.dot(M_00, Xt[:mr_adc.h0.dim])

    # h0-h1 and h1-h0 contributions
    if nval > 0:
        M_C_CAA, M_C_CCE, M_C_CVE, M_C_CAE, M_C_ACE, M_C_CCA, M_C_CVA, M_C_VCA = M_01
    else:
        M_C_CAA, M_C_CCE, M_C_CAE, M_C_ACE, M_C_CCA = M_01

    # C <-> CCE
    sigma[s_c:f_c] += np.dot(M_C_CCE, Xt[s_cce:f_cce])
    sigma[s_cce:f_cce] += np.dot(M_C_CCE.T, Xt[s_c:f_c])

    # C <-> CVE
    if nval > 0:
        sigma[s_c:f_c] += np.dot(M_C_CVE, Xt[s_cve:f_cve])
        sigma[s_cve:f_cve] += np.dot(M_C_CVE.T, Xt[s_c:f_c])

    # h1-h1 contributions
    # CCE <- CCE
    X_cce = Xt[s_cce:f_cce].copy()

    dim_cce = ncvs * ncvs * nextern
    dim_tril_cce = ncvs * (ncvs - 1)  * nextern // 2

    sigma_aaa_i = 0
    sigma_aaa_f = sigma_aaa_i + dim_tril_cce
    sigma_abb_i = sigma_aaa_f
    sigma_abb_f = sigma_abb_i + dim_cce

    X_aaa = np.zeros((ncvs, ncvs, nextern))
    X_aaa[cvs_tril_ind[0], cvs_tril_ind[1]] =  X_cce[sigma_aaa_i:sigma_aaa_f].reshape(-1, nextern).copy()
    X_aaa[cvs_tril_ind[1], cvs_tril_ind[0]] =- X_cce[sigma_aaa_i:sigma_aaa_f].reshape(-1, nextern).copy()

    X_abb = X_cce[sigma_abb_i:sigma_abb_f].reshape(ncvs, ncvs, nextern).copy()
    X_bab =- X_abb.transpose(1,0,2)

    sigma_cce_aaa =- 1/2 * einsum('KLB,B->KLB', X_aaa, e_extern, optimize = einsum_type)
    sigma_cce_aaa += 1/2 * einsum('KLB,K->KLB', X_aaa, e_cvs, optimize = einsum_type)
    sigma_cce_aaa += 1/2 * einsum('KLB,L->KLB', X_aaa, e_cvs, optimize = einsum_type)
    sigma_cce_aaa += 1/2 * einsum('LKB,B->KLB', X_aaa, e_extern, optimize = einsum_type)
    sigma_cce_aaa -= 1/2 * einsum('LKB,K->KLB', X_aaa, e_cvs, optimize = einsum_type)
    sigma_cce_aaa -= 1/2 * einsum('LKB,L->KLB', X_aaa, e_cvs, optimize = einsum_type)

    sigma_cce_abb =- 1/2 * einsum('KLB,B->KLB', X_abb, e_extern, optimize = einsum_type)
    sigma_cce_abb += 1/2 * einsum('KLB,K->KLB', X_abb, e_cvs, optimize = einsum_type)
    sigma_cce_abb += 1/2 * einsum('KLB,L->KLB', X_abb, e_cvs, optimize = einsum_type)
    sigma_cce_abb += 1/2 * einsum('LKB,B->KLB', X_bab, e_extern, optimize = einsum_type)
    sigma_cce_abb -= 1/2 * einsum('LKB,K->KLB', X_bab, e_cvs, optimize = einsum_type)
    sigma_cce_abb -= 1/2 * einsum('LKB,L->KLB', X_bab, e_cvs, optimize = einsum_type)

    ## Building C-CCE matrix
    dim_tril_cce = ncvs * (ncvs - 1) * nextern // 2
    dim_cce = ncvs * ncvs * nextern
    dim_sigma_cce = dim_cce + dim_tril_cce

    sigma_cce_aaa_i = 0
    sigma_cce_aaa_f = sigma_cce_aaa_i + dim_tril_cce
    sigma_cce_abb_i = sigma_cce_aaa_f
    sigma_cce_abb_f = sigma_cce_abb_i + dim_cce

    sigma_cce = np.zeros((dim_sigma_cce))
    sigma_cce[sigma_cce_aaa_i:sigma_cce_aaa_f] = sigma_cce_aaa[cvs_tril_ind[0], cvs_tril_ind[1]].reshape(-1).copy()
    sigma_cce[sigma_cce_abb_i:sigma_cce_abb_f] = sigma_cce_abb.reshape(-1).copy()

    sigma[s_cce:f_cce] += sigma_cce.reshape(-1).copy()

    if nval > 0:
        # CVE <- CVE
        X_aaa = Xt[s_cve_aaa:f_cve_aaa].reshape(ncvs, nval, nextern).copy()
        X_abb = Xt[s_cve_abb:f_cve_abb].reshape(ncvs, nval, nextern).copy()
        X_bab = Xt[s_cve_bab:f_cve_bab].reshape(ncvs, nval, nextern).copy()

        sigma_cve_aaa =- einsum('KLB,B->KLB', X_aaa, e_extern, optimize = einsum_type)
        sigma_cve_aaa += einsum('KLB,K->KLB', X_aaa, e_cvs, optimize = einsum_type)
        sigma_cve_aaa += einsum('KLB,L->KLB', X_aaa, e_val, optimize = einsum_type)

        sigma_cve_abb =- einsum('KLB,B->KLB', X_abb, e_extern, optimize = einsum_type)
        sigma_cve_abb += einsum('KLB,K->KLB', X_abb, e_cvs, optimize = einsum_type)
        sigma_cve_abb += einsum('KLB,L->KLB', X_abb, e_val, optimize = einsum_type)

        sigma_cve_bab =- einsum('KLB,B->KLB', X_bab, e_extern, optimize = einsum_type)
        sigma_cve_bab += einsum('KLB,K->KLB', X_bab, e_cvs, optimize = einsum_type)
        sigma_cve_bab += einsum('KLB,L->KLB', X_bab, e_val, optimize = einsum_type)

        # ## Building C-CVE matrix
        dim_cve = ncvs * nval * nextern
        dim_sigma_cve = 3 * dim_cve

        sigma_cve_aaa_i = 0
        sigma_cve_aaa_f = sigma_cve_aaa_i + dim_cve
        sigma_cve_abb_i = sigma_cve_aaa_f
        sigma_cve_abb_f = sigma_cve_abb_i + dim_cve
        sigma_cve_bab_i = sigma_cve_abb_f
        sigma_cve_bab_f = sigma_cve_bab_i + dim_cve

        sigma_cve = np.zeros((dim_sigma_cve))
        sigma_cve[sigma_cve_aaa_i:sigma_cve_aaa_f] = sigma_cve_aaa.reshape(-1).copy()
        sigma_cve[sigma_cve_abb_i:sigma_cve_abb_f] = sigma_cve_abb.reshape(-1).copy()
        sigma_cve[sigma_cve_bab_i:sigma_cve_bab_f] = sigma_cve_bab.reshape(-1).copy()

        sigma[s_cve:f_cve] += sigma_cve.copy()

    return sigma

## CCA block (CCA + CVA + VCA): Diagonal + M01
def compute_excitation_manifolds_cca(mr_adc):

    # MR-ADC(0) and MR-ADC(1)
    mr_adc.h0.n_c = mr_adc.ncvs
    mr_adc.h0.dim = mr_adc.h0.n_c # Total dimension of h0

    mr_adc.h0.s_c = 0
    mr_adc.h0.f_c = mr_adc.h0.s_c + mr_adc.h0.n_c

    print("Dimension of h0 excitation manifold:                       %d" % mr_adc.h0.dim)

    # MR-ADC(2)
    mr_adc.h1.dim = 0
    mr_adc.h_orth.dim = mr_adc.h0.dim

    if mr_adc.method in ("mr-adc(2)", "mr-adc(2)-x"):
        mr_adc.h1.n_caa = 0
        mr_adc.h1.n_cce = 0
        mr_adc.h1.n_cae = 0
        mr_adc.h1.n_ace = 0
        mr_adc.h1.n_cca_aaa = mr_adc.ncvs * (mr_adc.ncvs - 1) * mr_adc.ncas // 2
        mr_adc.h1.n_cca_abb = mr_adc.ncvs * mr_adc.ncvs * mr_adc.ncas
        mr_adc.h1.n_cca = mr_adc.h1.n_cca_aaa + mr_adc.h1.n_cca_abb
        if mr_adc.nval > 0:
            mr_adc.h1.n_cve = 0
            mr_adc.h1.n_vce = 0
            mr_adc.h1.n_cva_aaa = mr_adc.ncas * mr_adc.ncvs * mr_adc.nval
            mr_adc.h1.n_cva_abb = mr_adc.ncas * mr_adc.ncvs * mr_adc.nval
            mr_adc.h1.n_cva_bab = mr_adc.ncas * mr_adc.ncvs * mr_adc.nval
            mr_adc.h1.n_cva = mr_adc.h1.n_cva_aaa + mr_adc.h1.n_cva_abb + mr_adc.h1.n_cva_bab
            mr_adc.h1.dim = (mr_adc.h1.n_caa + mr_adc.h1.n_cce + mr_adc.h1.n_cve + mr_adc.h1.n_vce +
                             mr_adc.h1.n_cae + mr_adc.h1.n_ace + mr_adc.h1.n_cca + mr_adc.h1.n_cva)
        else:
            mr_adc.h1.dim = mr_adc.h1.n_caa + mr_adc.h1.n_cce + mr_adc.h1.n_cae + mr_adc.h1.n_cae + mr_adc.h1.n_cca

        if mr_adc.nval > 0:
            mr_adc.h1.s_caa = mr_adc.h0.f_c
            mr_adc.h1.f_caa = mr_adc.h1.s_caa + mr_adc.h1.n_caa
            mr_adc.h1.s_cce = mr_adc.h1.f_caa
            mr_adc.h1.f_cce = mr_adc.h1.s_cce + mr_adc.h1.n_cce
            mr_adc.h1.s_cve = mr_adc.h1.f_cce
            mr_adc.h1.f_cve = mr_adc.h1.s_cve + mr_adc.h1.n_cve
            mr_adc.h1.s_vce = mr_adc.h1.f_cve
            mr_adc.h1.f_vce = mr_adc.h1.s_vce + mr_adc.h1.n_vce
            mr_adc.h1.s_cae = mr_adc.h1.f_vce
            mr_adc.h1.f_cae = mr_adc.h1.s_cae + mr_adc.h1.n_cae
            mr_adc.h1.s_ace = mr_adc.h1.f_cae
            mr_adc.h1.f_ace = mr_adc.h1.s_ace + mr_adc.h1.n_ace
            mr_adc.h1.s_cca_aaa = mr_adc.h1.f_ace
            mr_adc.h1.f_cca_aaa = mr_adc.h1.s_cca_aaa + mr_adc.h1.n_cca_aaa
            mr_adc.h1.s_cca_abb = mr_adc.h1.f_cca_aaa
            mr_adc.h1.f_cca_abb = mr_adc.h1.s_cca_abb + mr_adc.h1.n_cca_abb
            mr_adc.h1.s_cva_aaa = mr_adc.h1.f_cca_abb
            mr_adc.h1.f_cva_aaa = mr_adc.h1.s_cva_aaa + mr_adc.h1.n_cva_aaa
            mr_adc.h1.s_cva_abb = mr_adc.h1.f_cva_aaa
            mr_adc.h1.f_cva_abb = mr_adc.h1.s_cva_abb + mr_adc.h1.n_cva_abb
            mr_adc.h1.s_cva_bab = mr_adc.h1.f_cva_abb
            mr_adc.h1.f_cva_bab = mr_adc.h1.s_cva_bab + mr_adc.h1.n_cva_bab
        else:
            mr_adc.h1.s_caa = mr_adc.h0.f_c
            mr_adc.h1.f_caa = mr_adc.h1.s_caa + mr_adc.h1.n_caa
            mr_adc.h1.s_cce = mr_adc.h1.f_caa
            mr_adc.h1.f_cce = mr_adc.h1.s_cce + mr_adc.h1.n_cce
            mr_adc.h1.s_cae = mr_adc.h1.f_cce
            mr_adc.h1.f_cae = mr_adc.h1.s_cae + mr_adc.h1.n_cae
            mr_adc.h1.s_ace = mr_adc.h1.f_cae
            mr_adc.h1.f_ace = mr_adc.h1.s_ace + mr_adc.h1.n_ace
            mr_adc.h1.s_cca_aaa = mr_adc.h1.f_ace
            mr_adc.h1.f_cca_aaa = mr_adc.h1.s_cca_aaa + mr_adc.h1.n_cca_aaa
            mr_adc.h1.s_cca_abb = mr_adc.h1.f_cca_aaa
            mr_adc.h1.f_cca_abb = mr_adc.h1.s_cca_abb + mr_adc.h1.n_cca_abb

        print("Dimension of h1 excitation manifold:                       %d" % mr_adc.h1.dim)

        # Overlap for c - caa
        mr_adc.S12.c_caa = mr_adc_overlap.compute_S12_0p_projector(mr_adc)
        mr_adc.S12.cae = mr_adc_overlap.compute_S12_m1(mr_adc)
        mr_adc.S12.cca = mr_adc_overlap.compute_S12_p1(mr_adc)

        # Determine dimensions of orthogonalized excitation spaces
        mr_adc.h_orth.n_c = mr_adc.ncvs
        mr_adc.h_orth.n_c_caa = 0
        mr_adc.h_orth.n_cce = 0
        mr_adc.h_orth.n_cce = 0
        mr_adc.h_orth.n_cae = 0
        mr_adc.h_orth.n_ace = 0
        mr_adc.h_orth.n_cca_aaa = mr_adc.S12.cca.shape[1] * mr_adc.ncvs * (mr_adc.ncvs - 1) // 2
        mr_adc.h_orth.n_cca_abb = mr_adc.S12.cca.shape[1] * mr_adc.ncvs * mr_adc.ncvs
        mr_adc.h_orth.n_cca = mr_adc.h_orth.n_cca_aaa + mr_adc.h_orth.n_cca_abb
        if mr_adc.nval > 0:
            mr_adc.h_orth.n_cve = 0
            mr_adc.h_orth.n_vce = 0
            mr_adc.h_orth.n_cva_aaa = mr_adc.S12.cca.shape[1] * mr_adc.ncvs * mr_adc.nval
            mr_adc.h_orth.n_cva_abb = mr_adc.S12.cca.shape[1] * mr_adc.ncvs * mr_adc.nval
            mr_adc.h_orth.n_cva_bab = mr_adc.S12.cca.shape[1] * mr_adc.ncvs * mr_adc.nval
            mr_adc.h_orth.n_cva = mr_adc.h_orth.n_cva_aaa + mr_adc.h_orth.n_cva_abb + mr_adc.h_orth.n_cva_bab
            mr_adc.h_orth.dim = (mr_adc.h_orth.n_c + mr_adc.h_orth.n_c_caa + mr_adc.h_orth.n_cce + mr_adc.h_orth.n_cve + mr_adc.h_orth.n_vce +
                                 mr_adc.h_orth.n_cae + mr_adc.h_orth.n_ace + mr_adc.h_orth.n_cca + mr_adc.h_orth.n_cva)
        else:
            mr_adc.h_orth.dim = mr_adc.h_orth.n_c + mr_adc.h_orth.n_c_caa + mr_adc.h_orth.n_cce + mr_adc.h_orth.n_cae + mr_adc.h_orth.n_ace + mr_adc.h_orth.n_cca

        if mr_adc.nval > 0:
            mr_adc.h_orth.s_c = 0
            mr_adc.h_orth.f_c = mr_adc.h_orth.n_c
            mr_adc.h_orth.s_c_caa = mr_adc.h_orth.f_c
            mr_adc.h_orth.f_c_caa = mr_adc.h_orth.s_c_caa + mr_adc.h_orth.n_c_caa
            mr_adc.h_orth.s_cce = mr_adc.h_orth.f_c_caa
            mr_adc.h_orth.f_cce = mr_adc.h_orth.s_cce + mr_adc.h_orth.n_cce
            mr_adc.h_orth.s_cve = mr_adc.h_orth.f_cce
            mr_adc.h_orth.f_cve = mr_adc.h_orth.s_cve + mr_adc.h_orth.n_cve
            mr_adc.h_orth.s_vce = mr_adc.h_orth.f_cve
            mr_adc.h_orth.f_vce = mr_adc.h_orth.s_vce + mr_adc.h_orth.n_vce
            mr_adc.h_orth.s_cae = mr_adc.h_orth.f_vce
            mr_adc.h_orth.f_cae = mr_adc.h_orth.s_cae + mr_adc.h_orth.n_cae
            mr_adc.h_orth.s_ace = mr_adc.h_orth.f_cae
            mr_adc.h_orth.f_ace = mr_adc.h_orth.s_ace + mr_adc.h_orth.n_ace
            mr_adc.h_orth.s_cca_aaa = mr_adc.h_orth.f_ace
            mr_adc.h_orth.f_cca_aaa = mr_adc.h_orth.s_cca_aaa + mr_adc.h_orth.n_cca_aaa
            mr_adc.h_orth.s_cca_abb = mr_adc.h_orth.f_cca_aaa
            mr_adc.h_orth.f_cca_abb = mr_adc.h_orth.s_cca_abb + mr_adc.h_orth.n_cca_abb
            mr_adc.h_orth.s_cva_aaa = mr_adc.h_orth.f_cca_abb
            mr_adc.h_orth.f_cva_aaa = mr_adc.h_orth.s_cva_aaa + mr_adc.h_orth.n_cva_aaa
            mr_adc.h_orth.s_cva_abb = mr_adc.h_orth.f_cva_aaa
            mr_adc.h_orth.f_cva_abb = mr_adc.h_orth.s_cva_abb + mr_adc.h_orth.n_cva_abb
            mr_adc.h_orth.s_cva_bab = mr_adc.h_orth.f_cva_abb
            mr_adc.h_orth.f_cva_bab = mr_adc.h_orth.s_cva_bab + mr_adc.h_orth.n_cva_bab
        else:
            mr_adc.h_orth.s_c = 0
            mr_adc.h_orth.f_c = mr_adc.h_orth.n_c
            mr_adc.h_orth.s_cce = mr_adc.h_orth.f_c
            mr_adc.h_orth.f_cce = mr_adc.h_orth.s_cce + mr_adc.h_orth.n_cce
            mr_adc.h_orth.s_cae = mr_adc.h_orth.f_cce
            mr_adc.h_orth.f_cae = mr_adc.h_orth.s_cae + mr_adc.h_orth.n_cae
            mr_adc.h_orth.s_ace = mr_adc.h_orth.f_cae
            mr_adc.h_orth.f_ace = mr_adc.h_orth.s_ace + mr_adc.h_orth.n_ace
            mr_adc.h_orth.s_cca_aaa = mr_adc.h_orth.f_ace
            mr_adc.h_orth.f_cca_aaa = mr_adc.h_orth.s_cca_aaa + mr_adc.h_orth.n_cca_aaa
            mr_adc.h_orth.s_cca_abb = mr_adc.h_orth.f_cca_aaa
            mr_adc.h_orth.f_cca_abb = mr_adc.h_orth.s_cca_abb + mr_adc.h_orth.n_cca_abb

    print("Total dimension of the excitation manifold:                %d" % (mr_adc.h0.dim + mr_adc.h1.dim))
    print("Dimension of the orthogonalized excitation manifold:       %d\n" % (mr_adc.h_orth.dim))
    sys.stdout.flush()

    if (mr_adc.h_orth.dim < mr_adc.nroots):
        mr_adc.nroots = mr_adc.h_orth.dim

    return mr_adc

def compute_M_01_cca(mr_adc):

    start_time = time.time()

    print ("Computing M(h0-h1) blocks...")
    sys.stdout.flush()

    # Einsum definition from kernel
    einsum = mr_adc.interface.einsum
    einsum_type = mr_adc.interface.einsum_type

    # Dimensions
    ncore = mr_adc.ncore
    ncas = mr_adc.ncas
    nocc = mr_adc.nocc
    nextern = mr_adc.nextern

    ncvs = mr_adc.ncvs
    nval = mr_adc.nval

    # cvs_tril_ind = np.tril_indices(ncvs, k=-1)

    # MOs Energy
    e_cvs = mr_adc.mo_energy.x
    if nval > 0:
        e_val = mr_adc.mo_energy.v
    e_core = mr_adc.mo_energy.c
    e_extern = mr_adc.mo_energy.e

    # Amplitudes
    t1_ce = mr_adc.t1.ce
    t1_ca = mr_adc.t1.ca
    t1_ae = mr_adc.t1.ae
    t1_caea = mr_adc.t1.caea
    t1_caae = mr_adc.t1.caae
    t1_caaa = mr_adc.t1.caaa
    t1_aaea = mr_adc.t1.aaea

    t1_xa = mr_adc.t1.xa
    t1_xaaa = mr_adc.t1.xaaa

    t1_xe = mr_adc.t1.xe
    t1_xaea = mr_adc.t1.xaea
    t1_xaae = mr_adc.t1.xaae

    if nval > 0:
        t1_ve = mr_adc.t1.ve
        t1_vaea = mr_adc.t1.vaea
        t1_vaae = mr_adc.t1.vaae

        t1_va = mr_adc.t1.va
        t1_vaaa = mr_adc.t1.vaaa

    # One-electron integrals
    h_aa = mr_adc.h1eff.aa

    h_xe = mr_adc.h1eff.xe
    h_xa = mr_adc.h1eff.xa

    if nval > 0:
        h_va = mr_adc.h1eff.va
        h_ve = mr_adc.h1eff.ve

    # Two-electrons integrals
    v_aaaa = mr_adc.v2e.aaaa

    v_xaxa = mr_adc.v2e.xaxa
    v_xaax = mr_adc.v2e.xaax

    if nval > 0:
        v_vxxe = mr_adc.v2e.vxxe
        v_xvxe = mr_adc.v2e.xvxe
        v_vxxa = mr_adc.v2e.vxxa
        v_xvxa = mr_adc.v2e.xvxa

    v_xaea = mr_adc.v2e.xaea
    v_xaae = mr_adc.v2e.xaae
    v_xxxe = mr_adc.v2e.xxxe

    v_xaaa = mr_adc.v2e.xaaa
    v_xxxa = mr_adc.v2e.xxxa

    if nval > 0:
        v_vaea = mr_adc.v2e.vaea
        v_vaae = mr_adc.v2e.vaae

        v_vaaa = mr_adc.v2e.vaaa

    # Reduced Density Matrices
    rdm_ca = mr_adc.rdm.ca
    rdm_ccaa = mr_adc.rdm.ccaa
    rdm_cccaaa = mr_adc.rdm.cccaaa

    # Indices
    cvs_tril_ind = np.tril_indices(ncvs, k=-1)

    # C - CCA
    M_C_CCA_a_abb  = einsum('KLIY->IKLY', v_xxxa, optimize = einsum_type).copy()
    M_C_CCA_a_abb -= einsum('LY,IK->IKLY', h_xa, np.identity(ncvs), optimize = einsum_type)
    M_C_CCA_a_abb -= 1/2 * einsum('KLIx,xY->IKLY', v_xxxa, rdm_ca, optimize = einsum_type)
    M_C_CCA_a_abb += einsum('L,IK,LY->IKLY', e_cvs, np.identity(ncvs), t1_xa, optimize = einsum_type)
    M_C_CCA_a_abb += 1/2 * einsum('Lx,IK,xY->IKLY', h_xa, np.identity(ncvs), rdm_ca, optimize = einsum_type)
    M_C_CCA_a_abb -= einsum('Yx,IK,Lx->IKLY', h_aa, np.identity(ncvs), t1_xa, optimize = einsum_type)
    M_C_CCA_a_abb -= einsum('IK,LxYy,yx->IKLY', np.identity(ncvs), v_xaaa, rdm_ca, optimize = einsum_type)
    M_C_CCA_a_abb += 1/2 * einsum('IK,LxyY,yx->IKLY', np.identity(ncvs), v_xaaa, rdm_ca, optimize = einsum_type)
    M_C_CCA_a_abb += 1/2 * einsum('IK,Lxyz,Yxyz->IKLY', np.identity(ncvs), v_xaaa, rdm_ccaa, optimize = einsum_type)
    M_C_CCA_a_abb -= 1/2 * einsum('L,IK,Lx,xY->IKLY', e_cvs, np.identity(ncvs), t1_xa, rdm_ca, optimize = einsum_type)
    M_C_CCA_a_abb += einsum('L,IK,LxYy,yx->IKLY', e_cvs, np.identity(ncvs), t1_xaaa, rdm_ca, optimize = einsum_type)
    M_C_CCA_a_abb -= 1/2 * einsum('L,IK,LxyY,yx->IKLY', e_cvs, np.identity(ncvs), t1_xaaa, rdm_ca, optimize = einsum_type)
    M_C_CCA_a_abb -= 1/2 * einsum('L,IK,Lxyz,Yxyz->IKLY', e_cvs, np.identity(ncvs), t1_xaaa, rdm_ccaa, optimize = einsum_type)
    M_C_CCA_a_abb -= einsum('Yx,IK,Lyxz,zy->IKLY', h_aa, np.identity(ncvs), t1_xaaa, rdm_ca, optimize = einsum_type)
    M_C_CCA_a_abb += 1/2 * einsum('Yx,IK,Lyzx,zy->IKLY', h_aa, np.identity(ncvs), t1_xaaa, rdm_ca, optimize = einsum_type)
    M_C_CCA_a_abb += 1/2 * einsum('xy,IK,Lx,yY->IKLY', h_aa, np.identity(ncvs), t1_xa, rdm_ca, optimize = einsum_type)
    M_C_CCA_a_abb += einsum('xy,IK,LxYz,zy->IKLY', h_aa, np.identity(ncvs), t1_xaaa, rdm_ca, optimize = einsum_type)
    M_C_CCA_a_abb -= 1/2 * einsum('xy,IK,LxzY,zy->IKLY', h_aa, np.identity(ncvs), t1_xaaa, rdm_ca, optimize = einsum_type)
    M_C_CCA_a_abb -= 1/2 * einsum('xy,IK,Lxzw,Yyzw->IKLY', h_aa, np.identity(ncvs), t1_xaaa, rdm_ccaa, optimize = einsum_type)
    M_C_CCA_a_abb -= einsum('xy,IK,LzYx,yz->IKLY', h_aa, np.identity(ncvs), t1_xaaa, rdm_ca, optimize = einsum_type)
    M_C_CCA_a_abb += 1/2 * einsum('xy,IK,LzxY,yz->IKLY', h_aa, np.identity(ncvs), t1_xaaa, rdm_ca, optimize = einsum_type)
    M_C_CCA_a_abb += 1/2 * einsum('xy,IK,Lzxw,Yzyw->IKLY', h_aa, np.identity(ncvs), t1_xaaa, rdm_ccaa, optimize = einsum_type)
    M_C_CCA_a_abb += 1/2 * einsum('xy,IK,Lzwx,Yzwy->IKLY', h_aa, np.identity(ncvs), t1_xaaa, rdm_ccaa, optimize = einsum_type)
    M_C_CCA_a_abb -= einsum('IK,Lx,Yyxz,yz->IKLY', np.identity(ncvs), t1_xa, v_aaaa, rdm_ca, optimize = einsum_type)
    M_C_CCA_a_abb += 1/2 * einsum('IK,Lx,Yyzx,yz->IKLY', np.identity(ncvs), t1_xa, v_aaaa, rdm_ca, optimize = einsum_type)
    M_C_CCA_a_abb += 1/2 * einsum('IK,Lx,xyzw,Yyzw->IKLY', np.identity(ncvs), t1_xa, v_aaaa, rdm_ccaa, optimize = einsum_type)
    M_C_CCA_a_abb += einsum('IK,LxYy,xzwu,yzwu->IKLY', np.identity(ncvs), t1_xaaa, v_aaaa, rdm_ccaa, optimize = einsum_type)
    M_C_CCA_a_abb -= einsum('IK,LxYy,yzwu,xzwu->IKLY', np.identity(ncvs), t1_xaaa, v_aaaa, rdm_ccaa, optimize = einsum_type)
    M_C_CCA_a_abb -= 1/2 * einsum('IK,LxyY,xzwu,yzwu->IKLY', np.identity(ncvs), t1_xaaa, v_aaaa, rdm_ccaa, optimize = einsum_type)
    M_C_CCA_a_abb += 1/2 * einsum('IK,LxyY,yzwu,xzwu->IKLY', np.identity(ncvs), t1_xaaa, v_aaaa, rdm_ccaa, optimize = einsum_type)
    M_C_CCA_a_abb -= 1/2 * einsum('IK,Lxyz,Yxwu,yzwu->IKLY', np.identity(ncvs), t1_xaaa, v_aaaa, rdm_ccaa, optimize = einsum_type)
    M_C_CCA_a_abb -= einsum('IK,Lxyz,Ywyz,wx->IKLY', np.identity(ncvs), t1_xaaa, v_aaaa, rdm_ca, optimize = einsum_type)
    M_C_CCA_a_abb -= einsum('IK,Lxyz,Ywyu,xuzw->IKLY', np.identity(ncvs), t1_xaaa, v_aaaa, rdm_ccaa, optimize = einsum_type)
    M_C_CCA_a_abb += 1/2 * einsum('IK,Lxyz,Ywzy,wx->IKLY', np.identity(ncvs), t1_xaaa, v_aaaa, rdm_ca, optimize = einsum_type)
    M_C_CCA_a_abb += 1/2 * einsum('IK,Lxyz,Ywzu,xuyw->IKLY', np.identity(ncvs), t1_xaaa, v_aaaa, rdm_ccaa, optimize = einsum_type)
    M_C_CCA_a_abb += 1/2 * einsum('IK,Lxyz,Ywuy,xuzw->IKLY', np.identity(ncvs), t1_xaaa, v_aaaa, rdm_ccaa, optimize = einsum_type)
    M_C_CCA_a_abb += 1/2 * einsum('IK,Lxyz,Ywuz,xuwy->IKLY', np.identity(ncvs), t1_xaaa, v_aaaa, rdm_ccaa, optimize = einsum_type)
    M_C_CCA_a_abb -= 5/12 * einsum('IK,Lxyz,xwuv,yzwYuv->IKLY', np.identity(ncvs), t1_xaaa, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    M_C_CCA_a_abb += 1/12 * einsum('IK,Lxyz,xwuv,yzwYvu->IKLY', np.identity(ncvs), t1_xaaa, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    M_C_CCA_a_abb += 1/12 * einsum('IK,Lxyz,xwuv,yzwuYv->IKLY', np.identity(ncvs), t1_xaaa, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    M_C_CCA_a_abb += 1/12 * einsum('IK,Lxyz,xwuv,yzwuvY->IKLY', np.identity(ncvs), t1_xaaa, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    M_C_CCA_a_abb += 1/12 * einsum('IK,Lxyz,xwuv,yzwvYu->IKLY', np.identity(ncvs), t1_xaaa, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    M_C_CCA_a_abb += 1/12 * einsum('IK,Lxyz,xwuv,yzwvuY->IKLY', np.identity(ncvs), t1_xaaa, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    M_C_CCA_a_abb += 1/2 * einsum('IK,Lxyz,yzwu,Yxwu->IKLY', np.identity(ncvs), t1_xaaa, v_aaaa, rdm_ccaa, optimize = einsum_type)
    M_C_CCA_a_abb -= 1/12 * einsum('IK,Lxyz,ywuv,zuvYxw->IKLY', np.identity(ncvs), t1_xaaa, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    M_C_CCA_a_abb -= 1/12 * einsum('IK,Lxyz,ywuv,zuvYwx->IKLY', np.identity(ncvs), t1_xaaa, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    M_C_CCA_a_abb += 5/12 * einsum('IK,Lxyz,ywuv,zuvxYw->IKLY', np.identity(ncvs), t1_xaaa, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    M_C_CCA_a_abb -= 1/12 * einsum('IK,Lxyz,ywuv,zuvxwY->IKLY', np.identity(ncvs), t1_xaaa, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    M_C_CCA_a_abb -= 1/12 * einsum('IK,Lxyz,ywuv,zuvwYx->IKLY', np.identity(ncvs), t1_xaaa, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    M_C_CCA_a_abb -= 1/12 * einsum('IK,Lxyz,ywuv,zuvwxY->IKLY', np.identity(ncvs), t1_xaaa, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    M_C_CCA_a_abb += 5/12 * einsum('IK,Lxyz,zwuv,yuvYxw->IKLY', np.identity(ncvs), t1_xaaa, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    M_C_CCA_a_abb -= 1/12 * einsum('IK,Lxyz,zwuv,yuvYwx->IKLY', np.identity(ncvs), t1_xaaa, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    M_C_CCA_a_abb -= 1/12 * einsum('IK,Lxyz,zwuv,yuvxYw->IKLY', np.identity(ncvs), t1_xaaa, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    M_C_CCA_a_abb -= 1/12 * einsum('IK,Lxyz,zwuv,yuvxwY->IKLY', np.identity(ncvs), t1_xaaa, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    M_C_CCA_a_abb -= 1/12 * einsum('IK,Lxyz,zwuv,yuvwYx->IKLY', np.identity(ncvs), t1_xaaa, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    M_C_CCA_a_abb -= 1/12 * einsum('IK,Lxyz,zwuv,yuvwxY->IKLY', np.identity(ncvs), t1_xaaa, v_aaaa, rdm_cccaaa, optimize = einsum_type)

    M_C_CCA_a_aaa = M_C_CCA_a_abb - M_C_CCA_a_abb.transpose(0,2,1,3)

    ## Reshape tensors to matrix form
    M_C_CCA_a_aaa = M_C_CCA_a_aaa[:, cvs_tril_ind[0], cvs_tril_ind[1]]

    M_C_CCA_a_aaa = M_C_CCA_a_aaa.reshape(ncvs, -1)
    M_C_CCA_a_abb = M_C_CCA_a_abb.reshape(ncvs, -1)

    ## Building C-CCA matrix
    dim_cca = ncvs * ncvs * ncas
    dim_tril_cca = ncvs * (ncvs - 1) * ncas // 2
    dim_c_cca = dim_tril_cca + dim_cca

    m_c_cca_aaa_i = 0
    m_c_cca_aaa_f = m_c_cca_aaa_i + dim_tril_cca
    m_c_cca_abb_i = m_c_cca_aaa_f
    m_c_cca_abb_f = m_c_cca_abb_i + dim_cca

    M_C_CCA = np.zeros((ncvs, dim_c_cca))
    M_C_CCA[:, m_c_cca_aaa_i:m_c_cca_aaa_f] = M_C_CCA_a_aaa.copy()
    M_C_CCA[:, m_c_cca_abb_i:m_c_cca_abb_f] = M_C_CCA_a_abb.copy()

    if nval > 0:
        M_C_CVA_a_abb  = einsum('KLIY->IKLY', v_xvxa, optimize = einsum_type).copy()
        M_C_CVA_a_abb -= einsum('LY,IK->IKLY', h_va, np.identity(ncvs), optimize = einsum_type)
        M_C_CVA_a_abb -= 1/2 * einsum('KLIx,xY->IKLY', v_xvxa, rdm_ca, optimize = einsum_type)
        M_C_CVA_a_abb += einsum('L,IK,LY->IKLY', e_val, np.identity(ncvs), t1_va, optimize = einsum_type)
        M_C_CVA_a_abb += 1/2 * einsum('Lx,IK,xY->IKLY', h_va, np.identity(ncvs), rdm_ca, optimize = einsum_type)
        M_C_CVA_a_abb -= einsum('Yx,IK,Lx->IKLY', h_aa, np.identity(ncvs), t1_va, optimize = einsum_type)
        M_C_CVA_a_abb -= einsum('IK,LxYy,yx->IKLY', np.identity(ncvs), v_vaaa, rdm_ca, optimize = einsum_type)
        M_C_CVA_a_abb += 1/2 * einsum('IK,LxyY,yx->IKLY', np.identity(ncvs), v_vaaa, rdm_ca, optimize = einsum_type)
        M_C_CVA_a_abb += 1/2 * einsum('IK,Lxyz,Yxyz->IKLY', np.identity(ncvs), v_vaaa, rdm_ccaa, optimize = einsum_type)
        M_C_CVA_a_abb -= 1/2 * einsum('L,IK,Lx,xY->IKLY', e_val, np.identity(ncvs), t1_va, rdm_ca, optimize = einsum_type)
        M_C_CVA_a_abb += einsum('L,IK,LxYy,yx->IKLY', e_val, np.identity(ncvs), t1_vaaa, rdm_ca, optimize = einsum_type)
        M_C_CVA_a_abb -= 1/2 * einsum('L,IK,LxyY,yx->IKLY', e_val, np.identity(ncvs), t1_vaaa, rdm_ca, optimize = einsum_type)
        M_C_CVA_a_abb -= 1/2 * einsum('L,IK,Lxyz,Yxyz->IKLY', e_val, np.identity(ncvs), t1_vaaa, rdm_ccaa, optimize = einsum_type)
        M_C_CVA_a_abb -= einsum('Yx,IK,Lyxz,zy->IKLY', h_aa, np.identity(ncvs), t1_vaaa, rdm_ca, optimize = einsum_type)
        M_C_CVA_a_abb += 1/2 * einsum('Yx,IK,Lyzx,zy->IKLY', h_aa, np.identity(ncvs), t1_vaaa, rdm_ca, optimize = einsum_type)
        M_C_CVA_a_abb += 1/2 * einsum('xy,IK,Lx,yY->IKLY', h_aa, np.identity(ncvs), t1_va, rdm_ca, optimize = einsum_type)
        M_C_CVA_a_abb += einsum('xy,IK,LxYz,zy->IKLY', h_aa, np.identity(ncvs), t1_vaaa, rdm_ca, optimize = einsum_type)
        M_C_CVA_a_abb -= 1/2 * einsum('xy,IK,LxzY,zy->IKLY', h_aa, np.identity(ncvs), t1_vaaa, rdm_ca, optimize = einsum_type)
        M_C_CVA_a_abb -= 1/2 * einsum('xy,IK,Lxzw,Yyzw->IKLY', h_aa, np.identity(ncvs), t1_vaaa, rdm_ccaa, optimize = einsum_type)
        M_C_CVA_a_abb -= einsum('xy,IK,LzYx,yz->IKLY', h_aa, np.identity(ncvs), t1_vaaa, rdm_ca, optimize = einsum_type)
        M_C_CVA_a_abb += 1/2 * einsum('xy,IK,LzxY,yz->IKLY', h_aa, np.identity(ncvs), t1_vaaa, rdm_ca, optimize = einsum_type)
        M_C_CVA_a_abb += 1/2 * einsum('xy,IK,Lzxw,Yzyw->IKLY', h_aa, np.identity(ncvs), t1_vaaa, rdm_ccaa, optimize = einsum_type)
        M_C_CVA_a_abb += 1/2 * einsum('xy,IK,Lzwx,Yzwy->IKLY', h_aa, np.identity(ncvs), t1_vaaa, rdm_ccaa, optimize = einsum_type)
        M_C_CVA_a_abb -= einsum('IK,Lx,Yyxz,yz->IKLY', np.identity(ncvs), t1_va, v_aaaa, rdm_ca, optimize = einsum_type)
        M_C_CVA_a_abb += 1/2 * einsum('IK,Lx,Yyzx,yz->IKLY', np.identity(ncvs), t1_va, v_aaaa, rdm_ca, optimize = einsum_type)
        M_C_CVA_a_abb += 1/2 * einsum('IK,Lx,xyzw,Yyzw->IKLY', np.identity(ncvs), t1_va, v_aaaa, rdm_ccaa, optimize = einsum_type)
        M_C_CVA_a_abb += einsum('IK,LxYy,xzwu,yzwu->IKLY', np.identity(ncvs), t1_vaaa, v_aaaa, rdm_ccaa, optimize = einsum_type)
        M_C_CVA_a_abb -= einsum('IK,LxYy,yzwu,xzwu->IKLY', np.identity(ncvs), t1_vaaa, v_aaaa, rdm_ccaa, optimize = einsum_type)
        M_C_CVA_a_abb -= 1/2 * einsum('IK,LxyY,xzwu,yzwu->IKLY', np.identity(ncvs), t1_vaaa, v_aaaa, rdm_ccaa, optimize = einsum_type)
        M_C_CVA_a_abb += 1/2 * einsum('IK,LxyY,yzwu,xzwu->IKLY', np.identity(ncvs), t1_vaaa, v_aaaa, rdm_ccaa, optimize = einsum_type)
        M_C_CVA_a_abb -= 1/2 * einsum('IK,Lxyz,Yxwu,yzwu->IKLY', np.identity(ncvs), t1_vaaa, v_aaaa, rdm_ccaa, optimize = einsum_type)
        M_C_CVA_a_abb -= einsum('IK,Lxyz,Ywyz,wx->IKLY', np.identity(ncvs), t1_vaaa, v_aaaa, rdm_ca, optimize = einsum_type)
        M_C_CVA_a_abb -= einsum('IK,Lxyz,Ywyu,xuzw->IKLY', np.identity(ncvs), t1_vaaa, v_aaaa, rdm_ccaa, optimize = einsum_type)
        M_C_CVA_a_abb += 1/2 * einsum('IK,Lxyz,Ywzy,wx->IKLY', np.identity(ncvs), t1_vaaa, v_aaaa, rdm_ca, optimize = einsum_type)
        M_C_CVA_a_abb += 1/2 * einsum('IK,Lxyz,Ywzu,xuyw->IKLY', np.identity(ncvs), t1_vaaa, v_aaaa, rdm_ccaa, optimize = einsum_type)
        M_C_CVA_a_abb += 1/2 * einsum('IK,Lxyz,Ywuy,xuzw->IKLY', np.identity(ncvs), t1_vaaa, v_aaaa, rdm_ccaa, optimize = einsum_type)
        M_C_CVA_a_abb += 1/2 * einsum('IK,Lxyz,Ywuz,xuwy->IKLY', np.identity(ncvs), t1_vaaa, v_aaaa, rdm_ccaa, optimize = einsum_type)
        M_C_CVA_a_abb -= 5/12 * einsum('IK,Lxyz,xwuv,yzwYuv->IKLY', np.identity(ncvs), t1_vaaa, v_aaaa, rdm_cccaaa, optimize = einsum_type)
        M_C_CVA_a_abb += 1/12 * einsum('IK,Lxyz,xwuv,yzwYvu->IKLY', np.identity(ncvs), t1_vaaa, v_aaaa, rdm_cccaaa, optimize = einsum_type)
        M_C_CVA_a_abb += 1/12 * einsum('IK,Lxyz,xwuv,yzwuYv->IKLY', np.identity(ncvs), t1_vaaa, v_aaaa, rdm_cccaaa, optimize = einsum_type)
        M_C_CVA_a_abb += 1/12 * einsum('IK,Lxyz,xwuv,yzwuvY->IKLY', np.identity(ncvs), t1_vaaa, v_aaaa, rdm_cccaaa, optimize = einsum_type)
        M_C_CVA_a_abb += 1/12 * einsum('IK,Lxyz,xwuv,yzwvYu->IKLY', np.identity(ncvs), t1_vaaa, v_aaaa, rdm_cccaaa, optimize = einsum_type)
        M_C_CVA_a_abb += 1/12 * einsum('IK,Lxyz,xwuv,yzwvuY->IKLY', np.identity(ncvs), t1_vaaa, v_aaaa, rdm_cccaaa, optimize = einsum_type)
        M_C_CVA_a_abb += 1/2 * einsum('IK,Lxyz,yzwu,Yxwu->IKLY', np.identity(ncvs), t1_vaaa, v_aaaa, rdm_ccaa, optimize = einsum_type)
        M_C_CVA_a_abb -= 1/12 * einsum('IK,Lxyz,ywuv,zuvYxw->IKLY', np.identity(ncvs), t1_vaaa, v_aaaa, rdm_cccaaa, optimize = einsum_type)
        M_C_CVA_a_abb -= 1/12 * einsum('IK,Lxyz,ywuv,zuvYwx->IKLY', np.identity(ncvs), t1_vaaa, v_aaaa, rdm_cccaaa, optimize = einsum_type)
        M_C_CVA_a_abb += 5/12 * einsum('IK,Lxyz,ywuv,zuvxYw->IKLY', np.identity(ncvs), t1_vaaa, v_aaaa, rdm_cccaaa, optimize = einsum_type)
        M_C_CVA_a_abb -= 1/12 * einsum('IK,Lxyz,ywuv,zuvxwY->IKLY', np.identity(ncvs), t1_vaaa, v_aaaa, rdm_cccaaa, optimize = einsum_type)
        M_C_CVA_a_abb -= 1/12 * einsum('IK,Lxyz,ywuv,zuvwYx->IKLY', np.identity(ncvs), t1_vaaa, v_aaaa, rdm_cccaaa, optimize = einsum_type)
        M_C_CVA_a_abb -= 1/12 * einsum('IK,Lxyz,ywuv,zuvwxY->IKLY', np.identity(ncvs), t1_vaaa, v_aaaa, rdm_cccaaa, optimize = einsum_type)
        M_C_CVA_a_abb += 5/12 * einsum('IK,Lxyz,zwuv,yuvYxw->IKLY', np.identity(ncvs), t1_vaaa, v_aaaa, rdm_cccaaa, optimize = einsum_type)
        M_C_CVA_a_abb -= 1/12 * einsum('IK,Lxyz,zwuv,yuvYwx->IKLY', np.identity(ncvs), t1_vaaa, v_aaaa, rdm_cccaaa, optimize = einsum_type)
        M_C_CVA_a_abb -= 1/12 * einsum('IK,Lxyz,zwuv,yuvxYw->IKLY', np.identity(ncvs), t1_vaaa, v_aaaa, rdm_cccaaa, optimize = einsum_type)
        M_C_CVA_a_abb -= 1/12 * einsum('IK,Lxyz,zwuv,yuvxwY->IKLY', np.identity(ncvs), t1_vaaa, v_aaaa, rdm_cccaaa, optimize = einsum_type)
        M_C_CVA_a_abb -= 1/12 * einsum('IK,Lxyz,zwuv,yuvwYx->IKLY', np.identity(ncvs), t1_vaaa, v_aaaa, rdm_cccaaa, optimize = einsum_type)
        M_C_CVA_a_abb -= 1/12 * einsum('IK,Lxyz,zwuv,yuvwxY->IKLY', np.identity(ncvs), t1_vaaa, v_aaaa, rdm_cccaaa, optimize = einsum_type)

        M_C_CVA_a_bab =- einsum('LKIY->IKLY', v_vxxa, optimize = einsum_type).copy()
        M_C_CVA_a_bab += 1/2 * einsum('LKIx,xY->IKLY', v_vxxa, rdm_ca, optimize = einsum_type)

        M_C_CVA_a_aaa = M_C_CVA_a_abb + M_C_CVA_a_bab

        M_C_CVA_a_aaa = M_C_CVA_a_aaa.reshape(ncvs, -1)
        M_C_CVA_a_abb = M_C_CVA_a_abb.reshape(ncvs, -1)
        M_C_CVA_a_bab = M_C_CVA_a_bab.reshape(ncvs, -1)

        ## Building C-CVA matrix
        dim_cva = ncvs * nval * ncas
        dim_c_cva = 3 * dim_cva

        m_c_cva_aaa_i = 0
        m_c_cva_aaa_f = m_c_cva_aaa_i + dim_cva
        m_c_cva_abb_i = m_c_cva_aaa_f
        m_c_cva_abb_f = m_c_cva_abb_i + dim_cva
        m_c_cva_bab_i = m_c_cva_abb_f
        m_c_cva_bab_f = m_c_cva_bab_i + dim_cva

        M_C_CVA = np.zeros((ncvs, dim_c_cva))
        M_C_CVA[:, m_c_cva_aaa_i:m_c_cva_aaa_f] = M_C_CVA_a_aaa.copy()
        M_C_CVA[:, m_c_cva_abb_i:m_c_cva_abb_f] = M_C_CVA_a_abb.copy()
        M_C_CVA[:, m_c_cva_bab_i:m_c_cva_bab_f] = M_C_CVA_a_bab.copy()

    print("Time for computing M(h0-h1) blocks:               %f sec\n" % (time.time() - start_time))
    sys.stdout.flush()

    shift = 100000.0
    M_C_CAA = shift
    M_C_CAE = shift
    M_C_ACE = shift
    M_C_CCE = shift

    nval = mr_adc.nval
    if nval > 0:
        M_C_CVE = shift
        M_C_VCE = shift

    if nval > 0:
        M_01 = (M_C_CAA, M_C_CCE, M_C_CVE, M_C_CAE, M_C_ACE, M_C_CCA, M_C_CVA)
    else:
        M_01 = (M_C_CAA, M_C_CCE, M_C_CAE, M_C_ACE, M_C_CCA)

    return M_01

def compute_preconditioner_cca(mr_adc, M_00):

    start_time = time.time()

    # Einsum definition from kernel
    einsum = mr_adc.interface.einsum
    einsum_type = mr_adc.interface.einsum_type

    if mr_adc.method in ("mr-adc(0)", "mr-adc(1)"):

        # Multiply by -1.0, since we are solving for -M C = -S C E
        return (-1.0 * np.diag(M_00))

    # Variables from kernel
    ncvs = mr_adc.ncvs
    nval = mr_adc.nval
    ncas = mr_adc.ncas
    nextern = mr_adc.nextern

    e_cvs = mr_adc.mo_energy.x
    if nval > 0:
        e_val = mr_adc.mo_energy.v
    e_extern = mr_adc.mo_energy.e

    # One-electron integrals
    h_aa = mr_adc.h1eff.aa

    # Two-electrons integrals
    v_aaaa = mr_adc.v2e.aaaa

    # Reduced Density Matrices
    rdm_ca = mr_adc.rdm.ca
    rdm_ccaa = mr_adc.rdm.ccaa

    # Overlap matrices
    S12_c_caa = mr_adc.S12.c_caa
    S12_cae = mr_adc.S12.cae
    S12_cca = mr_adc.S12.cca

    # Dimensions
    ho_s_c = mr_adc.h_orth.s_c
    ho_f_c = mr_adc.h_orth.f_c
    ho_s_cce = mr_adc.h_orth.s_cce
    ho_f_cce = mr_adc.h_orth.f_cce
    ho_s_cae = mr_adc.h_orth.s_cae
    ho_f_cae = mr_adc.h_orth.f_cae
    ho_s_ace = mr_adc.h_orth.s_ace
    ho_f_ace = mr_adc.h_orth.f_ace
    ho_s_cca_aaa = mr_adc.h_orth.s_cca_aaa
    ho_f_cca_aaa = mr_adc.h_orth.f_cca_aaa
    ho_s_cca_abb = mr_adc.h_orth.s_cca_abb
    ho_f_cca_abb = mr_adc.h_orth.f_cca_abb
    if nval > 0:
        ho_s_cve = mr_adc.h_orth.s_cve
        ho_f_cve = mr_adc.h_orth.f_cve
        ho_s_vce = mr_adc.h_orth.s_vce
        ho_f_vce = mr_adc.h_orth.f_vce

        ho_s_cva = mr_adc.h_orth.s_cva_aaa
        ho_f_cva = mr_adc.h_orth.f_cva_bab
        ho_s_cva_aaa = mr_adc.h_orth.s_cva_aaa
        ho_f_cva_aaa = mr_adc.h_orth.f_cva_aaa
        ho_s_cva_abb = mr_adc.h_orth.s_cva_abb
        ho_f_cva_abb = mr_adc.h_orth.f_cva_abb
        ho_s_cva_bab = mr_adc.h_orth.s_cva_bab
        ho_f_cva_bab = mr_adc.h_orth.f_cva_bab

    s_c = mr_adc.h0.s_c
    f_c = mr_adc.h0.f_c

    cvs_tril_ind = np.tril_indices(ncvs, k=-1)
    # cas_ind = np.tril_indices(ncas, k=-1)

    # Build the preconditioner
    precond = np.zeros(mr_adc.h_orth.dim)

    # C-C debug
    precond[ho_s_c:ho_f_c] = np.diag(M_00[s_c:f_c, s_c:f_c]).copy()

    # CCA
    precond_cca =- einsum('XY,II,JJ->IJXY', h_aa, np.identity(ncvs), np.identity(ncvs), optimize = einsum_type)
    precond_cca += einsum('I,II,JJ,XY->IJXY', e_cvs, np.identity(ncvs), np.identity(ncvs), np.identity(ncas), optimize = einsum_type)
    precond_cca += einsum('J,II,JJ,XY->IJXY', e_cvs, np.identity(ncvs), np.identity(ncvs), np.identity(ncas), optimize = einsum_type)
    precond_cca -= 1/2 * einsum('I,II,JJ,YX->IJXY', e_cvs, np.identity(ncvs), np.identity(ncvs), rdm_ca, optimize = einsum_type)
    precond_cca -= 1/2 * einsum('J,II,JJ,YX->IJXY', e_cvs, np.identity(ncvs), np.identity(ncvs), rdm_ca, optimize = einsum_type)
    precond_cca += 1/2 * einsum('Xx,II,JJ,Yx->IJXY', h_aa, np.identity(ncvs), np.identity(ncvs), rdm_ca, optimize = einsum_type)
    precond_cca -= einsum('XxYy,II,JJ,xy->IJXY', v_aaaa, np.identity(ncvs), np.identity(ncvs), rdm_ca, optimize = einsum_type)
    precond_cca += 1/2 * einsum('XxyY,II,JJ,xy->IJXY', v_aaaa, np.identity(ncvs), np.identity(ncvs), rdm_ca, optimize = einsum_type)
    precond_cca += 1/2 * einsum('Xxyz,II,JJ,Yxyz->IJXY', v_aaaa, np.identity(ncvs), np.identity(ncvs), rdm_ccaa, optimize = einsum_type)

    precond_cca = einsum("IJXY,XP,YP->IJP", precond_cca, S12_cca, S12_cca, optimize = einsum_type)
    precond[ho_s_cca_aaa:ho_f_cca_aaa] = precond_cca[cvs_tril_ind[0], cvs_tril_ind[1]].reshape(-1).copy()
    precond[ho_s_cca_abb:ho_f_cca_abb] = precond_cca.reshape(-1).copy()

    if nval > 0:
        # CVA
        precond_cva =- einsum('XY,II,JJ->IJXY', h_aa, np.identity(ncvs), np.identity(nval), optimize = einsum_type)
        precond_cva += einsum('I,II,JJ,XY->IJXY', e_cvs, np.identity(ncvs), np.identity(nval), np.identity(ncas), optimize = einsum_type)
        precond_cva += einsum('J,II,JJ,XY->IJXY', e_val, np.identity(ncvs), np.identity(nval), np.identity(ncas), optimize = einsum_type)
        precond_cva -= 1/2 * einsum('I,II,JJ,YX->IJXY', e_cvs, np.identity(ncvs), np.identity(nval), rdm_ca, optimize = einsum_type)
        precond_cva -= 1/2 * einsum('J,II,JJ,YX->IJXY', e_val, np.identity(ncvs), np.identity(nval), rdm_ca, optimize = einsum_type)
        precond_cva += 1/2 * einsum('Xx,II,JJ,Yx->IJXY', h_aa, np.identity(ncvs), np.identity(nval), rdm_ca, optimize = einsum_type)
        precond_cva -= einsum('XxYy,II,JJ,xy->IJXY', v_aaaa, np.identity(ncvs), np.identity(nval), rdm_ca, optimize = einsum_type)
        precond_cva += 1/2 * einsum('XxyY,II,JJ,xy->IJXY', v_aaaa, np.identity(ncvs), np.identity(nval), rdm_ca, optimize = einsum_type)
        precond_cva += 1/2 * einsum('Xxyz,II,JJ,Yxyz->IJXY', v_aaaa, np.identity(ncvs), np.identity(nval), rdm_ccaa, optimize = einsum_type)

        precond_cva = einsum("IJXY,XP,YP->IJP", precond_cva, S12_cca, S12_cca, optimize = einsum_type)
        precond[ho_s_cva_aaa:ho_f_cva_aaa] = precond_cva.reshape(-1).copy()
        precond[ho_s_cva_abb:ho_f_cva_abb] = precond_cva.reshape(-1).copy()
        precond[ho_s_cva_bab:ho_f_cva_bab] = precond_cva.reshape(-1).copy()

    # Multiply by -1.0, since we are solving for -M C = -S C E
    precond *= (-1.0)

    print ("Time for computing preconditioner:                %f sec\n" % (time.time() - start_time))
    sys.stdout.flush()

    return precond

def apply_S_12_cca(mr_adc, X, transpose = False):

    # Einsum definition from kernel
    einsum = mr_adc.interface.einsum
    einsum_type = mr_adc.interface.einsum_type

    # Dimensions
    nextern = mr_adc.nextern
    ncvs = mr_adc.ncvs
    nval = mr_adc.nval

    ho_s_c = mr_adc.h_orth.s_c
    ho_f_c = mr_adc.h_orth.f_c
    ho_s_cce = mr_adc.h_orth.s_cce
    ho_f_cce = mr_adc.h_orth.f_cce
    ho_s_cae = mr_adc.h_orth.s_cae
    ho_f_cae = mr_adc.h_orth.f_cae
    ho_s_ace = mr_adc.h_orth.s_ace
    ho_f_ace = mr_adc.h_orth.f_ace
    ho_s_cca_aaa = mr_adc.h_orth.s_cca_aaa
    ho_f_cca_aaa = mr_adc.h_orth.f_cca_aaa
    ho_s_cca_abb = mr_adc.h_orth.s_cca_abb
    ho_f_cca_abb = mr_adc.h_orth.f_cca_abb
    ho_s_cca = mr_adc.h_orth.s_cca_aaa
    ho_f_cca = mr_adc.h_orth.f_cca_abb
    s_c = mr_adc.h0.s_c
    f_c = mr_adc.h0.f_c
    s_caa = mr_adc.h1.s_caa
    f_caa = mr_adc.h1.f_caa
    s_cce = mr_adc.h1.s_cce
    f_cce = mr_adc.h1.f_cce
    s_cae = mr_adc.h1.s_cae
    f_cae = mr_adc.h1.f_cae
    s_ace = mr_adc.h1.s_ace
    f_ace = mr_adc.h1.f_ace
    s_cca = mr_adc.h1.s_cca_aaa
    f_cca = mr_adc.h1.f_cca_abb
    s_cca_aaa = mr_adc.h1.s_cca_aaa
    f_cca_aaa = mr_adc.h1.f_cca_aaa
    s_cca_abb = mr_adc.h1.s_cca_abb
    f_cca_abb = mr_adc.h1.f_cca_abb

    if nval > 0:
        ho_s_cve = mr_adc.h_orth.s_cve
        ho_f_cve = mr_adc.h_orth.f_cve
        ho_s_vce = mr_adc.h_orth.s_vce
        ho_f_vce = mr_adc.h_orth.f_vce

        ho_s_cva = mr_adc.h_orth.s_cva_aaa
        ho_f_cva = mr_adc.h_orth.f_cva_bab
        ho_s_cva_aaa = mr_adc.h_orth.s_cva_aaa
        ho_f_cva_aaa = mr_adc.h_orth.f_cva_aaa
        ho_s_cva_abb = mr_adc.h_orth.s_cva_abb
        ho_f_cva_abb = mr_adc.h_orth.f_cva_abb
        ho_s_cva_bab = mr_adc.h_orth.s_cva_bab
        ho_f_cva_bab = mr_adc.h_orth.f_cva_bab

        s_cve = mr_adc.h1.s_cve
        f_cve = mr_adc.h1.f_cve
        s_vce = mr_adc.h1.s_vce
        f_vce = mr_adc.h1.f_vce

        s_cva = mr_adc.h1.s_cva_aaa
        f_cva = mr_adc.h1.f_cva_bab
        s_cva_aaa = mr_adc.h1.s_cva_aaa
        f_cva_aaa = mr_adc.h1.f_cva_aaa
        s_cva_abb = mr_adc.h1.s_cva_abb
        f_cva_abb = mr_adc.h1.f_cva_abb
        s_cva_bab = mr_adc.h1.s_cva_bab
        f_cva_bab = mr_adc.h1.f_cva_bab

    # Overlap matrices
    S12_c_caa = mr_adc.S12.c_caa
    S12_cae = mr_adc.S12.cae
    S12_cca = mr_adc.S12.cca

    cc_ind = np.tril_indices(ncvs, k=-1)

    Xt = None

    if transpose:
        if (X.shape[0] != (mr_adc.h0.dim + mr_adc.h1.dim)):
            raise Exception("Dimensions do not match when applying S_12 transpose")

        Xt = np.zeros(mr_adc.h_orth.dim)

        # C-C DEBUG
        Xt[ho_s_c:ho_f_c] = X[s_c:f_c].copy()

        # CCA
        temp = X[s_cca:f_cca].reshape(-1, S12_cca.shape[0]).copy()
        Xt[ho_s_cca:ho_f_cca] = einsum("IX,XP->IP", temp, S12_cca).reshape(-1).copy()

        if nval > 0:
            # CVA
            temp = X[s_cva:f_cva].reshape(-1, S12_cca.shape[0]).copy()
            Xt[ho_s_cva:ho_f_cva] = einsum("IX,XP->IP", temp, S12_cca).reshape(-1).copy()

    else:
        if (X.shape[0] != (mr_adc.h_orth.dim)):
            raise Exception("Dimensions do not match when applying S_12")

        Xt = np.zeros(mr_adc.h0.dim + mr_adc.h1.dim)

        # C-C DEBUG
        Xt[s_c:f_c] = X[ho_s_c:ho_f_c].copy()

        # CCA
        temp = X[ho_s_cca:ho_f_cca].reshape(-1, S12_cca.shape[1]).copy()
        Xt[s_cca:f_cca] = einsum("IP,XP->IX", temp, S12_cca).reshape(-1).copy()

        if nval > 0:
            # CVA
            temp = X[ho_s_cva:ho_f_cva].reshape(-1, S12_cca.shape[1]).copy()
            Xt[s_cva:f_cva] = einsum("IP,XP->IX", temp, S12_cca).reshape(-1).copy()

    return Xt

def compute_sigma_vector_cca(mr_adc, M_00, M_01, M_11, Xt):

    # Einsum definition from kernel
    einsum = mr_adc.interface.einsum
    einsum_type = mr_adc.interface.einsum_type

    # Variables from kernel
    ncvs = mr_adc.ncvs
    nval = mr_adc.nval
    ncas = mr_adc.ncas
    nextern = mr_adc.nextern

    e_cvs = mr_adc.mo_energy.x
    e_core = mr_adc.mo_energy.c
    if nval > 0:
        e_val = mr_adc.mo_energy.v
        e_extern = mr_adc.mo_energy.e

    # One-electron integrals
    h_aa = mr_adc.h1eff.aa

    # Two-electrons integrals
    v_aaaa = mr_adc.v2e.aaaa

    # Reduced Density Matrices
    rdm_ca = mr_adc.rdm.ca
    rdm_ccaa = mr_adc.rdm.ccaa

    # Dimensions
    s_c = mr_adc.h0.s_c
    f_c = mr_adc.h0.f_c
    s_caa = mr_adc.h1.s_caa
    f_caa = mr_adc.h1.f_caa
    s_cce = mr_adc.h1.s_cce
    f_cce = mr_adc.h1.f_cce
    s_cae = mr_adc.h1.s_cae
    f_cae = mr_adc.h1.f_cae
    s_ace = mr_adc.h1.s_ace
    f_ace = mr_adc.h1.f_ace
    s_cca_aaa = mr_adc.h1.s_cca_aaa
    f_cca_aaa = mr_adc.h1.f_cca_aaa
    s_cca_abb = mr_adc.h1.s_cca_abb
    f_cca_abb = mr_adc.h1.f_cca_abb
    s_cca = mr_adc.h1.s_cca_aaa
    f_cca = mr_adc.h1.f_cca_abb
    if nval > 0:
        s_cve = mr_adc.h1.s_cve
        f_cve = mr_adc.h1.f_cve
        s_vce = mr_adc.h1.s_vce
        f_vce = mr_adc.h1.f_vce

        s_cva = mr_adc.h1.s_cva_aaa
        f_cva = mr_adc.h1.f_cva_bab
        s_cva_aaa = mr_adc.h1.s_cva_aaa
        f_cva_aaa = mr_adc.h1.f_cva_aaa
        s_cva_abb = mr_adc.h1.s_cva_abb
        f_cva_abb = mr_adc.h1.f_cva_abb
        s_cva_bab = mr_adc.h1.s_cva_bab
        f_cva_bab = mr_adc.h1.f_cva_bab

    cvs_tril_ind = np.tril_indices(ncvs, k=-1)

    # (CASCI + C) -> (CASCI + C)
    sigma = np.zeros_like(Xt)

    # h0-h0 contributions
    sigma[:mr_adc.h0.dim] = np.dot(M_00, Xt[:mr_adc.h0.dim])

    # h0-h1 and h1-h0 contributions
    if nval > 0:
        M_C_CAA, M_C_CCE, M_C_CVE, M_C_CAE, M_C_ACE, M_C_CCA, M_C_CVA = M_01
    else:
        M_C_CAA, M_C_CCE, M_C_CAE, M_C_ACE, M_C_CCA = M_01

    # C <-> CCA
    sigma[s_c:f_c] += np.dot(M_C_CCA, Xt[s_cca:f_cca])
    sigma[s_cca:f_cca] += np.dot(M_C_CCA.T, Xt[s_c:f_c])

    # C <-> CVA
    if nval > 0:
        sigma[s_c:f_c] += np.dot(M_C_CVA, Xt[s_cva:f_cva])
        sigma[s_cva:f_cva] += np.dot(M_C_CVA.T, Xt[s_c:f_c])

    # h1-h1 contributions
    # CCA <- CCA
    X_cca = Xt[s_cca:f_cca].copy()

    dim_cca = ncvs * ncvs * ncas
    dim_tril_cca = ncvs * (ncvs - 1) * ncas // 2

    sigma_aaa_i = 0
    sigma_aaa_f = sigma_aaa_i + dim_tril_cca
    sigma_abb_i = sigma_aaa_f
    sigma_abb_f = sigma_abb_i + dim_cca

    X_aaa = np.zeros((ncvs, ncvs, ncas))
    X_aaa[cvs_tril_ind[0], cvs_tril_ind[1]] =  X_cca[sigma_aaa_i:sigma_aaa_f].reshape(-1, ncas).copy()
    X_aaa[cvs_tril_ind[1], cvs_tril_ind[0]] =- X_cca[sigma_aaa_i:sigma_aaa_f].reshape(-1, ncas).copy()

    X_abb = X_cca[sigma_abb_i:sigma_abb_f].reshape(ncvs, ncvs, ncas).copy()

    sigma_cca_aaa  = einsum('KLW,K->KLW', X_aaa, e_cvs, optimize = einsum_type)
    sigma_cca_aaa += einsum('KLW,L->KLW', X_aaa, e_cvs, optimize = einsum_type)
    sigma_cca_aaa -= einsum('KLx,Wx->KLW', X_aaa, h_aa, optimize = einsum_type)
    sigma_cca_aaa -= 1/2 * einsum('KLx,K,Wx->KLW', X_aaa, e_cvs, rdm_ca, optimize = einsum_type)
    sigma_cca_aaa -= 1/2 * einsum('KLx,L,Wx->KLW', X_aaa, e_cvs, rdm_ca, optimize = einsum_type)
    sigma_cca_aaa += 1/2 * einsum('KLx,xy,Wy->KLW', X_aaa, h_aa, rdm_ca, optimize = einsum_type)
    sigma_cca_aaa -= einsum('KLx,Wyxz,zy->KLW', X_aaa, v_aaaa, rdm_ca, optimize = einsum_type)
    sigma_cca_aaa += 1/2 * einsum('KLx,Wyzx,zy->KLW', X_aaa, v_aaaa, rdm_ca, optimize = einsum_type)
    sigma_cca_aaa += 1/2 * einsum('KLx,xyzw,Wyzw->KLW', X_aaa, v_aaaa, rdm_ccaa, optimize = einsum_type)

    sigma_cca_abb  = einsum('KLW,K->KLW', X_abb, e_cvs, optimize = einsum_type)
    sigma_cca_abb += einsum('KLW,L->KLW', X_abb, e_cvs, optimize = einsum_type)
    sigma_cca_abb -= einsum('KLx,Wx->KLW', X_abb, h_aa, optimize = einsum_type)
    sigma_cca_abb -= 1/2 * einsum('KLx,K,Wx->KLW', X_abb, e_cvs, rdm_ca, optimize = einsum_type)
    sigma_cca_abb -= 1/2 * einsum('KLx,L,Wx->KLW', X_abb, e_cvs, rdm_ca, optimize = einsum_type)
    sigma_cca_abb += 1/2 * einsum('KLx,xy,Wy->KLW', X_abb, h_aa, rdm_ca, optimize = einsum_type)
    sigma_cca_abb -= einsum('KLx,Wyxz,zy->KLW', X_abb, v_aaaa, rdm_ca, optimize = einsum_type)
    sigma_cca_abb += 1/2 * einsum('KLx,Wyzx,zy->KLW', X_abb, v_aaaa, rdm_ca, optimize = einsum_type)
    sigma_cca_abb += 1/2 * einsum('KLx,xyzw,Wyzw->KLW', X_abb, v_aaaa, rdm_ccaa, optimize = einsum_type)

    ## Building C-CCA matrix
    dim_tril_cca = ncvs * (ncvs - 1) * ncas // 2
    dim_cca = ncvs * ncvs * ncas
    dim_sigma_cca = dim_cca + dim_tril_cca

    sigma_cca_aaa_i = 0
    sigma_cca_aaa_f = sigma_cca_aaa_i + dim_tril_cca
    sigma_cca_abb_i = sigma_cca_aaa_f
    sigma_cca_abb_f = sigma_cca_abb_i + dim_cca

    sigma_cca = np.zeros((dim_sigma_cca))
    sigma_cca[sigma_cca_aaa_i:sigma_cca_aaa_f] = sigma_cca_aaa[cvs_tril_ind[0], cvs_tril_ind[1]].reshape(-1).copy()
    sigma_cca[sigma_cca_abb_i:sigma_cca_abb_f] = sigma_cca_abb.reshape(-1).copy()

    sigma[s_cca:f_cca] += sigma_cca.reshape(-1).copy()

    if nval > 0:
        # CVA <- CVA
        X_aaa = Xt[s_cva_aaa:f_cva_aaa].reshape(ncvs, nval, ncas).copy()
        X_abb = Xt[s_cva_abb:f_cva_abb].reshape(ncvs, nval, ncas).copy()
        X_bab = Xt[s_cva_bab:f_cva_bab].reshape(ncvs, nval, ncas).copy()

        sigma_cva_aaa  = einsum('KLW,K->KLW', X_aaa, e_cvs, optimize = einsum_type)
        sigma_cva_aaa += einsum('KLW,L->KLW', X_aaa, e_val, optimize = einsum_type)
        sigma_cva_aaa -= einsum('KLx,Wx->KLW', X_aaa, h_aa, optimize = einsum_type)
        sigma_cva_aaa -= 1/2 * einsum('KLx,K,Wx->KLW', X_aaa, e_cvs, rdm_ca, optimize = einsum_type)
        sigma_cva_aaa -= 1/2 * einsum('KLx,L,Wx->KLW', X_aaa, e_val, rdm_ca, optimize = einsum_type)
        sigma_cva_aaa += 1/2 * einsum('KLx,xy,Wy->KLW', X_aaa, h_aa, rdm_ca, optimize = einsum_type)
        sigma_cva_aaa -= einsum('KLx,Wyxz,zy->KLW', X_aaa, v_aaaa, rdm_ca, optimize = einsum_type)
        sigma_cva_aaa += 1/2 * einsum('KLx,Wyzx,zy->KLW', X_aaa, v_aaaa, rdm_ca, optimize = einsum_type)
        sigma_cva_aaa += 1/2 * einsum('KLx,xyzw,Wyzw->KLW', X_aaa, v_aaaa, rdm_ccaa, optimize = einsum_type)

        sigma_cva_abb  = einsum('KLW,K->KLW', X_abb, e_cvs, optimize = einsum_type)
        sigma_cva_abb += einsum('KLW,L->KLW', X_abb, e_val, optimize = einsum_type)
        sigma_cva_abb -= einsum('KLx,Wx->KLW', X_abb, h_aa, optimize = einsum_type)
        sigma_cva_abb -= 1/2 * einsum('KLx,K,Wx->KLW', X_abb, e_cvs, rdm_ca, optimize = einsum_type)
        sigma_cva_abb -= 1/2 * einsum('KLx,L,Wx->KLW', X_abb, e_val, rdm_ca, optimize = einsum_type)
        sigma_cva_abb += 1/2 * einsum('KLx,xy,Wy->KLW', X_abb, h_aa, rdm_ca, optimize = einsum_type)
        sigma_cva_abb -= einsum('KLx,Wyxz,zy->KLW', X_abb, v_aaaa, rdm_ca, optimize = einsum_type)
        sigma_cva_abb += 1/2 * einsum('KLx,Wyzx,zy->KLW', X_abb, v_aaaa, rdm_ca, optimize = einsum_type)
        sigma_cva_abb += 1/2 * einsum('KLx,xyzw,Wyzw->KLW', X_abb, v_aaaa, rdm_ccaa, optimize = einsum_type)

        sigma_cva_bab  = einsum('KLW,K->KLW', X_bab, e_cvs, optimize = einsum_type)
        sigma_cva_bab += einsum('KLW,L->KLW', X_bab, e_val, optimize = einsum_type)
        sigma_cva_bab -= einsum('KLx,Wx->KLW', X_bab, h_aa, optimize = einsum_type)
        sigma_cva_bab -= 1/2 * einsum('KLx,K,Wx->KLW', X_bab, e_cvs, rdm_ca, optimize = einsum_type)
        sigma_cva_bab -= 1/2 * einsum('KLx,L,Wx->KLW', X_bab, e_val, rdm_ca, optimize = einsum_type)
        sigma_cva_bab += 1/2 * einsum('KLx,xy,Wy->KLW', X_bab, h_aa, rdm_ca, optimize = einsum_type)
        sigma_cva_bab -= einsum('KLx,Wyxz,zy->KLW', X_bab, v_aaaa, rdm_ca, optimize = einsum_type)
        sigma_cva_bab += 1/2 * einsum('KLx,Wyzx,zy->KLW', X_bab, v_aaaa, rdm_ca, optimize = einsum_type)
        sigma_cva_bab += 1/2 * einsum('KLx,xyzw,Wyzw->KLW', X_bab, v_aaaa, rdm_ccaa, optimize = einsum_type)

        ## Building C-CVA matrix
        dim_cva = ncvs * nval * ncas
        dim_sigma_cva = 3 * dim_cva

        sigma_cva_aaa_i = 0
        sigma_cva_aaa_f = sigma_cva_aaa_i + dim_cva
        sigma_cva_abb_i = sigma_cva_aaa_f
        sigma_cva_abb_f = sigma_cva_abb_i + dim_cva
        sigma_cva_bab_i = sigma_cva_abb_f
        sigma_cva_bab_f = sigma_cva_bab_i + dim_cva

        sigma_cva = np.zeros((dim_sigma_cva))
        sigma_cva[sigma_cva_aaa_i:sigma_cva_aaa_f] = sigma_cva_aaa.reshape(-1).copy()
        sigma_cva[sigma_cva_abb_i:sigma_cva_abb_f] = sigma_cva_abb.reshape(-1).copy()
        sigma_cva[sigma_cva_bab_i:sigma_cva_bab_f] = sigma_cva_bab.reshape(-1).copy()
        sigma[s_cva:f_cva] += sigma_cva.reshape(-1).copy()

    return sigma

## CAE block: Diagonal + M01
def compute_excitation_manifolds_cae(mr_adc):

    # MR-ADC(0) and MR-ADC(1)
    mr_adc.h0.n_c = mr_adc.ncvs
    mr_adc.h0.dim = mr_adc.h0.n_c # Total dimension of h0

    mr_adc.h0.s_c = 0
    mr_adc.h0.f_c = mr_adc.h0.s_c + mr_adc.h0.n_c

    print("Dimension of h0 excitation manifold:                       %d" % mr_adc.h0.dim)

    # MR-ADC(2)
    mr_adc.h1.dim = 0
    mr_adc.h_orth.dim = mr_adc.h0.dim

    if mr_adc.method in ("mr-adc(2)", "mr-adc(2)-x"):
        mr_adc.h1.n_caa = 0
        mr_adc.h1.n_cce = 0
        mr_adc.h1.n_cae = mr_adc.nextern * mr_adc.ncas * mr_adc.ncvs
        mr_adc.h1.n_cca = 0
        if mr_adc.nval > 0:
            mr_adc.h1.n_cve = 0
            mr_adc.h1.n_vce = 0
            mr_adc.h1.n_cva = 0
            mr_adc.h1.n_vca = 0
            mr_adc.h1.dim = (mr_adc.h1.n_caa + mr_adc.h1.n_cce + mr_adc.h1.n_cve + mr_adc.h1.n_vce +
                             3 * mr_adc.h1.n_cae + mr_adc.h1.n_cca + mr_adc.h1.n_cva + mr_adc.h1.n_vca)
        else:
            mr_adc.h1.dim = mr_adc.h1.n_caa + mr_adc.h1.n_cce + 3 * mr_adc.h1.n_cae + mr_adc.h1.n_cca

        if mr_adc.nval > 0:
            mr_adc.h1.s_caa = mr_adc.h0.f_c
            mr_adc.h1.f_caa = mr_adc.h1.s_caa + mr_adc.h1.n_caa
            mr_adc.h1.s_cce = mr_adc.h1.f_caa
            mr_adc.h1.f_cce = mr_adc.h1.s_cce + mr_adc.h1.n_cce
            mr_adc.h1.s_cve = mr_adc.h1.f_cce
            mr_adc.h1.f_cve = mr_adc.h1.s_cve + mr_adc.h1.n_cve
            mr_adc.h1.s_vce = mr_adc.h1.f_cve
            mr_adc.h1.f_vce = mr_adc.h1.s_vce + mr_adc.h1.n_vce
            mr_adc.h1.s_cae_aaa = mr_adc.h1.f_vce
            mr_adc.h1.f_cae_aaa = mr_adc.h1.s_cae_aaa + mr_adc.h1.n_cae
            mr_adc.h1.s_cae_abb = mr_adc.h1.f_cae_aaa
            mr_adc.h1.f_cae_abb = mr_adc.h1.s_cae_abb + mr_adc.h1.n_cae
            mr_adc.h1.s_cae_bab = mr_adc.h1.f_cae_abb
            mr_adc.h1.f_cae_bab = mr_adc.h1.s_cae_bab + mr_adc.h1.n_cae
            mr_adc.h1.s_cca = mr_adc.h1.f_cae_bab
            mr_adc.h1.f_cca = mr_adc.h1.s_cca + mr_adc.h1.n_cca
            mr_adc.h1.s_cva = mr_adc.h1.f_cca
            mr_adc.h1.f_cva = mr_adc.h1.s_cva + mr_adc.h1.n_cva
            mr_adc.h1.s_vca = mr_adc.h1.f_cva
            mr_adc.h1.f_vca = mr_adc.h1.s_vca + mr_adc.h1.n_vca
        else:
            mr_adc.h1.s_caa = mr_adc.h0.f_c
            mr_adc.h1.f_caa = mr_adc.h1.s_caa + mr_adc.h1.n_caa
            mr_adc.h1.s_cce = mr_adc.h1.f_caa
            mr_adc.h1.f_cce = mr_adc.h1.s_cce + mr_adc.h1.n_cce
            mr_adc.h1.s_cae_aaa = mr_adc.h1.f_cce
            mr_adc.h1.f_cae_aaa = mr_adc.h1.s_cae_aaa + mr_adc.h1.n_cae
            mr_adc.h1.s_cae_abb = mr_adc.h1.f_cae_aaa
            mr_adc.h1.f_cae_abb = mr_adc.h1.s_cae_abb + mr_adc.h1.n_cae
            mr_adc.h1.s_cae_bab = mr_adc.h1.f_cae_abb
            mr_adc.h1.f_cae_bab = mr_adc.h1.s_cae_bab + mr_adc.h1.n_cae
            mr_adc.h1.s_cca = mr_adc.h1.f_cae_bab
            mr_adc.h1.f_cca = mr_adc.h1.s_cca + mr_adc.h1.n_cca

        print("Dimension of h1 excitation manifold:                       %d" % mr_adc.h1.dim)

        # Overlap for c - caa
        mr_adc.S12.c_caa = mr_adc_overlap.compute_S12_0p_projector(mr_adc)
        mr_adc.S12.cae = mr_adc_overlap.compute_S12_m1(mr_adc)
        mr_adc.S12.cca = mr_adc_overlap.compute_S12_p1(mr_adc)

        # Determine dimensions of orthogonalized excitation spaces
        mr_adc.h_orth.n_c = mr_adc.ncvs
        mr_adc.h_orth.n_c_caa = 0
        mr_adc.h_orth.n_cce = 0
        mr_adc.h_orth.n_cce = 0
        mr_adc.h_orth.n_cae = mr_adc.nextern * mr_adc.ncvs * mr_adc.S12.cae.shape[1]
        mr_adc.h_orth.n_cca = 0
        if mr_adc.nval > 0:
            mr_adc.h_orth.n_cve = 0
            mr_adc.h_orth.n_vce = 0
            mr_adc.h_orth.n_cva = 0
            mr_adc.h_orth.n_vca = 0
            mr_adc.h_orth.dim = (mr_adc.h_orth.n_c + mr_adc.h_orth.n_c_caa + mr_adc.h_orth.n_cce + mr_adc.h_orth.n_cve + mr_adc.h_orth.n_vce +
                                 3 * mr_adc.h_orth.n_cae + mr_adc.h_orth.n_cca + mr_adc.h_orth.n_cva + mr_adc.h_orth.n_vca)
        else:
            mr_adc.h_orth.dim = mr_adc.h_orth.n_c + mr_adc.h_orth.n_c_caa + mr_adc.h_orth.n_cce + 3 * mr_adc.h_orth.n_cae + mr_adc.h_orth.n_cca

        if mr_adc.nval > 0:
            mr_adc.h_orth.s_c = 0
            mr_adc.h_orth.f_c = mr_adc.h_orth.n_c
            mr_adc.h_orth.s_cce = mr_adc.h_orth.f_c
            mr_adc.h_orth.f_cce = mr_adc.h_orth.s_cce + mr_adc.h_orth.n_cce
            mr_adc.h_orth.s_cve = mr_adc.h_orth.f_cce
            mr_adc.h_orth.f_cve = mr_adc.h_orth.s_cve + mr_adc.h_orth.n_cve
            mr_adc.h_orth.s_vce = mr_adc.h_orth.f_cve
            mr_adc.h_orth.f_vce = mr_adc.h_orth.s_vce + mr_adc.h_orth.n_vce
            mr_adc.h_orth.s_cae_aaa = mr_adc.h_orth.f_vce
            mr_adc.h_orth.f_cae_aaa = mr_adc.h_orth.s_cae_aaa + mr_adc.h_orth.n_cae
            mr_adc.h_orth.s_cae_abb = mr_adc.h_orth.f_cae_aaa
            mr_adc.h_orth.f_cae_abb = mr_adc.h_orth.s_cae_abb + mr_adc.h_orth.n_cae
            mr_adc.h_orth.s_cae_bab = mr_adc.h_orth.f_cae_abb
            mr_adc.h_orth.f_cae_bab = mr_adc.h_orth.s_cae_bab + mr_adc.h_orth.n_cae
            mr_adc.h_orth.s_cca = mr_adc.h_orth.f_cae_bab
            mr_adc.h_orth.f_cca = mr_adc.h_orth.s_cca + mr_adc.h_orth.n_cca
            mr_adc.h_orth.s_cva = mr_adc.h_orth.f_cca
            mr_adc.h_orth.f_cva = mr_adc.h_orth.s_cva + mr_adc.h_orth.n_cva
            mr_adc.h_orth.s_vca = mr_adc.h_orth.f_cva
            mr_adc.h_orth.f_vca = mr_adc.h_orth.s_vca + mr_adc.h_orth.n_vca
        else:
            mr_adc.h_orth.s_c = 0
            mr_adc.h_orth.f_c = mr_adc.h_orth.n_c
            mr_adc.h_orth.s_cce = mr_adc.h_orth.f_c
            mr_adc.h_orth.f_cce = mr_adc.h_orth.s_cce + mr_adc.h_orth.n_cce
            mr_adc.h_orth.s_cae_aaa = mr_adc.h_orth.f_cce
            mr_adc.h_orth.f_cae_aaa = mr_adc.h_orth.s_cae_aaa + mr_adc.h_orth.n_cae
            mr_adc.h_orth.s_cae_abb = mr_adc.h_orth.f_cae_aaa
            mr_adc.h_orth.f_cae_abb = mr_adc.h_orth.s_cae_abb + mr_adc.h_orth.n_cae
            mr_adc.h_orth.s_cae_bab = mr_adc.h_orth.f_cae_abb
            mr_adc.h_orth.f_cae_bab = mr_adc.h_orth.s_cae_bab + mr_adc.h_orth.n_cae
            mr_adc.h_orth.s_cca = mr_adc.h_orth.f_cae_bab
            mr_adc.h_orth.f_cca = mr_adc.h_orth.s_cca + mr_adc.h_orth.n_cca

    print("Total dimension of the excitation manifold:                %d" % (mr_adc.h0.dim + mr_adc.h1.dim))
    print("Dimension of the orthogonalized excitation manifold:       %d\n" % (mr_adc.h_orth.dim))
    sys.stdout.flush()

    if (mr_adc.h_orth.dim < mr_adc.nroots):
        mr_adc.nroots = mr_adc.h_orth.dim

    return mr_adc

def compute_M_01_cae(mr_adc):

    start_time = time.time()

    print ("Computing M(h0-h1) blocks...")
    sys.stdout.flush()

    # Einsum definition from kernel
    einsum = mr_adc.interface.einsum
    einsum_type = mr_adc.interface.einsum_type

    # Dimensions
    ncore = mr_adc.ncore
    ncas = mr_adc.ncas
    nocc = mr_adc.nocc
    nextern = mr_adc.nextern

    ncvs = mr_adc.ncvs
    nval = mr_adc.nval

    # cvs_tril_ind = np.tril_indices(ncvs, k=-1)

    # MOs Energy
    e_cvs = mr_adc.mo_energy.x
    if nval > 0:
        e_val = mr_adc.mo_energy.v
    e_core = mr_adc.mo_energy.c
    e_extern = mr_adc.mo_energy.e

    # Amplitudes
    t1_ce = mr_adc.t1.ce
    t1_ca = mr_adc.t1.ca
    t1_ae = mr_adc.t1.ae
    t1_caea = mr_adc.t1.caea
    t1_caae = mr_adc.t1.caae
    t1_caaa = mr_adc.t1.caaa
    t1_aaea = mr_adc.t1.aaea
    t1_aaae = mr_adc.t1.aaae

    t1_xe = mr_adc.t1.xe
    t1_xaea = mr_adc.t1.xaea
    t1_xaae = mr_adc.t1.xaae

    if nval > 0:
        t1_ve = mr_adc.t1.ve
        t1_vaea = mr_adc.t1.vaea
        t1_vaae = mr_adc.t1.vaae

    # One-electron integrals
    h_aa = mr_adc.h1eff.aa
    h_ae = mr_adc.h1eff.ae

    h_xe = mr_adc.h1eff.xe

    if nval > 0:
        h_ve = mr_adc.h1eff.ve

    # Two-electrons integrals
    v_aaaa = mr_adc.v2e.aaaa
    v_aaae = mr_adc.v2e.aaae

    v_xaxa = mr_adc.v2e.xaxa
    v_xaax = mr_adc.v2e.xaax

    v_xaxe = mr_adc.v2e.xaxe
    v_xaex = mr_adc.v2e.xaex

    if nval > 0:
        v_vxxe = mr_adc.v2e.vxxe
        v_xvxe = mr_adc.v2e.xvxe

    v_xaea = mr_adc.v2e.xaea
    v_xaae = mr_adc.v2e.xaae
    v_xxxe = mr_adc.v2e.xxxe

    if nval > 0:
        v_vaea = mr_adc.v2e.vaea
        v_vaae = mr_adc.v2e.vaae

    # Reduced Density Matrices
    rdm_ca = mr_adc.rdm.ca
    rdm_ccaa = mr_adc.rdm.ccaa
    rdm_cccaaa = mr_adc.rdm.cccaaa

    # Indices
    cvs_tril_ind = np.tril_indices(ncvs, k=-1)

    # C - CAE
    M_C_CAE_a_aaa =- 1/2 * einsum('KxBI,Yx->IKYB', v_xaex, rdm_ca, optimize = einsum_type)
    M_C_CAE_a_aaa += 1/2 * einsum('KxIB,Yx->IKYB', v_xaxe, rdm_ca, optimize = einsum_type)
    M_C_CAE_a_aaa -= 1/2 * einsum('xB,IK,Yx->IKYB', h_ae, np.identity(ncvs), rdm_ca, optimize = einsum_type)
    M_C_CAE_a_aaa -= 1/2 * einsum('IK,xyzB,Yzyx->IKYB', np.identity(ncvs), v_aaae, rdm_ccaa, optimize = einsum_type)
    M_C_CAE_a_aaa -= 1/2 * einsum('B,IK,xB,Yx->IKYB', e_extern, np.identity(ncvs), t1_ae, rdm_ca, optimize = einsum_type)
    M_C_CAE_a_aaa -= 1/2 * einsum('B,IK,xyzB,Yzyx->IKYB', e_extern, np.identity(ncvs), t1_aaae, rdm_ccaa, optimize = einsum_type)
    M_C_CAE_a_aaa += 1/2 * einsum('xy,IK,xB,Yy->IKYB', h_aa, np.identity(ncvs), t1_ae, rdm_ca, optimize = einsum_type)
    M_C_CAE_a_aaa -= 1/2 * einsum('xy,IK,zwxB,Yywz->IKYB', h_aa, np.identity(ncvs), t1_aaae, rdm_ccaa, optimize = einsum_type)
    M_C_CAE_a_aaa += 1/2 * einsum('xy,IK,xzwB,Ywzy->IKYB', h_aa, np.identity(ncvs), t1_aaae, rdm_ccaa, optimize = einsum_type)
    M_C_CAE_a_aaa += 1/2 * einsum('xy,IK,zxwB,Ywyz->IKYB', h_aa, np.identity(ncvs), t1_aaae, rdm_ccaa, optimize = einsum_type)
    M_C_CAE_a_aaa += 1/2 * einsum('IK,xB,xyzw,Yyzw->IKYB', np.identity(ncvs), t1_ae, v_aaaa, rdm_ccaa, optimize = einsum_type)
    M_C_CAE_a_aaa += 1/12 * einsum('IK,xyzB,zwuv,Yuvxyw->IKYB', np.identity(ncvs), t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    M_C_CAE_a_aaa += 1/12 * einsum('IK,xyzB,zwuv,Yuvxwy->IKYB', np.identity(ncvs), t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    M_C_CAE_a_aaa -= 5/12 * einsum('IK,xyzB,zwuv,Yuvyxw->IKYB', np.identity(ncvs), t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    M_C_CAE_a_aaa += 1/12 * einsum('IK,xyzB,zwuv,Yuvywx->IKYB', np.identity(ncvs), t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    M_C_CAE_a_aaa += 1/12 * einsum('IK,xyzB,zwuv,Yuvwxy->IKYB', np.identity(ncvs), t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    M_C_CAE_a_aaa += 1/12 * einsum('IK,xyzB,zwuv,Yuvwyx->IKYB', np.identity(ncvs), t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    M_C_CAE_a_aaa += 1/2 * einsum('IK,xyzB,xywu,Yzuw->IKYB', np.identity(ncvs), t1_aaae, v_aaaa, rdm_ccaa, optimize = einsum_type)
    M_C_CAE_a_aaa += 5/12 * einsum('IK,xyzB,xwuv,Yzwyuv->IKYB', np.identity(ncvs), t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    M_C_CAE_a_aaa -= 1/12 * einsum('IK,xyzB,xwuv,Yzwyvu->IKYB', np.identity(ncvs), t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    M_C_CAE_a_aaa -= 1/12 * einsum('IK,xyzB,xwuv,Yzwuyv->IKYB', np.identity(ncvs), t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    M_C_CAE_a_aaa -= 1/12 * einsum('IK,xyzB,xwuv,Yzwuvy->IKYB', np.identity(ncvs), t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    M_C_CAE_a_aaa -= 1/12 * einsum('IK,xyzB,xwuv,Yzwvyu->IKYB', np.identity(ncvs), t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    M_C_CAE_a_aaa -= 1/12 * einsum('IK,xyzB,xwuv,Yzwvuy->IKYB', np.identity(ncvs), t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    M_C_CAE_a_aaa -= 1/12 * einsum('IK,xyzB,ywuv,Yzwxuv->IKYB', np.identity(ncvs), t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    M_C_CAE_a_aaa -= 1/12 * einsum('IK,xyzB,ywuv,Yzwxvu->IKYB', np.identity(ncvs), t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    M_C_CAE_a_aaa += 5/12 * einsum('IK,xyzB,ywuv,Yzwuxv->IKYB', np.identity(ncvs), t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    M_C_CAE_a_aaa -= 1/12 * einsum('IK,xyzB,ywuv,Yzwuvx->IKYB', np.identity(ncvs), t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    M_C_CAE_a_aaa -= 1/12 * einsum('IK,xyzB,ywuv,Yzwvxu->IKYB', np.identity(ncvs), t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    M_C_CAE_a_aaa -= 1/12 * einsum('IK,xyzB,ywuv,Yzwvux->IKYB', np.identity(ncvs), t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)

    M_C_CAE_a_abb  = 1/2 * einsum('KxIB,Yx->IKYB', v_xaxe, rdm_ca, optimize = einsum_type)
    M_C_CAE_a_abb -= 1/2 * einsum('xB,IK,Yx->IKYB', h_ae, np.identity(ncvs), rdm_ca, optimize = einsum_type)
    M_C_CAE_a_abb -= 1/2 * einsum('IK,xyzB,Yzyx->IKYB', np.identity(ncvs), v_aaae, rdm_ccaa, optimize = einsum_type)
    M_C_CAE_a_abb -= 1/2 * einsum('B,IK,xB,Yx->IKYB', e_extern, np.identity(ncvs), t1_ae, rdm_ca, optimize = einsum_type)
    M_C_CAE_a_abb -= 1/2 * einsum('B,IK,xyzB,Yzyx->IKYB', e_extern, np.identity(ncvs), t1_aaae, rdm_ccaa, optimize = einsum_type)
    M_C_CAE_a_abb += 1/2 * einsum('xy,IK,xB,Yy->IKYB', h_aa, np.identity(ncvs), t1_ae, rdm_ca, optimize = einsum_type)
    M_C_CAE_a_abb -= 1/2 * einsum('xy,IK,zwxB,Yywz->IKYB', h_aa, np.identity(ncvs), t1_aaae, rdm_ccaa, optimize = einsum_type)
    M_C_CAE_a_abb += 1/2 * einsum('xy,IK,xzwB,Ywzy->IKYB', h_aa, np.identity(ncvs), t1_aaae, rdm_ccaa, optimize = einsum_type)
    M_C_CAE_a_abb += 1/2 * einsum('xy,IK,zxwB,Ywyz->IKYB', h_aa, np.identity(ncvs), t1_aaae, rdm_ccaa, optimize = einsum_type)
    M_C_CAE_a_abb += 1/2 * einsum('IK,xB,xyzw,Yyzw->IKYB', np.identity(ncvs), t1_ae, v_aaaa, rdm_ccaa, optimize = einsum_type)
    M_C_CAE_a_abb += 1/12 * einsum('IK,xyzB,zwuv,Yuvxyw->IKYB', np.identity(ncvs), t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    M_C_CAE_a_abb += 1/12 * einsum('IK,xyzB,zwuv,Yuvxwy->IKYB', np.identity(ncvs), t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    M_C_CAE_a_abb -= 5/12 * einsum('IK,xyzB,zwuv,Yuvyxw->IKYB', np.identity(ncvs), t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    M_C_CAE_a_abb += 1/12 * einsum('IK,xyzB,zwuv,Yuvywx->IKYB', np.identity(ncvs), t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    M_C_CAE_a_abb += 1/12 * einsum('IK,xyzB,zwuv,Yuvwxy->IKYB', np.identity(ncvs), t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    M_C_CAE_a_abb += 1/12 * einsum('IK,xyzB,zwuv,Yuvwyx->IKYB', np.identity(ncvs), t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    M_C_CAE_a_abb += 1/2 * einsum('IK,xyzB,xywu,Yzuw->IKYB', np.identity(ncvs), t1_aaae, v_aaaa, rdm_ccaa, optimize = einsum_type)
    M_C_CAE_a_abb += 5/12 * einsum('IK,xyzB,xwuv,Yzwyuv->IKYB', np.identity(ncvs), t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    M_C_CAE_a_abb -= 1/12 * einsum('IK,xyzB,xwuv,Yzwyvu->IKYB', np.identity(ncvs), t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    M_C_CAE_a_abb -= 1/12 * einsum('IK,xyzB,xwuv,Yzwuyv->IKYB', np.identity(ncvs), t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    M_C_CAE_a_abb -= 1/12 * einsum('IK,xyzB,xwuv,Yzwuvy->IKYB', np.identity(ncvs), t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    M_C_CAE_a_abb -= 1/12 * einsum('IK,xyzB,xwuv,Yzwvyu->IKYB', np.identity(ncvs), t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    M_C_CAE_a_abb -= 1/12 * einsum('IK,xyzB,xwuv,Yzwvuy->IKYB', np.identity(ncvs), t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    M_C_CAE_a_abb -= 1/12 * einsum('IK,xyzB,ywuv,Yzwxuv->IKYB', np.identity(ncvs), t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    M_C_CAE_a_abb -= 1/12 * einsum('IK,xyzB,ywuv,Yzwxvu->IKYB', np.identity(ncvs), t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    M_C_CAE_a_abb += 5/12 * einsum('IK,xyzB,ywuv,Yzwuxv->IKYB', np.identity(ncvs), t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    M_C_CAE_a_abb -= 1/12 * einsum('IK,xyzB,ywuv,Yzwuvx->IKYB', np.identity(ncvs), t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    M_C_CAE_a_abb -= 1/12 * einsum('IK,xyzB,ywuv,Yzwvxu->IKYB', np.identity(ncvs), t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    M_C_CAE_a_abb -= 1/12 * einsum('IK,xyzB,ywuv,Yzwvux->IKYB', np.identity(ncvs), t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)

    M_C_CAE_a_bab = M_C_CAE_a_aaa - M_C_CAE_a_abb

    ## Reshape tensors to matrix form
    M_C_CAE_a_aaa = M_C_CAE_a_aaa.reshape(ncvs, -1)
    M_C_CAE_a_abb = M_C_CAE_a_abb.reshape(ncvs, -1)
    M_C_CAE_a_bab = M_C_CAE_a_bab.reshape(ncvs, -1)

    ## Building C-CCE matrix
    dim_cae = ncvs * ncas * nextern
    dim_c_cae = 3 * dim_cae

    m_c_cae_aaa_i = 0
    m_c_cae_aaa_f = m_c_cae_aaa_i + dim_cae
    m_c_cae_abb_i = m_c_cae_aaa_f
    m_c_cae_abb_f = m_c_cae_abb_i + dim_cae
    m_c_cae_bab_i = m_c_cae_abb_f
    m_c_cae_bab_f = m_c_cae_bab_i + dim_cae

    M_C_CAE = np.zeros((ncvs, dim_c_cae))
    M_C_CAE[:, m_c_cae_aaa_i:m_c_cae_aaa_f] = M_C_CAE_a_aaa.copy()
    M_C_CAE[:, m_c_cae_abb_i:m_c_cae_abb_f] = M_C_CAE_a_abb.copy()
    M_C_CAE[:, m_c_cae_bab_i:m_c_cae_bab_f] = M_C_CAE_a_bab.copy()

    print("Time for computing M(h0-h1) blocks:               %f sec\n" % (time.time() - start_time))
    sys.stdout.flush()

    shift = 100000.0
    M_C_CAA = shift
    M_C_CCE = shift
    M_C_CCA = shift

    nval = mr_adc.nval
    if nval > 0:
        M_C_CVE = shift
        M_C_VCE = shift
        M_C_CVA = shift
        M_C_VCA = shift

    if nval > 0:
        M_01 = (M_C_CAA, M_C_CCE, M_C_CVE, M_C_CAE, M_C_CCA, M_C_CVA, M_C_VCA)
    else:
        M_01 = (M_C_CAA, M_C_CCE, M_C_CAE, M_C_CCA)

    return M_01

def compute_preconditioner_cae(mr_adc, M_00):

    start_time = time.time()

    # Einsum definition from kernel
    einsum = mr_adc.interface.einsum
    einsum_type = mr_adc.interface.einsum_type

    if mr_adc.method in ("mr-adc(0)", "mr-adc(1)"):

        # Multiply by -1.0, since we are solving for -M C = -S C E
        return (-1.0 * np.diag(M_00))

    # Variables from kernel
    ncvs = mr_adc.ncvs
    nval = mr_adc.nval
    ncas = mr_adc.ncas
    nextern = mr_adc.nextern

    e_cvs = mr_adc.mo_energy.x
    if nval > 0:
        e_val = mr_adc.mo_energy.v
    e_extern = mr_adc.mo_energy.e

    # One-electron integrals
    h_aa = mr_adc.h1eff.aa

    # Two-electrons integrals
    v_aaaa = mr_adc.v2e.aaaa

    # Reduced Density Matrices
    rdm_ca = mr_adc.rdm.ca
    rdm_ccaa = mr_adc.rdm.ccaa

    # Overlap matrices
    S12_c_caa = mr_adc.S12.c_caa
    S12_cae = mr_adc.S12.cae
    S12_cca = mr_adc.S12.cca

    # Dimensions
    ho_s_c = mr_adc.h_orth.s_c
    ho_f_c = mr_adc.h_orth.f_c
    ho_s_cce = mr_adc.h_orth.s_cce
    ho_f_cce = mr_adc.h_orth.f_cce
    ho_s_cae_aaa = mr_adc.h_orth.s_cae_aaa
    ho_f_cae_aaa = mr_adc.h_orth.f_cae_aaa
    ho_s_cae_abb = mr_adc.h_orth.s_cae_abb
    ho_f_cae_abb = mr_adc.h_orth.f_cae_abb
    ho_s_cae_bab = mr_adc.h_orth.s_cae_bab
    ho_f_cae_bab = mr_adc.h_orth.f_cae_bab
    ho_s_cca = mr_adc.h_orth.s_cca
    ho_f_cca = mr_adc.h_orth.f_cca
    if nval > 0:
        ho_s_cve = mr_adc.h_orth.s_cve
        ho_f_cve = mr_adc.h_orth.f_cve
        ho_s_vce = mr_adc.h_orth.s_vce
        ho_f_vce = mr_adc.h_orth.f_vce

        ho_s_cva = mr_adc.h_orth.s_cva
        ho_f_cva = mr_adc.h_orth.f_cva
        ho_s_vca = mr_adc.h_orth.s_vca
        ho_f_vca = mr_adc.h_orth.f_vca

    s_c = mr_adc.h0.s_c
    f_c = mr_adc.h0.f_c

    cvs_tril_ind = np.tril_indices(ncvs, k=-1)
    # cas_ind = np.tril_indices(ncas, k=-1)

    # Build the preconditioner
    precond = np.zeros(mr_adc.h_orth.dim)

    # C-C debug
    precond[ho_s_c:ho_f_c] = np.diag(M_00[s_c:f_c, s_c:f_c]).copy()

    # CAE
    precond_cae =- 1/2 * einsum('A,AA,II,XY->IAXY', e_extern, np.identity(nextern), np.identity(ncvs), rdm_ca, optimize = einsum_type)
    precond_cae += 1/2 * einsum('I,AA,II,XY->IAXY', e_cvs, np.identity(nextern), np.identity(ncvs), rdm_ca, optimize = einsum_type)
    precond_cae += 1/2 * einsum('Xx,AA,II,xY->IAXY', h_aa, np.identity(nextern), np.identity(ncvs), rdm_ca, optimize = einsum_type)
    precond_cae += 1/2 * einsum('Xxyz,AA,II,Yxyz->IAXY', v_aaaa, np.identity(nextern), np.identity(ncvs), rdm_ccaa, optimize = einsum_type)

    precond_cae = einsum("IAXY,XP,YP->IPA", precond_cae, S12_cae, S12_cae, optimize = einsum_type)
    precond[ho_s_cae_aaa:ho_f_cae_aaa] = precond_cae.reshape(-1).copy()
    precond[ho_s_cae_abb:ho_f_cae_abb] = precond_cae.reshape(-1).copy()
    precond[ho_s_cae_bab:ho_f_cae_bab] = precond_cae.reshape(-1).copy()

    # Multiply by -1.0, since we are solving for -M C = -S C E
    precond *= (-1.0)

    print ("Time for computing preconditioner:                %f sec\n" % (time.time() - start_time))
    sys.stdout.flush()

    return precond

def apply_S_12_cae(mr_adc, X, transpose = False):

    # Einsum definition from kernel
    einsum = mr_adc.interface.einsum
    einsum_type = mr_adc.interface.einsum_type

    # Dimensions
    nextern = mr_adc.nextern
    ncvs = mr_adc.ncvs
    nval = mr_adc.nval

    ho_s_c = mr_adc.h_orth.s_c
    ho_f_c = mr_adc.h_orth.f_c
    ho_s_cce = mr_adc.h_orth.s_cce
    ho_f_cce = mr_adc.h_orth.f_cce
    ho_s_cae_aaa = mr_adc.h_orth.s_cae_aaa
    ho_f_cae_aaa = mr_adc.h_orth.f_cae_aaa
    ho_s_cae_abb = mr_adc.h_orth.s_cae_abb
    ho_f_cae_abb = mr_adc.h_orth.f_cae_abb
    ho_s_cae_bab = mr_adc.h_orth.s_cae_bab
    ho_f_cae_bab = mr_adc.h_orth.f_cae_bab
    ho_s_cca = mr_adc.h_orth.s_cca
    ho_f_cca = mr_adc.h_orth.f_cca
    s_c = mr_adc.h0.s_c
    f_c = mr_adc.h0.f_c
    s_cce = mr_adc.h1.s_cce
    f_cce = mr_adc.h1.f_cce
    s_cae_aaa = mr_adc.h1.s_cae_aaa
    f_cae_aaa = mr_adc.h1.f_cae_aaa
    s_cae_abb = mr_adc.h1.s_cae_abb
    f_cae_abb = mr_adc.h1.f_cae_abb
    s_cae_bab = mr_adc.h1.s_cae_bab
    f_cae_bab = mr_adc.h1.f_cae_bab
    s_cca = mr_adc.h1.s_cca
    f_cca = mr_adc.h1.f_cca

    if nval > 0:
        ho_s_cve = mr_adc.h_orth.s_cve
        ho_f_cve = mr_adc.h_orth.f_cve
        ho_s_vce = mr_adc.h_orth.s_vce
        ho_f_vce = mr_adc.h_orth.f_vce

        ho_s_cva = mr_adc.h_orth.s_cva
        ho_f_cva = mr_adc.h_orth.f_cva
        ho_s_vca = mr_adc.h_orth.s_vca
        ho_f_vca = mr_adc.h_orth.f_vca

        s_cve = mr_adc.h1.s_cve
        f_cve = mr_adc.h1.f_cve
        s_vce = mr_adc.h1.s_vce
        f_vce = mr_adc.h1.f_vce

        s_cva = mr_adc.h1.s_cva
        f_cva = mr_adc.h1.f_cva
        s_vca = mr_adc.h1.s_vca
        f_vca = mr_adc.h1.f_vca

    # Overlap matrices
    S12_c_caa = mr_adc.S12.c_caa
    S12_cae = mr_adc.S12.cae
    S12_cca = mr_adc.S12.cca

    Xt = None

    if transpose:
        if (X.shape[0] != (mr_adc.h0.dim + mr_adc.h1.dim)):
            raise Exception("Dimensions do not match when applying S_12 transpose")

        Xt = np.zeros(mr_adc.h_orth.dim)

        # C-C DEBUG
        Xt[ho_s_c:ho_f_c] = X[s_c:f_c].copy()

        # CAE
        temp = X[s_cae_aaa:f_cae_aaa].reshape(ncvs, S12_cae.shape[0], nextern).copy()
        Xt[ho_s_cae_aaa:ho_f_cae_aaa] = einsum("IXA,XP->IPA", temp, S12_cae).reshape(-1).copy()

        temp = X[s_cae_abb:f_cae_abb].reshape(ncvs, S12_cae.shape[0], nextern).copy()
        Xt[ho_s_cae_abb:ho_f_cae_abb] = einsum("IXA,XP->IPA", temp, S12_cae).reshape(-1).copy()

        temp = X[s_cae_bab:f_cae_bab].reshape(ncvs, S12_cae.shape[0], nextern).copy()
        Xt[ho_s_cae_bab:ho_f_cae_bab] = einsum("IXA,XP->IPA", temp, S12_cae).reshape(-1).copy()

    else:
        if (X.shape[0] != (mr_adc.h_orth.dim)):
            raise Exception("Dimensions do not match when applying S_12")

        Xt = np.zeros(mr_adc.h0.dim + mr_adc.h1.dim)

        # C-C DEBUG
        Xt[s_c:f_c] = X[ho_s_c:ho_f_c].copy()

        # CAE
        temp = X[ho_s_cae_aaa:ho_f_cae_aaa].reshape(ncvs, S12_cae.shape[1], nextern).copy()
        Xt[s_cae_aaa:f_cae_aaa] = einsum("IPA,XP->IXA", temp, S12_cae).reshape(-1).copy()

        temp = X[ho_s_cae_abb:ho_f_cae_abb].reshape(ncvs, S12_cae.shape[1], nextern).copy()
        Xt[s_cae_abb:f_cae_abb] = einsum("IPA,XP->IXA", temp, S12_cae).reshape(-1).copy()

        temp = X[ho_s_cae_bab:ho_f_cae_bab].reshape(ncvs, S12_cae.shape[1], nextern).copy()
        Xt[s_cae_bab:f_cae_bab] = einsum("IPA,XP->IXA", temp, S12_cae).reshape(-1).copy()

    return Xt

def compute_sigma_vector_cae(mr_adc, M_00, M_01, M_11, Xt):

    # Einsum definition from kernel
    einsum = mr_adc.interface.einsum
    einsum_type = mr_adc.interface.einsum_type

    # Variables from kernel
    ncvs = mr_adc.ncvs
    nval = mr_adc.nval
    ncas = mr_adc.ncas
    nextern = mr_adc.nextern

    e_cvs = mr_adc.mo_energy.x
    e_core = mr_adc.mo_energy.c
    if nval > 0:
        e_val = mr_adc.mo_energy.v
    e_extern = mr_adc.mo_energy.e

    # One-electron integrals
    h_aa = mr_adc.h1eff.aa

    # Two-electrons integrals
    v_aaaa = mr_adc.v2e.aaaa

    # Reduced Density Matrices
    rdm_ca = mr_adc.rdm.ca
    rdm_ccaa = mr_adc.rdm.ccaa

    # Dimensions
    s_c = mr_adc.h0.s_c
    f_c = mr_adc.h0.f_c
    s_cce = mr_adc.h1.s_cce
    f_cce = mr_adc.h1.f_cce
    s_cae_aaa = mr_adc.h1.s_cae_aaa
    f_cae_aaa = mr_adc.h1.f_cae_aaa
    s_cae_abb = mr_adc.h1.s_cae_abb
    f_cae_abb = mr_adc.h1.f_cae_abb
    s_cae_bab = mr_adc.h1.s_cae_bab
    f_cae_bab = mr_adc.h1.f_cae_bab
    s_cae = mr_adc.h1.s_cae_aaa
    f_cae = mr_adc.h1.f_cae_bab
    if nval > 0:
        s_cve = mr_adc.h1.s_cve_aaa
        f_cve = mr_adc.h1.f_cve_bab

        s_cve_aaa = mr_adc.h1.s_cve_aaa
        f_cve_aaa = mr_adc.h1.f_cve_aaa
        s_cve_abb = mr_adc.h1.s_cve_abb
        f_cve_abb = mr_adc.h1.f_cve_abb
        s_cve_bab = mr_adc.h1.s_cve_bab
        f_cve_bab = mr_adc.h1.f_cve_bab

        s_cva = mr_adc.h1.s_cva
        f_cva = mr_adc.h1.f_cva
        s_vca = mr_adc.h1.s_vca
        f_vca = mr_adc.h1.f_vca

    cvs_tril_ind = np.tril_indices(ncvs, k=-1)

    # (CASCI + C) -> (CASCI + C)
    sigma = np.zeros_like(Xt)

    # h0-h0 contributions
    sigma[:mr_adc.h0.dim] = np.dot(M_00, Xt[:mr_adc.h0.dim])

    # h0-h1 and h1-h0 contributions
    if nval > 0:
        M_C_CAA, M_C_CCE, M_C_CVE, M_C_CAE, M_C_CCA, M_C_CVA, M_C_VCA = M_01
    else:
        M_C_CAA, M_C_CCE, M_C_CAE, M_C_CCA = M_01

    # C <-> CAE
    sigma[s_c:f_c] += np.dot(M_C_CAE, Xt[s_cae:f_cae])
    sigma[s_cae:f_cae] += np.dot(M_C_CAE.T, Xt[s_c:f_c])

    # h1-h1 contributions
    # CAE <- CAE
    X_cae = Xt[s_cae:f_cae].copy()

    dim_cae = ncvs * ncas * nextern

    sigma_aaa_i = 0
    sigma_aaa_f = sigma_aaa_i + dim_cae
    sigma_abb_i = sigma_aaa_f
    sigma_abb_f = sigma_abb_i + dim_cae
    sigma_bab_i = sigma_abb_f
    sigma_bab_f = sigma_bab_i + dim_cae

    X_aaa = X_cae[sigma_aaa_i:sigma_aaa_f].reshape(ncvs, ncas, nextern).copy()
    X_abb = X_cae[sigma_abb_i:sigma_abb_f].reshape(ncvs, ncas, nextern).copy()
    X_bab = X_cae[sigma_bab_i:sigma_bab_f].reshape(ncvs, ncas, nextern).copy()

    sigma_cae_aaa =- 1/2 * einsum('KxB,B,xZ->KZB', X_aaa, e_extern, rdm_ca, optimize = einsum_type)
    sigma_cae_aaa += 1/2 * einsum('KxB,K,xZ->KZB', X_aaa, e_cvs, rdm_ca, optimize = einsum_type)
    sigma_cae_aaa += 1/2 * einsum('KxB,xy,yZ->KZB', X_aaa, h_aa, rdm_ca, optimize = einsum_type)
    sigma_cae_aaa += 1/2 * einsum('KxB,xyzw,Zyzw->KZB', X_aaa, v_aaaa, rdm_ccaa, optimize = einsum_type)

    sigma_cae_abb =- 1/2 * einsum('KxB,B,xZ->KZB', X_abb, e_extern, rdm_ca, optimize = einsum_type)
    sigma_cae_abb += 1/2 * einsum('KxB,K,xZ->KZB', X_abb, e_cvs, rdm_ca, optimize = einsum_type)
    sigma_cae_abb += 1/2 * einsum('KxB,xy,yZ->KZB', X_abb, h_aa, rdm_ca, optimize = einsum_type)
    sigma_cae_abb += 1/2 * einsum('KxB,xyzw,Zyzw->KZB', X_abb, v_aaaa, rdm_ccaa, optimize = einsum_type)

    sigma_cae_bab =- 1/2 * einsum('KxB,B,xZ->KZB', X_bab, e_extern, rdm_ca, optimize = einsum_type)
    sigma_cae_bab += 1/2 * einsum('KxB,K,xZ->KZB', X_bab, e_cvs, rdm_ca, optimize = einsum_type)
    sigma_cae_bab += 1/2 * einsum('KxB,xy,yZ->KZB', X_bab, h_aa, rdm_ca, optimize = einsum_type)
    sigma_cae_bab += 1/2 * einsum('KxB,xyzw,Zyzw->KZB', X_bab, v_aaaa, rdm_ccaa, optimize = einsum_type)

    ## Building C-CAE matrix
    dim_cae = ncvs * ncas * nextern
    dim_sigma_cae = 3 * dim_cae

    sigma_cae_aaa_i = 0
    sigma_cae_aaa_f = sigma_cae_aaa_i + dim_cae
    sigma_cae_abb_i = sigma_cae_aaa_f
    sigma_cae_abb_f = sigma_cae_abb_i + dim_cae
    sigma_cae_bab_i = sigma_cae_abb_f
    sigma_cae_bab_f = sigma_cae_bab_i + dim_cae

    sigma_cae = np.zeros((dim_sigma_cae))
    sigma_cae[sigma_cae_aaa_i:sigma_cae_aaa_f] = sigma_cae_aaa.reshape(-1).copy()
    sigma_cae[sigma_cae_abb_i:sigma_cae_abb_f] = sigma_cae_abb.reshape(-1).copy()
    sigma_cae[sigma_cae_bab_i:sigma_cae_bab_f] = sigma_cae_bab.reshape(-1).copy()

    sigma[s_cae:f_cae] += sigma_cae.reshape(-1).copy()

    return sigma

## CAA block: Diagonal + M01 (v1)
def compute_excitation_manifolds_caa(mr_adc):

    # MR-ADC(0) and MR-ADC(1)
    mr_adc.h0.n_c = mr_adc.ncvs
    mr_adc.h0.dim = mr_adc.h0.n_c # Total dimension of h0

    mr_adc.h0.s_c = 0
    mr_adc.h0.f_c = mr_adc.h0.s_c + mr_adc.h0.n_c

    print("Dimension of h0 excitation manifold:                       %d" % mr_adc.h0.dim)

    # MR-ADC(2)
    mr_adc.h1.dim = 0
    mr_adc.h_orth.dim = mr_adc.h0.dim

    if mr_adc.method in ("mr-adc(2)", "mr-adc(2)-x"):
        # mr_adc.h1.n_caa = 2 * mr_adc.ncas * mr_adc.ncas * mr_adc.ncvs
        mr_adc.h1.n_caa = 3 * mr_adc.ncas * mr_adc.ncas * mr_adc.ncvs
        mr_adc.h1.n_cce = 0
        mr_adc.h1.n_cae = 0
        mr_adc.h1.n_ace = 0
        mr_adc.h1.n_cca = 0
        if mr_adc.nval > 0:
            mr_adc.h1.n_cve = 0
            mr_adc.h1.n_vce = 0
            mr_adc.h1.n_cva = 0
            mr_adc.h1.n_vca = 0
            mr_adc.h1.dim = (mr_adc.h1.n_caa + mr_adc.h1.n_cce + mr_adc.h1.n_cve + mr_adc.h1.n_vce +
                             mr_adc.h1.n_cae + mr_adc.h1.n_ace + mr_adc.h1.n_cca + mr_adc.h1.n_cva + mr_adc.h1.n_vca)
        else:
            mr_adc.h1.dim = mr_adc.h1.n_caa + mr_adc.h1.n_cce + mr_adc.h1.n_cae + mr_adc.h1.n_cae + mr_adc.h1.n_cca

        if mr_adc.nval > 0:
            mr_adc.h1.s_caa = mr_adc.h0.f_c
            mr_adc.h1.f_caa = mr_adc.h1.s_caa + mr_adc.h1.n_caa
            mr_adc.h1.s_cce = mr_adc.h1.f_caa
            mr_adc.h1.f_cce = mr_adc.h1.s_cce + mr_adc.h1.n_cce
            mr_adc.h1.s_cve = mr_adc.h1.f_cce
            mr_adc.h1.f_cve = mr_adc.h1.s_cve + mr_adc.h1.n_cve
            mr_adc.h1.s_vce = mr_adc.h1.f_cve
            mr_adc.h1.f_vce = mr_adc.h1.s_vce + mr_adc.h1.n_vce
            mr_adc.h1.s_cae = mr_adc.h1.f_vce
            mr_adc.h1.f_cae = mr_adc.h1.s_cae + mr_adc.h1.n_cae
            mr_adc.h1.s_ace = mr_adc.h1.f_cae
            mr_adc.h1.f_ace = mr_adc.h1.s_ace + mr_adc.h1.n_ace
            mr_adc.h1.s_cca = mr_adc.h1.f_ace
            mr_adc.h1.f_cca = mr_adc.h1.s_cca + mr_adc.h1.n_cca
            mr_adc.h1.s_cva = mr_adc.h1.f_cca
            mr_adc.h1.f_cva = mr_adc.h1.s_cva + mr_adc.h1.n_cva
            mr_adc.h1.s_vca = mr_adc.h1.f_cva
            mr_adc.h1.f_vca = mr_adc.h1.s_vca + mr_adc.h1.n_vca
        else:
            mr_adc.h1.s_caa = mr_adc.h0.f_c
            mr_adc.h1.f_caa = mr_adc.h1.s_caa + mr_adc.h1.n_caa
            mr_adc.h1.s_cce = mr_adc.h1.f_caa
            mr_adc.h1.f_cce = mr_adc.h1.s_cce + mr_adc.h1.n_cce
            mr_adc.h1.s_cae = mr_adc.h1.f_cce
            mr_adc.h1.f_cae = mr_adc.h1.s_cae + mr_adc.h1.n_cae
            mr_adc.h1.s_ace = mr_adc.h1.f_cae
            mr_adc.h1.f_ace = mr_adc.h1.s_ace + mr_adc.h1.n_ace
            mr_adc.h1.s_cca = mr_adc.h1.f_ace
            mr_adc.h1.f_cca = mr_adc.h1.s_cca + mr_adc.h1.n_cca

        print("Dimension of h1 excitation manifold:                       %d" % mr_adc.h1.dim)

        # Overlap for c - caa
        mr_adc.S12.c_caa = mr_adc_overlap.compute_S12_0p_projector(mr_adc)
        mr_adc.S12.cae = mr_adc_overlap.compute_S12_m1(mr_adc)
        mr_adc.S12.cca = mr_adc_overlap.compute_S12_p1(mr_adc)

        # Determine dimensions of orthogonalized excitation spaces
        mr_adc.h_orth.n_c = 0
        mr_adc.h_orth.n_c_caa = mr_adc.ncvs * mr_adc.S12.c_caa.shape[1]
        mr_adc.h_orth.n_cce = 0
        mr_adc.h_orth.n_cce = 0
        mr_adc.h_orth.n_cae = 0
        mr_adc.h_orth.n_ace = 0
        mr_adc.h_orth.n_cca = 0
        if mr_adc.nval > 0:
            mr_adc.h_orth.n_cve = 0
            mr_adc.h_orth.n_vce = 0
            mr_adc.h_orth.n_cva = 0
            mr_adc.h_orth.n_vca = 0
            mr_adc.h_orth.dim = (mr_adc.h_orth.n_c + mr_adc.h_orth.n_c_caa + mr_adc.h_orth.n_cce + mr_adc.h_orth.n_cve + mr_adc.h_orth.n_vce +
                                 mr_adc.h_orth.n_cae + mr_adc.h_orth.n_ace + mr_adc.h_orth.n_cca + mr_adc.h_orth.n_cva + mr_adc.h_orth.n_vca)
        else:
            mr_adc.h_orth.dim = mr_adc.h_orth.n_c + mr_adc.h_orth.n_c_caa + mr_adc.h_orth.n_cce + mr_adc.h_orth.n_cae + mr_adc.h_orth.n_ace + mr_adc.h_orth.n_cca

        if mr_adc.nval > 0:
            mr_adc.h_orth.s_c = 0
            mr_adc.h_orth.f_c = mr_adc.h_orth.n_c
            mr_adc.h_orth.s_c_caa = mr_adc.h_orth.f_c
            mr_adc.h_orth.f_c_caa = mr_adc.h_orth.s_c_caa + mr_adc.h_orth.n_c_caa
            mr_adc.h_orth.s_cce = mr_adc.h_orth.f_c_caa
            mr_adc.h_orth.f_cce = mr_adc.h_orth.s_cce + mr_adc.h_orth.n_cce
            mr_adc.h_orth.s_cve = mr_adc.h_orth.f_cce
            mr_adc.h_orth.f_cve = mr_adc.h_orth.s_cve + mr_adc.h_orth.n_cve
            mr_adc.h_orth.s_vce = mr_adc.h_orth.f_cve
            mr_adc.h_orth.f_vce = mr_adc.h_orth.s_vce + mr_adc.h_orth.n_vce
            mr_adc.h_orth.s_cae = mr_adc.h_orth.f_vce
            mr_adc.h_orth.f_cae = mr_adc.h_orth.s_cae + mr_adc.h_orth.n_cae
            mr_adc.h_orth.s_ace = mr_adc.h_orth.f_cae
            mr_adc.h_orth.f_ace = mr_adc.h_orth.s_ace + mr_adc.h_orth.n_ace
            mr_adc.h_orth.s_cca = mr_adc.h_orth.f_ace
            mr_adc.h_orth.f_cca = mr_adc.h_orth.s_cca + mr_adc.h_orth.n_cca
            mr_adc.h_orth.s_cva = mr_adc.h_orth.f_cca
            mr_adc.h_orth.f_cva = mr_adc.h_orth.s_cva + mr_adc.h_orth.n_cva
            mr_adc.h_orth.s_vca = mr_adc.h_orth.f_cva
            mr_adc.h_orth.f_vca = mr_adc.h_orth.s_vca + mr_adc.h_orth.n_vca
        else:
            mr_adc.h_orth.s_c = 0
            mr_adc.h_orth.f_c = mr_adc.h_orth.n_c
            mr_adc.h_orth.s_c_caa = mr_adc.h_orth.f_c
            mr_adc.h_orth.f_c_caa = mr_adc.h_orth.s_c_caa + mr_adc.h_orth.n_c_caa
            mr_adc.h_orth.s_cce = mr_adc.h_orth.f_c_caa
            mr_adc.h_orth.f_cce = mr_adc.h_orth.s_cce + mr_adc.h_orth.n_cce
            mr_adc.h_orth.s_cae = mr_adc.h_orth.f_cce
            mr_adc.h_orth.f_cae = mr_adc.h_orth.s_cae + mr_adc.h_orth.n_cae
            mr_adc.h_orth.s_ace = mr_adc.h_orth.f_cae
            mr_adc.h_orth.f_ace = mr_adc.h_orth.s_ace + mr_adc.h_orth.n_ace
            mr_adc.h_orth.s_cca = mr_adc.h_orth.f_ace
            mr_adc.h_orth.f_cca = mr_adc.h_orth.s_cca + mr_adc.h_orth.n_cca

    print("Total dimension of the excitation manifold:                %d" % (mr_adc.h0.dim + mr_adc.h1.dim))
    print("Dimension of the orthogonalized excitation manifold:       %d\n" % (mr_adc.h_orth.dim))
    sys.stdout.flush()

    if (mr_adc.h_orth.dim < mr_adc.nroots):
        mr_adc.nroots = mr_adc.h_orth.dim

    return mr_adc

def compute_M_01_caa(mr_adc):

    start_time = time.time()

    print ("Computing M(h0-h1) blocks...")
    sys.stdout.flush()

    # Einsum definition from kernel
    einsum = mr_adc.interface.einsum
    einsum_type = mr_adc.interface.einsum_type

    # Dimensions
    ncore = mr_adc.ncore
    ncas = mr_adc.ncas
    nocc = mr_adc.nocc
    nextern = mr_adc.nextern

    ncvs = mr_adc.ncvs
    nval = mr_adc.nval

    n_caa = mr_adc.h1.n_caa
    n_cce = mr_adc.h1.n_cce
    n_cae = mr_adc.h1.n_cae
    n_cca = mr_adc.h1.n_cca
    if nval > 0:
        n_cve = mr_adc.h1.n_cve
        n_cva = mr_adc.h1.n_cva

    cvs_tril_ind = np.tril_indices(ncvs, k=-1)

    # MOs Energy
    e_cvs = mr_adc.mo_energy.x
    e_core = mr_adc.mo_energy.c
    e_extern = mr_adc.mo_energy.e
    if nval > 0:
        e_val = mr_adc.mo_energy.v

    # Amplitudes
    t1_ce = mr_adc.t1.ce
    t1_ca = mr_adc.t1.ca
    t1_ae = mr_adc.t1.ae
    t1_caea = mr_adc.t1.caea
    t1_caae = mr_adc.t1.caae
    t1_caaa = mr_adc.t1.caaa
    t1_aaea = mr_adc.t1.aaea

    t1_xe = mr_adc.t1.xe
    t1_xaea = mr_adc.t1.xaea
    t1_xaae = mr_adc.t1.xaae

    if nval > 0:
        t1_ve = mr_adc.t1.ve
        t1_vaea = mr_adc.t1.vaea
        t1_vaae = mr_adc.t1.vaae

    # One-electron integrals
    h_aa = mr_adc.h1eff.aa

    h_xe = mr_adc.h1eff.xe

    if nval > 0:
        h_ve = mr_adc.h1eff.ve

    # Two-electrons integrals
    v_aaaa = mr_adc.v2e.aaaa

    v_xaxa = mr_adc.v2e.xaxa
    v_xaax = mr_adc.v2e.xaax

    v_xaea = mr_adc.v2e.xaea
    v_xaae = mr_adc.v2e.xaae
    v_xxxe = mr_adc.v2e.xxxe

    if nval > 0:
        v_vxxe = mr_adc.v2e.vxxe
        v_xvxe = mr_adc.v2e.xvxe

        v_vaea = mr_adc.v2e.vaea
        v_vaae = mr_adc.v2e.vaae

    # Reduced Density Matrices
    rdm_ca = mr_adc.rdm.ca
    rdm_ccaa = mr_adc.rdm.ccaa
    rdm_cccaaa = mr_adc.rdm.cccaaa

    # C - CAA
    # Oth-order
    M_C_CAA_a_aaa  = 1/2 * einsum('J,IJ,WZ->IJWZ', e_cvs, np.identity(ncvs), rdm_ca, optimize = einsum_type)
    M_C_CAA_a_aaa += 1/2 * einsum('Wx,IJ,xZ->IJWZ', h_aa, np.identity(ncvs), rdm_ca, optimize = einsum_type)
    M_C_CAA_a_aaa -= 1/2 * einsum('Zx,IJ,Wx->IJWZ', h_aa, np.identity(ncvs), rdm_ca, optimize = einsum_type)
    M_C_CAA_a_aaa += 1/2 * einsum('IJ,Wxyz,Zxyz->IJWZ', np.identity(ncvs), v_aaaa, rdm_ccaa, optimize = einsum_type)
    M_C_CAA_a_aaa -= 1/2 * einsum('IJ,Zxyz,Wxyz->IJWZ', np.identity(ncvs), v_aaaa, rdm_ccaa, optimize = einsum_type)

    M_C_CAA_a_abb  = 1/2 * einsum('J,IJ,WZ->IJWZ', e_cvs, np.identity(ncvs), rdm_ca, optimize = einsum_type)
    M_C_CAA_a_abb += 1/2 * einsum('Wx,IJ,xZ->IJWZ', h_aa, np.identity(ncvs), rdm_ca, optimize = einsum_type)
    M_C_CAA_a_abb -= 1/2 * einsum('Zx,IJ,Wx->IJWZ', h_aa, np.identity(ncvs), rdm_ca, optimize = einsum_type)
    M_C_CAA_a_abb += 1/2 * einsum('IJ,Wxyz,Zxyz->IJWZ', np.identity(ncvs), v_aaaa, rdm_ccaa, optimize = einsum_type)
    M_C_CAA_a_abb -= 1/2 * einsum('IJ,Zxyz,Wxyz->IJWZ', np.identity(ncvs), v_aaaa, rdm_ccaa, optimize = einsum_type)

    # 1st-order
    M_C_CAA_a_aaa += 1/2 * einsum('IZJx,Wx->IJWZ', v_xaxa, rdm_ca, optimize = einsum_type)
    M_C_CAA_a_aaa -= 1/2 * einsum('IZxJ,Wx->IJWZ', v_xaax, rdm_ca, optimize = einsum_type)
    M_C_CAA_a_aaa += 1/2 * einsum('IxJy,WxZy->IJWZ', v_xaxa, rdm_ccaa, optimize = einsum_type)
    M_C_CAA_a_aaa -= 1/6 * einsum('IxyJ,WxZy->IJWZ', v_xaax, rdm_ccaa, optimize = einsum_type)
    M_C_CAA_a_aaa += 1/6 * einsum('IxyJ,WxyZ->IJWZ', v_xaax, rdm_ccaa, optimize = einsum_type)
    M_C_CAA_a_aaa -= 1/2 * einsum('IxJy,yx,WZ->IJWZ', v_xaxa, rdm_ca, rdm_ca, optimize = einsum_type)
    M_C_CAA_a_aaa += 1/4 * einsum('IxyJ,yx,WZ->IJWZ', v_xaax, rdm_ca, rdm_ca, optimize = einsum_type)

    M_C_CAA_a_abb += 1/2 * einsum('IZJx,Wx->IJWZ', v_xaxa, rdm_ca, optimize = einsum_type)
    M_C_CAA_a_abb += 1/2 * einsum('IxJy,WxZy->IJWZ', v_xaxa, rdm_ccaa, optimize = einsum_type)
    M_C_CAA_a_abb -= 1/3 * einsum('IxyJ,WxZy->IJWZ', v_xaax, rdm_ccaa, optimize = einsum_type)
    M_C_CAA_a_abb -= 1/6 * einsum('IxyJ,WxyZ->IJWZ', v_xaax, rdm_ccaa, optimize = einsum_type)
    M_C_CAA_a_abb -= 1/2 * einsum('IxJy,yx,WZ->IJWZ', v_xaxa, rdm_ca, rdm_ca, optimize = einsum_type)
    M_C_CAA_a_abb += 1/4 * einsum('IxyJ,yx,WZ->IJWZ', v_xaax, rdm_ca, rdm_ca, optimize = einsum_type)

    M_C_CAA_a_bab =- 1/2 * einsum('IZxJ,Wx->IJWZ', v_xaax, rdm_ca, optimize = einsum_type)
    M_C_CAA_a_bab += 1/6 * einsum('IxyJ,WxZy->IJWZ', v_xaax, rdm_ccaa, optimize = einsum_type)
    M_C_CAA_a_bab += 1/3 * einsum('IxyJ,WxyZ->IJWZ', v_xaax, rdm_ccaa, optimize = einsum_type)

    M_C_CAA_a_aaa = M_C_CAA_a_aaa.reshape(ncvs, -1)
    M_C_CAA_a_abb = M_C_CAA_a_abb.reshape(ncvs, -1)
    M_C_CAA_a_bab = M_C_CAA_a_bab.reshape(ncvs, -1)

    ## Building C-CAA matrix
    dim_caa = ncvs * ncas * ncas
    dim_c_caa = 3 * dim_caa

    m_c_caa_aa_i = 0
    m_c_caa_aa_f = m_c_caa_aa_i + dim_caa
    m_c_caa_bb_i = m_c_caa_aa_f
    m_c_caa_bb_f = m_c_caa_bb_i + dim_caa
    m_c_caa_ab_i = m_c_caa_bb_f
    m_c_caa_ab_f = m_c_caa_ab_i + dim_caa

    M_C_CAA = np.zeros((ncvs, dim_c_caa))
    M_C_CAA[:, m_c_caa_aa_i:m_c_caa_aa_f] = M_C_CAA_a_aaa.reshape(ncvs, -1)
    M_C_CAA[:, m_c_caa_bb_i:m_c_caa_bb_f] = M_C_CAA_a_abb.reshape(ncvs, -1)
    M_C_CAA[:, m_c_caa_ab_i:m_c_caa_ab_f] = M_C_CAA_a_bab.reshape(ncvs, -1)

    print("Time for computing M(h0-h1) blocks:               %f sec\n" % (time.time() - start_time))
    sys.stdout.flush()

    shift = 100000.0
    M_C_CCE = shift
    M_C_CAE = shift
    M_C_CCA = shift

    nval = mr_adc.nval
    if nval > 0:
        M_C_CVE = shift
        M_C_CVA = shift

    if nval > 0:
        M_01 = (M_C_CAA, M_C_CCE, M_C_CVE, M_C_CAE, M_C_CCA, M_C_CVA)
    else:
        M_01 = (M_C_CAA, M_C_CCE, M_C_CAE, M_C_CCA)

    return M_01

def compute_preconditioner_caa(mr_adc, M_00):

    start_time = time.time()

    # Einsum definition from kernel
    einsum = mr_adc.interface.einsum
    einsum_type = mr_adc.interface.einsum_type

    if mr_adc.method in ("mr-adc(0)", "mr-adc(1)"):

        # Multiply by -1.0, since we are solving for -M C = -S C E
        return (-1.0 * np.diag(M_00))

    # Variables from kernel
    ncvs = mr_adc.ncvs
    nval = mr_adc.nval
    ncas = mr_adc.ncas
    nextern = mr_adc.nextern

    e_cvs = mr_adc.mo_energy.x
    e_extern = mr_adc.mo_energy.e

    if nval > 0:
        e_val = mr_adc.mo_energy.v

    # One-electron integrals
    h_aa = mr_adc.h1eff.aa

    # Two-electrons integrals
    v_aaaa = mr_adc.v2e.aaaa

    v_xaxa = mr_adc.v2e.xaxa
    v_xaax = mr_adc.v2e.xaax

    # Reduced Density Matrices
    rdm_ca = mr_adc.rdm.ca
    rdm_ccaa = mr_adc.rdm.ccaa
    rdm_cccaaa = mr_adc.rdm.cccaaa

    # Overlap matrices
    S12_c_caa = mr_adc.S12.c_caa
    S12_cae = mr_adc.S12.cae
    S12_cca = mr_adc.S12.cca

    # Dimensions
    ho_s_c = mr_adc.h_orth.s_c
    ho_f_c = mr_adc.h_orth.f_c
    ho_s_c_caa = mr_adc.h_orth.s_c_caa
    ho_f_c_caa = mr_adc.h_orth.f_c_caa
    ho_s_cce = mr_adc.h_orth.s_cce
    ho_f_cce = mr_adc.h_orth.f_cce
    ho_s_cae = mr_adc.h_orth.s_cae
    ho_f_cae = mr_adc.h_orth.f_cae
    ho_s_ace = mr_adc.h_orth.s_ace
    ho_f_ace = mr_adc.h_orth.f_ace
    ho_s_cca = mr_adc.h_orth.s_cca
    ho_f_cca = mr_adc.h_orth.f_cca
    if nval > 0:
        ho_s_cve = mr_adc.h_orth.s_cve
        ho_f_cve = mr_adc.h_orth.f_cve
        ho_s_vce = mr_adc.h_orth.s_vce
        ho_f_vce = mr_adc.h_orth.f_vce

        ho_s_cva = mr_adc.h_orth.s_cva
        ho_f_cva = mr_adc.h_orth.f_cva
        ho_s_vca = mr_adc.h_orth.s_vca
        ho_f_vca = mr_adc.h_orth.f_vca

    s_c = mr_adc.h0.s_c
    f_c = mr_adc.h0.f_c

    cvs_tril_ind = np.tril_indices(ncvs, k=-1)

    # Build the preconditioner
    precond = np.zeros(mr_adc.h_orth.dim)

    # C and CAA
    # 0th-order
    precond_c_caa_a_aaa  = 1/2 * einsum('I,II,XY->IXY', e_cvs, np.identity(ncvs), rdm_ca, optimize = einsum_type)

    precond_c_caa_a_abb  = 1/2 * einsum('I,II,XY->IXY', e_cvs, np.identity(ncvs), rdm_ca, optimize = einsum_type)

    # 1st-order
    precond_c_caa_a_aaa += 1/2 * einsum('IxIY,Xx->IXY', v_xaxa, rdm_ca, optimize = einsum_type)
    precond_c_caa_a_aaa += 1/2 * einsum('IxIy,XyYx->IXY', v_xaxa, rdm_ccaa, optimize = einsum_type)
    precond_c_caa_a_aaa -= 1/2 * einsum('IxYI,Xx->IXY', v_xaax, rdm_ca, optimize = einsum_type)
    precond_c_caa_a_aaa -= 1/6 * einsum('IxyI,XyYx->IXY', v_xaax, rdm_ccaa, optimize = einsum_type)
    precond_c_caa_a_aaa += 1/6 * einsum('IxyI,XyxY->IXY', v_xaax, rdm_ccaa, optimize = einsum_type)
    precond_c_caa_a_aaa -= 1/2 * einsum('IxIy,xy,XY->IXY', v_xaxa, rdm_ca, rdm_ca, optimize = einsum_type)
    precond_c_caa_a_aaa += 1/4 * einsum('IxyI,xy,XY->IXY', v_xaax, rdm_ca, rdm_ca, optimize = einsum_type)

    precond_c_caa_a_abb += 1/2 * einsum('IxIY,Xx->IXY', v_xaxa, rdm_ca, optimize = einsum_type)
    precond_c_caa_a_abb += 1/2 * einsum('IxIy,XyYx->IXY', v_xaxa, rdm_ccaa, optimize = einsum_type)
    precond_c_caa_a_abb -= 1/3 * einsum('IxyI,XyYx->IXY', v_xaax, rdm_ccaa, optimize = einsum_type)
    precond_c_caa_a_abb -= 1/6 * einsum('IxyI,XyxY->IXY', v_xaax, rdm_ccaa, optimize = einsum_type)
    precond_c_caa_a_abb -= 1/2 * einsum('IxIy,xy,XY->IXY', v_xaxa, rdm_ca, rdm_ca, optimize = einsum_type)
    precond_c_caa_a_abb += 1/4 * einsum('IxyI,xy,XY->IXY', v_xaax, rdm_ca, rdm_ca, optimize = einsum_type)

    precond_caa_caa_aaa_aaa =- 1/6 * einsum('I,II,WYXZ->IWZXY', e_cvs, np.identity(ncvs), rdm_ccaa, optimize = einsum_type)
    precond_caa_caa_aaa_aaa += 1/6 * einsum('I,II,WYZX->IWZXY', e_cvs, np.identity(ncvs), rdm_ccaa, optimize = einsum_type)
    precond_caa_caa_aaa_aaa += 1/6 * einsum('Xx,II,WYZx->IWZXY', h_aa, np.identity(ncvs), rdm_ccaa, optimize = einsum_type)
    precond_caa_caa_aaa_aaa -= 1/6 * einsum('Xx,II,WYxZ->IWZXY', h_aa, np.identity(ncvs), rdm_ccaa, optimize = einsum_type)
    precond_caa_caa_aaa_aaa -= 1/2 * einsum('YZ,II,XW->IWZXY', h_aa, np.identity(ncvs), rdm_ca, optimize = einsum_type)
    precond_caa_caa_aaa_aaa += 1/6 * einsum('Yx,II,WxXZ->IWZXY', h_aa, np.identity(ncvs), rdm_ccaa, optimize = einsum_type)
    precond_caa_caa_aaa_aaa -= 1/6 * einsum('Yx,II,WxZX->IWZXY', h_aa, np.identity(ncvs), rdm_ccaa, optimize = einsum_type)
    precond_caa_caa_aaa_aaa -= 1/6 * einsum('II,XZxy,WYxy->IWZXY', np.identity(ncvs), v_aaaa, rdm_ccaa, optimize = einsum_type)
    precond_caa_caa_aaa_aaa += 1/6 * einsum('II,XZxy,WYyx->IWZXY', np.identity(ncvs), v_aaaa, rdm_ccaa, optimize = einsum_type)
    precond_caa_caa_aaa_aaa += 1/6 * einsum('II,Xxyz,ZyzWYx->IWZXY', np.identity(ncvs), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    precond_caa_caa_aaa_aaa -= 1/6 * einsum('II,Xxyz,ZyzYWx->IWZXY', np.identity(ncvs), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    precond_caa_caa_aaa_aaa -= 1/2 * einsum('II,YxZy,WyXx->IWZXY', np.identity(ncvs), v_aaaa, rdm_ccaa, optimize = einsum_type)
    precond_caa_caa_aaa_aaa += 1/6 * einsum('II,YxyZ,WyXx->IWZXY', np.identity(ncvs), v_aaaa, rdm_ccaa, optimize = einsum_type)
    precond_caa_caa_aaa_aaa -= 1/6 * einsum('II,YxyZ,WyxX->IWZXY', np.identity(ncvs), v_aaaa, rdm_ccaa, optimize = einsum_type)
    precond_caa_caa_aaa_aaa += 1/6 * einsum('II,Yxyz,XZxWyz->IWZXY', np.identity(ncvs), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    precond_caa_caa_aaa_aaa -= 1/6 * einsum('II,Yxyz,XZxyWz->IWZXY', np.identity(ncvs), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    precond_caa_caa_aaa_aaa += 1/2 * einsum('I,II,YZ,XW->IWZXY', e_cvs, np.identity(ncvs), np.identity(ncas), rdm_ca, optimize = einsum_type)
    precond_caa_caa_aaa_aaa += 1/2 * einsum('Xx,II,YZ,xW->IWZXY', h_aa, np.identity(ncvs), np.identity(ncas), rdm_ca, optimize = einsum_type)
    precond_caa_caa_aaa_aaa += 1/2 * einsum('Xxyz,II,YZ,Wxyz->IWZXY', v_aaaa, np.identity(ncvs), np.identity(ncas), rdm_ccaa, optimize = einsum_type)

    precond_caa_caa_abb_abb =- 1/6 * einsum('I,II,WYXZ->IWZXY', e_cvs, np.identity(ncvs), rdm_ccaa, optimize = einsum_type)
    precond_caa_caa_abb_abb += 1/6 * einsum('I,II,WYZX->IWZXY', e_cvs, np.identity(ncvs), rdm_ccaa, optimize = einsum_type)
    precond_caa_caa_abb_abb += 1/6 * einsum('Xx,II,WYZx->IWZXY', h_aa, np.identity(ncvs), rdm_ccaa, optimize = einsum_type)
    precond_caa_caa_abb_abb -= 1/6 * einsum('Xx,II,WYxZ->IWZXY', h_aa, np.identity(ncvs), rdm_ccaa, optimize = einsum_type)
    precond_caa_caa_abb_abb -= 1/2 * einsum('YZ,II,XW->IWZXY', h_aa, np.identity(ncvs), rdm_ca, optimize = einsum_type)
    precond_caa_caa_abb_abb += 1/6 * einsum('Yx,II,WxXZ->IWZXY', h_aa, np.identity(ncvs), rdm_ccaa, optimize = einsum_type)
    precond_caa_caa_abb_abb -= 1/6 * einsum('Yx,II,WxZX->IWZXY', h_aa, np.identity(ncvs), rdm_ccaa, optimize = einsum_type)
    precond_caa_caa_abb_abb -= 1/6 * einsum('II,XZxy,WYxy->IWZXY', np.identity(ncvs), v_aaaa, rdm_ccaa, optimize = einsum_type)
    precond_caa_caa_abb_abb += 1/6 * einsum('II,XZxy,WYyx->IWZXY', np.identity(ncvs), v_aaaa, rdm_ccaa, optimize = einsum_type)
    precond_caa_caa_abb_abb += 1/6 * einsum('II,Xxyz,ZyzWYx->IWZXY', np.identity(ncvs), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    precond_caa_caa_abb_abb -= 1/6 * einsum('II,Xxyz,ZyzYWx->IWZXY', np.identity(ncvs), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    precond_caa_caa_abb_abb -= 1/2 * einsum('II,YxZy,WyXx->IWZXY', np.identity(ncvs), v_aaaa, rdm_ccaa, optimize = einsum_type)
    precond_caa_caa_abb_abb += 1/6 * einsum('II,YxyZ,WyXx->IWZXY', np.identity(ncvs), v_aaaa, rdm_ccaa, optimize = einsum_type)
    precond_caa_caa_abb_abb -= 1/6 * einsum('II,YxyZ,WyxX->IWZXY', np.identity(ncvs), v_aaaa, rdm_ccaa, optimize = einsum_type)
    precond_caa_caa_abb_abb += 1/6 * einsum('II,Yxyz,XZxWyz->IWZXY', np.identity(ncvs), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    precond_caa_caa_abb_abb -= 1/6 * einsum('II,Yxyz,XZxyWz->IWZXY', np.identity(ncvs), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    precond_caa_caa_abb_abb += 1/2 * einsum('I,II,YZ,XW->IWZXY', e_cvs, np.identity(ncvs), np.identity(ncas), rdm_ca, optimize = einsum_type)
    precond_caa_caa_abb_abb += 1/2 * einsum('Xx,II,YZ,xW->IWZXY', h_aa, np.identity(ncvs), np.identity(ncas), rdm_ca, optimize = einsum_type)
    precond_caa_caa_abb_abb += 1/2 * einsum('Xxyz,II,YZ,Wxyz->IWZXY', v_aaaa, np.identity(ncvs), np.identity(ncas), rdm_ccaa, optimize = einsum_type)

    precond_caa_caa_bab_bab =- 1/3 * einsum('I,II,WYXZ->IWZXY', e_cvs, np.identity(ncvs), rdm_ccaa, optimize = einsum_type)
    precond_caa_caa_bab_bab -= 1/6 * einsum('I,II,WYZX->IWZXY', e_cvs, np.identity(ncvs), rdm_ccaa, optimize = einsum_type)
    precond_caa_caa_bab_bab -= 1/6 * einsum('Xx,II,WYZx->IWZXY', h_aa, np.identity(ncvs), rdm_ccaa, optimize = einsum_type)
    precond_caa_caa_bab_bab -= 1/3 * einsum('Xx,II,WYxZ->IWZXY', h_aa, np.identity(ncvs), rdm_ccaa, optimize = einsum_type)
    precond_caa_caa_bab_bab -= 1/2 * einsum('YZ,II,XW->IWZXY', h_aa, np.identity(ncvs), rdm_ca, optimize = einsum_type)
    precond_caa_caa_bab_bab += 1/3 * einsum('Yx,II,WxXZ->IWZXY', h_aa, np.identity(ncvs), rdm_ccaa, optimize = einsum_type)
    precond_caa_caa_bab_bab += 1/6 * einsum('Yx,II,WxZX->IWZXY', h_aa, np.identity(ncvs), rdm_ccaa, optimize = einsum_type)
    precond_caa_caa_bab_bab -= 1/3 * einsum('II,XZxy,WYxy->IWZXY', np.identity(ncvs), v_aaaa, rdm_ccaa, optimize = einsum_type)
    precond_caa_caa_bab_bab -= 1/6 * einsum('II,XZxy,WYyx->IWZXY', np.identity(ncvs), v_aaaa, rdm_ccaa, optimize = einsum_type)
    precond_caa_caa_bab_bab -= 1/12 * einsum('II,Xxyz,ZyzWYx->IWZXY', np.identity(ncvs), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    precond_caa_caa_bab_bab += 1/12 * einsum('II,Xxyz,ZyzWxY->IWZXY', np.identity(ncvs), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    precond_caa_caa_bab_bab -= 1/4 * einsum('II,Xxyz,ZyzYWx->IWZXY', np.identity(ncvs), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    precond_caa_caa_bab_bab += 1/12 * einsum('II,Xxyz,ZyzYxW->IWZXY', np.identity(ncvs), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    precond_caa_caa_bab_bab += 1/12 * einsum('II,Xxyz,ZyzxWY->IWZXY', np.identity(ncvs), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    precond_caa_caa_bab_bab += 1/12 * einsum('II,Xxyz,ZyzxYW->IWZXY', np.identity(ncvs), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    precond_caa_caa_bab_bab -= 1/2 * einsum('II,YxZy,WyXx->IWZXY', np.identity(ncvs), v_aaaa, rdm_ccaa, optimize = einsum_type)
    precond_caa_caa_bab_bab += 1/3 * einsum('II,YxyZ,WyXx->IWZXY', np.identity(ncvs), v_aaaa, rdm_ccaa, optimize = einsum_type)
    precond_caa_caa_bab_bab += 1/6 * einsum('II,YxyZ,WyxX->IWZXY', np.identity(ncvs), v_aaaa, rdm_ccaa, optimize = einsum_type)
    precond_caa_caa_bab_bab += 1/4 * einsum('II,Yxyz,XZxWyz->IWZXY', np.identity(ncvs), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    precond_caa_caa_bab_bab -= 1/12 * einsum('II,Yxyz,XZxWzy->IWZXY', np.identity(ncvs), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    precond_caa_caa_bab_bab += 1/12 * einsum('II,Yxyz,XZxyWz->IWZXY', np.identity(ncvs), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    precond_caa_caa_bab_bab -= 1/12 * einsum('II,Yxyz,XZxyzW->IWZXY', np.identity(ncvs), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    precond_caa_caa_bab_bab -= 1/12 * einsum('II,Yxyz,XZxzWy->IWZXY', np.identity(ncvs), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    precond_caa_caa_bab_bab -= 1/12 * einsum('II,Yxyz,XZxzyW->IWZXY', np.identity(ncvs), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    precond_caa_caa_bab_bab += 1/2 * einsum('I,II,YZ,XW->IWZXY', e_cvs, np.identity(ncvs), np.identity(ncas), rdm_ca, optimize = einsum_type)
    precond_caa_caa_bab_bab += 1/2 * einsum('Xx,II,YZ,xW->IWZXY', h_aa, np.identity(ncvs), np.identity(ncas), rdm_ca, optimize = einsum_type)
    precond_caa_caa_bab_bab += 1/2 * einsum('Xxyz,II,YZ,Wxyz->IWZXY', v_aaaa, np.identity(ncvs), np.identity(ncas), rdm_ccaa, optimize = einsum_type)

    precond_caa_caa_aaa_abb  = 1/6 * einsum('I,II,WYXZ->IWZXY', e_cvs, np.identity(ncvs), rdm_ccaa, optimize = einsum_type)
    precond_caa_caa_aaa_abb += 1/3 * einsum('I,II,WYZX->IWZXY', e_cvs, np.identity(ncvs), rdm_ccaa, optimize = einsum_type)
    precond_caa_caa_aaa_abb += 1/3 * einsum('Xx,II,WYZx->IWZXY', h_aa, np.identity(ncvs), rdm_ccaa, optimize = einsum_type)
    precond_caa_caa_aaa_abb += 1/6 * einsum('Xx,II,WYxZ->IWZXY', h_aa, np.identity(ncvs), rdm_ccaa, optimize = einsum_type)
    precond_caa_caa_aaa_abb -= 1/6 * einsum('Yx,II,WxXZ->IWZXY', h_aa, np.identity(ncvs), rdm_ccaa, optimize = einsum_type)
    precond_caa_caa_aaa_abb -= 1/3 * einsum('Yx,II,WxZX->IWZXY', h_aa, np.identity(ncvs), rdm_ccaa, optimize = einsum_type)
    precond_caa_caa_aaa_abb += 1/6 * einsum('II,XZxy,WYxy->IWZXY', np.identity(ncvs), v_aaaa, rdm_ccaa, optimize = einsum_type)
    precond_caa_caa_aaa_abb += 1/3 * einsum('II,XZxy,WYyx->IWZXY', np.identity(ncvs), v_aaaa, rdm_ccaa, optimize = einsum_type)
    precond_caa_caa_aaa_abb += 1/4 * einsum('II,Xxyz,ZyzWYx->IWZXY', np.identity(ncvs), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    precond_caa_caa_aaa_abb -= 1/12 * einsum('II,Xxyz,ZyzWxY->IWZXY', np.identity(ncvs), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    precond_caa_caa_aaa_abb += 1/12 * einsum('II,Xxyz,ZyzYWx->IWZXY', np.identity(ncvs), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    precond_caa_caa_aaa_abb -= 1/12 * einsum('II,Xxyz,ZyzYxW->IWZXY', np.identity(ncvs), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    precond_caa_caa_aaa_abb -= 1/12 * einsum('II,Xxyz,ZyzxWY->IWZXY', np.identity(ncvs), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    precond_caa_caa_aaa_abb -= 1/12 * einsum('II,Xxyz,ZyzxYW->IWZXY', np.identity(ncvs), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    precond_caa_caa_aaa_abb -= 1/6 * einsum('II,YxyZ,WyXx->IWZXY', np.identity(ncvs), v_aaaa, rdm_ccaa, optimize = einsum_type)
    precond_caa_caa_aaa_abb -= 1/3 * einsum('II,YxyZ,WyxX->IWZXY', np.identity(ncvs), v_aaaa, rdm_ccaa, optimize = einsum_type)
    precond_caa_caa_aaa_abb -= 1/12 * einsum('II,Yxyz,XZxWyz->IWZXY', np.identity(ncvs), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    precond_caa_caa_aaa_abb += 1/12 * einsum('II,Yxyz,XZxWzy->IWZXY', np.identity(ncvs), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    precond_caa_caa_aaa_abb -= 1/4 * einsum('II,Yxyz,XZxyWz->IWZXY', np.identity(ncvs), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    precond_caa_caa_aaa_abb += 1/12 * einsum('II,Yxyz,XZxyzW->IWZXY', np.identity(ncvs), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    precond_caa_caa_aaa_abb += 1/12 * einsum('II,Yxyz,XZxzWy->IWZXY', np.identity(ncvs), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    precond_caa_caa_aaa_abb += 1/12 * einsum('II,Yxyz,XZxzyW->IWZXY', np.identity(ncvs), v_aaaa, rdm_cccaaa, optimize = einsum_type)

    precond_caa_caa_abb_aaa  = 1/6 * einsum('I,II,WYXZ->IWZXY', e_cvs, np.identity(ncvs), rdm_ccaa, optimize = einsum_type)
    precond_caa_caa_abb_aaa += 1/3 * einsum('I,II,WYZX->IWZXY', e_cvs, np.identity(ncvs), rdm_ccaa, optimize = einsum_type)
    precond_caa_caa_abb_aaa += 1/3 * einsum('Xx,II,WYZx->IWZXY', h_aa, np.identity(ncvs), rdm_ccaa, optimize = einsum_type)
    precond_caa_caa_abb_aaa += 1/6 * einsum('Xx,II,WYxZ->IWZXY', h_aa, np.identity(ncvs), rdm_ccaa, optimize = einsum_type)
    precond_caa_caa_abb_aaa -= 1/6 * einsum('Yx,II,WxXZ->IWZXY', h_aa, np.identity(ncvs), rdm_ccaa, optimize = einsum_type)
    precond_caa_caa_abb_aaa -= 1/3 * einsum('Yx,II,WxZX->IWZXY', h_aa, np.identity(ncvs), rdm_ccaa, optimize = einsum_type)
    precond_caa_caa_abb_aaa += 1/6 * einsum('II,XZxy,WYxy->IWZXY', np.identity(ncvs), v_aaaa, rdm_ccaa, optimize = einsum_type)
    precond_caa_caa_abb_aaa += 1/3 * einsum('II,XZxy,WYyx->IWZXY', np.identity(ncvs), v_aaaa, rdm_ccaa, optimize = einsum_type)
    precond_caa_caa_abb_aaa += 1/4 * einsum('II,Xxyz,ZyzWYx->IWZXY', np.identity(ncvs), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    precond_caa_caa_abb_aaa -= 1/12 * einsum('II,Xxyz,ZyzWxY->IWZXY', np.identity(ncvs), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    precond_caa_caa_abb_aaa += 1/12 * einsum('II,Xxyz,ZyzYWx->IWZXY', np.identity(ncvs), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    precond_caa_caa_abb_aaa -= 1/12 * einsum('II,Xxyz,ZyzYxW->IWZXY', np.identity(ncvs), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    precond_caa_caa_abb_aaa -= 1/12 * einsum('II,Xxyz,ZyzxWY->IWZXY', np.identity(ncvs), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    precond_caa_caa_abb_aaa -= 1/12 * einsum('II,Xxyz,ZyzxYW->IWZXY', np.identity(ncvs), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    precond_caa_caa_abb_aaa -= 1/6 * einsum('II,YxyZ,WyXx->IWZXY', np.identity(ncvs), v_aaaa, rdm_ccaa, optimize = einsum_type)
    precond_caa_caa_abb_aaa -= 1/3 * einsum('II,YxyZ,WyxX->IWZXY', np.identity(ncvs), v_aaaa, rdm_ccaa, optimize = einsum_type)
    precond_caa_caa_abb_aaa -= 1/12 * einsum('II,Yxyz,XZxWyz->IWZXY', np.identity(ncvs), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    precond_caa_caa_abb_aaa += 1/12 * einsum('II,Yxyz,XZxWzy->IWZXY', np.identity(ncvs), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    precond_caa_caa_abb_aaa -= 1/4 * einsum('II,Yxyz,XZxyWz->IWZXY', np.identity(ncvs), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    precond_caa_caa_abb_aaa += 1/12 * einsum('II,Yxyz,XZxyzW->IWZXY', np.identity(ncvs), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    precond_caa_caa_abb_aaa += 1/12 * einsum('II,Yxyz,XZxzWy->IWZXY', np.identity(ncvs), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    precond_caa_caa_abb_aaa += 1/12 * einsum('II,Yxyz,XZxzyW->IWZXY', np.identity(ncvs), v_aaaa, rdm_cccaaa, optimize = einsum_type)

    ## Building C-CAA matrix
    dim_XY = ncas * ncas
    dim_c_caa = 3 * dim_XY

    precond_aa_i = 1
    precond_aa_f = precond_aa_i + dim_XY
    precond_bb_i = precond_aa_f
    precond_bb_f = precond_bb_i + dim_XY
    precond_ab_i = precond_bb_f
    precond_ab_f = precond_ab_i + dim_XY

    precond_temp = np.zeros((ncvs, (1 + dim_c_caa), (1 + dim_c_caa)))
    precond_temp[:, 0, 0] = np.diag(M_00[s_c:f_c, s_c:f_c]).copy()

    precond_temp[:, 0, precond_aa_i:precond_aa_f] = precond_c_caa_a_aaa.reshape(ncvs, ncas * ncas).copy()
    precond_temp[:, 0, precond_bb_i:precond_bb_f] = precond_c_caa_a_abb.reshape(ncvs, ncas * ncas).copy()
    precond_temp[:, precond_aa_i:precond_ab_f, 0] = precond_temp[:, 0, precond_aa_i:precond_ab_f].copy()

    precond_temp[:, precond_aa_i:precond_aa_f, precond_aa_i:precond_aa_f] = precond_caa_caa_aaa_aaa.reshape(ncvs, ncas * ncas, ncas * ncas).copy()
    precond_temp[:, precond_aa_i:precond_aa_f, precond_bb_i:precond_bb_f] = precond_caa_caa_aaa_abb.reshape(ncvs, ncas * ncas, ncas * ncas).copy()

    precond_temp[:, precond_bb_i:precond_bb_f, precond_bb_i:precond_bb_f] = precond_caa_caa_abb_abb.reshape(ncvs, ncas * ncas, ncas * ncas).copy()
    precond_temp[:, precond_bb_i:precond_bb_f, precond_aa_i:precond_aa_f] = precond_caa_caa_abb_aaa.reshape(ncvs, ncas * ncas, ncas * ncas).copy()

    precond_temp[:, precond_ab_i:precond_ab_f, precond_ab_i:precond_ab_f] = precond_caa_caa_bab_bab.reshape(ncvs, ncas * ncas, ncas * ncas).copy()
    # print(">> SA precond CAA :\n{:}\n".format(precond_temp))

    # print(">>> SA M_CAA_CAA (aaa-aaa) : {:}".format(np.linalg.norm(precond_caa_caa_aaa_aaa)))
    # print(">>> SA M_CAA_CAA (aaa-abb) : {:}".format(np.linalg.norm(precond_caa_caa_aaa_abb)))
    # print(">>> SA M_CAA_CAA (abb-aaa) : {:}".format(np.linalg.norm(precond_caa_caa_abb_aaa)))
    # print(">>> SA M_CAA_CAA (abb-abb) : {:}".format(np.linalg.norm(precond_caa_caa_abb_abb)))
    # print(">>> SA M_CAA_CAA (bab-bab) : {:}".format(np.linalg.norm(precond_caa_caa_bab_bab)))

    # print(">> SA M_CAA_CAA (aaa-aaa) :\n{:}\n".format(precond_caa_caa_aaa_aaa))
    # print(">> SA M_CAA_CAA (aaa-abb) :\n{:}\n".format(precond_caa_caa_aaa_abb))
    # print(">> SA M_CAA_CAA (abb-aaa) :\n{:}\n".format(precond_caa_caa_abb_aaa))
    # print(">> SA M_CAA_CAA (abb-abb) :\n{:}\n".format(precond_caa_caa_abb_abb))
    # print(">> SA M_CAA_CAA (bab-bab) :\n{:}\n".format(precond_caa_caa_bab_bab))

    precond_temp = einsum('IXY,XP,YP->IP', precond_temp, S12_c_caa, S12_c_caa, optimize = einsum_type)
    # print(">>> SA precond CAA : {:}".format(precond_temp.shape))
    print(">> SA precond CAA :\n{:}\n".format(precond_temp))

    # print(">>> SA precond CAA (aaa) norm: {:}".format(np.linalg.norm(precond_temp[])))

    precond[ho_s_c_caa:ho_f_c_caa] = precond_temp.reshape(-1)

    # Multiply by -1.0, since we are solving for -M C = -S C E
    precond *= (-1.0)

    print ("Time for computing preconditioner:                %f sec\n" % (time.time() - start_time))
    sys.stdout.flush()

    return precond

def apply_S_12_caa(mr_adc, X, transpose = False):

    # Einsum definition from kernel
    einsum = mr_adc.interface.einsum
    einsum_type = mr_adc.interface.einsum_type

    # Dimensions
    nextern = mr_adc.nextern
    ncvs = mr_adc.ncvs
    nval = mr_adc.nval

    ho_s_c = mr_adc.h_orth.s_c
    ho_f_c = mr_adc.h_orth.f_c
    ho_s_c_caa = mr_adc.h_orth.s_c_caa
    ho_f_c_caa = mr_adc.h_orth.f_c_caa
    ho_s_cce = mr_adc.h_orth.s_cce
    ho_f_cce = mr_adc.h_orth.f_cce
    ho_s_cae = mr_adc.h_orth.s_cae
    ho_f_cae = mr_adc.h_orth.f_cae
    ho_s_ace = mr_adc.h_orth.s_ace
    ho_f_ace = mr_adc.h_orth.f_ace
    ho_s_cca = mr_adc.h_orth.s_cca
    ho_f_cca = mr_adc.h_orth.f_cca
    s_c = mr_adc.h0.s_c
    f_c = mr_adc.h0.f_c
    s_caa = mr_adc.h1.s_caa
    f_caa = mr_adc.h1.f_caa
    s_cce = mr_adc.h1.s_cce
    f_cce = mr_adc.h1.f_cce
    s_cae = mr_adc.h1.s_cae
    f_cae = mr_adc.h1.f_cae
    s_ace = mr_adc.h1.s_ace
    f_ace = mr_adc.h1.f_ace
    s_cca = mr_adc.h1.s_cca
    f_cca = mr_adc.h1.f_cca

    if nval > 0:
        ho_s_cve = mr_adc.h_orth.s_cve
        ho_f_cve = mr_adc.h_orth.f_cve
        ho_s_vce = mr_adc.h_orth.s_vce
        ho_f_vce = mr_adc.h_orth.f_vce

        ho_s_cva = mr_adc.h_orth.s_cva
        ho_f_cva = mr_adc.h_orth.f_cva
        ho_s_vca = mr_adc.h_orth.s_vca
        ho_f_vca = mr_adc.h_orth.f_vca

        s_cve = mr_adc.h1.s_cve
        f_cve = mr_adc.h1.f_cve
        s_vce = mr_adc.h1.s_vce
        f_vce = mr_adc.h1.f_vce

        s_cva = mr_adc.h1.s_cva
        f_cva = mr_adc.h1.f_cva
        s_vca = mr_adc.h1.s_vca
        f_vca = mr_adc.h1.f_vca

    # Overlap matrices
    S12_c_caa = mr_adc.S12.c_caa
    S12_cae = mr_adc.S12.cae
    S12_cca = mr_adc.S12.cca

    Xt = None

    if transpose:
        if (X.shape[0] != (mr_adc.h0.dim + mr_adc.h1.dim)):
            raise Exception("Dimensions do not match when applying S_12 transpose")

        Xt = np.zeros(mr_adc.h_orth.dim)

        # C and CAA -> C_CAA
        temp = np.zeros((ncvs, S12_c_caa.shape[0]))
        temp[:,0] = X[s_c:f_c].copy()
        temp[:,1:] = X[s_caa:f_caa].reshape(ncvs, -1).copy()
        Xt[ho_s_c_caa:ho_f_c_caa] = np.dot(temp, S12_c_caa).reshape(-1).copy()

    else:
        if (X.shape[0] != (mr_adc.h_orth.dim)):
            raise Exception("Dimensions do not match when applying S_12")

        Xt = np.zeros(mr_adc.h0.dim + mr_adc.h1.dim)

        # C_CAA -> C and CAA
        temp = X[ho_s_c_caa:ho_f_c_caa].reshape(ncvs, -1).copy()
        temp = np.dot(temp, S12_c_caa.T)
        Xt[s_c:f_c] = temp[:,0].copy()
        Xt[s_caa:f_caa] = temp[:,1:].reshape(-1).copy()

    return Xt

def compute_sigma_vector_caa(mr_adc, M_00, M_01, M_11, Xt):

    # Einsum definition from kernel
    einsum = mr_adc.interface.einsum
    einsum_type = mr_adc.interface.einsum_type

    # Variables from kernel
    ncvs = mr_adc.ncvs
    nval = mr_adc.nval
    ncas = mr_adc.ncas
    nextern = mr_adc.nextern

    e_cvs = mr_adc.mo_energy.x
    e_core = mr_adc.mo_energy.c
    e_extern = mr_adc.mo_energy.e
    if nval > 0:
        e_val = mr_adc.mo_energy.v

    # One-electron integrals
    h_aa = mr_adc.h1eff.aa

    # Two-electrons integrals
    v_aaaa = mr_adc.v2e.aaaa

    # Reduced Density Matrices
    rdm_ca = mr_adc.rdm.ca
    rdm_ccaa = mr_adc.rdm.ccaa
    rdm_cccaaa = mr_adc.rdm.cccaaa

    # Dimensions
    s_c = mr_adc.h0.s_c
    f_c = mr_adc.h0.f_c
    s_caa = mr_adc.h1.s_caa
    f_caa = mr_adc.h1.f_caa
    s_cce = mr_adc.h1.s_cce
    f_cce = mr_adc.h1.f_cce
    s_cae = mr_adc.h1.s_cae
    f_cae = mr_adc.h1.f_cae
    s_ace = mr_adc.h1.s_ace
    f_ace = mr_adc.h1.f_ace
    s_cca = mr_adc.h1.s_cca
    f_cca = mr_adc.h1.f_cca
    if nval > 0:
        s_cve = mr_adc.h1.s_cve
        f_cve = mr_adc.h1.f_cve
        s_vce = mr_adc.h1.s_vce
        f_vce = mr_adc.h1.f_vce

        s_cva = mr_adc.h1.s_cva
        f_cva = mr_adc.h1.f_cva
        s_vca = mr_adc.h1.s_vca
        f_vca = mr_adc.h1.f_vca

    cvs_tril_ind = np.tril_indices(ncvs, k=-1)

    # (CASCI + C) -> (CASCI + C)
    sigma = np.zeros_like(Xt)

    # h0-h0 contributions
    sigma[:mr_adc.h0.dim] = np.dot(M_00, Xt[:mr_adc.h0.dim])

    # h0-h1 and h1-h0 contributions
    if nval > 0:
        M_C_CAA, M_C_CCE, M_C_CVE, M_C_CAE, M_C_CCA, M_C_CVA = M_01
    else:
        M_C_CAA, M_C_CCE, M_C_CAE, M_C_CCA = M_01

    # C <-> CAA
    sigma[s_c:f_c] += np.dot(M_C_CAA, Xt[s_caa:f_caa])
    sigma[s_caa:f_caa] += np.dot(M_C_CAA.T, Xt[s_c:f_c])

    # h1-h1 contributions
    # CAA <- CAA
    X_caa = Xt[s_caa:f_caa].reshape(-1).copy()

    dim_WZ = ncas * ncas
    dim_c_caa = ncvs * dim_WZ

    sigma_aaa_i = 0
    sigma_aaa_f = sigma_aaa_i + dim_c_caa
    sigma_abb_i = sigma_aaa_f
    sigma_abb_f = sigma_abb_i + dim_c_caa
    sigma_bab_i = sigma_abb_f
    sigma_bab_f = sigma_bab_i + dim_c_caa

    X_aaa = X_caa[sigma_aaa_i:sigma_aaa_f].reshape(ncvs, ncas, ncas).copy()
    X_abb = X_caa[sigma_abb_i:sigma_abb_f].reshape(ncvs, ncas, ncas).copy()
    X_bab = X_caa[sigma_bab_i:sigma_bab_f].reshape(ncvs, ncas, ncas).copy()

    # C <-> CAA
    sigma_caa_aaa  = 1/2 * einsum('KxZ,K,xW->KWZ', X_aaa, e_cvs, rdm_ca, optimize = einsum_type)
    sigma_caa_aaa += 1/6 * einsum('Kxy,K,WyZx->KWZ', X_aaa, e_cvs, rdm_ccaa, optimize = einsum_type)
    sigma_caa_aaa -= 1/6 * einsum('Kxy,K,WyxZ->KWZ', X_aaa, e_cvs, rdm_ccaa, optimize = einsum_type)
    sigma_caa_aaa += 1/2 * einsum('KxZ,xy,yW->KWZ', X_aaa, h_aa, rdm_ca, optimize = einsum_type)
    sigma_caa_aaa -= 1/2 * einsum('Kxy,Zy,xW->KWZ', X_aaa, h_aa, rdm_ca, optimize = einsum_type)
    sigma_caa_aaa += 1/6 * einsum('Kxy,xz,WyZz->KWZ', X_aaa, h_aa, rdm_ccaa, optimize = einsum_type)
    sigma_caa_aaa -= 1/6 * einsum('Kxy,xz,WyzZ->KWZ', X_aaa, h_aa, rdm_ccaa, optimize = einsum_type)
    sigma_caa_aaa -= 1/6 * einsum('Kxy,yz,WzZx->KWZ', X_aaa, h_aa, rdm_ccaa, optimize = einsum_type)
    sigma_caa_aaa += 1/6 * einsum('Kxy,yz,WzxZ->KWZ', X_aaa, h_aa, rdm_ccaa, optimize = einsum_type)
    sigma_caa_aaa += 1/2 * einsum('KxZ,xyzw,Wyzw->KWZ', X_aaa, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_caa_aaa += 1/6 * einsum('Kxy,Zxzw,Wyzw->KWZ', X_aaa, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_caa_aaa -= 1/6 * einsum('Kxy,Zxzw,Wywz->KWZ', X_aaa, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_caa_aaa -= 1/2 * einsum('Kxy,Zzyw,Wzxw->KWZ', X_aaa, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_caa_aaa += 1/6 * einsum('Kxy,Zzwy,Wzxw->KWZ', X_aaa, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_caa_aaa -= 1/6 * einsum('Kxy,Zzwy,Wzwx->KWZ', X_aaa, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_caa_aaa += 1/6 * einsum('Kxy,xzwu,ZwuWyz->KWZ', X_aaa, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_caa_aaa -= 1/6 * einsum('Kxy,xzwu,ZwuyWz->KWZ', X_aaa, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_caa_aaa -= 1/6 * einsum('Kxy,yzwu,ZxzWwu->KWZ', X_aaa, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_caa_aaa += 1/6 * einsum('Kxy,yzwu,ZxzwWu->KWZ', X_aaa, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_caa_aaa += 1/3 * einsum('Kxy,K,WyZx->KWZ', X_abb, e_cvs, rdm_ccaa, optimize = einsum_type)
    sigma_caa_aaa += 1/6 * einsum('Kxy,K,WyxZ->KWZ', X_abb, e_cvs, rdm_ccaa, optimize = einsum_type)
    sigma_caa_aaa += 1/3 * einsum('Kxy,xz,WyZz->KWZ', X_abb, h_aa, rdm_ccaa, optimize = einsum_type)
    sigma_caa_aaa += 1/6 * einsum('Kxy,xz,WyzZ->KWZ', X_abb, h_aa, rdm_ccaa, optimize = einsum_type)
    sigma_caa_aaa -= 1/3 * einsum('Kxy,yz,WzZx->KWZ', X_abb, h_aa, rdm_ccaa, optimize = einsum_type)
    sigma_caa_aaa -= 1/6 * einsum('Kxy,yz,WzxZ->KWZ', X_abb, h_aa, rdm_ccaa, optimize = einsum_type)
    sigma_caa_aaa += 1/3 * einsum('Kxy,Zxzw,Wyzw->KWZ', X_abb, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_caa_aaa += 1/6 * einsum('Kxy,Zxzw,Wywz->KWZ', X_abb, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_caa_aaa -= 1/6 * einsum('Kxy,Zzwy,Wzxw->KWZ', X_abb, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_caa_aaa -= 1/3 * einsum('Kxy,Zzwy,Wzwx->KWZ', X_abb, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_caa_aaa += 1/4 * einsum('Kxy,xzwu,ZwuWyz->KWZ', X_abb, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_caa_aaa -= 1/12 * einsum('Kxy,xzwu,ZwuWzy->KWZ', X_abb, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_caa_aaa += 1/12 * einsum('Kxy,xzwu,ZwuyWz->KWZ', X_abb, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_caa_aaa -= 1/12 * einsum('Kxy,xzwu,ZwuyzW->KWZ', X_abb, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_caa_aaa -= 1/12 * einsum('Kxy,xzwu,ZwuzWy->KWZ', X_abb, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_caa_aaa -= 1/12 * einsum('Kxy,xzwu,ZwuzyW->KWZ', X_abb, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_caa_aaa -= 1/4 * einsum('Kxy,yzwu,ZxzWwu->KWZ', X_abb, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_caa_aaa += 1/12 * einsum('Kxy,yzwu,ZxzWuw->KWZ', X_abb, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_caa_aaa -= 1/12 * einsum('Kxy,yzwu,ZxzwWu->KWZ', X_abb, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_caa_aaa += 1/12 * einsum('Kxy,yzwu,ZxzwuW->KWZ', X_abb, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_caa_aaa += 1/12 * einsum('Kxy,yzwu,ZxzuWw->KWZ', X_abb, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_caa_aaa += 1/12 * einsum('Kxy,yzwu,ZxzuwW->KWZ', X_abb, v_aaaa, rdm_cccaaa, optimize = einsum_type)

    sigma_caa_abb  = 1/3 * einsum('Kxy,K,WyZx->KWZ', X_aaa, e_cvs, rdm_ccaa, optimize = einsum_type)
    sigma_caa_abb += 1/6 * einsum('Kxy,K,WyxZ->KWZ', X_aaa, e_cvs, rdm_ccaa, optimize = einsum_type)
    sigma_caa_abb += 1/3 * einsum('Kxy,xz,WyZz->KWZ', X_aaa, h_aa, rdm_ccaa, optimize = einsum_type)
    sigma_caa_abb += 1/6 * einsum('Kxy,xz,WyzZ->KWZ', X_aaa, h_aa, rdm_ccaa, optimize = einsum_type)
    sigma_caa_abb -= 1/3 * einsum('Kxy,yz,WzZx->KWZ', X_aaa, h_aa, rdm_ccaa, optimize = einsum_type)
    sigma_caa_abb -= 1/6 * einsum('Kxy,yz,WzxZ->KWZ', X_aaa, h_aa, rdm_ccaa, optimize = einsum_type)
    sigma_caa_abb += 1/3 * einsum('Kxy,Zxzw,Wyzw->KWZ', X_aaa, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_caa_abb += 1/6 * einsum('Kxy,Zxzw,Wywz->KWZ', X_aaa, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_caa_abb -= 1/6 * einsum('Kxy,Zzwy,Wzxw->KWZ', X_aaa, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_caa_abb -= 1/3 * einsum('Kxy,Zzwy,Wzwx->KWZ', X_aaa, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_caa_abb += 1/4 * einsum('Kxy,xzwu,ZwuWyz->KWZ', X_aaa, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_caa_abb -= 1/12 * einsum('Kxy,xzwu,ZwuWzy->KWZ', X_aaa, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_caa_abb += 1/12 * einsum('Kxy,xzwu,ZwuyWz->KWZ', X_aaa, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_caa_abb -= 1/12 * einsum('Kxy,xzwu,ZwuyzW->KWZ', X_aaa, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_caa_abb -= 1/12 * einsum('Kxy,xzwu,ZwuzWy->KWZ', X_aaa, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_caa_abb -= 1/12 * einsum('Kxy,xzwu,ZwuzyW->KWZ', X_aaa, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_caa_abb -= 1/4 * einsum('Kxy,yzwu,ZxzWwu->KWZ', X_aaa, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_caa_abb += 1/12 * einsum('Kxy,yzwu,ZxzWuw->KWZ', X_aaa, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_caa_abb -= 1/12 * einsum('Kxy,yzwu,ZxzwWu->KWZ', X_aaa, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_caa_abb += 1/12 * einsum('Kxy,yzwu,ZxzwuW->KWZ', X_aaa, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_caa_abb += 1/12 * einsum('Kxy,yzwu,ZxzuWw->KWZ', X_aaa, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_caa_abb += 1/12 * einsum('Kxy,yzwu,ZxzuwW->KWZ', X_aaa, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_caa_abb += 1/2 * einsum('KxZ,K,xW->KWZ', X_abb, e_cvs, rdm_ca, optimize = einsum_type)
    sigma_caa_abb += 1/6 * einsum('Kxy,K,WyZx->KWZ', X_abb, e_cvs, rdm_ccaa, optimize = einsum_type)
    sigma_caa_abb -= 1/6 * einsum('Kxy,K,WyxZ->KWZ', X_abb, e_cvs, rdm_ccaa, optimize = einsum_type)
    sigma_caa_abb += 1/2 * einsum('KxZ,xy,yW->KWZ', X_abb, h_aa, rdm_ca, optimize = einsum_type)
    sigma_caa_abb -= 1/2 * einsum('Kxy,Zy,xW->KWZ', X_abb, h_aa, rdm_ca, optimize = einsum_type)
    sigma_caa_abb += 1/6 * einsum('Kxy,xz,WyZz->KWZ', X_abb, h_aa, rdm_ccaa, optimize = einsum_type)
    sigma_caa_abb -= 1/6 * einsum('Kxy,xz,WyzZ->KWZ', X_abb, h_aa, rdm_ccaa, optimize = einsum_type)
    sigma_caa_abb -= 1/6 * einsum('Kxy,yz,WzZx->KWZ', X_abb, h_aa, rdm_ccaa, optimize = einsum_type)
    sigma_caa_abb += 1/6 * einsum('Kxy,yz,WzxZ->KWZ', X_abb, h_aa, rdm_ccaa, optimize = einsum_type)
    sigma_caa_abb += 1/2 * einsum('KxZ,xyzw,Wyzw->KWZ', X_abb, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_caa_abb += 1/6 * einsum('Kxy,Zxzw,Wyzw->KWZ', X_abb, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_caa_abb -= 1/6 * einsum('Kxy,Zxzw,Wywz->KWZ', X_abb, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_caa_abb -= 1/2 * einsum('Kxy,Zzyw,Wzxw->KWZ', X_abb, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_caa_abb += 1/6 * einsum('Kxy,Zzwy,Wzxw->KWZ', X_abb, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_caa_abb -= 1/6 * einsum('Kxy,Zzwy,Wzwx->KWZ', X_abb, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_caa_abb += 1/6 * einsum('Kxy,xzwu,ZwuWyz->KWZ', X_abb, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_caa_abb -= 1/6 * einsum('Kxy,xzwu,ZwuyWz->KWZ', X_abb, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_caa_abb -= 1/6 * einsum('Kxy,yzwu,ZxzWwu->KWZ', X_abb, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_caa_abb += 1/6 * einsum('Kxy,yzwu,ZxzwWu->KWZ', X_abb, v_aaaa, rdm_cccaaa, optimize = einsum_type)

    sigma_caa_bab  = 1/2 * einsum('KxZ,K,xW->KWZ', X_bab, e_cvs, rdm_ca, optimize = einsum_type)
    sigma_caa_bab -= 1/6 * einsum('Kxy,K,WyZx->KWZ', X_bab, e_cvs, rdm_ccaa, optimize = einsum_type)
    sigma_caa_bab -= 1/3 * einsum('Kxy,K,WyxZ->KWZ', X_bab, e_cvs, rdm_ccaa, optimize = einsum_type)
    sigma_caa_bab += 1/2 * einsum('KxZ,xy,yW->KWZ', X_bab, h_aa, rdm_ca, optimize = einsum_type)
    sigma_caa_bab -= 1/2 * einsum('Kxy,Zy,xW->KWZ', X_bab, h_aa, rdm_ca, optimize = einsum_type)
    sigma_caa_bab -= 1/6 * einsum('Kxy,xz,WyZz->KWZ', X_bab, h_aa, rdm_ccaa, optimize = einsum_type)
    sigma_caa_bab -= 1/3 * einsum('Kxy,xz,WyzZ->KWZ', X_bab, h_aa, rdm_ccaa, optimize = einsum_type)
    sigma_caa_bab += 1/6 * einsum('Kxy,yz,WzZx->KWZ', X_bab, h_aa, rdm_ccaa, optimize = einsum_type)
    sigma_caa_bab += 1/3 * einsum('Kxy,yz,WzxZ->KWZ', X_bab, h_aa, rdm_ccaa, optimize = einsum_type)
    sigma_caa_bab += 1/2 * einsum('KxZ,xyzw,Wyzw->KWZ', X_bab, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_caa_bab -= 1/6 * einsum('Kxy,Zxzw,Wyzw->KWZ', X_bab, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_caa_bab -= 1/3 * einsum('Kxy,Zxzw,Wywz->KWZ', X_bab, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_caa_bab -= 1/2 * einsum('Kxy,Zzyw,Wzxw->KWZ', X_bab, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_caa_bab += 1/3 * einsum('Kxy,Zzwy,Wzxw->KWZ', X_bab, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_caa_bab += 1/6 * einsum('Kxy,Zzwy,Wzwx->KWZ', X_bab, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_caa_bab -= 1/12 * einsum('Kxy,xzwu,ZwuWyz->KWZ', X_bab, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_caa_bab += 1/12 * einsum('Kxy,xzwu,ZwuWzy->KWZ', X_bab, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_caa_bab -= 1/4 * einsum('Kxy,xzwu,ZwuyWz->KWZ', X_bab, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_caa_bab += 1/12 * einsum('Kxy,xzwu,ZwuyzW->KWZ', X_bab, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_caa_bab += 1/12 * einsum('Kxy,xzwu,ZwuzWy->KWZ', X_bab, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_caa_bab += 1/12 * einsum('Kxy,xzwu,ZwuzyW->KWZ', X_bab, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_caa_bab += 1/12 * einsum('Kxy,yzwu,ZxzWwu->KWZ', X_bab, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_caa_bab -= 1/12 * einsum('Kxy,yzwu,ZxzWuw->KWZ', X_bab, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_caa_bab += 1/4 * einsum('Kxy,yzwu,ZxzwWu->KWZ', X_bab, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_caa_bab -= 1/12 * einsum('Kxy,yzwu,ZxzwuW->KWZ', X_bab, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_caa_bab -= 1/12 * einsum('Kxy,yzwu,ZxzuWw->KWZ', X_bab, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_caa_bab -= 1/12 * einsum('Kxy,yzwu,ZxzuwW->KWZ', X_bab, v_aaaa, rdm_cccaaa, optimize = einsum_type)

    ## Building C-CAA matrix
    dim_caa = ncvs * ncas * ncas

    sigma_aaa_i = 0
    sigma_aaa_f = sigma_aaa_i + dim_caa
    sigma_abb_i = sigma_aaa_f
    sigma_abb_f = sigma_abb_i + dim_caa
    sigma_bab_i = sigma_abb_f
    sigma_bab_f = sigma_bab_i + dim_caa

    sigma_caa = np.zeros((3 * dim_caa))
    sigma_caa[sigma_aaa_i:sigma_aaa_f] = sigma_caa_aaa.reshape(-1).copy()
    sigma_caa[sigma_abb_i:sigma_abb_f] = sigma_caa_abb.reshape(-1).copy()
    sigma_caa[sigma_bab_i:sigma_bab_f] = sigma_caa_bab.reshape(-1).copy()

    # print(">>> SA sigma_caa_aaa: {:}".format(np.linalg.norm(sigma_caa_aaa)))
    # print(">>> SA sigma_caa_abb: {:}".format(np.linalg.norm(sigma_caa_abb)))
    # print(">>> SA sigma_caa_bab: {:}".format(np.linalg.norm(sigma_caa_bab)))

    sigma[s_caa:f_caa] += sigma_caa.reshape(-1).copy()

    return sigma

### Ensembling (without CAA)
## Done: CCE, CVE, CAE, CCA
def compute_excitation_manifolds_dev(mr_adc):

    # MR-ADC(0) and MR-ADC(1)
    mr_adc.h0.n_c = mr_adc.ncvs

    ## Total dimension of h0
    mr_adc.h0.dim = mr_adc.h0.n_c

    mr_adc.h0.s_c = 0
    mr_adc.h0.f_c = mr_adc.h0.s_c + mr_adc.h0.n_c

    print("Dimension of h0 excitation manifold:                       %d" % mr_adc.h0.dim)

    # MR-ADC(2)
    mr_adc.h1.dim = 0
    mr_adc.h_orth.dim = mr_adc.h0.dim

    if mr_adc.method in ("mr-adc(2)", "mr-adc(2)-x"):
        mr_adc.h1.n_caa = 0
        mr_adc.h1.n_cce_aaa = mr_adc.ncvs * (mr_adc.ncvs - 1) * mr_adc.nextern // 2
        mr_adc.h1.n_cce_abb = mr_adc.ncvs * mr_adc.ncvs * mr_adc.nextern
        mr_adc.h1.n_cae = mr_adc.ncvs * mr_adc.ncas * mr_adc.nextern
        mr_adc.h1.n_cca_aaa = mr_adc.ncvs * (mr_adc.ncvs - 1) * mr_adc.ncas // 2
        mr_adc.h1.n_cca_abb = mr_adc.ncvs * mr_adc.ncvs * mr_adc.ncas

        mr_adc.h1.dim_caa = 3 * mr_adc.h1.n_caa
        mr_adc.h1.dim_cce = mr_adc.h1.n_cce_aaa + mr_adc.h1.n_cce_abb
        mr_adc.h1.dim_cae = 3 * mr_adc.h1.n_cae
        mr_adc.h1.dim_cca = mr_adc.h1.n_cca_aaa + mr_adc.h1.n_cca_abb

        if mr_adc.nval > 0:
            mr_adc.h1.n_cve = mr_adc.ncvs * mr_adc.nval * mr_adc.nextern
            mr_adc.h1.n_cva = mr_adc.ncvs * mr_adc.nval * mr_adc.ncas

            mr_adc.h1.dim_cve = 3 * mr_adc.h1.n_cve
            mr_adc.h1.dim_cva = 3 * mr_adc.h1.n_cva

            mr_adc.h1.dim = (mr_adc.h1.dim_caa + mr_adc.h1.dim_cce + mr_adc.h1.dim_cve +
                             mr_adc.h1.dim_cae + mr_adc.h1.dim_cca + mr_adc.h1.dim_cva)
        else:
            mr_adc.h1.dim = (mr_adc.h1.dim_caa + mr_adc.h1.dim_cce + mr_adc.h1.dim_cae + mr_adc.h1.dim_cca)

        if mr_adc.nval > 0:
            mr_adc.h1.s_caa = mr_adc.h0.f_c
            mr_adc.h1.f_caa = mr_adc.h1.s_caa + mr_adc.h1.dim_caa
            mr_adc.h1.s_cce = mr_adc.h1.f_caa
            mr_adc.h1.f_cce = mr_adc.h1.s_cce + mr_adc.h1.dim_cce
            mr_adc.h1.s_cve = mr_adc.h1.f_cce
            mr_adc.h1.f_cve = mr_adc.h1.s_cve + mr_adc.h1.dim_cve
            mr_adc.h1.s_cae = mr_adc.h1.f_cve
            mr_adc.h1.f_cae = mr_adc.h1.s_cae + mr_adc.h1.dim_cae
            mr_adc.h1.s_cca = mr_adc.h1.f_cae
            mr_adc.h1.f_cca = mr_adc.h1.s_cca + mr_adc.h1.dim_cca
            mr_adc.h1.s_cva = mr_adc.h1.f_cca
            mr_adc.h1.f_cva = mr_adc.h1.s_cva + mr_adc.h1.dim_cva

            mr_adc.h1.s_caa_aaa = mr_adc.h1.s_caa
            mr_adc.h1.f_caa_aaa = mr_adc.h1.s_caa_aaa + mr_adc.h1.n_caa
            mr_adc.h1.s_caa_abb = mr_adc.h1.f_caa_aaa
            mr_adc.h1.f_caa_abb = mr_adc.h1.s_caa_abb + mr_adc.h1.n_caa
            mr_adc.h1.s_caa_bab = mr_adc.h1.f_caa_abb
            mr_adc.h1.f_caa_bab = mr_adc.h1.s_caa_bab + mr_adc.h1.n_caa

            mr_adc.h1.s_cce_aaa = mr_adc.h1.s_cce
            mr_adc.h1.f_cce_aaa = mr_adc.h1.s_cce_aaa + mr_adc.h1.n_cce_aaa
            mr_adc.h1.s_cce_abb = mr_adc.h1.f_cce_aaa
            mr_adc.h1.f_cce_abb = mr_adc.h1.s_cce_abb + mr_adc.h1.n_cce_abb

            mr_adc.h1.s_cve_aaa = mr_adc.h1.s_cve
            mr_adc.h1.f_cve_aaa = mr_adc.h1.s_cve_aaa + mr_adc.h1.n_cve
            mr_adc.h1.s_cve_abb = mr_adc.h1.f_cve_aaa
            mr_adc.h1.f_cve_abb = mr_adc.h1.s_cve_abb + mr_adc.h1.n_cve
            mr_adc.h1.s_cve_bab = mr_adc.h1.f_cve_abb
            mr_adc.h1.f_cve_bab = mr_adc.h1.s_cve_bab + mr_adc.h1.n_cve

            mr_adc.h1.s_cae_aaa = mr_adc.h1.s_cae
            mr_adc.h1.f_cae_aaa = mr_adc.h1.s_cae_aaa + mr_adc.h1.n_cae
            mr_adc.h1.s_cae_abb = mr_adc.h1.f_cae_aaa
            mr_adc.h1.f_cae_abb = mr_adc.h1.s_cae_abb + mr_adc.h1.n_cae
            mr_adc.h1.s_cae_bab = mr_adc.h1.f_cae_abb
            mr_adc.h1.f_cae_bab = mr_adc.h1.s_cae_bab + mr_adc.h1.n_cae

            mr_adc.h1.s_cca_aaa = mr_adc.h1.s_cca
            mr_adc.h1.f_cca_aaa = mr_adc.h1.s_cca_aaa + mr_adc.h1.n_cca_aaa
            mr_adc.h1.s_cca_abb = mr_adc.h1.f_cca_aaa
            mr_adc.h1.f_cca_abb = mr_adc.h1.s_cca_abb + mr_adc.h1.n_cca_abb

            mr_adc.h1.s_cva_aaa = mr_adc.h1.s_cva
            mr_adc.h1.f_cva_aaa = mr_adc.h1.s_cva_aaa + mr_adc.h1.n_cva
            mr_adc.h1.s_cva_abb = mr_adc.h1.f_cva_aaa
            mr_adc.h1.f_cva_abb = mr_adc.h1.s_cva_abb + mr_adc.h1.n_cva
            mr_adc.h1.s_cva_bab = mr_adc.h1.f_cva_abb
            mr_adc.h1.f_cva_bab = mr_adc.h1.s_cva_bab + mr_adc.h1.n_cva

        else:
            mr_adc.h1.s_caa = mr_adc.h0.f_c
            mr_adc.h1.f_caa = mr_adc.h1.s_caa + mr_adc.h1.dim_caa
            mr_adc.h1.s_cce = mr_adc.h1.f_caa
            mr_adc.h1.f_cce = mr_adc.h1.s_cce + mr_adc.h1.dim_cce
            mr_adc.h1.s_cae = mr_adc.h1.f_cce
            mr_adc.h1.f_cae = mr_adc.h1.s_cae + mr_adc.h1.dim_cae
            mr_adc.h1.s_cca = mr_adc.h1.f_cae
            mr_adc.h1.f_cca = mr_adc.h1.s_cca + mr_adc.h1.dim_cca

            mr_adc.h1.s_caa_aaa = mr_adc.h1.s_caa
            mr_adc.h1.f_caa_aaa = mr_adc.h1.s_caa_aaa + mr_adc.h1.n_caa
            mr_adc.h1.s_caa_abb = mr_adc.h1.f_caa_aaa
            mr_adc.h1.f_caa_abb = mr_adc.h1.s_caa_abb + mr_adc.h1.n_caa
            mr_adc.h1.s_caa_bab = mr_adc.h1.f_caa_abb
            mr_adc.h1.f_caa_bab = mr_adc.h1.s_caa_bab + mr_adc.h1.n_caa

            mr_adc.h1.s_cce_aaa = mr_adc.h1.s_cce
            mr_adc.h1.f_cce_aaa = mr_adc.h1.s_cce_aaa + mr_adc.h1.n_cce_aaa
            mr_adc.h1.s_cce_abb = mr_adc.h1.f_cce_aaa
            mr_adc.h1.f_cce_abb = mr_adc.h1.s_cce_abb + mr_adc.h1.n_cce_abb

            mr_adc.h1.s_cae_aaa = mr_adc.h1.s_cae
            mr_adc.h1.f_cae_aaa = mr_adc.h1.s_cae_aaa + mr_adc.h1.n_cae
            mr_adc.h1.s_cae_abb = mr_adc.h1.f_cae_aaa
            mr_adc.h1.f_cae_abb = mr_adc.h1.s_cae_abb + mr_adc.h1.n_cae
            mr_adc.h1.s_cae_bab = mr_adc.h1.f_cae_abb
            mr_adc.h1.f_cae_bab = mr_adc.h1.s_cae_bab + mr_adc.h1.n_cae

            mr_adc.h1.s_cca_aaa = mr_adc.h1.s_cca
            mr_adc.h1.f_cca_aaa = mr_adc.h1.s_cca_aaa + mr_adc.h1.n_cca_aaa
            mr_adc.h1.s_cca_abb = mr_adc.h1.f_cca_aaa
            mr_adc.h1.f_cca_abb = mr_adc.h1.s_cca_abb + mr_adc.h1.n_cca_abb

        print("Dimension of h1 excitation manifold:                       %d" % mr_adc.h1.dim)

        # Overlap for c - caa
        mr_adc.S12.c_caa = mr_adc_overlap.compute_S12_0p_projector(mr_adc)
        mr_adc.S12.cae = mr_adc_overlap.compute_S12_m1(mr_adc)
        mr_adc.S12.cca = mr_adc_overlap.compute_S12_p1(mr_adc)

        # Determine dimensions of orthogonalized excitation spaces
        ### DEBUG
        mr_adc.h_orth.n_c = mr_adc.ncvs
        ### DEBUG
        mr_adc.h_orth.n_c_caa = 0
        mr_adc.h_orth.n_cce_aaa = mr_adc.h1.n_cce_aaa
        mr_adc.h_orth.n_cce_abb = mr_adc.h1.n_cce_abb
        mr_adc.h_orth.n_cae = mr_adc.ncvs * mr_adc.S12.cae.shape[1] * mr_adc.nextern
        mr_adc.h_orth.n_cca_aaa = mr_adc.ncvs * (mr_adc.ncvs - 1) * mr_adc.S12.cca.shape[1] // 2
        mr_adc.h_orth.n_cca_abb = mr_adc.ncvs * mr_adc.ncvs * mr_adc.S12.cca.shape[1]

        mr_adc.h_orth.dim_c_caa = 3 * mr_adc.h_orth.n_c_caa
        mr_adc.h_orth.dim_cce = mr_adc.h1.dim_cce
        mr_adc.h_orth.dim_cae = 3 * mr_adc.h_orth.n_cae
        mr_adc.h_orth.dim_cca = mr_adc.h_orth.n_cca_aaa + mr_adc.h_orth.n_cca_abb

        if mr_adc.nval > 0:
            mr_adc.h_orth.n_cve = mr_adc.h1.n_cve
            mr_adc.h_orth.n_cva = mr_adc.ncvs * mr_adc.nval * mr_adc.S12.cca.shape[1]

            mr_adc.h_orth.dim_cve = mr_adc.h1.dim_cve
            mr_adc.h_orth.dim_cva = 3 * mr_adc.h_orth.n_cva

            mr_adc.h_orth.dim = (mr_adc.h_orth.n_c + mr_adc.h_orth.dim_c_caa + mr_adc.h_orth.dim_cce + mr_adc.h_orth.dim_cve +
                                 mr_adc.h_orth.dim_cae + mr_adc.h_orth.dim_cca + mr_adc.h_orth.dim_cva)
        else:
            mr_adc.h_orth.dim = mr_adc.h_orth.n_c + mr_adc.h_orth.dim_c_caa + mr_adc.h_orth.dim_cce + mr_adc.h_orth.dim_cae + mr_adc.h_orth.dim_cca

        if mr_adc.nval > 0:
            ### DEBUG
            mr_adc.h_orth.s_c = 0
            mr_adc.h_orth.f_c = mr_adc.h_orth.n_c
            mr_adc.h_orth.s_c_caa = mr_adc.h_orth.f_c
            # mr_adc.h_orth.s_c_caa = 0
            ### DEBUG
            mr_adc.h_orth.f_c_caa = mr_adc.h_orth.s_c_caa + mr_adc.h_orth.dim_c_caa
            mr_adc.h_orth.s_cce = mr_adc.h_orth.f_c_caa
            mr_adc.h_orth.f_cce = mr_adc.h_orth.s_cce + mr_adc.h_orth.dim_cce
            mr_adc.h_orth.s_cve = mr_adc.h_orth.f_cce
            mr_adc.h_orth.f_cve = mr_adc.h_orth.s_cve + mr_adc.h_orth.dim_cve
            mr_adc.h_orth.s_cae = mr_adc.h_orth.f_cve
            mr_adc.h_orth.f_cae = mr_adc.h_orth.s_cae + mr_adc.h_orth.dim_cae
            mr_adc.h_orth.s_cca = mr_adc.h_orth.f_cae
            mr_adc.h_orth.f_cca = mr_adc.h_orth.s_cca + mr_adc.h_orth.dim_cca
            mr_adc.h_orth.s_cva = mr_adc.h_orth.f_cca
            mr_adc.h_orth.f_cva = mr_adc.h_orth.s_cva + mr_adc.h_orth.dim_cva

            mr_adc.h_orth.s_c_caa_aaa = mr_adc.h_orth.s_c_caa
            mr_adc.h_orth.f_c_caa_aaa = mr_adc.h_orth.s_c_caa_aaa + mr_adc.h_orth.n_c_caa
            mr_adc.h_orth.s_c_caa_abb = mr_adc.h_orth.f_c_caa_aaa
            mr_adc.h_orth.f_c_caa_abb = mr_adc.h_orth.s_c_caa_abb + mr_adc.h_orth.n_c_caa
            mr_adc.h_orth.s_c_caa_bab = mr_adc.h_orth.f_c_caa_abb
            mr_adc.h_orth.f_c_caa_bab = mr_adc.h_orth.s_c_caa_bab + mr_adc.h_orth.n_c_caa

            mr_adc.h_orth.s_cce_aaa = mr_adc.h_orth.s_cce
            mr_adc.h_orth.f_cce_aaa = mr_adc.h_orth.s_cce_aaa + mr_adc.h_orth.n_cce_aaa
            mr_adc.h_orth.s_cce_abb = mr_adc.h_orth.f_cce_aaa
            mr_adc.h_orth.f_cce_abb = mr_adc.h_orth.s_cce_abb + mr_adc.h_orth.n_cce_abb

            mr_adc.h_orth.s_cve_aaa = mr_adc.h_orth.s_cve
            mr_adc.h_orth.f_cve_aaa = mr_adc.h_orth.s_cve_aaa + mr_adc.h_orth.n_cve
            mr_adc.h_orth.s_cve_abb = mr_adc.h_orth.f_cve_aaa
            mr_adc.h_orth.f_cve_abb = mr_adc.h_orth.s_cve_abb + mr_adc.h_orth.n_cve
            mr_adc.h_orth.s_cve_bab = mr_adc.h_orth.f_cve_abb
            mr_adc.h_orth.f_cve_bab = mr_adc.h_orth.s_cve_bab + mr_adc.h_orth.n_cve

            mr_adc.h_orth.s_cae_aaa = mr_adc.h_orth.s_cae
            mr_adc.h_orth.f_cae_aaa = mr_adc.h_orth.s_cae_aaa + mr_adc.h_orth.n_cae
            mr_adc.h_orth.s_cae_abb = mr_adc.h_orth.f_cae_aaa
            mr_adc.h_orth.f_cae_abb = mr_adc.h_orth.s_cae_abb + mr_adc.h_orth.n_cae
            mr_adc.h_orth.s_cae_bab = mr_adc.h_orth.f_cae_abb
            mr_adc.h_orth.f_cae_bab = mr_adc.h_orth.s_cae_bab + mr_adc.h_orth.n_cae

            mr_adc.h_orth.s_cca_aaa = mr_adc.h_orth.s_cca
            mr_adc.h_orth.f_cca_aaa = mr_adc.h_orth.s_cca_aaa + mr_adc.h_orth.n_cca_aaa
            mr_adc.h_orth.s_cca_abb = mr_adc.h_orth.f_cca_aaa
            mr_adc.h_orth.f_cca_abb = mr_adc.h_orth.s_cca_abb + mr_adc.h_orth.n_cca_abb

            mr_adc.h_orth.s_cva_aaa = mr_adc.h_orth.s_cva
            mr_adc.h_orth.f_cva_aaa = mr_adc.h_orth.s_cva_aaa + mr_adc.h_orth.n_cva
            mr_adc.h_orth.s_cva_abb = mr_adc.h_orth.f_cva_aaa
            mr_adc.h_orth.f_cva_abb = mr_adc.h_orth.s_cva_abb + mr_adc.h_orth.n_cva
            mr_adc.h_orth.s_cva_bab = mr_adc.h_orth.f_cva_abb
            mr_adc.h_orth.f_cva_bab = mr_adc.h_orth.s_cva_bab + mr_adc.h_orth.n_cva

        else:
            ### DEBUG
            mr_adc.h_orth.s_c = 0
            mr_adc.h_orth.f_c = mr_adc.h_orth.n_c
            mr_adc.h_orth.s_c_caa = mr_adc.h_orth.f_c
            # mr_adc.h_orth.s_c_caa = 0
            ### DEBUG
            mr_adc.h_orth.f_c_caa = mr_adc.h_orth.s_c_caa + mr_adc.h_orth.dim_c_caa
            mr_adc.h_orth.s_cce = mr_adc.h_orth.f_c_caa
            mr_adc.h_orth.f_cce = mr_adc.h_orth.s_cce + mr_adc.h_orth.dim_cce
            mr_adc.h_orth.s_cae = mr_adc.h_orth.f_cce
            mr_adc.h_orth.f_cae = mr_adc.h_orth.s_cae + mr_adc.h_orth.dim_cae
            mr_adc.h_orth.s_cca = mr_adc.h_orth.f_cae
            mr_adc.h_orth.f_cca = mr_adc.h_orth.s_cca + mr_adc.h_orth.dim_cca

            mr_adc.h_orth.s_c_caa_aaa = mr_adc.h_orth.s_c_caa
            mr_adc.h_orth.f_c_caa_aaa = mr_adc.h_orth.s_c_caa_aaa + mr_adc.h_orth.n_c_caa
            mr_adc.h_orth.s_c_caa_abb = mr_adc.h_orth.f_c_caa_aaa
            mr_adc.h_orth.f_c_caa_abb = mr_adc.h_orth.s_c_caa_abb + mr_adc.h_orth.n_c_caa
            mr_adc.h_orth.s_c_caa_bab = mr_adc.h_orth.f_c_caa_abb
            mr_adc.h_orth.f_c_caa_bab = mr_adc.h_orth.s_c_caa_bab + mr_adc.h_orth.n_c_caa

            mr_adc.h_orth.s_cce_aaa = mr_adc.h_orth.s_cce
            mr_adc.h_orth.f_cce_aaa = mr_adc.h_orth.s_cce_aaa + mr_adc.h_orth.n_cce_aaa
            mr_adc.h_orth.s_cce_abb = mr_adc.h_orth.f_cce_aaa
            mr_adc.h_orth.f_cce_abb = mr_adc.h_orth.s_cce_abb + mr_adc.h_orth.n_cce_abb

            mr_adc.h_orth.s_cae_aaa = mr_adc.h_orth.s_cae
            mr_adc.h_orth.f_cae_aaa = mr_adc.h_orth.s_cae_aaa + mr_adc.h_orth.n_cae
            mr_adc.h_orth.s_cae_abb = mr_adc.h_orth.f_cae_aaa
            mr_adc.h_orth.f_cae_abb = mr_adc.h_orth.s_cae_abb + mr_adc.h_orth.n_cae
            mr_adc.h_orth.s_cae_bab = mr_adc.h_orth.f_cae_abb
            mr_adc.h_orth.f_cae_bab = mr_adc.h_orth.s_cae_bab + mr_adc.h_orth.n_cae

            mr_adc.h_orth.s_cca_aaa = mr_adc.h_orth.s_cca
            mr_adc.h_orth.f_cca_aaa = mr_adc.h_orth.s_cca_aaa + mr_adc.h_orth.n_cca_aaa
            mr_adc.h_orth.s_cca_abb = mr_adc.h_orth.f_cca_aaa
            mr_adc.h_orth.f_cca_abb = mr_adc.h_orth.s_cca_abb + mr_adc.h_orth.n_cca_abb

    print("Total dimension of the excitation manifold:                %d" % (mr_adc.h0.dim + mr_adc.h1.dim))
    print("Dimension of the orthogonalized excitation manifold:       %d\n" % (mr_adc.h_orth.dim))
    sys.stdout.flush()

    if (mr_adc.h_orth.dim < mr_adc.nroots):
        mr_adc.nroots = mr_adc.h_orth.dim

    return mr_adc

def compute_M_01_dev(mr_adc):

    start_time = time.time()

    print ("Computing M(h0-h1) blocks...")
    sys.stdout.flush()

    # Einsum definition from kernel
    einsum = mr_adc.interface.einsum
    einsum_type = mr_adc.interface.einsum_type

    # Dimensions
    ncore = mr_adc.ncore
    ncas = mr_adc.ncas
    nocc = mr_adc.nocc
    nextern = mr_adc.nextern

    ncvs = mr_adc.ncvs
    nval = mr_adc.nval

    # Indices
    cvs_tril_ind = np.tril_indices(ncvs, k=-1)

    # MOs Energy
    e_cvs = mr_adc.mo_energy.x
    e_core = mr_adc.mo_energy.c
    e_extern = mr_adc.mo_energy.e

    if nval > 0:
        e_val = mr_adc.mo_energy.v

    # Amplitudes
    t1_ce = mr_adc.t1.ce
    t1_ca = mr_adc.t1.ca
    t1_ae = mr_adc.t1.ae
    t1_caea = mr_adc.t1.caea
    t1_caae = mr_adc.t1.caae
    t1_caaa = mr_adc.t1.caaa
    t1_aaea = mr_adc.t1.aaea
    t1_aaae = mr_adc.t1.aaae

    t1_xa = mr_adc.t1.xa
    t1_xaaa = mr_adc.t1.xaaa

    t1_xe = mr_adc.t1.xe
    t1_xaea = mr_adc.t1.xaea
    t1_xaae = mr_adc.t1.xaae

    if nval > 0:
        t1_ve = mr_adc.t1.ve
        t1_vaea = mr_adc.t1.vaea
        t1_vaae = mr_adc.t1.vaae

        t1_va = mr_adc.t1.va
        t1_vaaa = mr_adc.t1.vaaa

    # One-electron integrals
    h_aa = mr_adc.h1eff.aa
    h_ae = mr_adc.h1eff.ae

    h_xe = mr_adc.h1eff.xe
    h_xa = mr_adc.h1eff.xa

    if nval > 0:
        h_ve = mr_adc.h1eff.ve
        h_va = mr_adc.h1eff.va

    # Two-electrons integrals
    v_aaaa = mr_adc.v2e.aaaa
    v_aaae = mr_adc.v2e.aaae

    v_xaxa = mr_adc.v2e.xaxa
    v_xaax = mr_adc.v2e.xaax

    v_xaxe = mr_adc.v2e.xaxe
    v_xaex = mr_adc.v2e.xaex

    v_xaea = mr_adc.v2e.xaea
    v_xaae = mr_adc.v2e.xaae
    v_xxxe = mr_adc.v2e.xxxe

    v_xxxa = mr_adc.v2e.xxxa
    v_xaaa = mr_adc.v2e.xaaa

    if nval > 0:
        v_vxxe = mr_adc.v2e.vxxe
        v_xvxe = mr_adc.v2e.xvxe

        v_vxxa = mr_adc.v2e.vxxa
        v_xvxa = mr_adc.v2e.xvxa

        v_vaea = mr_adc.v2e.vaea
        v_vaae = mr_adc.v2e.vaae

        v_vaaa = mr_adc.v2e.vaaa

    # Reduced Density Matrices
    rdm_ca = mr_adc.rdm.ca
    rdm_ccaa = mr_adc.rdm.ccaa
    rdm_cccaaa = mr_adc.rdm.cccaaa

    # Excitation Spaces
    dim_cce = mr_adc.h1.dim_cce
    dim_cae = mr_adc.h1.dim_cae
    dim_cca = mr_adc.h1.dim_cca

    n_cce_aaa = mr_adc.h1.n_cce_aaa
    n_cce_abb = mr_adc.h1.n_cce_abb
    n_cae = mr_adc.h1.n_cae
    n_cca_aaa = mr_adc.h1.n_cca_aaa
    n_cca_abb = mr_adc.h1.n_cca_abb

    if nval > 0:
        dim_cve = mr_adc.h1.dim_cve
        dim_cva = mr_adc.h1.dim_cva

        n_cve = mr_adc.h1.n_cve
        n_cva = mr_adc.h1.n_cva

    # C - CCE
    M_C_CCE_a_abb  = einsum('KLIB->IKLB', v_xxxe, optimize = einsum_type).copy()
    M_C_CCE_a_abb -= einsum('LB,IK->IKLB', h_xe, np.identity(ncvs), optimize = einsum_type)
    M_C_CCE_a_abb -= einsum('B,IK,LB->IKLB', e_extern, np.identity(ncvs), t1_xe, optimize = einsum_type)
    M_C_CCE_a_abb += einsum('L,IK,LB->IKLB', e_cvs, np.identity(ncvs), t1_xe, optimize = einsum_type)
    M_C_CCE_a_abb -= einsum('IK,LxBy,yx->IKLB', np.identity(ncvs), v_xaea, rdm_ca, optimize = einsum_type)
    M_C_CCE_a_abb += 1/2 * einsum('IK,LxyB,yx->IKLB', np.identity(ncvs), v_xaae, rdm_ca, optimize = einsum_type)
    M_C_CCE_a_abb -= einsum('B,IK,LxBy,yx->IKLB', e_extern, np.identity(ncvs), t1_xaea, rdm_ca, optimize = einsum_type)
    M_C_CCE_a_abb += 1/2 * einsum('B,IK,LxyB,yx->IKLB', e_extern, np.identity(ncvs), t1_xaae, rdm_ca, optimize = einsum_type)
    M_C_CCE_a_abb += einsum('L,IK,LxBy,yx->IKLB', e_cvs, np.identity(ncvs), t1_xaea, rdm_ca, optimize = einsum_type)
    M_C_CCE_a_abb -= 1/2 * einsum('L,IK,LxyB,yx->IKLB', e_cvs, np.identity(ncvs), t1_xaae, rdm_ca, optimize = einsum_type)
    M_C_CCE_a_abb += einsum('xy,IK,LxBz,zy->IKLB', h_aa, np.identity(ncvs), t1_xaea, rdm_ca, optimize = einsum_type)
    M_C_CCE_a_abb -= 1/2 * einsum('xy,IK,LxzB,zy->IKLB', h_aa, np.identity(ncvs), t1_xaae, rdm_ca, optimize = einsum_type)
    M_C_CCE_a_abb -= einsum('xy,IK,LzBx,yz->IKLB', h_aa, np.identity(ncvs), t1_xaea, rdm_ca, optimize = einsum_type)
    M_C_CCE_a_abb += 1/2 * einsum('xy,IK,LzxB,yz->IKLB', h_aa, np.identity(ncvs), t1_xaae, rdm_ca, optimize = einsum_type)
    M_C_CCE_a_abb += einsum('IK,LxBy,xzwu,yzwu->IKLB', np.identity(ncvs), t1_xaea, v_aaaa, rdm_ccaa, optimize = einsum_type)
    M_C_CCE_a_abb -= einsum('IK,LxBy,yzwu,xzwu->IKLB', np.identity(ncvs), t1_xaea, v_aaaa, rdm_ccaa, optimize = einsum_type)
    M_C_CCE_a_abb -= 1/2 * einsum('IK,LxyB,xzwu,yzwu->IKLB', np.identity(ncvs), t1_xaae, v_aaaa, rdm_ccaa, optimize = einsum_type)
    M_C_CCE_a_abb += 1/2 * einsum('IK,LxyB,yzwu,xzwu->IKLB', np.identity(ncvs), t1_xaae, v_aaaa, rdm_ccaa, optimize = einsum_type)

    M_C_CCE_a_aaa = M_C_CCE_a_abb - M_C_CCE_a_abb.transpose(0,2,1,3)

    ## Reshape tensors to matrix form
    M_C_CCE_a_aaa = M_C_CCE_a_aaa[:, cvs_tril_ind[0], cvs_tril_ind[1]]

    M_C_CCE_a_aaa = M_C_CCE_a_aaa.reshape(ncvs, -1)
    M_C_CCE_a_abb = M_C_CCE_a_abb.reshape(ncvs, -1)

    ## Building C-CCE matrix
    s_c_cce_aaa = 0
    f_c_cce_aaa = s_c_cce_aaa + n_cce_aaa
    s_c_cce_abb = f_c_cce_aaa
    f_c_cce_abb = s_c_cce_abb + n_cce_abb

    M_C_CCE = np.zeros((ncvs, dim_cce))
    M_C_CCE[:, s_c_cce_aaa:f_c_cce_aaa] = M_C_CCE_a_aaa.copy()
    M_C_CCE[:, s_c_cce_abb:f_c_cce_abb] = M_C_CCE_a_abb.copy()

    if nval > 0:
        # C - CVE
        M_C_CVE_a_abb  = einsum('KLIB->IKLB', v_xvxe, optimize = einsum_type).copy()
        M_C_CVE_a_abb -= einsum('LB,IK->IKLB', h_ve, np.identity(ncvs), optimize = einsum_type)
        M_C_CVE_a_abb -= einsum('B,IK,LB->IKLB', e_extern, np.identity(ncvs), t1_ve, optimize = einsum_type)
        M_C_CVE_a_abb += einsum('L,IK,LB->IKLB', e_val, np.identity(ncvs), t1_ve, optimize = einsum_type)
        M_C_CVE_a_abb -= einsum('IK,LxBy,yx->IKLB', np.identity(ncvs), v_vaea, rdm_ca, optimize = einsum_type)
        M_C_CVE_a_abb += 1/2 * einsum('IK,LxyB,yx->IKLB', np.identity(ncvs), v_vaae, rdm_ca, optimize = einsum_type)
        M_C_CVE_a_abb -= einsum('B,IK,LxBy,yx->IKLB', e_extern, np.identity(ncvs), t1_vaea, rdm_ca, optimize = einsum_type)
        M_C_CVE_a_abb += 1/2 * einsum('B,IK,LxyB,yx->IKLB', e_extern, np.identity(ncvs), t1_vaae, rdm_ca, optimize = einsum_type)
        M_C_CVE_a_abb += einsum('L,IK,LxBy,yx->IKLB', e_val, np.identity(ncvs), t1_vaea, rdm_ca, optimize = einsum_type)
        M_C_CVE_a_abb -= 1/2 * einsum('L,IK,LxyB,yx->IKLB', e_val, np.identity(ncvs), t1_vaae, rdm_ca, optimize = einsum_type)
        M_C_CVE_a_abb += einsum('xy,IK,LxBz,zy->IKLB', h_aa, np.identity(ncvs), t1_vaea, rdm_ca, optimize = einsum_type)
        M_C_CVE_a_abb -= 1/2 * einsum('xy,IK,LxzB,zy->IKLB', h_aa, np.identity(ncvs), t1_vaae, rdm_ca, optimize = einsum_type)
        M_C_CVE_a_abb -= einsum('xy,IK,LzBx,yz->IKLB', h_aa, np.identity(ncvs), t1_vaea, rdm_ca, optimize = einsum_type)
        M_C_CVE_a_abb += 1/2 * einsum('xy,IK,LzxB,yz->IKLB', h_aa, np.identity(ncvs), t1_vaae, rdm_ca, optimize = einsum_type)
        M_C_CVE_a_abb += einsum('IK,LxBy,xzwu,yzwu->IKLB', np.identity(ncvs), t1_vaea, v_aaaa, rdm_ccaa, optimize = einsum_type)
        M_C_CVE_a_abb -= einsum('IK,LxBy,yzwu,xzwu->IKLB', np.identity(ncvs), t1_vaea, v_aaaa, rdm_ccaa, optimize = einsum_type)
        M_C_CVE_a_abb -= 1/2 * einsum('IK,LxyB,xzwu,yzwu->IKLB', np.identity(ncvs), t1_vaae, v_aaaa, rdm_ccaa, optimize = einsum_type)
        M_C_CVE_a_abb += 1/2 * einsum('IK,LxyB,yzwu,xzwu->IKLB', np.identity(ncvs), t1_vaae, v_aaaa, rdm_ccaa, optimize = einsum_type)

        M_C_CVE_a_bab =- einsum('LKIB->IKLB', v_vxxe, optimize = einsum_type).copy()

        M_C_CVE_a_aaa = np.ascontiguousarray(M_C_CVE_a_abb + M_C_CVE_a_bab)

        ## Reshape tensors to matrix form
        M_C_CVE_a_aaa = M_C_CVE_a_aaa.reshape(ncvs, -1)
        M_C_CVE_a_abb = M_C_CVE_a_abb.reshape(ncvs, -1)
        M_C_CVE_a_bab = M_C_CVE_a_bab.reshape(ncvs, -1)

        ## Building C-CVE matrix
        n_cve = mr_adc.h1.n_cve

        s_c_cve_aaa = 0
        f_c_cve_aaa = s_c_cve_aaa + n_cve
        s_c_cve_abb = f_c_cve_aaa
        f_c_cve_abb = s_c_cve_abb + n_cve
        s_c_cve_bab = f_c_cve_abb
        f_c_cve_bab = s_c_cve_bab + n_cve

        M_C_CVE = np.zeros((ncvs, dim_cve))
        M_C_CVE[:, s_c_cve_aaa:f_c_cve_aaa] = M_C_CVE_a_aaa.copy()
        M_C_CVE[:, s_c_cve_abb:f_c_cve_abb] = M_C_CVE_a_abb.copy()
        M_C_CVE[:, s_c_cve_bab:f_c_cve_bab] = M_C_CVE_a_bab.copy()

    # C - CAE
    M_C_CAE_a_aaa =- 1/2 * einsum('KxBI,Yx->IKYB', v_xaex, rdm_ca, optimize = einsum_type)
    M_C_CAE_a_aaa += 1/2 * einsum('KxIB,Yx->IKYB', v_xaxe, rdm_ca, optimize = einsum_type)
    M_C_CAE_a_aaa -= 1/2 * einsum('xB,IK,Yx->IKYB', h_ae, np.identity(ncvs), rdm_ca, optimize = einsum_type)
    M_C_CAE_a_aaa -= 1/2 * einsum('IK,xyzB,Yzyx->IKYB', np.identity(ncvs), v_aaae, rdm_ccaa, optimize = einsum_type)
    M_C_CAE_a_aaa -= 1/2 * einsum('B,IK,xB,Yx->IKYB', e_extern, np.identity(ncvs), t1_ae, rdm_ca, optimize = einsum_type)
    M_C_CAE_a_aaa -= 1/2 * einsum('B,IK,xyzB,Yzyx->IKYB', e_extern, np.identity(ncvs), t1_aaae, rdm_ccaa, optimize = einsum_type)
    M_C_CAE_a_aaa += 1/2 * einsum('xy,IK,xB,Yy->IKYB', h_aa, np.identity(ncvs), t1_ae, rdm_ca, optimize = einsum_type)
    M_C_CAE_a_aaa -= 1/2 * einsum('xy,IK,zwxB,Yywz->IKYB', h_aa, np.identity(ncvs), t1_aaae, rdm_ccaa, optimize = einsum_type)
    M_C_CAE_a_aaa += 1/2 * einsum('xy,IK,xzwB,Ywzy->IKYB', h_aa, np.identity(ncvs), t1_aaae, rdm_ccaa, optimize = einsum_type)
    M_C_CAE_a_aaa += 1/2 * einsum('xy,IK,zxwB,Ywyz->IKYB', h_aa, np.identity(ncvs), t1_aaae, rdm_ccaa, optimize = einsum_type)
    M_C_CAE_a_aaa += 1/2 * einsum('IK,xB,xyzw,Yyzw->IKYB', np.identity(ncvs), t1_ae, v_aaaa, rdm_ccaa, optimize = einsum_type)
    M_C_CAE_a_aaa += 1/12 * einsum('IK,xyzB,zwuv,Yuvxyw->IKYB', np.identity(ncvs), t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    M_C_CAE_a_aaa += 1/12 * einsum('IK,xyzB,zwuv,Yuvxwy->IKYB', np.identity(ncvs), t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    M_C_CAE_a_aaa -= 5/12 * einsum('IK,xyzB,zwuv,Yuvyxw->IKYB', np.identity(ncvs), t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    M_C_CAE_a_aaa += 1/12 * einsum('IK,xyzB,zwuv,Yuvywx->IKYB', np.identity(ncvs), t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    M_C_CAE_a_aaa += 1/12 * einsum('IK,xyzB,zwuv,Yuvwxy->IKYB', np.identity(ncvs), t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    M_C_CAE_a_aaa += 1/12 * einsum('IK,xyzB,zwuv,Yuvwyx->IKYB', np.identity(ncvs), t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    M_C_CAE_a_aaa += 1/2 * einsum('IK,xyzB,xywu,Yzuw->IKYB', np.identity(ncvs), t1_aaae, v_aaaa, rdm_ccaa, optimize = einsum_type)
    M_C_CAE_a_aaa += 5/12 * einsum('IK,xyzB,xwuv,Yzwyuv->IKYB', np.identity(ncvs), t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    M_C_CAE_a_aaa -= 1/12 * einsum('IK,xyzB,xwuv,Yzwyvu->IKYB', np.identity(ncvs), t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    M_C_CAE_a_aaa -= 1/12 * einsum('IK,xyzB,xwuv,Yzwuyv->IKYB', np.identity(ncvs), t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    M_C_CAE_a_aaa -= 1/12 * einsum('IK,xyzB,xwuv,Yzwuvy->IKYB', np.identity(ncvs), t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    M_C_CAE_a_aaa -= 1/12 * einsum('IK,xyzB,xwuv,Yzwvyu->IKYB', np.identity(ncvs), t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    M_C_CAE_a_aaa -= 1/12 * einsum('IK,xyzB,xwuv,Yzwvuy->IKYB', np.identity(ncvs), t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    M_C_CAE_a_aaa -= 1/12 * einsum('IK,xyzB,ywuv,Yzwxuv->IKYB', np.identity(ncvs), t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    M_C_CAE_a_aaa -= 1/12 * einsum('IK,xyzB,ywuv,Yzwxvu->IKYB', np.identity(ncvs), t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    M_C_CAE_a_aaa += 5/12 * einsum('IK,xyzB,ywuv,Yzwuxv->IKYB', np.identity(ncvs), t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    M_C_CAE_a_aaa -= 1/12 * einsum('IK,xyzB,ywuv,Yzwuvx->IKYB', np.identity(ncvs), t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    M_C_CAE_a_aaa -= 1/12 * einsum('IK,xyzB,ywuv,Yzwvxu->IKYB', np.identity(ncvs), t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    M_C_CAE_a_aaa -= 1/12 * einsum('IK,xyzB,ywuv,Yzwvux->IKYB', np.identity(ncvs), t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)

    M_C_CAE_a_abb  = 1/2 * einsum('KxIB,Yx->IKYB', v_xaxe, rdm_ca, optimize = einsum_type)
    M_C_CAE_a_abb -= 1/2 * einsum('xB,IK,Yx->IKYB', h_ae, np.identity(ncvs), rdm_ca, optimize = einsum_type)
    M_C_CAE_a_abb -= 1/2 * einsum('IK,xyzB,Yzyx->IKYB', np.identity(ncvs), v_aaae, rdm_ccaa, optimize = einsum_type)
    M_C_CAE_a_abb -= 1/2 * einsum('B,IK,xB,Yx->IKYB', e_extern, np.identity(ncvs), t1_ae, rdm_ca, optimize = einsum_type)
    M_C_CAE_a_abb -= 1/2 * einsum('B,IK,xyzB,Yzyx->IKYB', e_extern, np.identity(ncvs), t1_aaae, rdm_ccaa, optimize = einsum_type)
    M_C_CAE_a_abb += 1/2 * einsum('xy,IK,xB,Yy->IKYB', h_aa, np.identity(ncvs), t1_ae, rdm_ca, optimize = einsum_type)
    M_C_CAE_a_abb -= 1/2 * einsum('xy,IK,zwxB,Yywz->IKYB', h_aa, np.identity(ncvs), t1_aaae, rdm_ccaa, optimize = einsum_type)
    M_C_CAE_a_abb += 1/2 * einsum('xy,IK,xzwB,Ywzy->IKYB', h_aa, np.identity(ncvs), t1_aaae, rdm_ccaa, optimize = einsum_type)
    M_C_CAE_a_abb += 1/2 * einsum('xy,IK,zxwB,Ywyz->IKYB', h_aa, np.identity(ncvs), t1_aaae, rdm_ccaa, optimize = einsum_type)
    M_C_CAE_a_abb += 1/2 * einsum('IK,xB,xyzw,Yyzw->IKYB', np.identity(ncvs), t1_ae, v_aaaa, rdm_ccaa, optimize = einsum_type)
    M_C_CAE_a_abb += 1/12 * einsum('IK,xyzB,zwuv,Yuvxyw->IKYB', np.identity(ncvs), t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    M_C_CAE_a_abb += 1/12 * einsum('IK,xyzB,zwuv,Yuvxwy->IKYB', np.identity(ncvs), t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    M_C_CAE_a_abb -= 5/12 * einsum('IK,xyzB,zwuv,Yuvyxw->IKYB', np.identity(ncvs), t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    M_C_CAE_a_abb += 1/12 * einsum('IK,xyzB,zwuv,Yuvywx->IKYB', np.identity(ncvs), t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    M_C_CAE_a_abb += 1/12 * einsum('IK,xyzB,zwuv,Yuvwxy->IKYB', np.identity(ncvs), t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    M_C_CAE_a_abb += 1/12 * einsum('IK,xyzB,zwuv,Yuvwyx->IKYB', np.identity(ncvs), t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    M_C_CAE_a_abb += 1/2 * einsum('IK,xyzB,xywu,Yzuw->IKYB', np.identity(ncvs), t1_aaae, v_aaaa, rdm_ccaa, optimize = einsum_type)
    M_C_CAE_a_abb += 5/12 * einsum('IK,xyzB,xwuv,Yzwyuv->IKYB', np.identity(ncvs), t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    M_C_CAE_a_abb -= 1/12 * einsum('IK,xyzB,xwuv,Yzwyvu->IKYB', np.identity(ncvs), t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    M_C_CAE_a_abb -= 1/12 * einsum('IK,xyzB,xwuv,Yzwuyv->IKYB', np.identity(ncvs), t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    M_C_CAE_a_abb -= 1/12 * einsum('IK,xyzB,xwuv,Yzwuvy->IKYB', np.identity(ncvs), t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    M_C_CAE_a_abb -= 1/12 * einsum('IK,xyzB,xwuv,Yzwvyu->IKYB', np.identity(ncvs), t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    M_C_CAE_a_abb -= 1/12 * einsum('IK,xyzB,xwuv,Yzwvuy->IKYB', np.identity(ncvs), t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    M_C_CAE_a_abb -= 1/12 * einsum('IK,xyzB,ywuv,Yzwxuv->IKYB', np.identity(ncvs), t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    M_C_CAE_a_abb -= 1/12 * einsum('IK,xyzB,ywuv,Yzwxvu->IKYB', np.identity(ncvs), t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    M_C_CAE_a_abb += 5/12 * einsum('IK,xyzB,ywuv,Yzwuxv->IKYB', np.identity(ncvs), t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    M_C_CAE_a_abb -= 1/12 * einsum('IK,xyzB,ywuv,Yzwuvx->IKYB', np.identity(ncvs), t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    M_C_CAE_a_abb -= 1/12 * einsum('IK,xyzB,ywuv,Yzwvxu->IKYB', np.identity(ncvs), t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    M_C_CAE_a_abb -= 1/12 * einsum('IK,xyzB,ywuv,Yzwvux->IKYB', np.identity(ncvs), t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)

    M_C_CAE_a_bab = M_C_CAE_a_aaa - M_C_CAE_a_abb

    ## Reshape tensors to matrix form
    M_C_CAE_a_aaa = M_C_CAE_a_aaa.reshape(ncvs, -1)
    M_C_CAE_a_abb = M_C_CAE_a_abb.reshape(ncvs, -1)
    M_C_CAE_a_bab = M_C_CAE_a_bab.reshape(ncvs, -1)

    ## Building C-CCE matrix
    s_c_cae_aaa = 0
    f_c_cae_aaa = s_c_cae_aaa + n_cae
    s_c_cae_abb = f_c_cae_aaa
    f_c_cae_abb = s_c_cae_abb + n_cae
    s_c_cae_bab = f_c_cae_abb
    f_c_cae_bab = s_c_cae_bab + n_cae

    M_C_CAE = np.zeros((ncvs, dim_cae))
    M_C_CAE[:, s_c_cae_aaa:f_c_cae_aaa] = M_C_CAE_a_aaa.copy()
    M_C_CAE[:, s_c_cae_abb:f_c_cae_abb] = M_C_CAE_a_abb.copy()
    M_C_CAE[:, s_c_cae_bab:f_c_cae_bab] = M_C_CAE_a_bab.copy()

    # C - CCA
    M_C_CCA_a_abb  = einsum('KLIY->IKLY', v_xxxa, optimize = einsum_type).copy()
    M_C_CCA_a_abb -= einsum('LY,IK->IKLY', h_xa, np.identity(ncvs), optimize = einsum_type)
    M_C_CCA_a_abb -= 1/2 * einsum('KLIx,xY->IKLY', v_xxxa, rdm_ca, optimize = einsum_type)
    M_C_CCA_a_abb += einsum('L,IK,LY->IKLY', e_cvs, np.identity(ncvs), t1_xa, optimize = einsum_type)
    M_C_CCA_a_abb += 1/2 * einsum('Lx,IK,xY->IKLY', h_xa, np.identity(ncvs), rdm_ca, optimize = einsum_type)
    M_C_CCA_a_abb -= einsum('Yx,IK,Lx->IKLY', h_aa, np.identity(ncvs), t1_xa, optimize = einsum_type)
    M_C_CCA_a_abb -= einsum('IK,LxYy,yx->IKLY', np.identity(ncvs), v_xaaa, rdm_ca, optimize = einsum_type)
    M_C_CCA_a_abb += 1/2 * einsum('IK,LxyY,yx->IKLY', np.identity(ncvs), v_xaaa, rdm_ca, optimize = einsum_type)
    M_C_CCA_a_abb += 1/2 * einsum('IK,Lxyz,Yxyz->IKLY', np.identity(ncvs), v_xaaa, rdm_ccaa, optimize = einsum_type)
    M_C_CCA_a_abb -= 1/2 * einsum('L,IK,Lx,xY->IKLY', e_cvs, np.identity(ncvs), t1_xa, rdm_ca, optimize = einsum_type)
    M_C_CCA_a_abb += einsum('L,IK,LxYy,yx->IKLY', e_cvs, np.identity(ncvs), t1_xaaa, rdm_ca, optimize = einsum_type)
    M_C_CCA_a_abb -= 1/2 * einsum('L,IK,LxyY,yx->IKLY', e_cvs, np.identity(ncvs), t1_xaaa, rdm_ca, optimize = einsum_type)
    M_C_CCA_a_abb -= 1/2 * einsum('L,IK,Lxyz,Yxyz->IKLY', e_cvs, np.identity(ncvs), t1_xaaa, rdm_ccaa, optimize = einsum_type)
    M_C_CCA_a_abb -= einsum('Yx,IK,Lyxz,zy->IKLY', h_aa, np.identity(ncvs), t1_xaaa, rdm_ca, optimize = einsum_type)
    M_C_CCA_a_abb += 1/2 * einsum('Yx,IK,Lyzx,zy->IKLY', h_aa, np.identity(ncvs), t1_xaaa, rdm_ca, optimize = einsum_type)
    M_C_CCA_a_abb += 1/2 * einsum('xy,IK,Lx,yY->IKLY', h_aa, np.identity(ncvs), t1_xa, rdm_ca, optimize = einsum_type)
    M_C_CCA_a_abb += einsum('xy,IK,LxYz,zy->IKLY', h_aa, np.identity(ncvs), t1_xaaa, rdm_ca, optimize = einsum_type)
    M_C_CCA_a_abb -= 1/2 * einsum('xy,IK,LxzY,zy->IKLY', h_aa, np.identity(ncvs), t1_xaaa, rdm_ca, optimize = einsum_type)
    M_C_CCA_a_abb -= 1/2 * einsum('xy,IK,Lxzw,Yyzw->IKLY', h_aa, np.identity(ncvs), t1_xaaa, rdm_ccaa, optimize = einsum_type)
    M_C_CCA_a_abb -= einsum('xy,IK,LzYx,yz->IKLY', h_aa, np.identity(ncvs), t1_xaaa, rdm_ca, optimize = einsum_type)
    M_C_CCA_a_abb += 1/2 * einsum('xy,IK,LzxY,yz->IKLY', h_aa, np.identity(ncvs), t1_xaaa, rdm_ca, optimize = einsum_type)
    M_C_CCA_a_abb += 1/2 * einsum('xy,IK,Lzxw,Yzyw->IKLY', h_aa, np.identity(ncvs), t1_xaaa, rdm_ccaa, optimize = einsum_type)
    M_C_CCA_a_abb += 1/2 * einsum('xy,IK,Lzwx,Yzwy->IKLY', h_aa, np.identity(ncvs), t1_xaaa, rdm_ccaa, optimize = einsum_type)
    M_C_CCA_a_abb -= einsum('IK,Lx,Yyxz,yz->IKLY', np.identity(ncvs), t1_xa, v_aaaa, rdm_ca, optimize = einsum_type)
    M_C_CCA_a_abb += 1/2 * einsum('IK,Lx,Yyzx,yz->IKLY', np.identity(ncvs), t1_xa, v_aaaa, rdm_ca, optimize = einsum_type)
    M_C_CCA_a_abb += 1/2 * einsum('IK,Lx,xyzw,Yyzw->IKLY', np.identity(ncvs), t1_xa, v_aaaa, rdm_ccaa, optimize = einsum_type)
    M_C_CCA_a_abb += einsum('IK,LxYy,xzwu,yzwu->IKLY', np.identity(ncvs), t1_xaaa, v_aaaa, rdm_ccaa, optimize = einsum_type)
    M_C_CCA_a_abb -= einsum('IK,LxYy,yzwu,xzwu->IKLY', np.identity(ncvs), t1_xaaa, v_aaaa, rdm_ccaa, optimize = einsum_type)
    M_C_CCA_a_abb -= 1/2 * einsum('IK,LxyY,xzwu,yzwu->IKLY', np.identity(ncvs), t1_xaaa, v_aaaa, rdm_ccaa, optimize = einsum_type)
    M_C_CCA_a_abb += 1/2 * einsum('IK,LxyY,yzwu,xzwu->IKLY', np.identity(ncvs), t1_xaaa, v_aaaa, rdm_ccaa, optimize = einsum_type)
    M_C_CCA_a_abb -= 1/2 * einsum('IK,Lxyz,Yxwu,yzwu->IKLY', np.identity(ncvs), t1_xaaa, v_aaaa, rdm_ccaa, optimize = einsum_type)
    M_C_CCA_a_abb -= einsum('IK,Lxyz,Ywyz,wx->IKLY', np.identity(ncvs), t1_xaaa, v_aaaa, rdm_ca, optimize = einsum_type)
    M_C_CCA_a_abb -= einsum('IK,Lxyz,Ywyu,xuzw->IKLY', np.identity(ncvs), t1_xaaa, v_aaaa, rdm_ccaa, optimize = einsum_type)
    M_C_CCA_a_abb += 1/2 * einsum('IK,Lxyz,Ywzy,wx->IKLY', np.identity(ncvs), t1_xaaa, v_aaaa, rdm_ca, optimize = einsum_type)
    M_C_CCA_a_abb += 1/2 * einsum('IK,Lxyz,Ywzu,xuyw->IKLY', np.identity(ncvs), t1_xaaa, v_aaaa, rdm_ccaa, optimize = einsum_type)
    M_C_CCA_a_abb += 1/2 * einsum('IK,Lxyz,Ywuy,xuzw->IKLY', np.identity(ncvs), t1_xaaa, v_aaaa, rdm_ccaa, optimize = einsum_type)
    M_C_CCA_a_abb += 1/2 * einsum('IK,Lxyz,Ywuz,xuwy->IKLY', np.identity(ncvs), t1_xaaa, v_aaaa, rdm_ccaa, optimize = einsum_type)
    M_C_CCA_a_abb -= 5/12 * einsum('IK,Lxyz,xwuv,yzwYuv->IKLY', np.identity(ncvs), t1_xaaa, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    M_C_CCA_a_abb += 1/12 * einsum('IK,Lxyz,xwuv,yzwYvu->IKLY', np.identity(ncvs), t1_xaaa, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    M_C_CCA_a_abb += 1/12 * einsum('IK,Lxyz,xwuv,yzwuYv->IKLY', np.identity(ncvs), t1_xaaa, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    M_C_CCA_a_abb += 1/12 * einsum('IK,Lxyz,xwuv,yzwuvY->IKLY', np.identity(ncvs), t1_xaaa, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    M_C_CCA_a_abb += 1/12 * einsum('IK,Lxyz,xwuv,yzwvYu->IKLY', np.identity(ncvs), t1_xaaa, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    M_C_CCA_a_abb += 1/12 * einsum('IK,Lxyz,xwuv,yzwvuY->IKLY', np.identity(ncvs), t1_xaaa, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    M_C_CCA_a_abb += 1/2 * einsum('IK,Lxyz,yzwu,Yxwu->IKLY', np.identity(ncvs), t1_xaaa, v_aaaa, rdm_ccaa, optimize = einsum_type)
    M_C_CCA_a_abb -= 1/12 * einsum('IK,Lxyz,ywuv,zuvYxw->IKLY', np.identity(ncvs), t1_xaaa, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    M_C_CCA_a_abb -= 1/12 * einsum('IK,Lxyz,ywuv,zuvYwx->IKLY', np.identity(ncvs), t1_xaaa, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    M_C_CCA_a_abb += 5/12 * einsum('IK,Lxyz,ywuv,zuvxYw->IKLY', np.identity(ncvs), t1_xaaa, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    M_C_CCA_a_abb -= 1/12 * einsum('IK,Lxyz,ywuv,zuvxwY->IKLY', np.identity(ncvs), t1_xaaa, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    M_C_CCA_a_abb -= 1/12 * einsum('IK,Lxyz,ywuv,zuvwYx->IKLY', np.identity(ncvs), t1_xaaa, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    M_C_CCA_a_abb -= 1/12 * einsum('IK,Lxyz,ywuv,zuvwxY->IKLY', np.identity(ncvs), t1_xaaa, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    M_C_CCA_a_abb += 5/12 * einsum('IK,Lxyz,zwuv,yuvYxw->IKLY', np.identity(ncvs), t1_xaaa, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    M_C_CCA_a_abb -= 1/12 * einsum('IK,Lxyz,zwuv,yuvYwx->IKLY', np.identity(ncvs), t1_xaaa, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    M_C_CCA_a_abb -= 1/12 * einsum('IK,Lxyz,zwuv,yuvxYw->IKLY', np.identity(ncvs), t1_xaaa, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    M_C_CCA_a_abb -= 1/12 * einsum('IK,Lxyz,zwuv,yuvxwY->IKLY', np.identity(ncvs), t1_xaaa, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    M_C_CCA_a_abb -= 1/12 * einsum('IK,Lxyz,zwuv,yuvwYx->IKLY', np.identity(ncvs), t1_xaaa, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    M_C_CCA_a_abb -= 1/12 * einsum('IK,Lxyz,zwuv,yuvwxY->IKLY', np.identity(ncvs), t1_xaaa, v_aaaa, rdm_cccaaa, optimize = einsum_type)

    M_C_CCA_a_aaa = M_C_CCA_a_abb - M_C_CCA_a_abb.transpose(0,2,1,3)

    ## Reshape tensors to matrix form
    M_C_CCA_a_aaa = M_C_CCA_a_aaa[:, cvs_tril_ind[0], cvs_tril_ind[1]]

    M_C_CCA_a_aaa = M_C_CCA_a_aaa.reshape(ncvs, -1)
    M_C_CCA_a_abb = M_C_CCA_a_abb.reshape(ncvs, -1)

    ## Building C-CCA matrix
    s_c_cca_aaa = 0
    f_c_cca_aaa = s_c_cca_aaa + n_cca_aaa
    s_c_cca_abb = f_c_cca_aaa
    f_c_cca_abb = s_c_cca_abb + n_cca_abb

    M_C_CCA = np.zeros((ncvs, dim_cca))
    M_C_CCA[:, s_c_cca_aaa:f_c_cca_aaa] = M_C_CCA_a_aaa.copy()
    M_C_CCA[:, s_c_cca_abb:f_c_cca_abb] = M_C_CCA_a_abb.copy()

    if nval > 0:
        M_C_CVA_a_abb  = einsum('KLIY->IKLY', v_xvxa, optimize = einsum_type).copy()
        M_C_CVA_a_abb -= einsum('LY,IK->IKLY', h_va, np.identity(ncvs), optimize = einsum_type)
        M_C_CVA_a_abb -= 1/2 * einsum('KLIx,xY->IKLY', v_xvxa, rdm_ca, optimize = einsum_type)
        M_C_CVA_a_abb += einsum('L,IK,LY->IKLY', e_val, np.identity(ncvs), t1_va, optimize = einsum_type)
        M_C_CVA_a_abb += 1/2 * einsum('Lx,IK,xY->IKLY', h_va, np.identity(ncvs), rdm_ca, optimize = einsum_type)
        M_C_CVA_a_abb -= einsum('Yx,IK,Lx->IKLY', h_aa, np.identity(ncvs), t1_va, optimize = einsum_type)
        M_C_CVA_a_abb -= einsum('IK,LxYy,yx->IKLY', np.identity(ncvs), v_vaaa, rdm_ca, optimize = einsum_type)
        M_C_CVA_a_abb += 1/2 * einsum('IK,LxyY,yx->IKLY', np.identity(ncvs), v_vaaa, rdm_ca, optimize = einsum_type)
        M_C_CVA_a_abb += 1/2 * einsum('IK,Lxyz,Yxyz->IKLY', np.identity(ncvs), v_vaaa, rdm_ccaa, optimize = einsum_type)
        M_C_CVA_a_abb -= 1/2 * einsum('L,IK,Lx,xY->IKLY', e_val, np.identity(ncvs), t1_va, rdm_ca, optimize = einsum_type)
        M_C_CVA_a_abb += einsum('L,IK,LxYy,yx->IKLY', e_val, np.identity(ncvs), t1_vaaa, rdm_ca, optimize = einsum_type)
        M_C_CVA_a_abb -= 1/2 * einsum('L,IK,LxyY,yx->IKLY', e_val, np.identity(ncvs), t1_vaaa, rdm_ca, optimize = einsum_type)
        M_C_CVA_a_abb -= 1/2 * einsum('L,IK,Lxyz,Yxyz->IKLY', e_val, np.identity(ncvs), t1_vaaa, rdm_ccaa, optimize = einsum_type)
        M_C_CVA_a_abb -= einsum('Yx,IK,Lyxz,zy->IKLY', h_aa, np.identity(ncvs), t1_vaaa, rdm_ca, optimize = einsum_type)
        M_C_CVA_a_abb += 1/2 * einsum('Yx,IK,Lyzx,zy->IKLY', h_aa, np.identity(ncvs), t1_vaaa, rdm_ca, optimize = einsum_type)
        M_C_CVA_a_abb += 1/2 * einsum('xy,IK,Lx,yY->IKLY', h_aa, np.identity(ncvs), t1_va, rdm_ca, optimize = einsum_type)
        M_C_CVA_a_abb += einsum('xy,IK,LxYz,zy->IKLY', h_aa, np.identity(ncvs), t1_vaaa, rdm_ca, optimize = einsum_type)
        M_C_CVA_a_abb -= 1/2 * einsum('xy,IK,LxzY,zy->IKLY', h_aa, np.identity(ncvs), t1_vaaa, rdm_ca, optimize = einsum_type)
        M_C_CVA_a_abb -= 1/2 * einsum('xy,IK,Lxzw,Yyzw->IKLY', h_aa, np.identity(ncvs), t1_vaaa, rdm_ccaa, optimize = einsum_type)
        M_C_CVA_a_abb -= einsum('xy,IK,LzYx,yz->IKLY', h_aa, np.identity(ncvs), t1_vaaa, rdm_ca, optimize = einsum_type)
        M_C_CVA_a_abb += 1/2 * einsum('xy,IK,LzxY,yz->IKLY', h_aa, np.identity(ncvs), t1_vaaa, rdm_ca, optimize = einsum_type)
        M_C_CVA_a_abb += 1/2 * einsum('xy,IK,Lzxw,Yzyw->IKLY', h_aa, np.identity(ncvs), t1_vaaa, rdm_ccaa, optimize = einsum_type)
        M_C_CVA_a_abb += 1/2 * einsum('xy,IK,Lzwx,Yzwy->IKLY', h_aa, np.identity(ncvs), t1_vaaa, rdm_ccaa, optimize = einsum_type)
        M_C_CVA_a_abb -= einsum('IK,Lx,Yyxz,yz->IKLY', np.identity(ncvs), t1_va, v_aaaa, rdm_ca, optimize = einsum_type)
        M_C_CVA_a_abb += 1/2 * einsum('IK,Lx,Yyzx,yz->IKLY', np.identity(ncvs), t1_va, v_aaaa, rdm_ca, optimize = einsum_type)
        M_C_CVA_a_abb += 1/2 * einsum('IK,Lx,xyzw,Yyzw->IKLY', np.identity(ncvs), t1_va, v_aaaa, rdm_ccaa, optimize = einsum_type)
        M_C_CVA_a_abb += einsum('IK,LxYy,xzwu,yzwu->IKLY', np.identity(ncvs), t1_vaaa, v_aaaa, rdm_ccaa, optimize = einsum_type)
        M_C_CVA_a_abb -= einsum('IK,LxYy,yzwu,xzwu->IKLY', np.identity(ncvs), t1_vaaa, v_aaaa, rdm_ccaa, optimize = einsum_type)
        M_C_CVA_a_abb -= 1/2 * einsum('IK,LxyY,xzwu,yzwu->IKLY', np.identity(ncvs), t1_vaaa, v_aaaa, rdm_ccaa, optimize = einsum_type)
        M_C_CVA_a_abb += 1/2 * einsum('IK,LxyY,yzwu,xzwu->IKLY', np.identity(ncvs), t1_vaaa, v_aaaa, rdm_ccaa, optimize = einsum_type)
        M_C_CVA_a_abb -= 1/2 * einsum('IK,Lxyz,Yxwu,yzwu->IKLY', np.identity(ncvs), t1_vaaa, v_aaaa, rdm_ccaa, optimize = einsum_type)
        M_C_CVA_a_abb -= einsum('IK,Lxyz,Ywyz,wx->IKLY', np.identity(ncvs), t1_vaaa, v_aaaa, rdm_ca, optimize = einsum_type)
        M_C_CVA_a_abb -= einsum('IK,Lxyz,Ywyu,xuzw->IKLY', np.identity(ncvs), t1_vaaa, v_aaaa, rdm_ccaa, optimize = einsum_type)
        M_C_CVA_a_abb += 1/2 * einsum('IK,Lxyz,Ywzy,wx->IKLY', np.identity(ncvs), t1_vaaa, v_aaaa, rdm_ca, optimize = einsum_type)
        M_C_CVA_a_abb += 1/2 * einsum('IK,Lxyz,Ywzu,xuyw->IKLY', np.identity(ncvs), t1_vaaa, v_aaaa, rdm_ccaa, optimize = einsum_type)
        M_C_CVA_a_abb += 1/2 * einsum('IK,Lxyz,Ywuy,xuzw->IKLY', np.identity(ncvs), t1_vaaa, v_aaaa, rdm_ccaa, optimize = einsum_type)
        M_C_CVA_a_abb += 1/2 * einsum('IK,Lxyz,Ywuz,xuwy->IKLY', np.identity(ncvs), t1_vaaa, v_aaaa, rdm_ccaa, optimize = einsum_type)
        M_C_CVA_a_abb -= 5/12 * einsum('IK,Lxyz,xwuv,yzwYuv->IKLY', np.identity(ncvs), t1_vaaa, v_aaaa, rdm_cccaaa, optimize = einsum_type)
        M_C_CVA_a_abb += 1/12 * einsum('IK,Lxyz,xwuv,yzwYvu->IKLY', np.identity(ncvs), t1_vaaa, v_aaaa, rdm_cccaaa, optimize = einsum_type)
        M_C_CVA_a_abb += 1/12 * einsum('IK,Lxyz,xwuv,yzwuYv->IKLY', np.identity(ncvs), t1_vaaa, v_aaaa, rdm_cccaaa, optimize = einsum_type)
        M_C_CVA_a_abb += 1/12 * einsum('IK,Lxyz,xwuv,yzwuvY->IKLY', np.identity(ncvs), t1_vaaa, v_aaaa, rdm_cccaaa, optimize = einsum_type)
        M_C_CVA_a_abb += 1/12 * einsum('IK,Lxyz,xwuv,yzwvYu->IKLY', np.identity(ncvs), t1_vaaa, v_aaaa, rdm_cccaaa, optimize = einsum_type)
        M_C_CVA_a_abb += 1/12 * einsum('IK,Lxyz,xwuv,yzwvuY->IKLY', np.identity(ncvs), t1_vaaa, v_aaaa, rdm_cccaaa, optimize = einsum_type)
        M_C_CVA_a_abb += 1/2 * einsum('IK,Lxyz,yzwu,Yxwu->IKLY', np.identity(ncvs), t1_vaaa, v_aaaa, rdm_ccaa, optimize = einsum_type)
        M_C_CVA_a_abb -= 1/12 * einsum('IK,Lxyz,ywuv,zuvYxw->IKLY', np.identity(ncvs), t1_vaaa, v_aaaa, rdm_cccaaa, optimize = einsum_type)
        M_C_CVA_a_abb -= 1/12 * einsum('IK,Lxyz,ywuv,zuvYwx->IKLY', np.identity(ncvs), t1_vaaa, v_aaaa, rdm_cccaaa, optimize = einsum_type)
        M_C_CVA_a_abb += 5/12 * einsum('IK,Lxyz,ywuv,zuvxYw->IKLY', np.identity(ncvs), t1_vaaa, v_aaaa, rdm_cccaaa, optimize = einsum_type)
        M_C_CVA_a_abb -= 1/12 * einsum('IK,Lxyz,ywuv,zuvxwY->IKLY', np.identity(ncvs), t1_vaaa, v_aaaa, rdm_cccaaa, optimize = einsum_type)
        M_C_CVA_a_abb -= 1/12 * einsum('IK,Lxyz,ywuv,zuvwYx->IKLY', np.identity(ncvs), t1_vaaa, v_aaaa, rdm_cccaaa, optimize = einsum_type)
        M_C_CVA_a_abb -= 1/12 * einsum('IK,Lxyz,ywuv,zuvwxY->IKLY', np.identity(ncvs), t1_vaaa, v_aaaa, rdm_cccaaa, optimize = einsum_type)
        M_C_CVA_a_abb += 5/12 * einsum('IK,Lxyz,zwuv,yuvYxw->IKLY', np.identity(ncvs), t1_vaaa, v_aaaa, rdm_cccaaa, optimize = einsum_type)
        M_C_CVA_a_abb -= 1/12 * einsum('IK,Lxyz,zwuv,yuvYwx->IKLY', np.identity(ncvs), t1_vaaa, v_aaaa, rdm_cccaaa, optimize = einsum_type)
        M_C_CVA_a_abb -= 1/12 * einsum('IK,Lxyz,zwuv,yuvxYw->IKLY', np.identity(ncvs), t1_vaaa, v_aaaa, rdm_cccaaa, optimize = einsum_type)
        M_C_CVA_a_abb -= 1/12 * einsum('IK,Lxyz,zwuv,yuvxwY->IKLY', np.identity(ncvs), t1_vaaa, v_aaaa, rdm_cccaaa, optimize = einsum_type)
        M_C_CVA_a_abb -= 1/12 * einsum('IK,Lxyz,zwuv,yuvwYx->IKLY', np.identity(ncvs), t1_vaaa, v_aaaa, rdm_cccaaa, optimize = einsum_type)
        M_C_CVA_a_abb -= 1/12 * einsum('IK,Lxyz,zwuv,yuvwxY->IKLY', np.identity(ncvs), t1_vaaa, v_aaaa, rdm_cccaaa, optimize = einsum_type)

        M_C_CVA_a_bab =- einsum('LKIY->IKLY', v_vxxa, optimize = einsum_type).copy()
        M_C_CVA_a_bab += 1/2 * einsum('LKIx,xY->IKLY', v_vxxa, rdm_ca, optimize = einsum_type)

        M_C_CVA_a_aaa = M_C_CVA_a_abb + M_C_CVA_a_bab

        M_C_CVA_a_aaa = M_C_CVA_a_aaa.reshape(ncvs, -1)
        M_C_CVA_a_abb = M_C_CVA_a_abb.reshape(ncvs, -1)
        M_C_CVA_a_bab = M_C_CVA_a_bab.reshape(ncvs, -1)

        # print(">>> SA M_C_CVA_a_aaa: {:}".format(np.linalg.norm(M_C_CVA_a_aaa)))
        # print(">>> SA M_C_CVA_a_abb: {:}".format(np.linalg.norm(M_C_CVA_a_abb)))
        # print(">>> SA M_C_CVA_a_bab: {:}".format(np.linalg.norm(M_C_CVA_a_bab)))

        ## Building C-CVA matrix
        s_c_cva_aaa = 0
        f_c_cva_aaa = s_c_cva_aaa + n_cva
        s_c_cva_abb = f_c_cva_aaa
        f_c_cva_abb = s_c_cva_abb + n_cva
        s_c_cva_bab = f_c_cva_abb
        f_c_cva_bab = s_c_cva_bab + n_cva

        M_C_CVA = np.zeros((ncvs, dim_cva))
        M_C_CVA[:, s_c_cva_aaa:f_c_cva_aaa] = M_C_CVA_a_aaa.copy()
        M_C_CVA[:, s_c_cva_abb:f_c_cva_abb] = M_C_CVA_a_abb.copy()
        M_C_CVA[:, s_c_cva_bab:f_c_cva_bab] = M_C_CVA_a_bab.copy()

    print("Time for computing M(h0-h1) blocks:               %f sec\n" % (time.time() - start_time))
    sys.stdout.flush()

    shift = 100000.0
    M_C_CAA = shift

    if nval > 0:
        M_01 = (M_C_CAA, M_C_CCE, M_C_CVE, M_C_CAE, M_C_CCA, M_C_CVA)
    else:
        M_01 = (M_C_CAA, M_C_CCE, M_C_CAE, M_C_CCA)

    return M_01

def compute_preconditioner_dev(mr_adc, M_00):

    start_time = time.time()

    # Einsum definition from kernel
    einsum = mr_adc.interface.einsum
    einsum_type = mr_adc.interface.einsum_type

    if mr_adc.method in ("mr-adc(0)", "mr-adc(1)"):

        # Multiply by -1.0, since we are solving for -M C = -S C E
        return (-1.0 * np.diag(M_00))

    # Variables from kernel
    ncvs = mr_adc.ncvs
    nval = mr_adc.nval
    ncas = mr_adc.ncas
    nextern = mr_adc.nextern

    e_cvs = mr_adc.mo_energy.x
    e_extern = mr_adc.mo_energy.e

    if nval > 0:
        e_val = mr_adc.mo_energy.v

    # One-electron integrals
    h_aa = mr_adc.h1eff.aa

    # Two-electrons integrals
    v_aaaa = mr_adc.v2e.aaaa

    v_xaxa = mr_adc.v2e.xaxa
    v_xaax = mr_adc.v2e.xaax

    # Reduced Density Matrices
    rdm_ca = mr_adc.rdm.ca
    rdm_ccaa = mr_adc.rdm.ccaa
    rdm_cccaaa = mr_adc.rdm.cccaaa

    # Overlap Matrices
    S12_c_caa = mr_adc.S12.c_caa
    S12_cae = mr_adc.S12.cae
    S12_cca = mr_adc.S12.cca

    # Excitation Spaces
    ### DEBUG
    ho_s_c = mr_adc.h_orth.s_c
    ho_f_c = mr_adc.h_orth.f_c
    ###
    ho_s_c_caa_aaa = mr_adc.h_orth.s_c_caa_aaa
    ho_f_c_caa_aaa = mr_adc.h_orth.f_c_caa_aaa
    ho_s_c_caa_abb = mr_adc.h_orth.s_c_caa_abb
    ho_f_c_caa_abb = mr_adc.h_orth.f_c_caa_abb
    ho_s_c_caa_bab = mr_adc.h_orth.s_c_caa_bab
    ho_f_c_caa_bab = mr_adc.h_orth.f_c_caa_bab

    ho_s_cce_aaa = mr_adc.h_orth.s_cce_aaa
    ho_f_cce_aaa = mr_adc.h_orth.f_cce_aaa
    ho_s_cce_abb = mr_adc.h_orth.s_cce_abb
    ho_f_cce_abb = mr_adc.h_orth.f_cce_abb

    ho_s_cae_aaa = mr_adc.h_orth.s_cae_aaa
    ho_f_cae_aaa = mr_adc.h_orth.f_cae_aaa
    ho_s_cae_abb = mr_adc.h_orth.s_cae_abb
    ho_f_cae_abb = mr_adc.h_orth.f_cae_abb
    ho_s_cae_bab = mr_adc.h_orth.s_cae_bab
    ho_f_cae_bab = mr_adc.h_orth.f_cae_bab

    ho_s_cca_aaa = mr_adc.h_orth.s_cca_aaa
    ho_f_cca_aaa = mr_adc.h_orth.f_cca_aaa
    ho_s_cca_abb = mr_adc.h_orth.s_cca_abb
    ho_f_cca_abb = mr_adc.h_orth.f_cca_abb

    if nval > 0:
        ho_s_cve = mr_adc.h_orth.s_cve
        ho_f_cve = mr_adc.h_orth.f_cve

        ho_s_cva = mr_adc.h_orth.s_cva
        ho_f_cva = mr_adc.h_orth.f_cva

        ho_s_cve_aaa = mr_adc.h_orth.s_cve_aaa
        ho_f_cve_aaa = mr_adc.h_orth.f_cve_aaa
        ho_s_cve_abb = mr_adc.h_orth.s_cve_abb
        ho_f_cve_abb = mr_adc.h_orth.f_cve_abb
        ho_s_cve_bab = mr_adc.h_orth.s_cve_bab
        ho_f_cve_bab = mr_adc.h_orth.f_cve_bab

        ho_s_cva_aaa = mr_adc.h_orth.s_cva_aaa
        ho_f_cva_aaa = mr_adc.h_orth.f_cva_aaa
        ho_s_cva_abb = mr_adc.h_orth.s_cva_abb
        ho_f_cva_abb = mr_adc.h_orth.f_cva_abb
        ho_s_cva_bab = mr_adc.h_orth.s_cva_bab
        ho_f_cva_bab = mr_adc.h_orth.f_cva_bab

    s_c = mr_adc.h0.s_c
    f_c = mr_adc.h0.f_c

    cvs_tril_ind = np.tril_indices(ncvs, k=-1)

    # Build the preconditioner
    precond = np.zeros(mr_adc.h_orth.dim)

    ### C-C debug
    precond[ho_s_c:ho_f_c] = np.diag(M_00[s_c:f_c, s_c:f_c]).copy()
    ###

    # CCE
    precond_cce =- einsum('A,AA,II,JJ->IJA', e_extern, np.identity(nextern), np.identity(ncvs), np.identity(ncvs), optimize = einsum_type)
    precond_cce += einsum('I,AA,II,JJ->IJA', e_cvs, np.identity(nextern), np.identity(ncvs), np.identity(ncvs), optimize = einsum_type)
    precond_cce += einsum('J,AA,II,JJ->IJA', e_cvs, np.identity(nextern), np.identity(ncvs), np.identity(ncvs), optimize = einsum_type)
    precond[ho_s_cce_aaa:ho_f_cce_aaa] = precond_cce[cvs_tril_ind[0], cvs_tril_ind[1]].reshape(-1).copy()
    precond[ho_s_cce_abb:ho_f_cce_abb] = precond_cce.reshape(-1).copy()

    if nval > 0:
        # CVE
        precond_cve =- einsum('A,AA,II,JJ->IJA', e_extern, np.identity(nextern), np.identity(ncvs), np.identity(nval), optimize = einsum_type)
        precond_cve += einsum('I,AA,II,JJ->IJA', e_cvs, np.identity(nextern), np.identity(ncvs), np.identity(nval), optimize = einsum_type)
        precond_cve += einsum('J,AA,II,JJ->IJA', e_val, np.identity(nextern), np.identity(ncvs), np.identity(nval), optimize = einsum_type)

        precond[ho_s_cve_aaa:ho_f_cve_aaa] = precond_cve.reshape(-1).copy()
        precond[ho_s_cve_abb:ho_f_cve_abb] = precond_cve.reshape(-1).copy()
        precond[ho_s_cve_bab:ho_f_cve_bab] = precond_cve.reshape(-1).copy()

    # CAE
    precond_cae =- 1/2 * einsum('A,AA,II,XY->IAXY', e_extern, np.identity(nextern), np.identity(ncvs), rdm_ca, optimize = einsum_type)
    precond_cae += 1/2 * einsum('I,AA,II,XY->IAXY', e_cvs, np.identity(nextern), np.identity(ncvs), rdm_ca, optimize = einsum_type)
    precond_cae += 1/2 * einsum('Xx,AA,II,xY->IAXY', h_aa, np.identity(nextern), np.identity(ncvs), rdm_ca, optimize = einsum_type)
    precond_cae += 1/2 * einsum('Xxyz,AA,II,Yxyz->IAXY', v_aaaa, np.identity(nextern), np.identity(ncvs), rdm_ccaa, optimize = einsum_type)

    precond_cae = einsum("IAXY,XP,YP->IPA", precond_cae, S12_cae, S12_cae, optimize = einsum_type)
    precond[ho_s_cae_aaa:ho_f_cae_aaa] = precond_cae.reshape(-1).copy()
    precond[ho_s_cae_abb:ho_f_cae_abb] = precond_cae.reshape(-1).copy()
    precond[ho_s_cae_bab:ho_f_cae_bab] = precond_cae.reshape(-1).copy()

    # CCA
    precond_cca =- einsum('XY,II,JJ->IJXY', h_aa, np.identity(ncvs), np.identity(ncvs), optimize = einsum_type)
    precond_cca += einsum('I,II,JJ,XY->IJXY', e_cvs, np.identity(ncvs), np.identity(ncvs), np.identity(ncas), optimize = einsum_type)
    precond_cca += einsum('J,II,JJ,XY->IJXY', e_cvs, np.identity(ncvs), np.identity(ncvs), np.identity(ncas), optimize = einsum_type)
    precond_cca -= 1/2 * einsum('I,II,JJ,YX->IJXY', e_cvs, np.identity(ncvs), np.identity(ncvs), rdm_ca, optimize = einsum_type)
    precond_cca -= 1/2 * einsum('J,II,JJ,YX->IJXY', e_cvs, np.identity(ncvs), np.identity(ncvs), rdm_ca, optimize = einsum_type)
    precond_cca += 1/2 * einsum('Xx,II,JJ,Yx->IJXY', h_aa, np.identity(ncvs), np.identity(ncvs), rdm_ca, optimize = einsum_type)
    precond_cca -= einsum('XxYy,II,JJ,xy->IJXY', v_aaaa, np.identity(ncvs), np.identity(ncvs), rdm_ca, optimize = einsum_type)
    precond_cca += 1/2 * einsum('XxyY,II,JJ,xy->IJXY', v_aaaa, np.identity(ncvs), np.identity(ncvs), rdm_ca, optimize = einsum_type)
    precond_cca += 1/2 * einsum('Xxyz,II,JJ,Yxyz->IJXY', v_aaaa, np.identity(ncvs), np.identity(ncvs), rdm_ccaa, optimize = einsum_type)

    precond_cca = einsum("IJXY,XP,YP->IJP", precond_cca, S12_cca, S12_cca, optimize = einsum_type)
    precond[ho_s_cca_aaa:ho_f_cca_aaa] = precond_cca[cvs_tril_ind[0], cvs_tril_ind[1]].reshape(-1).copy()
    precond[ho_s_cca_abb:ho_f_cca_abb] = precond_cca.reshape(-1).copy()

    if nval > 0:
        # CVA
        precond_cva =- einsum('XY,II,JJ->IJXY', h_aa, np.identity(ncvs), np.identity(nval), optimize = einsum_type)
        precond_cva += einsum('I,II,JJ,XY->IJXY', e_cvs, np.identity(ncvs), np.identity(nval), np.identity(ncas), optimize = einsum_type)
        precond_cva += einsum('J,II,JJ,XY->IJXY', e_val, np.identity(ncvs), np.identity(nval), np.identity(ncas), optimize = einsum_type)
        precond_cva -= 1/2 * einsum('I,II,JJ,YX->IJXY', e_cvs, np.identity(ncvs), np.identity(nval), rdm_ca, optimize = einsum_type)
        precond_cva -= 1/2 * einsum('J,II,JJ,YX->IJXY', e_val, np.identity(ncvs), np.identity(nval), rdm_ca, optimize = einsum_type)
        precond_cva += 1/2 * einsum('Xx,II,JJ,Yx->IJXY', h_aa, np.identity(ncvs), np.identity(nval), rdm_ca, optimize = einsum_type)
        precond_cva -= einsum('XxYy,II,JJ,xy->IJXY', v_aaaa, np.identity(ncvs), np.identity(nval), rdm_ca, optimize = einsum_type)
        precond_cva += 1/2 * einsum('XxyY,II,JJ,xy->IJXY', v_aaaa, np.identity(ncvs), np.identity(nval), rdm_ca, optimize = einsum_type)
        precond_cva += 1/2 * einsum('Xxyz,II,JJ,Yxyz->IJXY', v_aaaa, np.identity(ncvs), np.identity(nval), rdm_ccaa, optimize = einsum_type)

        precond_cva = einsum("IJXY,XP,YP->IJP", precond_cva, S12_cca, S12_cca, optimize = einsum_type)
        precond[ho_s_cva_aaa:ho_f_cva_aaa] = precond_cva.reshape(-1).copy()
        precond[ho_s_cva_abb:ho_f_cva_abb] = precond_cva.reshape(-1).copy()
        precond[ho_s_cva_bab:ho_f_cva_bab] = precond_cva.reshape(-1).copy()

    # Multiply by -1.0, since we are solving for -M C = -S C E
    precond *= (-1.0)

    print ("Time for computing preconditioner:                %f sec\n" % (time.time() - start_time))
    sys.stdout.flush()

    return precond

def apply_S_12_dev(mr_adc, X, transpose = False):

    # Einsum definition from kernel
    einsum = mr_adc.interface.einsum
    einsum_type = mr_adc.interface.einsum_type

    # Dimensions
    nextern = mr_adc.nextern
    ncvs = mr_adc.ncvs
    nval = mr_adc.nval

    ### DEBUG
    ho_s_c = mr_adc.h_orth.s_c
    ho_f_c = mr_adc.h_orth.f_c
    ####

    ho_s_c_caa = mr_adc.h_orth.s_c_caa
    ho_f_c_caa = mr_adc.h_orth.f_c_caa
    ho_s_cce = mr_adc.h_orth.s_cce
    ho_f_cce = mr_adc.h_orth.f_cce
    ho_s_cae = mr_adc.h_orth.s_cae
    ho_f_cae = mr_adc.h_orth.f_cae
    ho_s_cca = mr_adc.h_orth.s_cca
    ho_f_cca = mr_adc.h_orth.f_cca

    ho_s_cae_aaa = mr_adc.h_orth.s_cae_aaa
    ho_f_cae_aaa = mr_adc.h_orth.f_cae_aaa
    ho_s_cae_abb = mr_adc.h_orth.s_cae_abb
    ho_f_cae_abb = mr_adc.h_orth.f_cae_abb
    ho_s_cae_bab = mr_adc.h_orth.s_cae_bab
    ho_f_cae_bab = mr_adc.h_orth.f_cae_bab

    s_c = mr_adc.h0.s_c
    f_c = mr_adc.h0.f_c
    s_caa = mr_adc.h1.s_caa
    f_caa = mr_adc.h1.f_caa
    s_cce = mr_adc.h1.s_cce
    f_cce = mr_adc.h1.f_cce
    s_cae = mr_adc.h1.s_cae
    f_cae = mr_adc.h1.f_cae
    s_cca = mr_adc.h1.s_cca
    f_cca = mr_adc.h1.f_cca

    s_cae_aaa = mr_adc.h1.s_cae_aaa
    f_cae_aaa = mr_adc.h1.f_cae_aaa
    s_cae_abb = mr_adc.h1.s_cae_abb
    f_cae_abb = mr_adc.h1.f_cae_abb
    s_cae_bab = mr_adc.h1.s_cae_bab
    f_cae_bab = mr_adc.h1.f_cae_bab

    if nval > 0:
        ho_s_cve = mr_adc.h_orth.s_cve
        ho_f_cve = mr_adc.h_orth.f_cve

        ho_s_cva = mr_adc.h_orth.s_cva
        ho_f_cva = mr_adc.h_orth.f_cva

        ho_s_cva_aaa = mr_adc.h_orth.s_cva_aaa
        ho_f_cva_aaa = mr_adc.h_orth.f_cva_aaa
        ho_s_cva_abb = mr_adc.h_orth.s_cva_abb
        ho_f_cva_abb = mr_adc.h_orth.f_cva_abb
        ho_s_cva_bab = mr_adc.h_orth.s_cva_bab
        ho_f_cva_bab = mr_adc.h_orth.f_cva_bab

        s_cve = mr_adc.h1.s_cve
        f_cve = mr_adc.h1.f_cve

        s_cva = mr_adc.h1.s_cva
        f_cva = mr_adc.h1.f_cva

        s_cva_aaa = mr_adc.h1.s_cva_aaa
        f_cva_aaa = mr_adc.h1.f_cva_aaa
        s_cva_abb = mr_adc.h1.s_cva_abb
        f_cva_abb = mr_adc.h1.f_cva_abb
        s_cva_bab = mr_adc.h1.s_cva_bab
        f_cva_bab = mr_adc.h1.f_cva_bab

    # Overlap matrices
    S12_c_caa = mr_adc.S12.c_caa
    S12_cae = mr_adc.S12.cae
    S12_cca = mr_adc.S12.cca

    Xt = None

    if transpose:
        if (X.shape[0] != (mr_adc.h0.dim + mr_adc.h1.dim)):
            raise Exception("Dimensions do not match when applying S_12 transpose")

        Xt = np.zeros(mr_adc.h_orth.dim)

        ### C-C DEBUG
        Xt[ho_s_c:ho_f_c] = X[s_c:f_c].copy()
        ###

        # CCE
        Xt[ho_s_cce:ho_f_cce] = X[s_cce:f_cce].copy()

        if nval > 0:
            # CVE
            Xt[ho_s_cve:ho_f_cve] = X[s_cve:f_cve].copy()

        # CAE
        temp = X[s_cae_aaa:f_cae_aaa].reshape(ncvs, S12_cae.shape[0], nextern).copy()
        Xt[ho_s_cae_aaa:ho_f_cae_aaa] = einsum("IXA,XP->IPA", temp, S12_cae).reshape(-1).copy()

        temp = X[s_cae_abb:f_cae_abb].reshape(ncvs, S12_cae.shape[0], nextern).copy()
        Xt[ho_s_cae_abb:ho_f_cae_abb] = einsum("IXA,XP->IPA", temp, S12_cae).reshape(-1).copy()

        temp = X[s_cae_bab:f_cae_bab].reshape(ncvs, S12_cae.shape[0], nextern).copy()
        Xt[ho_s_cae_bab:ho_f_cae_bab] = einsum("IXA,XP->IPA", temp, S12_cae).reshape(-1).copy()

        # CCA
        temp = X[s_cca:f_cca].reshape(-1, S12_cca.shape[0]).copy()
        Xt[ho_s_cca:ho_f_cca] = einsum("IX,XP->IP", temp, S12_cca).reshape(-1).copy()

        if nval > 0:
            # CVA
            temp = X[s_cva:f_cva].reshape(-1, S12_cca.shape[0]).copy()
            Xt[ho_s_cva:ho_f_cva] = einsum("IX,XP->IP", temp, S12_cca).reshape(-1).copy()

    else:
        if (X.shape[0] != (mr_adc.h_orth.dim)):
            raise Exception("Dimensions do not match when applying S_12")

        Xt = np.zeros(mr_adc.h0.dim + mr_adc.h1.dim)

        ### C-C DEBUG
        Xt[s_c:f_c] = X[ho_s_c:ho_f_c].copy()
        ###

        # CCE
        Xt[s_cce:f_cce] = X[ho_s_cce:ho_f_cce].copy()

        if nval > 0:
            # CVE
            Xt[s_cve:f_cve] = X[ho_s_cve:ho_f_cve].copy()

        # CAE
        temp = X[ho_s_cae_aaa:ho_f_cae_aaa].reshape(ncvs, S12_cae.shape[1], nextern).copy()
        Xt[s_cae_aaa:f_cae_aaa] = einsum("IPA,XP->IXA", temp, S12_cae).reshape(-1).copy()

        temp = X[ho_s_cae_abb:ho_f_cae_abb].reshape(ncvs, S12_cae.shape[1], nextern).copy()
        Xt[s_cae_abb:f_cae_abb] = einsum("IPA,XP->IXA", temp, S12_cae).reshape(-1).copy()

        temp = X[ho_s_cae_bab:ho_f_cae_bab].reshape(ncvs, S12_cae.shape[1], nextern).copy()
        Xt[s_cae_bab:f_cae_bab] = einsum("IPA,XP->IXA", temp, S12_cae).reshape(-1).copy()

        # CCA
        temp = X[ho_s_cca:ho_f_cca].reshape(-1, S12_cca.shape[1]).copy()
        Xt[s_cca:f_cca] = einsum("IP,XP->IX", temp, S12_cca).reshape(-1).copy()

        if nval > 0:
            # CVA
            temp = X[ho_s_cva:ho_f_cva].reshape(-1, S12_cca.shape[1]).copy()
            Xt[s_cva:f_cva] = einsum("IP,XP->IX", temp, S12_cca).reshape(-1).copy()

    return Xt

def compute_sigma_vector_dev(mr_adc, M_00, M_01, M_11, Xt):

    # Einsum definition from kernel
    einsum = mr_adc.interface.einsum
    einsum_type = mr_adc.interface.einsum_type

    # Variables from kernel
    ncvs = mr_adc.ncvs
    nval = mr_adc.nval
    ncas = mr_adc.ncas
    nextern = mr_adc.nextern

    e_cvs = mr_adc.mo_energy.x
    e_core = mr_adc.mo_energy.c
    e_extern = mr_adc.mo_energy.e

    if nval > 0:
        e_val = mr_adc.mo_energy.v

    # One-electron integrals
    h_aa = mr_adc.h1eff.aa

    # Two-electrons integrals
    v_aaaa = mr_adc.v2e.aaaa

    # Reduced Density Matrices
    rdm_ca = mr_adc.rdm.ca
    rdm_ccaa = mr_adc.rdm.ccaa
    rdm_cccaaa = mr_adc.rdm.cccaaa

    # Dimensions
    s_c = mr_adc.h0.s_c
    f_c = mr_adc.h0.f_c
    s_caa = mr_adc.h1.s_caa
    f_caa = mr_adc.h1.f_caa
    s_cce = mr_adc.h1.s_cce
    f_cce = mr_adc.h1.f_cce
    s_cae = mr_adc.h1.s_cae
    f_cae = mr_adc.h1.f_cae
    s_cca = mr_adc.h1.s_cca
    f_cca = mr_adc.h1.f_cca

    s_caa_aaa = mr_adc.h1.s_caa_aaa
    f_caa_aaa = mr_adc.h1.f_caa_aaa
    s_caa_abb = mr_adc.h1.s_caa_abb
    f_caa_abb = mr_adc.h1.f_caa_abb
    s_caa_bab = mr_adc.h1.s_caa_bab
    f_caa_bab = mr_adc.h1.f_caa_bab

    s_cce_aaa = mr_adc.h1.s_cce_aaa
    f_cce_aaa = mr_adc.h1.f_cce_aaa
    s_cce_abb = mr_adc.h1.s_cce_abb
    f_cce_abb = mr_adc.h1.f_cce_abb

    s_cae_aaa = mr_adc.h1.s_cae_aaa
    f_cae_aaa = mr_adc.h1.f_cae_aaa
    s_cae_abb = mr_adc.h1.s_cae_abb
    f_cae_abb = mr_adc.h1.f_cae_abb
    s_cae_bab = mr_adc.h1.s_cae_bab
    f_cae_bab = mr_adc.h1.f_cae_bab

    s_cca_aaa = mr_adc.h1.s_cca_aaa
    f_cca_aaa = mr_adc.h1.f_cca_aaa
    s_cca_abb = mr_adc.h1.s_cca_abb
    f_cca_abb = mr_adc.h1.f_cca_abb

    if nval > 0:
        s_cve = mr_adc.h1.s_cve
        f_cve = mr_adc.h1.f_cve

        s_cva = mr_adc.h1.s_cva
        f_cva = mr_adc.h1.f_cva

        s_cve_aaa = mr_adc.h1.s_cve_aaa
        f_cve_aaa = mr_adc.h1.f_cve_aaa
        s_cve_abb = mr_adc.h1.s_cve_abb
        f_cve_abb = mr_adc.h1.f_cve_abb
        s_cve_bab = mr_adc.h1.s_cve_bab
        f_cve_bab = mr_adc.h1.f_cve_bab

        s_cva_aaa = mr_adc.h1.s_cva_aaa
        f_cva_aaa = mr_adc.h1.f_cva_aaa
        s_cva_abb = mr_adc.h1.s_cva_abb
        f_cva_abb = mr_adc.h1.f_cva_abb
        s_cva_bab = mr_adc.h1.s_cva_bab
        f_cva_bab = mr_adc.h1.f_cva_bab

    cvs_tril_ind = np.tril_indices(ncvs, k=-1)

    # (CASCI + C) -> (CASCI + C)
    sigma = np.zeros_like(Xt)

    # h0-h0 contributions
    sigma[:mr_adc.h0.dim] = np.dot(M_00, Xt[:mr_adc.h0.dim])

    # h0-h1 and h1-h0 contributions
    if nval > 0:
        M_C_CAA, M_C_CCE, M_C_CVE, M_C_CAE, M_C_CCA, M_C_CVA = M_01
    else:
        M_C_CAA, M_C_CCE, M_C_CAE, M_C_CCA = M_01

    # C <-> CCE
    sigma[s_c:f_c] += np.dot(M_C_CCE, Xt[s_cce:f_cce])
    sigma[s_cce:f_cce] += np.dot(M_C_CCE.T, Xt[s_c:f_c])

    # C <-> CVE
    if nval > 0:
        sigma[s_c:f_c] += np.dot(M_C_CVE, Xt[s_cve:f_cve])
        sigma[s_cve:f_cve] += np.dot(M_C_CVE.T, Xt[s_c:f_c])

    # C <-> CAE
    sigma[s_c:f_c] += np.dot(M_C_CAE, Xt[s_cae:f_cae])
    sigma[s_cae:f_cae] += np.dot(M_C_CAE.T, Xt[s_c:f_c])

    # C <-> CCA
    sigma[s_c:f_c] += np.dot(M_C_CCA, Xt[s_cca:f_cca])
    sigma[s_cca:f_cca] += np.dot(M_C_CCA.T, Xt[s_c:f_c])

    # C <-> CVA
    if nval > 0:
        sigma[s_c:f_c] += np.dot(M_C_CVA, Xt[s_cva:f_cva])
        sigma[s_cva:f_cva] += np.dot(M_C_CVA.T, Xt[s_c:f_c])

    # h1-h1 contributions
    # CCE <- CCE
    X_aaa = np.zeros((ncvs, ncvs, nextern))
    X_aaa[cvs_tril_ind[0], cvs_tril_ind[1]] =  Xt[s_cce_aaa:f_cce_aaa].reshape(-1, nextern).copy()
    X_aaa[cvs_tril_ind[1], cvs_tril_ind[0]] =- Xt[s_cce_aaa:f_cce_aaa].reshape(-1, nextern).copy()

    X_abb = Xt[s_cce_abb:f_cce_abb].reshape(ncvs, ncvs, nextern).copy()
    X_bab =- X_abb.transpose(1,0,2)

    sigma_cce =- 1/2 * einsum('KLB,B->KLB', X_aaa, e_extern, optimize = einsum_type)
    sigma_cce += 1/2 * einsum('KLB,K->KLB', X_aaa, e_cvs, optimize = einsum_type)
    sigma_cce += 1/2 * einsum('KLB,L->KLB', X_aaa, e_cvs, optimize = einsum_type)
    sigma_cce += 1/2 * einsum('LKB,B->KLB', X_aaa, e_extern, optimize = einsum_type)
    sigma_cce -= 1/2 * einsum('LKB,K->KLB', X_aaa, e_cvs, optimize = einsum_type)
    sigma_cce -= 1/2 * einsum('LKB,L->KLB', X_aaa, e_cvs, optimize = einsum_type)
    sigma[s_cce_aaa:f_cce_aaa] += sigma_cce[cvs_tril_ind[0], cvs_tril_ind[1]].reshape(-1).copy()

    sigma_cce =- 1/2 * einsum('KLB,B->KLB', X_abb, e_extern, optimize = einsum_type)
    sigma_cce += 1/2 * einsum('KLB,K->KLB', X_abb, e_cvs, optimize = einsum_type)
    sigma_cce += 1/2 * einsum('KLB,L->KLB', X_abb, e_cvs, optimize = einsum_type)
    sigma_cce += 1/2 * einsum('LKB,B->KLB', X_bab, e_extern, optimize = einsum_type)
    sigma_cce -= 1/2 * einsum('LKB,K->KLB', X_bab, e_cvs, optimize = einsum_type)
    sigma_cce -= 1/2 * einsum('LKB,L->KLB', X_bab, e_cvs, optimize = einsum_type)
    sigma[s_cce_abb:f_cce_abb] += sigma_cce.reshape(-1).copy()

    if nval > 0:
        # CVE <- CVE
        X_aaa = Xt[s_cve_aaa:f_cve_aaa].reshape(ncvs, nval, nextern).copy()
        X_abb = Xt[s_cve_abb:f_cve_abb].reshape(ncvs, nval, nextern).copy()
        X_bab = Xt[s_cve_bab:f_cve_bab].reshape(ncvs, nval, nextern).copy()

        sigma_cve =- einsum('KLB,B->KLB', X_aaa, e_extern, optimize = einsum_type)
        sigma_cve += einsum('KLB,K->KLB', X_aaa, e_cvs, optimize = einsum_type)
        sigma_cve += einsum('KLB,L->KLB', X_aaa, e_val, optimize = einsum_type)
        sigma[s_cve_aaa:f_cve_aaa] += sigma_cve.reshape(-1).copy()

        sigma_cve =- einsum('KLB,B->KLB', X_abb, e_extern, optimize = einsum_type)
        sigma_cve += einsum('KLB,K->KLB', X_abb, e_cvs, optimize = einsum_type)
        sigma_cve += einsum('KLB,L->KLB', X_abb, e_val, optimize = einsum_type)
        sigma[s_cve_abb:f_cve_abb] += sigma_cve.reshape(-1).copy()

        sigma_cve =- einsum('KLB,B->KLB', X_bab, e_extern, optimize = einsum_type)
        sigma_cve += einsum('KLB,K->KLB', X_bab, e_cvs, optimize = einsum_type)
        sigma_cve += einsum('KLB,L->KLB', X_bab, e_val, optimize = einsum_type)
        sigma[s_cve_bab:f_cve_bab] += sigma_cve.reshape(-1).copy()

    # CAE <- CAE
    X_aaa = Xt[s_cae_aaa:f_cae_aaa].reshape(ncvs, ncas, nextern).copy()
    X_abb = Xt[s_cae_abb:f_cae_abb].reshape(ncvs, ncas, nextern).copy()
    X_bab = Xt[s_cae_bab:f_cae_bab].reshape(ncvs, ncas, nextern).copy()

    sigma_cae =- 1/2 * einsum('KxB,B,xZ->KZB', X_aaa, e_extern, rdm_ca, optimize = einsum_type)
    sigma_cae += 1/2 * einsum('KxB,K,xZ->KZB', X_aaa, e_cvs, rdm_ca, optimize = einsum_type)
    sigma_cae += 1/2 * einsum('KxB,xy,yZ->KZB', X_aaa, h_aa, rdm_ca, optimize = einsum_type)
    sigma_cae += 1/2 * einsum('KxB,xyzw,Zyzw->KZB', X_aaa, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma[s_cae_aaa:f_cae_aaa] += sigma_cae.reshape(-1).copy()

    sigma_cae =- 1/2 * einsum('KxB,B,xZ->KZB', X_abb, e_extern, rdm_ca, optimize = einsum_type)
    sigma_cae += 1/2 * einsum('KxB,K,xZ->KZB', X_abb, e_cvs, rdm_ca, optimize = einsum_type)
    sigma_cae += 1/2 * einsum('KxB,xy,yZ->KZB', X_abb, h_aa, rdm_ca, optimize = einsum_type)
    sigma_cae += 1/2 * einsum('KxB,xyzw,Zyzw->KZB', X_abb, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma[s_cae_abb:f_cae_abb] += sigma_cae.reshape(-1).copy()

    sigma_cae =- 1/2 * einsum('KxB,B,xZ->KZB', X_bab, e_extern, rdm_ca, optimize = einsum_type)
    sigma_cae += 1/2 * einsum('KxB,K,xZ->KZB', X_bab, e_cvs, rdm_ca, optimize = einsum_type)
    sigma_cae += 1/2 * einsum('KxB,xy,yZ->KZB', X_bab, h_aa, rdm_ca, optimize = einsum_type)
    sigma_cae += 1/2 * einsum('KxB,xyzw,Zyzw->KZB', X_bab, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma[s_cae_bab:f_cae_bab] += sigma_cae.reshape(-1).copy()

    # CCA <- CCA
    X_aaa = np.zeros((ncvs, ncvs, ncas))
    X_aaa[cvs_tril_ind[0], cvs_tril_ind[1]] =  Xt[s_cca_aaa:f_cca_aaa].reshape(-1, ncas).copy()
    X_aaa[cvs_tril_ind[1], cvs_tril_ind[0]] =- Xt[s_cca_aaa:f_cca_aaa].reshape(-1, ncas).copy()

    X_abb = Xt[s_cca_abb:f_cca_abb].reshape(ncvs, ncvs, ncas).copy()

    sigma_cca  = einsum('KLW,K->KLW', X_aaa, e_cvs, optimize = einsum_type)
    sigma_cca += einsum('KLW,L->KLW', X_aaa, e_cvs, optimize = einsum_type)
    sigma_cca -= einsum('KLx,Wx->KLW', X_aaa, h_aa, optimize = einsum_type)
    sigma_cca -= 1/2 * einsum('KLx,K,Wx->KLW', X_aaa, e_cvs, rdm_ca, optimize = einsum_type)
    sigma_cca -= 1/2 * einsum('KLx,L,Wx->KLW', X_aaa, e_cvs, rdm_ca, optimize = einsum_type)
    sigma_cca += 1/2 * einsum('KLx,xy,Wy->KLW', X_aaa, h_aa, rdm_ca, optimize = einsum_type)
    sigma_cca -= einsum('KLx,Wyxz,zy->KLW', X_aaa, v_aaaa, rdm_ca, optimize = einsum_type)
    sigma_cca += 1/2 * einsum('KLx,Wyzx,zy->KLW', X_aaa, v_aaaa, rdm_ca, optimize = einsum_type)
    sigma_cca += 1/2 * einsum('KLx,xyzw,Wyzw->KLW', X_aaa, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma[s_cca_aaa:f_cca_aaa] += sigma_cca[cvs_tril_ind[0], cvs_tril_ind[1]].reshape(-1).copy()

    sigma_cca  = einsum('KLW,K->KLW', X_abb, e_cvs, optimize = einsum_type)
    sigma_cca += einsum('KLW,L->KLW', X_abb, e_cvs, optimize = einsum_type)
    sigma_cca -= einsum('KLx,Wx->KLW', X_abb, h_aa, optimize = einsum_type)
    sigma_cca -= 1/2 * einsum('KLx,K,Wx->KLW', X_abb, e_cvs, rdm_ca, optimize = einsum_type)
    sigma_cca -= 1/2 * einsum('KLx,L,Wx->KLW', X_abb, e_cvs, rdm_ca, optimize = einsum_type)
    sigma_cca += 1/2 * einsum('KLx,xy,Wy->KLW', X_abb, h_aa, rdm_ca, optimize = einsum_type)
    sigma_cca -= einsum('KLx,Wyxz,zy->KLW', X_abb, v_aaaa, rdm_ca, optimize = einsum_type)
    sigma_cca += 1/2 * einsum('KLx,Wyzx,zy->KLW', X_abb, v_aaaa, rdm_ca, optimize = einsum_type)
    sigma_cca += 1/2 * einsum('KLx,xyzw,Wyzw->KLW', X_abb, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma[s_cca_abb:f_cca_abb] += sigma_cca.reshape(-1).copy()

    if nval > 0:
        # CVA <- CVA
        X_aaa = Xt[s_cva_aaa:f_cva_aaa].reshape(ncvs, nval, ncas).copy()
        X_abb = Xt[s_cva_abb:f_cva_abb].reshape(ncvs, nval, ncas).copy()
        X_bab = Xt[s_cva_bab:f_cva_bab].reshape(ncvs, nval, ncas).copy()

        sigma_cva  = einsum('KLW,K->KLW', X_aaa, e_cvs, optimize = einsum_type)
        sigma_cva += einsum('KLW,L->KLW', X_aaa, e_val, optimize = einsum_type)
        sigma_cva -= einsum('KLx,Wx->KLW', X_aaa, h_aa, optimize = einsum_type)
        sigma_cva -= 1/2 * einsum('KLx,K,Wx->KLW', X_aaa, e_cvs, rdm_ca, optimize = einsum_type)
        sigma_cva -= 1/2 * einsum('KLx,L,Wx->KLW', X_aaa, e_val, rdm_ca, optimize = einsum_type)
        sigma_cva += 1/2 * einsum('KLx,xy,Wy->KLW', X_aaa, h_aa, rdm_ca, optimize = einsum_type)
        sigma_cva -= einsum('KLx,Wyxz,zy->KLW', X_aaa, v_aaaa, rdm_ca, optimize = einsum_type)
        sigma_cva += 1/2 * einsum('KLx,Wyzx,zy->KLW', X_aaa, v_aaaa, rdm_ca, optimize = einsum_type)
        sigma_cva += 1/2 * einsum('KLx,xyzw,Wyzw->KLW', X_aaa, v_aaaa, rdm_ccaa, optimize = einsum_type)
        sigma[s_cva_aaa:f_cva_aaa] += sigma_cva.reshape(-1).copy()

        sigma_cva  = einsum('KLW,K->KLW', X_abb, e_cvs, optimize = einsum_type)
        sigma_cva += einsum('KLW,L->KLW', X_abb, e_val, optimize = einsum_type)
        sigma_cva -= einsum('KLx,Wx->KLW', X_abb, h_aa, optimize = einsum_type)
        sigma_cva -= 1/2 * einsum('KLx,K,Wx->KLW', X_abb, e_cvs, rdm_ca, optimize = einsum_type)
        sigma_cva -= 1/2 * einsum('KLx,L,Wx->KLW', X_abb, e_val, rdm_ca, optimize = einsum_type)
        sigma_cva += 1/2 * einsum('KLx,xy,Wy->KLW', X_abb, h_aa, rdm_ca, optimize = einsum_type)
        sigma_cva -= einsum('KLx,Wyxz,zy->KLW', X_abb, v_aaaa, rdm_ca, optimize = einsum_type)
        sigma_cva += 1/2 * einsum('KLx,Wyzx,zy->KLW', X_abb, v_aaaa, rdm_ca, optimize = einsum_type)
        sigma_cva += 1/2 * einsum('KLx,xyzw,Wyzw->KLW', X_abb, v_aaaa, rdm_ccaa, optimize = einsum_type)
        sigma[s_cva_abb:f_cva_abb] += sigma_cva.reshape(-1).copy()

        sigma_cva  = einsum('KLW,K->KLW', X_bab, e_cvs, optimize = einsum_type)
        sigma_cva += einsum('KLW,L->KLW', X_bab, e_val, optimize = einsum_type)
        sigma_cva -= einsum('KLx,Wx->KLW', X_bab, h_aa, optimize = einsum_type)
        sigma_cva -= 1/2 * einsum('KLx,K,Wx->KLW', X_bab, e_cvs, rdm_ca, optimize = einsum_type)
        sigma_cva -= 1/2 * einsum('KLx,L,Wx->KLW', X_bab, e_val, rdm_ca, optimize = einsum_type)
        sigma_cva += 1/2 * einsum('KLx,xy,Wy->KLW', X_bab, h_aa, rdm_ca, optimize = einsum_type)
        sigma_cva -= einsum('KLx,Wyxz,zy->KLW', X_bab, v_aaaa, rdm_ca, optimize = einsum_type)
        sigma_cva += 1/2 * einsum('KLx,Wyzx,zy->KLW', X_bab, v_aaaa, rdm_ca, optimize = einsum_type)
        sigma_cva += 1/2 * einsum('KLx,xyzw,Wyzw->KLW', X_bab, v_aaaa, rdm_ccaa, optimize = einsum_type)
        sigma[s_cva_bab:f_cva_bab] += sigma_cva.reshape(-1).copy()

    return sigma
