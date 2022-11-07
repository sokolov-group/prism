import sys
import time
import numpy as np
import prism.mr_adc_overlap as mr_adc_overlap
import prism.mr_adc_rdms as mr_adc_rdms
import prism.mr_adc_efficient_4rdm as mr_adc_efficient_4rdm
#TODO: Check for redundant imports

def compute_excitation_manifolds(mr_adc):

    # MR-ADC(0) and MR-ADC(1)
    mr_adc.h0.n_c = mr_adc.ncvs
    mr_adc.h0.dim = mr_adc.h0.n_c # Total dimension of h0

    mr_adc.h0.s_c = 0
    mr_adc.h0.f_c = mr_adc.h0.s_c + mr_adc.h0.n_c

    print ("Dimension of h0 excitation manifold:                       %d" % mr_adc.h0.dim)

    # MR-ADC(2)
    mr_adc.h1.dim = 0
    mr_adc.h_orth.dim = mr_adc.h0.dim

    if mr_adc.method in ("mr-adc(2)", "mr-adc(2)-x"):
        mr_adc.h1.n_caa = mr_adc.ncas * mr_adc.ncas * mr_adc.ncvs
        mr_adc.h1.n_cce = mr_adc.nextern * mr_adc.ncvs * (mr_adc.ncvs - 1) // 2
        mr_adc.h1.n_cae = mr_adc.nextern * mr_adc.ncas * mr_adc.ncvs
        mr_adc.h1.n_cca = mr_adc.ncas * mr_adc.ncvs * (mr_adc.ncvs - 1) // 2
        if mr_adc.nval != 0:
            mr_adc.h1.n_cve = mr_adc.nextern * mr_adc.ncvs * mr_adc.nval
            mr_adc.h1.n_cva = mr_adc.ncas * mr_adc.ncvs * mr_adc.nval
            mr_adc.h1.dim = mr_adc.h1.n_caa + mr_adc.h1.n_cce + mr_adc.h1.n_cve + mr_adc.h1.n_cae + mr_adc.h1.n_cca + mr_adc.h1.n_cva
        else:
            mr_adc.h1.dim = mr_adc.h1.n_caa + mr_adc.h1.n_cce + mr_adc.h1.n_cae + mr_adc.h1.n_cca

        if mr_adc.nval != 0:
            mr_adc.h1.s_caa = mr_adc.h0.f_c
            mr_adc.h1.f_caa = mr_adc.h1.s_caa + mr_adc.h1.n_caa
            mr_adc.h1.s_cce = mr_adc.h1.f_caa
            mr_adc.h1.f_cce = mr_adc.h1.s_cce + mr_adc.h1.n_cce
            mr_adc.h1.s_cve = mr_adc.h1.f_cce
            mr_adc.h1.f_cve = mr_adc.h1.s_cve + mr_adc.h1.n_cve
            mr_adc.h1.s_cae = mr_adc.h1.f_cve
            mr_adc.h1.f_cae = mr_adc.h1.s_cae + mr_adc.h1.n_cae
            mr_adc.h1.s_cca = mr_adc.h1.f_cae
            mr_adc.h1.f_cca = mr_adc.h1.s_cca + mr_adc.h1.n_cca
            mr_adc.h1.s_cva = mr_adc.h1.f_cca
            mr_adc.h1.f_cva = mr_adc.h1.s_cva + mr_adc.h1.n_cva
        else:
            mr_adc.h1.s_caa = mr_adc.h0.f_c
            mr_adc.h1.f_caa = mr_adc.h1.s_caa + mr_adc.h1.n_caa
            mr_adc.h1.s_cce = mr_adc.h1.f_caa
            mr_adc.h1.f_cce = mr_adc.h1.s_cce + mr_adc.h1.n_cce
            mr_adc.h1.s_cae = mr_adc.h1.f_cce
            mr_adc.h1.f_cae = mr_adc.h1.s_cae + mr_adc.h1.n_cae
            mr_adc.h1.s_cca = mr_adc.h1.f_cae
            mr_adc.h1.f_cca = mr_adc.h1.s_cca + mr_adc.h1.n_cca

        print ("Dimension of h1 excitation manifold:                       %d" % mr_adc.h1.dim)

        # Overlap for c - caa
        mr_adc.S12.c_caa = mr_adc_overlap.compute_S12_0p_projector(mr_adc)
        mr_adc.S12.cae = mr_adc_overlap.compute_S12_m1(mr_adc)
        mr_adc.S12.cca = mr_adc_overlap.compute_S12_p1(mr_adc)

        # Determine dimensions of orthogonalized excitation spaces
        mr_adc.h_orth.n_c_caa = mr_adc.ncvs * mr_adc.S12.c_caa.shape[1]
        mr_adc.h_orth.n_cce = mr_adc.h1.n_cce
        mr_adc.h_orth.n_cae = mr_adc.nextern * mr_adc.ncvs * mr_adc.S12.cae.shape[1]
        mr_adc.h_orth.n_cca = mr_adc.S12.cca.shape[1] * mr_adc.ncvs * (mr_adc.ncvs - 1) // 2
        if mr_adc.nval != 0:
            mr_adc.h_orth.n_cve = mr_adc.h1.n_cve
            mr_adc.h_orth.n_cva = mr_adc.S12.cca.shape[1] * mr_adc.ncvs * mr_adc.nval
            mr_adc.h_orth.dim = mr_adc.h_orth.n_c_caa + mr_adc.h_orth.n_cce + mr_adc.h_orth.n_cve + mr_adc.h_orth.n_cae + mr_adc.h_orth.n_cca + mr_adc.h_orth.n_cva
        else:
            mr_adc.h_orth.dim = mr_adc.h_orth.n_c_caa + mr_adc.h_orth.n_cce + mr_adc.h_orth.n_cae + mr_adc.h_orth.n_cca

        if mr_adc.nval != 0:
            mr_adc.h_orth.s_c_caa = 0
            mr_adc.h_orth.f_c_caa = mr_adc.h_orth.n_c_caa
            mr_adc.h_orth.s_cce = mr_adc.h_orth.f_c_caa
            mr_adc.h_orth.f_cce = mr_adc.h_orth.s_cce + mr_adc.h_orth.n_cce
            mr_adc.h_orth.s_cve = mr_adc.h_orth.f_cce
            mr_adc.h_orth.f_cve = mr_adc.h_orth.s_cve + mr_adc.h_orth.n_cve
            mr_adc.h_orth.s_cae = mr_adc.h_orth.f_cve
            mr_adc.h_orth.f_cae = mr_adc.h_orth.s_cae + mr_adc.h_orth.n_cae
            mr_adc.h_orth.s_cca = mr_adc.h_orth.f_cae
            mr_adc.h_orth.f_cca = mr_adc.h_orth.s_cca + mr_adc.h_orth.n_cca
            mr_adc.h_orth.s_cva = mr_adc.h_orth.f_cca
            mr_adc.h_orth.f_cva = mr_adc.h_orth.s_cva + mr_adc.h_orth.n_cva
        else:
            mr_adc.h_orth.s_c_caa = 0
            mr_adc.h_orth.f_c_caa = mr_adc.h_orth.n_c_caa
            mr_adc.h_orth.s_cce = mr_adc.h_orth.f_c_caa
            mr_adc.h_orth.f_cce = mr_adc.h_orth.s_cce + mr_adc.h_orth.n_cce
            mr_adc.h_orth.s_cae = mr_adc.h_orth.f_cce
            mr_adc.h_orth.f_cae = mr_adc.h_orth.s_cae + mr_adc.h_orth.n_cae
            mr_adc.h_orth.s_cca = mr_adc.h_orth.f_cae
            mr_adc.h_orth.f_cca = mr_adc.h_orth.s_cca + mr_adc.h_orth.n_cca

    print ("Total dimension of the excitation manifold:                %d" % (mr_adc.h0.dim + mr_adc.h1.dim))
    print ("Dimension of the orthogonalized excitation manifold:       %d\n" % (mr_adc.h_orth.dim))
    sys.stdout.flush()

    if (mr_adc.h_orth.dim < mr_adc.nroots):
        mr_adc.nroots = mr_adc.h_orth.dim

    return mr_adc
