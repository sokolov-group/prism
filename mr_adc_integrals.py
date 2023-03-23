import sys
import time
import numpy as np
from functools import reduce

def transform_integrals_1e(mr_adc):

    start_time = time.time()

    print("Transforming 1e integrals to MO basis...")
    sys.stdout.flush()

    mo = mr_adc.mo

    mr_adc.h1e = reduce(np.dot, (mo.T, mr_adc.interface.h1e_ao, mo))

    if mr_adc.method_type in ('ee','cvs-ee'):

        sys.stdout.flush()
        mr_adc.dip_mom = np.zeros((3, mr_adc.nmo, mr_adc.nmo))

        # Dipole moments
        for i in range(3):
            mr_adc.dip_mom[i] = reduce(np.dot, (mo.T, mr_adc.interface.dip_mom_ao[i], mo))

    print("Time for transforming 1e integrals:                %f sec\n" % (time.time() - start_time))

def transform_2e_phys_incore(interface, mo_1, mo_2, mo_3, mo_4):
    'Two-electron integral transformation in Physicists notation'

    nmo_1 = mo_1.shape[1]
    nmo_2 = mo_2.shape[1]
    nmo_3 = mo_3.shape[1]
    nmo_4 = mo_4.shape[1]

    v2e = interface.transform_2e_chem_incore(interface.v2e_ao, (mo_1, mo_3, mo_2, mo_4), compact=False)
    v2e = v2e.reshape(nmo_1, nmo_3, nmo_2, nmo_4)
    v2e = v2e.transpose(0,2,1,3)

    return np.ascontiguousarray(v2e)

def transform_integrals_2e_incore(mr_adc):

    start_time = time.time()

    print("Transforming 2e integrals to MO basis (in-core)...")
    sys.stdout.flush()

    ncvs = mr_adc.ncvs
    nval = mr_adc.nval
    ncore = mr_adc.ncore
    nocc = mr_adc.nocc

    mo = mr_adc.mo
    mo_c = mo[:, :ncore].copy()
    mo_a = mo[:, ncore:nocc].copy()
    mo_e = mo[:, nocc:].copy()

    mr_adc.v2e.aaaa = transform_2e_phys_incore(mr_adc.interface, mo_a, mo_a, mo_a, mo_a)

    if mr_adc.method_type == "ip" or mr_adc.method_type == "ea" or mr_adc.method_type == "cvs-ip":
        if mr_adc.method in ("mr-adc(1)", "mr-adc(2)", "mr-adc(2)-x"):
            mr_adc.v2e.caaa = transform_2e_phys_incore(mr_adc.interface, mo_c, mo_a, mo_a, mo_a)
            mr_adc.v2e.caae = transform_2e_phys_incore(mr_adc.interface, mo_c, mo_a, mo_a, mo_e)
            mr_adc.v2e.caea = transform_2e_phys_incore(mr_adc.interface, mo_c, mo_a, mo_e, mo_a)
            mr_adc.v2e.caca = transform_2e_phys_incore(mr_adc.interface, mo_c, mo_a, mo_c, mo_a)
            mr_adc.v2e.caac = transform_2e_phys_incore(mr_adc.interface, mo_c, mo_a, mo_a, mo_c)
            mr_adc.v2e.aaae = transform_2e_phys_incore(mr_adc.interface, mo_a, mo_a, mo_a, mo_e)

            if mr_adc.method_type == "cvs-ip":
                mr_adc.v2e.xaaa = np.ascontiguousarray(mr_adc.v2e.caaa[:ncvs,:,:,:])
                mr_adc.v2e.xaae = np.ascontiguousarray(mr_adc.v2e.caae[:ncvs,:,:,:])
                mr_adc.v2e.xaea = np.ascontiguousarray(mr_adc.v2e.caea[:ncvs,:,:,:])
                mr_adc.v2e.xaxa = np.ascontiguousarray(mr_adc.v2e.caca[:ncvs,:,:ncvs,:])
                mr_adc.v2e.xaax = np.ascontiguousarray(mr_adc.v2e.caac[:ncvs,:,:,:ncvs])

                if nval > 0:
                    mr_adc.v2e.vaaa = np.ascontiguousarray(mr_adc.v2e.caaa[ncvs:,:,:,:])
                    mr_adc.v2e.vaae = np.ascontiguousarray(mr_adc.v2e.caae[ncvs:,:,:,:])
                    mr_adc.v2e.vaea = np.ascontiguousarray(mr_adc.v2e.caea[ncvs:,:,:,:])

        if mr_adc.method in ("mr-adc(2)", "mr-adc(2)-x"):
            mr_adc.v2e.ccee = transform_2e_phys_incore(mr_adc.interface, mo_c, mo_c, mo_e, mo_e)
            mr_adc.v2e.ccae = transform_2e_phys_incore(mr_adc.interface, mo_c, mo_c, mo_a, mo_e)
            mr_adc.v2e.caee = transform_2e_phys_incore(mr_adc.interface, mo_c, mo_a, mo_e, mo_e)
            mr_adc.v2e.ccaa = transform_2e_phys_incore(mr_adc.interface, mo_c, mo_c, mo_a, mo_a)
            mr_adc.v2e.aaee = transform_2e_phys_incore(mr_adc.interface, mo_a, mo_a, mo_e, mo_e)
            mr_adc.v2e.ccca = transform_2e_phys_incore(mr_adc.interface, mo_c, mo_c, mo_c, mo_a)
            mr_adc.v2e.ccce = transform_2e_phys_incore(mr_adc.interface, mo_c, mo_c, mo_c, mo_e)
            mr_adc.v2e.cace = transform_2e_phys_incore(mr_adc.interface, mo_c, mo_a, mo_c, mo_e)
            mr_adc.v2e.caec = transform_2e_phys_incore(mr_adc.interface, mo_c, mo_a, mo_e, mo_c)
            mr_adc.v2e.ceaa = transform_2e_phys_incore(mr_adc.interface, mo_c, mo_e, mo_a, mo_a)
            mr_adc.v2e.cece = transform_2e_phys_incore(mr_adc.interface, mo_c, mo_e, mo_c, mo_e)
            mr_adc.v2e.ceec = transform_2e_phys_incore(mr_adc.interface, mo_c, mo_e, mo_e, mo_c)
            mr_adc.v2e.ceee = transform_2e_phys_incore(mr_adc.interface, mo_c, mo_e, mo_e, mo_e)
            mr_adc.v2e.aeee = transform_2e_phys_incore(mr_adc.interface, mo_a, mo_e, mo_e, mo_e)
            mr_adc.v2e.ceae = transform_2e_phys_incore(mr_adc.interface, mo_c, mo_e, mo_a, mo_e)
            mr_adc.v2e.ceea = transform_2e_phys_incore(mr_adc.interface, mo_c, mo_e, mo_e, mo_a)
            mr_adc.v2e.aeae = transform_2e_phys_incore(mr_adc.interface, mo_a, mo_e, mo_a, mo_e)
            mr_adc.v2e.aeea = transform_2e_phys_incore(mr_adc.interface, mo_a, mo_e, mo_e, mo_a)

            if mr_adc.method_type == "cvs-ip":
                mr_adc.v2e.xcee = np.ascontiguousarray(mr_adc.v2e.ccee[:ncvs,:,:,:])
                mr_adc.v2e.xcae = np.ascontiguousarray(mr_adc.v2e.ccae[:ncvs,:,:,:])
                mr_adc.v2e.cxae = np.ascontiguousarray(mr_adc.v2e.ccae[:,:ncvs,:,:])
                mr_adc.v2e.xcaa = np.ascontiguousarray(mr_adc.v2e.ccaa[:ncvs,:,:,:])
                mr_adc.v2e.xaee = np.ascontiguousarray(mr_adc.v2e.caee[:ncvs,:,:,:])
                mr_adc.v2e.xaxe = np.ascontiguousarray(mr_adc.v2e.cace[:ncvs,:,:ncvs,:])
                mr_adc.v2e.xaex = np.ascontiguousarray(mr_adc.v2e.caec[:ncvs,:,:,:ncvs])
                mr_adc.v2e.xcxa = np.ascontiguousarray(mr_adc.v2e.ccca[:ncvs,:,:ncvs,:])
                mr_adc.v2e.cxxa = np.ascontiguousarray(mr_adc.v2e.ccca[:,:ncvs,:ncvs,:])
                mr_adc.v2e.xxxa = np.ascontiguousarray(mr_adc.v2e.ccca[:ncvs,:ncvs,:ncvs,:])
                mr_adc.v2e.xcxe = np.ascontiguousarray(mr_adc.v2e.ccce[:ncvs,:,:ncvs,:])
                mr_adc.v2e.cxxe = np.ascontiguousarray(mr_adc.v2e.ccce[:,:ncvs,:ncvs,:])
                mr_adc.v2e.xxxe = np.ascontiguousarray(mr_adc.v2e.ccce[:ncvs,:ncvs,:ncvs,:])

                if nval > 0:
                    mr_adc.v2e.vxxa = np.ascontiguousarray(mr_adc.v2e.ccca[ncvs:,:ncvs,:ncvs,:])
                    mr_adc.v2e.xvxa = np.ascontiguousarray(mr_adc.v2e.ccca[:ncvs,ncvs:,:ncvs,:])

                    mr_adc.v2e.vxxe = np.ascontiguousarray(mr_adc.v2e.ccce[ncvs:,:ncvs,:ncvs,:])
                    mr_adc.v2e.xvxe = np.ascontiguousarray(mr_adc.v2e.ccce[:ncvs,ncvs:,:ncvs,:])

            if mr_adc.method == "mr-adc(2)-x":
                mr_adc.v2e.cccc = transform_2e_phys_incore(mr_adc.interface, mo_c, mo_c, mo_c, mo_c)

            if (mr_adc.method == "mr-adc(2)-x" and mr_adc.method_type == "ea"):
                mr_adc.v2e.eeee = transform_2e_phys_incore(mr_adc.interface, mo_e, mo_e, mo_e, mo_e)

    # Effective one-electron integrals
    gcgc = transform_2e_phys_incore(mr_adc.interface, mo, mo_c, mo, mo_c)
    gccg = transform_2e_phys_incore(mr_adc.interface, mo, mo_c, mo_c, mo)
    h1eff = mr_adc.h1e + 2.0 * np.einsum('prqr->pq', gcgc) - np.einsum('prrq->pq', gccg)

    mr_adc.h1eff.ca = np.ascontiguousarray(h1eff[:ncore, ncore:nocc])
    mr_adc.h1eff.ce = np.ascontiguousarray(h1eff[:ncore, nocc:])
    mr_adc.h1eff.aa = np.ascontiguousarray(h1eff[ncore:nocc, ncore:nocc])
    mr_adc.h1eff.ae = np.ascontiguousarray(h1eff[ncore:nocc, nocc:])

    if mr_adc.method_type == "cvs-ip":
        mr_adc.h1eff.xa = np.ascontiguousarray(mr_adc.h1eff.ca[:ncvs,:])
        mr_adc.h1eff.xe = np.ascontiguousarray(mr_adc.h1eff.ce[:ncvs,:])

        if mr_adc.nval > 0:
            mr_adc.h1eff.va = np.ascontiguousarray(mr_adc.h1eff.ca[ncvs:,:])
            mr_adc.h1eff.ve = np.ascontiguousarray(mr_adc.h1eff.ce[ncvs:,:])

    # Store diagonal elements of the generalized Fock operator in spin-orbital basis
    mr_adc.mo_energy.c = mr_adc.interface.mo_energy[:ncore]
    mr_adc.mo_energy.e = mr_adc.interface.mo_energy[nocc:]

    if mr_adc.method_type == "cvs-ip":
        mr_adc.mo_energy.x = mr_adc.interface.mo_energy[:ncvs]

        if nval > 0:
            mr_adc.mo_energy.v = mr_adc.interface.mo_energy[ncvs:ncore]

    print("Time for transforming integrals:                   %f sec\n" % (time.time() - start_time))


