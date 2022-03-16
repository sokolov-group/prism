import sys
import time
import numpy as np
from functools import reduce
#import prism_beta.disk_helper as disk_helper

def transform_integrals_1e(mr_adc):

    start_time = time.time()

    print ("Transforming 1e integrals to MO basis...")
    sys.stdout.flush()

    mo = mr_adc.mo

    mr_adc.h1e = reduce(np.dot, (mo.T, mr_adc.interface.h1e_ao, mo))

    if mr_adc.method_type in ('ee','cvs-ee'):

        sys.stdout.flush()
        mr_adc.dip_mom = np.zeros((3, mr_adc.nmo, mr_adc.nmo))

        # Dipole moments
        for i in range(3):
            mr_adc.dip_mom[i] = reduce(np.dot, (mo.T, mr_adc.interface.dip_mom_ao[i], mo))

    print ("Time for transforming 1e integrals:                  %f sec\n" % (time.time() - start_time))


# Two-electron integral transformation in Physicists' notation
def transform_2e_phys_incore(interface, mo_1, mo_2, mo_3, mo_4):

    nmo_1 = mo_1.shape[1]
    nmo_2 = mo_2.shape[1]
    nmo_3 = mo_3.shape[1]
    nmo_4 = mo_4.shape[1]

    v2e = interface.transform_2e_chem_incore(interface.v2e_ao, (mo_1, mo_3, mo_2, mo_4), compact=False).reshape(nmo_1, nmo_3, nmo_2, nmo_4)
    v2e = v2e.transpose(0,2,1,3)

    return np.ascontiguousarray(v2e)


def transform_integrals_2e_incore(mr_adc):

    start_time = time.time()

    print ("Transforming 2e integrals to MO basis (in-core)...\n")
    sys.stdout.flush()

    mo = mr_adc.mo

    mo_c = mo[:, :mr_adc.ncore].copy()
    mo_a = mo[:, mr_adc.ncore:mr_adc.nocc].copy()
    mo_e = mo[:, mr_adc.nocc:].copy()

    mr_adc.v2e.aaaa = transform_2e_phys_incore(mr_adc.interface, mo_a, mo_a, mo_a, mo_a)

    #TODO: check for and remove redundant integrals
    if mr_adc.method_type == "ip" or mr_adc.method_type == "ea" or mr_adc.method_type == "cvs-ip":
        if mr_adc.method in ("mr-adc(1)", "mr-adc(2)", "mr-adc(2)-x"):
            mr_adc.v2e.caea = transform_2e_phys_incore(mr_adc.interface, mo_c, mo_a, mo_e, mo_a)
            mr_adc.v2e.caaa = transform_2e_phys_incore(mr_adc.interface, mo_c, mo_a, mo_a, mo_a)
            mr_adc.v2e.aaea = transform_2e_phys_incore(mr_adc.interface, mo_a, mo_a, mo_e, mo_a)
            mr_adc.v2e.caca = transform_2e_phys_incore(mr_adc.interface, mo_c, mo_a, mo_c, mo_a)

        if mr_adc.method in ("mr-adc(2)", "mr-adc(2)-x"):
            mr_adc.v2e.ccee = transform_2e_phys_incore(mr_adc.interface, mo_c, mo_c, mo_e, mo_e)
            mr_adc.v2e.ccea = transform_2e_phys_incore(mr_adc.interface, mo_c, mo_c, mo_e, mo_a)
            mr_adc.v2e.caee = transform_2e_phys_incore(mr_adc.interface, mo_c, mo_a, mo_e, mo_e)
            mr_adc.v2e.ccaa = transform_2e_phys_incore(mr_adc.interface, mo_c, mo_c, mo_a, mo_a)
            mr_adc.v2e.aaee = transform_2e_phys_incore(mr_adc.interface, mo_a, mo_a, mo_e, mo_e)
            mr_adc.v2e.ccca = transform_2e_phys_incore(mr_adc.interface, mo_c, mo_c, mo_c, mo_a)
            mr_adc.v2e.ccce = transform_2e_phys_incore(mr_adc.interface, mo_c, mo_c, mo_c, mo_e)
            mr_adc.v2e.cace = transform_2e_phys_incore(mr_adc.interface, mo_c, mo_a, mo_c, mo_e)
            mr_adc.v2e.ceaa = transform_2e_phys_incore(mr_adc.interface, mo_c, mo_e, mo_a, mo_a)
            mr_adc.v2e.cece = transform_2e_phys_incore(mr_adc.interface, mo_c, mo_e, mo_c, mo_e)
            mr_adc.v2e.ceee = transform_2e_phys_incore(mr_adc.interface, mo_c, mo_e, mo_e, mo_e)
            mr_adc.v2e.aeee = transform_2e_phys_incore(mr_adc.interface, mo_a, mo_e, mo_e, mo_e)
            mr_adc.v2e.ceae = transform_2e_phys_incore(mr_adc.interface, mo_c, mo_e, mo_a, mo_e)
            mr_adc.v2e.aeae = transform_2e_phys_incore(mr_adc.interface, mo_a, mo_e, mo_a, mo_e)

            # Need for mr-adc(2)-x
            if mr_adc.method == "mr-adc(2)-x":
                mr_adc.v2e.cccc = transform_2e_phys_incore(mr_adc.interface, mo_c, mo_c, mo_c, mo_c)

            if (mr_adc.method == "mr-adc(2)-x" and mr_adc.method_type == "ea"):
                mr_adc.v2e.eeee = transform_2e_phys_incore(mr_adc.interface, mo_e, mo_e, mo_e, mo_e)

    elif mr_adc.method_type == "ee":
        if mr_adc.method in ("mr-adc(1)", "mr-adc(2)", "mr-adc(2)-x"):
            mr_adc.v2e.caea = transform_2e_phys_incore(mr_adc.interface, mo_c, mo_a, mo_e, mo_a)
            mr_adc.v2e.caaa = transform_2e_phys_incore(mr_adc.interface, mo_c, mo_a, mo_a, mo_a)
            mr_adc.v2e.aaea = transform_2e_phys_incore(mr_adc.interface, mo_a, mo_a, mo_e, mo_a)
            mr_adc.v2e.caca = transform_2e_phys_incore(mr_adc.interface, mo_c, mo_a, mo_c, mo_a)
            mr_adc.v2e.ccee = transform_2e_phys_incore(mr_adc.interface, mo_c, mo_c, mo_e, mo_e)
            mr_adc.v2e.aaee = transform_2e_phys_incore(mr_adc.interface, mo_a, mo_a, mo_e, mo_e)
            mr_adc.v2e.caee = transform_2e_phys_incore(mr_adc.interface, mo_c, mo_a, mo_e, mo_e)
            mr_adc.v2e.cace = transform_2e_phys_incore(mr_adc.interface, mo_c, mo_a, mo_c, mo_e)
            mr_adc.v2e.cece = transform_2e_phys_incore(mr_adc.interface, mo_c, mo_e, mo_c, mo_e)
            mr_adc.v2e.aeae = transform_2e_phys_incore(mr_adc.interface, mo_a, mo_e, mo_a, mo_e)
            mr_adc.v2e.ceae = transform_2e_phys_incore(mr_adc.interface, mo_c, mo_e, mo_a, mo_e)
            mr_adc.v2e.ccea = transform_2e_phys_incore(mr_adc.interface, mo_c, mo_c, mo_e, mo_a)
            mr_adc.v2e.ceaa = transform_2e_phys_incore(mr_adc.interface, mo_c, mo_e, mo_a, mo_a)
            mr_adc.v2e.ccaa = transform_2e_phys_incore(mr_adc.interface, mo_c, mo_c, mo_a, mo_a)

        if mr_adc.method in ("mr-adc(2)", "mr-adc(2)-x"): 
            mr_adc.v2e.ccca = transform_2e_phys_incore(mr_adc.interface, mo_c, mo_c, mo_c, mo_a)
            mr_adc.v2e.ccce = transform_2e_phys_incore(mr_adc.interface, mo_c, mo_c, mo_c, mo_e)
            mr_adc.v2e.ceee = transform_2e_phys_incore(mr_adc.interface, mo_c, mo_e, mo_e, mo_e)
            mr_adc.v2e.aeee = transform_2e_phys_incore(mr_adc.interface, mo_a, mo_e, mo_e, mo_e)

            if mr_adc.method == "mr-adc(2)-x":
                mr_adc.v2e.cccc = transform_2e_phys_incore(mr_adc.interface, mo_c, mo_c, mo_c, mo_c)
                mr_adc.v2e.eeee = transform_2e_phys_incore(mr_adc.interface, mo_e, mo_e, mo_e, mo_e)

    elif mr_adc.method_type == "cvs-ee":
        if mr_adc.method in ("mr-adc(1)", "mr-adc(2)", "mr-adc(2)-x"):
            mr_adc.v2e.caea = transform_2e_phys_incore(mr_adc.interface, mo_c, mo_a, mo_e, mo_a)
            mr_adc.v2e.caaa = transform_2e_phys_incore(mr_adc.interface, mo_c, mo_a, mo_a, mo_a)
            mr_adc.v2e.aaea = transform_2e_phys_incore(mr_adc.interface, mo_a, mo_a, mo_e, mo_a)
            mr_adc.v2e.caca = transform_2e_phys_incore(mr_adc.interface, mo_c, mo_a, mo_c, mo_a)
            mr_adc.v2e.ccee = transform_2e_phys_incore(mr_adc.interface, mo_c, mo_c, mo_e, mo_e)
            mr_adc.v2e.aaee = transform_2e_phys_incore(mr_adc.interface, mo_a, mo_a, mo_e, mo_e)
            mr_adc.v2e.caee = transform_2e_phys_incore(mr_adc.interface, mo_c, mo_a, mo_e, mo_e)
            mr_adc.v2e.cace = transform_2e_phys_incore(mr_adc.interface, mo_c, mo_a, mo_c, mo_e)
            mr_adc.v2e.cece = transform_2e_phys_incore(mr_adc.interface, mo_c, mo_e, mo_c, mo_e)
            mr_adc.v2e.aeae = transform_2e_phys_incore(mr_adc.interface, mo_a, mo_e, mo_a, mo_e)
            mr_adc.v2e.ceae = transform_2e_phys_incore(mr_adc.interface, mo_c, mo_e, mo_a, mo_e)
            mr_adc.v2e.ccea = transform_2e_phys_incore(mr_adc.interface, mo_c, mo_c, mo_e, mo_a)
            mr_adc.v2e.ceaa = transform_2e_phys_incore(mr_adc.interface, mo_c, mo_e, mo_a, mo_a)
            mr_adc.v2e.ccaa = transform_2e_phys_incore(mr_adc.interface, mo_c, mo_c, mo_a, mo_a)

        if mr_adc.method in ("mr-adc(2)", "mr-adc(2)-x"): 
            mr_adc.v2e.ccca = transform_2e_phys_incore(mr_adc.interface, mo_c, mo_c, mo_c, mo_a)
            mr_adc.v2e.ccce = transform_2e_phys_incore(mr_adc.interface, mo_c, mo_c, mo_c, mo_e)
            mr_adc.v2e.ceee = transform_2e_phys_incore(mr_adc.interface, mo_c, mo_e, mo_e, mo_e)
            mr_adc.v2e.aeee = transform_2e_phys_incore(mr_adc.interface, mo_a, mo_e, mo_e, mo_e)

            if mr_adc.method == "mr-adc(2)-x":
                mr_adc.v2e.cccc = transform_2e_phys_incore(mr_adc.interface, mo_c, mo_c, mo_c, mo_c)
                mr_adc.v2e.eeee = transform_2e_phys_incore(mr_adc.interface, mo_e, mo_e, mo_e, mo_e)

    # Effective one-electron integrals
    gcgc = transform_2e_phys_incore(mr_adc.interface, mo, mo_c, mo, mo_c)
    gccg = transform_2e_phys_incore(mr_adc.interface, mo, mo_c, mo_c, mo)
    mr_adc.h1eff = mr_adc.h1e + 2.0 * np.einsum('prqr->pq', gcgc) - np.einsum('prrq->pq', gccg)
    mr_adc.h1eff_act = mr_adc.h1eff[mr_adc.ncore:mr_adc.nocc, mr_adc.ncore:mr_adc.nocc].copy()

    # Store diagonal elements of the generalized Fock operator in spin-orbital basis
    mr_adc.mo_energy.c = mr_adc.interface.mo_energy[:mr_adc.ncore]
    mr_adc.mo_energy.e = mr_adc.interface.mo_energy[(mr_adc.ncore + mr_adc.ncas):]

    print ("Time for transforming integrals:                  %f sec\n" % (time.time() - start_time))


