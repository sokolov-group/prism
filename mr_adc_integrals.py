import sys
import time
import numpy as np
from functools import reduce
#import prism_beta.disk_helper as disk_helper

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

    print("Time for transforming 1e integrals:                  %f sec\n" % (time.time() - start_time))


# Two-electron integral transformation in Physicists' notation
def transform_2e_phys_incore(interface, mo_1, mo_2, mo_3, mo_4):

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

    print("Transforming 2e integrals to MO basis (in-core)...\n")
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

    #TODO: check for and remove redundant integrals
    if mr_adc.method_type == "ip" or mr_adc.method_type == "ea" or mr_adc.method_type == "cvs-ip":
        if mr_adc.method in ("mr-adc(1)", "mr-adc(2)", "mr-adc(2)-x"):
            mr_adc.v2e.caea = transform_2e_phys_incore(mr_adc.interface, mo_c, mo_a, mo_e, mo_a)
            mr_adc.v2e.caae = transform_2e_phys_incore(mr_adc.interface, mo_c, mo_a, mo_a, mo_e)
            mr_adc.v2e.caaa = transform_2e_phys_incore(mr_adc.interface, mo_c, mo_a, mo_a, mo_a)
            mr_adc.v2e.aaae = transform_2e_phys_incore(mr_adc.interface, mo_a, mo_a, mo_a, mo_e)
            mr_adc.v2e.caca = transform_2e_phys_incore(mr_adc.interface, mo_c, mo_a, mo_c, mo_a)
            mr_adc.v2e.caac = transform_2e_phys_incore(mr_adc.interface, mo_c, mo_a, mo_a, mo_c)

            #TODO: Check if all integrals here are needed
            if mr_adc.method_type == "cvs-ip":
                mr_adc.v2e.xaea = np.ascontiguousarray(mr_adc.v2e.caea[:ncvs,:,:,:])
                mr_adc.v2e.xaae = np.ascontiguousarray(mr_adc.v2e.caae[:ncvs,:,:,:])
                mr_adc.v2e.xaaa = np.ascontiguousarray(mr_adc.v2e.caaa[:ncvs,:,:,:])
                mr_adc.v2e.xaxa = np.ascontiguousarray(mr_adc.v2e.caca[:ncvs,:,:ncvs,:])
                mr_adc.v2e.xaax = np.ascontiguousarray(mr_adc.v2e.caac[:ncvs,:,:,:ncvs])

                # mr_adc.v2e.xaca = np.ascontiguousarray(mr_adc.v2e.caca[:ncvs,:,:,:])
                # mr_adc.v2e.caxa = np.ascontiguousarray(mr_adc.v2e.caca[:,:,:ncvs,:])
                # mr_adc.v2e.xaac = np.ascontiguousarray(mr_adc.v2e.caac[:ncvs,:,:,:])
                # mr_adc.v2e.caax = np.ascontiguousarray(mr_adc.v2e.caac[:,:,:,:ncvs])

                if nval != 0:
                    mr_adc.v2e.vaea = np.ascontiguousarray(mr_adc.v2e.caea[ncvs:,:,:,:])
                    mr_adc.v2e.vaae = np.ascontiguousarray(mr_adc.v2e.caae[ncvs:,:,:,:])
                    mr_adc.v2e.vaaa = np.ascontiguousarray(mr_adc.v2e.caaa[ncvs:,:,:,:])
                    mr_adc.v2e.vava = np.ascontiguousarray(mr_adc.v2e.caca[ncvs:,:,ncvs:,:])
                    mr_adc.v2e.vaav = np.ascontiguousarray(mr_adc.v2e.caac[ncvs:,:,:,ncvs:])

                    # mr_adc.v2e.vaca = np.ascontiguousarray(mr_adc.v2e.caca[ncvs:,:,:,:])
                    # mr_adc.v2e.cava = np.ascontiguousarray(mr_adc.v2e.caca[:,:,ncvs:,:])
                    # mr_adc.v2e.vaac = np.ascontiguousarray(mr_adc.v2e.caac[ncvs:,:,:,:])
                    # mr_adc.v2e.caav = np.ascontiguousarray(mr_adc.v2e.caac[:,:,:,ncvs:])

                    # mr_adc.v2e.vaxa = np.ascontiguousarray(mr_adc.v2e.caca[ncvs:,:,:ncvs,:])
                    # mr_adc.v2e.xava = np.ascontiguousarray(mr_adc.v2e.caca[:ncvs,:,ncvs:,:])
                    # mr_adc.v2e.vaax = np.ascontiguousarray(mr_adc.v2e.caac[ncvs:,:,:,:ncvs])
                    # mr_adc.v2e.xaav = np.ascontiguousarray(mr_adc.v2e.caac[:ncvs,:,:,ncvs:])

        if mr_adc.method in ("mr-adc(2)", "mr-adc(2)-x"):
            mr_adc.v2e.ccee = transform_2e_phys_incore(mr_adc.interface, mo_c, mo_c, mo_e, mo_e)
            mr_adc.v2e.ccea = transform_2e_phys_incore(mr_adc.interface, mo_c, mo_c, mo_e, mo_a)
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

            #TODO: Check if all integrals here are needed
            if mr_adc.method_type == "cvs-ip":
                mr_adc.v2e.xcee = np.ascontiguousarray(mr_adc.v2e.ccee[:ncvs,:,:,:])
                # mr_adc.v2e.cxee = np.ascontiguousarray(mr_adc.v2e.ccee[:,:ncvs,:,:])
                # mr_adc.v2e.xxee = np.ascontiguousarray(mr_adc.v2e.ccee[:ncvs,:ncvs,:,:])
                mr_adc.v2e.xcea = np.ascontiguousarray(mr_adc.v2e.ccea[:ncvs,:,:,:])
                mr_adc.v2e.cxea = np.ascontiguousarray(mr_adc.v2e.ccea[:,:ncvs,:,:])
                mr_adc.v2e.xxea = np.ascontiguousarray(mr_adc.v2e.ccea[:ncvs,:ncvs,:,:])
                mr_adc.v2e.xcae = np.ascontiguousarray(mr_adc.v2e.ccae[:ncvs,:,:,:])
                mr_adc.v2e.cxae = np.ascontiguousarray(mr_adc.v2e.ccae[:,:ncvs,:,:])
                # mr_adc.v2e.xxae = np.ascontiguousarray(mr_adc.v2e.ccae[:ncvs,:ncvs,:,:])
                mr_adc.v2e.xcaa = np.ascontiguousarray(mr_adc.v2e.ccaa[:ncvs,:,:,:])
                # mr_adc.v2e.cxaa = np.ascontiguousarray(mr_adc.v2e.ccaa[:,:ncvs,:,:])
                # mr_adc.v2e.xxaa = np.ascontiguousarray(mr_adc.v2e.ccaa[:ncvs,:ncvs,:,:])
                mr_adc.v2e.xaee = np.ascontiguousarray(mr_adc.v2e.caee[:ncvs,:,:,:])
                mr_adc.v2e.xeee = np.ascontiguousarray(mr_adc.v2e.ceee[:ncvs,:,:,:])
                mr_adc.v2e.xeaa = np.ascontiguousarray(mr_adc.v2e.ceaa[:ncvs,:,:,:])
                mr_adc.v2e.xeae = np.ascontiguousarray(mr_adc.v2e.ceae[:ncvs,:,:,:])
                mr_adc.v2e.xeea = np.ascontiguousarray(mr_adc.v2e.ceea[:ncvs,:,:,:])
                # mr_adc.v2e.xace = np.ascontiguousarray(mr_adc.v2e.cace[:ncvs,:,:,:])
                # mr_adc.v2e.caxe = np.ascontiguousarray(mr_adc.v2e.cace[:,:,:ncvs,:])
                mr_adc.v2e.xaxe = np.ascontiguousarray(mr_adc.v2e.cace[:ncvs,:,:ncvs,:])
                # mr_adc.v2e.xaec = np.ascontiguousarray(mr_adc.v2e.caec[:ncvs,:,:,:])
                # mr_adc.v2e.caex = np.ascontiguousarray(mr_adc.v2e.caec[:,:,:,:ncvs])
                mr_adc.v2e.xaex = np.ascontiguousarray(mr_adc.v2e.caec[:ncvs,:,:,:ncvs])
                mr_adc.v2e.xece = np.ascontiguousarray(mr_adc.v2e.cece[:ncvs,:,:,:])
                mr_adc.v2e.cexe = np.ascontiguousarray(mr_adc.v2e.cece[:,:,:ncvs,:])
                mr_adc.v2e.xexe = np.ascontiguousarray(mr_adc.v2e.cece[:ncvs,:,:ncvs,:])
                mr_adc.v2e.xeec = np.ascontiguousarray(mr_adc.v2e.ceec[:ncvs,:,:,:])
                mr_adc.v2e.ceex = np.ascontiguousarray(mr_adc.v2e.ceec[:,:,:,:ncvs])
                mr_adc.v2e.xeex = np.ascontiguousarray(mr_adc.v2e.ceec[:ncvs,:,:,:ncvs])
                # mr_adc.v2e.xcca = np.ascontiguousarray(mr_adc.v2e.ccca[:ncvs,:,:,:])
                # mr_adc.v2e.cxca = np.ascontiguousarray(mr_adc.v2e.ccca[:,:ncvs,:,:])
                # mr_adc.v2e.ccxa = np.ascontiguousarray(mr_adc.v2e.ccca[:,:,:ncvs,:])
                # mr_adc.v2e.xxca = np.ascontiguousarray(mr_adc.v2e.ccca[:ncvs,:ncvs,:,:])
                mr_adc.v2e.xcxa = np.ascontiguousarray(mr_adc.v2e.ccca[:ncvs,:,:ncvs,:])
                mr_adc.v2e.cxxa = np.ascontiguousarray(mr_adc.v2e.ccca[:,:ncvs,:ncvs,:])
                mr_adc.v2e.xxxa = np.ascontiguousarray(mr_adc.v2e.ccca[:ncvs,:ncvs,:ncvs,:])
                # mr_adc.v2e.xcce = np.ascontiguousarray(mr_adc.v2e.ccce[:ncvs,:,:,:])
                # mr_adc.v2e.cxce = np.ascontiguousarray(mr_adc.v2e.ccce[:,:ncvs,:,:])
                # mr_adc.v2e.ccxe = np.ascontiguousarray(mr_adc.v2e.ccce[:,:,:ncvs,:])
                # mr_adc.v2e.xxce = np.ascontiguousarray(mr_adc.v2e.ccce[:ncvs,:ncvs,:,:])
                mr_adc.v2e.xcxe = np.ascontiguousarray(mr_adc.v2e.ccce[:ncvs,:,:ncvs,:])
                mr_adc.v2e.cxxe = np.ascontiguousarray(mr_adc.v2e.ccce[:,:ncvs,:ncvs,:])
                mr_adc.v2e.xxxe = np.ascontiguousarray(mr_adc.v2e.ccce[:ncvs,:ncvs,:ncvs,:])

                if nval != 0:
                    # mr_adc.v2e.vcee = np.ascontiguousarray(mr_adc.v2e.ccee[ncvs:,:,:,:])
                    # mr_adc.v2e.cvee = np.ascontiguousarray(mr_adc.v2e.ccee[:,ncvs:,:,:])
                    # mr_adc.v2e.vvee = np.ascontiguousarray(mr_adc.v2e.ccee[ncvs:,ncvs:,:,:])
                    # mr_adc.v2e.vcea = np.ascontiguousarray(mr_adc.v2e.ccea[ncvs:,:,:,:])
                    # mr_adc.v2e.cvea = np.ascontiguousarray(mr_adc.v2e.ccea[:,ncvs:,:,:])
                    # mr_adc.v2e.vvea = np.ascontiguousarray(mr_adc.v2e.ccea[ncvs:,ncvs:,:,:])
                    # mr_adc.v2e.vcae = np.ascontiguousarray(mr_adc.v2e.ccae[ncvs:,:,:,:])
                    # mr_adc.v2e.cvae = np.ascontiguousarray(mr_adc.v2e.ccae[:,ncvs:,:,:])
                    # mr_adc.v2e.vvae = np.ascontiguousarray(mr_adc.v2e.ccae[ncvs:,ncvs:,:,:])
                    # mr_adc.v2e.vcaa = np.ascontiguousarray(mr_adc.v2e.ccaa[ncvs:,:,:,:])
                    # mr_adc.v2e.cvaa = np.ascontiguousarray(mr_adc.v2e.ccaa[:,ncvs:,:,:])
                    # mr_adc.v2e.vvaa = np.ascontiguousarray(mr_adc.v2e.ccaa[ncvs:,ncvs:,:,:])
                    # mr_adc.v2e.vaee = np.ascontiguousarray(mr_adc.v2e.caee[ncvs:,:,:,:])
                    # mr_adc.v2e.veee = np.ascontiguousarray(mr_adc.v2e.ceee[ncvs:,:,:,:])
                    # mr_adc.v2e.veaa = np.ascontiguousarray(mr_adc.v2e.ceaa[ncvs:,:,:,:])
                    # mr_adc.v2e.veae = np.ascontiguousarray(mr_adc.v2e.ceae[ncvs:,:,:,:])
                    # mr_adc.v2e.veea = np.ascontiguousarray(mr_adc.v2e.ceea[ncvs:,:,:,:])
                    # mr_adc.v2e.vace = np.ascontiguousarray(mr_adc.v2e.cace[ncvs:,:,:,:])
                    # mr_adc.v2e.cave = np.ascontiguousarray(mr_adc.v2e.cace[:,:,ncvs:,:])
                    # mr_adc.v2e.vave = np.ascontiguousarray(mr_adc.v2e.cace[ncvs:,:,ncvs:,:])
                    # mr_adc.v2e.vaec = np.ascontiguousarray(mr_adc.v2e.caec[ncvs:,:,:,:])
                    # mr_adc.v2e.caev = np.ascontiguousarray(mr_adc.v2e.caec[:,:,:,ncvs:])
                    # mr_adc.v2e.vaev = np.ascontiguousarray(mr_adc.v2e.caec[ncvs:,:,:,ncvs:])
                    # mr_adc.v2e.vece = np.ascontiguousarray(mr_adc.v2e.cece[ncvs:,:,:,:])
                    # mr_adc.v2e.ceve = np.ascontiguousarray(mr_adc.v2e.cece[:,:,ncvs:,:])
                    # mr_adc.v2e.veve = np.ascontiguousarray(mr_adc.v2e.cece[ncvs:,:,ncvs:,:])
                    # mr_adc.v2e.veec = np.ascontiguousarray(mr_adc.v2e.ceec[ncvs:,:,:,:])
                    # mr_adc.v2e.ceev = np.ascontiguousarray(mr_adc.v2e.ceec[:,:,:,ncvs:])
                    # mr_adc.v2e.veev = np.ascontiguousarray(mr_adc.v2e.ceec[ncvs:,:,:,ncvs:])
                    # mr_adc.v2e.vcca = np.ascontiguousarray(mr_adc.v2e.ccca[ncvs:,:,:,:])
                    # mr_adc.v2e.cvca = np.ascontiguousarray(mr_adc.v2e.ccca[:,ncvs:,:,:])
                    # mr_adc.v2e.ccva = np.ascontiguousarray(mr_adc.v2e.ccca[:,:,ncvs:,:])
                    # mr_adc.v2e.vvca = np.ascontiguousarray(mr_adc.v2e.ccca[ncvs:,ncvs:,:,:])
                    # mr_adc.v2e.vcva = np.ascontiguousarray(mr_adc.v2e.ccca[ncvs:,:,ncvs:,:])
                    # mr_adc.v2e.cvva = np.ascontiguousarray(mr_adc.v2e.ccca[:,ncvs:,ncvs:,:])
                    # mr_adc.v2e.vvva = np.ascontiguousarray(mr_adc.v2e.ccca[ncvs:,ncvs:,ncvs:,:])
                    # mr_adc.v2e.vcce = np.ascontiguousarray(mr_adc.v2e.ccce[ncvs:,:,:,:])
                    # mr_adc.v2e.cvce = np.ascontiguousarray(mr_adc.v2e.ccce[:,ncvs:,:,:])
                    # mr_adc.v2e.ccve = np.ascontiguousarray(mr_adc.v2e.ccce[:,:,ncvs:,:])
                    # mr_adc.v2e.vvce = np.ascontiguousarray(mr_adc.v2e.ccce[ncvs:,ncvs:,:,:])
                    # mr_adc.v2e.vcve = np.ascontiguousarray(mr_adc.v2e.ccce[ncvs:,:,ncvs:,:])
                    # mr_adc.v2e.cvve = np.ascontiguousarray(mr_adc.v2e.ccce[:,ncvs:,ncvs:,:])
                    # mr_adc.v2e.vvve = np.ascontiguousarray(mr_adc.v2e.ccce[ncvs:,ncvs:,ncvs:,:])

                    # mr_adc.v2e.vxee = np.ascontiguousarray(mr_adc.v2e.ccee[ncvs:,:ncvs,:,:])
                    # mr_adc.v2e.xvee = np.ascontiguousarray(mr_adc.v2e.ccee[:ncvs,ncvs:,:,:])
                    # mr_adc.v2e.vxea = np.ascontiguousarray(mr_adc.v2e.ccea[ncvs:,:ncvs,:,:])
                    # mr_adc.v2e.xvea = np.ascontiguousarray(mr_adc.v2e.ccea[:ncvs,ncvs:,:,:])
                    # mr_adc.v2e.vxae = np.ascontiguousarray(mr_adc.v2e.ccae[ncvs:,:ncvs,:,:])
                    # mr_adc.v2e.xvae = np.ascontiguousarray(mr_adc.v2e.ccae[:ncvs,ncvs:,:,:])
                    # mr_adc.v2e.vxaa = np.ascontiguousarray(mr_adc.v2e.ccaa[ncvs:,:ncvs,:,:])
                    # mr_adc.v2e.xvaa = np.ascontiguousarray(mr_adc.v2e.ccaa[:ncvs,ncvs:,:,:])
                    # mr_adc.v2e.vaxe = np.ascontiguousarray(mr_adc.v2e.cace[ncvs:,:,:ncvs,:])
                    # mr_adc.v2e.xave = np.ascontiguousarray(mr_adc.v2e.cace[:ncvs,:,ncvs:,:])
                    # mr_adc.v2e.vaex = np.ascontiguousarray(mr_adc.v2e.caec[ncvs:,:,:,:ncvs])
                    # mr_adc.v2e.xaev = np.ascontiguousarray(mr_adc.v2e.caec[:ncvs,:,:,ncvs:])
                    # mr_adc.v2e.vexe = np.ascontiguousarray(mr_adc.v2e.cece[ncvs:,:,:ncvs,:])
                    # mr_adc.v2e.xeve = np.ascontiguousarray(mr_adc.v2e.cece[:ncvs,:,ncvs:,:])
                    # mr_adc.v2e.veex = np.ascontiguousarray(mr_adc.v2e.ceec[ncvs:,:,:,:ncvs])
                    # mr_adc.v2e.xeev = np.ascontiguousarray(mr_adc.v2e.ceec[:ncvs,:,:,ncvs:])
                    # mr_adc.v2e.vvxa = np.ascontiguousarray(mr_adc.v2e.ccca[ncvs:,ncvs:,:ncvs,:])
                    # mr_adc.v2e.vxva = np.ascontiguousarray(mr_adc.v2e.ccca[ncvs:,:ncvs,ncvs:,:])
                    # mr_adc.v2e.xvva = np.ascontiguousarray(mr_adc.v2e.ccca[:ncvs,ncvs:,ncvs:,:])
                    # mr_adc.v2e.vvxe = np.ascontiguousarray(mr_adc.v2e.ccce[ncvs:,ncvs:,:ncvs,:])
                    # mr_adc.v2e.vxve = np.ascontiguousarray(mr_adc.v2e.ccce[ncvs:,:ncvs,ncvs:,:])
                    # mr_adc.v2e.xvve = np.ascontiguousarray(mr_adc.v2e.ccce[:ncvs,ncvs:,ncvs:,:])

                    # mr_adc.v2e.vxca = np.ascontiguousarray(mr_adc.v2e.ccca[ncvs:,:ncvs,:,:])
                    # mr_adc.v2e.vcxa = np.ascontiguousarray(mr_adc.v2e.ccca[ncvs:,:,:ncvs,:])
                    mr_adc.v2e.vxxa = np.ascontiguousarray(mr_adc.v2e.ccca[ncvs:,:ncvs,:ncvs,:])

                    # mr_adc.v2e.xvca = np.ascontiguousarray(mr_adc.v2e.ccca[:ncvs,ncvs:,:,:])
                    # mr_adc.v2e.cvxa = np.ascontiguousarray(mr_adc.v2e.ccca[:,ncvs:,:ncvs,:])
                    mr_adc.v2e.xvxa = np.ascontiguousarray(mr_adc.v2e.ccca[:ncvs,ncvs:,:ncvs,:])

                    # mr_adc.v2e.xcva = np.ascontiguousarray(mr_adc.v2e.ccca[:ncvs,:,ncvs:,:])
                    # mr_adc.v2e.cxva = np.ascontiguousarray(mr_adc.v2e.ccca[:,:ncvs,ncvs:,:])
                    # mr_adc.v2e.xxva = np.ascontiguousarray(mr_adc.v2e.ccca[:ncvs,:ncvs,ncvs:,:])

                    # mr_adc.v2e.vxce = np.ascontiguousarray(mr_adc.v2e.ccce[ncvs:,:ncvs,:,:])
                    # mr_adc.v2e.vcxe = np.ascontiguousarray(mr_adc.v2e.ccce[ncvs:,:,:ncvs,:])
                    mr_adc.v2e.vxxe = np.ascontiguousarray(mr_adc.v2e.ccce[ncvs:,:ncvs,:ncvs,:])

                    # mr_adc.v2e.xvce = np.ascontiguousarray(mr_adc.v2e.ccce[:ncvs,ncvs:,:,:])
                    # mr_adc.v2e.cvxe = np.ascontiguousarray(mr_adc.v2e.ccce[:,ncvs:,:ncvs,:])
                    mr_adc.v2e.xvxe = np.ascontiguousarray(mr_adc.v2e.ccce[:ncvs,ncvs:,:ncvs,:])

                    # mr_adc.v2e.xcve = np.ascontiguousarray(mr_adc.v2e.ccce[:ncvs,:,ncvs:,:])
                    # mr_adc.v2e.cxve = np.ascontiguousarray(mr_adc.v2e.ccce[:,:ncvs,ncvs:,:])
                    # mr_adc.v2e.xxve = np.ascontiguousarray(mr_adc.v2e.ccce[:ncvs,:ncvs,ncvs:,:])

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

    #TODO: Check if all integrals here are needed
    if mr_adc.method_type == "cvs-ip":
        mr_adc.h1eff.xa = np.ascontiguousarray(mr_adc.h1eff.ca[:ncvs,:])
        mr_adc.h1eff.xe = np.ascontiguousarray(mr_adc.h1eff.ce[:ncvs,:])

        if mr_adc.nval != 0:
            mr_adc.h1eff.va = np.ascontiguousarray(mr_adc.h1eff.ca[ncvs:,:])
            mr_adc.h1eff.ve = np.ascontiguousarray(mr_adc.h1eff.ce[ncvs:,:])

    # Store diagonal elements of the generalized Fock operator in spin-orbital basis
    mr_adc.mo_energy.c = mr_adc.interface.mo_energy[:ncore]
    mr_adc.mo_energy.e = mr_adc.interface.mo_energy[nocc:]

    if mr_adc.method_type == "cvs-ip":
        mr_adc.mo_energy.x = mr_adc.interface.mo_energy[:ncvs]

        if nval != 0:
            mr_adc.mo_energy.v = mr_adc.interface.mo_energy[ncvs:ncore]

    print("Time for transforming integrals:                  %f sec\n" % (time.time() - start_time))


