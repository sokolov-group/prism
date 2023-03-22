import sys
import time
import numpy as np

def compute_gs_rdms(mr_adc):

    start_time = time.time()

    print("Computing ground-state RDMs...")
    sys.stdout.flush()

    # TODO: for open-shells, this needs to perform state-averaging
    # Compute ground-state RDMs
    if mr_adc.ncas != 0:
        mr_adc.rdm.ca, mr_adc.rdm.ccaa, mr_adc.rdm.cccaaa, mr_adc.rdm.ccccaaaa = mr_adc.interface.compute_rdm1234(mr_adc.wfn_casscf, mr_adc.wfn_casscf, mr_adc.nelecas)
    else:
        mr_adc.rdm.ca = np.zeros((mr_adc.ncas, mr_adc.ncas))
        mr_adc.rdm.ccaa =  np.zeros((mr_adc.ncas, mr_adc.ncas, mr_adc.ncas, mr_adc.ncas))
        mr_adc.rdm.cccaaa =  np.zeros((mr_adc.ncas, mr_adc.ncas, mr_adc.ncas, mr_adc.ncas, mr_adc.ncas, mr_adc.ncas))

    print("Time for computing ground-state RDMs:                          %f sec\n" % (time.time() - start_time))
