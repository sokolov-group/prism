import sys
import numpy as np
from functools import reduce
import prism_beta.disk_helper as disk_helper

# Transform one-electron integrals to spin-orbital basis
def transform_1e_integrals_so(interface, mo, int_ao = None):

    so = np.zeros((mo.shape[0], 2 * mo.shape[1]))
    so[:, ::2] = mo
    so[:,1::2] = mo
    
    if int_ao is None:
        int_ao = interface.h1e_ao

    int_so = reduce(np.dot, (so.T, int_ao, so))
    int_so[::2,1::2] = int_so[1::2,::2] = 0

    return int_so


# Transform given two-electron integral tensor to spin-orbital antisymmetrized form in Physicists' notation
def transform_asym_integrals_so(interface, mo_1, mo_2, mo_3, mo_4):

    so_1 = np.zeros((mo_1.shape[0], 2 * mo_1.shape[1]))
    so_1[:,  ::2] = mo_1
    so_1[:, 1::2] = mo_1
    so_2 = np.zeros((mo_2.shape[0], 2 * mo_2.shape[1]))
    so_2[:,  ::2] = mo_2
    so_2[:, 1::2] = mo_2
    so_3 = np.zeros((mo_3.shape[0], 2 * mo_3.shape[1]))
    so_3[:,  ::2] = mo_3
    so_3[:, 1::2] = mo_3
    so_4 = np.zeros((mo_4.shape[0], 2 * mo_4.shape[1]))
    so_4[:,  ::2] = mo_4
    so_4[:, 1::2] = mo_4

    v2e = interface.transform_2e_integrals(so_1, so_3, so_2, so_4)
    v2e = v2e.reshape(so_1.shape[1], so_3.shape[1], so_2.shape[1], so_4.shape[1])
    v2e = zero_spin_cases(v2e)
    v2e = v2e.transpose(0,2,1,3).copy()

    if (mo_1 is mo_2):
        v2e -= v2e.transpose(1,0,2,3)
    elif (mo_3 is mo_4):
        v2e -= v2e.transpose(0,1,3,2)
    else:
        v2e_temp = interface.transform_2e_integrals(so_1, so_4, so_2, so_3)
        v2e_temp = v2e_temp.reshape(so_1.shape[1], so_4.shape[1], so_2.shape[1], so_3.shape[1])
        v2e_temp = zero_spin_cases(v2e_temp)
        v2e -= v2e_temp.transpose(0,2,3,1)

    return v2e


def transform_asym_integrals_so_disk(interface, mo_1, mo_2, mo_3, mo_4):

    so_1 = np.zeros((mo_1.shape[0], 2 * mo_1.shape[1]))
    so_1[:,  ::2] = mo_1
    so_1[:, 1::2] = mo_1
    so_2 = np.zeros((mo_2.shape[0], 2 * mo_2.shape[1]))
    so_2[:,  ::2] = mo_2
    so_2[:, 1::2] = mo_2
    so_3 = np.zeros((mo_3.shape[0], 2 * mo_3.shape[1]))
    so_3[:,  ::2] = mo_3
    so_3[:, 1::2] = mo_3
    so_4 = np.zeros((mo_4.shape[0], 2 * mo_4.shape[1]))
    so_4[:,  ::2] = mo_4
    so_4[:, 1::2] = mo_4

    v2e_disk = []

    for p in range(mo_1.shape[1]):

        so_1_slice = so_1[:, 2*p:2*(p+1)]

        v2e = interface.transform_2e_integrals(so_1_slice, so_3, so_2, so_4)
        v2e = v2e.reshape(so_1_slice.shape[1], so_3.shape[1], so_2.shape[1], so_4.shape[1])
        v2e = zero_spin_cases(v2e)
        v2e = v2e.transpose(0,2,1,3).copy()

        if (mo_3 is mo_4):
            v2e -= v2e.transpose(0,1,3,2)
        else:
            v2e_temp = interface.transform_2e_integrals(so_1_slice, so_4, so_2, so_3)
            v2e_temp = v2e_temp.reshape(so_1_slice.shape[1], so_4.shape[1], so_2.shape[1], so_3.shape[1])
            v2e_temp = zero_spin_cases(v2e_temp)
            v2e -= v2e_temp.transpose(0,2,3,1)

        v2e_a = disk_helper.dataset(v2e[0])
        v2e_disk.append(v2e_a)

        v2e_b = disk_helper.dataset(v2e[1])
        v2e_disk.append(v2e_b)

    return v2e_disk


def zero_spin_cases(v2e):

    v2e[::2,1::2] = v2e[1::2,::2] = v2e[:,:,::2,1::2] = v2e[:,:,1::2,::2] = 0.0
    
    return v2e


def remove_dataset(dataset):

    for p in range(len(dataset)):
        disk_helper.remove_dataset(dataset[p])


