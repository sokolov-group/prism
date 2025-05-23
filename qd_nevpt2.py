# Copyright 2025 Prism Developers. All Rights Reserved.
#
# Licensed under the GNU General Public License v3.0;
# you may not use this file except in compliance with the License.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
# either express or implied.
#
# See the License file for the specific language governing
# permissions and limitations.
#
# Available at https://github.com/sokolov-group/prism
#
# Authors: Alexander Yu. Sokolov <alexander.y.sokolov@gmail.com>
#

import numpy as np
from functools import reduce

import prism.lib.logger as logger

def compute_energy(nevpt, e_diag, t1, t1_0):

    # Einsum definition from kernel
    einsum = nevpt.interface.einsum
    einsum_type = nevpt.interface.einsum_type

    ncore = nevpt.ncore - nevpt.nfrozen
    ncas = nevpt.ncas
    nelecas = nevpt.ref_nelecas
    nextern = nevpt.nextern

    h_eff = np.diag(e_diag)
    dim = h_eff.shape[0]

    t1_ccee = t1_0

    ## One-electron integrals
    h_ca = nevpt.h1eff.ca
    h_ce = nevpt.h1eff.ce
    h_ae = nevpt.h1eff.ae

    ## Two-electron integrals
    v_cace = nevpt.v2e.cace
    v_caca = nevpt.v2e.caca
    v_ceae = nevpt.v2e.ceae
    v_ceaa = nevpt.v2e.ceaa
    v_caae = nevpt.v2e.caae
    v_caaa = nevpt.v2e.caaa
    v_aeae = nevpt.v2e.aeae
    v_aaae = nevpt.v2e.aaae

    # TODO: Optimize memory usage for the ee integrals
    # TODO: test frozen core

    for I in range(dim):
        for J in range(I-1):
            # Compute transition density matrices
            trdm_ca, trdm_ccaa, trdm_cccaaa = nevpt.interface.compute_rdm123(nevpt.ref_wfn[I], nevpt.ref_wfn[J], nevpt.ref_nelecas[I])

            # Compute the effective Hamiltonian matrix elements
            # TODO: optimize memory usage by grouping terms that depend on the same amplitudes and freeing memory after use
            # 0.5 * < Psi_I | V * T | Psi_J >
            t1_caea = t1[J].caea
            t1_caae = t1[J].caae
            t1_caaa = t1[J].caaa
            t1_aaae = t1[J].aaae
            t1_ccae = t1[J].ccae
            t1_caee = t1[J].caee
            t1_ccaa = t1[J].ccaa
            t1_aaee = t1[J].aaee

            H_IJ  = einsum('ia,ixay,yx', h_ce, t1_caea, trdm_ca, optimize = einsum_type)
            H_IJ -= 1/2 * einsum('ia,ixya,yx', h_ce, t1_caae, trdm_ca, optimize = einsum_type)
#            H_IJ -= 1/2 * einsum('ix,iy,yx', h_ca, t1_ca, trdm_ca, optimize = einsum_type)
            H_IJ += einsum('ix,iyxz,zy', h_ca, t1_caaa, trdm_ca, optimize = einsum_type)
            H_IJ -= 2/3 * einsum('ix,iyzw,zwxy', h_ca, t1_caaa, trdm_ccaa, optimize = einsum_type)
            H_IJ += 1/6 * einsum('ix,iyzw,zwyx', h_ca, t1_caaa, trdm_ccaa, optimize = einsum_type)
            H_IJ -= 1/2 * einsum('ix,iyzx,zy', h_ca, t1_caaa, trdm_ca, optimize = einsum_type)
#            H_IJ += 1/2 * einsum('xa,ya,xy', h_ae, t1_ae, trdm_ca, optimize = einsum_type)
            H_IJ -= 1/6 * einsum('xa,yzwa,xwyz', h_ae, t1_aaae, trdm_ccaa, optimize = einsum_type)
            H_IJ += 2/3 * einsum('xa,yzwa,xwzy', h_ae, t1_aaae, trdm_ccaa, optimize = einsum_type)
#            H_IJ += einsum('ia,iaxy,xy', t1_ce, v_ceaa, trdm_ca, optimize = einsum_type)
#            H_IJ -= 1/2 * einsum('ia,ixya,yx', t1_ce, v_caae, trdm_ca, optimize = einsum_type)
            H_IJ -= einsum('ijxa,iyja,xy', t1_ccae, v_cace, trdm_ca, optimize = einsum_type)
            H_IJ += 1/2 * einsum('ijxa,jyia,xy', t1_ccae, v_cace, trdm_ca, optimize = einsum_type)
            H_IJ -= einsum('ijxy,ixjz,yz', t1_ccaa, v_caca, trdm_ca, optimize = einsum_type)
            H_IJ += 1/2 * einsum('ijxy,iyjz,xz', t1_ccaa, v_caca, trdm_ca, optimize = einsum_type)
            H_IJ -= 1/12 * einsum('ijxy,izjw,xywz', t1_ccaa, v_caca, trdm_ccaa, optimize = einsum_type)
            H_IJ += 1/3 * einsum('ijxy,izjw,xyzw', t1_ccaa, v_caca, trdm_ccaa, optimize = einsum_type)
#            H_IJ += einsum('ix,ixyz,yz', t1_ca, v_caaa, trdm_ca, optimize = einsum_type)
#            H_IJ += 1/6 * einsum('ix,iyzw,xzwy', t1_ca, v_caaa, trdm_ccaa, optimize = einsum_type)
#            H_IJ -= 2/3 * einsum('ix,iyzw,xzyw', t1_ca, v_caaa, trdm_ccaa, optimize = einsum_type)
#            H_IJ -= 1/2 * einsum('ix,iyzx,zy', t1_ca, v_caaa, trdm_ca, optimize = einsum_type)
            H_IJ += einsum('ixab,iayb,yx', t1_caee, v_ceae, trdm_ca, optimize = einsum_type)
            H_IJ -= 1/2 * einsum('ixab,ibya,yx', t1_caee, v_ceae, trdm_ca, optimize = einsum_type)
            H_IJ -= 1/3 * einsum('ixay,iazw,yzwx', t1_caea, v_ceaa, trdm_ccaa, optimize = einsum_type)
            H_IJ += 4/3 * einsum('ixay,iazw,yzxw', t1_caea, v_ceaa, trdm_ccaa, optimize = einsum_type)
            H_IJ += einsum('ixay,iazy,zx', t1_caea, v_ceaa, trdm_ca, optimize = einsum_type)
            H_IJ -= 1/2 * einsum('ixay,iyza,zx', t1_caea, v_caae, trdm_ca, optimize = einsum_type)
            H_IJ -= 2/3 * einsum('ixay,izwa,ywxz', t1_caea, v_caae, trdm_ccaa, optimize = einsum_type)
            H_IJ += 1/6 * einsum('ixay,izwa,ywzx', t1_caea, v_caae, trdm_ccaa, optimize = einsum_type)
            H_IJ += 1/6 * einsum('ixya,iazw,yzwx', t1_caae, v_ceaa, trdm_ccaa, optimize = einsum_type)
            H_IJ -= 2/3 * einsum('ixya,iazw,yzxw', t1_caae, v_ceaa, trdm_ccaa, optimize = einsum_type)
            H_IJ -= 1/2 * einsum('ixya,iazy,zx', t1_caae, v_ceaa, trdm_ca, optimize = einsum_type)
            H_IJ += einsum('ixya,iyza,zx', t1_caae, v_caae, trdm_ca, optimize = einsum_type)
            H_IJ += 1/6 * einsum('ixya,izwa,ywxz', t1_caae, v_caae, trdm_ccaa, optimize = einsum_type)
            H_IJ -= 2/3 * einsum('ixya,izwa,ywzx', t1_caae, v_caae, trdm_ccaa, optimize = einsum_type)
            H_IJ += 1/6 * einsum('ixyz,iwuv,yzuvwx', t1_caaa, v_caaa, trdm_cccaaa, optimize = einsum_type)
            H_IJ += 1/6 * einsum('ixyz,iwuv,yzuvxw', t1_caaa, v_caaa, trdm_cccaaa, optimize = einsum_type)
            H_IJ += 1/6 * einsum('ixyz,iwuv,yzuwvx', t1_caaa, v_caaa, trdm_cccaaa, optimize = einsum_type)
            H_IJ -= 1/3 * einsum('ixyz,iwuv,yzuwxv', t1_caaa, v_caaa, trdm_cccaaa, optimize = einsum_type)
            H_IJ += 1/6 * einsum('ixyz,iwuv,yzuxvw', t1_caaa, v_caaa, trdm_cccaaa, optimize = einsum_type)
            H_IJ += 1/6 * einsum('ixyz,iwuv,yzuxwv', t1_caaa, v_caaa, trdm_cccaaa, optimize = einsum_type)
            H_IJ += 1/6 * einsum('ixyz,iwuy,zuwx', t1_caaa, v_caaa, trdm_ccaa, optimize = einsum_type)
            H_IJ -= 2/3 * einsum('ixyz,iwuy,zuxw', t1_caaa, v_caaa, trdm_ccaa, optimize = einsum_type)
            H_IJ -= 2/3 * einsum('ixyz,iwuz,yuwx', t1_caaa, v_caaa, trdm_ccaa, optimize = einsum_type)
            H_IJ += 1/6 * einsum('ixyz,iwuz,yuxw', t1_caaa, v_caaa, trdm_ccaa, optimize = einsum_type)
            H_IJ -= 1/3 * einsum('ixyz,iywu,zwux', t1_caaa, v_caaa, trdm_ccaa, optimize = einsum_type)
            H_IJ += 4/3 * einsum('ixyz,iywu,zwxu', t1_caaa, v_caaa, trdm_ccaa, optimize = einsum_type)
            H_IJ += einsum('ixyz,iywz,wx', t1_caaa, v_caaa, trdm_ca, optimize = einsum_type)
            H_IJ += 1/6 * einsum('ixyz,izwu,ywux', t1_caaa, v_caaa, trdm_ccaa, optimize = einsum_type)
            H_IJ -= 2/3 * einsum('ixyz,izwu,ywxu', t1_caaa, v_caaa, trdm_ccaa, optimize = einsum_type)
            H_IJ -= 1/2 * einsum('ixyz,izwy,wx', t1_caaa, v_caaa, trdm_ca, optimize = einsum_type)
#            H_IJ += 2/3 * einsum('xa,yzwa,wyxz', t1_ae, v_aaae, trdm_ccaa, optimize = einsum_type)
#            H_IJ -= 1/6 * einsum('xa,yzwa,wyzx', t1_ae, v_aaae, trdm_ccaa, optimize = einsum_type)
            H_IJ += 1/3 * einsum('xyab,zawb,zwxy', t1_aaee, v_aeae, trdm_ccaa, optimize = einsum_type)
            H_IJ -= 1/12 * einsum('xyab,zawb,zwyx', t1_aaee, v_aeae, trdm_ccaa, optimize = einsum_type)
            H_IJ -= 1/6 * einsum('xyza,wuva,zvwuxy', t1_aaae, v_aaae, trdm_cccaaa, optimize = einsum_type)
            H_IJ -= 1/6 * einsum('xyza,wuva,zvwuyx', t1_aaae, v_aaae, trdm_cccaaa, optimize = einsum_type)
            H_IJ -= 1/6 * einsum('xyza,wuva,zvwxuy', t1_aaae, v_aaae, trdm_cccaaa, optimize = einsum_type)
            H_IJ += 1/3 * einsum('xyza,wuva,zvwxyu', t1_aaae, v_aaae, trdm_cccaaa, optimize = einsum_type)
            H_IJ -= 1/6 * einsum('xyza,wuva,zvwyux', t1_aaae, v_aaae, trdm_cccaaa, optimize = einsum_type)
            H_IJ -= 1/6 * einsum('xyza,wuva,zvwyxu', t1_aaae, v_aaae, trdm_cccaaa, optimize = einsum_type)
            H_IJ += 2/3 * einsum('xyza,wzua,wuxy', t1_aaae, v_aaae, trdm_ccaa, optimize = einsum_type)
            H_IJ -= 1/6 * einsum('xyza,wzua,wuyx', t1_aaae, v_aaae, trdm_ccaa, optimize = einsum_type)

            t1_caea = t1[I].caea
            t1_caae = t1[I].caae
            t1_caaa = t1[I].caaa
            t1_aaae = t1[I].aaae
            t1_ccae = t1[I].ccae
            t1_caee = t1[I].caee
            t1_ccaa = t1[I].ccaa
            t1_aaee = t1[I].aaee

            H_IJ += einsum('ia,ixay,xy', h_ce, t1_caea, trdm_ca, optimize = einsum_type)
            H_IJ -= 1/2 * einsum('ia,ixya,xy', h_ce, t1_caae, trdm_ca, optimize = einsum_type)
#            H_IJ -= 1/2 * einsum('ix,iy,xy', h_ca, t1_ca, trdm_ca, optimize = einsum_type)
            H_IJ += einsum('ix,iyxz,yz', h_ca, t1_caaa, trdm_ca, optimize = einsum_type)
            H_IJ += 1/6 * einsum('ix,iyzw,xywz', h_ca, t1_caaa, trdm_ccaa, optimize = einsum_type)
            H_IJ -= 2/3 * einsum('ix,iyzw,xyzw', h_ca, t1_caaa, trdm_ccaa, optimize = einsum_type)
            H_IJ -= 1/2 * einsum('ix,iyzx,yz', h_ca, t1_caaa, trdm_ca, optimize = einsum_type)
#            H_IJ += 1/2 * einsum('xa,ya,yx', h_ae, t1_ae, trdm_ca, optimize = einsum_type)
            H_IJ -= 1/6 * einsum('xa,yzwa,zywx', h_ae, t1_aaae, trdm_ccaa, optimize = einsum_type)
            H_IJ += 2/3 * einsum('xa,yzwa,zyxw', h_ae, t1_aaae, trdm_ccaa, optimize = einsum_type)
#            H_IJ += einsum('ia,iaxy,yx', t1_ce, v_ceaa, trdm_ca, optimize = einsum_type)
#            H_IJ -= 1/2 * einsum('ia,ixya,xy', t1_ce, v_caae, trdm_ca, optimize = einsum_type)
            H_IJ -= einsum('ijxa,iyja,yx', t1_ccae, v_cace, trdm_ca, optimize = einsum_type)
            H_IJ += 1/2 * einsum('ijxa,jyia,yx', t1_ccae, v_cace, trdm_ca, optimize = einsum_type)
            H_IJ -= einsum('ijxy,ixjz,zy', t1_ccaa, v_caca, trdm_ca, optimize = einsum_type)
            H_IJ += 1/2 * einsum('ijxy,iyjz,zx', t1_ccaa, v_caca, trdm_ca, optimize = einsum_type)
            H_IJ += 1/3 * einsum('ijxy,izjw,zwxy', t1_ccaa, v_caca, trdm_ccaa, optimize = einsum_type)
            H_IJ -= 1/12 * einsum('ijxy,izjw,zwyx', t1_ccaa, v_caca, trdm_ccaa, optimize = einsum_type)
#            H_IJ += einsum('ix,ixyz,zy', t1_ca, v_caaa, trdm_ca, optimize = einsum_type)
#            H_IJ -= 2/3 * einsum('ix,iyzw,ywxz', t1_ca, v_caaa, trdm_ccaa, optimize = einsum_type)
#            H_IJ += 1/6 * einsum('ix,iyzw,ywzx', t1_ca, v_caaa, trdm_ccaa, optimize = einsum_type)
#            H_IJ -= 1/2 * einsum('ix,iyzx,yz', t1_ca, v_caaa, trdm_ca, optimize = einsum_type)
            H_IJ += einsum('ixab,iayb,xy', t1_caee, v_ceae, trdm_ca, optimize = einsum_type)
            H_IJ -= 1/2 * einsum('ixab,ibya,xy', t1_caee, v_ceae, trdm_ca, optimize = einsum_type)
            H_IJ += 4/3 * einsum('ixay,iazw,xwyz', t1_caea, v_ceaa, trdm_ccaa, optimize = einsum_type)
            H_IJ -= 1/3 * einsum('ixay,iazw,xwzy', t1_caea, v_ceaa, trdm_ccaa, optimize = einsum_type)
            H_IJ += einsum('ixay,iazy,xz', t1_caea, v_ceaa, trdm_ca, optimize = einsum_type)
            H_IJ -= 1/2 * einsum('ixay,iyza,xz', t1_caea, v_caae, trdm_ca, optimize = einsum_type)
            H_IJ += 1/6 * einsum('ixay,izwa,xzwy', t1_caea, v_caae, trdm_ccaa, optimize = einsum_type)
            H_IJ -= 2/3 * einsum('ixay,izwa,xzyw', t1_caea, v_caae, trdm_ccaa, optimize = einsum_type)
            H_IJ -= 2/3 * einsum('ixya,iazw,xwyz', t1_caae, v_ceaa, trdm_ccaa, optimize = einsum_type)
            H_IJ += 1/6 * einsum('ixya,iazw,xwzy', t1_caae, v_ceaa, trdm_ccaa, optimize = einsum_type)
            H_IJ -= 1/2 * einsum('ixya,iazy,xz', t1_caae, v_ceaa, trdm_ca, optimize = einsum_type)
            H_IJ += einsum('ixya,iyza,xz', t1_caae, v_caae, trdm_ca, optimize = einsum_type)
            H_IJ -= 2/3 * einsum('ixya,izwa,xzwy', t1_caae, v_caae, trdm_ccaa, optimize = einsum_type)
            H_IJ += 1/6 * einsum('ixya,izwa,xzyw', t1_caae, v_caae, trdm_ccaa, optimize = einsum_type)
            H_IJ += 1/6 * einsum('ixyz,iwuv,xwvuyz', t1_caaa, v_caaa, trdm_cccaaa, optimize = einsum_type)
            H_IJ += 1/6 * einsum('ixyz,iwuv,xwvuzy', t1_caaa, v_caaa, trdm_cccaaa, optimize = einsum_type)
            H_IJ += 1/6 * einsum('ixyz,iwuv,xwvyuz', t1_caaa, v_caaa, trdm_cccaaa, optimize = einsum_type)
            H_IJ += 1/6 * einsum('ixyz,iwuv,xwvyzu', t1_caaa, v_caaa, trdm_cccaaa, optimize = einsum_type)
            H_IJ += 1/6 * einsum('ixyz,iwuv,xwvzuy', t1_caaa, v_caaa, trdm_cccaaa, optimize = einsum_type)
            H_IJ -= 1/3 * einsum('ixyz,iwuv,xwvzyu', t1_caaa, v_caaa, trdm_cccaaa, optimize = einsum_type)
            H_IJ += 1/6 * einsum('ixyz,iwuy,xwuz', t1_caaa, v_caaa, trdm_ccaa, optimize = einsum_type)
            H_IJ -= 2/3 * einsum('ixyz,iwuy,xwzu', t1_caaa, v_caaa, trdm_ccaa, optimize = einsum_type)
            H_IJ -= 2/3 * einsum('ixyz,iwuz,xwuy', t1_caaa, v_caaa, trdm_ccaa, optimize = einsum_type)
            H_IJ += 1/6 * einsum('ixyz,iwuz,xwyu', t1_caaa, v_caaa, trdm_ccaa, optimize = einsum_type)
            H_IJ -= 1/3 * einsum('ixyz,iywu,xuwz', t1_caaa, v_caaa, trdm_ccaa, optimize = einsum_type)
            H_IJ += 4/3 * einsum('ixyz,iywu,xuzw', t1_caaa, v_caaa, trdm_ccaa, optimize = einsum_type)
            H_IJ += einsum('ixyz,iywz,xw', t1_caaa, v_caaa, trdm_ca, optimize = einsum_type)
            H_IJ += 1/6 * einsum('ixyz,izwu,xuwy', t1_caaa, v_caaa, trdm_ccaa, optimize = einsum_type)
            H_IJ -= 2/3 * einsum('ixyz,izwu,xuyw', t1_caaa, v_caaa, trdm_ccaa, optimize = einsum_type)
            H_IJ -= 1/2 * einsum('ixyz,izwy,xw', t1_caaa, v_caaa, trdm_ca, optimize = einsum_type)
#            H_IJ += 2/3 * einsum('xa,yzwa,xzwy', t1_ae, v_aaae, trdm_ccaa, optimize = einsum_type)
#            H_IJ -= 1/6 * einsum('xa,yzwa,xzyw', t1_ae, v_aaae, trdm_ccaa, optimize = einsum_type)
            H_IJ -= 1/12 * einsum('xyab,zawb,xywz', t1_aaee, v_aeae, trdm_ccaa, optimize = einsum_type)
            H_IJ += 1/3 * einsum('xyab,zawb,xyzw', t1_aaee, v_aeae, trdm_ccaa, optimize = einsum_type)
            H_IJ -= 1/6 * einsum('xyza,wuva,yxuvwz', t1_aaae, v_aaae, trdm_cccaaa, optimize = einsum_type)
            H_IJ += 1/3 * einsum('xyza,wuva,yxuvzw', t1_aaae, v_aaae, trdm_cccaaa, optimize = einsum_type)
            H_IJ -= 1/6 * einsum('xyza,wuva,yxuwvz', t1_aaae, v_aaae, trdm_cccaaa, optimize = einsum_type)
            H_IJ -= 1/6 * einsum('xyza,wuva,yxuwzv', t1_aaae, v_aaae, trdm_cccaaa, optimize = einsum_type)
            H_IJ -= 1/6 * einsum('xyza,wuva,yxuzvw', t1_aaae, v_aaae, trdm_cccaaa, optimize = einsum_type)
            H_IJ -= 1/6 * einsum('xyza,wuva,yxuzwv', t1_aaae, v_aaae, trdm_cccaaa, optimize = einsum_type)
            H_IJ -= 1/6 * einsum('xyza,wzua,xyuw', t1_aaae, v_aaae, trdm_ccaa, optimize = einsum_type)
            H_IJ += 2/3 * einsum('xyza,wzua,xywu', t1_aaae, v_aaae, trdm_ccaa, optimize = einsum_type)

            # Generated assuming braket symmetry:

            # Generated assuming braket symmetry:
#####            t1_caea = t1[I].caea + t1[J].caea
#####            t1_caae = t1[I].caae + t1[J].caae
#####            t1_caaa = t1[I].caaa + t1[J].caaa
#####            t1_aaae = t1[I].aaae + t1[J].aaae
#####            t1_ccae = t1[I].ccae + t1[J].ccae
#####            t1_caee = t1[I].caee + t1[J].caee
#####            t1_ccaa = t1[I].ccaa + t1[J].ccaa
#####            t1_aaee = t1[I].aaee + t1[J].aaee
#####
######            H_IJ  = einsum('ia,ia', h_ce, t1_ce, optimize = einsum_type)
######            H_IJ += einsum('ix,ix', h_ca, t1_ca, optimize = einsum_type)
######            H_IJ += einsum('ijab,iajb', t1_ccee, v_cece, optimize = einsum_type)
######            H_IJ -= 1/2 * einsum('ijab,jaib', t1_ccee, v_cece, optimize = einsum_type)
######            H_IJ += 2 * einsum('ijxa,ixja', t1_ccae, v_cace, optimize = einsum_type)
######            H_IJ -= einsum('ijxa,jxia', t1_ccae, v_cace, optimize = einsum_type)
######            H_IJ += einsum('ijxy,ixjy', t1_ccaa, v_caca, optimize = einsum_type)
######            H_IJ -= 1/2 * einsum('ijxy,jxiy', t1_ccaa, v_caca, optimize = einsum_type)
#####            H_IJ = einsum('ia,ixay,yx', h_ce, t1_caea, rdm_ca, optimize = einsum_type)
#####            H_IJ -= 1/2 * einsum('ia,ixya,yx', h_ce, t1_caae, rdm_ca, optimize = einsum_type)
######            H_IJ -= 1/2 * einsum('ix,iy,xy', h_ca, t1_ca, rdm_ca, optimize = einsum_type)
#####            H_IJ += einsum('ix,iyxz,zy', h_ca, t1_caaa, rdm_ca, optimize = einsum_type)
#####            H_IJ -= 1/2 * einsum('ix,iyzw,xyzw', h_ca, t1_caaa, rdm_ccaa, optimize = einsum_type)
#####            H_IJ -= 1/2 * einsum('ix,iyzx,zy', h_ca, t1_caaa, rdm_ca, optimize = einsum_type)
######            H_IJ += 1/2 * einsum('xa,ya,xy', h_ae, t1_ae, rdm_ca, optimize = einsum_type)
#####            H_IJ += 1/2 * einsum('xa,yzwa,xwzy', h_ae, t1_aaae, rdm_ccaa, optimize = einsum_type)
######            H_IJ += einsum('ia,iaxy,yx', t1_ce, v_ceaa, rdm_ca, optimize = einsum_type)
######            H_IJ -= 1/2 * einsum('ia,ixya,xy', t1_ce, v_caae, rdm_ca, optimize = einsum_type)
#####            H_IJ -= einsum('ijxa,iyja,xy', t1_ccae, v_cace, rdm_ca, optimize = einsum_type)
#####            H_IJ += 1/2 * einsum('ijxa,jyia,xy', t1_ccae, v_cace, rdm_ca, optimize = einsum_type)
#####            H_IJ -= einsum('ijxy,ixjz,yz', t1_ccaa, v_caca, rdm_ca, optimize = einsum_type)
#####            H_IJ += 1/2 * einsum('ijxy,iyjz,xz', t1_ccaa, v_caca, rdm_ca, optimize = einsum_type)
#####            H_IJ += 1/4 * einsum('ijxy,izjw,xyzw', t1_ccaa, v_caca, rdm_ccaa, optimize = einsum_type)
######            H_IJ += einsum('ix,ixyz,zy', t1_ca, v_caaa, rdm_ca, optimize = einsum_type)
######            H_IJ -= 1/2 * einsum('ix,iyzw,xzyw', t1_ca, v_caaa, rdm_ccaa, optimize = einsum_type)
######            H_IJ -= 1/2 * einsum('ix,iyzx,yz', t1_ca, v_caaa, rdm_ca, optimize = einsum_type)
#####            H_IJ += einsum('ixab,iayb,xy', t1_caee, v_ceae, rdm_ca, optimize = einsum_type)
#####            H_IJ -= 1/2 * einsum('ixab,ibya,xy', t1_caee, v_ceae, rdm_ca, optimize = einsum_type)
#####            H_IJ += einsum('ixay,iazw,yzxw', t1_caea, v_ceaa, rdm_ccaa, optimize = einsum_type)
#####            H_IJ += einsum('ixay,iazy,xz', t1_caea, v_ceaa, rdm_ca, optimize = einsum_type)
#####            H_IJ -= 1/2 * einsum('ixay,iyza,xz', t1_caea, v_caae, rdm_ca, optimize = einsum_type)
#####            H_IJ -= 1/2 * einsum('ixay,izwa,ywxz', t1_caea, v_caae, rdm_ccaa, optimize = einsum_type)
#####            H_IJ -= 1/2 * einsum('ixya,iazw,yzxw', t1_caae, v_ceaa, rdm_ccaa, optimize = einsum_type)
#####            H_IJ -= 1/2 * einsum('ixya,iazy,xz', t1_caae, v_ceaa, rdm_ca, optimize = einsum_type)
#####            H_IJ += einsum('ixya,iyza,xz', t1_caae, v_caae, rdm_ca, optimize = einsum_type)
#####            H_IJ -= 1/2 * einsum('ixya,izwa,ywzx', t1_caae, v_caae, rdm_ccaa, optimize = einsum_type)
#####            H_IJ += 1/6 * einsum('ixyz,iwuv,xwvuyz', t1_caaa, v_caaa, rdm_cccaaa, optimize = einsum_type)
#####            H_IJ += 1/6 * einsum('ixyz,iwuv,xwvuzy', t1_caaa, v_caaa, rdm_cccaaa, optimize = einsum_type)
#####            H_IJ += 1/6 * einsum('ixyz,iwuv,xwvyuz', t1_caaa, v_caaa, rdm_cccaaa, optimize = einsum_type)
#####            H_IJ += 1/6 * einsum('ixyz,iwuv,xwvyzu', t1_caaa, v_caaa, rdm_cccaaa, optimize = einsum_type)
#####            H_IJ += 1/6 * einsum('ixyz,iwuv,xwvzuy', t1_caaa, v_caaa, rdm_cccaaa, optimize = einsum_type)
#####            H_IJ -= 1/3 * einsum('ixyz,iwuv,xwvzyu', t1_caaa, v_caaa, rdm_cccaaa, optimize = einsum_type)
#####            H_IJ -= 1/2 * einsum('ixyz,iwuy,zuxw', t1_caaa, v_caaa, rdm_ccaa, optimize = einsum_type)
#####            H_IJ -= 1/2 * einsum('ixyz,iwuz,yuwx', t1_caaa, v_caaa, rdm_ccaa, optimize = einsum_type)
#####            H_IJ += einsum('ixyz,iywu,zwxu', t1_caaa, v_caaa, rdm_ccaa, optimize = einsum_type)
#####            H_IJ += einsum('ixyz,iywz,xw', t1_caaa, v_caaa, rdm_ca, optimize = einsum_type)
#####            H_IJ -= 1/2 * einsum('ixyz,izwu,ywxu', t1_caaa, v_caaa, rdm_ccaa, optimize = einsum_type)
#####            H_IJ -= 1/2 * einsum('ixyz,izwy,xw', t1_caaa, v_caaa, rdm_ca, optimize = einsum_type)
######            H_IJ += 1/2 * einsum('xa,yzwa,xzwy', t1_ae, v_aaae, rdm_ccaa, optimize = einsum_type)
#####            H_IJ += 1/4 * einsum('xyab,zawb,xyzw', t1_aaee, v_aeae, rdm_ccaa, optimize = einsum_type)
#####            H_IJ -= 1/6 * einsum('xyza,wuva,zvwuxy', t1_aaae, v_aaae, rdm_cccaaa, optimize = einsum_type)
#####            H_IJ -= 1/6 * einsum('xyza,wuva,zvwuyx', t1_aaae, v_aaae, rdm_cccaaa, optimize = einsum_type)
#####            H_IJ -= 1/6 * einsum('xyza,wuva,zvwxuy', t1_aaae, v_aaae, rdm_cccaaa, optimize = einsum_type)
#####            H_IJ += 1/3 * einsum('xyza,wuva,zvwxyu', t1_aaae, v_aaae, rdm_cccaaa, optimize = einsum_type)
#####            H_IJ -= 1/6 * einsum('xyza,wuva,zvwyux', t1_aaae, v_aaae, rdm_cccaaa, optimize = einsum_type)
#####            H_IJ -= 1/6 * einsum('xyza,wuva,zvwyxu', t1_aaae, v_aaae, rdm_cccaaa, optimize = einsum_type)
#####            H_IJ += 1/2 * einsum('xyza,wzua,xywu', t1_aaae, v_aaae, rdm_ccaa, optimize = einsum_type)

            h_eff[I, J] += H_IJ
            h_eff[J, I] += H_IJ

    h_eval, h_evec = np.linalg.eigh(h_eff)

    print (h_eval)

    exit()

    return h_eval, h_evec


