# Copyright 2023 Prism Developers. All Rights Reserved.
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
#          Carlos E. V. de Moura <carlosevmoura@gmail.com>
#

import numpy as np

def compute_K_ac(mr_adc):

    # Einsum definition from kernel
    einsum = mr_adc.interface.einsum
    einsum_type = mr_adc.interface.einsum_type

    # Variables from kernel
    ## One-electron integrals
    h_aa = mr_adc.h1eff.aa

    ## Two-electron integrals
    v_aaaa = mr_adc.v2e.aaaa

    ## Reduced density matrices
    rdm_ca = mr_adc.rdm.ca
    rdm_ccaa = mr_adc.rdm.ccaa

    # Compute K_ac: < Psi_0 | a_X [H_{act}, a^{\dag}_Y] | Psi_0 >
    K_ac  = einsum('XY->XY', h_aa, optimize = einsum_type).copy()
    K_ac -= 1/2 * einsum('Yx,Xx->XY', h_aa, rdm_ca, optimize = einsum_type)
    K_ac += einsum('XYxy,xy->XY', v_aaaa, rdm_ca, optimize = einsum_type)
    K_ac -= 1/2 * einsum('XxyY,yx->XY', v_aaaa, rdm_ca, optimize = einsum_type)
    K_ac -= 1/2 * einsum('Yxyz,Xyxz->XY', v_aaaa, rdm_ccaa, optimize = einsum_type)

    return K_ac

def compute_K_ca(mr_adc):

    # Einsum definition from kernel
    einsum = mr_adc.interface.einsum
    einsum_type = mr_adc.interface.einsum_type

    # Variables from kernel
    ## One-electron integrals
    h_aa = mr_adc.h1eff.aa

    ## Two-electron integrals
    v_aaaa = mr_adc.v2e.aaaa

    ## Reduced density matrices
    rdm_ca = mr_adc.rdm.ca
    rdm_ccaa = mr_adc.rdm.ccaa

    # Compute K_ca: < Psi_0 | a^{\dag}_X [H_{act}, a_Y] | Psi_0 >
    K_ca =- 1/2 * einsum('Yx,Xx->XY', h_aa, rdm_ca, optimize = einsum_type)
    K_ca -= 1/2 * einsum('Yxyz,Xyxz->XY', v_aaaa, rdm_ccaa, optimize = einsum_type)

    return K_ca

def compute_K_aacc(mr_adc):

    # Einsum definition from kernel
    einsum = mr_adc.interface.einsum
    einsum_type = mr_adc.interface.einsum_type

    # Variables from kernel
    ncas = mr_adc.ncas

    ## One-electron integrals
    h_aa = mr_adc.h1eff.aa

    ## Two-electron integrals
    v_aaaa = mr_adc.v2e.aaaa

    ## Reduced density matrices
    rdm_ca = mr_adc.rdm.ca
    rdm_ccaa = mr_adc.rdm.ccaa
    rdm_cccaaa = mr_adc.rdm.cccaaa

    # Compute K_aacc: < Psi_0 | a_X a_Y [H_{act}, a^{\dag}_Z a^{\dag}_W] | Psi_0 >
    K_aacc  = einsum('WXZY->XYWZ', v_aaaa, optimize = einsum_type).copy()
    K_aacc += einsum('WX,YZ->XYWZ', h_aa, np.identity(ncas), optimize = einsum_type)
    K_aacc += einsum('YZ,WX->XYWZ', h_aa, np.identity(ncas), optimize = einsum_type)
    K_aacc -= 1/2 * einsum('WX,YZ->XYWZ', h_aa, rdm_ca, optimize = einsum_type)
    K_aacc += 1/6 * einsum('Wx,XYZx->XYWZ', h_aa, rdm_ccaa, optimize = einsum_type)
    K_aacc += 1/3 * einsum('Wx,XYxZ->XYWZ', h_aa, rdm_ccaa, optimize = einsum_type)
    K_aacc -= 1/2 * einsum('YZ,WX->XYWZ', h_aa, rdm_ca, optimize = einsum_type)
    K_aacc += 1/3 * einsum('Zx,WxXY->XYWZ', h_aa, rdm_ccaa, optimize = einsum_type)
    K_aacc += 1/6 * einsum('Zx,WxYX->XYWZ', h_aa, rdm_ccaa, optimize = einsum_type)
    K_aacc -= 1/2 * einsum('WXZx,Yx->XYWZ', v_aaaa, rdm_ca, optimize = einsum_type)
    K_aacc -= 1/2 * einsum('WXxY,Zx->XYWZ', v_aaaa, rdm_ca, optimize = einsum_type)
    K_aacc -= 1/2 * einsum('WXxy,YxZy->XYWZ', v_aaaa, rdm_ccaa, optimize = einsum_type)
    K_aacc -= 1/2 * einsum('WxZY,Xx->XYWZ', v_aaaa, rdm_ca, optimize = einsum_type)
    K_aacc += 1/3 * einsum('WxZy,XYxy->XYWZ', v_aaaa, rdm_ccaa, optimize = einsum_type)
    K_aacc += 1/6 * einsum('WxZy,XYyx->XYWZ', v_aaaa, rdm_ccaa, optimize = einsum_type)
    K_aacc += 1/3 * einsum('WxyX,YyZx->XYWZ', v_aaaa, rdm_ccaa, optimize = einsum_type)
    K_aacc += 1/6 * einsum('WxyX,YyxZ->XYWZ', v_aaaa, rdm_ccaa, optimize = einsum_type)
    K_aacc += 1/6 * einsum('WxyY,XyZx->XYWZ', v_aaaa, rdm_ccaa, optimize = einsum_type)
    K_aacc += 1/3 * einsum('WxyY,XyxZ->XYWZ', v_aaaa, rdm_ccaa, optimize = einsum_type)
    K_aacc += 1/24 * einsum('Wxyz,XYyZxz->XYWZ', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K_aacc -= 1/8 * einsum('Wxyz,XYyZzx->XYWZ', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K_aacc += 5/24 * einsum('Wxyz,XYyxZz->XYWZ', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K_aacc -= 1/8 * einsum('Wxyz,XYyxzZ->XYWZ', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K_aacc -= 1/8 * einsum('Wxyz,XYyzZx->XYWZ', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K_aacc -= 1/8 * einsum('Wxyz,XYyzxZ->XYWZ', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K_aacc -= 1/2 * einsum('XxYZ,Wx->XYWZ', v_aaaa, rdm_ca, optimize = einsum_type)
    K_aacc += 1/6 * einsum('XxyZ,WyYx->XYWZ', v_aaaa, rdm_ccaa, optimize = einsum_type)
    K_aacc += 1/3 * einsum('XxyZ,WyxY->XYWZ', v_aaaa, rdm_ccaa, optimize = einsum_type)
    K_aacc -= 1/2 * einsum('YZxy,WxXy->XYWZ', v_aaaa, rdm_ccaa, optimize = einsum_type)
    K_aacc += 1/3 * einsum('YxyZ,WyXx->XYWZ', v_aaaa, rdm_ccaa, optimize = einsum_type)
    K_aacc += 1/6 * einsum('YxyZ,WyxX->XYWZ', v_aaaa, rdm_ccaa, optimize = einsum_type)
    K_aacc += 5/24 * einsum('Zxyz,WxzXYy->XYWZ', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K_aacc -= 1/8 * einsum('Zxyz,WxzXyY->XYWZ', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K_aacc += 1/24 * einsum('Zxyz,WxzYXy->XYWZ', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K_aacc -= 1/8 * einsum('Zxyz,WxzYyX->XYWZ', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K_aacc -= 1/8 * einsum('Zxyz,WxzyXY->XYWZ', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K_aacc -= 1/8 * einsum('Zxyz,WxzyYX->XYWZ', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K_aacc -= 1/2 * einsum('Wx,YZ,Xx->XYWZ', h_aa, np.identity(ncas), rdm_ca, optimize = einsum_type)
    K_aacc -= 1/2 * einsum('Zx,WX,Yx->XYWZ', h_aa, np.identity(ncas), rdm_ca, optimize = einsum_type)
    K_aacc += einsum('WX,YZxy,xy->XYWZ', np.identity(ncas), v_aaaa, rdm_ca, optimize = einsum_type)
    K_aacc -= 1/2 * einsum('WX,YxyZ,yx->XYWZ', np.identity(ncas), v_aaaa, rdm_ca, optimize = einsum_type)
    K_aacc -= 1/2 * einsum('WX,Zxyz,Yyxz->XYWZ', np.identity(ncas), v_aaaa, rdm_ccaa, optimize = einsum_type)
    K_aacc += einsum('YZ,WXxy,xy->XYWZ', np.identity(ncas), v_aaaa, rdm_ca, optimize = einsum_type)
    K_aacc -= 1/2 * einsum('YZ,WxyX,yx->XYWZ', np.identity(ncas), v_aaaa, rdm_ca, optimize = einsum_type)
    K_aacc -= 1/2 * einsum('YZ,Wxyz,Xyxz->XYWZ', np.identity(ncas), v_aaaa, rdm_ccaa, optimize = einsum_type)
    K_aacc = K_aacc.reshape(ncas**2, ncas**2)

    return K_aacc

def compute_K_ccaa(mr_adc):

    # Einsum definition from kernel
    einsum = mr_adc.interface.einsum
    einsum_type = mr_adc.interface.einsum_type

    # Variables from kernel
    ncas = mr_adc.ncas

    ## One-electron integrals
    h_aa = mr_adc.h1eff.aa

    ## Two-electron integrals
    v_aaaa = mr_adc.v2e.aaaa

    ## Reduced density matrices
    rdm_ccaa = mr_adc.rdm.ccaa
    rdm_cccaaa = mr_adc.rdm.cccaaa

    # Compute K_ccaa: < Psi_0 | a^{\dag}_X a^{\dag}_Y [H_{act}, a_Z a_W] | Psi_0 >
    K_ccaa =- 1/6 * einsum('Wx,XYZx->XYWZ', h_aa, rdm_ccaa, optimize = einsum_type)
    K_ccaa -= 1/3 * einsum('Wx,XYxZ->XYWZ', h_aa, rdm_ccaa, optimize = einsum_type)
    K_ccaa -= 1/3 * einsum('Zx,WxXY->XYWZ', h_aa, rdm_ccaa, optimize = einsum_type)
    K_ccaa -= 1/6 * einsum('Zx,WxYX->XYWZ', h_aa, rdm_ccaa, optimize = einsum_type)
    K_ccaa -= 1/3 * einsum('WxZy,XYxy->XYWZ', v_aaaa, rdm_ccaa, optimize = einsum_type)
    K_ccaa -= 1/6 * einsum('WxZy,XYyx->XYWZ', v_aaaa, rdm_ccaa, optimize = einsum_type)
    K_ccaa -= 1/24 * einsum('Wxyz,XYyZxz->XYWZ', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K_ccaa += 1/8 * einsum('Wxyz,XYyZzx->XYWZ', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K_ccaa -= 5/24 * einsum('Wxyz,XYyxZz->XYWZ', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K_ccaa += 1/8 * einsum('Wxyz,XYyxzZ->XYWZ', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K_ccaa += 1/8 * einsum('Wxyz,XYyzZx->XYWZ', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K_ccaa += 1/8 * einsum('Wxyz,XYyzxZ->XYWZ', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K_ccaa -= 5/24 * einsum('Zxyz,WxzXYy->XYWZ', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K_ccaa += 1/8 * einsum('Zxyz,WxzXyY->XYWZ', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K_ccaa -= 1/24 * einsum('Zxyz,WxzYXy->XYWZ', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K_ccaa += 1/8 * einsum('Zxyz,WxzYyX->XYWZ', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K_ccaa += 1/8 * einsum('Zxyz,WxzyXY->XYWZ', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K_ccaa += 1/8 * einsum('Zxyz,WxzyYX->XYWZ', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K_ccaa = K_ccaa.reshape(ncas**2, ncas**2)

    return K_ccaa

def compute_K_caca(mr_adc):

    # Einsum definition from kernel
    einsum = mr_adc.interface.einsum
    einsum_type = mr_adc.interface.einsum_type

    # Variables from kernel
    ncas = mr_adc.ncas

    ## One-electron integrals
    h_aa = mr_adc.h1eff.aa

    ## Two-electron integrals
    v_aaaa = mr_adc.v2e.aaaa

    ## Reduced density matrices
    rdm_ca = mr_adc.rdm.ca
    rdm_ccaa = mr_adc.rdm.ccaa
    rdm_cccaaa = mr_adc.rdm.cccaaa

    # Compute K_caca: < Psi_0 | a^{\dag}_X a_Y [H_{act}, a^{\dag}_Z a_W] | Psi_0 >
    K_caca_aa_aa =- 1/6 * einsum('Wx,XZYx->XYWZ', h_aa, rdm_ccaa, optimize = einsum_type)
    K_caca_aa_aa += 1/6 * einsum('Wx,XZxY->XYWZ', h_aa, rdm_ccaa, optimize = einsum_type)
    K_caca_aa_aa += 1/2 * einsum('YZ,WX->XYWZ', h_aa, rdm_ca, optimize = einsum_type)
    K_caca_aa_aa -= 1/6 * einsum('Zx,WYXx->XYWZ', h_aa, rdm_ccaa, optimize = einsum_type)
    K_caca_aa_aa += 1/6 * einsum('Zx,WYxX->XYWZ', h_aa, rdm_ccaa, optimize = einsum_type)
    K_caca_aa_aa += 1/6 * einsum('WxYy,XZxy->XYWZ', v_aaaa, rdm_ccaa, optimize = einsum_type)
    K_caca_aa_aa -= 1/6 * einsum('WxYy,XZyx->XYWZ', v_aaaa, rdm_ccaa, optimize = einsum_type)
    K_caca_aa_aa -= 5/24 * einsum('Wxyz,XZyYxz->XYWZ', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K_caca_aa_aa -= 1/24 * einsum('Wxyz,XZyYzx->XYWZ', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K_caca_aa_aa += 1/8 * einsum('Wxyz,XZyxYz->XYWZ', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K_caca_aa_aa -= 1/24 * einsum('Wxyz,XZyxzY->XYWZ', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K_caca_aa_aa -= 1/24 * einsum('Wxyz,XZyzYx->XYWZ', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K_caca_aa_aa -= 1/24 * einsum('Wxyz,XZyzxY->XYWZ', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K_caca_aa_aa += 1/2 * einsum('YZxy,WyXx->XYWZ', v_aaaa, rdm_ccaa, optimize = einsum_type)
    K_caca_aa_aa -= 1/6 * einsum('YxyZ,WxXy->XYWZ', v_aaaa, rdm_ccaa, optimize = einsum_type)
    K_caca_aa_aa += 1/6 * einsum('YxyZ,WxyX->XYWZ', v_aaaa, rdm_ccaa, optimize = einsum_type)
    K_caca_aa_aa -= 5/24 * einsum('Zxyz,WYyXxz->XYWZ', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K_caca_aa_aa -= 1/24 * einsum('Zxyz,WYyXzx->XYWZ', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K_caca_aa_aa += 1/8 * einsum('Zxyz,WYyxXz->XYWZ', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K_caca_aa_aa -= 1/24 * einsum('Zxyz,WYyxzX->XYWZ', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K_caca_aa_aa -= 1/24 * einsum('Zxyz,WYyzXx->XYWZ', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K_caca_aa_aa -= 1/24 * einsum('Zxyz,WYyzxX->XYWZ', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K_caca_aa_aa -= 1/2 * einsum('Wx,YZ,Xx->XYWZ', h_aa, np.identity(ncas), rdm_ca, optimize = einsum_type)
    K_caca_aa_aa -= 1/2 * einsum('YZ,Wxyz,Xyxz->XYWZ', np.identity(ncas), v_aaaa, rdm_ccaa, optimize = einsum_type)

    K_caca_aa_bb =- 1/3 * einsum('Wx,XZYx->XYWZ', h_aa, rdm_ccaa, optimize = einsum_type)
    K_caca_aa_bb -= 1/6 * einsum('Wx,XZxY->XYWZ', h_aa, rdm_ccaa, optimize = einsum_type)
    K_caca_aa_bb += 1/6 * einsum('Zx,WYXx->XYWZ', h_aa, rdm_ccaa, optimize = einsum_type)
    K_caca_aa_bb += 1/3 * einsum('Zx,WYxX->XYWZ', h_aa, rdm_ccaa, optimize = einsum_type)
    K_caca_aa_bb -= 1/6 * einsum('WxYy,XZxy->XYWZ', v_aaaa, rdm_ccaa, optimize = einsum_type)
    K_caca_aa_bb -= 1/3 * einsum('WxYy,XZyx->XYWZ', v_aaaa, rdm_ccaa, optimize = einsum_type)
    K_caca_aa_bb -= 5/24 * einsum('Wxyz,XZyYxz->XYWZ', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K_caca_aa_bb += 1/8 * einsum('Wxyz,XZyYzx->XYWZ', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K_caca_aa_bb -= 1/24 * einsum('Wxyz,XZyxYz->XYWZ', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K_caca_aa_bb += 1/8 * einsum('Wxyz,XZyxzY->XYWZ', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K_caca_aa_bb += 1/8 * einsum('Wxyz,XZyzYx->XYWZ', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K_caca_aa_bb += 1/8 * einsum('Wxyz,XZyzxY->XYWZ', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K_caca_aa_bb += 1/6 * einsum('YxyZ,WxXy->XYWZ', v_aaaa, rdm_ccaa, optimize = einsum_type)
    K_caca_aa_bb += 1/3 * einsum('YxyZ,WxyX->XYWZ', v_aaaa, rdm_ccaa, optimize = einsum_type)
    K_caca_aa_bb += 1/8 * einsum('Zxyz,WYyXxz->XYWZ', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K_caca_aa_bb -= 1/24 * einsum('Zxyz,WYyXzx->XYWZ', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K_caca_aa_bb += 7/24 * einsum('Zxyz,WYyxXz->XYWZ', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K_caca_aa_bb -= 1/24 * einsum('Zxyz,WYyxzX->XYWZ', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K_caca_aa_bb -= 1/24 * einsum('Zxyz,WYyzXx->XYWZ', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K_caca_aa_bb -= 1/24 * einsum('Zxyz,WYyzxX->XYWZ', v_aaaa, rdm_cccaaa, optimize = einsum_type)

    ## Reshape tensors to matrix form
    dim_wz = ncas * ncas
    dim_caca = 2 * dim_wz

    K_caca = np.zeros((dim_caca, dim_caca))

    # Building K_caca matrix
    s_aa = 0
    f_aa = s_aa + dim_wz
    s_bb = f_aa
    f_bb = s_bb + dim_wz

    K_caca_aa_aa = K_caca_aa_aa.reshape(dim_wz, dim_wz)
    K_caca_aa_bb = K_caca_aa_bb.reshape(dim_wz, dim_wz)

    K_caca[s_aa:f_aa, s_aa:f_aa] = K_caca_aa_aa
    K_caca[s_bb:f_bb, s_bb:f_bb] = K_caca_aa_aa

    K_caca[s_aa:f_aa, s_bb:f_bb] = K_caca_aa_bb
    K_caca[s_bb:f_bb, s_aa:f_aa] = K_caca_aa_bb

    return K_caca

def compute_K_p1p(mr_adc):

    # Einsum definition from kernel
    einsum = mr_adc.interface.einsum
    einsum_type = mr_adc.interface.einsum_type

    # Variables from kernel
    ncas = mr_adc.ncas

    ## One-electron integrals
    h_aa = mr_adc.h1eff.aa

    ## Two-electron integrals
    v_aaaa = mr_adc.v2e.aaaa

    ## Reduced density matrices
    rdm_ca = mr_adc.rdm.ca
    rdm_ccaa = mr_adc.rdm.ccaa
    rdm_cccaaa = mr_adc.rdm.cccaaa
    rdm_ccccaaaa = mr_adc.rdm.ccccaaaa

    # Computing K11
    # K11 block: < Psi_0 | a_X [H_{act}, a^{\dag}_Y] | Psi_0 >
    K11_a_a  = einsum('XY->XY', h_aa, optimize = einsum_type).copy()
    K11_a_a -= 1/2 * einsum('Yx,Xx->XY', h_aa, rdm_ca, optimize = einsum_type)
    K11_a_a += einsum('XYxy,xy->XY', v_aaaa, rdm_ca, optimize = einsum_type)
    K11_a_a -= 1/2 * einsum('XxyY,yx->XY', v_aaaa, rdm_ca, optimize = einsum_type)
    K11_a_a -= 1/2 * einsum('Yxyz,Xyxz->XY', v_aaaa, rdm_ccaa, optimize = einsum_type)

    # K12 block: < Psi_0 | a_X [H_{act}, a^{\dag}_Y a^{\dag}_Z a_W] | Psi_0 >
    K12_a_bba  = 1/3 * einsum('Wx,XxYZ->XWZY', h_aa, rdm_ccaa, optimize = einsum_type)
    K12_a_bba += 1/6 * einsum('Wx,XxZY->XWZY', h_aa, rdm_ccaa, optimize = einsum_type)
    K12_a_bba += 1/2 * einsum('XY,WZ->XWZY', h_aa, rdm_ca, optimize = einsum_type)
    K12_a_bba -= 1/3 * einsum('Yx,WXZx->XWZY', h_aa, rdm_ccaa, optimize = einsum_type)
    K12_a_bba -= 1/6 * einsum('Yx,WXxZ->XWZY', h_aa, rdm_ccaa, optimize = einsum_type)
    K12_a_bba -= 1/6 * einsum('Zx,WXYx->XWZY', h_aa, rdm_ccaa, optimize = einsum_type)
    K12_a_bba -= 1/3 * einsum('Zx,WXxY->XWZY', h_aa, rdm_ccaa, optimize = einsum_type)
    K12_a_bba += 1/6 * einsum('WxXy,YZxy->XWZY', v_aaaa, rdm_ccaa, optimize = einsum_type)
    K12_a_bba += 1/3 * einsum('WxXy,YZyx->XWZY', v_aaaa, rdm_ccaa, optimize = einsum_type)
    K12_a_bba += 5/24 * einsum('Wxyz,XxzYZy->XWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K12_a_bba -= 1/8 * einsum('Wxyz,XxzYyZ->XWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K12_a_bba += 1/24 * einsum('Wxyz,XxzZYy->XWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K12_a_bba -= 1/8 * einsum('Wxyz,XxzZyY->XWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K12_a_bba -= 1/8 * einsum('Wxyz,XxzyYZ->XWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K12_a_bba -= 1/8 * einsum('Wxyz,XxzyZY->XWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K12_a_bba += 1/2 * einsum('XYxZ,Wx->XWZY', v_aaaa, rdm_ca, optimize = einsum_type)
    K12_a_bba += 1/2 * einsum('XYxy,WyZx->XWZY', v_aaaa, rdm_ccaa, optimize = einsum_type)
    K12_a_bba -= 1/3 * einsum('XxyY,WxZy->XWZY', v_aaaa, rdm_ccaa, optimize = einsum_type)
    K12_a_bba -= 1/6 * einsum('XxyY,WxyZ->XWZY', v_aaaa, rdm_ccaa, optimize = einsum_type)
    K12_a_bba -= 1/6 * einsum('XxyZ,WxYy->XWZY', v_aaaa, rdm_ccaa, optimize = einsum_type)
    K12_a_bba -= 1/3 * einsum('XxyZ,WxyY->XWZY', v_aaaa, rdm_ccaa, optimize = einsum_type)
    K12_a_bba -= 1/6 * einsum('YxZy,WXxy->XWZY', v_aaaa, rdm_ccaa, optimize = einsum_type)
    K12_a_bba -= 1/3 * einsum('YxZy,WXyx->XWZY', v_aaaa, rdm_ccaa, optimize = einsum_type)
    K12_a_bba -= 7/24 * einsum('Yxyz,WXyZxz->XWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K12_a_bba += 1/24 * einsum('Yxyz,WXyZzx->XWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K12_a_bba -= 1/8 * einsum('Yxyz,WXyxZz->XWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K12_a_bba += 1/24 * einsum('Yxyz,WXyxzZ->XWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K12_a_bba += 1/24 * einsum('Yxyz,WXyzZx->XWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K12_a_bba += 1/24 * einsum('Yxyz,WXyzxZ->XWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K12_a_bba -= 1/8 * einsum('Zxyz,WXyYxz->XWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K12_a_bba += 1/24 * einsum('Zxyz,WXyYzx->XWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K12_a_bba -= 7/24 * einsum('Zxyz,WXyxYz->XWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K12_a_bba += 1/24 * einsum('Zxyz,WXyxzY->XWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K12_a_bba += 1/24 * einsum('Zxyz,WXyzYx->XWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K12_a_bba += 1/24 * einsum('Zxyz,WXyzxY->XWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K12_a_bba -= 1/2 * einsum('Wx,XY,Zx->XWZY', h_aa, np.identity(ncas), rdm_ca, optimize = einsum_type)
    K12_a_bba += 1/2 * einsum('Zx,XY,Wx->XWZY', h_aa, np.identity(ncas), rdm_ca, optimize = einsum_type)
    K12_a_bba -= 1/2 * einsum('XY,Wxyz,Zyxz->XWZY', np.identity(ncas), v_aaaa, rdm_ccaa, optimize = einsum_type)
    K12_a_bba += 1/2 * einsum('XY,Zxyz,Wyxz->XWZY', np.identity(ncas), v_aaaa, rdm_ccaa, optimize = einsum_type)

    K12_a_aaa = np.ascontiguousarray(K12_a_bba - K12_a_bba.transpose(0,1,3,2))

    # K22 block: < Psi_0 | a^{\dag}_U a_V a_X [H_{act}, a^{\dag}_Y a^{\dag}_Z a_W] | Psi_0 >
    K22_aaa_aaa  = 1/6 * einsum('VY,UZWX->UVXWZY', h_aa, rdm_ccaa, optimize = einsum_type)
    K22_aaa_aaa -= 1/6 * einsum('VY,UZXW->UVXWZY', h_aa, rdm_ccaa, optimize = einsum_type)
    K22_aaa_aaa -= 1/6 * einsum('VZ,UYWX->UVXWZY', h_aa, rdm_ccaa, optimize = einsum_type)
    K22_aaa_aaa += 1/6 * einsum('VZ,UYXW->UVXWZY', h_aa, rdm_ccaa, optimize = einsum_type)
    K22_aaa_aaa -= 1/12 * einsum('Wx,UYZVxX->UVXWZY', h_aa, rdm_cccaaa, optimize = einsum_type)
    K22_aaa_aaa -= 1/12 * einsum('Wx,UYZXVx->UVXWZY', h_aa, rdm_cccaaa, optimize = einsum_type)
    K22_aaa_aaa -= 1/12 * einsum('Wx,UYZxXV->UVXWZY', h_aa, rdm_cccaaa, optimize = einsum_type)
    K22_aaa_aaa += 1/6 * einsum('XY,UZVW->UVXWZY', h_aa, rdm_ccaa, optimize = einsum_type)
    K22_aaa_aaa -= 1/6 * einsum('XY,UZWV->UVXWZY', h_aa, rdm_ccaa, optimize = einsum_type)
    K22_aaa_aaa -= 1/6 * einsum('XZ,UYVW->UVXWZY', h_aa, rdm_ccaa, optimize = einsum_type)
    K22_aaa_aaa += 1/6 * einsum('XZ,UYWV->UVXWZY', h_aa, rdm_ccaa, optimize = einsum_type)
    K22_aaa_aaa += 1/12 * einsum('Yx,UZxVXW->UVXWZY', h_aa, rdm_cccaaa, optimize = einsum_type)
    K22_aaa_aaa += 1/12 * einsum('Yx,UZxWVX->UVXWZY', h_aa, rdm_cccaaa, optimize = einsum_type)
    K22_aaa_aaa += 1/12 * einsum('Yx,UZxXWV->UVXWZY', h_aa, rdm_cccaaa, optimize = einsum_type)
    K22_aaa_aaa -= 1/12 * einsum('Zx,UYxVXW->UVXWZY', h_aa, rdm_cccaaa, optimize = einsum_type)
    K22_aaa_aaa -= 1/12 * einsum('Zx,UYxWVX->UVXWZY', h_aa, rdm_cccaaa, optimize = einsum_type)
    K22_aaa_aaa -= 1/12 * einsum('Zx,UYxXWV->UVXWZY', h_aa, rdm_cccaaa, optimize = einsum_type)
    K22_aaa_aaa -= 1/2 * einsum('VYXZ,UW->UVXWZY', v_aaaa, rdm_ca, optimize = einsum_type)
    K22_aaa_aaa += 1/6 * einsum('VYXx,UZWx->UVXWZY', v_aaaa, rdm_ccaa, optimize = einsum_type)
    K22_aaa_aaa -= 1/6 * einsum('VYXx,UZxW->UVXWZY', v_aaaa, rdm_ccaa, optimize = einsum_type)
    K22_aaa_aaa += 1/6 * einsum('VYxZ,UxWX->UVXWZY', v_aaaa, rdm_ccaa, optimize = einsum_type)
    K22_aaa_aaa -= 1/6 * einsum('VYxZ,UxXW->UVXWZY', v_aaaa, rdm_ccaa, optimize = einsum_type)
    K22_aaa_aaa += 1/6 * einsum('VYxy,UZxWXy->UVXWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_aaa_aaa -= 1/6 * einsum('VYxy,UZxXWy->UVXWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_aaa_aaa += 1/2 * einsum('VZXY,UW->UVXWZY', v_aaaa, rdm_ca, optimize = einsum_type)
    K22_aaa_aaa -= 1/6 * einsum('VZXx,UYWx->UVXWZY', v_aaaa, rdm_ccaa, optimize = einsum_type)
    K22_aaa_aaa += 1/6 * einsum('VZXx,UYxW->UVXWZY', v_aaaa, rdm_ccaa, optimize = einsum_type)
    K22_aaa_aaa -= 1/6 * einsum('VZxY,UxWX->UVXWZY', v_aaaa, rdm_ccaa, optimize = einsum_type)
    K22_aaa_aaa += 1/6 * einsum('VZxY,UxXW->UVXWZY', v_aaaa, rdm_ccaa, optimize = einsum_type)
    K22_aaa_aaa -= 1/6 * einsum('VZxy,UYxWXy->UVXWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_aaa_aaa += 1/6 * einsum('VZxy,UYxXWy->UVXWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_aaa_aaa -= 1/24 * einsum('VxWy,UYZXxy->UVXWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_aaa_aaa += 1/24 * einsum('VxWy,UYZXyx->UVXWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_aaa_aaa += 1/24 * einsum('VxWy,UYZxXy->UVXWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_aaa_aaa -= 1/24 * einsum('VxWy,UYZxyX->UVXWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_aaa_aaa -= 1/24 * einsum('VxWy,UYZyXx->UVXWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_aaa_aaa += 1/24 * einsum('VxWy,UYZyxX->UVXWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_aaa_aaa -= 1/6 * einsum('VxXY,UZWx->UVXWZY', v_aaaa, rdm_ccaa, optimize = einsum_type)
    K22_aaa_aaa += 1/6 * einsum('VxXY,UZxW->UVXWZY', v_aaaa, rdm_ccaa, optimize = einsum_type)
    K22_aaa_aaa += 1/6 * einsum('VxXZ,UYWx->UVXWZY', v_aaaa, rdm_ccaa, optimize = einsum_type)
    K22_aaa_aaa -= 1/6 * einsum('VxXZ,UYxW->UVXWZY', v_aaaa, rdm_ccaa, optimize = einsum_type)
    K22_aaa_aaa += 1/12 * einsum('VxyY,UZyWxX->UVXWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_aaa_aaa += 1/12 * einsum('VxyY,UZyXWx->UVXWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_aaa_aaa += 1/12 * einsum('VxyY,UZyxXW->UVXWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_aaa_aaa -= 1/12 * einsum('VxyZ,UYyWxX->UVXWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_aaa_aaa -= 1/12 * einsum('VxyZ,UYyXWx->UVXWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_aaa_aaa -= 1/12 * einsum('VxyZ,UYyxXW->UVXWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_aaa_aaa -= 1/24 * einsum('WxXy,UYZVxy->UVXWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_aaa_aaa += 1/24 * einsum('WxXy,UYZVyx->UVXWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_aaa_aaa += 1/24 * einsum('WxXy,UYZxVy->UVXWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_aaa_aaa -= 1/24 * einsum('WxXy,UYZxyV->UVXWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_aaa_aaa -= 1/24 * einsum('WxXy,UYZyVx->UVXWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_aaa_aaa += 1/24 * einsum('WxXy,UYZyxV->UVXWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_aaa_aaa += 3/40 * einsum('Wxyz,UYZyVXxz->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aaa_aaa -= 1/40 * einsum('Wxyz,UYZyVXzx->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aaa_aaa += 1/120 * einsum('Wxyz,UYZyVxXz->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aaa_aaa -= 1/120 * einsum('Wxyz,UYZyVxzX->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aaa_aaa += 1/120 * einsum('Wxyz,UYZyVzXx->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aaa_aaa += 1/120 * einsum('Wxyz,UYZyVzxX->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aaa_aaa += 1/120 * einsum('Wxyz,UYZyXVxz->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aaa_aaa -= 1/120 * einsum('Wxyz,UYZyXVzx->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aaa_aaa += 3/40 * einsum('Wxyz,UYZyXxVz->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aaa_aaa -= 1/40 * einsum('Wxyz,UYZyXxzV->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aaa_aaa -= 1/120 * einsum('Wxyz,UYZyXzVx->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aaa_aaa -= 1/120 * einsum('Wxyz,UYZyXzxV->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aaa_aaa += 1/12 * einsum('Wxyz,UYZyxVXz->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aaa_aaa -= 1/60 * einsum('Wxyz,UYZyxVzX->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aaa_aaa -= 1/30 * einsum('Wxyz,UYZyxXVz->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aaa_aaa -= 1/20 * einsum('Wxyz,UYZyxXzV->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aaa_aaa -= 1/60 * einsum('Wxyz,UYZyxzVX->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aaa_aaa -= 1/60 * einsum('Wxyz,UYZyxzXV->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aaa_aaa -= 1/60 * einsum('Wxyz,UYZyzVXx->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aaa_aaa -= 1/60 * einsum('Wxyz,UYZyzVxX->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aaa_aaa -= 1/20 * einsum('Wxyz,UYZyzXVx->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aaa_aaa -= 1/20 * einsum('Wxyz,UYZyzXxV->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aaa_aaa -= 1/30 * einsum('Wxyz,UYZyzxVX->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aaa_aaa -= 1/30 * einsum('Wxyz,UYZyzxXV->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aaa_aaa += 1/6 * einsum('XYxZ,UxVW->UVXWZY', v_aaaa, rdm_ccaa, optimize = einsum_type)
    K22_aaa_aaa -= 1/6 * einsum('XYxZ,UxWV->UVXWZY', v_aaaa, rdm_ccaa, optimize = einsum_type)
    K22_aaa_aaa += 1/6 * einsum('XYxy,UZxVWy->UVXWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_aaa_aaa -= 1/6 * einsum('XYxy,UZxWVy->UVXWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_aaa_aaa -= 1/6 * einsum('XZxY,UxVW->UVXWZY', v_aaaa, rdm_ccaa, optimize = einsum_type)
    K22_aaa_aaa += 1/6 * einsum('XZxY,UxWV->UVXWZY', v_aaaa, rdm_ccaa, optimize = einsum_type)
    K22_aaa_aaa -= 1/6 * einsum('XZxy,UYxVWy->UVXWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_aaa_aaa += 1/6 * einsum('XZxy,UYxWVy->UVXWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_aaa_aaa += 1/12 * einsum('XxyY,UZyVxW->UVXWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_aaa_aaa += 1/12 * einsum('XxyY,UZyWVx->UVXWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_aaa_aaa += 1/12 * einsum('XxyY,UZyxWV->UVXWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_aaa_aaa -= 1/12 * einsum('XxyZ,UYyVxW->UVXWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_aaa_aaa -= 1/12 * einsum('XxyZ,UYyWVx->UVXWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_aaa_aaa -= 1/12 * einsum('XxyZ,UYyxWV->UVXWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_aaa_aaa += 1/24 * einsum('YxZy,UxyVWX->UVXWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_aaa_aaa -= 1/24 * einsum('YxZy,UxyVXW->UVXWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_aaa_aaa -= 1/24 * einsum('YxZy,UxyWVX->UVXWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_aaa_aaa += 1/24 * einsum('YxZy,UxyWXV->UVXWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_aaa_aaa += 1/24 * einsum('YxZy,UxyXVW->UVXWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_aaa_aaa -= 1/24 * einsum('YxZy,UxyXWV->UVXWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_aaa_aaa -= 3/40 * einsum('Yxyz,UZxzVWXy->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aaa_aaa += 1/40 * einsum('Yxyz,UZxzVWyX->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aaa_aaa -= 1/120 * einsum('Yxyz,UZxzVXWy->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aaa_aaa -= 1/120 * einsum('Yxyz,UZxzVXyW->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aaa_aaa += 1/120 * einsum('Yxyz,UZxzVyWX->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aaa_aaa -= 1/120 * einsum('Yxyz,UZxzVyXW->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aaa_aaa -= 1/120 * einsum('Yxyz,UZxzWVXy->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aaa_aaa += 1/120 * einsum('Yxyz,UZxzWVyX->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aaa_aaa -= 1/12 * einsum('Yxyz,UZxzWXVy->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aaa_aaa += 1/60 * einsum('Yxyz,UZxzWXyV->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aaa_aaa += 1/60 * einsum('Yxyz,UZxzWyVX->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aaa_aaa += 1/60 * einsum('Yxyz,UZxzWyXV->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aaa_aaa -= 3/40 * einsum('Yxyz,UZxzXVWy->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aaa_aaa += 1/120 * einsum('Yxyz,UZxzXVyW->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aaa_aaa += 1/30 * einsum('Yxyz,UZxzXWVy->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aaa_aaa += 1/20 * einsum('Yxyz,UZxzXWyV->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aaa_aaa += 1/60 * einsum('Yxyz,UZxzXyVW->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aaa_aaa += 1/30 * einsum('Yxyz,UZxzXyWV->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aaa_aaa += 1/40 * einsum('Yxyz,UZxzyVWX->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aaa_aaa += 1/120 * einsum('Yxyz,UZxzyVXW->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aaa_aaa += 1/20 * einsum('Yxyz,UZxzyWVX->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aaa_aaa += 1/20 * einsum('Yxyz,UZxzyWXV->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aaa_aaa += 1/60 * einsum('Yxyz,UZxzyXVW->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aaa_aaa += 1/30 * einsum('Yxyz,UZxzyXWV->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aaa_aaa += 3/40 * einsum('Zxyz,UYxzVWXy->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aaa_aaa -= 1/40 * einsum('Zxyz,UYxzVWyX->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aaa_aaa += 1/120 * einsum('Zxyz,UYxzVXWy->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aaa_aaa += 1/120 * einsum('Zxyz,UYxzVXyW->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aaa_aaa -= 1/120 * einsum('Zxyz,UYxzVyWX->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aaa_aaa += 1/120 * einsum('Zxyz,UYxzVyXW->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aaa_aaa += 1/120 * einsum('Zxyz,UYxzWVXy->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aaa_aaa -= 1/120 * einsum('Zxyz,UYxzWVyX->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aaa_aaa += 1/12 * einsum('Zxyz,UYxzWXVy->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aaa_aaa -= 1/60 * einsum('Zxyz,UYxzWXyV->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aaa_aaa -= 1/60 * einsum('Zxyz,UYxzWyVX->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aaa_aaa -= 1/60 * einsum('Zxyz,UYxzWyXV->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aaa_aaa += 3/40 * einsum('Zxyz,UYxzXVWy->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aaa_aaa -= 1/120 * einsum('Zxyz,UYxzXVyW->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aaa_aaa -= 1/30 * einsum('Zxyz,UYxzXWVy->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aaa_aaa -= 1/20 * einsum('Zxyz,UYxzXWyV->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aaa_aaa -= 1/60 * einsum('Zxyz,UYxzXyVW->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aaa_aaa -= 1/30 * einsum('Zxyz,UYxzXyWV->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aaa_aaa -= 1/40 * einsum('Zxyz,UYxzyVWX->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aaa_aaa -= 1/120 * einsum('Zxyz,UYxzyVXW->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aaa_aaa -= 1/20 * einsum('Zxyz,UYxzyWVX->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aaa_aaa -= 1/20 * einsum('Zxyz,UYxzyWXV->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aaa_aaa -= 1/60 * einsum('Zxyz,UYxzyXVW->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aaa_aaa -= 1/30 * einsum('Zxyz,UYxzyXWV->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aaa_aaa -= 1/2 * einsum('VY,XZ,UW->UVXWZY', h_aa, np.identity(ncas), rdm_ca, optimize = einsum_type)
    K22_aaa_aaa += 1/2 * einsum('VZ,XY,UW->UVXWZY', h_aa, np.identity(ncas), rdm_ca, optimize = einsum_type)
    K22_aaa_aaa += 1/6 * einsum('Wx,VY,UZXx->UVXWZY', h_aa, np.identity(ncas), rdm_ccaa, optimize = einsum_type)
    K22_aaa_aaa -= 1/6 * einsum('Wx,VY,UZxX->UVXWZY', h_aa, np.identity(ncas), rdm_ccaa, optimize = einsum_type)
    K22_aaa_aaa -= 1/6 * einsum('Wx,VZ,UYXx->UVXWZY', h_aa, np.identity(ncas), rdm_ccaa, optimize = einsum_type)
    K22_aaa_aaa += 1/6 * einsum('Wx,VZ,UYxX->UVXWZY', h_aa, np.identity(ncas), rdm_ccaa, optimize = einsum_type)
    K22_aaa_aaa -= 1/6 * einsum('Wx,XY,UZVx->UVXWZY', h_aa, np.identity(ncas), rdm_ccaa, optimize = einsum_type)
    K22_aaa_aaa += 1/6 * einsum('Wx,XY,UZxV->UVXWZY', h_aa, np.identity(ncas), rdm_ccaa, optimize = einsum_type)
    K22_aaa_aaa += 1/6 * einsum('Wx,XZ,UYVx->UVXWZY', h_aa, np.identity(ncas), rdm_ccaa, optimize = einsum_type)
    K22_aaa_aaa -= 1/6 * einsum('Wx,XZ,UYxV->UVXWZY', h_aa, np.identity(ncas), rdm_ccaa, optimize = einsum_type)
    K22_aaa_aaa += 1/2 * einsum('XY,VZ,UW->UVXWZY', h_aa, np.identity(ncas), rdm_ca, optimize = einsum_type)
    K22_aaa_aaa -= 1/2 * einsum('XZ,VY,UW->UVXWZY', h_aa, np.identity(ncas), rdm_ca, optimize = einsum_type)
    K22_aaa_aaa -= 1/6 * einsum('Yx,VZ,UxWX->UVXWZY', h_aa, np.identity(ncas), rdm_ccaa, optimize = einsum_type)
    K22_aaa_aaa += 1/6 * einsum('Yx,VZ,UxXW->UVXWZY', h_aa, np.identity(ncas), rdm_ccaa, optimize = einsum_type)
    K22_aaa_aaa -= 1/6 * einsum('Yx,XZ,UxVW->UVXWZY', h_aa, np.identity(ncas), rdm_ccaa, optimize = einsum_type)
    K22_aaa_aaa += 1/6 * einsum('Yx,XZ,UxWV->UVXWZY', h_aa, np.identity(ncas), rdm_ccaa, optimize = einsum_type)
    K22_aaa_aaa += 1/6 * einsum('Zx,VY,UxWX->UVXWZY', h_aa, np.identity(ncas), rdm_ccaa, optimize = einsum_type)
    K22_aaa_aaa -= 1/6 * einsum('Zx,VY,UxXW->UVXWZY', h_aa, np.identity(ncas), rdm_ccaa, optimize = einsum_type)
    K22_aaa_aaa += 1/6 * einsum('Zx,XY,UxVW->UVXWZY', h_aa, np.identity(ncas), rdm_ccaa, optimize = einsum_type)
    K22_aaa_aaa -= 1/6 * einsum('Zx,XY,UxWV->UVXWZY', h_aa, np.identity(ncas), rdm_ccaa, optimize = einsum_type)
    K22_aaa_aaa -= 1/6 * einsum('VY,WxXy,UZxy->UVXWZY', np.identity(ncas), v_aaaa, rdm_ccaa, optimize = einsum_type)
    K22_aaa_aaa += 1/6 * einsum('VY,WxXy,UZyx->UVXWZY', np.identity(ncas), v_aaaa, rdm_ccaa, optimize = einsum_type)
    K22_aaa_aaa += 5/24 * einsum('VY,Wxyz,UZyXxz->UVXWZY', np.identity(ncas), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_aaa_aaa += 1/24 * einsum('VY,Wxyz,UZyXzx->UVXWZY', np.identity(ncas), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_aaa_aaa -= 1/8 * einsum('VY,Wxyz,UZyxXz->UVXWZY', np.identity(ncas), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_aaa_aaa += 1/24 * einsum('VY,Wxyz,UZyxzX->UVXWZY', np.identity(ncas), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_aaa_aaa += 1/24 * einsum('VY,Wxyz,UZyzXx->UVXWZY', np.identity(ncas), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_aaa_aaa += 1/24 * einsum('VY,Wxyz,UZyzxX->UVXWZY', np.identity(ncas), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_aaa_aaa -= 1/2 * einsum('VY,XZxy,UxWy->UVXWZY', np.identity(ncas), v_aaaa, rdm_ccaa, optimize = einsum_type)
    K22_aaa_aaa += 1/6 * einsum('VY,XxyZ,UyWx->UVXWZY', np.identity(ncas), v_aaaa, rdm_ccaa, optimize = einsum_type)
    K22_aaa_aaa -= 1/6 * einsum('VY,XxyZ,UyxW->UVXWZY', np.identity(ncas), v_aaaa, rdm_ccaa, optimize = einsum_type)
    K22_aaa_aaa += 5/24 * einsum('VY,Zxyz,UxzWXy->UVXWZY', np.identity(ncas), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_aaa_aaa += 1/24 * einsum('VY,Zxyz,UxzWyX->UVXWZY', np.identity(ncas), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_aaa_aaa -= 1/8 * einsum('VY,Zxyz,UxzXWy->UVXWZY', np.identity(ncas), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_aaa_aaa += 1/24 * einsum('VY,Zxyz,UxzXyW->UVXWZY', np.identity(ncas), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_aaa_aaa += 1/24 * einsum('VY,Zxyz,UxzyWX->UVXWZY', np.identity(ncas), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_aaa_aaa += 1/24 * einsum('VY,Zxyz,UxzyXW->UVXWZY', np.identity(ncas), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_aaa_aaa += 1/6 * einsum('VZ,WxXy,UYxy->UVXWZY', np.identity(ncas), v_aaaa, rdm_ccaa, optimize = einsum_type)
    K22_aaa_aaa -= 1/6 * einsum('VZ,WxXy,UYyx->UVXWZY', np.identity(ncas), v_aaaa, rdm_ccaa, optimize = einsum_type)
    K22_aaa_aaa -= 5/24 * einsum('VZ,Wxyz,UYyXxz->UVXWZY', np.identity(ncas), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_aaa_aaa -= 1/24 * einsum('VZ,Wxyz,UYyXzx->UVXWZY', np.identity(ncas), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_aaa_aaa += 1/8 * einsum('VZ,Wxyz,UYyxXz->UVXWZY', np.identity(ncas), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_aaa_aaa -= 1/24 * einsum('VZ,Wxyz,UYyxzX->UVXWZY', np.identity(ncas), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_aaa_aaa -= 1/24 * einsum('VZ,Wxyz,UYyzXx->UVXWZY', np.identity(ncas), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_aaa_aaa -= 1/24 * einsum('VZ,Wxyz,UYyzxX->UVXWZY', np.identity(ncas), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_aaa_aaa += 1/2 * einsum('VZ,XYxy,UxWy->UVXWZY', np.identity(ncas), v_aaaa, rdm_ccaa, optimize = einsum_type)
    K22_aaa_aaa -= 1/6 * einsum('VZ,XxyY,UyWx->UVXWZY', np.identity(ncas), v_aaaa, rdm_ccaa, optimize = einsum_type)
    K22_aaa_aaa += 1/6 * einsum('VZ,XxyY,UyxW->UVXWZY', np.identity(ncas), v_aaaa, rdm_ccaa, optimize = einsum_type)
    K22_aaa_aaa -= 5/24 * einsum('VZ,Yxyz,UxzWXy->UVXWZY', np.identity(ncas), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_aaa_aaa -= 1/24 * einsum('VZ,Yxyz,UxzWyX->UVXWZY', np.identity(ncas), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_aaa_aaa += 1/8 * einsum('VZ,Yxyz,UxzXWy->UVXWZY', np.identity(ncas), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_aaa_aaa -= 1/24 * einsum('VZ,Yxyz,UxzXyW->UVXWZY', np.identity(ncas), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_aaa_aaa -= 1/24 * einsum('VZ,Yxyz,UxzyWX->UVXWZY', np.identity(ncas), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_aaa_aaa -= 1/24 * einsum('VZ,Yxyz,UxzyXW->UVXWZY', np.identity(ncas), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_aaa_aaa += 1/2 * einsum('XY,VZxy,UxWy->UVXWZY', np.identity(ncas), v_aaaa, rdm_ccaa, optimize = einsum_type)
    K22_aaa_aaa -= 1/6 * einsum('XY,VxWy,UZxy->UVXWZY', np.identity(ncas), v_aaaa, rdm_ccaa, optimize = einsum_type)
    K22_aaa_aaa += 1/6 * einsum('XY,VxWy,UZyx->UVXWZY', np.identity(ncas), v_aaaa, rdm_ccaa, optimize = einsum_type)
    K22_aaa_aaa -= 1/6 * einsum('XY,VxyZ,UyWx->UVXWZY', np.identity(ncas), v_aaaa, rdm_ccaa, optimize = einsum_type)
    K22_aaa_aaa += 1/6 * einsum('XY,VxyZ,UyxW->UVXWZY', np.identity(ncas), v_aaaa, rdm_ccaa, optimize = einsum_type)
    K22_aaa_aaa -= 5/24 * einsum('XY,Wxyz,UZyVxz->UVXWZY', np.identity(ncas), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_aaa_aaa -= 1/24 * einsum('XY,Wxyz,UZyVzx->UVXWZY', np.identity(ncas), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_aaa_aaa += 1/8 * einsum('XY,Wxyz,UZyxVz->UVXWZY', np.identity(ncas), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_aaa_aaa -= 1/24 * einsum('XY,Wxyz,UZyxzV->UVXWZY', np.identity(ncas), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_aaa_aaa -= 1/24 * einsum('XY,Wxyz,UZyzVx->UVXWZY', np.identity(ncas), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_aaa_aaa -= 1/24 * einsum('XY,Wxyz,UZyzxV->UVXWZY', np.identity(ncas), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_aaa_aaa += 5/24 * einsum('XY,Zxyz,UxzVWy->UVXWZY', np.identity(ncas), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_aaa_aaa += 1/24 * einsum('XY,Zxyz,UxzVyW->UVXWZY', np.identity(ncas), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_aaa_aaa -= 1/8 * einsum('XY,Zxyz,UxzWVy->UVXWZY', np.identity(ncas), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_aaa_aaa += 1/24 * einsum('XY,Zxyz,UxzWyV->UVXWZY', np.identity(ncas), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_aaa_aaa += 1/24 * einsum('XY,Zxyz,UxzyVW->UVXWZY', np.identity(ncas), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_aaa_aaa += 1/24 * einsum('XY,Zxyz,UxzyWV->UVXWZY', np.identity(ncas), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_aaa_aaa -= 1/2 * einsum('XZ,VYxy,UxWy->UVXWZY', np.identity(ncas), v_aaaa, rdm_ccaa, optimize = einsum_type)
    K22_aaa_aaa += 1/6 * einsum('XZ,VxWy,UYxy->UVXWZY', np.identity(ncas), v_aaaa, rdm_ccaa, optimize = einsum_type)
    K22_aaa_aaa -= 1/6 * einsum('XZ,VxWy,UYyx->UVXWZY', np.identity(ncas), v_aaaa, rdm_ccaa, optimize = einsum_type)
    K22_aaa_aaa += 1/6 * einsum('XZ,VxyY,UyWx->UVXWZY', np.identity(ncas), v_aaaa, rdm_ccaa, optimize = einsum_type)
    K22_aaa_aaa -= 1/6 * einsum('XZ,VxyY,UyxW->UVXWZY', np.identity(ncas), v_aaaa, rdm_ccaa, optimize = einsum_type)
    K22_aaa_aaa += 5/24 * einsum('XZ,Wxyz,UYyVxz->UVXWZY', np.identity(ncas), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_aaa_aaa += 1/24 * einsum('XZ,Wxyz,UYyVzx->UVXWZY', np.identity(ncas), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_aaa_aaa -= 1/8 * einsum('XZ,Wxyz,UYyxVz->UVXWZY', np.identity(ncas), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_aaa_aaa += 1/24 * einsum('XZ,Wxyz,UYyxzV->UVXWZY', np.identity(ncas), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_aaa_aaa += 1/24 * einsum('XZ,Wxyz,UYyzVx->UVXWZY', np.identity(ncas), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_aaa_aaa += 1/24 * einsum('XZ,Wxyz,UYyzxV->UVXWZY', np.identity(ncas), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_aaa_aaa -= 5/24 * einsum('XZ,Yxyz,UxzVWy->UVXWZY', np.identity(ncas), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_aaa_aaa -= 1/24 * einsum('XZ,Yxyz,UxzVyW->UVXWZY', np.identity(ncas), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_aaa_aaa += 1/8 * einsum('XZ,Yxyz,UxzWVy->UVXWZY', np.identity(ncas), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_aaa_aaa -= 1/24 * einsum('XZ,Yxyz,UxzWyV->UVXWZY', np.identity(ncas), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_aaa_aaa -= 1/24 * einsum('XZ,Yxyz,UxzyVW->UVXWZY', np.identity(ncas), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_aaa_aaa -= 1/24 * einsum('XZ,Yxyz,UxzyWV->UVXWZY', np.identity(ncas), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_aaa_aaa += 1/2 * einsum('Wx,VY,XZ,Ux->UVXWZY', h_aa, np.identity(ncas), np.identity(ncas), rdm_ca, optimize = einsum_type)
    K22_aaa_aaa -= 1/2 * einsum('Wx,VZ,XY,Ux->UVXWZY', h_aa, np.identity(ncas), np.identity(ncas), rdm_ca, optimize = einsum_type)
    K22_aaa_aaa += 1/2 * einsum('Wxyz,VY,XZ,Uyxz->UVXWZY', v_aaaa, np.identity(ncas), np.identity(ncas), rdm_ccaa, optimize = einsum_type)
    K22_aaa_aaa -= 1/2 * einsum('Wxyz,VZ,XY,Uyxz->UVXWZY', v_aaaa, np.identity(ncas), np.identity(ncas), rdm_ccaa, optimize = einsum_type)

    K22_bba_bba =- 1/3 * einsum('VZ,UYWX->UVXWZY', h_aa, rdm_ccaa, optimize = einsum_type)
    K22_bba_bba -= 1/6 * einsum('VZ,UYXW->UVXWZY', h_aa, rdm_ccaa, optimize = einsum_type)
    K22_bba_bba += 1/6 * einsum('Wx,UYZVXx->UVXWZY', h_aa, rdm_cccaaa, optimize = einsum_type)
    K22_bba_bba += 1/12 * einsum('Wx,UYZVxX->UVXWZY', h_aa, rdm_cccaaa, optimize = einsum_type)
    K22_bba_bba += 1/12 * einsum('Wx,UYZXVx->UVXWZY', h_aa, rdm_cccaaa, optimize = einsum_type)
    K22_bba_bba -= 1/12 * einsum('Wx,UYZxXV->UVXWZY', h_aa, rdm_cccaaa, optimize = einsum_type)
    K22_bba_bba += 1/6 * einsum('XY,UZVW->UVXWZY', h_aa, rdm_ccaa, optimize = einsum_type)
    K22_bba_bba -= 1/6 * einsum('XY,UZWV->UVXWZY', h_aa, rdm_ccaa, optimize = einsum_type)
    K22_bba_bba -= 1/6 * einsum('Yx,UZxVWX->UVXWZY', h_aa, rdm_cccaaa, optimize = einsum_type)
    K22_bba_bba -= 1/12 * einsum('Yx,UZxVXW->UVXWZY', h_aa, rdm_cccaaa, optimize = einsum_type)
    K22_bba_bba += 1/12 * einsum('Yx,UZxWVX->UVXWZY', h_aa, rdm_cccaaa, optimize = einsum_type)
    K22_bba_bba -= 1/12 * einsum('Yx,UZxXWV->UVXWZY', h_aa, rdm_cccaaa, optimize = einsum_type)
    K22_bba_bba -= 1/12 * einsum('Zx,UYxVXW->UVXWZY', h_aa, rdm_cccaaa, optimize = einsum_type)
    K22_bba_bba += 1/12 * einsum('Zx,UYxWVX->UVXWZY', h_aa, rdm_cccaaa, optimize = einsum_type)
    K22_bba_bba += 1/6 * einsum('Zx,UYxWXV->UVXWZY', h_aa, rdm_cccaaa, optimize = einsum_type)
    K22_bba_bba += 1/12 * einsum('Zx,UYxXWV->UVXWZY', h_aa, rdm_cccaaa, optimize = einsum_type)
    K22_bba_bba += 1/2 * einsum('VZXY,UW->UVXWZY', v_aaaa, rdm_ca, optimize = einsum_type)
    K22_bba_bba -= 1/3 * einsum('VZXx,UYWx->UVXWZY', v_aaaa, rdm_ccaa, optimize = einsum_type)
    K22_bba_bba -= 1/6 * einsum('VZXx,UYxW->UVXWZY', v_aaaa, rdm_ccaa, optimize = einsum_type)
    K22_bba_bba -= 1/3 * einsum('VZxY,UxWX->UVXWZY', v_aaaa, rdm_ccaa, optimize = einsum_type)
    K22_bba_bba -= 1/6 * einsum('VZxY,UxXW->UVXWZY', v_aaaa, rdm_ccaa, optimize = einsum_type)
    K22_bba_bba -= 1/3 * einsum('VZxy,UYxWXy->UVXWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_bba_bba -= 1/6 * einsum('VZxy,UYxXWy->UVXWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_bba_bba += 1/24 * einsum('VxWy,UYZXxy->UVXWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_bba_bba -= 1/24 * einsum('VxWy,UYZXyx->UVXWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_bba_bba += 1/8 * einsum('VxWy,UYZxXy->UVXWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_bba_bba += 1/24 * einsum('VxWy,UYZxyX->UVXWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_bba_bba -= 1/8 * einsum('VxWy,UYZyXx->UVXWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_bba_bba -= 1/24 * einsum('VxWy,UYZyxX->UVXWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_bba_bba -= 1/6 * einsum('VxXY,UZWx->UVXWZY', v_aaaa, rdm_ccaa, optimize = einsum_type)
    K22_bba_bba += 1/6 * einsum('VxXY,UZxW->UVXWZY', v_aaaa, rdm_ccaa, optimize = einsum_type)
    K22_bba_bba += 1/12 * einsum('VxyY,UZyWxX->UVXWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_bba_bba -= 1/12 * einsum('VxyY,UZyXWx->UVXWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_bba_bba -= 1/6 * einsum('VxyY,UZyxWX->UVXWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_bba_bba -= 1/12 * einsum('VxyY,UZyxXW->UVXWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_bba_bba += 1/6 * einsum('VxyZ,UYyWXx->UVXWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_bba_bba += 1/12 * einsum('VxyZ,UYyWxX->UVXWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_bba_bba += 1/12 * einsum('VxyZ,UYyXWx->UVXWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_bba_bba -= 1/12 * einsum('VxyZ,UYyxXW->UVXWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_bba_bba += 1/12 * einsum('WxXy,UYZVxy->UVXWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_bba_bba += 1/6 * einsum('WxXy,UYZVyx->UVXWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_bba_bba -= 1/12 * einsum('WxXy,UYZxyV->UVXWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_bba_bba += 1/12 * einsum('WxXy,UYZyVx->UVXWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_bba_bba += 1/20 * einsum('Wxyz,UYZyVXxz->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_bba_bba += 1/60 * einsum('Wxyz,UYZyVXzx->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_bba_bba -= 1/30 * einsum('Wxyz,UYZyVxzX->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_bba_bba -= 1/30 * einsum('Wxyz,UYZyVzXx->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_bba_bba -= 7/60 * einsum('Wxyz,UYZyVzxX->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_bba_bba -= 1/120 * einsum('Wxyz,UYZyXVxz->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_bba_bba += 1/24 * einsum('Wxyz,UYZyXVzx->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_bba_bba -= 1/60 * einsum('Wxyz,UYZyXxVz->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_bba_bba += 1/60 * einsum('Wxyz,UYZyXxzV->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_bba_bba += 1/30 * einsum('Wxyz,UYZyXzVx->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_bba_bba -= 1/15 * einsum('Wxyz,UYZyXzxV->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_bba_bba -= 1/120 * einsum('Wxyz,UYZyxVXz->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_bba_bba += 1/24 * einsum('Wxyz,UYZyxVzX->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_bba_bba -= 1/20 * einsum('Wxyz,UYZyxXVz->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_bba_bba += 1/15 * einsum('Wxyz,UYZyxXzV->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_bba_bba += 1/30 * einsum('Wxyz,UYZyxzVX->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_bba_bba += 1/60 * einsum('Wxyz,UYZyxzXV->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_bba_bba += 1/120 * einsum('Wxyz,UYZyzVXx->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_bba_bba -= 3/40 * einsum('Wxyz,UYZyzVxX->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_bba_bba += 1/20 * einsum('Wxyz,UYZyzXVx->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_bba_bba -= 1/20 * einsum('Wxyz,UYZyzXxV->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_bba_bba -= 1/60 * einsum('Wxyz,UYZyzxXV->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_bba_bba += 1/6 * einsum('XYxZ,UxVW->UVXWZY', v_aaaa, rdm_ccaa, optimize = einsum_type)
    K22_bba_bba -= 1/6 * einsum('XYxZ,UxWV->UVXWZY', v_aaaa, rdm_ccaa, optimize = einsum_type)
    K22_bba_bba += 1/6 * einsum('XYxy,UZxVWy->UVXWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_bba_bba -= 1/6 * einsum('XYxy,UZxWVy->UVXWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_bba_bba -= 1/6 * einsum('XxyY,UZyVWx->UVXWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_bba_bba -= 1/12 * einsum('XxyY,UZyVxW->UVXWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_bba_bba += 1/12 * einsum('XxyY,UZyWVx->UVXWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_bba_bba -= 1/12 * einsum('XxyY,UZyxWV->UVXWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_bba_bba -= 1/12 * einsum('XxyZ,UYyVxW->UVXWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_bba_bba += 1/12 * einsum('XxyZ,UYyWVx->UVXWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_bba_bba += 1/6 * einsum('XxyZ,UYyWxV->UVXWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_bba_bba += 1/12 * einsum('XxyZ,UYyxWV->UVXWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_bba_bba -= 1/12 * einsum('YxZy,UxyVXW->UVXWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_bba_bba += 1/12 * einsum('YxZy,UxyWVX->UVXWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_bba_bba += 1/6 * einsum('YxZy,UxyWXV->UVXWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_bba_bba += 1/12 * einsum('YxZy,UxyXWV->UVXWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_bba_bba -= 1/8 * einsum('Yxyz,UZxzVWXy->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_bba_bba += 3/40 * einsum('Yxyz,UZxzVWyX->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_bba_bba -= 1/60 * einsum('Yxyz,UZxzVXWy->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_bba_bba += 1/60 * einsum('Yxyz,UZxzVXyW->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_bba_bba += 1/30 * einsum('Yxyz,UZxzVyWX->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_bba_bba -= 1/20 * einsum('Yxyz,UZxzVyXW->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_bba_bba += 1/40 * einsum('Yxyz,UZxzWVXy->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_bba_bba -= 1/40 * einsum('Yxyz,UZxzWVyX->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_bba_bba += 1/30 * einsum('Yxyz,UZxzWXVy->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_bba_bba -= 1/15 * einsum('Yxyz,UZxzWyXV->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_bba_bba += 1/24 * einsum('Yxyz,UZxzXVWy->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_bba_bba -= 1/120 * einsum('Yxyz,UZxzXVyW->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_bba_bba += 1/40 * einsum('Yxyz,UZxzXWVy->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_bba_bba += 3/40 * einsum('Yxyz,UZxzXWyV->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_bba_bba += 1/60 * einsum('Yxyz,UZxzXyVW->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_bba_bba += 1/30 * einsum('Yxyz,UZxzXyWV->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_bba_bba += 1/120 * einsum('Yxyz,UZxzyVWX->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_bba_bba -= 3/40 * einsum('Yxyz,UZxzyVXW->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_bba_bba += 3/40 * einsum('Yxyz,UZxzyWVX->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_bba_bba += 1/120 * einsum('Yxyz,UZxzyWXV->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_bba_bba += 1/60 * einsum('Yxyz,UZxzyXVW->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_bba_bba += 1/30 * einsum('Yxyz,UZxzyXWV->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_bba_bba += 1/30 * einsum('Zxyz,UYxzVWyX->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_bba_bba -= 1/20 * einsum('Zxyz,UYxzVXWy->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_bba_bba -= 1/60 * einsum('Zxyz,UYxzVXyW->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_bba_bba += 7/60 * einsum('Zxyz,UYxzVyWX->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_bba_bba += 1/30 * einsum('Zxyz,UYxzVyXW->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_bba_bba += 1/60 * einsum('Zxyz,UYxzWVXy->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_bba_bba -= 1/30 * einsum('Zxyz,UYxzWVyX->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_bba_bba += 1/20 * einsum('Zxyz,UYxzWXVy->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_bba_bba -= 1/20 * einsum('Zxyz,UYxzWXyV->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_bba_bba -= 1/30 * einsum('Zxyz,UYxzWyVX->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_bba_bba += 1/120 * einsum('Zxyz,UYxzXVWy->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_bba_bba -= 1/24 * einsum('Zxyz,UYxzXVyW->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_bba_bba += 1/120 * einsum('Zxyz,UYxzXWVy->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_bba_bba -= 1/120 * einsum('Zxyz,UYxzXWyV->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_bba_bba -= 1/24 * einsum('Zxyz,UYxzXyVW->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_bba_bba += 3/40 * einsum('Zxyz,UYxzXyWV->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_bba_bba += 1/15 * einsum('Zxyz,UYxzyVWX->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_bba_bba -= 1/60 * einsum('Zxyz,UYxzyVXW->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_bba_bba -= 1/60 * einsum('Zxyz,UYxzyWVX->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_bba_bba += 1/60 * einsum('Zxyz,UYxzyWXV->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_bba_bba -= 1/15 * einsum('Zxyz,UYxzyXVW->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_bba_bba += 1/20 * einsum('Zxyz,UYxzyXWV->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_bba_bba += 1/2 * einsum('VZ,XY,UW->UVXWZY', h_aa, np.identity(ncas), rdm_ca, optimize = einsum_type)
    K22_bba_bba += 1/6 * einsum('Wx,VZ,UYXx->UVXWZY', h_aa, np.identity(ncas), rdm_ccaa, optimize = einsum_type)
    K22_bba_bba += 1/3 * einsum('Wx,VZ,UYxX->UVXWZY', h_aa, np.identity(ncas), rdm_ccaa, optimize = einsum_type)
    K22_bba_bba -= 1/6 * einsum('Wx,XY,UZVx->UVXWZY', h_aa, np.identity(ncas), rdm_ccaa, optimize = einsum_type)
    K22_bba_bba += 1/6 * einsum('Wx,XY,UZxV->UVXWZY', h_aa, np.identity(ncas), rdm_ccaa, optimize = einsum_type)
    K22_bba_bba += 1/2 * einsum('XY,VZ,UW->UVXWZY', h_aa, np.identity(ncas), rdm_ca, optimize = einsum_type)
    K22_bba_bba -= 1/3 * einsum('Yx,VZ,UxWX->UVXWZY', h_aa, np.identity(ncas), rdm_ccaa, optimize = einsum_type)
    K22_bba_bba -= 1/6 * einsum('Yx,VZ,UxXW->UVXWZY', h_aa, np.identity(ncas), rdm_ccaa, optimize = einsum_type)
    K22_bba_bba += 1/6 * einsum('Zx,XY,UxVW->UVXWZY', h_aa, np.identity(ncas), rdm_ccaa, optimize = einsum_type)
    K22_bba_bba -= 1/6 * einsum('Zx,XY,UxWV->UVXWZY', h_aa, np.identity(ncas), rdm_ccaa, optimize = einsum_type)
    K22_bba_bba += 1/3 * einsum('VZ,WxXy,UYxy->UVXWZY', np.identity(ncas), v_aaaa, rdm_ccaa, optimize = einsum_type)
    K22_bba_bba += 1/6 * einsum('VZ,WxXy,UYyx->UVXWZY', np.identity(ncas), v_aaaa, rdm_ccaa, optimize = einsum_type)
    K22_bba_bba += 1/8 * einsum('VZ,Wxyz,UYyXxz->UVXWZY', np.identity(ncas), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_bba_bba -= 1/24 * einsum('VZ,Wxyz,UYyXzx->UVXWZY', np.identity(ncas), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_bba_bba += 7/24 * einsum('VZ,Wxyz,UYyxXz->UVXWZY', np.identity(ncas), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_bba_bba -= 1/24 * einsum('VZ,Wxyz,UYyxzX->UVXWZY', np.identity(ncas), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_bba_bba -= 1/24 * einsum('VZ,Wxyz,UYyzXx->UVXWZY', np.identity(ncas), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_bba_bba -= 1/24 * einsum('VZ,Wxyz,UYyzxX->UVXWZY', np.identity(ncas), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_bba_bba += 1/2 * einsum('VZ,XYxy,UxWy->UVXWZY', np.identity(ncas), v_aaaa, rdm_ccaa, optimize = einsum_type)
    K22_bba_bba -= 1/3 * einsum('VZ,XxyY,UyWx->UVXWZY', np.identity(ncas), v_aaaa, rdm_ccaa, optimize = einsum_type)
    K22_bba_bba -= 1/6 * einsum('VZ,XxyY,UyxW->UVXWZY', np.identity(ncas), v_aaaa, rdm_ccaa, optimize = einsum_type)
    K22_bba_bba -= 7/24 * einsum('VZ,Yxyz,UxzWXy->UVXWZY', np.identity(ncas), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_bba_bba += 1/24 * einsum('VZ,Yxyz,UxzWyX->UVXWZY', np.identity(ncas), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_bba_bba -= 1/8 * einsum('VZ,Yxyz,UxzXWy->UVXWZY', np.identity(ncas), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_bba_bba += 1/24 * einsum('VZ,Yxyz,UxzXyW->UVXWZY', np.identity(ncas), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_bba_bba += 1/24 * einsum('VZ,Yxyz,UxzyWX->UVXWZY', np.identity(ncas), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_bba_bba += 1/24 * einsum('VZ,Yxyz,UxzyXW->UVXWZY', np.identity(ncas), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_bba_bba += 1/2 * einsum('XY,VZxy,UxWy->UVXWZY', np.identity(ncas), v_aaaa, rdm_ccaa, optimize = einsum_type)
    K22_bba_bba -= 1/6 * einsum('XY,VxWy,UZxy->UVXWZY', np.identity(ncas), v_aaaa, rdm_ccaa, optimize = einsum_type)
    K22_bba_bba += 1/6 * einsum('XY,VxWy,UZyx->UVXWZY', np.identity(ncas), v_aaaa, rdm_ccaa, optimize = einsum_type)
    K22_bba_bba -= 1/6 * einsum('XY,VxyZ,UyWx->UVXWZY', np.identity(ncas), v_aaaa, rdm_ccaa, optimize = einsum_type)
    K22_bba_bba += 1/6 * einsum('XY,VxyZ,UyxW->UVXWZY', np.identity(ncas), v_aaaa, rdm_ccaa, optimize = einsum_type)
    K22_bba_bba -= 1/8 * einsum('XY,Wxyz,UZyVxz->UVXWZY', np.identity(ncas), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_bba_bba += 1/24 * einsum('XY,Wxyz,UZyVzx->UVXWZY', np.identity(ncas), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_bba_bba += 5/24 * einsum('XY,Wxyz,UZyxVz->UVXWZY', np.identity(ncas), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_bba_bba += 1/24 * einsum('XY,Wxyz,UZyxzV->UVXWZY', np.identity(ncas), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_bba_bba += 1/24 * einsum('XY,Wxyz,UZyzVx->UVXWZY', np.identity(ncas), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_bba_bba += 1/24 * einsum('XY,Wxyz,UZyzxV->UVXWZY', np.identity(ncas), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_bba_bba += 1/8 * einsum('XY,Zxyz,UxzVWy->UVXWZY', np.identity(ncas), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_bba_bba -= 1/24 * einsum('XY,Zxyz,UxzVyW->UVXWZY', np.identity(ncas), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_bba_bba -= 5/24 * einsum('XY,Zxyz,UxzWVy->UVXWZY', np.identity(ncas), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_bba_bba -= 1/24 * einsum('XY,Zxyz,UxzWyV->UVXWZY', np.identity(ncas), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_bba_bba -= 1/24 * einsum('XY,Zxyz,UxzyVW->UVXWZY', np.identity(ncas), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_bba_bba -= 1/24 * einsum('XY,Zxyz,UxzyWV->UVXWZY', np.identity(ncas), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_bba_bba -= 1/2 * einsum('Wx,VZ,XY,Ux->UVXWZY', h_aa, np.identity(ncas), np.identity(ncas), rdm_ca, optimize = einsum_type)
    K22_bba_bba -= 1/2 * einsum('Wxyz,VZ,XY,Uyxz->UVXWZY', v_aaaa, np.identity(ncas), np.identity(ncas), rdm_ccaa, optimize = einsum_type)

    K22_aaa_bba = np.ascontiguousarray(K22_aaa_aaa - K22_bba_bba.transpose(0,2,1,3,5,4) + K22_bba_bba.transpose(0,1,2,3,5,4))
    K22_bba_aaa = np.ascontiguousarray(K22_aaa_aaa - K22_bba_bba.transpose(0,2,1,3,5,4) + K22_bba_bba.transpose(0,2,1,3,4,5))

    # Reshape tensors to matrix form
    dim_x = ncas
    dim_wzy = ncas * ncas * ncas
    dim_tril_wzy = ncas * ncas * (ncas - 1) // 2

    dim_act = dim_x + dim_wzy + dim_tril_wzy

    tril_ind = np.tril_indices(ncas, k=-1)

    K12_a_aaa = K12_a_aaa[:, :, tril_ind[0], tril_ind[1]]

    K22_aaa_aaa = K22_aaa_aaa[:, :, :, :, tril_ind[0], tril_ind[1]]
    K22_aaa_aaa = K22_aaa_aaa[:, tril_ind[0], tril_ind[1]]

    K22_aaa_bba = K22_aaa_bba[:, tril_ind[0], tril_ind[1]]
    K22_bba_aaa = K22_bba_aaa[:, :, :, :, tril_ind[0], tril_ind[1]]

    K12_a_aaa = K12_a_aaa.reshape(dim_x, dim_tril_wzy)
    K12_a_bba = K12_a_bba.reshape(dim_x, dim_wzy)

    K22_aaa_aaa = K22_aaa_aaa.reshape(dim_tril_wzy, dim_tril_wzy)
    K22_aaa_bba = K22_aaa_bba.reshape(dim_tril_wzy, dim_wzy)

    K22_bba_aaa = K22_bba_aaa.reshape(dim_wzy, dim_tril_wzy)
    K22_bba_bba = K22_bba_bba.reshape(dim_wzy, dim_wzy)

    # Build K_p1p matrix
    s_a = 0
    f_a = dim_x
    s_aaa = f_a
    f_aaa = s_aaa + dim_tril_wzy
    s_bba = f_aaa
    f_bba = s_bba + dim_wzy

    K_p1p = np.zeros((dim_act, dim_act))

    K_p1p[s_a:f_a, s_a:f_a] = K11_a_a

    K_p1p[s_a:f_a, s_aaa:f_aaa] = K12_a_aaa
    K_p1p[s_a:f_a, s_bba:f_bba] = K12_a_bba

    K_p1p[s_aaa:f_aaa, s_a:f_a] = K12_a_aaa.T
    K_p1p[s_bba:f_bba, s_a:f_a] = K12_a_bba.T

    K_p1p[s_aaa:f_aaa, s_aaa:f_aaa] = K22_aaa_aaa
    K_p1p[s_aaa:f_aaa, s_bba:f_bba] = K22_aaa_bba

    K_p1p[s_bba:f_bba, s_aaa:f_aaa] = K22_bba_aaa
    K_p1p[s_bba:f_bba, s_bba:f_bba] = K22_bba_bba

    return K_p1p

def compute_K_m1p(mr_adc):

    # Einsum definition from kernel
    einsum = mr_adc.interface.einsum
    einsum_type = mr_adc.interface.einsum_type

    # Variables from kernel
    ncas = mr_adc.ncas

    ## One-electron integrals
    h_aa = mr_adc.h1eff.aa

    ## Two-electron integrals
    v_aaaa = mr_adc.v2e.aaaa

    ## Reduced density matrices
    rdm_ca = mr_adc.rdm.ca
    rdm_ccaa = mr_adc.rdm.ccaa
    rdm_cccaaa = mr_adc.rdm.cccaaa
    rdm_ccccaaaa = mr_adc.rdm.ccccaaaa

    # Computing K11
    # K11 block: < Psi_0 | a^{\dag}_X [H_{act}, a_Y] | Psi_0 >
    K11_a_a =- 1/2 * einsum('Yx,Xx->XY', h_aa, rdm_ca, optimize = einsum_type)
    K11_a_a -= 1/2 * einsum('Yxyz,Xyxz->XY', v_aaaa, rdm_ccaa, optimize = einsum_type)

    # K12 block: < Psi_0 | a^{\dag}_X [H_{act}, a^{\dag}_Y a_Z a_W] | Psi_0 >
    K12_a_abb =- 1/6 * einsum('Wx,XYZx->XWZY', h_aa, rdm_ccaa, optimize = einsum_type)
    K12_a_abb -= 1/3 * einsum('Wx,XYxZ->XWZY', h_aa, rdm_ccaa, optimize = einsum_type)
    K12_a_abb += 1/3 * einsum('Yx,WZXx->XWZY', h_aa, rdm_ccaa, optimize = einsum_type)
    K12_a_abb += 1/6 * einsum('Yx,WZxX->XWZY', h_aa, rdm_ccaa, optimize = einsum_type)
    K12_a_abb -= 1/3 * einsum('Zx,WxXY->XWZY', h_aa, rdm_ccaa, optimize = einsum_type)
    K12_a_abb -= 1/6 * einsum('Zx,WxYX->XWZY', h_aa, rdm_ccaa, optimize = einsum_type)
    K12_a_abb -= 1/3 * einsum('WxZy,XYxy->XWZY', v_aaaa, rdm_ccaa, optimize = einsum_type)
    K12_a_abb -= 1/6 * einsum('WxZy,XYyx->XWZY', v_aaaa, rdm_ccaa, optimize = einsum_type)
    K12_a_abb -= 1/24 * einsum('Wxyz,XYyZxz->XWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K12_a_abb += 1/8 * einsum('Wxyz,XYyZzx->XWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K12_a_abb -= 5/24 * einsum('Wxyz,XYyxZz->XWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K12_a_abb += 1/8 * einsum('Wxyz,XYyxzZ->XWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K12_a_abb += 1/8 * einsum('Wxyz,XYyzZx->XWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K12_a_abb += 1/8 * einsum('Wxyz,XYyzxZ->XWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K12_a_abb += 5/24 * einsum('Yxyz,WZyXxz->XWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K12_a_abb -= 1/8 * einsum('Yxyz,WZyXzx->XWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K12_a_abb += 1/24 * einsum('Yxyz,WZyxXz->XWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K12_a_abb -= 1/8 * einsum('Yxyz,WZyxzX->XWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K12_a_abb -= 1/8 * einsum('Yxyz,WZyzXx->XWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K12_a_abb -= 1/8 * einsum('Yxyz,WZyzxX->XWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K12_a_abb -= 5/24 * einsum('Zxyz,WxzXYy->XWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K12_a_abb += 1/8 * einsum('Zxyz,WxzXyY->XWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K12_a_abb -= 1/24 * einsum('Zxyz,WxzYXy->XWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K12_a_abb += 1/8 * einsum('Zxyz,WxzYyX->XWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K12_a_abb += 1/8 * einsum('Zxyz,WxzyXY->XWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K12_a_abb += 1/8 * einsum('Zxyz,WxzyYX->XWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)

    K12_a_aaa = np.ascontiguousarray(K12_a_abb - K12_a_abb.transpose(0,2,1,3))

    # K22 block: < Psi_0 | a^{\dag}_U a^{\dag}_V a_X [H_{act}, a^{\dag}_Y a_Z a_W] | Psi_0 >
    K22_aaa_aaa  = 1/12 * einsum('Wx,UVYXxZ->UVXWZY', h_aa, rdm_cccaaa, optimize = einsum_type)
    K22_aaa_aaa += 1/12 * einsum('Wx,UVYZXx->UVXWZY', h_aa, rdm_cccaaa, optimize = einsum_type)
    K22_aaa_aaa += 1/12 * einsum('Wx,UVYxZX->UVXWZY', h_aa, rdm_cccaaa, optimize = einsum_type)
    K22_aaa_aaa += 1/6 * einsum('XY,UVWZ->UVXWZY', h_aa, rdm_ccaa, optimize = einsum_type)
    K22_aaa_aaa -= 1/6 * einsum('XY,UVZW->UVXWZY', h_aa, rdm_ccaa, optimize = einsum_type)
    K22_aaa_aaa -= 1/12 * einsum('Yx,UVxWZX->UVXWZY', h_aa, rdm_cccaaa, optimize = einsum_type)
    K22_aaa_aaa -= 1/12 * einsum('Yx,UVxXWZ->UVXWZY', h_aa, rdm_cccaaa, optimize = einsum_type)
    K22_aaa_aaa -= 1/12 * einsum('Yx,UVxZXW->UVXWZY', h_aa, rdm_cccaaa, optimize = einsum_type)
    K22_aaa_aaa += 1/12 * einsum('Zx,UVYWxX->UVXWZY', h_aa, rdm_cccaaa, optimize = einsum_type)
    K22_aaa_aaa += 1/12 * einsum('Zx,UVYXWx->UVXWZY', h_aa, rdm_cccaaa, optimize = einsum_type)
    K22_aaa_aaa += 1/12 * einsum('Zx,UVYxXW->UVXWZY', h_aa, rdm_cccaaa, optimize = einsum_type)
    K22_aaa_aaa -= 1/24 * einsum('WxXy,UVYZxy->UVXWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_aaa_aaa += 1/24 * einsum('WxXy,UVYZyx->UVXWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_aaa_aaa += 1/24 * einsum('WxXy,UVYxZy->UVXWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_aaa_aaa -= 1/24 * einsum('WxXy,UVYxyZ->UVXWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_aaa_aaa -= 1/24 * einsum('WxXy,UVYyZx->UVXWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_aaa_aaa += 1/24 * einsum('WxXy,UVYyxZ->UVXWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_aaa_aaa += 1/24 * einsum('WxZy,UVYXxy->UVXWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_aaa_aaa -= 1/24 * einsum('WxZy,UVYXyx->UVXWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_aaa_aaa -= 1/24 * einsum('WxZy,UVYxXy->UVXWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_aaa_aaa += 1/24 * einsum('WxZy,UVYxyX->UVXWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_aaa_aaa += 1/24 * einsum('WxZy,UVYyXx->UVXWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_aaa_aaa -= 1/24 * einsum('WxZy,UVYyxX->UVXWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_aaa_aaa -= 3/40 * einsum('Wxyz,UVYyXZxz->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aaa_aaa += 1/40 * einsum('Wxyz,UVYyXZzx->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aaa_aaa -= 1/120 * einsum('Wxyz,UVYyXxZz->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aaa_aaa += 1/120 * einsum('Wxyz,UVYyXxzZ->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aaa_aaa -= 1/120 * einsum('Wxyz,UVYyXzZx->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aaa_aaa -= 1/120 * einsum('Wxyz,UVYyXzxZ->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aaa_aaa -= 1/120 * einsum('Wxyz,UVYyZXxz->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aaa_aaa += 1/120 * einsum('Wxyz,UVYyZXzx->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aaa_aaa -= 3/40 * einsum('Wxyz,UVYyZxXz->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aaa_aaa += 1/40 * einsum('Wxyz,UVYyZxzX->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aaa_aaa += 1/120 * einsum('Wxyz,UVYyZzXx->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aaa_aaa += 1/120 * einsum('Wxyz,UVYyZzxX->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aaa_aaa -= 1/12 * einsum('Wxyz,UVYyxXZz->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aaa_aaa += 1/60 * einsum('Wxyz,UVYyxXzZ->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aaa_aaa += 1/30 * einsum('Wxyz,UVYyxZXz->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aaa_aaa += 1/20 * einsum('Wxyz,UVYyxZzX->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aaa_aaa += 1/60 * einsum('Wxyz,UVYyxzXZ->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aaa_aaa += 1/60 * einsum('Wxyz,UVYyxzZX->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aaa_aaa += 1/60 * einsum('Wxyz,UVYyzXZx->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aaa_aaa += 1/60 * einsum('Wxyz,UVYyzXxZ->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aaa_aaa += 1/20 * einsum('Wxyz,UVYyzZXx->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aaa_aaa += 1/20 * einsum('Wxyz,UVYyzZxX->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aaa_aaa += 1/30 * einsum('Wxyz,UVYyzxXZ->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aaa_aaa += 1/30 * einsum('Wxyz,UVYyzxZX->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aaa_aaa += 1/6 * einsum('XYxy,UVxWZy->UVXWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_aaa_aaa -= 1/6 * einsum('XYxy,UVxZWy->UVXWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_aaa_aaa -= 1/24 * einsum('XxZy,UVYWxy->UVXWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_aaa_aaa += 1/24 * einsum('XxZy,UVYWyx->UVXWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_aaa_aaa += 1/24 * einsum('XxZy,UVYxWy->UVXWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_aaa_aaa -= 1/24 * einsum('XxZy,UVYxyW->UVXWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_aaa_aaa -= 1/24 * einsum('XxZy,UVYyWx->UVXWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_aaa_aaa += 1/24 * einsum('XxZy,UVYyxW->UVXWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_aaa_aaa += 1/12 * einsum('XxyY,UVyWxZ->UVXWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_aaa_aaa += 1/12 * einsum('XxyY,UVyZWx->UVXWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_aaa_aaa += 1/12 * einsum('XxyY,UVyxZW->UVXWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_aaa_aaa += 3/40 * einsum('Yxyz,UVxzWXZy->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aaa_aaa -= 1/40 * einsum('Yxyz,UVxzWXyZ->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aaa_aaa += 1/120 * einsum('Yxyz,UVxzWZXy->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aaa_aaa += 1/120 * einsum('Yxyz,UVxzWZyX->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aaa_aaa -= 1/120 * einsum('Yxyz,UVxzWyXZ->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aaa_aaa += 1/120 * einsum('Yxyz,UVxzWyZX->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aaa_aaa += 1/120 * einsum('Yxyz,UVxzXWZy->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aaa_aaa -= 1/120 * einsum('Yxyz,UVxzXWyZ->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aaa_aaa += 1/12 * einsum('Yxyz,UVxzXZWy->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aaa_aaa -= 1/60 * einsum('Yxyz,UVxzXZyW->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aaa_aaa -= 1/60 * einsum('Yxyz,UVxzXyWZ->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aaa_aaa -= 1/60 * einsum('Yxyz,UVxzXyZW->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aaa_aaa += 3/40 * einsum('Yxyz,UVxzZWXy->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aaa_aaa -= 1/120 * einsum('Yxyz,UVxzZWyX->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aaa_aaa -= 1/30 * einsum('Yxyz,UVxzZXWy->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aaa_aaa -= 1/20 * einsum('Yxyz,UVxzZXyW->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aaa_aaa -= 1/60 * einsum('Yxyz,UVxzZyWX->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aaa_aaa -= 1/30 * einsum('Yxyz,UVxzZyXW->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aaa_aaa -= 1/40 * einsum('Yxyz,UVxzyWXZ->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aaa_aaa -= 1/120 * einsum('Yxyz,UVxzyWZX->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aaa_aaa -= 1/20 * einsum('Yxyz,UVxzyXWZ->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aaa_aaa -= 1/20 * einsum('Yxyz,UVxzyXZW->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aaa_aaa -= 1/60 * einsum('Yxyz,UVxzyZWX->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aaa_aaa -= 1/30 * einsum('Yxyz,UVxzyZXW->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aaa_aaa -= 3/40 * einsum('Zxyz,UVYyWXxz->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aaa_aaa += 1/40 * einsum('Zxyz,UVYyWXzx->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aaa_aaa -= 1/120 * einsum('Zxyz,UVYyWxXz->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aaa_aaa += 1/120 * einsum('Zxyz,UVYyWxzX->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aaa_aaa -= 1/120 * einsum('Zxyz,UVYyWzXx->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aaa_aaa -= 1/120 * einsum('Zxyz,UVYyWzxX->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aaa_aaa -= 1/120 * einsum('Zxyz,UVYyXWxz->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aaa_aaa += 1/120 * einsum('Zxyz,UVYyXWzx->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aaa_aaa -= 3/40 * einsum('Zxyz,UVYyXxWz->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aaa_aaa += 1/40 * einsum('Zxyz,UVYyXxzW->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aaa_aaa += 1/120 * einsum('Zxyz,UVYyXzWx->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aaa_aaa += 1/120 * einsum('Zxyz,UVYyXzxW->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aaa_aaa -= 1/12 * einsum('Zxyz,UVYyxWXz->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aaa_aaa += 1/60 * einsum('Zxyz,UVYyxWzX->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aaa_aaa += 1/30 * einsum('Zxyz,UVYyxXWz->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aaa_aaa += 1/20 * einsum('Zxyz,UVYyxXzW->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aaa_aaa += 1/60 * einsum('Zxyz,UVYyxzWX->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aaa_aaa += 1/60 * einsum('Zxyz,UVYyxzXW->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aaa_aaa += 1/60 * einsum('Zxyz,UVYyzWXx->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aaa_aaa += 1/60 * einsum('Zxyz,UVYyzWxX->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aaa_aaa += 1/20 * einsum('Zxyz,UVYyzXWx->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aaa_aaa += 1/20 * einsum('Zxyz,UVYyzXxW->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aaa_aaa += 1/30 * einsum('Zxyz,UVYyzxWX->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aaa_aaa += 1/30 * einsum('Zxyz,UVYyzxXW->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aaa_aaa += 1/6 * einsum('Wx,XY,UVZx->UVXWZY', h_aa, np.identity(ncas), rdm_ccaa, optimize = einsum_type)
    K22_aaa_aaa -= 1/6 * einsum('Wx,XY,UVxZ->UVXWZY', h_aa, np.identity(ncas), rdm_ccaa, optimize = einsum_type)
    K22_aaa_aaa -= 1/6 * einsum('Zx,XY,UVWx->UVXWZY', h_aa, np.identity(ncas), rdm_ccaa, optimize = einsum_type)
    K22_aaa_aaa += 1/6 * einsum('Zx,XY,UVxW->UVXWZY', h_aa, np.identity(ncas), rdm_ccaa, optimize = einsum_type)
    K22_aaa_aaa -= 1/6 * einsum('XY,WxZy,UVxy->UVXWZY', np.identity(ncas), v_aaaa, rdm_ccaa, optimize = einsum_type)
    K22_aaa_aaa += 1/6 * einsum('XY,WxZy,UVyx->UVXWZY', np.identity(ncas), v_aaaa, rdm_ccaa, optimize = einsum_type)
    K22_aaa_aaa += 5/24 * einsum('XY,Wxyz,UVyZxz->UVXWZY', np.identity(ncas), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_aaa_aaa += 1/24 * einsum('XY,Wxyz,UVyZzx->UVXWZY', np.identity(ncas), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_aaa_aaa -= 1/8 * einsum('XY,Wxyz,UVyxZz->UVXWZY', np.identity(ncas), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_aaa_aaa += 1/24 * einsum('XY,Wxyz,UVyxzZ->UVXWZY', np.identity(ncas), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_aaa_aaa += 1/24 * einsum('XY,Wxyz,UVyzZx->UVXWZY', np.identity(ncas), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_aaa_aaa += 1/24 * einsum('XY,Wxyz,UVyzxZ->UVXWZY', np.identity(ncas), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_aaa_aaa -= 5/24 * einsum('XY,Zxyz,UVyWxz->UVXWZY', np.identity(ncas), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_aaa_aaa -= 1/24 * einsum('XY,Zxyz,UVyWzx->UVXWZY', np.identity(ncas), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_aaa_aaa += 1/8 * einsum('XY,Zxyz,UVyxWz->UVXWZY', np.identity(ncas), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_aaa_aaa -= 1/24 * einsum('XY,Zxyz,UVyxzW->UVXWZY', np.identity(ncas), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_aaa_aaa -= 1/24 * einsum('XY,Zxyz,UVyzWx->UVXWZY', np.identity(ncas), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_aaa_aaa -= 1/24 * einsum('XY,Zxyz,UVyzxW->UVXWZY', np.identity(ncas), v_aaaa, rdm_cccaaa, optimize = einsum_type)

    K22_abb_abb =- 1/12 * einsum('Wx,UVYXxZ->UVXWZY', h_aa, rdm_cccaaa, optimize = einsum_type)
    K22_abb_abb -= 1/12 * einsum('Wx,UVYZXx->UVXWZY', h_aa, rdm_cccaaa, optimize = einsum_type)
    K22_abb_abb -= 1/6 * einsum('Wx,UVYxXZ->UVXWZY', h_aa, rdm_cccaaa, optimize = einsum_type)
    K22_abb_abb += 1/12 * einsum('Wx,UVYxZX->UVXWZY', h_aa, rdm_cccaaa, optimize = einsum_type)
    K22_abb_abb += 1/3 * einsum('XY,UVWZ->UVXWZY', h_aa, rdm_ccaa, optimize = einsum_type)
    K22_abb_abb += 1/6 * einsum('XY,UVZW->UVXWZY', h_aa, rdm_ccaa, optimize = einsum_type)
    K22_abb_abb += 1/6 * einsum('Yx,UVxWXZ->UVXWZY', h_aa, rdm_cccaaa, optimize = einsum_type)
    K22_abb_abb -= 1/12 * einsum('Yx,UVxWZX->UVXWZY', h_aa, rdm_cccaaa, optimize = einsum_type)
    K22_abb_abb += 1/12 * einsum('Yx,UVxXWZ->UVXWZY', h_aa, rdm_cccaaa, optimize = einsum_type)
    K22_abb_abb += 1/12 * einsum('Yx,UVxZXW->UVXWZY', h_aa, rdm_cccaaa, optimize = einsum_type)
    K22_abb_abb -= 1/6 * einsum('Zx,UVYWXx->UVXWZY', h_aa, rdm_cccaaa, optimize = einsum_type)
    K22_abb_abb += 1/12 * einsum('Zx,UVYWxX->UVXWZY', h_aa, rdm_cccaaa, optimize = einsum_type)
    K22_abb_abb -= 1/12 * einsum('Zx,UVYXWx->UVXWZY', h_aa, rdm_cccaaa, optimize = einsum_type)
    K22_abb_abb -= 1/12 * einsum('Zx,UVYxXW->UVXWZY', h_aa, rdm_cccaaa, optimize = einsum_type)
    K22_abb_abb -= 1/12 * einsum('WxXy,UVYZyx->UVXWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_abb_abb += 1/12 * einsum('WxXy,UVYxZy->UVXWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_abb_abb -= 1/6 * einsum('WxXy,UVYxyZ->UVXWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_abb_abb -= 1/12 * einsum('WxXy,UVYyxZ->UVXWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_abb_abb += 1/12 * einsum('WxZy,UVYXyx->UVXWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_abb_abb -= 1/12 * einsum('WxZy,UVYxXy->UVXWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_abb_abb += 1/6 * einsum('WxZy,UVYxyX->UVXWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_abb_abb += 1/12 * einsum('WxZy,UVYyxX->UVXWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_abb_abb += 1/24 * einsum('Wxyz,UVYyXZxz->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_abb_abb -= 1/120 * einsum('Wxyz,UVYyXZzx->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_abb_abb += 1/40 * einsum('Wxyz,UVYyXxZz->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_abb_abb += 1/60 * einsum('Wxyz,UVYyXxzZ->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_abb_abb += 3/40 * einsum('Wxyz,UVYyXzZx->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_abb_abb += 1/30 * einsum('Wxyz,UVYyXzxZ->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_abb_abb -= 1/60 * einsum('Wxyz,UVYyZXxz->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_abb_abb += 1/60 * einsum('Wxyz,UVYyZXzx->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_abb_abb += 1/30 * einsum('Wxyz,UVYyZxXz->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_abb_abb += 1/60 * einsum('Wxyz,UVYyZxzX->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_abb_abb += 1/30 * einsum('Wxyz,UVYyZzxX->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_abb_abb -= 1/8 * einsum('Wxyz,UVYyxXZz->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_abb_abb -= 1/20 * einsum('Wxyz,UVYyxXzZ->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_abb_abb += 1/40 * einsum('Wxyz,UVYyxZXz->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_abb_abb -= 3/40 * einsum('Wxyz,UVYyxZzX->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_abb_abb -= 1/15 * einsum('Wxyz,UVYyxzXZ->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_abb_abb += 1/120 * einsum('Wxyz,UVYyxzZX->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_abb_abb += 3/40 * einsum('Wxyz,UVYyzXZx->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_abb_abb += 1/30 * einsum('Wxyz,UVYyzXxZ->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_abb_abb -= 1/40 * einsum('Wxyz,UVYyzZXx->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_abb_abb += 1/120 * einsum('Wxyz,UVYyzZxX->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_abb_abb += 3/40 * einsum('Wxyz,UVYyzxZX->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_abb_abb += 1/3 * einsum('XYxy,UVxWZy->UVXWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_abb_abb += 1/6 * einsum('XYxy,UVxZWy->UVXWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_abb_abb -= 1/8 * einsum('XxZy,UVYWxy->UVXWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_abb_abb += 1/8 * einsum('XxZy,UVYWyx->UVXWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_abb_abb -= 1/24 * einsum('XxZy,UVYxWy->UVXWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_abb_abb += 1/24 * einsum('XxZy,UVYxyW->UVXWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_abb_abb += 1/24 * einsum('XxZy,UVYyWx->UVXWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_abb_abb -= 1/24 * einsum('XxZy,UVYyxW->UVXWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_abb_abb -= 1/6 * einsum('XxyY,UVyWZx->UVXWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_abb_abb += 1/12 * einsum('XxyY,UVyWxZ->UVXWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_abb_abb -= 1/12 * einsum('XxyY,UVyZWx->UVXWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_abb_abb -= 1/12 * einsum('XxyY,UVyxZW->UVXWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_abb_abb += 1/20 * einsum('Yxyz,UVxzWXZy->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_abb_abb += 1/60 * einsum('Yxyz,UVxzWXyZ->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_abb_abb -= 1/20 * einsum('Yxyz,UVxzWZXy->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_abb_abb += 1/20 * einsum('Yxyz,UVxzWZyX->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_abb_abb += 1/15 * einsum('Yxyz,UVxzWyXZ->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_abb_abb -= 1/20 * einsum('Yxyz,UVxzWyZX->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_abb_abb -= 1/120 * einsum('Yxyz,UVxzXWZy->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_abb_abb += 1/24 * einsum('Yxyz,UVxzXWyZ->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_abb_abb -= 1/60 * einsum('Yxyz,UVxzXZWy->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_abb_abb += 1/30 * einsum('Yxyz,UVxzXZyW->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_abb_abb += 1/60 * einsum('Yxyz,UVxzXyWZ->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_abb_abb -= 1/15 * einsum('Yxyz,UVxzXyZW->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_abb_abb -= 1/120 * einsum('Yxyz,UVxzZWXy->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_abb_abb += 1/120 * einsum('Yxyz,UVxzZWyX->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_abb_abb -= 1/30 * einsum('Yxyz,UVxzZXyW->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_abb_abb -= 1/60 * einsum('Yxyz,UVxzZyWX->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_abb_abb += 1/60 * einsum('Yxyz,UVxzZyXW->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_abb_abb += 1/24 * einsum('Yxyz,UVxzyWXZ->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_abb_abb -= 3/40 * einsum('Yxyz,UVxzyWZX->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_abb_abb -= 1/30 * einsum('Yxyz,UVxzyXWZ->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_abb_abb -= 7/60 * einsum('Yxyz,UVxzyXZW->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_abb_abb += 1/30 * einsum('Yxyz,UVxzyZXW->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_abb_abb -= 1/20 * einsum('Zxyz,UVYyWXxz->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_abb_abb -= 1/60 * einsum('Zxyz,UVYyWXzx->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_abb_abb += 1/20 * einsum('Zxyz,UVYyWxXz->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_abb_abb -= 1/15 * einsum('Zxyz,UVYyWxzX->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_abb_abb -= 1/20 * einsum('Zxyz,UVYyWzXx->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_abb_abb += 1/20 * einsum('Zxyz,UVYyWzxX->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_abb_abb += 1/120 * einsum('Zxyz,UVYyXWxz->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_abb_abb -= 1/24 * einsum('Zxyz,UVYyXWzx->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_abb_abb += 1/120 * einsum('Zxyz,UVYyXxWz->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_abb_abb -= 1/24 * einsum('Zxyz,UVYyXxzW->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_abb_abb -= 1/120 * einsum('Zxyz,UVYyXzWx->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_abb_abb += 3/40 * einsum('Zxyz,UVYyXzxW->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_abb_abb += 1/60 * einsum('Zxyz,UVYyxWXz->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_abb_abb -= 1/60 * einsum('Zxyz,UVYyxWzX->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_abb_abb += 1/30 * einsum('Zxyz,UVYyxXzW->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_abb_abb += 1/60 * einsum('Zxyz,UVYyxzWX->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_abb_abb -= 1/30 * einsum('Zxyz,UVYyzWXx->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_abb_abb += 1/15 * einsum('Zxyz,UVYyzWxX->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_abb_abb += 1/30 * einsum('Zxyz,UVYyzXWx->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_abb_abb += 7/60 * einsum('Zxyz,UVYyzXxW->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_abb_abb -= 1/60 * einsum('Zxyz,UVYyzxWX->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_abb_abb -= 1/30 * einsum('Zxyz,UVYyzxXW->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_abb_abb -= 1/6 * einsum('Wx,XY,UVZx->UVXWZY', h_aa, np.identity(ncas), rdm_ccaa, optimize = einsum_type)
    K22_abb_abb -= 1/3 * einsum('Wx,XY,UVxZ->UVXWZY', h_aa, np.identity(ncas), rdm_ccaa, optimize = einsum_type)
    K22_abb_abb -= 1/3 * einsum('Zx,XY,UVWx->UVXWZY', h_aa, np.identity(ncas), rdm_ccaa, optimize = einsum_type)
    K22_abb_abb -= 1/6 * einsum('Zx,XY,UVxW->UVXWZY', h_aa, np.identity(ncas), rdm_ccaa, optimize = einsum_type)
    K22_abb_abb -= 1/3 * einsum('XY,WxZy,UVxy->UVXWZY', np.identity(ncas), v_aaaa, rdm_ccaa, optimize = einsum_type)
    K22_abb_abb -= 1/6 * einsum('XY,WxZy,UVyx->UVXWZY', np.identity(ncas), v_aaaa, rdm_ccaa, optimize = einsum_type)
    K22_abb_abb -= 1/24 * einsum('XY,Wxyz,UVyZxz->UVXWZY', np.identity(ncas), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_abb_abb += 1/8 * einsum('XY,Wxyz,UVyZzx->UVXWZY', np.identity(ncas), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_abb_abb -= 5/24 * einsum('XY,Wxyz,UVyxZz->UVXWZY', np.identity(ncas), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_abb_abb += 1/8 * einsum('XY,Wxyz,UVyxzZ->UVXWZY', np.identity(ncas), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_abb_abb += 1/8 * einsum('XY,Wxyz,UVyzZx->UVXWZY', np.identity(ncas), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_abb_abb += 1/8 * einsum('XY,Wxyz,UVyzxZ->UVXWZY', np.identity(ncas), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_abb_abb -= 5/24 * einsum('XY,Zxyz,UVyWxz->UVXWZY', np.identity(ncas), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_abb_abb += 1/8 * einsum('XY,Zxyz,UVyWzx->UVXWZY', np.identity(ncas), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_abb_abb -= 1/24 * einsum('XY,Zxyz,UVyxWz->UVXWZY', np.identity(ncas), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_abb_abb += 1/8 * einsum('XY,Zxyz,UVyxzW->UVXWZY', np.identity(ncas), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_abb_abb += 1/8 * einsum('XY,Zxyz,UVyzWx->UVXWZY', np.identity(ncas), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_abb_abb += 1/8 * einsum('XY,Zxyz,UVyzxW->UVXWZY', np.identity(ncas), v_aaaa, rdm_cccaaa, optimize = einsum_type)

    K22_aaa_abb = np.ascontiguousarray(K22_aaa_aaa - K22_abb_abb.transpose(1,0,2,4,3,5) + K22_abb_abb.transpose(0,1,2,4,3,5))
    K22_abb_aaa = np.ascontiguousarray(K22_aaa_aaa - K22_abb_abb.transpose(1,0,2,4,3,5) + K22_abb_abb.transpose(1,0,2,3,4,5))

    # Reshape tensors to matrix form
    dim_x = ncas
    dim_wzy = ncas * ncas * ncas
    dim_tril_wzy = ncas * ncas * (ncas - 1) // 2

    dim_act = dim_x + dim_wzy + dim_tril_wzy

    tril_ind = np.tril_indices(ncas, k=-1)

    K12_a_aaa = K12_a_aaa[:, tril_ind[0], tril_ind[1]]

    K22_aaa_aaa = K22_aaa_aaa[:, :, :, tril_ind[0], tril_ind[1]]
    K22_aaa_aaa = K22_aaa_aaa[tril_ind[0], tril_ind[1]]

    K22_aaa_abb = K22_aaa_abb[tril_ind[0], tril_ind[1]]
    K22_abb_aaa = K22_abb_aaa[:, :, :, tril_ind[0], tril_ind[1]]

    K12_a_aaa = K12_a_aaa.reshape(dim_x, dim_tril_wzy)
    K12_a_abb = K12_a_abb.reshape(dim_x, dim_wzy)

    K22_aaa_aaa = K22_aaa_aaa.reshape(dim_tril_wzy, dim_tril_wzy)
    K22_aaa_abb = K22_aaa_abb.reshape(dim_tril_wzy, dim_wzy)

    K22_abb_aaa = K22_abb_aaa.reshape(dim_wzy, dim_tril_wzy)
    K22_abb_abb = K22_abb_abb.reshape(dim_wzy, dim_wzy)

    # Build K_m1p matrix
    s_a = 0
    f_a = dim_x
    s_aaa = f_a
    f_aaa = s_aaa + dim_tril_wzy
    s_abb = f_aaa
    f_abb = s_abb + dim_wzy

    K_m1p = np.zeros((dim_act, dim_act))

    K_m1p[s_a:f_a, s_a:f_a] = K11_a_a

    K_m1p[s_a:f_a, s_aaa:f_aaa] = K12_a_aaa
    K_m1p[s_a:f_a, s_abb:f_abb] = K12_a_abb

    K_m1p[s_aaa:f_aaa, s_a:f_a] = K12_a_aaa.T
    K_m1p[s_abb:f_abb, s_a:f_a] = K12_a_abb.T

    K_m1p[s_aaa:f_aaa, s_aaa:f_aaa] = K22_aaa_aaa
    K_m1p[s_aaa:f_aaa, s_abb:f_abb] = K22_aaa_abb

    K_m1p[s_abb:f_abb, s_aaa:f_aaa] = K22_abb_aaa
    K_m1p[s_abb:f_abb, s_abb:f_abb] = K22_abb_abb

    return K_m1p
