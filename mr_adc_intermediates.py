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
    K_ac += einsum('XxYy,xy->XY', v_aaaa, rdm_ca, optimize = einsum_type)
    K_ac -= 1/2 * einsum('XxyY,xy->XY', v_aaaa, rdm_ca, optimize = einsum_type)
    K_ac -= 1/2 * einsum('Yxyz,yzXx->XY', v_aaaa, rdm_ccaa, optimize = einsum_type)

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
    K_ca -= 1/2 * einsum('Yxyz,Xxyz->XY', v_aaaa, rdm_ccaa, optimize = einsum_type)

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
    K_aacc  = einsum('WZXY->XYWZ', v_aaaa, optimize = einsum_type).copy()
    K_aacc += einsum('WX,YZ->XYWZ', h_aa, np.identity(ncas), optimize = einsum_type)
    K_aacc += einsum('YZ,WX->XYWZ', h_aa, np.identity(ncas), optimize = einsum_type)
    K_aacc -= 1/2 * einsum('WX,ZY->XYWZ', h_aa, rdm_ca, optimize = einsum_type)
    K_aacc += 1/6 * einsum('Wx,XYZx->XYWZ', h_aa, rdm_ccaa, optimize = einsum_type)
    K_aacc += 1/3 * einsum('Wx,XYxZ->XYWZ', h_aa, rdm_ccaa, optimize = einsum_type)
    K_aacc -= 1/2 * einsum('YZ,WX->XYWZ', h_aa, rdm_ca, optimize = einsum_type)
    K_aacc += 1/3 * einsum('Zx,WxXY->XYWZ', h_aa, rdm_ccaa, optimize = einsum_type)
    K_aacc += 1/6 * einsum('Zx,WxYX->XYWZ', h_aa, rdm_ccaa, optimize = einsum_type)
    K_aacc -= 1/2 * einsum('WZXx,xY->XYWZ', v_aaaa, rdm_ca, optimize = einsum_type)
    K_aacc -= 1/2 * einsum('WZxY,xX->XYWZ', v_aaaa, rdm_ca, optimize = einsum_type)
    K_aacc += 1/3 * einsum('WZxy,XYxy->XYWZ', v_aaaa, rdm_ccaa, optimize = einsum_type)
    K_aacc += 1/6 * einsum('WZxy,XYyx->XYWZ', v_aaaa, rdm_ccaa, optimize = einsum_type)
    K_aacc -= 1/2 * einsum('WxXY,Zx->XYWZ', v_aaaa, rdm_ca, optimize = einsum_type)
    K_aacc -= 1/2 * einsum('WxXy,YxZy->XYWZ', v_aaaa, rdm_ccaa, optimize = einsum_type)
    K_aacc += 1/3 * einsum('WxyX,YxZy->XYWZ', v_aaaa, rdm_ccaa, optimize = einsum_type)
    K_aacc += 1/6 * einsum('WxyX,YxyZ->XYWZ', v_aaaa, rdm_ccaa, optimize = einsum_type)
    K_aacc += 1/6 * einsum('WxyY,XxZy->XYWZ', v_aaaa, rdm_ccaa, optimize = einsum_type)
    K_aacc += 1/3 * einsum('WxyY,XxyZ->XYWZ', v_aaaa, rdm_ccaa, optimize = einsum_type)
    K_aacc += 1/12 * einsum('Wxyz,ZyzXYx->XYWZ', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K_aacc -= 1/12 * einsum('Wxyz,ZyzXxY->XYWZ', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K_aacc += 1/4 * einsum('Wxyz,ZyzYXx->XYWZ', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K_aacc -= 1/12 * einsum('Wxyz,ZyzYxX->XYWZ', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K_aacc -= 1/12 * einsum('Wxyz,ZyzxXY->XYWZ', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K_aacc -= 1/12 * einsum('Wxyz,ZyzxYX->XYWZ', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K_aacc -= 1/2 * einsum('XYxZ,Wx->XYWZ', v_aaaa, rdm_ca, optimize = einsum_type)
    K_aacc += 1/6 * einsum('XxyZ,WxYy->XYWZ', v_aaaa, rdm_ccaa, optimize = einsum_type)
    K_aacc += 1/3 * einsum('XxyZ,WxyY->XYWZ', v_aaaa, rdm_ccaa, optimize = einsum_type)
    K_aacc -= 1/2 * einsum('YxZy,WxXy->XYWZ', v_aaaa, rdm_ccaa, optimize = einsum_type)
    K_aacc += 1/3 * einsum('YxyZ,WxXy->XYWZ', v_aaaa, rdm_ccaa, optimize = einsum_type)
    K_aacc += 1/6 * einsum('YxyZ,WxyX->XYWZ', v_aaaa, rdm_ccaa, optimize = einsum_type)
    K_aacc += 1/4 * einsum('Zxyz,WyzXYx->XYWZ', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K_aacc -= 1/12 * einsum('Zxyz,WyzXxY->XYWZ', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K_aacc += 1/12 * einsum('Zxyz,WyzYXx->XYWZ', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K_aacc -= 1/12 * einsum('Zxyz,WyzYxX->XYWZ', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K_aacc -= 1/12 * einsum('Zxyz,WyzxXY->XYWZ', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K_aacc -= 1/12 * einsum('Zxyz,WyzxYX->XYWZ', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K_aacc -= 1/2 * einsum('Wx,YZ,xX->XYWZ', h_aa, np.identity(ncas), rdm_ca, optimize = einsum_type)
    K_aacc -= 1/2 * einsum('Zx,WX,xY->XYWZ', h_aa, np.identity(ncas), rdm_ca, optimize = einsum_type)
    K_aacc += einsum('WX,YxZy,xy->XYWZ', np.identity(ncas), v_aaaa, rdm_ca, optimize = einsum_type)
    K_aacc -= 1/2 * einsum('WX,YxyZ,xy->XYWZ', np.identity(ncas), v_aaaa, rdm_ca, optimize = einsum_type)
    K_aacc -= 1/2 * einsum('WX,Zxyz,Yxyz->XYWZ', np.identity(ncas), v_aaaa, rdm_ccaa, optimize = einsum_type)
    K_aacc += einsum('YZ,WxXy,yx->XYWZ', np.identity(ncas), v_aaaa, rdm_ca, optimize = einsum_type)
    K_aacc -= 1/2 * einsum('YZ,WxyX,yx->XYWZ', np.identity(ncas), v_aaaa, rdm_ca, optimize = einsum_type)
    K_aacc -= 1/2 * einsum('YZ,Wxyz,Xxyz->XYWZ', np.identity(ncas), v_aaaa, rdm_ccaa, optimize = einsum_type)
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
    K_ccaa -= 1/3 * einsum('Zx,XYWx->XYWZ', h_aa, rdm_ccaa, optimize = einsum_type)
    K_ccaa -= 1/6 * einsum('Zx,XYxW->XYWZ', h_aa, rdm_ccaa, optimize = einsum_type)
    K_ccaa -= 1/3 * einsum('WZxy,XYxy->XYWZ', v_aaaa, rdm_ccaa, optimize = einsum_type)
    K_ccaa -= 1/6 * einsum('WZxy,XYyx->XYWZ', v_aaaa, rdm_ccaa, optimize = einsum_type)
    K_ccaa -= 1/12 * einsum('Wxyz,XYxZyz->XYWZ', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K_ccaa += 1/12 * einsum('Wxyz,XYxZzy->XYWZ', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K_ccaa -= 1/4 * einsum('Wxyz,XYxyZz->XYWZ', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K_ccaa += 1/12 * einsum('Wxyz,XYxyzZ->XYWZ', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K_ccaa += 1/12 * einsum('Wxyz,XYxzZy->XYWZ', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K_ccaa += 1/12 * einsum('Wxyz,XYxzyZ->XYWZ', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K_ccaa -= 1/4 * einsum('Zxyz,XYxWyz->XYWZ', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K_ccaa += 1/12 * einsum('Zxyz,XYxWzy->XYWZ', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K_ccaa -= 1/12 * einsum('Zxyz,XYxyWz->XYWZ', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K_ccaa += 1/12 * einsum('Zxyz,XYxyzW->XYWZ', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K_ccaa += 1/12 * einsum('Zxyz,XYxzWy->XYWZ', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K_ccaa += 1/12 * einsum('Zxyz,XYxzyW->XYWZ', v_aaaa, rdm_cccaaa, optimize = einsum_type)
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
    K_caca_aa_aa += 1/2 * einsum('YZ,XW->XYWZ', h_aa, rdm_ca, optimize = einsum_type)
    K_caca_aa_aa -= 1/6 * einsum('Zx,WYXx->XYWZ', h_aa, rdm_ccaa, optimize = einsum_type)
    K_caca_aa_aa += 1/6 * einsum('Zx,WYxX->XYWZ', h_aa, rdm_ccaa, optimize = einsum_type)
    K_caca_aa_aa += 1/6 * einsum('WYxy,XZxy->XYWZ', v_aaaa, rdm_ccaa, optimize = einsum_type)
    K_caca_aa_aa -= 1/6 * einsum('WYxy,XZyx->XYWZ', v_aaaa, rdm_ccaa, optimize = einsum_type)
    K_caca_aa_aa -= 1/6 * einsum('Wxyz,XZxYyz->XYWZ', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K_caca_aa_aa += 1/6 * einsum('Wxyz,XZxyYz->XYWZ', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K_caca_aa_aa += 1/2 * einsum('YxZy,WyXx->XYWZ', v_aaaa, rdm_ccaa, optimize = einsum_type)
    K_caca_aa_aa -= 1/6 * einsum('YxyZ,WyXx->XYWZ', v_aaaa, rdm_ccaa, optimize = einsum_type)
    K_caca_aa_aa += 1/6 * einsum('YxyZ,WyxX->XYWZ', v_aaaa, rdm_ccaa, optimize = einsum_type)
    K_caca_aa_aa -= 1/6 * einsum('Zxyz,XyzWYx->XYWZ', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K_caca_aa_aa += 1/6 * einsum('Zxyz,XyzYWx->XYWZ', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K_caca_aa_aa -= 1/2 * einsum('Wx,YZ,Xx->XYWZ', h_aa, np.identity(ncas), rdm_ca, optimize = einsum_type)
    K_caca_aa_aa -= 1/2 * einsum('YZ,Wxyz,Xxyz->XYWZ', np.identity(ncas), v_aaaa, rdm_ccaa, optimize = einsum_type)

    K_caca_aa_bb =- 1/3 * einsum('Wx,XZYx->XYWZ', h_aa, rdm_ccaa, optimize = einsum_type)
    K_caca_aa_bb -= 1/6 * einsum('Wx,XZxY->XYWZ', h_aa, rdm_ccaa, optimize = einsum_type)
    K_caca_aa_bb += 1/6 * einsum('Zx,WYXx->XYWZ', h_aa, rdm_ccaa, optimize = einsum_type)
    K_caca_aa_bb += 1/3 * einsum('Zx,WYxX->XYWZ', h_aa, rdm_ccaa, optimize = einsum_type)
    K_caca_aa_bb -= 1/6 * einsum('WYxy,XZxy->XYWZ', v_aaaa, rdm_ccaa, optimize = einsum_type)
    K_caca_aa_bb -= 1/3 * einsum('WYxy,XZyx->XYWZ', v_aaaa, rdm_ccaa, optimize = einsum_type)
    K_caca_aa_bb -= 1/4 * einsum('Wxyz,XZxYyz->XYWZ', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K_caca_aa_bb += 1/12 * einsum('Wxyz,XZxYzy->XYWZ', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K_caca_aa_bb -= 1/12 * einsum('Wxyz,XZxyYz->XYWZ', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K_caca_aa_bb += 1/12 * einsum('Wxyz,XZxyzY->XYWZ', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K_caca_aa_bb += 1/12 * einsum('Wxyz,XZxzYy->XYWZ', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K_caca_aa_bb += 1/12 * einsum('Wxyz,XZxzyY->XYWZ', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K_caca_aa_bb += 1/6 * einsum('YxyZ,WyXx->XYWZ', v_aaaa, rdm_ccaa, optimize = einsum_type)
    K_caca_aa_bb += 1/3 * einsum('YxyZ,WyxX->XYWZ', v_aaaa, rdm_ccaa, optimize = einsum_type)
    K_caca_aa_bb += 1/12 * einsum('Zxyz,XyzWYx->XYWZ', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K_caca_aa_bb -= 1/12 * einsum('Zxyz,XyzWxY->XYWZ', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K_caca_aa_bb += 1/4 * einsum('Zxyz,XyzYWx->XYWZ', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K_caca_aa_bb -= 1/12 * einsum('Zxyz,XyzYxW->XYWZ', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K_caca_aa_bb -= 1/12 * einsum('Zxyz,XyzxWY->XYWZ', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K_caca_aa_bb -= 1/12 * einsum('Zxyz,XyzxYW->XYWZ', v_aaaa, rdm_cccaaa, optimize = einsum_type)

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
    # K11 block: < Psi_0 | a_X [H_{act}, a^{\dag}_Y] | Psi_0>
    K11_a_a  = einsum('XY->XY', h_aa, optimize = einsum_type).copy()
    K11_a_a += einsum('XxYy,xy->XY', v_aaaa, rdm_ca, optimize = einsum_type)
    K11_a_a -= 1/2 * einsum('XxyY,xy->XY', v_aaaa, rdm_ca, optimize = einsum_type)
    K11_a_a -= 1/2 * einsum('Yxyz,Xxyz->XY', v_aaaa, rdm_ccaa, optimize = einsum_type)

    # K12 block: < Psi_0 | a_X [H_{act}, a^{\dag}_Y a^{\dag}_Z a_W] | Psi_0>
    K12_a_bba  = 1/3 * einsum('Wx,XxYZ->XWZY', h_aa, rdm_ccaa, optimize = einsum_type)
    K12_a_bba += 1/6 * einsum('Wx,XxZY->XWZY', h_aa, rdm_ccaa, optimize = einsum_type)
    K12_a_bba += 1/2 * einsum('XY,ZW->XWZY', h_aa, rdm_ca, optimize = einsum_type)
    K12_a_bba -= 1/3 * einsum('Yx,WXZx->XWZY', h_aa, rdm_ccaa, optimize = einsum_type)
    K12_a_bba -= 1/6 * einsum('Yx,WXxZ->XWZY', h_aa, rdm_ccaa, optimize = einsum_type)
    K12_a_bba -= 1/6 * einsum('Zx,WXYx->XWZY', h_aa, rdm_ccaa, optimize = einsum_type)
    K12_a_bba -= 1/3 * einsum('Zx,WXxY->XWZY', h_aa, rdm_ccaa, optimize = einsum_type)
    K12_a_bba += 1/6 * einsum('WXxy,YZxy->XWZY', v_aaaa, rdm_ccaa, optimize = einsum_type)
    K12_a_bba += 1/3 * einsum('WXxy,YZyx->XWZY', v_aaaa, rdm_ccaa, optimize = einsum_type)
    K12_a_bba += 1/4 * einsum('Wxyz,YZxXyz->XWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K12_a_bba -= 1/12 * einsum('Wxyz,YZxXzy->XWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K12_a_bba += 1/12 * einsum('Wxyz,YZxyXz->XWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K12_a_bba -= 1/12 * einsum('Wxyz,YZxyzX->XWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K12_a_bba -= 1/12 * einsum('Wxyz,YZxzXy->XWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K12_a_bba -= 1/12 * einsum('Wxyz,YZxzyX->XWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K12_a_bba += 1/2 * einsum('XxYZ,xW->XWZY', v_aaaa, rdm_ca, optimize = einsum_type)
    K12_a_bba += 1/2 * einsum('XxYy,WyZx->XWZY', v_aaaa, rdm_ccaa, optimize = einsum_type)
    K12_a_bba -= 1/3 * einsum('XxyY,WyZx->XWZY', v_aaaa, rdm_ccaa, optimize = einsum_type)
    K12_a_bba -= 1/6 * einsum('XxyY,WyxZ->XWZY', v_aaaa, rdm_ccaa, optimize = einsum_type)
    K12_a_bba -= 1/6 * einsum('XxyZ,WyYx->XWZY', v_aaaa, rdm_ccaa, optimize = einsum_type)
    K12_a_bba -= 1/3 * einsum('XxyZ,WyxY->XWZY', v_aaaa, rdm_ccaa, optimize = einsum_type)
    K12_a_bba -= 1/6 * einsum('YZxy,WXxy->XWZY', v_aaaa, rdm_ccaa, optimize = einsum_type)
    K12_a_bba -= 1/3 * einsum('YZxy,WXyx->XWZY', v_aaaa, rdm_ccaa, optimize = einsum_type)
    K12_a_bba -= 1/4 * einsum('Yxyz,ZyzWXx->XWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K12_a_bba += 1/12 * einsum('Yxyz,ZyzWxX->XWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K12_a_bba -= 1/12 * einsum('Yxyz,ZyzXWx->XWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K12_a_bba += 1/12 * einsum('Yxyz,ZyzXxW->XWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K12_a_bba += 1/12 * einsum('Yxyz,ZyzxWX->XWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K12_a_bba += 1/12 * einsum('Yxyz,ZyzxXW->XWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K12_a_bba -= 1/12 * einsum('Zxyz,YyzWXx->XWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K12_a_bba += 1/12 * einsum('Zxyz,YyzWxX->XWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K12_a_bba -= 1/4 * einsum('Zxyz,YyzXWx->XWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K12_a_bba += 1/12 * einsum('Zxyz,YyzXxW->XWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K12_a_bba += 1/12 * einsum('Zxyz,YyzxWX->XWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K12_a_bba += 1/12 * einsum('Zxyz,YyzxXW->XWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K12_a_bba -= 1/2 * einsum('Wx,XY,Zx->XWZY', h_aa, np.identity(ncas), rdm_ca, optimize = einsum_type)
    K12_a_bba += 1/2 * einsum('Zx,XY,xW->XWZY', h_aa, np.identity(ncas), rdm_ca, optimize = einsum_type)
    K12_a_bba -= 1/2 * einsum('XY,Wxyz,Zxyz->XWZY', np.identity(ncas), v_aaaa, rdm_ccaa, optimize = einsum_type)
    K12_a_bba += 1/2 * einsum('XY,Zxyz,Wxyz->XWZY', np.identity(ncas), v_aaaa, rdm_ccaa, optimize = einsum_type)

    K12_a_aaa = np.ascontiguousarray(K12_a_bba - K12_a_bba.transpose(0,1,3,2))

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
    K22_aaa_aaa -= 1/24 * einsum('VWxy,UYZXxy->UVXWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_aaa_aaa += 1/24 * einsum('VWxy,UYZXyx->UVXWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_aaa_aaa += 1/24 * einsum('VWxy,UYZxXy->UVXWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_aaa_aaa -= 1/24 * einsum('VWxy,UYZxyX->UVXWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_aaa_aaa -= 1/24 * einsum('VWxy,UYZyXx->UVXWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_aaa_aaa += 1/24 * einsum('VWxy,UYZyxX->UVXWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_aaa_aaa -= 1/2 * einsum('VXYZ,UW->UVXWZY', v_aaaa, rdm_ca, optimize = einsum_type)
    K22_aaa_aaa += 1/6 * einsum('VXYx,UZWx->UVXWZY', v_aaaa, rdm_ccaa, optimize = einsum_type)
    K22_aaa_aaa -= 1/6 * einsum('VXYx,UZxW->UVXWZY', v_aaaa, rdm_ccaa, optimize = einsum_type)
    K22_aaa_aaa += 1/2 * einsum('VXZY,UW->UVXWZY', v_aaaa, rdm_ca, optimize = einsum_type)
    K22_aaa_aaa -= 1/6 * einsum('VXZx,UYWx->UVXWZY', v_aaaa, rdm_ccaa, optimize = einsum_type)
    K22_aaa_aaa += 1/6 * einsum('VXZx,UYxW->UVXWZY', v_aaaa, rdm_ccaa, optimize = einsum_type)
    K22_aaa_aaa -= 1/6 * einsum('VXxY,UZWx->UVXWZY', v_aaaa, rdm_ccaa, optimize = einsum_type)
    K22_aaa_aaa += 1/6 * einsum('VXxY,UZxW->UVXWZY', v_aaaa, rdm_ccaa, optimize = einsum_type)
    K22_aaa_aaa += 1/6 * einsum('VXxZ,UYWx->UVXWZY', v_aaaa, rdm_ccaa, optimize = einsum_type)
    K22_aaa_aaa -= 1/6 * einsum('VXxZ,UYxW->UVXWZY', v_aaaa, rdm_ccaa, optimize = einsum_type)
    K22_aaa_aaa += 1/6 * einsum('VxYZ,UxWX->UVXWZY', v_aaaa, rdm_ccaa, optimize = einsum_type)
    K22_aaa_aaa -= 1/6 * einsum('VxYZ,UxXW->UVXWZY', v_aaaa, rdm_ccaa, optimize = einsum_type)
    K22_aaa_aaa += 1/6 * einsum('VxYy,UZxWXy->UVXWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_aaa_aaa -= 1/6 * einsum('VxYy,UZxXWy->UVXWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_aaa_aaa -= 1/6 * einsum('VxZY,UxWX->UVXWZY', v_aaaa, rdm_ccaa, optimize = einsum_type)
    K22_aaa_aaa += 1/6 * einsum('VxZY,UxXW->UVXWZY', v_aaaa, rdm_ccaa, optimize = einsum_type)
    K22_aaa_aaa -= 1/6 * einsum('VxZy,UYxWXy->UVXWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_aaa_aaa += 1/6 * einsum('VxZy,UYxXWy->UVXWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_aaa_aaa += 1/12 * einsum('VxyY,UZxWyX->UVXWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_aaa_aaa += 1/12 * einsum('VxyY,UZxXWy->UVXWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_aaa_aaa += 1/12 * einsum('VxyY,UZxyXW->UVXWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_aaa_aaa -= 1/12 * einsum('VxyZ,UYxWyX->UVXWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_aaa_aaa -= 1/12 * einsum('VxyZ,UYxXWy->UVXWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_aaa_aaa -= 1/12 * einsum('VxyZ,UYxyXW->UVXWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_aaa_aaa -= 1/24 * einsum('WXxy,UYZVxy->UVXWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_aaa_aaa += 1/24 * einsum('WXxy,UYZVyx->UVXWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_aaa_aaa += 1/24 * einsum('WXxy,UYZxVy->UVXWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_aaa_aaa -= 1/24 * einsum('WXxy,UYZxyV->UVXWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_aaa_aaa -= 1/24 * einsum('WXxy,UYZyVx->UVXWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_aaa_aaa += 1/24 * einsum('WXxy,UYZyxV->UVXWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_aaa_aaa -= 1/40 * einsum('Wxyz,UYZxVyXz->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aaa_aaa -= 1/30 * einsum('Wxyz,UYZxVyzX->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aaa_aaa += 11/120 * einsum('Wxyz,UYZxVzXy->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aaa_aaa -= 1/60 * einsum('Wxyz,UYZxXVyz->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aaa_aaa += 1/15 * einsum('Wxyz,UYZxXVzy->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aaa_aaa -= 1/120 * einsum('Wxyz,UYZxXyVz->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aaa_aaa -= 1/60 * einsum('Wxyz,UYZxXyzV->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aaa_aaa += 1/40 * einsum('Wxyz,UYZxXzVy->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aaa_aaa += 1/60 * einsum('Wxyz,UYZxXzyV->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aaa_aaa += 5/24 * einsum('Wxyz,UYZxyVXz->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aaa_aaa += 7/60 * einsum('Wxyz,UYZxyVzX->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aaa_aaa -= 1/120 * einsum('Wxyz,UYZxyXVz->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aaa_aaa += 1/15 * einsum('Wxyz,UYZxyXzV->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aaa_aaa += 3/40 * einsum('Wxyz,UYZxyzVX->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aaa_aaa += 19/120 * einsum('Wxyz,UYZxyzXV->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aaa_aaa += 1/8 * einsum('Wxyz,UYZxzVXy->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aaa_aaa += 1/30 * einsum('Wxyz,UYZxzVyX->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aaa_aaa -= 1/120 * einsum('Wxyz,UYZxzXVy->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aaa_aaa -= 1/60 * einsum('Wxyz,UYZxzXyV->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aaa_aaa -= 1/24 * einsum('Wxyz,UYZxzyVX->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aaa_aaa += 1/24 * einsum('Wxyz,UYZxzyXV->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aaa_aaa += 1/6 * einsum('XxYZ,UxVW->UVXWZY', v_aaaa, rdm_ccaa, optimize = einsum_type)
    K22_aaa_aaa -= 1/6 * einsum('XxYZ,UxWV->UVXWZY', v_aaaa, rdm_ccaa, optimize = einsum_type)
    K22_aaa_aaa += 1/6 * einsum('XxYy,UZxVWy->UVXWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_aaa_aaa -= 1/6 * einsum('XxYy,UZxWVy->UVXWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_aaa_aaa -= 1/6 * einsum('XxZY,UxVW->UVXWZY', v_aaaa, rdm_ccaa, optimize = einsum_type)
    K22_aaa_aaa += 1/6 * einsum('XxZY,UxWV->UVXWZY', v_aaaa, rdm_ccaa, optimize = einsum_type)
    K22_aaa_aaa -= 1/6 * einsum('XxZy,UYxVWy->UVXWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_aaa_aaa += 1/6 * einsum('XxZy,UYxWVy->UVXWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_aaa_aaa += 1/12 * einsum('XxyY,UZxVyW->UVXWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_aaa_aaa += 1/12 * einsum('XxyY,UZxWVy->UVXWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_aaa_aaa += 1/12 * einsum('XxyY,UZxyWV->UVXWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_aaa_aaa -= 1/12 * einsum('XxyZ,UYxVyW->UVXWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_aaa_aaa -= 1/12 * einsum('XxyZ,UYxWVy->UVXWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_aaa_aaa -= 1/12 * einsum('XxyZ,UYxyWV->UVXWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_aaa_aaa += 1/24 * einsum('YZxy,UxyVWX->UVXWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_aaa_aaa -= 1/24 * einsum('YZxy,UxyVXW->UVXWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_aaa_aaa -= 1/24 * einsum('YZxy,UxyWVX->UVXWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_aaa_aaa += 1/24 * einsum('YZxy,UxyWXV->UVXWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_aaa_aaa += 1/24 * einsum('YZxy,UxyXVW->UVXWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_aaa_aaa -= 1/24 * einsum('YZxy,UxyXWV->UVXWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_aaa_aaa += 1/30 * einsum('Yxyz,UZyzVXxW->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aaa_aaa -= 11/120 * einsum('Yxyz,UZyzVxWX->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aaa_aaa += 1/40 * einsum('Yxyz,UZyzVxXW->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aaa_aaa += 1/60 * einsum('Yxyz,UZyzWVXx->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aaa_aaa -= 1/15 * einsum('Yxyz,UZyzWVxX->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aaa_aaa -= 1/120 * einsum('Yxyz,UZyzWxVX->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aaa_aaa -= 1/120 * einsum('Yxyz,UZyzWxXV->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aaa_aaa -= 13/60 * einsum('Yxyz,UZyzXVWx->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aaa_aaa -= 1/10 * einsum('Yxyz,UZyzXVxW->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aaa_aaa += 1/60 * einsum('Yxyz,UZyzXWVx->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aaa_aaa -= 1/15 * einsum('Yxyz,UZyzXWxV->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aaa_aaa -= 1/24 * einsum('Yxyz,UZyzXxVW->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aaa_aaa -= 19/120 * einsum('Yxyz,UZyzXxWV->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aaa_aaa -= 17/120 * einsum('Yxyz,UZyzxVWX->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aaa_aaa -= 1/40 * einsum('Yxyz,UZyzxVXW->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aaa_aaa += 1/120 * einsum('Yxyz,UZyzxWVX->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aaa_aaa += 1/120 * einsum('Yxyz,UZyzxWXV->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aaa_aaa += 1/24 * einsum('Yxyz,UZyzxXVW->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aaa_aaa -= 3/40 * einsum('Yxyz,UZyzxXWV->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aaa_aaa -= 1/30 * einsum('Zxyz,UYyzVXxW->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aaa_aaa += 11/120 * einsum('Zxyz,UYyzVxWX->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aaa_aaa -= 1/40 * einsum('Zxyz,UYyzVxXW->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aaa_aaa -= 1/60 * einsum('Zxyz,UYyzWVXx->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aaa_aaa += 1/15 * einsum('Zxyz,UYyzWVxX->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aaa_aaa += 1/120 * einsum('Zxyz,UYyzWxVX->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aaa_aaa += 1/120 * einsum('Zxyz,UYyzWxXV->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aaa_aaa += 13/60 * einsum('Zxyz,UYyzXVWx->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aaa_aaa += 1/10 * einsum('Zxyz,UYyzXVxW->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aaa_aaa -= 1/60 * einsum('Zxyz,UYyzXWVx->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aaa_aaa += 1/15 * einsum('Zxyz,UYyzXWxV->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aaa_aaa += 1/24 * einsum('Zxyz,UYyzXxVW->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aaa_aaa += 19/120 * einsum('Zxyz,UYyzXxWV->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aaa_aaa += 17/120 * einsum('Zxyz,UYyzxVWX->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aaa_aaa += 1/40 * einsum('Zxyz,UYyzxVXW->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aaa_aaa -= 1/120 * einsum('Zxyz,UYyzxWVX->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aaa_aaa -= 1/120 * einsum('Zxyz,UYyzxWXV->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aaa_aaa -= 1/24 * einsum('Zxyz,UYyzxXVW->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aaa_aaa += 3/40 * einsum('Zxyz,UYyzxXWV->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
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
    K22_aaa_aaa -= 1/6 * einsum('VY,WXxy,UZxy->UVXWZY', np.identity(ncas), v_aaaa, rdm_ccaa, optimize = einsum_type)
    K22_aaa_aaa += 1/6 * einsum('VY,WXxy,UZyx->UVXWZY', np.identity(ncas), v_aaaa, rdm_ccaa, optimize = einsum_type)
    K22_aaa_aaa += 1/6 * einsum('VY,Wxyz,UZxXyz->UVXWZY', np.identity(ncas), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_aaa_aaa -= 1/6 * einsum('VY,Wxyz,UZxyXz->UVXWZY', np.identity(ncas), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_aaa_aaa -= 1/2 * einsum('VY,XxZy,UxWy->UVXWZY', np.identity(ncas), v_aaaa, rdm_ccaa, optimize = einsum_type)
    K22_aaa_aaa += 1/6 * einsum('VY,XxyZ,UxWy->UVXWZY', np.identity(ncas), v_aaaa, rdm_ccaa, optimize = einsum_type)
    K22_aaa_aaa -= 1/6 * einsum('VY,XxyZ,UxyW->UVXWZY', np.identity(ncas), v_aaaa, rdm_ccaa, optimize = einsum_type)
    K22_aaa_aaa += 1/6 * einsum('VY,Zxyz,UyzWXx->UVXWZY', np.identity(ncas), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_aaa_aaa -= 1/6 * einsum('VY,Zxyz,UyzXWx->UVXWZY', np.identity(ncas), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_aaa_aaa += 1/6 * einsum('VZ,WXxy,UYxy->UVXWZY', np.identity(ncas), v_aaaa, rdm_ccaa, optimize = einsum_type)
    K22_aaa_aaa -= 1/6 * einsum('VZ,WXxy,UYyx->UVXWZY', np.identity(ncas), v_aaaa, rdm_ccaa, optimize = einsum_type)
    K22_aaa_aaa -= 1/6 * einsum('VZ,Wxyz,UYxXyz->UVXWZY', np.identity(ncas), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_aaa_aaa += 1/6 * einsum('VZ,Wxyz,UYxyXz->UVXWZY', np.identity(ncas), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_aaa_aaa += 1/2 * einsum('VZ,XxYy,UxWy->UVXWZY', np.identity(ncas), v_aaaa, rdm_ccaa, optimize = einsum_type)
    K22_aaa_aaa -= 1/6 * einsum('VZ,XxyY,UxWy->UVXWZY', np.identity(ncas), v_aaaa, rdm_ccaa, optimize = einsum_type)
    K22_aaa_aaa += 1/6 * einsum('VZ,XxyY,UxyW->UVXWZY', np.identity(ncas), v_aaaa, rdm_ccaa, optimize = einsum_type)
    K22_aaa_aaa -= 1/6 * einsum('VZ,Yxyz,UyzWXx->UVXWZY', np.identity(ncas), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_aaa_aaa += 1/6 * einsum('VZ,Yxyz,UyzXWx->UVXWZY', np.identity(ncas), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_aaa_aaa -= 1/6 * einsum('XY,VWxy,UZxy->UVXWZY', np.identity(ncas), v_aaaa, rdm_ccaa, optimize = einsum_type)
    K22_aaa_aaa += 1/6 * einsum('XY,VWxy,UZyx->UVXWZY', np.identity(ncas), v_aaaa, rdm_ccaa, optimize = einsum_type)
    K22_aaa_aaa += 1/2 * einsum('XY,VxZy,UxWy->UVXWZY', np.identity(ncas), v_aaaa, rdm_ccaa, optimize = einsum_type)
    K22_aaa_aaa -= 1/6 * einsum('XY,VxyZ,UxWy->UVXWZY', np.identity(ncas), v_aaaa, rdm_ccaa, optimize = einsum_type)
    K22_aaa_aaa += 1/6 * einsum('XY,VxyZ,UxyW->UVXWZY', np.identity(ncas), v_aaaa, rdm_ccaa, optimize = einsum_type)
    K22_aaa_aaa -= 1/6 * einsum('XY,Wxyz,UZxVyz->UVXWZY', np.identity(ncas), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_aaa_aaa += 1/6 * einsum('XY,Wxyz,UZxyVz->UVXWZY', np.identity(ncas), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_aaa_aaa += 1/6 * einsum('XY,Zxyz,UyzVWx->UVXWZY', np.identity(ncas), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_aaa_aaa -= 1/6 * einsum('XY,Zxyz,UyzWVx->UVXWZY', np.identity(ncas), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_aaa_aaa += 1/6 * einsum('XZ,VWxy,UYxy->UVXWZY', np.identity(ncas), v_aaaa, rdm_ccaa, optimize = einsum_type)
    K22_aaa_aaa -= 1/6 * einsum('XZ,VWxy,UYyx->UVXWZY', np.identity(ncas), v_aaaa, rdm_ccaa, optimize = einsum_type)
    K22_aaa_aaa -= 1/2 * einsum('XZ,VxYy,UxWy->UVXWZY', np.identity(ncas), v_aaaa, rdm_ccaa, optimize = einsum_type)
    K22_aaa_aaa += 1/6 * einsum('XZ,VxyY,UxWy->UVXWZY', np.identity(ncas), v_aaaa, rdm_ccaa, optimize = einsum_type)
    K22_aaa_aaa -= 1/6 * einsum('XZ,VxyY,UxyW->UVXWZY', np.identity(ncas), v_aaaa, rdm_ccaa, optimize = einsum_type)
    K22_aaa_aaa += 1/6 * einsum('XZ,Wxyz,UYxVyz->UVXWZY', np.identity(ncas), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_aaa_aaa -= 1/6 * einsum('XZ,Wxyz,UYxyVz->UVXWZY', np.identity(ncas), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_aaa_aaa -= 1/6 * einsum('XZ,Yxyz,UyzVWx->UVXWZY', np.identity(ncas), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_aaa_aaa += 1/6 * einsum('XZ,Yxyz,UyzWVx->UVXWZY', np.identity(ncas), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_aaa_aaa += 1/2 * einsum('Wx,VY,XZ,Ux->UVXWZY', h_aa, np.identity(ncas), np.identity(ncas), rdm_ca, optimize = einsum_type)
    K22_aaa_aaa -= 1/2 * einsum('Wx,VZ,XY,Ux->UVXWZY', h_aa, np.identity(ncas), np.identity(ncas), rdm_ca, optimize = einsum_type)
    K22_aaa_aaa += 1/2 * einsum('Wxyz,VY,XZ,Uxyz->UVXWZY', v_aaaa, np.identity(ncas), np.identity(ncas), rdm_ccaa, optimize = einsum_type)
    K22_aaa_aaa -= 1/2 * einsum('Wxyz,VZ,XY,Uxyz->UVXWZY', v_aaaa, np.identity(ncas), np.identity(ncas), rdm_ccaa, optimize = einsum_type)

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
    K22_bba_bba += 1/24 * einsum('VWxy,UYZXxy->UVXWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_bba_bba -= 1/24 * einsum('VWxy,UYZXyx->UVXWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_bba_bba += 1/8 * einsum('VWxy,UYZxXy->UVXWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_bba_bba += 1/24 * einsum('VWxy,UYZxyX->UVXWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_bba_bba -= 1/8 * einsum('VWxy,UYZyXx->UVXWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_bba_bba -= 1/24 * einsum('VWxy,UYZyxX->UVXWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_bba_bba += 1/2 * einsum('VXZY,UW->UVXWZY', v_aaaa, rdm_ca, optimize = einsum_type)
    K22_bba_bba -= 1/3 * einsum('VXZx,UYWx->UVXWZY', v_aaaa, rdm_ccaa, optimize = einsum_type)
    K22_bba_bba -= 1/6 * einsum('VXZx,UYxW->UVXWZY', v_aaaa, rdm_ccaa, optimize = einsum_type)
    K22_bba_bba -= 1/6 * einsum('VXxY,UZWx->UVXWZY', v_aaaa, rdm_ccaa, optimize = einsum_type)
    K22_bba_bba += 1/6 * einsum('VXxY,UZxW->UVXWZY', v_aaaa, rdm_ccaa, optimize = einsum_type)
    K22_bba_bba -= 1/3 * einsum('VxZY,UxWX->UVXWZY', v_aaaa, rdm_ccaa, optimize = einsum_type)
    K22_bba_bba -= 1/6 * einsum('VxZY,UxXW->UVXWZY', v_aaaa, rdm_ccaa, optimize = einsum_type)
    K22_bba_bba -= 1/3 * einsum('VxZy,UYxWXy->UVXWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_bba_bba -= 1/6 * einsum('VxZy,UYxXWy->UVXWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_bba_bba += 1/12 * einsum('VxyY,UZxWyX->UVXWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_bba_bba -= 1/12 * einsum('VxyY,UZxXWy->UVXWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_bba_bba -= 1/6 * einsum('VxyY,UZxyWX->UVXWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_bba_bba -= 1/12 * einsum('VxyY,UZxyXW->UVXWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_bba_bba += 1/6 * einsum('VxyZ,UYxWXy->UVXWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_bba_bba += 1/12 * einsum('VxyZ,UYxWyX->UVXWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_bba_bba += 1/12 * einsum('VxyZ,UYxXWy->UVXWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_bba_bba -= 1/12 * einsum('VxyZ,UYxyXW->UVXWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_bba_bba += 1/24 * einsum('WXxy,UYZVxy->UVXWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_bba_bba += 1/8 * einsum('WXxy,UYZVyx->UVXWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_bba_bba -= 1/24 * einsum('WXxy,UYZxVy->UVXWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_bba_bba -= 1/8 * einsum('WXxy,UYZxyV->UVXWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_bba_bba += 1/24 * einsum('WXxy,UYZyVx->UVXWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_bba_bba -= 1/24 * einsum('WXxy,UYZyxV->UVXWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_bba_bba -= 1/30 * einsum('Wxyz,UYZxVyXz->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_bba_bba += 1/12 * einsum('Wxyz,UYZxVyzX->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_bba_bba -= 1/60 * einsum('Wxyz,UYZxVzXy->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_bba_bba += 1/60 * einsum('Wxyz,UYZxVzyX->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_bba_bba -= 1/40 * einsum('Wxyz,UYZxXVyz->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_bba_bba += 7/120 * einsum('Wxyz,UYZxXVzy->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_bba_bba += 1/20 * einsum('Wxyz,UYZxXyVz->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_bba_bba += 1/15 * einsum('Wxyz,UYZxXyzV->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_bba_bba += 3/20 * einsum('Wxyz,UYZxXzVy->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_bba_bba -= 1/8 * einsum('Wxyz,UYZxyVXz->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_bba_bba += 3/40 * einsum('Wxyz,UYZxyVzX->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_bba_bba -= 1/10 * einsum('Wxyz,UYZxyXVz->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_bba_bba += 1/6 * einsum('Wxyz,UYZxyzVX->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_bba_bba -= 1/60 * einsum('Wxyz,UYZxyzXV->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_bba_bba += 1/24 * einsum('Wxyz,UYZxzVXy->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_bba_bba += 3/40 * einsum('Wxyz,UYZxzVyX->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_bba_bba += 3/20 * einsum('Wxyz,UYZxzXVy->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_bba_bba += 7/30 * einsum('Wxyz,UYZxzyVX->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_bba_bba += 1/20 * einsum('Wxyz,UYZxzyXV->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_bba_bba += 1/6 * einsum('XxYZ,UxVW->UVXWZY', v_aaaa, rdm_ccaa, optimize = einsum_type)
    K22_bba_bba -= 1/6 * einsum('XxYZ,UxWV->UVXWZY', v_aaaa, rdm_ccaa, optimize = einsum_type)
    K22_bba_bba += 1/6 * einsum('XxYy,UZxVWy->UVXWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_bba_bba -= 1/6 * einsum('XxYy,UZxWVy->UVXWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_bba_bba -= 1/6 * einsum('XxyY,UZxVWy->UVXWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_bba_bba -= 1/12 * einsum('XxyY,UZxVyW->UVXWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_bba_bba += 1/12 * einsum('XxyY,UZxWVy->UVXWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_bba_bba -= 1/12 * einsum('XxyY,UZxyWV->UVXWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_bba_bba -= 1/12 * einsum('XxyZ,UYxVyW->UVXWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_bba_bba += 1/12 * einsum('XxyZ,UYxWVy->UVXWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_bba_bba += 1/6 * einsum('XxyZ,UYxWyV->UVXWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_bba_bba += 1/12 * einsum('XxyZ,UYxyWV->UVXWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_bba_bba -= 1/24 * einsum('YZxy,UxyVWX->UVXWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_bba_bba -= 1/8 * einsum('YZxy,UxyVXW->UVXWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_bba_bba += 1/24 * einsum('YZxy,UxyWVX->UVXWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_bba_bba += 1/8 * einsum('YZxy,UxyWXV->UVXWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_bba_bba -= 1/24 * einsum('YZxy,UxyXVW->UVXWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_bba_bba += 1/24 * einsum('YZxy,UxyXWV->UVXWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_bba_bba -= 1/40 * einsum('Yxyz,UZyzVXWx->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_bba_bba -= 1/24 * einsum('Yxyz,UZyzVXxW->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_bba_bba -= 1/60 * einsum('Yxyz,UZyzVxWX->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_bba_bba += 1/20 * einsum('Yxyz,UZyzVxXW->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_bba_bba += 1/10 * einsum('Yxyz,UZyzWVXx->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_bba_bba -= 3/20 * einsum('Yxyz,UZyzWVxX->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_bba_bba += 1/40 * einsum('Yxyz,UZyzWXVx->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_bba_bba -= 1/24 * einsum('Yxyz,UZyzWXxV->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_bba_bba -= 1/20 * einsum('Yxyz,UZyzWxVX->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_bba_bba += 1/20 * einsum('Yxyz,UZyzWxXV->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_bba_bba -= 1/30 * einsum('Yxyz,UZyzXVWx->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_bba_bba -= 2/15 * einsum('Yxyz,UZyzXVxW->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_bba_bba += 1/60 * einsum('Yxyz,UZyzXWxV->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_bba_bba -= 1/30 * einsum('Yxyz,UZyzXxVW->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_bba_bba -= 23/120 * einsum('Yxyz,UZyzxVWX->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_bba_bba -= 1/8 * einsum('Yxyz,UZyzxVXW->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_bba_bba -= 3/40 * einsum('Yxyz,UZyzxWVX->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_bba_bba += 1/40 * einsum('Yxyz,UZyzxWXV->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_bba_bba -= 7/60 * einsum('Yxyz,UZyzxXVW->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_bba_bba -= 1/12 * einsum('Yxyz,UZyzxXWV->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_bba_bba -= 1/60 * einsum('Zxyz,UYyzVWXx->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_bba_bba -= 1/12 * einsum('Zxyz,UYyzVWxX->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_bba_bba += 1/30 * einsum('Zxyz,UYyzVxWX->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_bba_bba += 1/60 * einsum('Zxyz,UYyzVxXW->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_bba_bba += 1/20 * einsum('Zxyz,UYyzWVXx->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_bba_bba -= 1/10 * einsum('Zxyz,UYyzWVxX->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_bba_bba -= 13/60 * einsum('Zxyz,UYyzWxVX->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_bba_bba += 1/60 * einsum('Zxyz,UYyzWxXV->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_bba_bba += 1/40 * einsum('Zxyz,UYyzXVWx->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_bba_bba -= 7/120 * einsum('Zxyz,UYyzXVxW->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_bba_bba -= 1/8 * einsum('Zxyz,UYyzXWVx->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_bba_bba -= 1/24 * einsum('Zxyz,UYyzXWxV->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_bba_bba -= 7/40 * einsum('Zxyz,UYyzXxVW->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_bba_bba += 3/40 * einsum('Zxyz,UYyzXxWV->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_bba_bba -= 1/60 * einsum('Zxyz,UYyzxVXW->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_bba_bba -= 7/30 * einsum('Zxyz,UYyzxWVX->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_bba_bba -= 3/20 * einsum('Zxyz,UYyzxXVW->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_bba_bba += 1/10 * einsum('Zxyz,UYyzxXWV->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
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
    K22_bba_bba += 1/3 * einsum('VZ,WXxy,UYxy->UVXWZY', np.identity(ncas), v_aaaa, rdm_ccaa, optimize = einsum_type)
    K22_bba_bba += 1/6 * einsum('VZ,WXxy,UYyx->UVXWZY', np.identity(ncas), v_aaaa, rdm_ccaa, optimize = einsum_type)
    K22_bba_bba += 1/12 * einsum('VZ,Wxyz,UYxXyz->UVXWZY', np.identity(ncas), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_bba_bba -= 1/12 * einsum('VZ,Wxyz,UYxXzy->UVXWZY', np.identity(ncas), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_bba_bba += 1/4 * einsum('VZ,Wxyz,UYxyXz->UVXWZY', np.identity(ncas), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_bba_bba -= 1/12 * einsum('VZ,Wxyz,UYxyzX->UVXWZY', np.identity(ncas), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_bba_bba -= 1/12 * einsum('VZ,Wxyz,UYxzXy->UVXWZY', np.identity(ncas), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_bba_bba -= 1/12 * einsum('VZ,Wxyz,UYxzyX->UVXWZY', np.identity(ncas), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_bba_bba += 1/2 * einsum('VZ,XxYy,UxWy->UVXWZY', np.identity(ncas), v_aaaa, rdm_ccaa, optimize = einsum_type)
    K22_bba_bba -= 1/3 * einsum('VZ,XxyY,UxWy->UVXWZY', np.identity(ncas), v_aaaa, rdm_ccaa, optimize = einsum_type)
    K22_bba_bba -= 1/6 * einsum('VZ,XxyY,UxyW->UVXWZY', np.identity(ncas), v_aaaa, rdm_ccaa, optimize = einsum_type)
    K22_bba_bba -= 1/4 * einsum('VZ,Yxyz,UyzWXx->UVXWZY', np.identity(ncas), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_bba_bba += 1/12 * einsum('VZ,Yxyz,UyzWxX->UVXWZY', np.identity(ncas), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_bba_bba -= 1/12 * einsum('VZ,Yxyz,UyzXWx->UVXWZY', np.identity(ncas), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_bba_bba += 1/12 * einsum('VZ,Yxyz,UyzXxW->UVXWZY', np.identity(ncas), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_bba_bba += 1/12 * einsum('VZ,Yxyz,UyzxWX->UVXWZY', np.identity(ncas), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_bba_bba += 1/12 * einsum('VZ,Yxyz,UyzxXW->UVXWZY', np.identity(ncas), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_bba_bba -= 1/6 * einsum('XY,VWxy,UZxy->UVXWZY', np.identity(ncas), v_aaaa, rdm_ccaa, optimize = einsum_type)
    K22_bba_bba += 1/6 * einsum('XY,VWxy,UZyx->UVXWZY', np.identity(ncas), v_aaaa, rdm_ccaa, optimize = einsum_type)
    K22_bba_bba += 1/2 * einsum('XY,VxZy,UxWy->UVXWZY', np.identity(ncas), v_aaaa, rdm_ccaa, optimize = einsum_type)
    K22_bba_bba -= 1/6 * einsum('XY,VxyZ,UxWy->UVXWZY', np.identity(ncas), v_aaaa, rdm_ccaa, optimize = einsum_type)
    K22_bba_bba += 1/6 * einsum('XY,VxyZ,UxyW->UVXWZY', np.identity(ncas), v_aaaa, rdm_ccaa, optimize = einsum_type)
    K22_bba_bba -= 1/6 * einsum('XY,Wxyz,UZxVyz->UVXWZY', np.identity(ncas), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_bba_bba += 1/6 * einsum('XY,Wxyz,UZxyVz->UVXWZY', np.identity(ncas), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_bba_bba += 1/6 * einsum('XY,Zxyz,UyzVWx->UVXWZY', np.identity(ncas), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_bba_bba -= 1/6 * einsum('XY,Zxyz,UyzWVx->UVXWZY', np.identity(ncas), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_bba_bba -= 1/2 * einsum('Wx,VZ,XY,Ux->UVXWZY', h_aa, np.identity(ncas), np.identity(ncas), rdm_ca, optimize = einsum_type)
    K22_bba_bba -= 1/2 * einsum('Wxyz,VZ,XY,Uxyz->UVXWZY', v_aaaa, np.identity(ncas), np.identity(ncas), rdm_ccaa, optimize = einsum_type)

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
    # K11 block: < Psi_0 | a^{\dag}_X [H_{act}, a_Y] | Psi_0>
    K11_a_a =- 1/2 * einsum('Yx,Xx->XY', h_aa, rdm_ca, optimize = einsum_type)
    K11_a_a -= 1/2 * einsum('Yxyz,Xxyz->XY', v_aaaa, rdm_ccaa, optimize = einsum_type)

    # K12 block: < Psi_0 | a^{\dag}_X [H_{act}, a^{\dag}_Y a_Z a_W] | Psi_0>
    K12_a_abb =- 1/6 * einsum('Wx,XYZx->XWZY', h_aa, rdm_ccaa, optimize = einsum_type)
    K12_a_abb -= 1/3 * einsum('Wx,XYxZ->XWZY', h_aa, rdm_ccaa, optimize = einsum_type)
    K12_a_abb += 1/3 * einsum('Yx,WZXx->XWZY', h_aa, rdm_ccaa, optimize = einsum_type)
    K12_a_abb += 1/6 * einsum('Yx,WZxX->XWZY', h_aa, rdm_ccaa, optimize = einsum_type)
    K12_a_abb -= 1/3 * einsum('Zx,WxXY->XWZY', h_aa, rdm_ccaa, optimize = einsum_type)
    K12_a_abb -= 1/6 * einsum('Zx,WxYX->XWZY', h_aa, rdm_ccaa, optimize = einsum_type)
    K12_a_abb -= 1/3 * einsum('WZxy,XYxy->XWZY', v_aaaa, rdm_ccaa, optimize = einsum_type)
    K12_a_abb -= 1/6 * einsum('WZxy,XYyx->XWZY', v_aaaa, rdm_ccaa, optimize = einsum_type)
    K12_a_abb -= 1/12 * einsum('Wxyz,XYxZyz->XWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K12_a_abb += 1/12 * einsum('Wxyz,XYxZzy->XWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K12_a_abb -= 1/4 * einsum('Wxyz,XYxyZz->XWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K12_a_abb += 1/12 * einsum('Wxyz,XYxyzZ->XWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K12_a_abb += 1/12 * einsum('Wxyz,XYxzZy->XWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K12_a_abb += 1/12 * einsum('Wxyz,XYxzyZ->XWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K12_a_abb += 1/4 * einsum('Yxyz,XyzWZx->XWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K12_a_abb -= 1/12 * einsum('Yxyz,XyzWxZ->XWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K12_a_abb += 1/12 * einsum('Yxyz,XyzZWx->XWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K12_a_abb -= 1/12 * einsum('Yxyz,XyzZxW->XWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K12_a_abb -= 1/12 * einsum('Yxyz,XyzxWZ->XWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K12_a_abb -= 1/12 * einsum('Yxyz,XyzxZW->XWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K12_a_abb -= 1/4 * einsum('Zxyz,XYxWyz->XWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K12_a_abb += 1/12 * einsum('Zxyz,XYxWzy->XWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K12_a_abb -= 1/12 * einsum('Zxyz,XYxyWz->XWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K12_a_abb += 1/12 * einsum('Zxyz,XYxyzW->XWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K12_a_abb += 1/12 * einsum('Zxyz,XYxzWy->XWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K12_a_abb += 1/12 * einsum('Zxyz,XYxzyW->XWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)

    K12_a_aaa = np.ascontiguousarray(K12_a_abb - K12_a_abb.transpose(0,2,1,3))

    # K22 block: < Psi_0 | a^{\dag}_U a^{\dag}_V a_X [H_{act}, a^{\dag}_Y a_Z a_W] | Psi_0>
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
    K22_aaa_aaa -= 1/24 * einsum('WXxy,UVYZxy->UVXWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_aaa_aaa += 1/24 * einsum('WXxy,UVYZyx->UVXWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_aaa_aaa += 1/24 * einsum('WXxy,UVYxZy->UVXWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_aaa_aaa -= 1/24 * einsum('WXxy,UVYxyZ->UVXWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_aaa_aaa -= 1/24 * einsum('WXxy,UVYyZx->UVXWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_aaa_aaa += 1/24 * einsum('WXxy,UVYyxZ->UVXWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_aaa_aaa += 1/24 * einsum('WZxy,UVYXxy->UVXWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_aaa_aaa -= 1/24 * einsum('WZxy,UVYXyx->UVXWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_aaa_aaa -= 1/24 * einsum('WZxy,UVYxXy->UVXWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_aaa_aaa += 1/24 * einsum('WZxy,UVYxyX->UVXWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_aaa_aaa += 1/24 * einsum('WZxy,UVYyXx->UVXWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_aaa_aaa -= 1/24 * einsum('WZxy,UVYyxX->UVXWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_aaa_aaa += 1/40 * einsum('Wxyz,UVYxXyZz->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aaa_aaa += 1/30 * einsum('Wxyz,UVYxXyzZ->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aaa_aaa -= 11/120 * einsum('Wxyz,UVYxXzZy->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aaa_aaa += 1/60 * einsum('Wxyz,UVYxZXyz->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aaa_aaa -= 1/15 * einsum('Wxyz,UVYxZXzy->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aaa_aaa += 1/120 * einsum('Wxyz,UVYxZyXz->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aaa_aaa += 1/60 * einsum('Wxyz,UVYxZyzX->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aaa_aaa -= 1/40 * einsum('Wxyz,UVYxZzXy->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aaa_aaa -= 1/60 * einsum('Wxyz,UVYxZzyX->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aaa_aaa -= 5/24 * einsum('Wxyz,UVYxyXZz->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aaa_aaa -= 7/60 * einsum('Wxyz,UVYxyXzZ->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aaa_aaa += 1/120 * einsum('Wxyz,UVYxyZXz->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aaa_aaa -= 1/15 * einsum('Wxyz,UVYxyZzX->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aaa_aaa -= 3/40 * einsum('Wxyz,UVYxyzXZ->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aaa_aaa -= 19/120 * einsum('Wxyz,UVYxyzZX->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aaa_aaa -= 1/8 * einsum('Wxyz,UVYxzXZy->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aaa_aaa -= 1/30 * einsum('Wxyz,UVYxzXyZ->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aaa_aaa += 1/120 * einsum('Wxyz,UVYxzZXy->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aaa_aaa += 1/60 * einsum('Wxyz,UVYxzZyX->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aaa_aaa += 1/24 * einsum('Wxyz,UVYxzyXZ->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aaa_aaa -= 1/24 * einsum('Wxyz,UVYxzyZX->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aaa_aaa -= 1/24 * einsum('XZxy,UVYWxy->UVXWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_aaa_aaa += 1/24 * einsum('XZxy,UVYWyx->UVXWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_aaa_aaa += 1/24 * einsum('XZxy,UVYxWy->UVXWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_aaa_aaa -= 1/24 * einsum('XZxy,UVYxyW->UVXWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_aaa_aaa -= 1/24 * einsum('XZxy,UVYyWx->UVXWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_aaa_aaa += 1/24 * einsum('XZxy,UVYyxW->UVXWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_aaa_aaa += 1/6 * einsum('XxYy,UVxWZy->UVXWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_aaa_aaa -= 1/6 * einsum('XxYy,UVxZWy->UVXWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_aaa_aaa += 1/12 * einsum('XxyY,UVxWyZ->UVXWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_aaa_aaa += 1/12 * einsum('XxyY,UVxZWy->UVXWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_aaa_aaa += 1/12 * einsum('XxyY,UVxyZW->UVXWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_aaa_aaa -= 1/30 * einsum('Yxyz,UVyzWZxX->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aaa_aaa += 11/120 * einsum('Yxyz,UVyzWxXZ->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aaa_aaa -= 1/40 * einsum('Yxyz,UVyzWxZX->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aaa_aaa -= 1/60 * einsum('Yxyz,UVyzXWZx->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aaa_aaa += 1/15 * einsum('Yxyz,UVyzXWxZ->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aaa_aaa += 1/120 * einsum('Yxyz,UVyzXxWZ->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aaa_aaa += 1/120 * einsum('Yxyz,UVyzXxZW->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aaa_aaa += 13/60 * einsum('Yxyz,UVyzZWXx->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aaa_aaa += 1/10 * einsum('Yxyz,UVyzZWxX->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aaa_aaa -= 1/60 * einsum('Yxyz,UVyzZXWx->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aaa_aaa += 1/15 * einsum('Yxyz,UVyzZXxW->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aaa_aaa += 1/24 * einsum('Yxyz,UVyzZxWX->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aaa_aaa += 19/120 * einsum('Yxyz,UVyzZxXW->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aaa_aaa += 17/120 * einsum('Yxyz,UVyzxWXZ->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aaa_aaa += 1/40 * einsum('Yxyz,UVyzxWZX->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aaa_aaa -= 1/120 * einsum('Yxyz,UVyzxXWZ->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aaa_aaa -= 1/120 * einsum('Yxyz,UVyzxXZW->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aaa_aaa -= 1/24 * einsum('Yxyz,UVyzxZWX->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aaa_aaa += 3/40 * einsum('Yxyz,UVyzxZXW->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aaa_aaa += 1/40 * einsum('Zxyz,UVYxWyXz->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aaa_aaa += 1/30 * einsum('Zxyz,UVYxWyzX->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aaa_aaa -= 11/120 * einsum('Zxyz,UVYxWzXy->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aaa_aaa += 1/60 * einsum('Zxyz,UVYxXWyz->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aaa_aaa -= 1/15 * einsum('Zxyz,UVYxXWzy->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aaa_aaa += 1/120 * einsum('Zxyz,UVYxXyWz->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aaa_aaa += 1/60 * einsum('Zxyz,UVYxXyzW->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aaa_aaa -= 1/40 * einsum('Zxyz,UVYxXzWy->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aaa_aaa -= 1/60 * einsum('Zxyz,UVYxXzyW->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aaa_aaa -= 5/24 * einsum('Zxyz,UVYxyWXz->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aaa_aaa -= 7/60 * einsum('Zxyz,UVYxyWzX->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aaa_aaa += 1/120 * einsum('Zxyz,UVYxyXWz->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aaa_aaa -= 1/15 * einsum('Zxyz,UVYxyXzW->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aaa_aaa -= 3/40 * einsum('Zxyz,UVYxyzWX->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aaa_aaa -= 19/120 * einsum('Zxyz,UVYxyzXW->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aaa_aaa -= 1/8 * einsum('Zxyz,UVYxzWXy->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aaa_aaa -= 1/30 * einsum('Zxyz,UVYxzWyX->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aaa_aaa += 1/120 * einsum('Zxyz,UVYxzXWy->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aaa_aaa += 1/60 * einsum('Zxyz,UVYxzXyW->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aaa_aaa += 1/24 * einsum('Zxyz,UVYxzyWX->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aaa_aaa -= 1/24 * einsum('Zxyz,UVYxzyXW->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aaa_aaa += 1/6 * einsum('Wx,XY,UVZx->UVXWZY', h_aa, np.identity(ncas), rdm_ccaa, optimize = einsum_type)
    K22_aaa_aaa -= 1/6 * einsum('Wx,XY,UVxZ->UVXWZY', h_aa, np.identity(ncas), rdm_ccaa, optimize = einsum_type)
    K22_aaa_aaa -= 1/6 * einsum('Zx,XY,UVWx->UVXWZY', h_aa, np.identity(ncas), rdm_ccaa, optimize = einsum_type)
    K22_aaa_aaa += 1/6 * einsum('Zx,XY,UVxW->UVXWZY', h_aa, np.identity(ncas), rdm_ccaa, optimize = einsum_type)
    K22_aaa_aaa -= 1/6 * einsum('XY,WZxy,UVxy->UVXWZY', np.identity(ncas), v_aaaa, rdm_ccaa, optimize = einsum_type)
    K22_aaa_aaa += 1/6 * einsum('XY,WZxy,UVyx->UVXWZY', np.identity(ncas), v_aaaa, rdm_ccaa, optimize = einsum_type)
    K22_aaa_aaa += 1/6 * einsum('XY,Wxyz,UVxZyz->UVXWZY', np.identity(ncas), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_aaa_aaa -= 1/6 * einsum('XY,Wxyz,UVxyZz->UVXWZY', np.identity(ncas), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_aaa_aaa -= 1/6 * einsum('XY,Zxyz,UVxWyz->UVXWZY', np.identity(ncas), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_aaa_aaa += 1/6 * einsum('XY,Zxyz,UVxyWz->UVXWZY', np.identity(ncas), v_aaaa, rdm_cccaaa, optimize = einsum_type)

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
    K22_abb_abb += 1/24 * einsum('WXxy,UVYZxy->UVXWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_abb_abb -= 1/24 * einsum('WXxy,UVYZyx->UVXWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_abb_abb += 1/8 * einsum('WXxy,UVYxZy->UVXWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_abb_abb -= 1/8 * einsum('WXxy,UVYxyZ->UVXWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_abb_abb += 1/24 * einsum('WXxy,UVYyZx->UVXWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_abb_abb -= 1/24 * einsum('WXxy,UVYyxZ->UVXWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_abb_abb -= 1/24 * einsum('WZxy,UVYXxy->UVXWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_abb_abb += 1/24 * einsum('WZxy,UVYXyx->UVXWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_abb_abb -= 1/8 * einsum('WZxy,UVYxXy->UVXWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_abb_abb += 1/8 * einsum('WZxy,UVYxyX->UVXWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_abb_abb -= 1/24 * einsum('WZxy,UVYyXx->UVXWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_abb_abb += 1/24 * einsum('WZxy,UVYyxX->UVXWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_abb_abb += 3/40 * einsum('Wxyz,UVYxXZyz->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_abb_abb -= 3/40 * einsum('Wxyz,UVYxXZzy->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_abb_abb += 1/40 * einsum('Wxyz,UVYxXyZz->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_abb_abb -= 1/12 * einsum('Wxyz,UVYxXyzZ->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_abb_abb -= 3/40 * einsum('Wxyz,UVYxXzZy->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_abb_abb -= 7/60 * einsum('Wxyz,UVYxXzyZ->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_abb_abb += 1/20 * einsum('Wxyz,UVYxZXyz->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_abb_abb -= 1/60 * einsum('Wxyz,UVYxZXzy->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_abb_abb += 1/60 * einsum('Wxyz,UVYxZyXz->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_abb_abb -= 1/6 * einsum('Wxyz,UVYxZzXy->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_abb_abb -= 1/30 * einsum('Wxyz,UVYxZzyX->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_abb_abb -= 1/40 * einsum('Wxyz,UVYxyXzZ->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_abb_abb += 1/10 * einsum('Wxyz,UVYxyZXz->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_abb_abb -= 7/40 * einsum('Wxyz,UVYxyzXZ->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_abb_abb -= 1/24 * einsum('Wxyz,UVYxzXyZ->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_abb_abb -= 3/20 * einsum('Wxyz,UVYxzZXy->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_abb_abb -= 1/60 * einsum('Wxyz,UVYxzZyX->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_abb_abb -= 19/120 * einsum('Wxyz,UVYxzyXZ->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_abb_abb += 1/60 * einsum('Wxyz,UVYxzyZX->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_abb_abb -= 1/8 * einsum('XZxy,UVYWxy->UVXWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_abb_abb += 1/8 * einsum('XZxy,UVYWyx->UVXWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_abb_abb -= 1/24 * einsum('XZxy,UVYxWy->UVXWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_abb_abb += 1/24 * einsum('XZxy,UVYxyW->UVXWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_abb_abb += 1/24 * einsum('XZxy,UVYyWx->UVXWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_abb_abb -= 1/24 * einsum('XZxy,UVYyxW->UVXWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_abb_abb += 1/3 * einsum('XxYy,UVxWZy->UVXWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_abb_abb += 1/6 * einsum('XxYy,UVxZWy->UVXWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_abb_abb -= 1/6 * einsum('XxyY,UVxWZy->UVXWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_abb_abb += 1/12 * einsum('XxyY,UVxWyZ->UVXWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_abb_abb -= 1/12 * einsum('XxyY,UVxZWy->UVXWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_abb_abb -= 1/12 * einsum('XxyY,UVxyZW->UVXWZY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_abb_abb += 3/20 * einsum('Yxyz,UVyzWxXZ->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_abb_abb -= 1/10 * einsum('Yxyz,UVyzWxZX->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_abb_abb -= 1/40 * einsum('Yxyz,UVyzXWZx->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_abb_abb += 7/120 * einsum('Yxyz,UVyzXWxZ->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_abb_abb -= 1/20 * einsum('Yxyz,UVyzXZWx->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_abb_abb += 1/10 * einsum('Yxyz,UVyzXZxW->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_abb_abb += 1/60 * einsum('Yxyz,UVyzXxWZ->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_abb_abb += 1/8 * einsum('Yxyz,UVyzZWXx->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_abb_abb += 1/24 * einsum('Yxyz,UVyzZWxX->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_abb_abb += 1/60 * einsum('Yxyz,UVyzZXWx->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_abb_abb += 1/12 * einsum('Yxyz,UVyzZXxW->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_abb_abb += 7/30 * einsum('Yxyz,UVyzZxXW->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_abb_abb += 7/40 * einsum('Yxyz,UVyzxWXZ->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_abb_abb -= 3/40 * einsum('Yxyz,UVyzxWZX->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_abb_abb -= 1/60 * einsum('Yxyz,UVyzxXWZ->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_abb_abb -= 1/30 * einsum('Yxyz,UVyzxXZW->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_abb_abb -= 1/60 * einsum('Yxyz,UVyzxZWX->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_abb_abb += 13/60 * einsum('Yxyz,UVyzxZXW->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_abb_abb += 1/10 * einsum('Zxyz,UVYxWyXz->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_abb_abb -= 3/20 * einsum('Zxyz,UVYxWzXy->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_abb_abb += 1/40 * einsum('Zxyz,UVYxXWyz->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_abb_abb -= 7/120 * einsum('Zxyz,UVYxXWzy->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_abb_abb += 1/8 * einsum('Zxyz,UVYxXyWz->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_abb_abb -= 3/40 * einsum('Zxyz,UVYxXyzW->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_abb_abb -= 1/24 * einsum('Zxyz,UVYxXzWy->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_abb_abb -= 3/40 * einsum('Zxyz,UVYxXzyW->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_abb_abb -= 1/20 * einsum('Zxyz,UVYxyWXz->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_abb_abb -= 1/15 * einsum('Zxyz,UVYxyWzX->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_abb_abb += 1/30 * einsum('Zxyz,UVYxyXWz->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_abb_abb -= 1/12 * einsum('Zxyz,UVYxyXzW->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_abb_abb -= 1/20 * einsum('Zxyz,UVYxyzWX->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_abb_abb -= 7/30 * einsum('Zxyz,UVYxyzXW->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_abb_abb -= 3/20 * einsum('Zxyz,UVYxzWXy->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_abb_abb += 1/60 * einsum('Zxyz,UVYxzXWy->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_abb_abb -= 1/60 * einsum('Zxyz,UVYxzXyW->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_abb_abb += 1/60 * einsum('Zxyz,UVYxzyWX->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_abb_abb -= 1/6 * einsum('Zxyz,UVYxzyXW->UVXWZY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_abb_abb -= 1/6 * einsum('Wx,XY,UVZx->UVXWZY', h_aa, np.identity(ncas), rdm_ccaa, optimize = einsum_type)
    K22_abb_abb -= 1/3 * einsum('Wx,XY,UVxZ->UVXWZY', h_aa, np.identity(ncas), rdm_ccaa, optimize = einsum_type)
    K22_abb_abb -= 1/3 * einsum('Zx,XY,UVWx->UVXWZY', h_aa, np.identity(ncas), rdm_ccaa, optimize = einsum_type)
    K22_abb_abb -= 1/6 * einsum('Zx,XY,UVxW->UVXWZY', h_aa, np.identity(ncas), rdm_ccaa, optimize = einsum_type)
    K22_abb_abb -= 1/3 * einsum('XY,WZxy,UVxy->UVXWZY', np.identity(ncas), v_aaaa, rdm_ccaa, optimize = einsum_type)
    K22_abb_abb -= 1/6 * einsum('XY,WZxy,UVyx->UVXWZY', np.identity(ncas), v_aaaa, rdm_ccaa, optimize = einsum_type)
    K22_abb_abb -= 1/12 * einsum('XY,Wxyz,UVxZyz->UVXWZY', np.identity(ncas), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_abb_abb += 1/12 * einsum('XY,Wxyz,UVxZzy->UVXWZY', np.identity(ncas), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_abb_abb -= 1/4 * einsum('XY,Wxyz,UVxyZz->UVXWZY', np.identity(ncas), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_abb_abb += 1/12 * einsum('XY,Wxyz,UVxyzZ->UVXWZY', np.identity(ncas), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_abb_abb += 1/12 * einsum('XY,Wxyz,UVxzZy->UVXWZY', np.identity(ncas), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_abb_abb += 1/12 * einsum('XY,Wxyz,UVxzyZ->UVXWZY', np.identity(ncas), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_abb_abb -= 1/4 * einsum('XY,Zxyz,UVxWyz->UVXWZY', np.identity(ncas), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_abb_abb += 1/12 * einsum('XY,Zxyz,UVxWzy->UVXWZY', np.identity(ncas), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_abb_abb -= 1/12 * einsum('XY,Zxyz,UVxyWz->UVXWZY', np.identity(ncas), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_abb_abb += 1/12 * einsum('XY,Zxyz,UVxyzW->UVXWZY', np.identity(ncas), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_abb_abb += 1/12 * einsum('XY,Zxyz,UVxzWy->UVXWZY', np.identity(ncas), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_abb_abb += 1/12 * einsum('XY,Zxyz,UVxzyW->UVXWZY', np.identity(ncas), v_aaaa, rdm_cccaaa, optimize = einsum_type)

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

# Spin-Orbital Sanity Check Functions
def compute_K_caca_sanity_check(mr_adc):

    # Einsum definition from kernel
    einsum = mr_adc.interface.einsum
    einsum_type = mr_adc.interface.einsum_type

    # Variables from kernel
    ncas = mr_adc.ncas

    rdm_ca = mr_adc.rdm.ca
    rdm_ccaa = mr_adc.rdm.ccaa
    rdm_cccaaa = mr_adc.rdm.cccaaa

    h_aa = mr_adc.h1eff.aa
    v_aaaa = mr_adc.v2e.aaaa

    K_caca = np.zeros((ncas * 2, ncas * 2, ncas * 2, ncas * 2))

    K_caca_ab_ba  = 1/2 * einsum('WY,XZ->XYWZ', h_aa, rdm_ca, optimize = einsum_type)
    K_caca_ab_ba -= 1/6 * einsum('Wx,XxYZ->XYWZ', h_aa, rdm_ccaa, optimize = einsum_type)
    K_caca_ab_ba -= 1/3 * einsum('Wx,XxZY->XYWZ', h_aa, rdm_ccaa, optimize = einsum_type)
    K_caca_ab_ba += 1/3 * einsum('Zx,WXYx->XYWZ', h_aa, rdm_ccaa, optimize = einsum_type)
    K_caca_ab_ba += 1/6 * einsum('Zx,WXxY->XYWZ', h_aa, rdm_ccaa, optimize = einsum_type)
    K_caca_ab_ba += 1/2 * einsum('WxYy,XyZx->XYWZ', v_aaaa, rdm_ccaa, optimize = einsum_type)
    K_caca_ab_ba -= 1/3 * einsum('WxyY,XyZx->XYWZ', v_aaaa, rdm_ccaa, optimize = einsum_type)
    K_caca_ab_ba -= 1/6 * einsum('WxyY,XyxZ->XYWZ', v_aaaa, rdm_ccaa, optimize = einsum_type)
    K_caca_ab_ba -= 1/12 * einsum('Wxyz,XyzYZx->XYWZ', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K_caca_ab_ba += 1/12 * einsum('Wxyz,XyzYxZ->XYWZ', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K_caca_ab_ba -= 1/4 * einsum('Wxyz,XyzZYx->XYWZ', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K_caca_ab_ba += 1/12 * einsum('Wxyz,XyzZxY->XYWZ', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K_caca_ab_ba += 1/12 * einsum('Wxyz,XyzxYZ->XYWZ', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K_caca_ab_ba += 1/12 * einsum('Wxyz,XyzxZY->XYWZ', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K_caca_ab_ba += 1/3 * einsum('YZxy,WXxy->XYWZ', v_aaaa, rdm_ccaa, optimize = einsum_type)
    K_caca_ab_ba += 1/6 * einsum('YZxy,WXyx->XYWZ', v_aaaa, rdm_ccaa, optimize = einsum_type)
    K_caca_ab_ba += 1/4 * einsum('Zxyz,WXxYyz->XYWZ', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K_caca_ab_ba -= 1/12 * einsum('Zxyz,WXxYzy->XYWZ', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K_caca_ab_ba += 1/12 * einsum('Zxyz,WXxyYz->XYWZ', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K_caca_ab_ba -= 1/12 * einsum('Zxyz,WXxyzY->XYWZ', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K_caca_ab_ba -= 1/12 * einsum('Zxyz,WXxzYy->XYWZ', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K_caca_ab_ba -= 1/12 * einsum('Zxyz,WXxzyY->XYWZ', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K_caca_ab_ba -= 1/2 * einsum('Zx,WY,Xx->XYWZ', h_aa, np.identity(ncas), rdm_ca, optimize = einsum_type)
    K_caca_ab_ba -= 1/2 * einsum('WY,Zxyz,Xxyz->XYWZ', np.identity(ncas), v_aaaa, rdm_ccaa, optimize = einsum_type)

    K_caca_aa_bb  = 1/3 * einsum('Wx,XxYZ->XYWZ', h_aa, rdm_ccaa, optimize = einsum_type)
    K_caca_aa_bb += 1/6 * einsum('Wx,XxZY->XYWZ', h_aa, rdm_ccaa, optimize = einsum_type)
    K_caca_aa_bb -= 1/6 * einsum('Zx,WXYx->XYWZ', h_aa, rdm_ccaa, optimize = einsum_type)
    K_caca_aa_bb -= 1/3 * einsum('Zx,WXxY->XYWZ', h_aa, rdm_ccaa, optimize = einsum_type)
    K_caca_aa_bb += 1/6 * einsum('WxyY,XyZx->XYWZ', v_aaaa, rdm_ccaa, optimize = einsum_type)
    K_caca_aa_bb += 1/3 * einsum('WxyY,XyxZ->XYWZ', v_aaaa, rdm_ccaa, optimize = einsum_type)
    K_caca_aa_bb += 1/4 * einsum('Wxyz,XyzYZx->XYWZ', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K_caca_aa_bb -= 1/12 * einsum('Wxyz,XyzYxZ->XYWZ', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K_caca_aa_bb += 1/12 * einsum('Wxyz,XyzZYx->XYWZ', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K_caca_aa_bb -= 1/12 * einsum('Wxyz,XyzZxY->XYWZ', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K_caca_aa_bb -= 1/12 * einsum('Wxyz,XyzxYZ->XYWZ', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K_caca_aa_bb -= 1/12 * einsum('Wxyz,XyzxZY->XYWZ', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K_caca_aa_bb -= 1/6 * einsum('YZxy,WXxy->XYWZ', v_aaaa, rdm_ccaa, optimize = einsum_type)
    K_caca_aa_bb -= 1/3 * einsum('YZxy,WXyx->XYWZ', v_aaaa, rdm_ccaa, optimize = einsum_type)
    K_caca_aa_bb -= 1/12 * einsum('Zxyz,WXxYyz->XYWZ', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K_caca_aa_bb += 1/12 * einsum('Zxyz,WXxYzy->XYWZ', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K_caca_aa_bb -= 1/4 * einsum('Zxyz,WXxyYz->XYWZ', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K_caca_aa_bb += 1/12 * einsum('Zxyz,WXxyzY->XYWZ', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K_caca_aa_bb += 1/12 * einsum('Zxyz,WXxzYy->XYWZ', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K_caca_aa_bb += 1/12 * einsum('Zxyz,WXxzyY->XYWZ', v_aaaa, rdm_cccaaa, optimize = einsum_type)

    K_caca[::2,1::2,1::2,::2] = K_caca_ab_ba.copy()
    K_caca[1::2,::2,::2,1::2] = K_caca_ab_ba.copy()

    K_caca[::2,::2,1::2,1::2] = K_caca_aa_bb.copy()
    K_caca[1::2,1::2,::2,::2] = K_caca_aa_bb.copy()

    K_caca[::2,::2,::2,::2]  = K_caca_ab_ba.copy()
    K_caca[::2,::2,::2,::2] += K_caca_aa_bb.copy()
    K_caca[1::2,1::2,1::2,1::2] = K_caca[::2,::2,::2,::2].copy()

    return K_caca

def compute_K_p1p_sanity_check(mr_adc):

    # Einsum definition from kernel
    einsum = mr_adc.interface.einsum
    einsum_type = mr_adc.interface.einsum_type

    # Variables from kernel
    ncas = mr_adc.ncas

    rdm_ca = mr_adc.rdm.ca
    rdm_ccaa = mr_adc.rdm.ccaa
    rdm_cccaaa = mr_adc.rdm.cccaaa
    rdm_ccccaaaa = mr_adc.rdm.ccccaaaa

    h_aa = mr_adc.h1eff.aa
    v_aaaa = mr_adc.v2e.aaaa

    n_x = ncas * 2
    n_xzw = ncas * 2 * ncas * 2 * (ncas * 2 - 1) // 2
    dim_act = n_x + n_xzw
    aa_ind = np.tril_indices(ncas * 2, k=-1)

    # Computing K11
    K11 = np.zeros((ncas * 2, ncas * 2))

    K11_a_a  = einsum('XY->XY', h_aa, optimize = einsum_type).copy()
    # K11_a_a -= 1/2 * einsum('Yx,xX->XY', h_aa, rdm_ca, optimize = einsum_type)
    K11_a_a += einsum('XxYy,xy->XY', v_aaaa, rdm_ca, optimize = einsum_type)
    K11_a_a -= 1/2 * einsum('XxyY,xy->XY', v_aaaa, rdm_ca, optimize = einsum_type)
    K11_a_a -= 1/2 * einsum('Yxyz,yzXx->XY', v_aaaa, rdm_ccaa, optimize = einsum_type)

    K11[::2,::2] = K11_a_a.copy()
    K11[1::2,1::2] = K11_a_a.copy()

    # Computing K12
    K12 = np.zeros((ncas * 2, ncas * 2, ncas * 2, ncas * 2))

    K12_a_bba =- 1/3 * einsum('Wx,YxXZ->XZWY', h_aa, rdm_ccaa, optimize = einsum_type)
    K12_a_bba -= 1/6 * einsum('Wx,YxZX->XZWY', h_aa, rdm_ccaa, optimize = einsum_type)
    K12_a_bba += 1/2 * einsum('XY,WZ->XZWY', h_aa, rdm_ca, optimize = einsum_type)
    K12_a_bba -= 1/6 * einsum('Yx,WxXZ->XZWY', h_aa, rdm_ccaa, optimize = einsum_type)
    K12_a_bba -= 1/3 * einsum('Yx,WxZX->XZWY', h_aa, rdm_ccaa, optimize = einsum_type)
    K12_a_bba += 1/6 * einsum('Zx,WYXx->XZWY', h_aa, rdm_ccaa, optimize = einsum_type)
    K12_a_bba += 1/3 * einsum('Zx,WYxX->XZWY', h_aa, rdm_ccaa, optimize = einsum_type)
    K12_a_bba += 1/2 * einsum('WYxX,xZ->XZWY', v_aaaa, rdm_ca, optimize = einsum_type)
    K12_a_bba -= 1/6 * einsum('WYxy,xyXZ->XZWY', v_aaaa, rdm_ccaa, optimize = einsum_type)
    K12_a_bba -= 1/3 * einsum('WYxy,xyZX->XZWY', v_aaaa, rdm_ccaa, optimize = einsum_type)
    K12_a_bba -= 1/6 * einsum('WxyX,YyZx->XZWY', v_aaaa, rdm_ccaa, optimize = einsum_type)
    K12_a_bba -= 1/3 * einsum('WxyX,YyxZ->XZWY', v_aaaa, rdm_ccaa, optimize = einsum_type)
    K12_a_bba -= 1/4 * einsum('Wxyz,YyzXZx->XZWY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K12_a_bba += 1/12 * einsum('Wxyz,YyzXxZ->XZWY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K12_a_bba -= 1/12 * einsum('Wxyz,YyzZXx->XZWY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K12_a_bba += 1/12 * einsum('Wxyz,YyzZxX->XZWY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K12_a_bba += 1/12 * einsum('Wxyz,YyzxXZ->XZWY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K12_a_bba += 1/12 * einsum('Wxyz,YyzxZX->XZWY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K12_a_bba += 1/6 * einsum('XZxy,WYxy->XZWY', v_aaaa, rdm_ccaa, optimize = einsum_type)
    K12_a_bba += 1/3 * einsum('XZxy,WYyx->XZWY', v_aaaa, rdm_ccaa, optimize = einsum_type)
    K12_a_bba += 1/2 * einsum('XxYy,WxZy->XZWY', v_aaaa, rdm_ccaa, optimize = einsum_type)
    K12_a_bba -= 1/3 * einsum('XxyY,WxZy->XZWY', v_aaaa, rdm_ccaa, optimize = einsum_type)
    K12_a_bba -= 1/6 * einsum('XxyY,WxyZ->XZWY', v_aaaa, rdm_ccaa, optimize = einsum_type)
    K12_a_bba -= 1/12 * einsum('Yxyz,WyzXZx->XZWY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K12_a_bba += 1/12 * einsum('Yxyz,WyzXxZ->XZWY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K12_a_bba -= 1/4 * einsum('Yxyz,WyzZXx->XZWY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K12_a_bba += 1/12 * einsum('Yxyz,WyzZxX->XZWY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K12_a_bba += 1/12 * einsum('Yxyz,WyzxXZ->XZWY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K12_a_bba += 1/12 * einsum('Yxyz,WyzxZX->XZWY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K12_a_bba += 1/12 * einsum('Zxyz,WYxXyz->XZWY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K12_a_bba -= 1/12 * einsum('Zxyz,WYxXzy->XZWY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K12_a_bba += 1/4 * einsum('Zxyz,WYxyXz->XZWY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K12_a_bba -= 1/12 * einsum('Zxyz,WYxyzX->XZWY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K12_a_bba -= 1/12 * einsum('Zxyz,WYxzXy->XZWY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K12_a_bba -= 1/12 * einsum('Zxyz,WYxzyX->XZWY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K12_a_bba += 1/2 * einsum('Wx,XY,xZ->XZWY', h_aa, np.identity(ncas), rdm_ca, optimize = einsum_type)
    K12_a_bba -= 1/2 * einsum('Zx,XY,Wx->XZWY', h_aa, np.identity(ncas), rdm_ca, optimize = einsum_type)
    K12_a_bba += 1/2 * einsum('XY,Wxyz,yzZx->XZWY', np.identity(ncas), v_aaaa, rdm_ccaa, optimize = einsum_type)
    K12_a_bba -= 1/2 * einsum('XY,Zxyz,Wxyz->XZWY', np.identity(ncas), v_aaaa, rdm_ccaa, optimize = einsum_type)

    K12[::2,1::2,1::2,::2] = K12_a_bba.copy()
    K12[1::2,::2,::2,1::2] = K12[::2,1::2,1::2,::2].copy()

    K12[::2,1::2,::2,1::2] -= K12_a_bba.transpose(0,1,3,2).copy()
    K12[1::2,::2,1::2,::2]  = K12[::2,1::2,::2,1::2].copy()

    K12[::2,::2,::2,::2]  = K12_a_bba.copy()
    K12[::2,::2,::2,::2] += K12[::2,1::2,::2,1::2].copy()
    K12[1::2,1::2,1::2,1::2] = K12[::2,::2,::2,::2].copy()

    # Computing K22
    K22 = np.zeros((ncas * 2, ncas * 2, ncas * 2, ncas * 2, ncas * 2, ncas * 2))

    K22_aab_aab =- 1/6 * einsum('VW,UYXZ->UVXZWY', h_aa, rdm_ccaa, optimize = einsum_type)
    K22_aab_aab -= 1/3 * einsum('VW,UYZX->UVXZWY', h_aa, rdm_ccaa, optimize = einsum_type)
    K22_aab_aab -= 1/6 * einsum('Wx,UYxVXZ->UVXZWY', h_aa, rdm_cccaaa, optimize = einsum_type)
    K22_aab_aab -= 1/12 * einsum('Wx,UYxVZX->UVXZWY', h_aa, rdm_cccaaa, optimize = einsum_type)
    K22_aab_aab -= 1/12 * einsum('Wx,UYxXVZ->UVXZWY', h_aa, rdm_cccaaa, optimize = einsum_type)
    K22_aab_aab += 1/12 * einsum('Wx,UYxZXV->UVXZWY', h_aa, rdm_cccaaa, optimize = einsum_type)
    K22_aab_aab += 1/6 * einsum('XY,UWVZ->UVXZWY', h_aa, rdm_ccaa, optimize = einsum_type)
    K22_aab_aab -= 1/6 * einsum('XY,UWZV->UVXZWY', h_aa, rdm_ccaa, optimize = einsum_type)
    K22_aab_aab -= 1/12 * einsum('Yx,UWxVZX->UVXZWY', h_aa, rdm_cccaaa, optimize = einsum_type)
    K22_aab_aab += 1/12 * einsum('Yx,UWxXVZ->UVXZWY', h_aa, rdm_cccaaa, optimize = einsum_type)
    K22_aab_aab += 1/6 * einsum('Yx,UWxZVX->UVXZWY', h_aa, rdm_cccaaa, optimize = einsum_type)
    K22_aab_aab += 1/12 * einsum('Yx,UWxZXV->UVXZWY', h_aa, rdm_cccaaa, optimize = einsum_type)
    K22_aab_aab += 1/12 * einsum('Zx,UWYVxX->UVXZWY', h_aa, rdm_cccaaa, optimize = einsum_type)
    K22_aab_aab -= 1/12 * einsum('Zx,UWYXVx->UVXZWY', h_aa, rdm_cccaaa, optimize = einsum_type)
    K22_aab_aab -= 1/6 * einsum('Zx,UWYxVX->UVXZWY', h_aa, rdm_cccaaa, optimize = einsum_type)
    K22_aab_aab -= 1/12 * einsum('Zx,UWYxXV->UVXZWY', h_aa, rdm_cccaaa, optimize = einsum_type)
    K22_aab_aab += 1/2 * einsum('VXWY,UZ->UVXZWY', v_aaaa, rdm_ca, optimize = einsum_type)
    K22_aab_aab -= 1/3 * einsum('VXWx,UYZx->UVXZWY', v_aaaa, rdm_ccaa, optimize = einsum_type)
    K22_aab_aab -= 1/6 * einsum('VXWx,UYxZ->UVXZWY', v_aaaa, rdm_ccaa, optimize = einsum_type)
    K22_aab_aab -= 1/6 * einsum('VXxY,UWZx->UVXZWY', v_aaaa, rdm_ccaa, optimize = einsum_type)
    K22_aab_aab += 1/6 * einsum('VXxY,UWxZ->UVXZWY', v_aaaa, rdm_ccaa, optimize = einsum_type)
    K22_aab_aab -= 1/24 * einsum('VZxy,UWYXxy->UVXZWY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_aab_aab += 1/24 * einsum('VZxy,UWYXyx->UVXZWY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_aab_aab += 1/24 * einsum('VZxy,UWYxXy->UVXZWY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_aab_aab += 1/8 * einsum('VZxy,UWYxyX->UVXZWY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_aab_aab -= 1/24 * einsum('VZxy,UWYyXx->UVXZWY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_aab_aab -= 1/8 * einsum('VZxy,UWYyxX->UVXZWY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_aab_aab -= 1/6 * einsum('VxWY,UxXZ->UVXZWY', v_aaaa, rdm_ccaa, optimize = einsum_type)
    K22_aab_aab -= 1/3 * einsum('VxWY,UxZX->UVXZWY', v_aaaa, rdm_ccaa, optimize = einsum_type)
    K22_aab_aab += 1/6 * einsum('VxWy,UYxXyZ->UVXZWY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_aab_aab -= 1/6 * einsum('VxWy,UYxZXy->UVXZWY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_aab_aab += 1/6 * einsum('VxWy,UYxZyX->UVXZWY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_aab_aab += 1/6 * einsum('VxWy,UYxyXZ->UVXZWY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_aab_aab += 1/6 * einsum('VxWy,UYxyZX->UVXZWY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_aab_aab -= 1/12 * einsum('VxyW,UYxXyZ->UVXZWY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_aab_aab += 1/12 * einsum('VxyW,UYxZXy->UVXZWY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_aab_aab -= 1/6 * einsum('VxyW,UYxyXZ->UVXZWY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_aab_aab -= 1/12 * einsum('VxyW,UYxyZX->UVXZWY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_aab_aab += 1/12 * einsum('VxyY,UWxXyZ->UVXZWY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_aab_aab += 1/12 * einsum('VxyY,UWxZXy->UVXZWY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_aab_aab += 1/6 * einsum('VxyY,UWxZyX->UVXZWY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_aab_aab -= 1/12 * einsum('VxyY,UWxyZX->UVXZWY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_aab_aab += 1/6 * einsum('WYxX,UxVZ->UVXZWY', v_aaaa, rdm_ccaa, optimize = einsum_type)
    K22_aab_aab -= 1/6 * einsum('WYxX,UxZV->UVXZWY', v_aaaa, rdm_ccaa, optimize = einsum_type)
    K22_aab_aab -= 1/24 * einsum('WYxy,UxyVXZ->UVXZWY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_aab_aab -= 1/8 * einsum('WYxy,UxyVZX->UVXZWY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_aab_aab += 1/24 * einsum('WYxy,UxyXVZ->UVXZWY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_aab_aab -= 1/24 * einsum('WYxy,UxyXZV->UVXZWY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_aab_aab += 1/8 * einsum('WYxy,UxyZVX->UVXZWY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_aab_aab += 1/24 * einsum('WYxy,UxyZXV->UVXZWY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_aab_aab -= 1/12 * einsum('WxyX,UYyVxZ->UVXZWY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_aab_aab += 1/12 * einsum('WxyX,UYyZVx->UVXZWY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_aab_aab += 1/6 * einsum('WxyX,UYyZxV->UVXZWY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_aab_aab += 1/12 * einsum('WxyX,UYyxZV->UVXZWY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_aab_aab -= 3/10 * einsum('Wxyz,UYyzVXZx->UVXZWY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aab_aab += 1/10 * einsum('Wxyz,UYyzVXxZ->UVXZWY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aab_aab -= 2/15 * einsum('Wxyz,UYyzVZXx->UVXZWY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aab_aab += 1/30 * einsum('Wxyz,UYyzVZxX->UVXZWY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aab_aab += 1/10 * einsum('Wxyz,UYyzVxXZ->UVXZWY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aab_aab -= 1/20 * einsum('Wxyz,UYyzVxZX->UVXZWY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aab_aab -= 11/40 * einsum('Wxyz,UYyzXVZx->UVXZWY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aab_aab += 1/24 * einsum('Wxyz,UYyzXVxZ->UVXZWY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aab_aab -= 1/8 * einsum('Wxyz,UYyzXZVx->UVXZWY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aab_aab -= 1/24 * einsum('Wxyz,UYyzXZxV->UVXZWY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aab_aab += 1/40 * einsum('Wxyz,UYyzXxVZ->UVXZWY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aab_aab -= 1/8 * einsum('Wxyz,UYyzXxZV->UVXZWY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aab_aab -= 1/15 * einsum('Wxyz,UYyzZVXx->UVXZWY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aab_aab += 1/60 * einsum('Wxyz,UYyzZVxX->UVXZWY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aab_aab += 1/15 * einsum('Wxyz,UYyzxVXZ->UVXZWY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aab_aab -= 1/12 * einsum('Wxyz,UYyzxVZX->UVXZWY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aab_aab += 1/20 * einsum('Wxyz,UYyzxXVZ->UVXZWY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aab_aab -= 1/10 * einsum('Wxyz,UYyzxXZV->UVXZWY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aab_aab -= 1/60 * einsum('Wxyz,UYyzxZVX->UVXZWY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aab_aab -= 1/60 * einsum('Wxyz,UYyzxZXV->UVXZWY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aab_aab += 1/24 * einsum('XZxy,UWYVxy->UVXZWY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_aab_aab += 1/8 * einsum('XZxy,UWYVyx->UVXZWY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_aab_aab -= 1/24 * einsum('XZxy,UWYxVy->UVXZWY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_aab_aab += 1/24 * einsum('XZxy,UWYxyV->UVXZWY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_aab_aab -= 1/8 * einsum('XZxy,UWYyVx->UVXZWY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_aab_aab -= 1/24 * einsum('XZxy,UWYyxV->UVXZWY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_aab_aab += 1/6 * einsum('XxYy,UWxVZy->UVXZWY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_aab_aab -= 1/6 * einsum('XxYy,UWxZVy->UVXZWY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_aab_aab -= 1/6 * einsum('XxyY,UWxVZy->UVXZWY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_aab_aab -= 1/12 * einsum('XxyY,UWxVyZ->UVXZWY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_aab_aab += 1/12 * einsum('XxyY,UWxZVy->UVXZWY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_aab_aab -= 1/12 * einsum('XxyY,UWxyZV->UVXZWY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_aab_aab -= 1/40 * einsum('Yxyz,UWyzVXZx->UVXZWY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aab_aab += 9/40 * einsum('Yxyz,UWyzVXxZ->UVXZWY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aab_aab -= 1/10 * einsum('Yxyz,UWyzVZXx->UVXZWY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aab_aab += 3/20 * einsum('Yxyz,UWyzVZxX->UVXZWY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aab_aab += 1/5 * einsum('Yxyz,UWyzVxXZ->UVXZWY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aab_aab += 7/60 * einsum('Yxyz,UWyzVxZX->UVXZWY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aab_aab -= 1/15 * einsum('Yxyz,UWyzXVZx->UVXZWY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aab_aab += 1/10 * einsum('Yxyz,UWyzXVxZ->UVXZWY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aab_aab -= 1/15 * einsum('Yxyz,UWyzXZVx->UVXZWY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aab_aab += 1/20 * einsum('Yxyz,UWyzXZxV->UVXZWY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aab_aab += 3/20 * einsum('Yxyz,UWyzXxVZ->UVXZWY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aab_aab += 1/60 * einsum('Yxyz,UWyzXxZV->UVXZWY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aab_aab -= 1/120 * einsum('Yxyz,UWyzZXVx->UVXZWY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aab_aab += 1/40 * einsum('Yxyz,UWyzZXxV->UVXZWY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aab_aab += 1/20 * einsum('Yxyz,UWyzZxVX->UVXZWY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aab_aab += 1/40 * einsum('Yxyz,UWyzxVXZ->UVXZWY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aab_aab -= 7/120 * einsum('Yxyz,UWyzxVZX->UVXZWY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aab_aab += 1/10 * einsum('Yxyz,UWyzxXVZ->UVXZWY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aab_aab -= 1/30 * einsum('Yxyz,UWyzxXZV->UVXZWY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aab_aab += 1/40 * einsum('Yxyz,UWyzxZVX->UVXZWY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aab_aab -= 1/40 * einsum('Yxyz,UWyzxZXV->UVXZWY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aab_aab += 3/20 * einsum('Zxyz,UWYxVXyz->UVXZWY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aab_aab -= 1/10 * einsum('Zxyz,UWYxVXzy->UVXZWY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aab_aab += 3/10 * einsum('Zxyz,UWYxVyXz->UVXZWY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aab_aab += 1/30 * einsum('Zxyz,UWYxVyzX->UVXZWY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aab_aab -= 1/10 * einsum('Zxyz,UWYxVzXy->UVXZWY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aab_aab -= 1/30 * einsum('Zxyz,UWYxVzyX->UVXZWY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aab_aab += 2/15 * einsum('Zxyz,UWYxXVyz->UVXZWY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aab_aab -= 1/30 * einsum('Zxyz,UWYxXVzy->UVXZWY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aab_aab += 11/40 * einsum('Zxyz,UWYxXyVz->UVXZWY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aab_aab += 7/60 * einsum('Zxyz,UWYxXyzV->UVXZWY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aab_aab -= 1/24 * einsum('Zxyz,UWYxXzVy->UVXZWY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aab_aab += 1/20 * einsum('Zxyz,UWYxXzyV->UVXZWY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aab_aab -= 1/60 * einsum('Zxyz,UWYxyVzX->UVXZWY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aab_aab += 3/40 * einsum('Zxyz,UWYxyXVz->UVXZWY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aab_aab -= 1/40 * einsum('Zxyz,UWYxyzVX->UVXZWY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aab_aab -= 1/20 * einsum('Zxyz,UWYxzVXy->UVXZWY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aab_aab += 1/60 * einsum('Zxyz,UWYxzVyX->UVXZWY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aab_aab -= 7/120 * einsum('Zxyz,UWYxzXVy->UVXZWY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aab_aab += 1/30 * einsum('Zxyz,UWYxzXyV->UVXZWY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aab_aab += 3/40 * einsum('Zxyz,UWYxzyVX->UVXZWY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aab_aab += 1/10 * einsum('Zxyz,UWYxzyXV->UVXZWY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aab_aab += 1/2 * einsum('VW,XY,UZ->UVXZWY', h_aa, np.identity(ncas), rdm_ca, optimize = einsum_type)
    K22_aab_aab += 1/6 * einsum('Wx,XY,UxVZ->UVXZWY', h_aa, np.identity(ncas), rdm_ccaa, optimize = einsum_type)
    K22_aab_aab -= 1/6 * einsum('Wx,XY,UxZV->UVXZWY', h_aa, np.identity(ncas), rdm_ccaa, optimize = einsum_type)
    K22_aab_aab += 1/2 * einsum('XY,VW,UZ->UVXZWY', h_aa, np.identity(ncas), rdm_ca, optimize = einsum_type)
    K22_aab_aab -= 1/6 * einsum('Yx,VW,UxXZ->UVXZWY', h_aa, np.identity(ncas), rdm_ccaa, optimize = einsum_type)
    K22_aab_aab -= 1/3 * einsum('Yx,VW,UxZX->UVXZWY', h_aa, np.identity(ncas), rdm_ccaa, optimize = einsum_type)
    K22_aab_aab += 1/6 * einsum('Zx,VW,UYXx->UVXZWY', h_aa, np.identity(ncas), rdm_ccaa, optimize = einsum_type)
    K22_aab_aab += 1/3 * einsum('Zx,VW,UYxX->UVXZWY', h_aa, np.identity(ncas), rdm_ccaa, optimize = einsum_type)
    K22_aab_aab -= 1/6 * einsum('Zx,XY,UWVx->UVXZWY', h_aa, np.identity(ncas), rdm_ccaa, optimize = einsum_type)
    K22_aab_aab += 1/6 * einsum('Zx,XY,UWxV->UVXZWY', h_aa, np.identity(ncas), rdm_ccaa, optimize = einsum_type)
    K22_aab_aab += 1/6 * einsum('VW,XZxy,UYxy->UVXZWY', np.identity(ncas), v_aaaa, rdm_ccaa, optimize = einsum_type)
    K22_aab_aab += 1/3 * einsum('VW,XZxy,UYyx->UVXZWY', np.identity(ncas), v_aaaa, rdm_ccaa, optimize = einsum_type)
    K22_aab_aab += 1/2 * einsum('VW,XxYy,UxZy->UVXZWY', np.identity(ncas), v_aaaa, rdm_ccaa, optimize = einsum_type)
    K22_aab_aab -= 1/3 * einsum('VW,XxyY,UxZy->UVXZWY', np.identity(ncas), v_aaaa, rdm_ccaa, optimize = einsum_type)
    K22_aab_aab -= 1/6 * einsum('VW,XxyY,UxyZ->UVXZWY', np.identity(ncas), v_aaaa, rdm_ccaa, optimize = einsum_type)
    K22_aab_aab -= 1/12 * einsum('VW,Yxyz,UyzXZx->UVXZWY', np.identity(ncas), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_aab_aab += 1/12 * einsum('VW,Yxyz,UyzXxZ->UVXZWY', np.identity(ncas), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_aab_aab -= 1/4 * einsum('VW,Yxyz,UyzZXx->UVXZWY', np.identity(ncas), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_aab_aab += 1/12 * einsum('VW,Yxyz,UyzZxX->UVXZWY', np.identity(ncas), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_aab_aab += 1/12 * einsum('VW,Yxyz,UyzxXZ->UVXZWY', np.identity(ncas), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_aab_aab += 1/12 * einsum('VW,Yxyz,UyzxZX->UVXZWY', np.identity(ncas), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_aab_aab += 1/12 * einsum('VW,Zxyz,UYxXyz->UVXZWY', np.identity(ncas), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_aab_aab -= 1/12 * einsum('VW,Zxyz,UYxXzy->UVXZWY', np.identity(ncas), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_aab_aab += 1/4 * einsum('VW,Zxyz,UYxyXz->UVXZWY', np.identity(ncas), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_aab_aab -= 1/12 * einsum('VW,Zxyz,UYxyzX->UVXZWY', np.identity(ncas), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_aab_aab -= 1/12 * einsum('VW,Zxyz,UYxzXy->UVXZWY', np.identity(ncas), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_aab_aab -= 1/12 * einsum('VW,Zxyz,UYxzyX->UVXZWY', np.identity(ncas), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_aab_aab -= 1/6 * einsum('XY,VZxy,UWxy->UVXZWY', np.identity(ncas), v_aaaa, rdm_ccaa, optimize = einsum_type)
    K22_aab_aab += 1/6 * einsum('XY,VZxy,UWyx->UVXZWY', np.identity(ncas), v_aaaa, rdm_ccaa, optimize = einsum_type)
    K22_aab_aab += 1/2 * einsum('XY,VxWy,UxZy->UVXZWY', np.identity(ncas), v_aaaa, rdm_ccaa, optimize = einsum_type)
    K22_aab_aab -= 1/6 * einsum('XY,VxyW,UxZy->UVXZWY', np.identity(ncas), v_aaaa, rdm_ccaa, optimize = einsum_type)
    K22_aab_aab += 1/6 * einsum('XY,VxyW,UxyZ->UVXZWY', np.identity(ncas), v_aaaa, rdm_ccaa, optimize = einsum_type)
    K22_aab_aab += 1/6 * einsum('XY,Wxyz,UyzVZx->UVXZWY', np.identity(ncas), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_aab_aab -= 1/6 * einsum('XY,Wxyz,UyzZVx->UVXZWY', np.identity(ncas), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_aab_aab -= 1/6 * einsum('XY,Zxyz,UWxVyz->UVXZWY', np.identity(ncas), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_aab_aab += 1/6 * einsum('XY,Zxyz,UWxyVz->UVXZWY', np.identity(ncas), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_aab_aab -= 1/2 * einsum('Zx,VW,XY,Ux->UVXZWY', h_aa, np.identity(ncas), np.identity(ncas), rdm_ca, optimize = einsum_type)
    K22_aab_aab -= 1/2 * einsum('Zxyz,VW,XY,Uxyz->UVXZWY', v_aaaa, np.identity(ncas), np.identity(ncas), rdm_ccaa, optimize = einsum_type)

    K22_baa_baa =- 1/6 * einsum('VW,UYXZ->UVXZWY', h_aa, rdm_ccaa, optimize = einsum_type)
    K22_baa_baa -= 1/3 * einsum('VW,UYZX->UVXZWY', h_aa, rdm_ccaa, optimize = einsum_type)
    K22_baa_baa += 1/6 * einsum('VY,UWXZ->UVXZWY', h_aa, rdm_ccaa, optimize = einsum_type)
    K22_baa_baa += 1/3 * einsum('VY,UWZX->UVXZWY', h_aa, rdm_ccaa, optimize = einsum_type)
    K22_baa_baa += 1/6 * einsum('WX,UYVZ->UVXZWY', h_aa, rdm_ccaa, optimize = einsum_type)
    K22_baa_baa += 1/3 * einsum('WX,UYZV->UVXZWY', h_aa, rdm_ccaa, optimize = einsum_type)
    K22_baa_baa -= 1/12 * einsum('Wx,UYxVZX->UVXZWY', h_aa, rdm_cccaaa, optimize = einsum_type)
    K22_baa_baa -= 1/12 * einsum('Wx,UYxXVZ->UVXZWY', h_aa, rdm_cccaaa, optimize = einsum_type)
    K22_baa_baa -= 1/6 * einsum('Wx,UYxZVX->UVXZWY', h_aa, rdm_cccaaa, optimize = einsum_type)
    K22_baa_baa += 1/12 * einsum('Wx,UYxZXV->UVXZWY', h_aa, rdm_cccaaa, optimize = einsum_type)
    K22_baa_baa -= 1/6 * einsum('XY,UWVZ->UVXZWY', h_aa, rdm_ccaa, optimize = einsum_type)
    K22_baa_baa -= 1/3 * einsum('XY,UWZV->UVXZWY', h_aa, rdm_ccaa, optimize = einsum_type)
    K22_baa_baa += 1/12 * einsum('Yx,UWxVZX->UVXZWY', h_aa, rdm_cccaaa, optimize = einsum_type)
    K22_baa_baa += 1/12 * einsum('Yx,UWxXVZ->UVXZWY', h_aa, rdm_cccaaa, optimize = einsum_type)
    K22_baa_baa += 1/6 * einsum('Yx,UWxZVX->UVXZWY', h_aa, rdm_cccaaa, optimize = einsum_type)
    K22_baa_baa -= 1/12 * einsum('Yx,UWxZXV->UVXZWY', h_aa, rdm_cccaaa, optimize = einsum_type)
    K22_baa_baa -= 1/12 * einsum('Zx,UWYVxX->UVXZWY', h_aa, rdm_cccaaa, optimize = einsum_type)
    K22_baa_baa -= 1/12 * einsum('Zx,UWYXVx->UVXZWY', h_aa, rdm_cccaaa, optimize = einsum_type)
    K22_baa_baa -= 1/6 * einsum('Zx,UWYxVX->UVXZWY', h_aa, rdm_cccaaa, optimize = einsum_type)
    K22_baa_baa += 1/12 * einsum('Zx,UWYxXV->UVXZWY', h_aa, rdm_cccaaa, optimize = einsum_type)
    K22_baa_baa += 1/2 * einsum('VXWY,UZ->UVXZWY', v_aaaa, rdm_ca, optimize = einsum_type)
    K22_baa_baa -= 1/3 * einsum('VXWx,UYZx->UVXZWY', v_aaaa, rdm_ccaa, optimize = einsum_type)
    K22_baa_baa -= 1/6 * einsum('VXWx,UYxZ->UVXZWY', v_aaaa, rdm_ccaa, optimize = einsum_type)
    K22_baa_baa -= 1/2 * einsum('VXYW,UZ->UVXZWY', v_aaaa, rdm_ca, optimize = einsum_type)
    K22_baa_baa += 1/3 * einsum('VXYx,UWZx->UVXZWY', v_aaaa, rdm_ccaa, optimize = einsum_type)
    K22_baa_baa += 1/6 * einsum('VXYx,UWxZ->UVXZWY', v_aaaa, rdm_ccaa, optimize = einsum_type)
    K22_baa_baa += 1/3 * einsum('VXxW,UYZx->UVXZWY', v_aaaa, rdm_ccaa, optimize = einsum_type)
    K22_baa_baa += 1/6 * einsum('VXxW,UYxZ->UVXZWY', v_aaaa, rdm_ccaa, optimize = einsum_type)
    K22_baa_baa -= 1/3 * einsum('VXxY,UWZx->UVXZWY', v_aaaa, rdm_ccaa, optimize = einsum_type)
    K22_baa_baa -= 1/6 * einsum('VXxY,UWxZ->UVXZWY', v_aaaa, rdm_ccaa, optimize = einsum_type)
    K22_baa_baa -= 1/24 * einsum('VZxy,UWYXxy->UVXZWY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_baa_baa += 1/24 * einsum('VZxy,UWYXyx->UVXZWY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_baa_baa += 1/24 * einsum('VZxy,UWYxXy->UVXZWY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_baa_baa -= 1/24 * einsum('VZxy,UWYxyX->UVXZWY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_baa_baa += 1/8 * einsum('VZxy,UWYyXx->UVXZWY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_baa_baa -= 1/8 * einsum('VZxy,UWYyxX->UVXZWY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_baa_baa -= 1/6 * einsum('VxWY,UxXZ->UVXZWY', v_aaaa, rdm_ccaa, optimize = einsum_type)
    K22_baa_baa -= 1/3 * einsum('VxWY,UxZX->UVXZWY', v_aaaa, rdm_ccaa, optimize = einsum_type)
    K22_baa_baa += 1/6 * einsum('VxWy,UYxXyZ->UVXZWY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_baa_baa -= 1/6 * einsum('VxWy,UYxZXy->UVXZWY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_baa_baa += 1/6 * einsum('VxWy,UYxZyX->UVXZWY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_baa_baa += 1/6 * einsum('VxWy,UYxyXZ->UVXZWY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_baa_baa += 1/6 * einsum('VxWy,UYxyZX->UVXZWY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_baa_baa += 1/6 * einsum('VxYW,UxXZ->UVXZWY', v_aaaa, rdm_ccaa, optimize = einsum_type)
    K22_baa_baa += 1/3 * einsum('VxYW,UxZX->UVXZWY', v_aaaa, rdm_ccaa, optimize = einsum_type)
    K22_baa_baa -= 1/6 * einsum('VxYy,UWxXyZ->UVXZWY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_baa_baa += 1/6 * einsum('VxYy,UWxZXy->UVXZWY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_baa_baa -= 1/6 * einsum('VxYy,UWxZyX->UVXZWY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_baa_baa -= 1/6 * einsum('VxYy,UWxyXZ->UVXZWY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_baa_baa -= 1/6 * einsum('VxYy,UWxyZX->UVXZWY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_baa_baa -= 1/12 * einsum('VxyW,UYxXyZ->UVXZWY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_baa_baa += 1/12 * einsum('VxyW,UYxZXy->UVXZWY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_baa_baa -= 1/6 * einsum('VxyW,UYxZyX->UVXZWY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_baa_baa -= 1/12 * einsum('VxyW,UYxyZX->UVXZWY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_baa_baa += 1/12 * einsum('VxyY,UWxXyZ->UVXZWY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_baa_baa -= 1/12 * einsum('VxyY,UWxZXy->UVXZWY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_baa_baa += 1/6 * einsum('VxyY,UWxZyX->UVXZWY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_baa_baa += 1/12 * einsum('VxyY,UWxyZX->UVXZWY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_baa_baa += 1/6 * einsum('WYXx,UxVZ->UVXZWY', v_aaaa, rdm_ccaa, optimize = einsum_type)
    K22_baa_baa += 1/3 * einsum('WYXx,UxZV->UVXZWY', v_aaaa, rdm_ccaa, optimize = einsum_type)
    K22_baa_baa -= 1/6 * einsum('WYxX,UxVZ->UVXZWY', v_aaaa, rdm_ccaa, optimize = einsum_type)
    K22_baa_baa -= 1/3 * einsum('WYxX,UxZV->UVXZWY', v_aaaa, rdm_ccaa, optimize = einsum_type)
    K22_baa_baa -= 1/24 * einsum('WYxy,UxyVXZ->UVXZWY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_baa_baa += 1/24 * einsum('WYxy,UxyVZX->UVXZWY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_baa_baa += 1/24 * einsum('WYxy,UxyXVZ->UVXZWY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_baa_baa -= 1/24 * einsum('WYxy,UxyXZV->UVXZWY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_baa_baa += 1/8 * einsum('WYxy,UxyZVX->UVXZWY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_baa_baa -= 1/8 * einsum('WYxy,UxyZXV->UVXZWY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_baa_baa -= 1/6 * einsum('WxXy,UYyVxZ->UVXZWY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_baa_baa += 1/6 * einsum('WxXy,UYyZVx->UVXZWY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_baa_baa -= 1/6 * einsum('WxXy,UYyZxV->UVXZWY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_baa_baa -= 1/6 * einsum('WxXy,UYyxVZ->UVXZWY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_baa_baa -= 1/6 * einsum('WxXy,UYyxZV->UVXZWY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_baa_baa += 1/12 * einsum('WxyX,UYyVxZ->UVXZWY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_baa_baa -= 1/12 * einsum('WxyX,UYyZVx->UVXZWY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_baa_baa += 1/6 * einsum('WxyX,UYyZxV->UVXZWY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_baa_baa += 1/12 * einsum('WxyX,UYyxZV->UVXZWY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_baa_baa -= 1/15 * einsum('Wxyz,UYyzVXZx->UVXZWY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_baa_baa += 1/60 * einsum('Wxyz,UYyzVXxZ->UVXZWY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_baa_baa -= 11/40 * einsum('Wxyz,UYyzVZXx->UVXZWY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_baa_baa += 1/24 * einsum('Wxyz,UYyzVZxX->UVXZWY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_baa_baa -= 1/12 * einsum('Wxyz,UYyzVxXZ->UVXZWY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_baa_baa += 1/15 * einsum('Wxyz,UYyzVxZX->UVXZWY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_baa_baa -= 2/15 * einsum('Wxyz,UYyzXVZx->UVXZWY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_baa_baa += 1/30 * einsum('Wxyz,UYyzXVxZ->UVXZWY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_baa_baa -= 1/8 * einsum('Wxyz,UYyzXZVx->UVXZWY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_baa_baa -= 1/24 * einsum('Wxyz,UYyzXZxV->UVXZWY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_baa_baa -= 1/60 * einsum('Wxyz,UYyzXxVZ->UVXZWY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_baa_baa -= 1/60 * einsum('Wxyz,UYyzXxZV->UVXZWY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_baa_baa -= 3/10 * einsum('Wxyz,UYyzZVXx->UVXZWY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_baa_baa += 1/10 * einsum('Wxyz,UYyzZVxX->UVXZWY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_baa_baa += 1/20 * einsum('Wxyz,UYyzZxVX->UVXZWY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_baa_baa -= 1/10 * einsum('Wxyz,UYyzZxXV->UVXZWY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_baa_baa -= 1/20 * einsum('Wxyz,UYyzxVXZ->UVXZWY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_baa_baa += 1/10 * einsum('Wxyz,UYyzxVZX->UVXZWY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_baa_baa += 1/40 * einsum('Wxyz,UYyzxZVX->UVXZWY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_baa_baa -= 1/8 * einsum('Wxyz,UYyzxZXV->UVXZWY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_baa_baa += 1/24 * einsum('XZxy,UWYVxy->UVXZWY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_baa_baa -= 1/24 * einsum('XZxy,UWYVyx->UVXZWY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_baa_baa -= 1/24 * einsum('XZxy,UWYxVy->UVXZWY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_baa_baa += 1/24 * einsum('XZxy,UWYxyV->UVXZWY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_baa_baa -= 1/8 * einsum('XZxy,UWYyVx->UVXZWY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_baa_baa += 1/8 * einsum('XZxy,UWYyxV->UVXZWY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_baa_baa += 1/6 * einsum('XxYy,UWxVyZ->UVXZWY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_baa_baa -= 1/6 * einsum('XxYy,UWxZVy->UVXZWY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_baa_baa += 1/6 * einsum('XxYy,UWxZyV->UVXZWY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_baa_baa += 1/6 * einsum('XxYy,UWxyVZ->UVXZWY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_baa_baa += 1/6 * einsum('XxYy,UWxyZV->UVXZWY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_baa_baa -= 1/12 * einsum('XxyY,UWxVyZ->UVXZWY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_baa_baa += 1/12 * einsum('XxyY,UWxZVy->UVXZWY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_baa_baa -= 1/6 * einsum('XxyY,UWxZyV->UVXZWY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_baa_baa -= 1/12 * einsum('XxyY,UWxyZV->UVXZWY', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_baa_baa += 1/15 * einsum('Yxyz,UWyzVXZx->UVXZWY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_baa_baa -= 1/60 * einsum('Yxyz,UWyzVXxZ->UVXZWY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_baa_baa += 11/40 * einsum('Yxyz,UWyzVZXx->UVXZWY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_baa_baa -= 1/24 * einsum('Yxyz,UWyzVZxX->UVXZWY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_baa_baa += 1/12 * einsum('Yxyz,UWyzVxXZ->UVXZWY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_baa_baa -= 1/15 * einsum('Yxyz,UWyzVxZX->UVXZWY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_baa_baa += 2/15 * einsum('Yxyz,UWyzXVZx->UVXZWY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_baa_baa -= 1/30 * einsum('Yxyz,UWyzXVxZ->UVXZWY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_baa_baa += 1/8 * einsum('Yxyz,UWyzXZVx->UVXZWY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_baa_baa += 1/24 * einsum('Yxyz,UWyzXZxV->UVXZWY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_baa_baa += 1/60 * einsum('Yxyz,UWyzXxVZ->UVXZWY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_baa_baa += 1/60 * einsum('Yxyz,UWyzXxZV->UVXZWY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_baa_baa += 3/10 * einsum('Yxyz,UWyzZVXx->UVXZWY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_baa_baa -= 1/10 * einsum('Yxyz,UWyzZVxX->UVXZWY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_baa_baa -= 1/20 * einsum('Yxyz,UWyzZxVX->UVXZWY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_baa_baa += 1/10 * einsum('Yxyz,UWyzZxXV->UVXZWY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_baa_baa += 1/20 * einsum('Yxyz,UWyzxVXZ->UVXZWY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_baa_baa -= 1/10 * einsum('Yxyz,UWyzxVZX->UVXZWY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_baa_baa -= 1/40 * einsum('Yxyz,UWyzxZVX->UVXZWY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_baa_baa += 1/8 * einsum('Yxyz,UWyzxZXV->UVXZWY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_baa_baa -= 1/120 * einsum('Zxyz,UWYxVXyz->UVXZWY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_baa_baa += 3/40 * einsum('Zxyz,UWYxVXzy->UVXZWY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_baa_baa -= 11/120 * einsum('Zxyz,UWYxVyXz->UVXZWY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_baa_baa += 1/10 * einsum('Zxyz,UWYxVyzX->UVXZWY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_baa_baa += 3/40 * einsum('Zxyz,UWYxVzXy->UVXZWY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_baa_baa += 1/10 * einsum('Zxyz,UWYxVzyX->UVXZWY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_baa_baa += 1/6 * einsum('Zxyz,UWYxXVzy->UVXZWY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_baa_baa -= 1/15 * einsum('Zxyz,UWYxXyVz->UVXZWY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_baa_baa += 1/60 * einsum('Zxyz,UWYxXyzV->UVXZWY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_baa_baa += 1/60 * einsum('Zxyz,UWYxXzVy->UVXZWY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_baa_baa += 1/60 * einsum('Zxyz,UWYxXzyV->UVXZWY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_baa_baa -= 1/10 * einsum('Zxyz,UWYxyVXz->UVXZWY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_baa_baa += 7/40 * einsum('Zxyz,UWYxyVzX->UVXZWY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_baa_baa += 1/40 * einsum('Zxyz,UWYxyzVX->UVXZWY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_baa_baa += 3/20 * einsum('Zxyz,UWYxzVXy->UVXZWY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_baa_baa += 7/40 * einsum('Zxyz,UWYxzVyX->UVXZWY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_baa_baa += 1/40 * einsum('Zxyz,UWYxzyVX->UVXZWY', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_baa_baa += 1/2 * einsum('VW,XY,UZ->UVXZWY', h_aa, np.identity(ncas), rdm_ca, optimize = einsum_type)
    K22_baa_baa -= 1/2 * einsum('VY,WX,UZ->UVXZWY', h_aa, np.identity(ncas), rdm_ca, optimize = einsum_type)
    K22_baa_baa -= 1/2 * einsum('WX,VY,UZ->UVXZWY', h_aa, np.identity(ncas), rdm_ca, optimize = einsum_type)
    K22_baa_baa += 1/6 * einsum('Wx,VY,UxXZ->UVXZWY', h_aa, np.identity(ncas), rdm_ccaa, optimize = einsum_type)
    K22_baa_baa += 1/3 * einsum('Wx,VY,UxZX->UVXZWY', h_aa, np.identity(ncas), rdm_ccaa, optimize = einsum_type)
    K22_baa_baa -= 1/6 * einsum('Wx,XY,UxVZ->UVXZWY', h_aa, np.identity(ncas), rdm_ccaa, optimize = einsum_type)
    K22_baa_baa -= 1/3 * einsum('Wx,XY,UxZV->UVXZWY', h_aa, np.identity(ncas), rdm_ccaa, optimize = einsum_type)
    K22_baa_baa += 1/2 * einsum('XY,VW,UZ->UVXZWY', h_aa, np.identity(ncas), rdm_ca, optimize = einsum_type)
    K22_baa_baa -= 1/6 * einsum('Yx,VW,UxXZ->UVXZWY', h_aa, np.identity(ncas), rdm_ccaa, optimize = einsum_type)
    K22_baa_baa -= 1/3 * einsum('Yx,VW,UxZX->UVXZWY', h_aa, np.identity(ncas), rdm_ccaa, optimize = einsum_type)
    K22_baa_baa += 1/6 * einsum('Yx,WX,UxVZ->UVXZWY', h_aa, np.identity(ncas), rdm_ccaa, optimize = einsum_type)
    K22_baa_baa += 1/3 * einsum('Yx,WX,UxZV->UVXZWY', h_aa, np.identity(ncas), rdm_ccaa, optimize = einsum_type)
    K22_baa_baa += 1/6 * einsum('Zx,VW,UYXx->UVXZWY', h_aa, np.identity(ncas), rdm_ccaa, optimize = einsum_type)
    K22_baa_baa += 1/3 * einsum('Zx,VW,UYxX->UVXZWY', h_aa, np.identity(ncas), rdm_ccaa, optimize = einsum_type)
    K22_baa_baa -= 1/6 * einsum('Zx,VY,UWXx->UVXZWY', h_aa, np.identity(ncas), rdm_ccaa, optimize = einsum_type)
    K22_baa_baa -= 1/3 * einsum('Zx,VY,UWxX->UVXZWY', h_aa, np.identity(ncas), rdm_ccaa, optimize = einsum_type)
    K22_baa_baa -= 1/6 * einsum('Zx,WX,UYVx->UVXZWY', h_aa, np.identity(ncas), rdm_ccaa, optimize = einsum_type)
    K22_baa_baa -= 1/3 * einsum('Zx,WX,UYxV->UVXZWY', h_aa, np.identity(ncas), rdm_ccaa, optimize = einsum_type)
    K22_baa_baa += 1/6 * einsum('Zx,XY,UWVx->UVXZWY', h_aa, np.identity(ncas), rdm_ccaa, optimize = einsum_type)
    K22_baa_baa += 1/3 * einsum('Zx,XY,UWxV->UVXZWY', h_aa, np.identity(ncas), rdm_ccaa, optimize = einsum_type)
    K22_baa_baa += 1/6 * einsum('VW,XZxy,UYxy->UVXZWY', np.identity(ncas), v_aaaa, rdm_ccaa, optimize = einsum_type)
    K22_baa_baa += 1/3 * einsum('VW,XZxy,UYyx->UVXZWY', np.identity(ncas), v_aaaa, rdm_ccaa, optimize = einsum_type)
    K22_baa_baa += 1/2 * einsum('VW,XxYy,UxZy->UVXZWY', np.identity(ncas), v_aaaa, rdm_ccaa, optimize = einsum_type)
    K22_baa_baa -= 1/3 * einsum('VW,XxyY,UxZy->UVXZWY', np.identity(ncas), v_aaaa, rdm_ccaa, optimize = einsum_type)
    K22_baa_baa -= 1/6 * einsum('VW,XxyY,UxyZ->UVXZWY', np.identity(ncas), v_aaaa, rdm_ccaa, optimize = einsum_type)
    K22_baa_baa -= 1/12 * einsum('VW,Yxyz,UyzXZx->UVXZWY', np.identity(ncas), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_baa_baa += 1/12 * einsum('VW,Yxyz,UyzXxZ->UVXZWY', np.identity(ncas), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_baa_baa -= 1/4 * einsum('VW,Yxyz,UyzZXx->UVXZWY', np.identity(ncas), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_baa_baa += 1/12 * einsum('VW,Yxyz,UyzZxX->UVXZWY', np.identity(ncas), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_baa_baa += 1/12 * einsum('VW,Yxyz,UyzxXZ->UVXZWY', np.identity(ncas), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_baa_baa += 1/12 * einsum('VW,Yxyz,UyzxZX->UVXZWY', np.identity(ncas), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_baa_baa += 1/12 * einsum('VW,Zxyz,UYxXyz->UVXZWY', np.identity(ncas), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_baa_baa -= 1/12 * einsum('VW,Zxyz,UYxXzy->UVXZWY', np.identity(ncas), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_baa_baa += 1/4 * einsum('VW,Zxyz,UYxyXz->UVXZWY', np.identity(ncas), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_baa_baa -= 1/12 * einsum('VW,Zxyz,UYxyzX->UVXZWY', np.identity(ncas), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_baa_baa -= 1/12 * einsum('VW,Zxyz,UYxzXy->UVXZWY', np.identity(ncas), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_baa_baa -= 1/12 * einsum('VW,Zxyz,UYxzyX->UVXZWY', np.identity(ncas), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_baa_baa -= 1/2 * einsum('VY,WxXy,UyZx->UVXZWY', np.identity(ncas), v_aaaa, rdm_ccaa, optimize = einsum_type)
    K22_baa_baa += 1/3 * einsum('VY,WxyX,UyZx->UVXZWY', np.identity(ncas), v_aaaa, rdm_ccaa, optimize = einsum_type)
    K22_baa_baa += 1/6 * einsum('VY,WxyX,UyxZ->UVXZWY', np.identity(ncas), v_aaaa, rdm_ccaa, optimize = einsum_type)
    K22_baa_baa += 1/12 * einsum('VY,Wxyz,UyzXZx->UVXZWY', np.identity(ncas), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_baa_baa -= 1/12 * einsum('VY,Wxyz,UyzXxZ->UVXZWY', np.identity(ncas), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_baa_baa += 1/4 * einsum('VY,Wxyz,UyzZXx->UVXZWY', np.identity(ncas), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_baa_baa -= 1/12 * einsum('VY,Wxyz,UyzZxX->UVXZWY', np.identity(ncas), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_baa_baa -= 1/12 * einsum('VY,Wxyz,UyzxXZ->UVXZWY', np.identity(ncas), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_baa_baa -= 1/12 * einsum('VY,Wxyz,UyzxZX->UVXZWY', np.identity(ncas), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_baa_baa -= 1/6 * einsum('VY,XZxy,UWxy->UVXZWY', np.identity(ncas), v_aaaa, rdm_ccaa, optimize = einsum_type)
    K22_baa_baa -= 1/3 * einsum('VY,XZxy,UWyx->UVXZWY', np.identity(ncas), v_aaaa, rdm_ccaa, optimize = einsum_type)
    K22_baa_baa -= 1/12 * einsum('VY,Zxyz,UWxXyz->UVXZWY', np.identity(ncas), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_baa_baa += 1/12 * einsum('VY,Zxyz,UWxXzy->UVXZWY', np.identity(ncas), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_baa_baa -= 1/4 * einsum('VY,Zxyz,UWxyXz->UVXZWY', np.identity(ncas), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_baa_baa += 1/12 * einsum('VY,Zxyz,UWxyzX->UVXZWY', np.identity(ncas), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_baa_baa += 1/12 * einsum('VY,Zxyz,UWxzXy->UVXZWY', np.identity(ncas), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_baa_baa += 1/12 * einsum('VY,Zxyz,UWxzyX->UVXZWY', np.identity(ncas), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_baa_baa -= 1/6 * einsum('WX,VZxy,UYxy->UVXZWY', np.identity(ncas), v_aaaa, rdm_ccaa, optimize = einsum_type)
    K22_baa_baa -= 1/3 * einsum('WX,VZxy,UYyx->UVXZWY', np.identity(ncas), v_aaaa, rdm_ccaa, optimize = einsum_type)
    K22_baa_baa -= 1/2 * einsum('WX,VxYy,UxZy->UVXZWY', np.identity(ncas), v_aaaa, rdm_ccaa, optimize = einsum_type)
    K22_baa_baa += 1/3 * einsum('WX,VxyY,UxZy->UVXZWY', np.identity(ncas), v_aaaa, rdm_ccaa, optimize = einsum_type)
    K22_baa_baa += 1/6 * einsum('WX,VxyY,UxyZ->UVXZWY', np.identity(ncas), v_aaaa, rdm_ccaa, optimize = einsum_type)
    K22_baa_baa += 1/12 * einsum('WX,Yxyz,UyzVZx->UVXZWY', np.identity(ncas), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_baa_baa -= 1/12 * einsum('WX,Yxyz,UyzVxZ->UVXZWY', np.identity(ncas), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_baa_baa += 1/4 * einsum('WX,Yxyz,UyzZVx->UVXZWY', np.identity(ncas), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_baa_baa -= 1/12 * einsum('WX,Yxyz,UyzZxV->UVXZWY', np.identity(ncas), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_baa_baa -= 1/12 * einsum('WX,Yxyz,UyzxVZ->UVXZWY', np.identity(ncas), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_baa_baa -= 1/12 * einsum('WX,Yxyz,UyzxZV->UVXZWY', np.identity(ncas), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_baa_baa -= 1/12 * einsum('WX,Zxyz,UYxVyz->UVXZWY', np.identity(ncas), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_baa_baa += 1/12 * einsum('WX,Zxyz,UYxVzy->UVXZWY', np.identity(ncas), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_baa_baa -= 1/4 * einsum('WX,Zxyz,UYxyVz->UVXZWY', np.identity(ncas), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_baa_baa += 1/12 * einsum('WX,Zxyz,UYxyzV->UVXZWY', np.identity(ncas), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_baa_baa += 1/12 * einsum('WX,Zxyz,UYxzVy->UVXZWY', np.identity(ncas), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_baa_baa += 1/12 * einsum('WX,Zxyz,UYxzyV->UVXZWY', np.identity(ncas), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_baa_baa += 1/6 * einsum('XY,VZxy,UWxy->UVXZWY', np.identity(ncas), v_aaaa, rdm_ccaa, optimize = einsum_type)
    K22_baa_baa += 1/3 * einsum('XY,VZxy,UWyx->UVXZWY', np.identity(ncas), v_aaaa, rdm_ccaa, optimize = einsum_type)
    K22_baa_baa += 1/2 * einsum('XY,VxWy,UxZy->UVXZWY', np.identity(ncas), v_aaaa, rdm_ccaa, optimize = einsum_type)
    K22_baa_baa -= 1/3 * einsum('XY,VxyW,UxZy->UVXZWY', np.identity(ncas), v_aaaa, rdm_ccaa, optimize = einsum_type)
    K22_baa_baa -= 1/6 * einsum('XY,VxyW,UxyZ->UVXZWY', np.identity(ncas), v_aaaa, rdm_ccaa, optimize = einsum_type)
    K22_baa_baa -= 1/12 * einsum('XY,Wxyz,UyzVZx->UVXZWY', np.identity(ncas), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_baa_baa += 1/12 * einsum('XY,Wxyz,UyzVxZ->UVXZWY', np.identity(ncas), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_baa_baa -= 1/4 * einsum('XY,Wxyz,UyzZVx->UVXZWY', np.identity(ncas), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_baa_baa += 1/12 * einsum('XY,Wxyz,UyzZxV->UVXZWY', np.identity(ncas), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_baa_baa += 1/12 * einsum('XY,Wxyz,UyzxVZ->UVXZWY', np.identity(ncas), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_baa_baa += 1/12 * einsum('XY,Wxyz,UyzxZV->UVXZWY', np.identity(ncas), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_baa_baa += 1/12 * einsum('XY,Zxyz,UWxVyz->UVXZWY', np.identity(ncas), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_baa_baa -= 1/12 * einsum('XY,Zxyz,UWxVzy->UVXZWY', np.identity(ncas), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_baa_baa += 1/4 * einsum('XY,Zxyz,UWxyVz->UVXZWY', np.identity(ncas), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_baa_baa -= 1/12 * einsum('XY,Zxyz,UWxyzV->UVXZWY', np.identity(ncas), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_baa_baa -= 1/12 * einsum('XY,Zxyz,UWxzVy->UVXZWY', np.identity(ncas), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_baa_baa -= 1/12 * einsum('XY,Zxyz,UWxzyV->UVXZWY', np.identity(ncas), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_baa_baa -= 1/2 * einsum('Zx,VW,XY,Ux->UVXZWY', h_aa, np.identity(ncas), np.identity(ncas), rdm_ca, optimize = einsum_type)
    K22_baa_baa += 1/2 * einsum('Zx,VY,WX,Ux->UVXZWY', h_aa, np.identity(ncas), np.identity(ncas), rdm_ca, optimize = einsum_type)
    K22_baa_baa -= 1/2 * einsum('Zxyz,VW,XY,Uxyz->UVXZWY', v_aaaa, np.identity(ncas), np.identity(ncas), rdm_ccaa, optimize = einsum_type)
    K22_baa_baa += 1/2 * einsum('Zxyz,VY,WX,Uxyz->UVXZWY', v_aaaa, np.identity(ncas), np.identity(ncas), rdm_ccaa, optimize = einsum_type)

    K22[::2,::2,1::2,::2,::2,1::2] = K22_aab_aab.copy()
    K22[1::2,1::2,::2,1::2,1::2,::2] = K22_aab_aab.copy()

    # K22[1::2,::2,::2,1::2,::2,::2] = K22_baa_baa.copy()
    # K22[::2,1::2,1::2,::2,1::2,1::2] = K22_baa_baa.copy()

    K22[::2,1::2,::2,::2,1::2,::2] = K22_aab_aab.transpose(0,2,1,3,5,4).copy()
    K22[1::2,::2,1::2,1::2,::2,1::2] = K22[::2,1::2,::2,::2,1::2,::2].copy()

    K22[::2,1::2,::2,::2,::2,1::2] -= K22_aab_aab.transpose(0,2,1,3,4,5).copy()
    K22[1::2,::2,1::2,1::2,1::2,::2] = K22[::2,1::2,::2,::2,::2,1::2].copy()

    K22[::2,::2,1::2,::2,1::2,::2] -= K22_aab_aab.transpose(0,1,2,3,5,4).copy()
    K22[1::2,1::2,::2,1::2,::2,1::2] = K22[::2,::2,1::2,::2,1::2,::2].copy()

    K22[::2,::2,::2,1::2,1::2,::2]  = K22_aab_aab.copy()
    K22[::2,::2,::2,1::2,1::2,::2] -= K22_baa_baa.copy()
    K22[::2,::2,::2,1::2,1::2,::2] += K22[::2,1::2,::2,::2,::2,1::2].copy()
    K22[1::2,1::2,1::2,::2,::2,1::2] = K22[::2,::2,::2,1::2,1::2,::2].copy()

    K22[1::2,1::2,::2,::2,::2,::2]  = K22_aab_aab.copy()
    K22[1::2,1::2,::2,::2,::2,::2] -= K22_baa_baa.copy()
    K22[1::2,1::2,::2,::2,::2,::2] += K22[::2,::2,1::2,::2,1::2,::2].copy()
    K22[::2,::2,1::2,1::2,1::2,1::2] = K22[1::2,1::2,::2,::2,::2,::2].copy()

    K22[1::2,::2,1::2,::2,::2,::2] -= K22[1::2,1::2,::2,::2,::2,::2].transpose(0,2,1,3,4,5).copy()
    K22[::2,1::2,::2,1::2,1::2,1::2] = K22[1::2,::2,1::2,::2,::2,::2].copy()

    K22[::2,::2,::2,1::2,::2,1::2] -= K22[::2,::2,::2,1::2,1::2,::2].transpose(0,1,2,3,5,4).copy()
    K22[1::2,1::2,1::2,::2,1::2,::2] = K22[::2,::2,::2,1::2,::2,1::2].copy()

    K22[::2,::2,::2,::2,::2,::2]  = K22[::2,::2,::2,1::2,1::2,::2].copy()
    K22[::2,::2,::2,::2,::2,::2] += K22[1::2,1::2,::2,1::2,::2,1::2].copy()
    K22[::2,::2,::2,::2,::2,::2] += K22[1::2,::2,1::2,1::2,::2,1::2].copy()
    K22[1::2,1::2,1::2,1::2,1::2,1::2] = K22[::2,::2,::2,::2,::2,::2].copy()

    print(">>> SA K22_aaa_aaa (sanity): {:}".format(np.linalg.norm(K22[::2,::2,::2,::2,::2,::2])))
    print(">>> SA K22_bba_bba (sanity): {:}".format(np.linalg.norm(K22[1::2,1::2,::2,1::2,1::2,::2])))

    ###
    # # K22[::2,::2,1::2,::2,::2,1::2] = K22_aab_aab.copy()
    # K22[1::2,1::2,::2,1::2,1::2,::2] = K22_aab_aab.copy()

    # # K22[1::2,::2,::2,1::2,::2,::2] = K22_baa_baa.copy()
    # # K22[::2,1::2,1::2,::2,1::2,1::2] = K22_baa_baa.copy()

    # # K22[::2,1::2,::2,::2,1::2,::2] = K22_aab_aab.transpose(0,2,1,3,5,4).copy()
    # K22[1::2,::2,1::2,1::2,::2,1::2] = K22_aab_aab.transpose(0,2,1,3,5,4).copy()

    # # K22[::2,1::2,::2,::2,::2,1::2] -= K22_aab_aab.transpose(0,2,1,3,4,5).copy()
    # K22[1::2,::2,1::2,1::2,1::2,::2] -= K22_aab_aab.transpose(0,2,1,3,4,5).copy()

    # # K22[::2,::2,1::2,::2,1::2,::2] -= K22_aab_aab.transpose(0,1,2,3,5,4).copy()
    # K22[1::2,1::2,::2,1::2,::2,1::2] -= K22_aab_aab.transpose(0,1,2,3,5,4).copy()

    # K22[::2,::2,::2,1::2,1::2,::2]  = K22_aab_aab.copy()
    # K22[::2,::2,::2,1::2,1::2,::2] -= K22_baa_baa.copy()
    # K22[::2,::2,::2,1::2,1::2,::2] -= K22_aab_aab.transpose(0,2,1,3,4,5).copy()

    # # K22[1::2,1::2,1::2,::2,::2,1::2]  = K22_aab_aab.copy()
    # # K22[1::2,1::2,1::2,::2,::2,1::2] -= K22_baa_baa.copy()
    # # K22[1::2,1::2,1::2,::2,::2,1::2] -= K22_aab_aab.transpose(0,2,1,3,4,5).copy()

    # K22[1::2,1::2,::2,::2,::2,::2]  = K22_aab_aab.copy()
    # K22[1::2,1::2,::2,::2,::2,::2] -= K22_baa_baa.copy()
    # K22[1::2,1::2,::2,::2,::2,::2] -= K22_aab_aab.transpose(0,1,2,3,5,4).copy()

    # # K22[::2,::2,1::2,1::2,1::2,1::2]  = K22_aab_aab.copy()
    # # K22[::2,::2,1::2,1::2,1::2,1::2] -= K22_baa_baa.copy()
    # # K22[::2,::2,1::2,1::2,1::2,1::2] -= K22_aab_aab.transpose(0,1,2,3,5,4).copy()

    # K22[1::2,::2,1::2,::2,::2,::2] -= K22_aab_aab.transpose(0,2,1,3,4,5).copy()
    # K22[1::2,::2,1::2,::2,::2,::2] -= K22_baa_baa.copy()
    # K22[1::2,::2,1::2,::2,::2,::2] += K22_aab_aab.transpose(0,2,1,3,5,4).copy()

    # # K22[::2,1::2,::2,1::2,1::2,1::2] -= K22_aab_aab.transpose(0,2,1,3,4,5).copy()
    # # K22[::2,1::2,::2,1::2,1::2,1::2] -= K22_baa_baa.copy()
    # # K22[::2,1::2,::2,1::2,1::2,1::2] += K22_aab_aab.transpose(0,2,1,3,5,4).copy()

    # K22[::2,::2,::2,1::2,::2,1::2] -= K22_aab_aab.transpose(0,1,2,3,5,4).copy()
    # K22[::2,::2,::2,1::2,::2,1::2] -= K22_baa_baa.copy()
    # K22[::2,::2,::2,1::2,::2,1::2] += K22_aab_aab.transpose(0,2,1,3,5,4).copy()

    # # K22[1::2,1::2,1::2,::2,1::2,::2] -= K22_aab_aab.transpose(0,1,2,3,5,4).copy()
    # # K22[1::2,1::2,1::2,::2,1::2,::2] -= K22_baa_baa.copy()
    # # K22[1::2,1::2,1::2,::2,1::2,::2] += K22_aab_aab.transpose(0,2,1,3,5,4).copy()

    # K22[::2,::2,::2,::2,::2,::2]  = K22_aab_aab.copy()
    # K22[::2,::2,::2,::2,::2,::2] -= K22_baa_baa.copy()
    # K22[::2,::2,::2,::2,::2,::2] -= K22_aab_aab.transpose(0,2,1,3,4,5).copy()
    # K22[::2,::2,::2,::2,::2,::2] -= K22_aab_aab.transpose(0,1,2,3,5,4).copy()
    # K22[::2,::2,::2,::2,::2,::2] += K22_aab_aab.transpose(0,2,1,3,5,4).copy()

    # # K22[1::2,1::2,1::2,1::2,1::2,1::2]  = K22_aab_aab.copy()
    # # K22[1::2,1::2,1::2,1::2,1::2,1::2] -= K22_baa_baa.copy()
    # # K22[1::2,1::2,1::2,1::2,1::2,1::2] -= K22_aab_aab.transpose(0,2,1,3,4,5).copy()
    # # K22[1::2,1::2,1::2,1::2,1::2,1::2] -= K22_aab_aab.transpose(0,1,2,3,5,4).copy()
    # # K22[1::2,1::2,1::2,1::2,1::2,1::2] += K22_aab_aab.transpose(0,2,1,3,5,4).copy()
    ###

    K12 = K12[:,:,aa_ind[0],aa_ind[1]]
    K22 = K22[:,:,:,:,aa_ind[0],aa_ind[1]]
    K22 = K22[:,aa_ind[0],aa_ind[1]]

    K_p1p = np.zeros((dim_act, dim_act))

    K_p1p[:n_x,:n_x] = K11.copy()
    K_p1p[:n_x,n_x:] = K12.reshape(n_x, n_xzw).copy()
    K_p1p[n_x:,:n_x] = K12.reshape(n_x, n_xzw).T.copy()
    K_p1p[n_x:,n_x:] = K22.reshape(n_xzw, n_xzw).copy()

    return K_p1p

def compute_K_m1p_sanity_check(mr_adc):

    # Einsum definition from kernel
    einsum = mr_adc.interface.einsum
    einsum_type = mr_adc.interface.einsum_type

    # Variables from kernel
    ncas = mr_adc.ncas

    rdm_ca = mr_adc.rdm.ca
    rdm_ccaa = mr_adc.rdm.ccaa
    rdm_cccaaa = mr_adc.rdm.cccaaa
    rdm_ccccaaaa = mr_adc.rdm.ccccaaaa

    h_aa = mr_adc.h1eff.aa
    v_aaaa = mr_adc.v2e.aaaa

    n_x = ncas * 2
    n_xzw = ncas * 2 * ncas * 2 * (ncas * 2 - 1) // 2
    dim_act = n_x + n_xzw
    aa_ind = np.tril_indices(ncas * 2, k=-1)

    # Computing K11
    K11 = np.zeros((ncas * 2, ncas * 2))

    K11_a_a =- 1/2 * einsum('Yx,Xx->XY', h_aa, rdm_ca, optimize = einsum_type)
    K11_a_a -= 1/2 * einsum('Yxyz,Xxyz->XY', v_aaaa, rdm_ccaa, optimize = einsum_type)

    K11[::2,::2] = K11_a_a.copy()
    K11[1::2,1::2] = K11_a_a.copy()

    # Computing K12
    K12 = np.zeros((ncas * 2, ncas * 2, ncas * 2, ncas * 2))

    K12_a_abb =- 1/3 * einsum('Wx,XZYx->XYWZ', h_aa, rdm_ccaa, optimize = einsum_type)
    K12_a_abb -= 1/6 * einsum('Wx,XZxY->XYWZ', h_aa, rdm_ccaa, optimize = einsum_type)
    K12_a_abb -= 1/6 * einsum('Yx,XZWx->XYWZ', h_aa, rdm_ccaa, optimize = einsum_type)
    K12_a_abb -= 1/3 * einsum('Yx,XZxW->XYWZ', h_aa, rdm_ccaa, optimize = einsum_type)
    K12_a_abb += 1/6 * einsum('Zx,XxWY->XYWZ', h_aa, rdm_ccaa, optimize = einsum_type)
    K12_a_abb += 1/3 * einsum('Zx,XxYW->XYWZ', h_aa, rdm_ccaa, optimize = einsum_type)
    K12_a_abb -= 1/6 * einsum('WYxy,XZxy->XYWZ', v_aaaa, rdm_ccaa, optimize = einsum_type)
    K12_a_abb -= 1/3 * einsum('WYxy,XZyx->XYWZ', v_aaaa, rdm_ccaa, optimize = einsum_type)
    K12_a_abb -= 1/4 * einsum('Wxyz,XZxYyz->XYWZ', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K12_a_abb += 1/12 * einsum('Wxyz,XZxYzy->XYWZ', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K12_a_abb -= 1/12 * einsum('Wxyz,XZxyYz->XYWZ', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K12_a_abb += 1/12 * einsum('Wxyz,XZxyzY->XYWZ', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K12_a_abb += 1/12 * einsum('Wxyz,XZxzYy->XYWZ', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K12_a_abb += 1/12 * einsum('Wxyz,XZxzyY->XYWZ', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K12_a_abb -= 1/12 * einsum('Yxyz,XZxWyz->XYWZ', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K12_a_abb += 1/12 * einsum('Yxyz,XZxWzy->XYWZ', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K12_a_abb -= 1/4 * einsum('Yxyz,XZxyWz->XYWZ', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K12_a_abb += 1/12 * einsum('Yxyz,XZxyzW->XYWZ', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K12_a_abb += 1/12 * einsum('Yxyz,XZxzWy->XYWZ', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K12_a_abb += 1/12 * einsum('Yxyz,XZxzyW->XYWZ', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K12_a_abb += 1/12 * einsum('Zxyz,XyzWYx->XYWZ', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K12_a_abb -= 1/12 * einsum('Zxyz,XyzWxY->XYWZ', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K12_a_abb += 1/4 * einsum('Zxyz,XyzYWx->XYWZ', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K12_a_abb -= 1/12 * einsum('Zxyz,XyzYxW->XYWZ', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K12_a_abb -= 1/12 * einsum('Zxyz,XyzxWY->XYWZ', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K12_a_abb -= 1/12 * einsum('Zxyz,XyzxYW->XYWZ', v_aaaa, rdm_cccaaa, optimize = einsum_type)

    K12[::2,::2,1::2,1::2] = K12_a_abb.copy()
    K12[1::2,1::2,::2,::2] = K12_a_abb.copy()

    K12[::2,1::2,::2,1::2] -= K12[::2,::2,1::2,1::2].transpose(0,2,1,3).copy()
    K12[1::2,::2,1::2,::2]  = K12[::2,1::2,::2,1::2].copy()

    K12[::2,::2,::2,::2]  = K12_a_abb.copy()
    K12[::2,::2,::2,::2] += K12[::2,1::2,::2,1::2].copy()
    K12[1::2,1::2,1::2,1::2] = K12[::2,::2,::2,::2].copy()

    # Computing K22
    K22 = np.zeros((ncas * 2, ncas * 2, ncas * 2, ncas * 2, ncas * 2, ncas * 2))

    K22_aab_aab  = 1/6 * einsum('VZ,UXWY->XUVYWZ', h_aa, rdm_ccaa, optimize = einsum_type)
    K22_aab_aab -= 1/6 * einsum('VZ,UXYW->XUVYWZ', h_aa, rdm_ccaa, optimize = einsum_type)
    K22_aab_aab -= 1/12 * einsum('Wx,UXZVxY->XUVYWZ', h_aa, rdm_cccaaa, optimize = einsum_type)
    K22_aab_aab -= 1/12 * einsum('Wx,UXZYVx->XUVYWZ', h_aa, rdm_cccaaa, optimize = einsum_type)
    K22_aab_aab -= 1/6 * einsum('Wx,UXZYxV->XUVYWZ', h_aa, rdm_cccaaa, optimize = einsum_type)
    K22_aab_aab += 1/12 * einsum('Wx,UXZxYV->XUVYWZ', h_aa, rdm_cccaaa, optimize = einsum_type)
    K22_aab_aab += 1/12 * einsum('Yx,UXZVxW->XUVYWZ', h_aa, rdm_cccaaa, optimize = einsum_type)
    K22_aab_aab += 1/12 * einsum('Yx,UXZWVx->XUVYWZ', h_aa, rdm_cccaaa, optimize = einsum_type)
    K22_aab_aab += 1/6 * einsum('Yx,UXZWxV->XUVYWZ', h_aa, rdm_cccaaa, optimize = einsum_type)
    K22_aab_aab -= 1/12 * einsum('Yx,UXZxWV->XUVYWZ', h_aa, rdm_cccaaa, optimize = einsum_type)
    K22_aab_aab -= 1/12 * einsum('Zx,UXxVYW->XUVYWZ', h_aa, rdm_cccaaa, optimize = einsum_type)
    K22_aab_aab -= 1/12 * einsum('Zx,UXxWVY->XUVYWZ', h_aa, rdm_cccaaa, optimize = einsum_type)
    K22_aab_aab -= 1/6 * einsum('Zx,UXxWYV->XUVYWZ', h_aa, rdm_cccaaa, optimize = einsum_type)
    K22_aab_aab += 1/12 * einsum('Zx,UXxYWV->XUVYWZ', h_aa, rdm_cccaaa, optimize = einsum_type)
    K22_aab_aab -= 1/24 * einsum('VWxy,UXZYxy->XUVYWZ', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_aab_aab -= 1/8 * einsum('VWxy,UXZYyx->XUVYWZ', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_aab_aab += 1/24 * einsum('VWxy,UXZxYy->XUVYWZ', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_aab_aab -= 1/24 * einsum('VWxy,UXZxyY->XUVYWZ', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_aab_aab += 1/8 * einsum('VWxy,UXZyYx->XUVYWZ', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_aab_aab += 1/24 * einsum('VWxy,UXZyxY->XUVYWZ', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_aab_aab += 1/24 * einsum('VYxy,UXZWxy->XUVYWZ', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_aab_aab += 1/8 * einsum('VYxy,UXZWyx->XUVYWZ', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_aab_aab -= 1/24 * einsum('VYxy,UXZxWy->XUVYWZ', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_aab_aab += 1/24 * einsum('VYxy,UXZxyW->XUVYWZ', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_aab_aab -= 1/8 * einsum('VYxy,UXZyWx->XUVYWZ', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_aab_aab -= 1/24 * einsum('VYxy,UXZyxW->XUVYWZ', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_aab_aab += 1/6 * einsum('VxZy,UXxWYy->XUVYWZ', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_aab_aab -= 1/6 * einsum('VxZy,UXxYWy->XUVYWZ', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_aab_aab -= 1/6 * einsum('VxyZ,UXxWYy->XUVYWZ', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_aab_aab -= 1/12 * einsum('VxyZ,UXxWyY->XUVYWZ', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_aab_aab += 1/12 * einsum('VxyZ,UXxYWy->XUVYWZ', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_aab_aab -= 1/12 * einsum('VxyZ,UXxyYW->XUVYWZ', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_aab_aab -= 1/24 * einsum('WYxy,UXZVxy->XUVYWZ', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_aab_aab += 1/24 * einsum('WYxy,UXZVyx->XUVYWZ', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_aab_aab += 1/24 * einsum('WYxy,UXZxVy->XUVYWZ', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_aab_aab += 1/8 * einsum('WYxy,UXZxyV->XUVYWZ', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_aab_aab -= 1/24 * einsum('WYxy,UXZyVx->XUVYWZ', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_aab_aab -= 1/8 * einsum('WYxy,UXZyxV->XUVYWZ', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_aab_aab -= 2/15 * einsum('Wxyz,UXZxVYyz->XUVYWZ', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aab_aab += 1/30 * einsum('Wxyz,UXZxVYzy->XUVYWZ', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aab_aab -= 11/40 * einsum('Wxyz,UXZxVyYz->XUVYWZ', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aab_aab -= 7/60 * einsum('Wxyz,UXZxVyzY->XUVYWZ', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aab_aab += 1/24 * einsum('Wxyz,UXZxVzYy->XUVYWZ', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aab_aab -= 1/20 * einsum('Wxyz,UXZxVzyY->XUVYWZ', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aab_aab -= 3/20 * einsum('Wxyz,UXZxYVyz->XUVYWZ', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aab_aab += 1/10 * einsum('Wxyz,UXZxYVzy->XUVYWZ', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aab_aab -= 3/10 * einsum('Wxyz,UXZxYyVz->XUVYWZ', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aab_aab -= 1/30 * einsum('Wxyz,UXZxYyzV->XUVYWZ', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aab_aab += 1/10 * einsum('Wxyz,UXZxYzVy->XUVYWZ', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aab_aab += 1/30 * einsum('Wxyz,UXZxYzyV->XUVYWZ', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aab_aab -= 3/40 * einsum('Wxyz,UXZxyVYz->XUVYWZ', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aab_aab += 1/60 * einsum('Wxyz,UXZxyYzV->XUVYWZ', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aab_aab += 1/40 * einsum('Wxyz,UXZxyzYV->XUVYWZ', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aab_aab += 7/120 * einsum('Wxyz,UXZxzVYy->XUVYWZ', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aab_aab -= 1/30 * einsum('Wxyz,UXZxzVyY->XUVYWZ', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aab_aab += 1/20 * einsum('Wxyz,UXZxzYVy->XUVYWZ', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aab_aab -= 1/60 * einsum('Wxyz,UXZxzYyV->XUVYWZ', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aab_aab -= 1/10 * einsum('Wxyz,UXZxzyVY->XUVYWZ', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aab_aab -= 3/40 * einsum('Wxyz,UXZxzyYV->XUVYWZ', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aab_aab += 2/15 * einsum('Yxyz,UXZxVWyz->XUVYWZ', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aab_aab -= 1/30 * einsum('Yxyz,UXZxVWzy->XUVYWZ', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aab_aab += 11/40 * einsum('Yxyz,UXZxVyWz->XUVYWZ', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aab_aab += 7/60 * einsum('Yxyz,UXZxVyzW->XUVYWZ', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aab_aab -= 1/24 * einsum('Yxyz,UXZxVzWy->XUVYWZ', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aab_aab += 1/20 * einsum('Yxyz,UXZxVzyW->XUVYWZ', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aab_aab += 3/20 * einsum('Yxyz,UXZxWVyz->XUVYWZ', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aab_aab -= 1/10 * einsum('Yxyz,UXZxWVzy->XUVYWZ', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aab_aab += 3/10 * einsum('Yxyz,UXZxWyVz->XUVYWZ', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aab_aab += 1/30 * einsum('Yxyz,UXZxWyzV->XUVYWZ', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aab_aab -= 1/10 * einsum('Yxyz,UXZxWzVy->XUVYWZ', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aab_aab -= 1/30 * einsum('Yxyz,UXZxWzyV->XUVYWZ', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aab_aab += 3/40 * einsum('Yxyz,UXZxyVWz->XUVYWZ', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aab_aab -= 1/60 * einsum('Yxyz,UXZxyWzV->XUVYWZ', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aab_aab -= 1/40 * einsum('Yxyz,UXZxyzWV->XUVYWZ', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aab_aab -= 7/120 * einsum('Yxyz,UXZxzVWy->XUVYWZ', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aab_aab += 1/30 * einsum('Yxyz,UXZxzVyW->XUVYWZ', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aab_aab -= 1/20 * einsum('Yxyz,UXZxzWVy->XUVYWZ', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aab_aab += 1/60 * einsum('Yxyz,UXZxzWyV->XUVYWZ', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aab_aab += 1/10 * einsum('Yxyz,UXZxzyVW->XUVYWZ', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aab_aab += 3/40 * einsum('Yxyz,UXZxzyWV->XUVYWZ', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aab_aab -= 1/15 * einsum('Zxyz,UXyzVWYx->XUVYWZ', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aab_aab += 1/10 * einsum('Zxyz,UXyzVWxY->XUVYWZ', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aab_aab -= 1/15 * einsum('Zxyz,UXyzVYWx->XUVYWZ', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aab_aab += 1/20 * einsum('Zxyz,UXyzVYxW->XUVYWZ', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aab_aab += 3/20 * einsum('Zxyz,UXyzVxWY->XUVYWZ', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aab_aab += 1/60 * einsum('Zxyz,UXyzVxYW->XUVYWZ', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aab_aab -= 1/40 * einsum('Zxyz,UXyzWVYx->XUVYWZ', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aab_aab += 9/40 * einsum('Zxyz,UXyzWVxY->XUVYWZ', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aab_aab -= 1/10 * einsum('Zxyz,UXyzWYVx->XUVYWZ', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aab_aab += 3/20 * einsum('Zxyz,UXyzWYxV->XUVYWZ', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aab_aab += 1/5 * einsum('Zxyz,UXyzWxVY->XUVYWZ', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aab_aab += 7/60 * einsum('Zxyz,UXyzWxYV->XUVYWZ', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aab_aab -= 1/120 * einsum('Zxyz,UXyzYVWx->XUVYWZ', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aab_aab += 1/40 * einsum('Zxyz,UXyzYVxW->XUVYWZ', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aab_aab += 1/20 * einsum('Zxyz,UXyzYxWV->XUVYWZ', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aab_aab += 1/10 * einsum('Zxyz,UXyzxVWY->XUVYWZ', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aab_aab -= 1/30 * einsum('Zxyz,UXyzxVYW->XUVYWZ', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aab_aab += 1/40 * einsum('Zxyz,UXyzxWVY->XUVYWZ', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aab_aab -= 7/120 * einsum('Zxyz,UXyzxWYV->XUVYWZ', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aab_aab -= 1/40 * einsum('Zxyz,UXyzxYVW->XUVYWZ', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aab_aab += 1/40 * einsum('Zxyz,UXyzxYWV->XUVYWZ', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_aab_aab += 1/6 * einsum('Wx,VZ,UXYx->XUVYWZ', h_aa, np.identity(ncas), rdm_ccaa, optimize = einsum_type)
    K22_aab_aab -= 1/6 * einsum('Wx,VZ,UXxY->XUVYWZ', h_aa, np.identity(ncas), rdm_ccaa, optimize = einsum_type)
    K22_aab_aab -= 1/6 * einsum('Yx,VZ,UXWx->XUVYWZ', h_aa, np.identity(ncas), rdm_ccaa, optimize = einsum_type)
    K22_aab_aab += 1/6 * einsum('Yx,VZ,UXxW->XUVYWZ', h_aa, np.identity(ncas), rdm_ccaa, optimize = einsum_type)
    K22_aab_aab -= 1/6 * einsum('VZ,WYxy,UXxy->XUVYWZ', np.identity(ncas), v_aaaa, rdm_ccaa, optimize = einsum_type)
    K22_aab_aab += 1/6 * einsum('VZ,WYxy,UXyx->XUVYWZ', np.identity(ncas), v_aaaa, rdm_ccaa, optimize = einsum_type)
    K22_aab_aab += 1/6 * einsum('VZ,Wxyz,UXxYyz->XUVYWZ', np.identity(ncas), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_aab_aab -= 1/6 * einsum('VZ,Wxyz,UXxyYz->XUVYWZ', np.identity(ncas), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_aab_aab -= 1/6 * einsum('VZ,Yxyz,UXxWyz->XUVYWZ', np.identity(ncas), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_aab_aab += 1/6 * einsum('VZ,Yxyz,UXxyWz->XUVYWZ', np.identity(ncas), v_aaaa, rdm_cccaaa, optimize = einsum_type)

    K22_baa_baa  = 1/3 * einsum('VZ,UXWY->XUVYWZ', h_aa, rdm_ccaa, optimize = einsum_type)
    K22_baa_baa += 1/6 * einsum('VZ,UXYW->XUVYWZ', h_aa, rdm_ccaa, optimize = einsum_type)
    K22_baa_baa -= 1/6 * einsum('Wx,UXZVYx->XUVYWZ', h_aa, rdm_cccaaa, optimize = einsum_type)
    K22_baa_baa -= 1/12 * einsum('Wx,UXZVxY->XUVYWZ', h_aa, rdm_cccaaa, optimize = einsum_type)
    K22_baa_baa -= 1/12 * einsum('Wx,UXZYVx->XUVYWZ', h_aa, rdm_cccaaa, optimize = einsum_type)
    K22_baa_baa += 1/12 * einsum('Wx,UXZxYV->XUVYWZ', h_aa, rdm_cccaaa, optimize = einsum_type)
    K22_baa_baa -= 1/12 * einsum('Yx,UXZVxW->XUVYWZ', h_aa, rdm_cccaaa, optimize = einsum_type)
    K22_baa_baa += 1/12 * einsum('Yx,UXZWVx->XUVYWZ', h_aa, rdm_cccaaa, optimize = einsum_type)
    K22_baa_baa += 1/6 * einsum('Yx,UXZWxV->XUVYWZ', h_aa, rdm_cccaaa, optimize = einsum_type)
    K22_baa_baa += 1/12 * einsum('Yx,UXZxWV->XUVYWZ', h_aa, rdm_cccaaa, optimize = einsum_type)
    K22_baa_baa += 1/12 * einsum('Zx,UXxVYW->XUVYWZ', h_aa, rdm_cccaaa, optimize = einsum_type)
    K22_baa_baa -= 1/12 * einsum('Zx,UXxWVY->XUVYWZ', h_aa, rdm_cccaaa, optimize = einsum_type)
    K22_baa_baa -= 1/6 * einsum('Zx,UXxWYV->XUVYWZ', h_aa, rdm_cccaaa, optimize = einsum_type)
    K22_baa_baa -= 1/12 * einsum('Zx,UXxYWV->XUVYWZ', h_aa, rdm_cccaaa, optimize = einsum_type)
    K22_baa_baa -= 1/24 * einsum('VWxy,UXZYxy->XUVYWZ', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_baa_baa += 1/24 * einsum('VWxy,UXZYyx->XUVYWZ', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_baa_baa -= 1/8 * einsum('VWxy,UXZxYy->XUVYWZ', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_baa_baa -= 1/24 * einsum('VWxy,UXZxyY->XUVYWZ', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_baa_baa += 1/8 * einsum('VWxy,UXZyYx->XUVYWZ', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_baa_baa += 1/24 * einsum('VWxy,UXZyxY->XUVYWZ', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_baa_baa += 1/24 * einsum('VYxy,UXZWxy->XUVYWZ', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_baa_baa += 1/8 * einsum('VYxy,UXZWyx->XUVYWZ', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_baa_baa -= 1/24 * einsum('VYxy,UXZxWy->XUVYWZ', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_baa_baa -= 1/8 * einsum('VYxy,UXZxyW->XUVYWZ', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_baa_baa += 1/24 * einsum('VYxy,UXZyWx->XUVYWZ', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_baa_baa -= 1/24 * einsum('VYxy,UXZyxW->XUVYWZ', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_baa_baa += 1/3 * einsum('VxZy,UXxWYy->XUVYWZ', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_baa_baa += 1/6 * einsum('VxZy,UXxYWy->XUVYWZ', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_baa_baa -= 1/6 * einsum('VxyZ,UXxWYy->XUVYWZ', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_baa_baa -= 1/12 * einsum('VxyZ,UXxWyY->XUVYWZ', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_baa_baa -= 1/12 * einsum('VxyZ,UXxYWy->XUVYWZ', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_baa_baa += 1/12 * einsum('VxyZ,UXxyYW->XUVYWZ', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_baa_baa -= 1/24 * einsum('WYxy,UXZVxy->XUVYWZ', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_baa_baa -= 1/8 * einsum('WYxy,UXZVyx->XUVYWZ', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_baa_baa += 1/24 * einsum('WYxy,UXZxVy->XUVYWZ', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_baa_baa += 1/8 * einsum('WYxy,UXZxyV->XUVYWZ', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_baa_baa -= 1/24 * einsum('WYxy,UXZyVx->XUVYWZ', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_baa_baa += 1/24 * einsum('WYxy,UXZyxV->XUVYWZ', v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_baa_baa -= 3/10 * einsum('Wxyz,UXZxVYyz->XUVYWZ', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_baa_baa += 1/10 * einsum('Wxyz,UXZxVYzy->XUVYWZ', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_baa_baa -= 3/20 * einsum('Wxyz,UXZxVyYz->XUVYWZ', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_baa_baa += 1/30 * einsum('Wxyz,UXZxVyzY->XUVYWZ', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_baa_baa += 1/10 * einsum('Wxyz,UXZxVzYy->XUVYWZ', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_baa_baa -= 1/30 * einsum('Wxyz,UXZxVzyY->XUVYWZ', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_baa_baa -= 11/40 * einsum('Wxyz,UXZxYVyz->XUVYWZ', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_baa_baa += 1/24 * einsum('Wxyz,UXZxYVzy->XUVYWZ', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_baa_baa -= 2/15 * einsum('Wxyz,UXZxYyVz->XUVYWZ', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_baa_baa -= 1/20 * einsum('Wxyz,UXZxYyzV->XUVYWZ', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_baa_baa += 1/30 * einsum('Wxyz,UXZxYzVy->XUVYWZ', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_baa_baa -= 7/60 * einsum('Wxyz,UXZxYzyV->XUVYWZ', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_baa_baa -= 3/40 * einsum('Wxyz,UXZxyVYz->XUVYWZ', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_baa_baa += 1/40 * einsum('Wxyz,UXZxyVzY->XUVYWZ', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_baa_baa += 1/60 * einsum('Wxyz,UXZxyzVY->XUVYWZ', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_baa_baa += 7/120 * einsum('Wxyz,UXZxzVYy->XUVYWZ', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_baa_baa -= 3/40 * einsum('Wxyz,UXZxzVyY->XUVYWZ', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_baa_baa += 1/20 * einsum('Wxyz,UXZxzYVy->XUVYWZ', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_baa_baa -= 1/10 * einsum('Wxyz,UXZxzYyV->XUVYWZ', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_baa_baa -= 1/60 * einsum('Wxyz,UXZxzyVY->XUVYWZ', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_baa_baa -= 1/30 * einsum('Wxyz,UXZxzyYV->XUVYWZ', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_baa_baa += 1/6 * einsum('Yxyz,UXZxVWzy->XUVYWZ', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_baa_baa -= 1/10 * einsum('Yxyz,UXZxVyWz->XUVYWZ', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_baa_baa += 7/40 * einsum('Yxyz,UXZxVyzW->XUVYWZ', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_baa_baa += 3/20 * einsum('Yxyz,UXZxVzWy->XUVYWZ', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_baa_baa += 7/40 * einsum('Yxyz,UXZxVzyW->XUVYWZ', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_baa_baa -= 1/120 * einsum('Yxyz,UXZxWVyz->XUVYWZ', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_baa_baa += 3/40 * einsum('Yxyz,UXZxWVzy->XUVYWZ', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_baa_baa -= 11/120 * einsum('Yxyz,UXZxyVWz->XUVYWZ', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_baa_baa += 1/10 * einsum('Yxyz,UXZxyVzW->XUVYWZ', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_baa_baa -= 1/15 * einsum('Yxyz,UXZxyWVz->XUVYWZ', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_baa_baa += 1/60 * einsum('Yxyz,UXZxyWzV->XUVYWZ', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_baa_baa += 1/40 * einsum('Yxyz,UXZxyzVW->XUVYWZ', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_baa_baa += 3/40 * einsum('Yxyz,UXZxzVWy->XUVYWZ', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_baa_baa += 1/10 * einsum('Yxyz,UXZxzVyW->XUVYWZ', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_baa_baa += 1/60 * einsum('Yxyz,UXZxzWVy->XUVYWZ', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_baa_baa += 1/60 * einsum('Yxyz,UXZxzWyV->XUVYWZ', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_baa_baa += 1/40 * einsum('Yxyz,UXZxzyVW->XUVYWZ', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_baa_baa += 2/15 * einsum('Zxyz,UXyzVWYx->XUVYWZ', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_baa_baa -= 1/30 * einsum('Zxyz,UXyzVWxY->XUVYWZ', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_baa_baa += 3/10 * einsum('Zxyz,UXyzVYWx->XUVYWZ', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_baa_baa -= 1/10 * einsum('Zxyz,UXyzVYxW->XUVYWZ', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_baa_baa += 1/20 * einsum('Zxyz,UXyzVxWY->XUVYWZ', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_baa_baa -= 1/10 * einsum('Zxyz,UXyzVxYW->XUVYWZ', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_baa_baa += 1/15 * einsum('Zxyz,UXyzWVYx->XUVYWZ', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_baa_baa -= 1/60 * einsum('Zxyz,UXyzWVxY->XUVYWZ', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_baa_baa += 11/40 * einsum('Zxyz,UXyzYVWx->XUVYWZ', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_baa_baa -= 1/24 * einsum('Zxyz,UXyzYVxW->XUVYWZ', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_baa_baa += 1/8 * einsum('Zxyz,UXyzYWVx->XUVYWZ', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_baa_baa += 1/24 * einsum('Zxyz,UXyzYWxV->XUVYWZ', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_baa_baa -= 1/40 * einsum('Zxyz,UXyzYxVW->XUVYWZ', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_baa_baa += 1/8 * einsum('Zxyz,UXyzYxWV->XUVYWZ', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_baa_baa += 1/12 * einsum('Zxyz,UXyzxVWY->XUVYWZ', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_baa_baa -= 1/15 * einsum('Zxyz,UXyzxVYW->XUVYWZ', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_baa_baa += 1/60 * einsum('Zxyz,UXyzxWVY->XUVYWZ', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_baa_baa += 1/60 * einsum('Zxyz,UXyzxWYV->XUVYWZ', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_baa_baa -= 1/20 * einsum('Zxyz,UXyzxYVW->XUVYWZ', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_baa_baa += 1/10 * einsum('Zxyz,UXyzxYWV->XUVYWZ', v_aaaa, rdm_ccccaaaa, optimize = einsum_type)
    K22_baa_baa -= 1/6 * einsum('Wx,VZ,UXYx->XUVYWZ', h_aa, np.identity(ncas), rdm_ccaa, optimize = einsum_type)
    K22_baa_baa -= 1/3 * einsum('Wx,VZ,UXxY->XUVYWZ', h_aa, np.identity(ncas), rdm_ccaa, optimize = einsum_type)
    K22_baa_baa -= 1/3 * einsum('Yx,VZ,UXWx->XUVYWZ', h_aa, np.identity(ncas), rdm_ccaa, optimize = einsum_type)
    K22_baa_baa -= 1/6 * einsum('Yx,VZ,UXxW->XUVYWZ', h_aa, np.identity(ncas), rdm_ccaa, optimize = einsum_type)
    K22_baa_baa -= 1/3 * einsum('VZ,WYxy,UXxy->XUVYWZ', np.identity(ncas), v_aaaa, rdm_ccaa, optimize = einsum_type)
    K22_baa_baa -= 1/6 * einsum('VZ,WYxy,UXyx->XUVYWZ', np.identity(ncas), v_aaaa, rdm_ccaa, optimize = einsum_type)
    K22_baa_baa -= 1/12 * einsum('VZ,Wxyz,UXxYyz->XUVYWZ', np.identity(ncas), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_baa_baa += 1/12 * einsum('VZ,Wxyz,UXxYzy->XUVYWZ', np.identity(ncas), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_baa_baa -= 1/4 * einsum('VZ,Wxyz,UXxyYz->XUVYWZ', np.identity(ncas), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_baa_baa += 1/12 * einsum('VZ,Wxyz,UXxyzY->XUVYWZ', np.identity(ncas), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_baa_baa += 1/12 * einsum('VZ,Wxyz,UXxzYy->XUVYWZ', np.identity(ncas), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_baa_baa += 1/12 * einsum('VZ,Wxyz,UXxzyY->XUVYWZ', np.identity(ncas), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_baa_baa -= 1/4 * einsum('VZ,Yxyz,UXxWyz->XUVYWZ', np.identity(ncas), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_baa_baa += 1/12 * einsum('VZ,Yxyz,UXxWzy->XUVYWZ', np.identity(ncas), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_baa_baa -= 1/12 * einsum('VZ,Yxyz,UXxyWz->XUVYWZ', np.identity(ncas), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_baa_baa += 1/12 * einsum('VZ,Yxyz,UXxyzW->XUVYWZ', np.identity(ncas), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_baa_baa += 1/12 * einsum('VZ,Yxyz,UXxzWy->XUVYWZ', np.identity(ncas), v_aaaa, rdm_cccaaa, optimize = einsum_type)
    K22_baa_baa += 1/12 * einsum('VZ,Yxyz,UXxzyW->XUVYWZ', np.identity(ncas), v_aaaa, rdm_cccaaa, optimize = einsum_type)

    K22[::2,::2,1::2,::2,::2,1::2] = K22_aab_aab.copy()
    K22[1::2,1::2,::2,1::2,1::2,::2] = K22_aab_aab.copy()

    K22[1::2,::2,::2,1::2,::2,::2] = K22_baa_baa.copy()
    K22[::2,1::2,1::2,::2,1::2,1::2] = K22_baa_baa.copy()

    K22[1::2,::2,::2,::2,1::2,::2] -= K22_baa_baa.transpose(0,1,2,4,3,5).copy()
    K22[::2,1::2,1::2,1::2,::2,1::2] = K22[1::2,::2,::2,::2,1::2,::2].copy()

    K22[::2,1::2,::2,1::2,::2,::2] -= K22_baa_baa.transpose(1,0,2,3,4,5).copy()
    K22[1::2,::2,1::2,::2,1::2,1::2] = K22[::2,1::2,::2,1::2,::2,::2].copy()

    K22[::2,1::2,::2,::2,1::2,::2] = K22_baa_baa.transpose(1,0,2,4,3,5).copy()
    K22[1::2,::2,1::2,1::2,::2,1::2] = K22[::2,1::2,::2,::2,1::2,::2].copy()

    K22[::2,::2,::2,::2,1::2,1::2]  = K22_baa_baa.copy()
    K22[::2,::2,::2,::2,1::2,1::2] -= K22_aab_aab.copy()
    K22[::2,::2,::2,::2,1::2,1::2] += K22[::2,1::2,::2,1::2,::2,::2].copy()
    K22[1::2,1::2,1::2,1::2,::2,::2] = K22[::2,::2,::2,::2,1::2,1::2].copy()

    K22[::2,::2,::2,1::2,::2,1::2] -= K22[::2,::2,::2,::2,1::2,1::2].transpose(0,1,2,4,3,5).copy()
    K22[1::2,1::2,1::2,::2,1::2,::2] = K22[::2,::2,::2,1::2,::2,1::2].copy()

    K22[::2,1::2,1::2,::2,::2,::2]  = K22_baa_baa.copy()
    K22[::2,1::2,1::2,::2,::2,::2] -= K22_aab_aab.copy()
    K22[::2,1::2,1::2,::2,::2,::2] += K22[1::2,::2,::2,::2,1::2,::2].copy()
    K22[1::2,::2,::2,1::2,1::2,1::2] = K22[::2,1::2,1::2,::2,::2,::2].copy()

    K22[1::2,::2,1::2,::2,::2,::2] -= K22[::2,1::2,1::2,::2,::2,::2].transpose(1,0,2,3,4,5).copy()
    K22[::2,1::2,::2,1::2,1::2,1::2] = K22[1::2,::2,1::2,::2,::2,::2].copy()

    K22[::2,::2,::2,::2,::2,::2]  = K22[::2,::2,::2,::2,1::2,1::2].copy()
    K22[::2,::2,::2,::2,::2,::2] += K22[::2,1::2,1::2,1::2,::2,1::2].copy()
    K22[::2,::2,::2,::2,::2,::2] += K22[1::2,::2,1::2,1::2,::2,1::2].copy()
    K22[1::2,1::2,1::2,1::2,1::2,1::2] = K22[::2,::2,::2,::2,::2,::2].copy()

    K12 = K12[:,aa_ind[0],aa_ind[1]]
    K22 = K22[:,:,:,aa_ind[0],aa_ind[1]]
    K22 = K22[aa_ind[0],aa_ind[1]]

    K_m1p = np.zeros((dim_act, dim_act))

    K_m1p[:n_x,:n_x] = K11.copy()
    K_m1p[:n_x,n_x:] = K12.reshape(n_x, n_xzw).copy()
    K_m1p[n_x:,:n_x] = K12.reshape(n_x, n_xzw).T.copy()
    K_m1p[n_x:,n_x:] = K22.reshape(n_xzw, n_xzw).copy()

    return K_m1p
