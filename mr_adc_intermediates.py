import numpy as np

def compute_K_ac(mr_adc):

    # Einsum
    einsum = mr_adc.interface.einsum
    einsum_type = mr_adc.interface.einsum_type

    rdm_ca = mr_adc.rdm.ca
    rdm_ccaa = mr_adc.rdm.ccaa

    h1eff = mr_adc.h1eff_act
    v2e_act = mr_adc.v2e.aaaa

    ## DEBUG
    h_aa = mr_adc.h1e[mr_adc.ncore:mr_adc.nocc, mr_adc.ncore:mr_adc.nocc].copy()
    v_aaaa = mr_adc.v2e.aaaa
    v_caca = mr_adc.v2e.caca

    import prism.mr_adc_integrals as mr_adc_integrals
    mo_c = mr_adc.mo[:, :mr_adc.ncore].copy()
    mo_a = mr_adc.mo[:, mr_adc.ncore:mr_adc.nocc].copy()
    v_acca = mr_adc_integrals.transform_2e_phys_incore(mr_adc.interface, mo_a, mo_c, mo_c, mo_a)

    K_ac  = einsum('XY->XY', h_aa, optimize = einsum_type).copy()
    K_ac -= einsum('XiiY->XY', v_acca, optimize = einsum_type).copy()
    K_ac += 2.0 * einsum('iXiY->XY', v_caca, optimize = einsum_type).copy()
    K_ac -= 0.5 * einsum('Yz,zX->XY', h_aa, rdm_ca, optimize = einsum_type)
    K_ac += einsum('XzYw,zw->XY', v_aaaa, rdm_ca, optimize = einsum_type)
    K_ac += 0.5 * einsum('Yiiz,zX->XY', v_acca, rdm_ca, optimize = einsum_type)
    K_ac -= 0.0833333333333 * einsum('Yzwu,uwXz->XY', v_aaaa, rdm_ccaa, optimize = einsum_type)
    K_ac -= 0.416666666667 * einsum('Yzwu,wuXz->XY', v_aaaa, rdm_ccaa, optimize = einsum_type)
    K_ac -= einsum('iYiz,zX->XY', v_caca, rdm_ca, optimize = einsum_type)
    K_ac -= 0.5 * einsum('zXYw,zw->XY', v_aaaa, rdm_ca, optimize = einsum_type)
    K_ac -= 0.0833333333333 * einsum('zYwu,uwXz->XY', v_aaaa, rdm_ccaa, optimize = einsum_type)
    K_ac += 0.0833333333333 * einsum('zYwu,wuXz->XY', v_aaaa, rdm_ccaa, optimize = einsum_type)
    print ("\n>>> SQA K_ac: {:}".format(np.linalg.norm(K_ac)))

    return K_ac
