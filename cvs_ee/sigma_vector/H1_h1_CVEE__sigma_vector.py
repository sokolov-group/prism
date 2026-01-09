## h1 <- h1 coupling contributions
from . import logger, tools, ascontiguousarray, zeros
from prism.mr_adc_integrals import get_eeee_df, unpack_v2e_eeee

# CCAA <- CVEE: NO CONTRIBUTION

# CCEA <- CVEE
def compute_sigma_vector__H1__h1_h1__CCEA_CVEE(mr_adc, X, sigma):
    cput1 = (logger.process_clock(), logger.perf_counter())

    # Einsum definition from kernel
    einsum = mr_adc.interface.einsum
    einsum_type = mr_adc.interface.einsum_type

    #Excitation Manifold
    ccea = mr_adc.h1.ccea

    ## Two-electron integrals
    v_xaev = mr_adc.v2e.xaev
    v_vxae = mr_adc.v2e.vxae

    # Reduced Density Matrices
    rdm_ca = mr_adc.rdm.ca

    sigma_KLCW  = 2 * einsum('KiCa,LWai->KLCW', X, v_xaev, optimize = einsum_type)
    sigma_KLCW -= einsum('KiCa,iLWa->KLCW', X, v_vxae, optimize = einsum_type)
    sigma_KLCW -= einsum('KiaC,LWai->KLCW', X, v_xaev, optimize = einsum_type)
    sigma_KLCW -= einsum('LiaC,iKWa->KLCW', X, v_vxae, optimize = einsum_type)
    sigma_KLCW -= einsum('KiCa,Lxai,Wx->KLCW', X, v_xaev, rdm_ca, optimize = einsum_type)
    sigma_KLCW += 1/2 * einsum('KiCa,iLxa,Wx->KLCW', X, v_vxae, rdm_ca, optimize = einsum_type)
    sigma_KLCW += 1/2 * einsum('KiaC,Lxai,Wx->KLCW', X, v_xaev, rdm_ca, optimize = einsum_type)
    sigma_KLCW += 1/2 * einsum('LiaC,iKxa,Wx->KLCW', X, v_vxae, rdm_ca, optimize = einsum_type)
    sigma[ccea] += ascontiguousarray(sigma_KLCW).reshape(-1)

    mr_adc.log.timer_debug("computing sigma H1 h1-h1 CCEA-CVEE", *cput1)

# CCEE <- CVEE
def compute_sigma_vector__H1__h1_h1__CCEE_CVEE(mr_adc, X, sigma):
    cput1 = (logger.process_clock(), logger.perf_counter())

    # Einsum definition from kernel
    einsum = mr_adc.interface.einsum
    einsum_type = mr_adc.interface.einsum_type

    #Excitation Manifold
    ccee = mr_adc.h1.ccee

    ## Two-electron integrals
    v_xxxv = mr_adc.v2e.xxxv
    v_xvxx = mr_adc.v2e.xvxx
    v_xeev = mr_adc.v2e.xeev
    v_vxee = mr_adc.v2e.vxee

    sigma_KLCD  = 2 * einsum('KiCa,LDai->KLCD', X, v_xeev, optimize = einsum_type)
    sigma_KLCD -= einsum('KiCa,iLDa->KLCD', X, v_vxee, optimize = einsum_type)
    sigma_KLCD -= einsum('KiaC,LDai->KLCD', X, v_xeev, optimize = einsum_type)
    sigma_KLCD -= einsum('KiaD,iLCa->KLCD', X, v_vxee, optimize = einsum_type)
    sigma_KLCD += 2 * einsum('LiDa,KCai->KLCD', X, v_xeev, optimize = einsum_type)
    sigma_KLCD -= einsum('LiDa,iKCa->KLCD', X, v_vxee, optimize = einsum_type)
    sigma_KLCD -= einsum('LiaC,iKDa->KLCD', X, v_vxee, optimize = einsum_type)
    sigma_KLCD -= einsum('LiaD,KCai->KLCD', X, v_xeev, optimize = einsum_type)
    sigma_KLCD += einsum('ijCD,KiLj->KLCD', X, v_xxxv, optimize = einsum_type)
    sigma_KLCD += einsum('ijDC,KjLi->KLCD', X, v_xvxx, optimize = einsum_type)
    sigma[ccee] += ascontiguousarray(sigma_KLCD).reshape(-1)

    mr_adc.log.timer_debug("computing sigma H1 h1-h1 CCEE-CVEE", *cput1)

# CAEE <- CVEE
def compute_sigma_vector__H1__h1_h1__CAEE_CVEE(mr_adc, X, sigma):
    cput1 = (logger.process_clock(), logger.perf_counter())

    # Einsum definition from kernel
    einsum = mr_adc.interface.einsum
    einsum_type = mr_adc.interface.einsum_type

    #Excitation Manifold
    caee = mr_adc.h1.caee

    ## Molecular Orbitals Energies
    e_val = mr_adc.mo_energy.v

    ## One-electron integrals
    h_va = mr_adc.h1eff.va
    h_aa = mr_adc.h1eff.aa

    ## Two-electron integrals
    v_xxva = mr_adc.v2e.xxva
    v_vxxa = mr_adc.v2e.vxxa
    v_vaaa = mr_adc.v2e.vaaa
    v_vaee = mr_adc.v2e.vaee
    v_veea = mr_adc.v2e.veea
    v_aaaa = mr_adc.v2e.aaaa
 
    ## Amplitudes
    t1_va = mr_adc.t1.va
    t1_vaaa = mr_adc.t1.vaaa

    # Reduced Density Matrices
    rdm_ca = mr_adc.rdm.ca
    rdm_ccaa = mr_adc.rdm.ccaa

    sigma_KWCD =- einsum('KiCD,iW->KWCD', X, h_va, optimize = einsum_type)
    sigma_KWCD += einsum('KiCD,i,iW->KWCD', X, e_val, t1_va, optimize = einsum_type)
    sigma_KWCD -= einsum('KiCD,Wx,ix->KWCD', X, h_aa, t1_va, optimize = einsum_type)
    sigma_KWCD -= einsum('KiCD,iWxy,yx->KWCD', X, v_vaaa, rdm_ca, optimize = einsum_type)
    sigma_KWCD += 1/2 * einsum('KiCD,ixyW,xy->KWCD', X, v_vaaa, rdm_ca, optimize = einsum_type)
    sigma_KWCD += einsum('KiCa,iaDx,Wx->KWCD', X, v_veea, rdm_ca, optimize = einsum_type)
    sigma_KWCD -= 1/2 * einsum('KiCa,ixDa,Wx->KWCD', X, v_vaee, rdm_ca, optimize = einsum_type)
    sigma_KWCD -= 1/2 * einsum('KiaC,iaDx,Wx->KWCD', X, v_veea, rdm_ca, optimize = einsum_type)
    sigma_KWCD -= 1/2 * einsum('KiaD,ixCa,Wx->KWCD', X, v_vaee, rdm_ca, optimize = einsum_type)
    sigma_KWCD += 1/2 * einsum('ijCD,iKjx,Wx->KWCD', X, v_xxva, rdm_ca, optimize = einsum_type)
    sigma_KWCD += 1/2 * einsum('ijDC,jKix,Wx->KWCD', X, v_vxxa, rdm_ca, optimize = einsum_type)
    sigma_KWCD += einsum('KiCD,i,ixWy,yx->KWCD', X, e_val, t1_vaaa, rdm_ca, optimize = einsum_type)
    sigma_KWCD -= 1/2 * einsum('KiCD,i,ixyW,yx->KWCD', X, e_val, t1_vaaa, rdm_ca, optimize = einsum_type)
    sigma_KWCD -= einsum('KiCD,Wx,iyxz,yz->KWCD', X, h_aa, t1_vaaa, rdm_ca, optimize = einsum_type)
    sigma_KWCD += 1/2 * einsum('KiCD,Wx,iyzx,yz->KWCD', X, h_aa, t1_vaaa, rdm_ca, optimize = einsum_type)
    sigma_KWCD += einsum('KiCD,xy,ixWz,yz->KWCD', X, h_aa, t1_vaaa, rdm_ca, optimize = einsum_type)
    sigma_KWCD -= 1/2 * einsum('KiCD,xy,ixzW,yz->KWCD', X, h_aa, t1_vaaa, rdm_ca, optimize = einsum_type)
    sigma_KWCD -= einsum('KiCD,xy,izWx,yz->KWCD', X, h_aa, t1_vaaa, rdm_ca, optimize = einsum_type)
    sigma_KWCD += 1/2 * einsum('KiCD,xy,izxW,yz->KWCD', X, h_aa, t1_vaaa, rdm_ca, optimize = einsum_type)
    sigma_KWCD -= einsum('KiCD,ix,Wxyz,yz->KWCD', X, t1_va, v_aaaa, rdm_ca, optimize = einsum_type)
    sigma_KWCD += 1/2 * einsum('KiCD,ix,Wyzx,zy->KWCD', X, t1_va, v_aaaa, rdm_ca, optimize = einsum_type)
    sigma_KWCD += einsum('KiCD,ixWy,xzwu,ywzu->KWCD', X, t1_vaaa, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWCD -= einsum('KiCD,ixWy,yzwu,xwzu->KWCD', X, t1_vaaa, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWCD -= 1/2 * einsum('KiCD,ixyW,xzwu,ywzu->KWCD', X, t1_vaaa, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWCD += 1/2 * einsum('KiCD,ixyW,yzwu,xwzu->KWCD', X, t1_vaaa, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWCD += 1/2 * einsum('KiCD,ixyz,Wwuy,xwzu->KWCD', X, t1_vaaa, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWCD += 1/2 * einsum('KiCD,ixyz,Wwuz,xwuy->KWCD', X, t1_vaaa, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWCD -= 1/2 * einsum('KiCD,ixyz,Wwxu,yzwu->KWCD', X, t1_vaaa, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWCD -= einsum('KiCD,ixyz,Wywu,xuzw->KWCD', X, t1_vaaa, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWCD -= einsum('KiCD,ixyz,Wywz,xw->KWCD', X, t1_vaaa, v_aaaa, rdm_ca, optimize = einsum_type)
    sigma_KWCD += 1/2 * einsum('KiCD,ixyz,Wzwu,xuyw->KWCD', X, t1_vaaa, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWCD += 1/2 * einsum('KiCD,ixyz,Wzwy,xw->KWCD', X, t1_vaaa, v_aaaa, rdm_ca, optimize = einsum_type)
    sigma[caee] += ascontiguousarray(sigma_KWCD).reshape(-1)

    mr_adc.log.timer_debug("computing sigma H1 h1-h1 CAEE-CVEE", *cput1)

# CAAA <- CVEE
def compute_sigma_vector__H1__h1_h1__CAAA_CVEE(mr_adc, X, sigma):
    cput1 = (logger.process_clock(), logger.perf_counter())

    # Einsum definition from kernel
    einsum = mr_adc.interface.einsum
    einsum_type = mr_adc.interface.einsum_type

    #Excitation Manifold
    caaa__aaaa = mr_adc.h1.caaa__aaaa
    caaa__abab = mr_adc.h1.caaa__abab

    ## Indices
    aa_tril_ind = mr_adc.h1.aa_tril_ind 

    ## Molecular Orbitals Energies
    e_val = mr_adc.mo_energy.v
    e_extern = mr_adc.mo_energy.e

    ## One-electron integrals
    h_aa = mr_adc.h1eff.aa

    ## Two-electron integrals
    v_veae = mr_adc.v2e.veae
    v_aaaa = mr_adc.v2e.aaaa
  
    ## Amplitudes
    t1_vaee = mr_adc.t1.vaee
  
    # Reduced Density Matrices
    rdm_ca = mr_adc.rdm.ca
    rdm_ccaa = mr_adc.rdm.ccaa

    sigma_KWUV_aaaa =- 1/2 * einsum('Kiab,iaUb,VW->KWUV', X, v_veae, rdm_ca, optimize = einsum_type)
    sigma_KWUV_aaaa += 1/2 * einsum('Kiab,iaVb,UW->KWUV', X, v_veae, rdm_ca, optimize = einsum_type)
    sigma_KWUV_aaaa += einsum('Kiab,ibUa,VW->KWUV', X, v_veae, rdm_ca, optimize = einsum_type)
    sigma_KWUV_aaaa -= einsum('Kiab,ibVa,UW->KWUV', X, v_veae, rdm_ca, optimize = einsum_type)
    sigma_KWUV_aaaa -= 1/2 * einsum('Kiab,a,iUab,VW->KWUV', X, e_extern, t1_vaee, rdm_ca, optimize = einsum_type)
    sigma_KWUV_aaaa += einsum('Kiab,a,iUba,VW->KWUV', X, e_extern, t1_vaee, rdm_ca, optimize = einsum_type)
    sigma_KWUV_aaaa += 1/2 * einsum('Kiab,a,iVab,UW->KWUV', X, e_extern, t1_vaee, rdm_ca, optimize = einsum_type)
    sigma_KWUV_aaaa -= einsum('Kiab,a,iVba,UW->KWUV', X, e_extern, t1_vaee, rdm_ca, optimize = einsum_type)
    sigma_KWUV_aaaa -= 1/2 * einsum('Kiab,b,iUab,VW->KWUV', X, e_extern, t1_vaee, rdm_ca, optimize = einsum_type)
    sigma_KWUV_aaaa += einsum('Kiab,b,iUba,VW->KWUV', X, e_extern, t1_vaee, rdm_ca, optimize = einsum_type)
    sigma_KWUV_aaaa += 1/2 * einsum('Kiab,b,iVab,UW->KWUV', X, e_extern, t1_vaee, rdm_ca, optimize = einsum_type)
    sigma_KWUV_aaaa -= einsum('Kiab,b,iVba,UW->KWUV', X, e_extern, t1_vaee, rdm_ca, optimize = einsum_type)
    sigma_KWUV_aaaa += 1/2 * einsum('Kiab,i,iUab,VW->KWUV', X, e_val, t1_vaee, rdm_ca, optimize = einsum_type)
    sigma_KWUV_aaaa -= einsum('Kiab,i,iUba,VW->KWUV', X, e_val, t1_vaee, rdm_ca, optimize = einsum_type)
    sigma_KWUV_aaaa -= 1/2 * einsum('Kiab,i,iVab,UW->KWUV', X, e_val, t1_vaee, rdm_ca, optimize = einsum_type)
    sigma_KWUV_aaaa += einsum('Kiab,i,iVba,UW->KWUV', X, e_val, t1_vaee, rdm_ca, optimize = einsum_type)
    sigma_KWUV_aaaa += 1/2 * einsum('Kiab,Ux,ixab,VW->KWUV', X, h_aa, t1_vaee, rdm_ca, optimize = einsum_type)
    sigma_KWUV_aaaa -= einsum('Kiab,Ux,ixba,VW->KWUV', X, h_aa, t1_vaee, rdm_ca, optimize = einsum_type)
    sigma_KWUV_aaaa -= 1/2 * einsum('Kiab,Vx,ixab,UW->KWUV', X, h_aa, t1_vaee, rdm_ca, optimize = einsum_type)
    sigma_KWUV_aaaa += einsum('Kiab,Vx,ixba,UW->KWUV', X, h_aa, t1_vaee, rdm_ca, optimize = einsum_type)
    sigma_KWUV_aaaa += 1/2 * einsum('Kiab,ixab,UxVy,Wy->KWUV', X, t1_vaee, v_aaaa, rdm_ca, optimize = einsum_type)
    sigma_KWUV_aaaa += 1/2 * einsum('Kiab,ixab,Uxyz,VzWy->KWUV', X, t1_vaee, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWUV_aaaa -= 1/2 * einsum('Kiab,ixab,UyVx,Wy->KWUV', X, t1_vaee, v_aaaa, rdm_ca, optimize = einsum_type)
    sigma_KWUV_aaaa -= 1/6 * einsum('Kiab,ixab,Uyzx,VyWz->KWUV', X, t1_vaee, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWUV_aaaa += 1/6 * einsum('Kiab,ixab,Uyzx,VyzW->KWUV', X, t1_vaee, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWUV_aaaa -= 1/2 * einsum('Kiab,ixab,Vxyz,UzWy->KWUV', X, t1_vaee, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWUV_aaaa += 1/6 * einsum('Kiab,ixab,Vyzx,UyWz->KWUV', X, t1_vaee, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWUV_aaaa -= 1/6 * einsum('Kiab,ixab,Vyzx,UyzW->KWUV', X, t1_vaee, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWUV_aaaa -= 1/6 * einsum('Kiab,ixab,Wyxz,UVyz->KWUV', X, t1_vaee, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWUV_aaaa += 1/6 * einsum('Kiab,ixab,Wyxz,UVzy->KWUV', X, t1_vaee, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWUV_aaaa -= einsum('Kiab,ixba,UxVy,Wy->KWUV', X, t1_vaee, v_aaaa, rdm_ca, optimize = einsum_type)
    sigma_KWUV_aaaa -= einsum('Kiab,ixba,Uxyz,VzWy->KWUV', X, t1_vaee, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWUV_aaaa += einsum('Kiab,ixba,UyVx,Wy->KWUV', X, t1_vaee, v_aaaa, rdm_ca, optimize = einsum_type)
    sigma_KWUV_aaaa += 1/3 * einsum('Kiab,ixba,Uyzx,VyWz->KWUV', X, t1_vaee, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWUV_aaaa -= 1/3 * einsum('Kiab,ixba,Uyzx,VyzW->KWUV', X, t1_vaee, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWUV_aaaa += einsum('Kiab,ixba,Vxyz,UzWy->KWUV', X, t1_vaee, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWUV_aaaa -= 1/3 * einsum('Kiab,ixba,Vyzx,UyWz->KWUV', X, t1_vaee, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWUV_aaaa += 1/3 * einsum('Kiab,ixba,Vyzx,UyzW->KWUV', X, t1_vaee, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWUV_aaaa += 1/3 * einsum('Kiab,ixba,Wyxz,UVyz->KWUV', X, t1_vaee, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWUV_aaaa -= 1/3 * einsum('Kiab,ixba,Wyxz,UVzy->KWUV', X, t1_vaee, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma[caaa__aaaa] += ascontiguousarray(sigma_KWUV_aaaa[:, :, aa_tril_ind[0], aa_tril_ind[1]]).reshape(-1)

    sigma_KWUV_abab =- 1/2 * einsum('Kiab,iaUb,VW->KWUV', X, v_veae, rdm_ca, optimize = einsum_type)
    sigma_KWUV_abab += einsum('Kiab,ibUa,VW->KWUV', X, v_veae, rdm_ca, optimize = einsum_type)
    sigma_KWUV_abab -= 1/2 * einsum('Kiab,a,iUab,VW->KWUV', X, e_extern, t1_vaee, rdm_ca, optimize = einsum_type)
    sigma_KWUV_abab += einsum('Kiab,a,iUba,VW->KWUV', X, e_extern, t1_vaee, rdm_ca, optimize = einsum_type)
    sigma_KWUV_abab -= 1/2 * einsum('Kiab,b,iUab,VW->KWUV', X, e_extern, t1_vaee, rdm_ca, optimize = einsum_type)
    sigma_KWUV_abab += einsum('Kiab,b,iUba,VW->KWUV', X, e_extern, t1_vaee, rdm_ca, optimize = einsum_type)
    sigma_KWUV_abab += 1/2 * einsum('Kiab,i,iUab,VW->KWUV', X, e_val, t1_vaee, rdm_ca, optimize = einsum_type)
    sigma_KWUV_abab -= einsum('Kiab,i,iUba,VW->KWUV', X, e_val, t1_vaee, rdm_ca, optimize = einsum_type)
    sigma_KWUV_abab += 1/2 * einsum('Kiab,Ux,ixab,VW->KWUV', X, h_aa, t1_vaee, rdm_ca, optimize = einsum_type)
    sigma_KWUV_abab -= einsum('Kiab,Ux,ixba,VW->KWUV', X, h_aa, t1_vaee, rdm_ca, optimize = einsum_type)
    sigma_KWUV_abab += 1/2 * einsum('Kiab,ixab,UxVy,Wy->KWUV', X, t1_vaee, v_aaaa, rdm_ca, optimize = einsum_type)
    sigma_KWUV_abab += 1/2 * einsum('Kiab,ixab,Uxyz,VzWy->KWUV', X, t1_vaee, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWUV_abab -= 1/3 * einsum('Kiab,ixab,Uyzx,VyWz->KWUV', X, t1_vaee, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWUV_abab -= 1/6 * einsum('Kiab,ixab,Uyzx,VyzW->KWUV', X, t1_vaee, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWUV_abab -= 1/6 * einsum('Kiab,ixab,Vyzx,UyWz->KWUV', X, t1_vaee, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWUV_abab -= 1/3 * einsum('Kiab,ixab,Vyzx,UyzW->KWUV', X, t1_vaee, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWUV_abab += 1/6 * einsum('Kiab,ixab,Wyxz,UVyz->KWUV', X, t1_vaee, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWUV_abab += 1/3 * einsum('Kiab,ixab,Wyxz,UVzy->KWUV', X, t1_vaee, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWUV_abab -= einsum('Kiab,ixba,UxVy,Wy->KWUV', X, t1_vaee, v_aaaa, rdm_ca, optimize = einsum_type)
    sigma_KWUV_abab -= einsum('Kiab,ixba,Uxyz,VzWy->KWUV', X, t1_vaee, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWUV_abab += 2/3 * einsum('Kiab,ixba,Uyzx,VyWz->KWUV', X, t1_vaee, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWUV_abab += 1/3 * einsum('Kiab,ixba,Uyzx,VyzW->KWUV', X, t1_vaee, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWUV_abab += 1/3 * einsum('Kiab,ixba,Vyzx,UyWz->KWUV', X, t1_vaee, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWUV_abab += 2/3 * einsum('Kiab,ixba,Vyzx,UyzW->KWUV', X, t1_vaee, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWUV_abab -= 1/3 * einsum('Kiab,ixba,Wyxz,UVyz->KWUV', X, t1_vaee, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWUV_abab -= 2/3 * einsum('Kiab,ixba,Wyxz,UVzy->KWUV', X, t1_vaee, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma[caaa__abab] += ascontiguousarray(sigma_KWUV_abab).reshape(-1)

    mr_adc.log.timer_debug("computing sigma H1 h1-h1 CAAA-CVEE", *cput1)

# CAEA <- CVEE
def compute_sigma_vector__H1__h1_h1__CAEA_CVEE(mr_adc, X, sigma):
    cput1 = (logger.process_clock(), logger.perf_counter())

    # Einsum definition from kernel
    einsum = mr_adc.interface.einsum
    einsum_type = mr_adc.interface.einsum_type

    #Excitation Manifold
    caea__aaaa = mr_adc.h1.caea__aaaa
    caea__abab = mr_adc.h1.caea__abab
    caea__baab = mr_adc.h1.caea__baab

    ## Molecular Orbitals Energies
    e_val = mr_adc.mo_energy.v
    e_extern = mr_adc.mo_energy.e
 
    ## One-electron integrals
    h_aa = mr_adc.h1eff.aa
 
    ## Two-electron integrals
    v_xxve = mr_adc.v2e.xxve
    v_vxxe = mr_adc.v2e.vxxe
    v_vaae = mr_adc.v2e.vaae
    v_veaa = mr_adc.v2e.veaa
    #v_veee = mr_adc.v2e.veee
    v_aaaa = mr_adc.v2e.aaaa

    ## Amplitudes
    t1_vaae = mr_adc.t1.vaae
    t1_vaea = mr_adc.t1.vaea
   
    # Reduced Density Matrices
    rdm_ca = mr_adc.rdm.ca
    rdm_ccaa = mr_adc.rdm.ccaa

    sigma_KWCU_aaaa  = 1/2 * einsum('KiCa,iWxa,Ux->KWCU', X, v_vaae, rdm_ca, optimize = einsum_type)
    sigma_KWCU_aaaa += einsum('KiCa,iaUx,Wx->KWCU', X, v_veaa, rdm_ca, optimize = einsum_type)
    sigma_KWCU_aaaa -= einsum('KiCa,iaxW,Ux->KWCU', X, v_veaa, rdm_ca, optimize = einsum_type)
    sigma_KWCU_aaaa -= 1/2 * einsum('KiCa,ixUa,Wx->KWCU', X, v_vaae, rdm_ca, optimize = einsum_type)
    sigma_KWCU_aaaa -= 1/2 * einsum('KiaC,iWxa,Ux->KWCU', X, v_vaae, rdm_ca, optimize = einsum_type)
    sigma_KWCU_aaaa -= 1/2 * einsum('KiaC,iaUx,Wx->KWCU', X, v_veaa, rdm_ca, optimize = einsum_type)
    sigma_KWCU_aaaa += 1/2 * einsum('KiaC,iaxW,Ux->KWCU', X, v_veaa, rdm_ca, optimize = einsum_type)
    sigma_KWCU_aaaa += 1/2 * einsum('KiaC,ixUa,Wx->KWCU', X, v_vaae, rdm_ca, optimize = einsum_type)
    #sigma_KWCU_aaaa -= 1/2 * einsum('Kiab,iaCb,UW->KWCU', X, v_veee, rdm_ca, optimize = einsum_type)
    #sigma_KWCU_aaaa += einsum('Kiab,ibCa,UW->KWCU', X, v_veee, rdm_ca, optimize = einsum_type)
    sigma_KWCU_aaaa -= einsum('ijCa,iKja,UW->KWCU', X, v_xxve, rdm_ca, optimize = einsum_type)
    sigma_KWCU_aaaa += 1/2 * einsum('ijCa,jKia,UW->KWCU', X, v_vxxe, rdm_ca, optimize = einsum_type)
    sigma_KWCU_aaaa += 1/2 * einsum('ijaC,iKja,UW->KWCU', X, v_xxve, rdm_ca, optimize = einsum_type)
    sigma_KWCU_aaaa -= einsum('ijaC,jKia,UW->KWCU', X, v_vxxe, rdm_ca, optimize = einsum_type)
    sigma_KWCU_aaaa += einsum('KiCa,a,iUax,Wx->KWCU', X, e_extern, t1_vaea, rdm_ca, optimize = einsum_type)
    sigma_KWCU_aaaa -= 1/2 * einsum('KiCa,a,iUxa,Wx->KWCU', X, e_extern, t1_vaae, rdm_ca, optimize = einsum_type)
    sigma_KWCU_aaaa += 1/2 * einsum('KiCa,a,ixWa,Ux->KWCU', X, e_extern, t1_vaae, rdm_ca, optimize = einsum_type)
    sigma_KWCU_aaaa -= einsum('KiCa,a,ixaW,Ux->KWCU', X, e_extern, t1_vaea, rdm_ca, optimize = einsum_type)
    sigma_KWCU_aaaa -= einsum('KiCa,i,iUax,Wx->KWCU', X, e_val, t1_vaea, rdm_ca, optimize = einsum_type)
    sigma_KWCU_aaaa += 1/2 * einsum('KiCa,i,iUxa,Wx->KWCU', X, e_val, t1_vaae, rdm_ca, optimize = einsum_type)
    sigma_KWCU_aaaa -= 1/2 * einsum('KiCa,i,ixWa,Ux->KWCU', X, e_val, t1_vaae, rdm_ca, optimize = einsum_type)
    sigma_KWCU_aaaa += einsum('KiCa,i,ixaW,Ux->KWCU', X, e_val, t1_vaea, rdm_ca, optimize = einsum_type)
    sigma_KWCU_aaaa -= 1/2 * einsum('KiaC,a,iUax,Wx->KWCU', X, e_extern, t1_vaea, rdm_ca, optimize = einsum_type)
    sigma_KWCU_aaaa += 1/2 * einsum('KiaC,a,iUxa,Wx->KWCU', X, e_extern, t1_vaae, rdm_ca, optimize = einsum_type)
    sigma_KWCU_aaaa -= 1/2 * einsum('KiaC,a,ixWa,Ux->KWCU', X, e_extern, t1_vaae, rdm_ca, optimize = einsum_type)
    sigma_KWCU_aaaa += 1/2 * einsum('KiaC,a,ixaW,Ux->KWCU', X, e_extern, t1_vaea, rdm_ca, optimize = einsum_type)
    sigma_KWCU_aaaa += 1/2 * einsum('KiaC,i,iUax,Wx->KWCU', X, e_val, t1_vaea, rdm_ca, optimize = einsum_type)
    sigma_KWCU_aaaa -= 1/2 * einsum('KiaC,i,iUxa,Wx->KWCU', X, e_val, t1_vaae, rdm_ca, optimize = einsum_type)
    sigma_KWCU_aaaa += 1/2 * einsum('KiaC,i,ixWa,Ux->KWCU', X, e_val, t1_vaae, rdm_ca, optimize = einsum_type)
    sigma_KWCU_aaaa -= 1/2 * einsum('KiaC,i,ixaW,Ux->KWCU', X, e_val, t1_vaea, rdm_ca, optimize = einsum_type)
    sigma_KWCU_aaaa -= einsum('KiCa,Ux,ixay,Wy->KWCU', X, h_aa, t1_vaea, rdm_ca, optimize = einsum_type)
    sigma_KWCU_aaaa += 1/2 * einsum('KiCa,Ux,ixya,Wy->KWCU', X, h_aa, t1_vaae, rdm_ca, optimize = einsum_type)
    sigma_KWCU_aaaa -= einsum('KiCa,Wx,iyax,Uy->KWCU', X, h_aa, t1_vaea, rdm_ca, optimize = einsum_type)
    sigma_KWCU_aaaa += 1/2 * einsum('KiCa,Wx,iyxa,Uy->KWCU', X, h_aa, t1_vaae, rdm_ca, optimize = einsum_type)
    sigma_KWCU_aaaa += einsum('KiCa,xy,iUax,Wy->KWCU', X, h_aa, t1_vaea, rdm_ca, optimize = einsum_type)
    sigma_KWCU_aaaa -= 1/2 * einsum('KiCa,xy,iUxa,Wy->KWCU', X, h_aa, t1_vaae, rdm_ca, optimize = einsum_type)
    sigma_KWCU_aaaa -= 1/2 * einsum('KiCa,xy,ixWa,Uy->KWCU', X, h_aa, t1_vaae, rdm_ca, optimize = einsum_type)
    sigma_KWCU_aaaa += einsum('KiCa,xy,ixaW,Uy->KWCU', X, h_aa, t1_vaea, rdm_ca, optimize = einsum_type)
    sigma_KWCU_aaaa += 1/2 * einsum('KiaC,Ux,ixay,Wy->KWCU', X, h_aa, t1_vaea, rdm_ca, optimize = einsum_type)
    sigma_KWCU_aaaa -= 1/2 * einsum('KiaC,Ux,ixya,Wy->KWCU', X, h_aa, t1_vaae, rdm_ca, optimize = einsum_type)
    sigma_KWCU_aaaa += 1/2 * einsum('KiaC,Wx,iyax,Uy->KWCU', X, h_aa, t1_vaea, rdm_ca, optimize = einsum_type)
    sigma_KWCU_aaaa -= 1/2 * einsum('KiaC,Wx,iyxa,Uy->KWCU', X, h_aa, t1_vaae, rdm_ca, optimize = einsum_type)
    sigma_KWCU_aaaa -= 1/2 * einsum('KiaC,xy,iUax,Wy->KWCU', X, h_aa, t1_vaea, rdm_ca, optimize = einsum_type)
    sigma_KWCU_aaaa += 1/2 * einsum('KiaC,xy,iUxa,Wy->KWCU', X, h_aa, t1_vaae, rdm_ca, optimize = einsum_type)
    sigma_KWCU_aaaa += 1/2 * einsum('KiaC,xy,ixWa,Uy->KWCU', X, h_aa, t1_vaae, rdm_ca, optimize = einsum_type)
    sigma_KWCU_aaaa -= 1/2 * einsum('KiaC,xy,ixaW,Uy->KWCU', X, h_aa, t1_vaea, rdm_ca, optimize = einsum_type)
    sigma_KWCU_aaaa += einsum('KiCa,iUax,xyzw,Wzyw->KWCU', X, t1_vaea, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_aaaa -= 1/2 * einsum('KiCa,iUxa,xyzw,Wzyw->KWCU', X, t1_vaae, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_aaaa -= 1/2 * einsum('KiCa,ixWa,xyzw,Uzyw->KWCU', X, t1_vaae, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_aaaa += einsum('KiCa,ixaW,xyzw,Uzyw->KWCU', X, t1_vaea, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_aaaa -= einsum('KiCa,ixay,Uxzw,Wzyw->KWCU', X, t1_vaea, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_aaaa -= einsum('KiCa,ixay,Uzwx,Wwzy->KWCU', X, t1_vaea, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_aaaa += einsum('KiCa,ixay,Uzyw,Wxzw->KWCU', X, t1_vaea, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_aaaa -= einsum('KiCa,ixay,Wyzw,Uzxw->KWCU', X, t1_vaea, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_aaaa -= einsum('KiCa,ixay,Wzwy,Uwzx->KWCU', X, t1_vaea, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_aaaa += einsum('KiCa,ixay,Wzxw,Uyzw->KWCU', X, t1_vaea, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_aaaa += 1/2 * einsum('KiCa,ixya,Uxzw,Wzyw->KWCU', X, t1_vaae, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_aaaa += 1/2 * einsum('KiCa,ixya,Uzwx,Wwzy->KWCU', X, t1_vaae, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_aaaa -= 1/2 * einsum('KiCa,ixya,Uzyw,Wxzw->KWCU', X, t1_vaae, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_aaaa += 1/2 * einsum('KiCa,ixya,Wyzw,Uzxw->KWCU', X, t1_vaae, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_aaaa += 1/2 * einsum('KiCa,ixya,Wzwy,Uwzx->KWCU', X, t1_vaae, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_aaaa -= 1/2 * einsum('KiCa,ixya,Wzxw,Uyzw->KWCU', X, t1_vaae, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_aaaa -= 1/2 * einsum('KiaC,iUax,xyzw,Wzyw->KWCU', X, t1_vaea, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_aaaa += 1/2 * einsum('KiaC,iUxa,xyzw,Wzyw->KWCU', X, t1_vaae, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_aaaa += 1/2 * einsum('KiaC,ixWa,xyzw,Uzyw->KWCU', X, t1_vaae, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_aaaa -= 1/2 * einsum('KiaC,ixaW,xyzw,Uzyw->KWCU', X, t1_vaea, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_aaaa += 1/2 * einsum('KiaC,ixay,Uxzw,Wzyw->KWCU', X, t1_vaea, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_aaaa += 1/2 * einsum('KiaC,ixay,Uzwx,Wwzy->KWCU', X, t1_vaea, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_aaaa -= 1/2 * einsum('KiaC,ixay,Uzyw,Wxzw->KWCU', X, t1_vaea, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_aaaa += 1/2 * einsum('KiaC,ixay,Wyzw,Uzxw->KWCU', X, t1_vaea, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_aaaa += 1/2 * einsum('KiaC,ixay,Wzwy,Uwzx->KWCU', X, t1_vaea, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_aaaa -= 1/2 * einsum('KiaC,ixay,Wzxw,Uyzw->KWCU', X, t1_vaea, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_aaaa -= 1/2 * einsum('KiaC,ixya,Uxzw,Wzyw->KWCU', X, t1_vaae, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_aaaa += 1/6 * einsum('KiaC,ixya,Uzwx,Wwyz->KWCU', X, t1_vaae, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_aaaa -= 1/6 * einsum('KiaC,ixya,Uzwx,Wwzy->KWCU', X, t1_vaae, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_aaaa -= 1/6 * einsum('KiaC,ixya,Uzyw,Wxwz->KWCU', X, t1_vaae, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_aaaa += 1/6 * einsum('KiaC,ixya,Uzyw,Wxzw->KWCU', X, t1_vaae, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_aaaa -= 1/2 * einsum('KiaC,ixya,Wyzw,Uzxw->KWCU', X, t1_vaae, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_aaaa += 1/6 * einsum('KiaC,ixya,Wzwy,Uwxz->KWCU', X, t1_vaae, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_aaaa -= 1/6 * einsum('KiaC,ixya,Wzwy,Uwzx->KWCU', X, t1_vaae, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_aaaa -= 1/6 * einsum('KiaC,ixya,Wzxw,Uywz->KWCU', X, t1_vaae, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_aaaa += 1/6 * einsum('KiaC,ixya,Wzxw,Uyzw->KWCU', X, t1_vaae, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma[caea__aaaa] += ascontiguousarray(sigma_KWCU_aaaa).reshape(-1)

    sigma_KWCU_abab  = 1/2 * einsum('KiCa,iWxa,Ux->KWCU', X, v_vaae, rdm_ca, optimize = einsum_type)
    sigma_KWCU_abab += einsum('KiCa,iaUx,Wx->KWCU', X, v_veaa, rdm_ca, optimize = einsum_type)
    sigma_KWCU_abab -= einsum('KiCa,iaxW,Ux->KWCU', X, v_veaa, rdm_ca, optimize = einsum_type)
    sigma_KWCU_abab -= 1/2 * einsum('KiCa,ixUa,Wx->KWCU', X, v_vaae, rdm_ca, optimize = einsum_type)
    sigma_KWCU_abab -= 1/2 * einsum('KiaC,iaUx,Wx->KWCU', X, v_veaa, rdm_ca, optimize = einsum_type)
    sigma_KWCU_abab += 1/2 * einsum('KiaC,iaxW,Ux->KWCU', X, v_veaa, rdm_ca, optimize = einsum_type)
    #sigma_KWCU_abab -= 1/2 * einsum('Kiab,iaCb,UW->KWCU', X, v_veee, rdm_ca, optimize = einsum_type)
    #sigma_KWCU_abab += einsum('Kiab,ibCa,UW->KWCU', X, v_veee, rdm_ca, optimize = einsum_type)
    sigma_KWCU_abab -= einsum('ijCa,iKja,UW->KWCU', X, v_xxve, rdm_ca, optimize = einsum_type)
    sigma_KWCU_abab += 1/2 * einsum('ijCa,jKia,UW->KWCU', X, v_vxxe, rdm_ca, optimize = einsum_type)
    sigma_KWCU_abab += 1/2 * einsum('ijaC,iKja,UW->KWCU', X, v_xxve, rdm_ca, optimize = einsum_type)
    sigma_KWCU_abab -= einsum('ijaC,jKia,UW->KWCU', X, v_vxxe, rdm_ca, optimize = einsum_type)
    sigma_KWCU_abab += einsum('KiCa,a,iUax,Wx->KWCU', X, e_extern, t1_vaea, rdm_ca, optimize = einsum_type)
    sigma_KWCU_abab -= 1/2 * einsum('KiCa,a,iUxa,Wx->KWCU', X, e_extern, t1_vaae, rdm_ca, optimize = einsum_type)
    sigma_KWCU_abab += 1/2 * einsum('KiCa,a,ixWa,Ux->KWCU', X, e_extern, t1_vaae, rdm_ca, optimize = einsum_type)
    sigma_KWCU_abab -= einsum('KiCa,a,ixaW,Ux->KWCU', X, e_extern, t1_vaea, rdm_ca, optimize = einsum_type)
    sigma_KWCU_abab -= einsum('KiCa,i,iUax,Wx->KWCU', X, e_val, t1_vaea, rdm_ca, optimize = einsum_type)
    sigma_KWCU_abab += 1/2 * einsum('KiCa,i,iUxa,Wx->KWCU', X, e_val, t1_vaae, rdm_ca, optimize = einsum_type)
    sigma_KWCU_abab -= 1/2 * einsum('KiCa,i,ixWa,Ux->KWCU', X, e_val, t1_vaae, rdm_ca, optimize = einsum_type)
    sigma_KWCU_abab += einsum('KiCa,i,ixaW,Ux->KWCU', X, e_val, t1_vaea, rdm_ca, optimize = einsum_type)
    sigma_KWCU_abab -= 1/2 * einsum('KiaC,a,iUax,Wx->KWCU', X, e_extern, t1_vaea, rdm_ca, optimize = einsum_type)
    sigma_KWCU_abab += 1/2 * einsum('KiaC,a,ixaW,Ux->KWCU', X, e_extern, t1_vaea, rdm_ca, optimize = einsum_type)
    sigma_KWCU_abab += 1/2 * einsum('KiaC,i,iUax,Wx->KWCU', X, e_val, t1_vaea, rdm_ca, optimize = einsum_type)
    sigma_KWCU_abab -= 1/2 * einsum('KiaC,i,ixaW,Ux->KWCU', X, e_val, t1_vaea, rdm_ca, optimize = einsum_type)
    sigma_KWCU_abab -= einsum('KiCa,Ux,ixay,Wy->KWCU', X, h_aa, t1_vaea, rdm_ca, optimize = einsum_type)
    sigma_KWCU_abab += 1/2 * einsum('KiCa,Ux,ixya,Wy->KWCU', X, h_aa, t1_vaae, rdm_ca, optimize = einsum_type)
    sigma_KWCU_abab -= einsum('KiCa,Wx,iyax,Uy->KWCU', X, h_aa, t1_vaea, rdm_ca, optimize = einsum_type)
    sigma_KWCU_abab += 1/2 * einsum('KiCa,Wx,iyxa,Uy->KWCU', X, h_aa, t1_vaae, rdm_ca, optimize = einsum_type)
    sigma_KWCU_abab += einsum('KiCa,xy,iUax,Wy->KWCU', X, h_aa, t1_vaea, rdm_ca, optimize = einsum_type)
    sigma_KWCU_abab -= 1/2 * einsum('KiCa,xy,iUxa,Wy->KWCU', X, h_aa, t1_vaae, rdm_ca, optimize = einsum_type)
    sigma_KWCU_abab -= 1/2 * einsum('KiCa,xy,ixWa,Uy->KWCU', X, h_aa, t1_vaae, rdm_ca, optimize = einsum_type)
    sigma_KWCU_abab += einsum('KiCa,xy,ixaW,Uy->KWCU', X, h_aa, t1_vaea, rdm_ca, optimize = einsum_type)
    sigma_KWCU_abab += 1/2 * einsum('KiaC,Ux,ixay,Wy->KWCU', X, h_aa, t1_vaea, rdm_ca, optimize = einsum_type)
    sigma_KWCU_abab += 1/2 * einsum('KiaC,Wx,iyax,Uy->KWCU', X, h_aa, t1_vaea, rdm_ca, optimize = einsum_type)
    sigma_KWCU_abab -= 1/2 * einsum('KiaC,xy,iUax,Wy->KWCU', X, h_aa, t1_vaea, rdm_ca, optimize = einsum_type)
    sigma_KWCU_abab -= 1/2 * einsum('KiaC,xy,ixaW,Uy->KWCU', X, h_aa, t1_vaea, rdm_ca, optimize = einsum_type)
    sigma_KWCU_abab += einsum('KiCa,iUax,xyzw,Wzyw->KWCU', X, t1_vaea, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_abab -= 1/2 * einsum('KiCa,iUxa,xyzw,Wzyw->KWCU', X, t1_vaae, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_abab -= 1/2 * einsum('KiCa,ixWa,xyzw,Uzyw->KWCU', X, t1_vaae, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_abab += einsum('KiCa,ixaW,xyzw,Uzyw->KWCU', X, t1_vaea, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_abab -= einsum('KiCa,ixay,Uxzw,Wzyw->KWCU', X, t1_vaea, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_abab -= einsum('KiCa,ixay,Uzwx,Wwzy->KWCU', X, t1_vaea, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_abab += einsum('KiCa,ixay,Uzyw,Wxzw->KWCU', X, t1_vaea, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_abab -= einsum('KiCa,ixay,Wyzw,Uzxw->KWCU', X, t1_vaea, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_abab -= einsum('KiCa,ixay,Wzwy,Uwzx->KWCU', X, t1_vaea, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_abab += einsum('KiCa,ixay,Wzxw,Uyzw->KWCU', X, t1_vaea, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_abab += 1/2 * einsum('KiCa,ixya,Uxzw,Wzyw->KWCU', X, t1_vaae, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_abab += 1/2 * einsum('KiCa,ixya,Uzwx,Wwzy->KWCU', X, t1_vaae, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_abab -= 1/2 * einsum('KiCa,ixya,Uzyw,Wxzw->KWCU', X, t1_vaae, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_abab += 1/2 * einsum('KiCa,ixya,Wyzw,Uzxw->KWCU', X, t1_vaae, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_abab += 1/2 * einsum('KiCa,ixya,Wzwy,Uwzx->KWCU', X, t1_vaae, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_abab -= 1/2 * einsum('KiCa,ixya,Wzxw,Uyzw->KWCU', X, t1_vaae, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_abab -= 1/2 * einsum('KiaC,iUax,xyzw,Wzyw->KWCU', X, t1_vaea, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_abab -= 1/2 * einsum('KiaC,ixaW,xyzw,Uzyw->KWCU', X, t1_vaea, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_abab += 1/2 * einsum('KiaC,ixay,Uxzw,Wzyw->KWCU', X, t1_vaea, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_abab += 1/2 * einsum('KiaC,ixay,Uzwx,Wwzy->KWCU', X, t1_vaea, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_abab -= 1/2 * einsum('KiaC,ixay,Uzyw,Wxzw->KWCU', X, t1_vaea, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_abab += 1/2 * einsum('KiaC,ixay,Wyzw,Uzxw->KWCU', X, t1_vaea, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_abab += 1/2 * einsum('KiaC,ixay,Wzwy,Uwzx->KWCU', X, t1_vaea, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_abab -= 1/2 * einsum('KiaC,ixay,Wzxw,Uyzw->KWCU', X, t1_vaea, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_abab -= 1/6 * einsum('KiaC,ixya,Uzwx,Wwyz->KWCU', X, t1_vaae, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_abab -= 1/3 * einsum('KiaC,ixya,Uzwx,Wwzy->KWCU', X, t1_vaae, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_abab += 1/6 * einsum('KiaC,ixya,Uzyw,Wxwz->KWCU', X, t1_vaae, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_abab += 1/3 * einsum('KiaC,ixya,Uzyw,Wxzw->KWCU', X, t1_vaae, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_abab -= 1/6 * einsum('KiaC,ixya,Wzwy,Uwxz->KWCU', X, t1_vaae, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_abab -= 1/3 * einsum('KiaC,ixya,Wzwy,Uwzx->KWCU', X, t1_vaae, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_abab += 1/6 * einsum('KiaC,ixya,Wzxw,Uywz->KWCU', X, t1_vaae, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_abab += 1/3 * einsum('KiaC,ixya,Wzxw,Uyzw->KWCU', X, t1_vaae, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma[caea__abab] += ascontiguousarray(sigma_KWCU_abab).reshape(-1)

    sigma_KWCU_baab =- 1/2 * einsum('KiaC,iWxa,Ux->KWCU', X, v_vaae, rdm_ca, optimize = einsum_type)
    sigma_KWCU_baab += 1/2 * einsum('KiaC,ixUa,Wx->KWCU', X, v_vaae, rdm_ca, optimize = einsum_type)
    sigma_KWCU_baab += 1/2 * einsum('KiaC,a,iUxa,Wx->KWCU', X, e_extern, t1_vaae, rdm_ca, optimize = einsum_type)
    sigma_KWCU_baab -= 1/2 * einsum('KiaC,a,ixWa,Ux->KWCU', X, e_extern, t1_vaae, rdm_ca, optimize = einsum_type)
    sigma_KWCU_baab -= 1/2 * einsum('KiaC,i,iUxa,Wx->KWCU', X, e_val, t1_vaae, rdm_ca, optimize = einsum_type)
    sigma_KWCU_baab += 1/2 * einsum('KiaC,i,ixWa,Ux->KWCU', X, e_val, t1_vaae, rdm_ca, optimize = einsum_type)
    sigma_KWCU_baab -= 1/2 * einsum('KiaC,Ux,ixya,Wy->KWCU', X, h_aa, t1_vaae, rdm_ca, optimize = einsum_type)
    sigma_KWCU_baab -= 1/2 * einsum('KiaC,Wx,iyxa,Uy->KWCU', X, h_aa, t1_vaae, rdm_ca, optimize = einsum_type)
    sigma_KWCU_baab += 1/2 * einsum('KiaC,xy,iUxa,Wy->KWCU', X, h_aa, t1_vaae, rdm_ca, optimize = einsum_type)
    sigma_KWCU_baab += 1/2 * einsum('KiaC,xy,ixWa,Uy->KWCU', X, h_aa, t1_vaae, rdm_ca, optimize = einsum_type)
    sigma_KWCU_baab += 1/2 * einsum('KiaC,iUxa,xyzw,Wzyw->KWCU', X, t1_vaae, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_baab += 1/2 * einsum('KiaC,ixWa,xyzw,Uzyw->KWCU', X, t1_vaae, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_baab -= 1/2 * einsum('KiaC,ixya,Uxzw,Wzyw->KWCU', X, t1_vaae, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_baab += 1/3 * einsum('KiaC,ixya,Uzwx,Wwyz->KWCU', X, t1_vaae, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_baab += 1/6 * einsum('KiaC,ixya,Uzwx,Wwzy->KWCU', X, t1_vaae, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_baab -= 1/3 * einsum('KiaC,ixya,Uzyw,Wxwz->KWCU', X, t1_vaae, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_baab -= 1/6 * einsum('KiaC,ixya,Uzyw,Wxzw->KWCU', X, t1_vaae, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_baab -= 1/2 * einsum('KiaC,ixya,Wyzw,Uzxw->KWCU', X, t1_vaae, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_baab += 1/3 * einsum('KiaC,ixya,Wzwy,Uwxz->KWCU', X, t1_vaae, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_baab += 1/6 * einsum('KiaC,ixya,Wzwy,Uwzx->KWCU', X, t1_vaae, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_baab -= 1/3 * einsum('KiaC,ixya,Wzxw,Uywz->KWCU', X, t1_vaae, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_baab -= 1/6 * einsum('KiaC,ixya,Wzxw,Uyzw->KWCU', X, t1_vaae, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma[caea__baab] += ascontiguousarray(sigma_KWCU_baab).reshape(-1)

    mr_adc.log.timer_debug("computing sigma H1 h1-h1 CAEA-CVEE", *cput1)

def compute_sigma_vector__H1__h1_h1__CAEA_CVEE__V_VEEE(mr_adc, X, sigma, v_veee):
    cput1 = (logger.process_clock(), logger.perf_counter())

    # Einsum definition from kernel
    einsum = mr_adc.interface.einsum
    einsum_type = mr_adc.interface.einsum_type
  
    # Reduced Density Matrices
    rdm_ca = mr_adc.rdm.ca

    temp =- 1/2 * einsum('Kiab,iaCb,UW->KWCU', X, v_veee, rdm_ca, optimize = einsum_type)
    temp += einsum('Kiab,ibCa,UW->KWCU', X, v_veee, rdm_ca, optimize = einsum_type)
    # temp = sigma_KWCU_aaaa = sigma_KWCU_abab

    mr_adc.log.timer_debug("contracting v2e.veee", *cput1)

    return temp

# CVAA <- CVEE
def compute_sigma_vector__H1__h1_h1__CVAA_CVEE(mr_adc, X, sigma):
    cput1 = (logger.process_clock(), logger.perf_counter())

    # Einsum definition from kernel
    einsum = mr_adc.interface.einsum
    einsum_type = mr_adc.interface.einsum_type

    #Excitation Manifold
    cvaa = mr_adc.h1.cvaa

    ## Molecular Orbitals Energies
    e_extern = mr_adc.mo_energy.e
    
    ## One-electron integrals
    h_aa = mr_adc.h1eff.aa

    ## Two-electron integrals
    v_aeae = mr_adc.v2e.aeae
    v_aaaa = mr_adc.v2e.aaaa

    ## Amplitudes
    t1_aaee = mr_adc.t1.aaee

    # Reduced Density Matrices
    rdm_ca = mr_adc.rdm.ca
    rdm_ccaa = mr_adc.rdm.ccaa

    sigma_KLWU  = einsum('KLab,UbWa->KLWU', X, v_aeae, optimize = einsum_type)
    sigma_KLWU += einsum('KLab,a,UWba->KLWU', X, e_extern, t1_aaee, optimize = einsum_type)
    sigma_KLWU += einsum('KLab,b,UWba->KLWU', X, e_extern, t1_aaee, optimize = einsum_type)
    sigma_KLWU -= einsum('KLab,Ux,Wxab->KLWU', X, h_aa, t1_aaee, optimize = einsum_type)
    sigma_KLWU -= einsum('KLab,Wx,Uxba->KLWU', X, h_aa, t1_aaee, optimize = einsum_type)
    sigma_KLWU -= einsum('KLab,xyab,UyWx->KLWU', X, t1_aaee, v_aaaa, optimize = einsum_type)
    sigma_KLWU -= 1/2 * einsum('KLab,Ubxa,Wx->KLWU', X, v_aeae, rdm_ca, optimize = einsum_type)
    sigma_KLWU -= 1/2 * einsum('KLab,Waxb,Ux->KLWU', X, v_aeae, rdm_ca, optimize = einsum_type)
    sigma_KLWU -= 1/2 * einsum('KLab,a,Uxba,Wx->KLWU', X, e_extern, t1_aaee, rdm_ca, optimize = einsum_type)
    sigma_KLWU -= 1/2 * einsum('KLab,a,Wxab,Ux->KLWU', X, e_extern, t1_aaee, rdm_ca, optimize = einsum_type)
    sigma_KLWU -= 1/2 * einsum('KLab,b,Uxba,Wx->KLWU', X, e_extern, t1_aaee, rdm_ca, optimize = einsum_type)
    sigma_KLWU -= 1/2 * einsum('KLab,b,Wxab,Ux->KLWU', X, e_extern, t1_aaee, rdm_ca, optimize = einsum_type)
    sigma_KLWU += 1/2 * einsum('KLab,Ux,yxab,Wy->KLWU', X, h_aa, t1_aaee, rdm_ca, optimize = einsum_type)
    sigma_KLWU += 1/2 * einsum('KLab,Wx,xyab,Uy->KLWU', X, h_aa, t1_aaee, rdm_ca, optimize = einsum_type)
    sigma_KLWU += 1/2 * einsum('KLab,xy,Uxba,Wy->KLWU', X, h_aa, t1_aaee, rdm_ca, optimize = einsum_type)
    sigma_KLWU += 1/2 * einsum('KLab,xy,Wxab,Uy->KLWU', X, h_aa, t1_aaee, rdm_ca, optimize = einsum_type)
    sigma_KLWU -= einsum('KLab,Uxba,Wxyz,yz->KLWU', X, t1_aaee, v_aaaa, rdm_ca, optimize = einsum_type)
    sigma_KLWU += 1/2 * einsum('KLab,Uxba,Wyzx,zy->KLWU', X, t1_aaee, v_aaaa, rdm_ca, optimize = einsum_type)
    sigma_KLWU += 1/2 * einsum('KLab,Uxba,xyzw,Wzyw->KLWU', X, t1_aaee, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KLWU -= einsum('KLab,Wxab,Uxyz,yz->KLWU', X, t1_aaee, v_aaaa, rdm_ca, optimize = einsum_type)
    sigma_KLWU += 1/2 * einsum('KLab,Wxab,Uyzx,zy->KLWU', X, t1_aaee, v_aaaa, rdm_ca, optimize = einsum_type)
    sigma_KLWU += 1/2 * einsum('KLab,Wxab,xyzw,Uzyw->KLWU', X, t1_aaee, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KLWU += 1/2 * einsum('KLab,xyab,UyWz,xz->KLWU', X, t1_aaee, v_aaaa, rdm_ca, optimize = einsum_type)
    sigma_KLWU += 1/2 * einsum('KLab,xyab,Uyzw,Wwxz->KLWU', X, t1_aaee, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KLWU += 1/2 * einsum('KLab,xyab,Uyzx,Wz->KLWU', X, t1_aaee, v_aaaa, rdm_ca, optimize = einsum_type)
    sigma_KLWU += 1/2 * einsum('KLab,xyab,UzWx,yz->KLWU', X, t1_aaee, v_aaaa, rdm_ca, optimize = einsum_type)
    sigma_KLWU -= 1/3 * einsum('KLab,xyab,Uzwx,Wzwy->KLWU', X, t1_aaee, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KLWU -= 1/6 * einsum('KLab,xyab,Uzwx,Wzyw->KLWU', X, t1_aaee, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KLWU -= 1/6 * einsum('KLab,xyab,Uzwy,Wzwx->KLWU', X, t1_aaee, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KLWU -= 1/3 * einsum('KLab,xyab,Uzwy,Wzxw->KLWU', X, t1_aaee, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KLWU += 1/2 * einsum('KLab,xyab,Wxzw,Uwyz->KLWU', X, t1_aaee, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KLWU += 1/2 * einsum('KLab,xyab,Wxzy,Uz->KLWU', X, t1_aaee, v_aaaa, rdm_ca, optimize = einsum_type)
    sigma_KLWU -= 1/6 * einsum('KLab,xyab,Wzwx,Uzwy->KLWU', X, t1_aaee, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KLWU -= 1/3 * einsum('KLab,xyab,Wzwx,Uzyw->KLWU', X, t1_aaee, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KLWU -= 1/3 * einsum('KLab,xyab,Wzwy,Uzwx->KLWU', X, t1_aaee, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KLWU -= 1/6 * einsum('KLab,xyab,Wzwy,Uzxw->KLWU', X, t1_aaee, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma[cvaa] += ascontiguousarray(sigma_KLWU).reshape(-1) 

    mr_adc.log.timer_debug("computing sigma H1 h1-h1 CVAA-CVEE", *cput1)

# CVEA <- CVEE
def compute_sigma_vector__H1__h1_h1__CVEA_CVEE(mr_adc, X, sigma):
    cput1 = (logger.process_clock(), logger.perf_counter())

    # Einsum definition from kernel
    einsum = mr_adc.interface.einsum
    einsum_type = mr_adc.interface.einsum_type

    #Excitation Manifold
    cvea__abab = mr_adc.h1.cvea__abab
    cvea__baab = mr_adc.h1.cvea__baab
 
    ## Molecular Orbitals Energies
    e_extern = mr_adc.mo_energy.e
    
    ## One-electron integrals
    h_aa = mr_adc.h1eff.aa
    h_ae = mr_adc.h1eff.ae

    ## Two-electron integrals
    v_xxae = mr_adc.v2e.xxae
    v_xaex = mr_adc.v2e.xaex
    v_vvae = mr_adc.v2e.vvae
    v_vaev = mr_adc.v2e.vaev
    v_aaaa = mr_adc.v2e.aaaa
    v_aaae = mr_adc.v2e.aaae
    v_aeee = mr_adc.v2e.aeee

    ## Amplitudes
    t1_ae = mr_adc.t1.ae
    t1_aaae = mr_adc.t1.aaae
    
    # Reduced Density Matrices
    rdm_ca = mr_adc.rdm.ca
    rdm_ccaa = mr_adc.rdm.ccaa

    sigma_KLCW_abab  = einsum('KLCa,Wa->KLCW', X, h_ae, optimize = einsum_type)
    sigma_KLCW_abab += einsum('KLab,WbCa->KLCW', X, v_aeee, optimize = einsum_type)
    sigma_KLCW_abab += 2 * einsum('KiCa,LWai->KLCW', X, v_vaev, optimize = einsum_type)
    sigma_KLCW_abab -= einsum('KiCa,iLWa->KLCW', X, v_vvae, optimize = einsum_type)
    sigma_KLCW_abab -= einsum('KiaC,LWai->KLCW', X, v_vaev, optimize = einsum_type)
    sigma_KLCW_abab -= einsum('iLCa,iKWa->KLCW', X, v_xxae, optimize = einsum_type)
    sigma_KLCW_abab += einsum('KLCa,a,Wa->KLCW', X, e_extern, t1_ae, optimize = einsum_type)
    sigma_KLCW_abab -= einsum('KLCa,Wx,xa->KLCW', X, h_aa, t1_ae, optimize = einsum_type)
    sigma_KLCW_abab -= 1/2 * einsum('KLCa,Wxya,yx->KLCW', X, v_aaae, rdm_ca, optimize = einsum_type)
    sigma_KLCW_abab += einsum('KLCa,xyWa,xy->KLCW', X, v_aaae, rdm_ca, optimize = einsum_type)
    sigma_KLCW_abab -= 1/2 * einsum('KLab,xbCa,Wx->KLCW', X, v_aeee, rdm_ca, optimize = einsum_type)
    sigma_KLCW_abab -= einsum('KiCa,Lxai,Wx->KLCW', X, v_vaev, rdm_ca, optimize = einsum_type)
    sigma_KLCW_abab += 1/2 * einsum('KiCa,iLxa,Wx->KLCW', X, v_vvae, rdm_ca, optimize = einsum_type)
    sigma_KLCW_abab += 1/2 * einsum('KiaC,Lxai,Wx->KLCW', X, v_vaev, rdm_ca, optimize = einsum_type)
    sigma_KLCW_abab += 1/2 * einsum('iLCa,iKxa,Wx->KLCW', X, v_xxae, rdm_ca, optimize = einsum_type)
    sigma_KLCW_abab -= 1/2 * einsum('KLCa,a,Wxya,xy->KLCW', X, e_extern, t1_aaae, rdm_ca, optimize = einsum_type)
    sigma_KLCW_abab += einsum('KLCa,a,xWya,xy->KLCW', X, e_extern, t1_aaae, rdm_ca, optimize = einsum_type)
    sigma_KLCW_abab += 1/2 * einsum('KLCa,Wx,xyza,zy->KLCW', X, h_aa, t1_aaae, rdm_ca, optimize = einsum_type)
    sigma_KLCW_abab -= einsum('KLCa,Wx,yxza,zy->KLCW', X, h_aa, t1_aaae, rdm_ca, optimize = einsum_type)
    sigma_KLCW_abab += 1/2 * einsum('KLCa,xy,Wxza,yz->KLCW', X, h_aa, t1_aaae, rdm_ca, optimize = einsum_type)
    sigma_KLCW_abab -= 1/2 * einsum('KLCa,xy,Wzxa,yz->KLCW', X, h_aa, t1_aaae, rdm_ca, optimize = einsum_type)
    sigma_KLCW_abab -= einsum('KLCa,xy,xWza,yz->KLCW', X, h_aa, t1_aaae, rdm_ca, optimize = einsum_type)
    sigma_KLCW_abab += einsum('KLCa,xy,zWxa,yz->KLCW', X, h_aa, t1_aaae, rdm_ca, optimize = einsum_type)
    sigma_KLCW_abab += 1/2 * einsum('KLCa,Wxya,xzwu,ywzu->KLCW', X, t1_aaae, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KLCW_abab -= 1/2 * einsum('KLCa,Wxya,yzwu,xwzu->KLCW', X, t1_aaae, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KLCW_abab -= einsum('KLCa,xWya,xzwu,ywzu->KLCW', X, t1_aaae, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KLCW_abab += einsum('KLCa,xWya,yzwu,xwzu->KLCW', X, t1_aaae, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KLCW_abab -= einsum('KLCa,xa,Wxyz,yz->KLCW', X, t1_ae, v_aaaa, rdm_ca, optimize = einsum_type)
    sigma_KLCW_abab += 1/2 * einsum('KLCa,xa,Wyzx,zy->KLCW', X, t1_ae, v_aaaa, rdm_ca, optimize = einsum_type)
    sigma_KLCW_abab += 1/2 * einsum('KLCa,xyza,Wwux,zwuy->KLCW', X, t1_aaae, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KLCW_abab += 1/2 * einsum('KLCa,xyza,Wwuy,zwxu->KLCW', X, t1_aaae, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KLCW_abab -= 1/2 * einsum('KLCa,xyza,Wwzu,yxwu->KLCW', X, t1_aaae, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KLCW_abab += 1/2 * einsum('KLCa,xyza,Wxwu,zuyw->KLCW', X, t1_aaae, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KLCW_abab += 1/2 * einsum('KLCa,xyza,Wxwy,zw->KLCW', X, t1_aaae, v_aaaa, rdm_ca, optimize = einsum_type)
    sigma_KLCW_abab -= einsum('KLCa,xyza,Wywu,zuxw->KLCW', X, t1_aaae, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KLCW_abab -= einsum('KLCa,xyza,Wywx,zw->KLCW', X, t1_aaae, v_aaaa, rdm_ca, optimize = einsum_type)
    sigma[cvea__abab] += ascontiguousarray(sigma_KLCW_abab).reshape(-1)

    sigma_KLCW_baab =- einsum('KLaC,Wa->KLCW', X, h_ae, optimize = einsum_type)
    sigma_KLCW_baab -= einsum('KLab,WaCb->KLCW', X, v_aeee, optimize = einsum_type)
    sigma_KLCW_baab += einsum('KiaC,iLWa->KLCW', X, v_vvae, optimize = einsum_type)
    sigma_KLCW_baab += einsum('iLCa,KWai->KLCW', X, v_xaex, optimize = einsum_type)
    sigma_KLCW_baab -= 2 * einsum('iLaC,KWai->KLCW', X, v_xaex, optimize = einsum_type)
    sigma_KLCW_baab += einsum('iLaC,iKWa->KLCW', X, v_xxae, optimize = einsum_type)
    sigma_KLCW_baab -= einsum('KLaC,a,Wa->KLCW', X, e_extern, t1_ae, optimize = einsum_type)
    sigma_KLCW_baab += einsum('KLaC,Wx,xa->KLCW', X, h_aa, t1_ae, optimize = einsum_type)
    sigma_KLCW_baab += 1/2 * einsum('KLaC,Wxya,yx->KLCW', X, v_aaae, rdm_ca, optimize = einsum_type)
    sigma_KLCW_baab -= einsum('KLaC,xyWa,xy->KLCW', X, v_aaae, rdm_ca, optimize = einsum_type)
    sigma_KLCW_baab += 1/2 * einsum('KLab,xaCb,Wx->KLCW', X, v_aeee, rdm_ca, optimize = einsum_type)
    sigma_KLCW_baab -= 1/2 * einsum('KiaC,iLxa,Wx->KLCW', X, v_vvae, rdm_ca, optimize = einsum_type)
    sigma_KLCW_baab -= 1/2 * einsum('iLCa,Kxai,Wx->KLCW', X, v_xaex, rdm_ca, optimize = einsum_type)
    sigma_KLCW_baab += einsum('iLaC,Kxai,Wx->KLCW', X, v_xaex, rdm_ca, optimize = einsum_type)
    sigma_KLCW_baab -= 1/2 * einsum('iLaC,iKxa,Wx->KLCW', X, v_xxae, rdm_ca, optimize = einsum_type)
    sigma_KLCW_baab += 1/2 * einsum('KLaC,a,Wxya,xy->KLCW', X, e_extern, t1_aaae, rdm_ca, optimize = einsum_type)
    sigma_KLCW_baab -= einsum('KLaC,a,xWya,xy->KLCW', X, e_extern, t1_aaae, rdm_ca, optimize = einsum_type)
    sigma_KLCW_baab -= 1/2 * einsum('KLaC,Wx,xyza,zy->KLCW', X, h_aa, t1_aaae, rdm_ca, optimize = einsum_type)
    sigma_KLCW_baab += einsum('KLaC,Wx,yxza,zy->KLCW', X, h_aa, t1_aaae, rdm_ca, optimize = einsum_type)
    sigma_KLCW_baab -= 1/2 * einsum('KLaC,xy,Wxza,yz->KLCW', X, h_aa, t1_aaae, rdm_ca, optimize = einsum_type)
    sigma_KLCW_baab += 1/2 * einsum('KLaC,xy,Wzxa,yz->KLCW', X, h_aa, t1_aaae, rdm_ca, optimize = einsum_type)
    sigma_KLCW_baab += einsum('KLaC,xy,xWza,yz->KLCW', X, h_aa, t1_aaae, rdm_ca, optimize = einsum_type)
    sigma_KLCW_baab -= einsum('KLaC,xy,zWxa,yz->KLCW', X, h_aa, t1_aaae, rdm_ca, optimize = einsum_type)
    sigma_KLCW_baab -= 1/2 * einsum('KLaC,Wxya,xzwu,ywzu->KLCW', X, t1_aaae, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KLCW_baab += 1/2 * einsum('KLaC,Wxya,yzwu,xwzu->KLCW', X, t1_aaae, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KLCW_baab += einsum('KLaC,xWya,xzwu,ywzu->KLCW', X, t1_aaae, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KLCW_baab -= einsum('KLaC,xWya,yzwu,xwzu->KLCW', X, t1_aaae, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KLCW_baab += einsum('KLaC,xa,Wxyz,yz->KLCW', X, t1_ae, v_aaaa, rdm_ca, optimize = einsum_type)
    sigma_KLCW_baab -= 1/2 * einsum('KLaC,xa,Wyzx,zy->KLCW', X, t1_ae, v_aaaa, rdm_ca, optimize = einsum_type)
    sigma_KLCW_baab -= 1/2 * einsum('KLaC,xyza,Wwux,zwuy->KLCW', X, t1_aaae, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KLCW_baab -= 1/2 * einsum('KLaC,xyza,Wwuy,zwxu->KLCW', X, t1_aaae, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KLCW_baab += 1/2 * einsum('KLaC,xyza,Wwzu,yxwu->KLCW', X, t1_aaae, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KLCW_baab -= 1/2 * einsum('KLaC,xyza,Wxwu,zuyw->KLCW', X, t1_aaae, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KLCW_baab -= 1/2 * einsum('KLaC,xyza,Wxwy,zw->KLCW', X, t1_aaae, v_aaaa, rdm_ca, optimize = einsum_type)
    sigma_KLCW_baab += einsum('KLaC,xyza,Wywu,zuxw->KLCW', X, t1_aaae, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KLCW_baab += einsum('KLaC,xyza,Wywx,zw->KLCW', X, t1_aaae, v_aaaa, rdm_ca, optimize = einsum_type)
    sigma[cvea__baab] += ascontiguousarray(sigma_KLCW_baab).reshape(-1)

    mr_adc.log.timer_debug("computing sigma H1 h1-h1 CVEA-CVEE", *cput1)

def compute_sigma_vector__H1__h1_h1__CVEA_CVEE__V_AEEE(mr_adc, X, sigma, v_aeee):
    cput1 = (logger.process_clock(), logger.perf_counter())

    # Einsum definition from kernel
    einsum = mr_adc.interface.einsum
    einsum_type = mr_adc.interface.einsum_type

    # Reduced Density Matrices
    rdm_ca = mr_adc.rdm.ca

    sigma_KLCW_abab  = einsum('KLab,WbCa->KLCW', X, v_aeee, optimize = einsum_type)
    sigma_KLCW_abab -= 1/2 * einsum('KLab,xbCa,Wx->KLCW', X, v_aeee, rdm_ca, optimize = einsum_type)

    sigma_KLCW_baab =- einsum('KLab,WaCb->KLCW', X, v_aeee, optimize = einsum_type)
    sigma_KLCW_baab += 1/2 * einsum('KLab,xaCb,Wx->KLCW', X, v_aeee, rdm_ca, optimize = einsum_type)

    mr_adc.log.timer_debug("contracting v2e.aeee", *cput1)
    return sigma_KLCW_abab, sigma_KLCW_baab

# CVEE <- CVEE
def compute_sigma_vector__H1__h1_h1__CVEE_CVEE(mr_adc, X, sigma):        
    cput1 = (logger.process_clock(), logger.perf_counter())

    # Einsum definition from kernel
    einsum = mr_adc.interface.einsum
    einsum_type = mr_adc.interface.einsum_type
    dot = mr_adc.interface.dot

    #Excitation Manifold
    cvee = mr_adc.h1.cvee

    ## Two-electron integrals
    v_xxvv = mr_adc.v2e.xxvv
    v_xxee = mr_adc.v2e.xxee
    v_xvvx = mr_adc.v2e.xvvx
    v_xeex = mr_adc.v2e.xeex
    v_vvee = mr_adc.v2e.vvee
    v_veev = mr_adc.v2e.veev
    #v_eeee = mr_adc.v2e.eeee

    # Variables from kernel
    ncvs    = mr_adc.ncvs
    nval    = mr_adc.nval
    nextern = mr_adc.nextern

    #sigma_KLCD  = einsum('KLab,CaDb->KLCD', X, v_eeee, optimize = einsum_type)
    sigma_KLCD  = 2 * einsum('KiCa,LDai->KLCD', X, v_veev, optimize = einsum_type)
    sigma_KLCD -= einsum('KiCa,iLDa->KLCD', X, v_vvee, optimize = einsum_type)
    sigma_KLCD -= einsum('KiaC,LDai->KLCD', X, v_veev, optimize = einsum_type)
    sigma_KLCD -= einsum('KiaD,iLCa->KLCD', X, v_vvee, optimize = einsum_type)
    sigma_KLCD -= einsum('iLCa,iKDa->KLCD', X, v_xxee, optimize = einsum_type)
    sigma_KLCD -= einsum('iLDa,KCai->KLCD', X, v_xeex, optimize = einsum_type)
    sigma_KLCD += 2 * einsum('iLaD,KCai->KLCD', X, v_xeex, optimize = einsum_type)
    sigma_KLCD -= einsum('iLaD,iKCa->KLCD', X, v_xxee, optimize = einsum_type)
    sigma_KLCD += einsum('ijCD,KiLj->KLCD', X, v_xxvv, optimize = einsum_type)
    sigma_KLCD += einsum('ijDC,KjLi->KLCD', X, v_xvvx, optimize = einsum_type)

    # Pack X for ladder contractions
    X = ascontiguousarray(X.reshape(ncvs*nval, -1).T) 
    temp = zeros((nextern, nextern, ncvs*nval))

    chunks = tools.calculate_chunks(mr_adc, nextern, [nextern, nextern, nextern], ntensors=2)
    for i_chunk, (s_chunk, f_chunk) in enumerate(chunks):
        cput2 = (logger.process_clock(), logger.perf_counter())
        mr_adc.log.debug("v2e.eeee [%i/%i], chunk [%i:%i]", i_chunk + 1, len(chunks), s_chunk, f_chunk)
    
        ## Two-electron integral
        if mr_adc.interface.with_df:
            v_eeee = get_eeee_df(mr_adc, mr_adc.v2e.Lee, s_chunk, f_chunk)
        else:
            v_eeee = unpack_v2e_eeee(mr_adc, mr_adc.v2e.eeee, s_chunk, f_chunk)
        
        # Contractions using dot products
        temp[s_chunk:f_chunk] += dot(v_eeee, X).reshape(-1, nextern, ncvs*nval)

        del(v_eeee)
        mr_adc.log.timer_debug("contracting v2e.eeee", *cput2)

    sigma_KLCD += temp.transpose(2,0,1).reshape((ncvs, nval, nextern, nextern))
    del(X, temp)
 
    sigma[cvee] += ascontiguousarray(sigma_KLCD).reshape(-1)

    mr_adc.log.timer_debug("computing sigma H1 h1-h1 CVEE-CVEE", *cput1)

