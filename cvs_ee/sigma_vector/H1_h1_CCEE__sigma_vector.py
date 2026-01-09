## h1 <- h1 coupling contributions
from . import logger, tools, ascontiguousarray, zeros
from prism.mr_adc_integrals import get_eeee_df, unpack_v2e_eeee

# CCAA <- CCEE
def compute_sigma_vector__H1__h1_h1__CCAA_CCEE(mr_adc, X, sigma):
    cput1 = (logger.process_clock(), logger.perf_counter())

    # Einsum definition from kernel
    einsum = mr_adc.interface.einsum
    einsum_type = mr_adc.interface.einsum_type

    #Excitation Manifold
    ccaa = mr_adc.h1.ccaa

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
    sigma[ccaa] += ascontiguousarray(sigma_KLWU).reshape(-1)

    mr_adc.log.timer_debug("computing sigma H1 h1-h1 CCAA-CCEE", *cput1)

# CCEA <- CCEE
def compute_sigma_vector__H1__h1_h1__CCEA_CCEE(mr_adc, X, sigma):
    cput1 = (logger.process_clock(), logger.perf_counter())

    # Einsum definition from kernel
    einsum = mr_adc.interface.einsum
    einsum_type = mr_adc.interface.einsum_type

    #Excitation Manifold
    ccea = mr_adc.h1.ccea

    ## Molecular Orbitals Energies
    e_extern = mr_adc.mo_energy.e
    
    ## One-electron integrals
    h_aa = mr_adc.h1eff.aa
    h_ae = mr_adc.h1eff.ae

    ## Two-electron integrals
    v_aaae = mr_adc.v2e.aaae
    v_xxae = mr_adc.v2e.xxae
    v_xaex = mr_adc.v2e.xaex
    v_aaaa = mr_adc.v2e.aaaa
    #v_aeee = mr_adc.v2e.aeee

    ## Amplitudes
    t1_ae = mr_adc.t1.ae 
    t1_aaae = mr_adc.t1.aaae

    # Reduced Density Matrices
    rdm_ca = mr_adc.rdm.ca
    rdm_ccaa = mr_adc.rdm.ccaa

    sigma_KLCW  = einsum('KLCa,Wa->KLCW', X, h_ae, optimize = einsum_type)
    #sigma_KLCW += einsum('KLab,WbCa->KLCW', X, v_aeee, optimize = einsum_type)
    sigma_KLCW += 2 * einsum('KiCa,LWai->KLCW', X, v_xaex, optimize = einsum_type)
    sigma_KLCW -= einsum('KiCa,iLWa->KLCW', X, v_xxae, optimize = einsum_type)
    sigma_KLCW -= einsum('KiaC,LWai->KLCW', X, v_xaex, optimize = einsum_type)
    sigma_KLCW -= einsum('iLCa,iKWa->KLCW', X, v_xxae, optimize = einsum_type)
    sigma_KLCW += einsum('KLCa,a,Wa->KLCW', X, e_extern, t1_ae, optimize = einsum_type)
    sigma_KLCW -= einsum('KLCa,Wx,xa->KLCW', X, h_aa, t1_ae, optimize = einsum_type)
    sigma_KLCW -= 1/2 * einsum('KLCa,Wxya,yx->KLCW', X, v_aaae, rdm_ca, optimize = einsum_type)
    sigma_KLCW += einsum('KLCa,xyWa,xy->KLCW', X, v_aaae, rdm_ca, optimize = einsum_type)
    #sigma_KLCW -= 1/2 * einsum('KLab,xbCa,Wx->KLCW', X, v_aeee, rdm_ca, optimize = einsum_type)
    sigma_KLCW -= einsum('KiCa,Lxai,Wx->KLCW', X, v_xaex, rdm_ca, optimize = einsum_type)
    sigma_KLCW += 1/2 * einsum('KiCa,iLxa,Wx->KLCW', X, v_xxae, rdm_ca, optimize = einsum_type)
    sigma_KLCW += 1/2 * einsum('KiaC,Lxai,Wx->KLCW', X, v_xaex, rdm_ca, optimize = einsum_type)
    sigma_KLCW += 1/2 * einsum('iLCa,iKxa,Wx->KLCW', X, v_xxae, rdm_ca, optimize = einsum_type)
    sigma_KLCW -= 1/2 * einsum('KLCa,a,Wxya,xy->KLCW', X, e_extern, t1_aaae, rdm_ca, optimize = einsum_type)
    sigma_KLCW += einsum('KLCa,a,xWya,xy->KLCW', X, e_extern, t1_aaae, rdm_ca, optimize = einsum_type)
    sigma_KLCW += 1/2 * einsum('KLCa,Wx,xyza,zy->KLCW', X, h_aa, t1_aaae, rdm_ca, optimize = einsum_type)
    sigma_KLCW -= einsum('KLCa,Wx,yxza,zy->KLCW', X, h_aa, t1_aaae, rdm_ca, optimize = einsum_type)
    sigma_KLCW += 1/2 * einsum('KLCa,xy,Wxza,yz->KLCW', X, h_aa, t1_aaae, rdm_ca, optimize = einsum_type)
    sigma_KLCW -= 1/2 * einsum('KLCa,xy,Wzxa,yz->KLCW', X, h_aa, t1_aaae, rdm_ca, optimize = einsum_type)
    sigma_KLCW -= einsum('KLCa,xy,xWza,yz->KLCW', X, h_aa, t1_aaae, rdm_ca, optimize = einsum_type)
    sigma_KLCW += einsum('KLCa,xy,zWxa,yz->KLCW', X, h_aa, t1_aaae, rdm_ca, optimize = einsum_type)
    sigma_KLCW += 1/2 * einsum('KLCa,Wxya,xzwu,ywzu->KLCW', X, t1_aaae, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KLCW -= 1/2 * einsum('KLCa,Wxya,yzwu,xwzu->KLCW', X, t1_aaae, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KLCW -= einsum('KLCa,xWya,xzwu,ywzu->KLCW', X, t1_aaae, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KLCW += einsum('KLCa,xWya,yzwu,xwzu->KLCW', X, t1_aaae, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KLCW -= einsum('KLCa,xa,Wxyz,yz->KLCW', X, t1_ae, v_aaaa, rdm_ca, optimize = einsum_type)
    sigma_KLCW += 1/2 * einsum('KLCa,xa,Wyzx,zy->KLCW', X, t1_ae, v_aaaa, rdm_ca, optimize = einsum_type)
    sigma_KLCW += 1/2 * einsum('KLCa,xyza,Wwux,zwuy->KLCW', X, t1_aaae, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KLCW += 1/2 * einsum('KLCa,xyza,Wwuy,zwxu->KLCW', X, t1_aaae, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KLCW -= 1/2 * einsum('KLCa,xyza,Wwzu,yxwu->KLCW', X, t1_aaae, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KLCW += 1/2 * einsum('KLCa,xyza,Wxwu,zuyw->KLCW', X, t1_aaae, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KLCW += 1/2 * einsum('KLCa,xyza,Wxwy,zw->KLCW', X, t1_aaae, v_aaaa, rdm_ca, optimize = einsum_type)
    sigma_KLCW -= einsum('KLCa,xyza,Wywu,zuxw->KLCW', X, t1_aaae, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KLCW -= einsum('KLCa,xyza,Wywx,zw->KLCW', X, t1_aaae, v_aaaa, rdm_ca, optimize = einsum_type)
    sigma[ccea] += ascontiguousarray(sigma_KLCW).reshape(-1)

    mr_adc.log.timer_debug("computing sigma H1 h1-h1 CCEA-CCEE", *cput1)

def compute_sigma_vector__H1__h1_h1__CCEA_CCEE__V_AEEE(mr_adc, X, sigma, v_aeee):
    cput1 = (logger.process_clock(), logger.perf_counter())

    # Einsum definition from kernel
    einsum = mr_adc.interface.einsum
    einsum_type = mr_adc.interface.einsum_type

    # Reduced Density Matrices
    rdm_ca = mr_adc.rdm.ca
    
    sigma_KLCW  = einsum('KLab,WbCa->KLCW', X, v_aeee, optimize = einsum_type)
    sigma_KLCW -= 1/2 * einsum('KLab,xbCa,Wx->KLCW', X, v_aeee, rdm_ca, optimize = einsum_type)

    mr_adc.log.timer_debug("contracting v2e.aeee", *cput1)
    return sigma_KLCW

# CCEE <- CCEE
def compute_sigma_vector__H1__h1_h1__CCEE_CCEE(mr_adc, X, sigma):
    cput1 = (logger.process_clock(), logger.perf_counter())

    # Einsum definition from kernel
    einsum = mr_adc.interface.einsum
    einsum_type = mr_adc.interface.einsum_type
    dot = mr_adc.interface.dot

    #Excitation Manifold
    ccee = mr_adc.h1.ccee

    ## Indices
    cc_tril_ind = mr_adc.h1.cc_tril_ind 
    ee_tril_ind = mr_adc.h1.ee_tril_ind 

    ## Two-electron integrals
    v_xxxx = mr_adc.v2e.xxxx
    v_xxee = mr_adc.v2e.xxee
    v_xeex = mr_adc.v2e.xeex
    #v_eeee = mr_adc.v2e.eeee

    # Variables from kernel
    ncvs = mr_adc.ncvs
    nextern = mr_adc.nextern

    #sigma_KLCD  = einsum('KLab,CaDb->KLCD', X, v_eeee, optimize = einsum_type)
    sigma_KLCD  = 2 * einsum('KiCa,LDai->KLCD', X, v_xeex, optimize = einsum_type)
    sigma_KLCD -= einsum('KiCa,iLDa->KLCD', X, v_xxee, optimize = einsum_type)
    sigma_KLCD -= einsum('KiaC,LDai->KLCD', X, v_xeex, optimize = einsum_type)
    sigma_KLCD -= einsum('KiaD,iLCa->KLCD', X, v_xxee, optimize = einsum_type)
    sigma_KLCD += 2 * einsum('LiDa,KCai->KLCD', X, v_xeex, optimize = einsum_type)
    sigma_KLCD -= einsum('LiDa,iKCa->KLCD', X, v_xxee, optimize = einsum_type)
    sigma_KLCD -= einsum('LiaC,iKDa->KLCD', X, v_xxee, optimize = einsum_type)
    sigma_KLCD -= einsum('LiaD,KCai->KLCD', X, v_xeex, optimize = einsum_type)
    sigma_KLCD += einsum('ijCD,KiLj->KLCD', X, v_xxxx, optimize = einsum_type)

    # Pack X for ladder contractions
    X = ascontiguousarray(X.reshape(ncvs*ncvs, -1).T)
    temp = zeros((nextern, nextern, ncvs*ncvs))

    chunks = tools.calculate_chunks(mr_adc, nextern, [nextern, nextern, nextern], ntensors = 2)
    for i_chunk, (s_chunk, f_chunk) in enumerate(chunks):
        cput2 = (logger.process_clock(), logger.perf_counter())
        mr_adc.log.debug("v2e.eeee [%i/%i], chunk [%i:%i]", i_chunk + 1, len(chunks), s_chunk, f_chunk)
    
        ## Two-electron integral
        if mr_adc.interface.with_df:
            v_eeee = get_eeee_df(mr_adc, mr_adc.v2e.Lee, s_chunk, f_chunk)
        else:
            v_eeee = unpack_v2e_eeee(mr_adc, mr_adc.v2e.eeee, s_chunk, f_chunk)
        
        # Contractions using dot products
        temp[s_chunk:f_chunk] += dot(v_eeee, X).reshape(-1, nextern, ncvs*ncvs)

        del(v_eeee)
        mr_adc.log.timer_debug("contracting v2e.eeee", *cput2)

    sigma_KLCD += temp.transpose(2,0,1).reshape((ncvs, ncvs, nextern, nextern))
    del(temp, X)
 
    sigma[ccee] += ascontiguousarray(sigma_KLCD).reshape(-1)

    mr_adc.log.timer_debug("computing sigma H1 h1-h1 CCEE-CCEE", *cput1)

# CAEE <- CCEE
def compute_sigma_vector__H1__h1_h1__CAEE_CCEE(mr_adc, X, sigma):
    cput1 = (logger.process_clock(), logger.perf_counter())

    # Einsum definition from kernel
    einsum = mr_adc.interface.einsum
    einsum_type = mr_adc.interface.einsum_type

    #Excitation Manifold
    caee = mr_adc.h1.caee

    ## Molecular Orbitals Energies
    e_cvs = mr_adc.mo_energy.x
    
    ## One-electron integrals
    h_xa = mr_adc.h1eff.xa
    h_aa = mr_adc.h1eff.aa

    ## Two-electron integrals
    v_xxxa = mr_adc.v2e.xxxa
    v_xaaa = mr_adc.v2e.xaaa
    v_xaee = mr_adc.v2e.xaee
    v_xeea = mr_adc.v2e.xeea
    v_aaaa = mr_adc.v2e.aaaa

    ## Amplitudes
    t1_xa = mr_adc.t1.xa
    t1_xaaa = mr_adc.t1.xaaa

    # Reduced Density Matrices
    rdm_ca = mr_adc.rdm.ca
    rdm_ccaa = mr_adc.rdm.ccaa

    sigma_KWCD =- einsum('KiCD,iW->KWCD', X, h_xa, optimize = einsum_type)
    sigma_KWCD += einsum('KiCD,i,iW->KWCD', X, e_cvs, t1_xa, optimize = einsum_type)
    sigma_KWCD -= einsum('KiCD,Wx,ix->KWCD', X, h_aa, t1_xa, optimize = einsum_type)
    sigma_KWCD -= einsum('KiCD,iWxy,yx->KWCD', X, v_xaaa, rdm_ca, optimize = einsum_type)
    sigma_KWCD += 1/2 * einsum('KiCD,ixyW,xy->KWCD', X, v_xaaa, rdm_ca, optimize = einsum_type)
    sigma_KWCD += einsum('KiCa,iaDx,Wx->KWCD', X, v_xeea, rdm_ca, optimize = einsum_type)
    sigma_KWCD -= 1/2 * einsum('KiCa,ixDa,Wx->KWCD', X, v_xaee, rdm_ca, optimize = einsum_type)
    sigma_KWCD -= 1/2 * einsum('KiaC,iaDx,Wx->KWCD', X, v_xeea, rdm_ca, optimize = einsum_type)
    sigma_KWCD -= 1/2 * einsum('KiaD,ixCa,Wx->KWCD', X, v_xaee, rdm_ca, optimize = einsum_type)
    sigma_KWCD += 1/2 * einsum('ijCD,iKjx,Wx->KWCD', X, v_xxxa, rdm_ca, optimize = einsum_type)
    sigma_KWCD += einsum('KiCD,i,ixWy,yx->KWCD', X, e_cvs, t1_xaaa, rdm_ca, optimize = einsum_type)
    sigma_KWCD -= 1/2 * einsum('KiCD,i,ixyW,yx->KWCD', X, e_cvs, t1_xaaa, rdm_ca, optimize = einsum_type)
    sigma_KWCD -= einsum('KiCD,Wx,iyxz,yz->KWCD', X, h_aa, t1_xaaa, rdm_ca, optimize = einsum_type)
    sigma_KWCD += 1/2 * einsum('KiCD,Wx,iyzx,yz->KWCD', X, h_aa, t1_xaaa, rdm_ca, optimize = einsum_type)
    sigma_KWCD += einsum('KiCD,xy,ixWz,yz->KWCD', X, h_aa, t1_xaaa, rdm_ca, optimize = einsum_type)
    sigma_KWCD -= 1/2 * einsum('KiCD,xy,ixzW,yz->KWCD', X, h_aa, t1_xaaa, rdm_ca, optimize = einsum_type)
    sigma_KWCD -= einsum('KiCD,xy,izWx,yz->KWCD', X, h_aa, t1_xaaa, rdm_ca, optimize = einsum_type)
    sigma_KWCD += 1/2 * einsum('KiCD,xy,izxW,yz->KWCD', X, h_aa, t1_xaaa, rdm_ca, optimize = einsum_type)
    sigma_KWCD -= einsum('KiCD,ix,Wxyz,yz->KWCD', X, t1_xa, v_aaaa, rdm_ca, optimize = einsum_type)
    sigma_KWCD += 1/2 * einsum('KiCD,ix,Wyzx,zy->KWCD', X, t1_xa, v_aaaa, rdm_ca, optimize = einsum_type)
    sigma_KWCD += einsum('KiCD,ixWy,xzwu,ywzu->KWCD', X, t1_xaaa, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWCD -= einsum('KiCD,ixWy,yzwu,xwzu->KWCD', X, t1_xaaa, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWCD -= 1/2 * einsum('KiCD,ixyW,xzwu,ywzu->KWCD', X, t1_xaaa, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWCD += 1/2 * einsum('KiCD,ixyW,yzwu,xwzu->KWCD', X, t1_xaaa, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWCD += 1/2 * einsum('KiCD,ixyz,Wwuy,xwzu->KWCD', X, t1_xaaa, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWCD += 1/2 * einsum('KiCD,ixyz,Wwuz,xwuy->KWCD', X, t1_xaaa, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWCD -= 1/2 * einsum('KiCD,ixyz,Wwxu,yzwu->KWCD', X, t1_xaaa, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWCD -= einsum('KiCD,ixyz,Wywu,xuzw->KWCD', X, t1_xaaa, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWCD -= einsum('KiCD,ixyz,Wywz,xw->KWCD', X, t1_xaaa, v_aaaa, rdm_ca, optimize = einsum_type)
    sigma_KWCD += 1/2 * einsum('KiCD,ixyz,Wzwu,xuyw->KWCD', X, t1_xaaa, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWCD += 1/2 * einsum('KiCD,ixyz,Wzwy,xw->KWCD', X, t1_xaaa, v_aaaa, rdm_ca, optimize = einsum_type)
    sigma[caee] += ascontiguousarray(sigma_KWCD).reshape(-1)

    mr_adc.log.timer_debug("computing sigma H1 h1-h1 CAEE-CCEE", *cput1)

# CAAA <- CCEE
def compute_sigma_vector__H1__h1_h1__CAAA_CCEE(mr_adc, X, sigma):
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
    e_cvs = mr_adc.mo_energy.x
    e_extern = mr_adc.mo_energy.e
 
    ## One-electron integrals
    h_aa = mr_adc.h1eff.aa
 
    ## Two-electron integrals
    v_xeae = mr_adc.v2e.xeae
    v_aaaa = mr_adc.v2e.aaaa
  
    ## Amplitudes
    t1_xaee = mr_adc.t1.xaee
  
    # Reduced Density Matrices
    rdm_ca = mr_adc.rdm.ca
    rdm_ccaa = mr_adc.rdm.ccaa

    sigma_KWUV_aaaa =- 1/2 * einsum('Kiab,iaUb,VW->KWUV', X, v_xeae, rdm_ca, optimize = einsum_type)
    sigma_KWUV_aaaa += 1/2 * einsum('Kiab,iaVb,UW->KWUV', X, v_xeae, rdm_ca, optimize = einsum_type)
    sigma_KWUV_aaaa += einsum('Kiab,ibUa,VW->KWUV', X, v_xeae, rdm_ca, optimize = einsum_type)
    sigma_KWUV_aaaa -= einsum('Kiab,ibVa,UW->KWUV', X, v_xeae, rdm_ca, optimize = einsum_type)
    sigma_KWUV_aaaa -= 1/2 * einsum('Kiab,a,iUab,VW->KWUV', X, e_extern, t1_xaee, rdm_ca, optimize = einsum_type)
    sigma_KWUV_aaaa += einsum('Kiab,a,iUba,VW->KWUV', X, e_extern, t1_xaee, rdm_ca, optimize = einsum_type)
    sigma_KWUV_aaaa += 1/2 * einsum('Kiab,a,iVab,UW->KWUV', X, e_extern, t1_xaee, rdm_ca, optimize = einsum_type)
    sigma_KWUV_aaaa -= einsum('Kiab,a,iVba,UW->KWUV', X, e_extern, t1_xaee, rdm_ca, optimize = einsum_type)
    sigma_KWUV_aaaa -= 1/2 * einsum('Kiab,b,iUab,VW->KWUV', X, e_extern, t1_xaee, rdm_ca, optimize = einsum_type)
    sigma_KWUV_aaaa += einsum('Kiab,b,iUba,VW->KWUV', X, e_extern, t1_xaee, rdm_ca, optimize = einsum_type)
    sigma_KWUV_aaaa += 1/2 * einsum('Kiab,b,iVab,UW->KWUV', X, e_extern, t1_xaee, rdm_ca, optimize = einsum_type)
    sigma_KWUV_aaaa -= einsum('Kiab,b,iVba,UW->KWUV', X, e_extern, t1_xaee, rdm_ca, optimize = einsum_type)
    sigma_KWUV_aaaa += 1/2 * einsum('Kiab,i,iUab,VW->KWUV', X, e_cvs, t1_xaee, rdm_ca, optimize = einsum_type)
    sigma_KWUV_aaaa -= einsum('Kiab,i,iUba,VW->KWUV', X, e_cvs, t1_xaee, rdm_ca, optimize = einsum_type)
    sigma_KWUV_aaaa -= 1/2 * einsum('Kiab,i,iVab,UW->KWUV', X, e_cvs, t1_xaee, rdm_ca, optimize = einsum_type)
    sigma_KWUV_aaaa += einsum('Kiab,i,iVba,UW->KWUV', X, e_cvs, t1_xaee, rdm_ca, optimize = einsum_type)
    sigma_KWUV_aaaa += 1/2 * einsum('Kiab,Ux,ixab,VW->KWUV', X, h_aa, t1_xaee, rdm_ca, optimize = einsum_type)
    sigma_KWUV_aaaa -= einsum('Kiab,Ux,ixba,VW->KWUV', X, h_aa, t1_xaee, rdm_ca, optimize = einsum_type)
    sigma_KWUV_aaaa -= 1/2 * einsum('Kiab,Vx,ixab,UW->KWUV', X, h_aa, t1_xaee, rdm_ca, optimize = einsum_type)
    sigma_KWUV_aaaa += einsum('Kiab,Vx,ixba,UW->KWUV', X, h_aa, t1_xaee, rdm_ca, optimize = einsum_type)
    sigma_KWUV_aaaa += 1/2 * einsum('Kiab,ixab,UxVy,Wy->KWUV', X, t1_xaee, v_aaaa, rdm_ca, optimize = einsum_type)
    sigma_KWUV_aaaa += 1/2 * einsum('Kiab,ixab,Uxyz,VzWy->KWUV', X, t1_xaee, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWUV_aaaa -= 1/2 * einsum('Kiab,ixab,UyVx,Wy->KWUV', X, t1_xaee, v_aaaa, rdm_ca, optimize = einsum_type)
    sigma_KWUV_aaaa -= 1/6 * einsum('Kiab,ixab,Uyzx,VyWz->KWUV', X, t1_xaee, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWUV_aaaa += 1/6 * einsum('Kiab,ixab,Uyzx,VyzW->KWUV', X, t1_xaee, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWUV_aaaa -= 1/2 * einsum('Kiab,ixab,Vxyz,UzWy->KWUV', X, t1_xaee, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWUV_aaaa += 1/6 * einsum('Kiab,ixab,Vyzx,UyWz->KWUV', X, t1_xaee, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWUV_aaaa -= 1/6 * einsum('Kiab,ixab,Vyzx,UyzW->KWUV', X, t1_xaee, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWUV_aaaa -= 1/6 * einsum('Kiab,ixab,Wyxz,UVyz->KWUV', X, t1_xaee, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWUV_aaaa += 1/6 * einsum('Kiab,ixab,Wyxz,UVzy->KWUV', X, t1_xaee, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWUV_aaaa -= einsum('Kiab,ixba,UxVy,Wy->KWUV', X, t1_xaee, v_aaaa, rdm_ca, optimize = einsum_type)
    sigma_KWUV_aaaa -= einsum('Kiab,ixba,Uxyz,VzWy->KWUV', X, t1_xaee, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWUV_aaaa += einsum('Kiab,ixba,UyVx,Wy->KWUV', X, t1_xaee, v_aaaa, rdm_ca, optimize = einsum_type)
    sigma_KWUV_aaaa += 1/3 * einsum('Kiab,ixba,Uyzx,VyWz->KWUV', X, t1_xaee, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWUV_aaaa -= 1/3 * einsum('Kiab,ixba,Uyzx,VyzW->KWUV', X, t1_xaee, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWUV_aaaa += einsum('Kiab,ixba,Vxyz,UzWy->KWUV', X, t1_xaee, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWUV_aaaa -= 1/3 * einsum('Kiab,ixba,Vyzx,UyWz->KWUV', X, t1_xaee, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWUV_aaaa += 1/3 * einsum('Kiab,ixba,Vyzx,UyzW->KWUV', X, t1_xaee, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWUV_aaaa += 1/3 * einsum('Kiab,ixba,Wyxz,UVyz->KWUV', X, t1_xaee, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWUV_aaaa -= 1/3 * einsum('Kiab,ixba,Wyxz,UVzy->KWUV', X, t1_xaee, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma[caaa__aaaa] += ascontiguousarray(sigma_KWUV_aaaa[:, :, aa_tril_ind[0], aa_tril_ind[1]]).reshape(-1)

    sigma_KWUV_abab =- 1/2 * einsum('Kiab,iaUb,VW->KWUV', X, v_xeae, rdm_ca, optimize = einsum_type)
    sigma_KWUV_abab += einsum('Kiab,ibUa,VW->KWUV', X, v_xeae, rdm_ca, optimize = einsum_type)
    sigma_KWUV_abab -= 1/2 * einsum('Kiab,a,iUab,VW->KWUV', X, e_extern, t1_xaee, rdm_ca, optimize = einsum_type)
    sigma_KWUV_abab += einsum('Kiab,a,iUba,VW->KWUV', X, e_extern, t1_xaee, rdm_ca, optimize = einsum_type)
    sigma_KWUV_abab -= 1/2 * einsum('Kiab,b,iUab,VW->KWUV', X, e_extern, t1_xaee, rdm_ca, optimize = einsum_type)
    sigma_KWUV_abab += einsum('Kiab,b,iUba,VW->KWUV', X, e_extern, t1_xaee, rdm_ca, optimize = einsum_type)
    sigma_KWUV_abab += 1/2 * einsum('Kiab,i,iUab,VW->KWUV', X, e_cvs, t1_xaee, rdm_ca, optimize = einsum_type)
    sigma_KWUV_abab -= einsum('Kiab,i,iUba,VW->KWUV', X, e_cvs, t1_xaee, rdm_ca, optimize = einsum_type)
    sigma_KWUV_abab += 1/2 * einsum('Kiab,Ux,ixab,VW->KWUV', X, h_aa, t1_xaee, rdm_ca, optimize = einsum_type)
    sigma_KWUV_abab -= einsum('Kiab,Ux,ixba,VW->KWUV', X, h_aa, t1_xaee, rdm_ca, optimize = einsum_type)
    sigma_KWUV_abab += 1/2 * einsum('Kiab,ixab,UxVy,Wy->KWUV', X, t1_xaee, v_aaaa, rdm_ca, optimize = einsum_type)
    sigma_KWUV_abab += 1/2 * einsum('Kiab,ixab,Uxyz,VzWy->KWUV', X, t1_xaee, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWUV_abab -= 1/3 * einsum('Kiab,ixab,Uyzx,VyWz->KWUV', X, t1_xaee, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWUV_abab -= 1/6 * einsum('Kiab,ixab,Uyzx,VyzW->KWUV', X, t1_xaee, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWUV_abab -= 1/6 * einsum('Kiab,ixab,Vyzx,UyWz->KWUV', X, t1_xaee, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWUV_abab -= 1/3 * einsum('Kiab,ixab,Vyzx,UyzW->KWUV', X, t1_xaee, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWUV_abab += 1/6 * einsum('Kiab,ixab,Wyxz,UVyz->KWUV', X, t1_xaee, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWUV_abab += 1/3 * einsum('Kiab,ixab,Wyxz,UVzy->KWUV', X, t1_xaee, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWUV_abab -= einsum('Kiab,ixba,UxVy,Wy->KWUV', X, t1_xaee, v_aaaa, rdm_ca, optimize = einsum_type)
    sigma_KWUV_abab -= einsum('Kiab,ixba,Uxyz,VzWy->KWUV', X, t1_xaee, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWUV_abab += 2/3 * einsum('Kiab,ixba,Uyzx,VyWz->KWUV', X, t1_xaee, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWUV_abab += 1/3 * einsum('Kiab,ixba,Uyzx,VyzW->KWUV', X, t1_xaee, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWUV_abab += 1/3 * einsum('Kiab,ixba,Vyzx,UyWz->KWUV', X, t1_xaee, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWUV_abab += 2/3 * einsum('Kiab,ixba,Vyzx,UyzW->KWUV', X, t1_xaee, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWUV_abab -= 1/3 * einsum('Kiab,ixba,Wyxz,UVyz->KWUV', X, t1_xaee, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWUV_abab -= 2/3 * einsum('Kiab,ixba,Wyxz,UVzy->KWUV', X, t1_xaee, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma[caaa__abab] += ascontiguousarray(sigma_KWUV_abab).reshape(-1)

    mr_adc.log.timer_debug("computing sigma H1 h1-h1 CAAA-CCEE", *cput1)

# CAEA <- CCEE
def compute_sigma_vector__H1__h1_h1__CAEA_CCEE(mr_adc, X, sigma):
    cput1 = (logger.process_clock(), logger.perf_counter())

    # Einsum definition from kernel
    einsum = mr_adc.interface.einsum
    einsum_type = mr_adc.interface.einsum_type

    #Excitation Manifold
    caea__aaaa = mr_adc.h1.caea__aaaa
    caea__abab = mr_adc.h1.caea__abab
    caea__baab = mr_adc.h1.caea__baab
 
    ## Molecular Orbitals Energies
    e_cvs = mr_adc.mo_energy.x
    e_extern = mr_adc.mo_energy.e
 
    ## One-electron integrals
    h_xe = mr_adc.h1eff.xe
    h_aa = mr_adc.h1eff.aa
 
    ## Two-electron integrals
    v_xxxe = mr_adc.v2e.xxxe
    v_xaae = mr_adc.v2e.xaae
    v_xeaa = mr_adc.v2e.xeaa
    v_aaaa = mr_adc.v2e.aaaa
    #v_xeee = mr_adc.v2e.xeee
  
    ## Amplitudes
    t1_xe = mr_adc.t1.xe
    t1_xaae = mr_adc.t1.xaae
    t1_xaea = mr_adc.t1.xaea
    
    # Reduced Density Matrices
    rdm_ca = mr_adc.rdm.ca
    rdm_ccaa = mr_adc.rdm.ccaa

    sigma_KWCU_aaaa  = 1/2 * einsum('KiCa,iWxa,Ux->KWCU', X, v_xaae, rdm_ca, optimize = einsum_type)
    sigma_KWCU_aaaa += einsum('KiCa,iaUx,Wx->KWCU', X, v_xeaa, rdm_ca, optimize = einsum_type)
    sigma_KWCU_aaaa -= einsum('KiCa,iaxW,Ux->KWCU', X, v_xeaa, rdm_ca, optimize = einsum_type)
    sigma_KWCU_aaaa -= 1/2 * einsum('KiCa,ixUa,Wx->KWCU', X, v_xaae, rdm_ca, optimize = einsum_type)
    sigma_KWCU_aaaa -= 1/2 * einsum('KiaC,iWxa,Ux->KWCU', X, v_xaae, rdm_ca, optimize = einsum_type)
    sigma_KWCU_aaaa -= 1/2 * einsum('KiaC,iaUx,Wx->KWCU', X, v_xeaa, rdm_ca, optimize = einsum_type)
    sigma_KWCU_aaaa += 1/2 * einsum('KiaC,iaxW,Ux->KWCU', X, v_xeaa, rdm_ca, optimize = einsum_type)
    sigma_KWCU_aaaa += 1/2 * einsum('KiaC,ixUa,Wx->KWCU', X, v_xaae, rdm_ca, optimize = einsum_type)
    #sigma_KWCU_aaaa -= 1/2 * einsum('Kiab,iaCb,UW->KWCU', X, v_xeee, rdm_ca, optimize = einsum_type)
    #sigma_KWCU_aaaa += einsum('Kiab,ibCa,UW->KWCU', X, v_xeee, rdm_ca, optimize = einsum_type)
    sigma_KWCU_aaaa -= einsum('ijCa,iKja,UW->KWCU', X, v_xxxe, rdm_ca, optimize = einsum_type)
    sigma_KWCU_aaaa += 1/2 * einsum('ijCa,jKia,UW->KWCU', X, v_xxxe, rdm_ca, optimize = einsum_type)
    sigma_KWCU_aaaa += einsum('KiCa,a,iUax,Wx->KWCU', X, e_extern, t1_xaea, rdm_ca, optimize = einsum_type)
    sigma_KWCU_aaaa -= 1/2 * einsum('KiCa,a,iUxa,Wx->KWCU', X, e_extern, t1_xaae, rdm_ca, optimize = einsum_type)
    sigma_KWCU_aaaa += 1/2 * einsum('KiCa,a,ixWa,Ux->KWCU', X, e_extern, t1_xaae, rdm_ca, optimize = einsum_type)
    sigma_KWCU_aaaa -= einsum('KiCa,a,ixaW,Ux->KWCU', X, e_extern, t1_xaea, rdm_ca, optimize = einsum_type)
    sigma_KWCU_aaaa -= einsum('KiCa,i,iUax,Wx->KWCU', X, e_cvs, t1_xaea, rdm_ca, optimize = einsum_type)
    sigma_KWCU_aaaa += 1/2 * einsum('KiCa,i,iUxa,Wx->KWCU', X, e_cvs, t1_xaae, rdm_ca, optimize = einsum_type)
    sigma_KWCU_aaaa -= 1/2 * einsum('KiCa,i,ixWa,Ux->KWCU', X, e_cvs, t1_xaae, rdm_ca, optimize = einsum_type)
    sigma_KWCU_aaaa += einsum('KiCa,i,ixaW,Ux->KWCU', X, e_cvs, t1_xaea, rdm_ca, optimize = einsum_type)
    sigma_KWCU_aaaa -= 1/2 * einsum('KiaC,a,iUax,Wx->KWCU', X, e_extern, t1_xaea, rdm_ca, optimize = einsum_type)
    sigma_KWCU_aaaa += 1/2 * einsum('KiaC,a,iUxa,Wx->KWCU', X, e_extern, t1_xaae, rdm_ca, optimize = einsum_type)
    sigma_KWCU_aaaa -= 1/2 * einsum('KiaC,a,ixWa,Ux->KWCU', X, e_extern, t1_xaae, rdm_ca, optimize = einsum_type)
    sigma_KWCU_aaaa += 1/2 * einsum('KiaC,a,ixaW,Ux->KWCU', X, e_extern, t1_xaea, rdm_ca, optimize = einsum_type)
    sigma_KWCU_aaaa += 1/2 * einsum('KiaC,i,iUax,Wx->KWCU', X, e_cvs, t1_xaea, rdm_ca, optimize = einsum_type)
    sigma_KWCU_aaaa -= 1/2 * einsum('KiaC,i,iUxa,Wx->KWCU', X, e_cvs, t1_xaae, rdm_ca, optimize = einsum_type)
    sigma_KWCU_aaaa += 1/2 * einsum('KiaC,i,ixWa,Ux->KWCU', X, e_cvs, t1_xaae, rdm_ca, optimize = einsum_type)
    sigma_KWCU_aaaa -= 1/2 * einsum('KiaC,i,ixaW,Ux->KWCU', X, e_cvs, t1_xaea, rdm_ca, optimize = einsum_type)
    sigma_KWCU_aaaa -= einsum('KiCa,Ux,ixay,Wy->KWCU', X, h_aa, t1_xaea, rdm_ca, optimize = einsum_type)
    sigma_KWCU_aaaa += 1/2 * einsum('KiCa,Ux,ixya,Wy->KWCU', X, h_aa, t1_xaae, rdm_ca, optimize = einsum_type)
    sigma_KWCU_aaaa -= einsum('KiCa,Wx,iyax,Uy->KWCU', X, h_aa, t1_xaea, rdm_ca, optimize = einsum_type)
    sigma_KWCU_aaaa += 1/2 * einsum('KiCa,Wx,iyxa,Uy->KWCU', X, h_aa, t1_xaae, rdm_ca, optimize = einsum_type)
    sigma_KWCU_aaaa += einsum('KiCa,xy,iUax,Wy->KWCU', X, h_aa, t1_xaea, rdm_ca, optimize = einsum_type)
    sigma_KWCU_aaaa -= 1/2 * einsum('KiCa,xy,iUxa,Wy->KWCU', X, h_aa, t1_xaae, rdm_ca, optimize = einsum_type)
    sigma_KWCU_aaaa -= 1/2 * einsum('KiCa,xy,ixWa,Uy->KWCU', X, h_aa, t1_xaae, rdm_ca, optimize = einsum_type)
    sigma_KWCU_aaaa += einsum('KiCa,xy,ixaW,Uy->KWCU', X, h_aa, t1_xaea, rdm_ca, optimize = einsum_type)
    sigma_KWCU_aaaa += 1/2 * einsum('KiaC,Ux,ixay,Wy->KWCU', X, h_aa, t1_xaea, rdm_ca, optimize = einsum_type)
    sigma_KWCU_aaaa -= 1/2 * einsum('KiaC,Ux,ixya,Wy->KWCU', X, h_aa, t1_xaae, rdm_ca, optimize = einsum_type)
    sigma_KWCU_aaaa += 1/2 * einsum('KiaC,Wx,iyax,Uy->KWCU', X, h_aa, t1_xaea, rdm_ca, optimize = einsum_type)
    sigma_KWCU_aaaa -= 1/2 * einsum('KiaC,Wx,iyxa,Uy->KWCU', X, h_aa, t1_xaae, rdm_ca, optimize = einsum_type)
    sigma_KWCU_aaaa -= 1/2 * einsum('KiaC,xy,iUax,Wy->KWCU', X, h_aa, t1_xaea, rdm_ca, optimize = einsum_type)
    sigma_KWCU_aaaa += 1/2 * einsum('KiaC,xy,iUxa,Wy->KWCU', X, h_aa, t1_xaae, rdm_ca, optimize = einsum_type)
    sigma_KWCU_aaaa += 1/2 * einsum('KiaC,xy,ixWa,Uy->KWCU', X, h_aa, t1_xaae, rdm_ca, optimize = einsum_type)
    sigma_KWCU_aaaa -= 1/2 * einsum('KiaC,xy,ixaW,Uy->KWCU', X, h_aa, t1_xaea, rdm_ca, optimize = einsum_type)
    sigma_KWCU_aaaa += einsum('KiCa,iUax,xyzw,Wzyw->KWCU', X, t1_xaea, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_aaaa -= 1/2 * einsum('KiCa,iUxa,xyzw,Wzyw->KWCU', X, t1_xaae, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_aaaa -= 1/2 * einsum('KiCa,ixWa,xyzw,Uzyw->KWCU', X, t1_xaae, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_aaaa += einsum('KiCa,ixaW,xyzw,Uzyw->KWCU', X, t1_xaea, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_aaaa -= einsum('KiCa,ixay,Uxzw,Wzyw->KWCU', X, t1_xaea, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_aaaa -= einsum('KiCa,ixay,Uzwx,Wwzy->KWCU', X, t1_xaea, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_aaaa += einsum('KiCa,ixay,Uzyw,Wxzw->KWCU', X, t1_xaea, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_aaaa -= einsum('KiCa,ixay,Wyzw,Uzxw->KWCU', X, t1_xaea, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_aaaa -= einsum('KiCa,ixay,Wzwy,Uwzx->KWCU', X, t1_xaea, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_aaaa += einsum('KiCa,ixay,Wzxw,Uyzw->KWCU', X, t1_xaea, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_aaaa += 1/2 * einsum('KiCa,ixya,Uxzw,Wzyw->KWCU', X, t1_xaae, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_aaaa += 1/2 * einsum('KiCa,ixya,Uzwx,Wwzy->KWCU', X, t1_xaae, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_aaaa -= 1/2 * einsum('KiCa,ixya,Uzyw,Wxzw->KWCU', X, t1_xaae, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_aaaa += 1/2 * einsum('KiCa,ixya,Wyzw,Uzxw->KWCU', X, t1_xaae, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_aaaa += 1/2 * einsum('KiCa,ixya,Wzwy,Uwzx->KWCU', X, t1_xaae, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_aaaa -= 1/2 * einsum('KiCa,ixya,Wzxw,Uyzw->KWCU', X, t1_xaae, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_aaaa -= 1/2 * einsum('KiaC,iUax,xyzw,Wzyw->KWCU', X, t1_xaea, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_aaaa += 1/2 * einsum('KiaC,iUxa,xyzw,Wzyw->KWCU', X, t1_xaae, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_aaaa += 1/2 * einsum('KiaC,ixWa,xyzw,Uzyw->KWCU', X, t1_xaae, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_aaaa -= 1/2 * einsum('KiaC,ixaW,xyzw,Uzyw->KWCU', X, t1_xaea, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_aaaa += 1/2 * einsum('KiaC,ixay,Uxzw,Wzyw->KWCU', X, t1_xaea, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_aaaa += 1/2 * einsum('KiaC,ixay,Uzwx,Wwzy->KWCU', X, t1_xaea, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_aaaa -= 1/2 * einsum('KiaC,ixay,Uzyw,Wxzw->KWCU', X, t1_xaea, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_aaaa += 1/2 * einsum('KiaC,ixay,Wyzw,Uzxw->KWCU', X, t1_xaea, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_aaaa += 1/2 * einsum('KiaC,ixay,Wzwy,Uwzx->KWCU', X, t1_xaea, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_aaaa -= 1/2 * einsum('KiaC,ixay,Wzxw,Uyzw->KWCU', X, t1_xaea, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_aaaa -= 1/2 * einsum('KiaC,ixya,Uxzw,Wzyw->KWCU', X, t1_xaae, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_aaaa += 1/6 * einsum('KiaC,ixya,Uzwx,Wwyz->KWCU', X, t1_xaae, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_aaaa -= 1/6 * einsum('KiaC,ixya,Uzwx,Wwzy->KWCU', X, t1_xaae, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_aaaa -= 1/6 * einsum('KiaC,ixya,Uzyw,Wxwz->KWCU', X, t1_xaae, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_aaaa += 1/6 * einsum('KiaC,ixya,Uzyw,Wxzw->KWCU', X, t1_xaae, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_aaaa -= 1/2 * einsum('KiaC,ixya,Wyzw,Uzxw->KWCU', X, t1_xaae, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_aaaa += 1/6 * einsum('KiaC,ixya,Wzwy,Uwxz->KWCU', X, t1_xaae, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_aaaa -= 1/6 * einsum('KiaC,ixya,Wzwy,Uwzx->KWCU', X, t1_xaae, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_aaaa -= 1/6 * einsum('KiaC,ixya,Wzxw,Uywz->KWCU', X, t1_xaae, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_aaaa += 1/6 * einsum('KiaC,ixya,Wzxw,Uyzw->KWCU', X, t1_xaae, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma[caea__aaaa] += ascontiguousarray(sigma_KWCU_aaaa).reshape(-1)

    sigma_KWCU_abab  = 1/2 * einsum('KiCa,iWxa,Ux->KWCU', X, v_xaae, rdm_ca, optimize = einsum_type)
    sigma_KWCU_abab += einsum('KiCa,iaUx,Wx->KWCU', X, v_xeaa, rdm_ca, optimize = einsum_type)
    sigma_KWCU_abab -= einsum('KiCa,iaxW,Ux->KWCU', X, v_xeaa, rdm_ca, optimize = einsum_type)
    sigma_KWCU_abab -= 1/2 * einsum('KiCa,ixUa,Wx->KWCU', X, v_xaae, rdm_ca, optimize = einsum_type)
    sigma_KWCU_abab -= 1/2 * einsum('KiaC,iaUx,Wx->KWCU', X, v_xeaa, rdm_ca, optimize = einsum_type)
    sigma_KWCU_abab += 1/2 * einsum('KiaC,iaxW,Ux->KWCU', X, v_xeaa, rdm_ca, optimize = einsum_type)
    #sigma_KWCU_abab -= 1/2 * einsum('Kiab,iaCb,UW->KWCU', X, v_xeee, rdm_ca, optimize = einsum_type)
    #sigma_KWCU_abab += einsum('Kiab,ibCa,UW->KWCU', X, v_xeee, rdm_ca, optimize = einsum_type)
    sigma_KWCU_abab -= einsum('ijCa,iKja,UW->KWCU', X, v_xxxe, rdm_ca, optimize = einsum_type)
    sigma_KWCU_abab += 1/2 * einsum('ijCa,jKia,UW->KWCU', X, v_xxxe, rdm_ca, optimize = einsum_type)
    sigma_KWCU_abab += einsum('KiCa,a,iUax,Wx->KWCU', X, e_extern, t1_xaea, rdm_ca, optimize = einsum_type)
    sigma_KWCU_abab -= 1/2 * einsum('KiCa,a,iUxa,Wx->KWCU', X, e_extern, t1_xaae, rdm_ca, optimize = einsum_type)
    sigma_KWCU_abab += 1/2 * einsum('KiCa,a,ixWa,Ux->KWCU', X, e_extern, t1_xaae, rdm_ca, optimize = einsum_type)
    sigma_KWCU_abab -= einsum('KiCa,a,ixaW,Ux->KWCU', X, e_extern, t1_xaea, rdm_ca, optimize = einsum_type)
    sigma_KWCU_abab -= einsum('KiCa,i,iUax,Wx->KWCU', X, e_cvs, t1_xaea, rdm_ca, optimize = einsum_type)
    sigma_KWCU_abab += 1/2 * einsum('KiCa,i,iUxa,Wx->KWCU', X, e_cvs, t1_xaae, rdm_ca, optimize = einsum_type)
    sigma_KWCU_abab -= 1/2 * einsum('KiCa,i,ixWa,Ux->KWCU', X, e_cvs, t1_xaae, rdm_ca, optimize = einsum_type)
    sigma_KWCU_abab += einsum('KiCa,i,ixaW,Ux->KWCU', X, e_cvs, t1_xaea, rdm_ca, optimize = einsum_type)
    sigma_KWCU_abab -= 1/2 * einsum('KiaC,a,iUax,Wx->KWCU', X, e_extern, t1_xaea, rdm_ca, optimize = einsum_type)
    sigma_KWCU_abab += 1/2 * einsum('KiaC,a,ixaW,Ux->KWCU', X, e_extern, t1_xaea, rdm_ca, optimize = einsum_type)
    sigma_KWCU_abab += 1/2 * einsum('KiaC,i,iUax,Wx->KWCU', X, e_cvs, t1_xaea, rdm_ca, optimize = einsum_type)
    sigma_KWCU_abab -= 1/2 * einsum('KiaC,i,ixaW,Ux->KWCU', X, e_cvs, t1_xaea, rdm_ca, optimize = einsum_type)
    sigma_KWCU_abab -= einsum('KiCa,Ux,ixay,Wy->KWCU', X, h_aa, t1_xaea, rdm_ca, optimize = einsum_type)
    sigma_KWCU_abab += 1/2 * einsum('KiCa,Ux,ixya,Wy->KWCU', X, h_aa, t1_xaae, rdm_ca, optimize = einsum_type)
    sigma_KWCU_abab -= einsum('KiCa,Wx,iyax,Uy->KWCU', X, h_aa, t1_xaea, rdm_ca, optimize = einsum_type)
    sigma_KWCU_abab += 1/2 * einsum('KiCa,Wx,iyxa,Uy->KWCU', X, h_aa, t1_xaae, rdm_ca, optimize = einsum_type)
    sigma_KWCU_abab += einsum('KiCa,xy,iUax,Wy->KWCU', X, h_aa, t1_xaea, rdm_ca, optimize = einsum_type)
    sigma_KWCU_abab -= 1/2 * einsum('KiCa,xy,iUxa,Wy->KWCU', X, h_aa, t1_xaae, rdm_ca, optimize = einsum_type)
    sigma_KWCU_abab -= 1/2 * einsum('KiCa,xy,ixWa,Uy->KWCU', X, h_aa, t1_xaae, rdm_ca, optimize = einsum_type)
    sigma_KWCU_abab += einsum('KiCa,xy,ixaW,Uy->KWCU', X, h_aa, t1_xaea, rdm_ca, optimize = einsum_type)
    sigma_KWCU_abab += 1/2 * einsum('KiaC,Ux,ixay,Wy->KWCU', X, h_aa, t1_xaea, rdm_ca, optimize = einsum_type)
    sigma_KWCU_abab += 1/2 * einsum('KiaC,Wx,iyax,Uy->KWCU', X, h_aa, t1_xaea, rdm_ca, optimize = einsum_type)
    sigma_KWCU_abab -= 1/2 * einsum('KiaC,xy,iUax,Wy->KWCU', X, h_aa, t1_xaea, rdm_ca, optimize = einsum_type)
    sigma_KWCU_abab -= 1/2 * einsum('KiaC,xy,ixaW,Uy->KWCU', X, h_aa, t1_xaea, rdm_ca, optimize = einsum_type)
    sigma_KWCU_abab += einsum('KiCa,iUax,xyzw,Wzyw->KWCU', X, t1_xaea, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_abab -= 1/2 * einsum('KiCa,iUxa,xyzw,Wzyw->KWCU', X, t1_xaae, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_abab -= 1/2 * einsum('KiCa,ixWa,xyzw,Uzyw->KWCU', X, t1_xaae, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_abab += einsum('KiCa,ixaW,xyzw,Uzyw->KWCU', X, t1_xaea, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_abab -= einsum('KiCa,ixay,Uxzw,Wzyw->KWCU', X, t1_xaea, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_abab -= einsum('KiCa,ixay,Uzwx,Wwzy->KWCU', X, t1_xaea, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_abab += einsum('KiCa,ixay,Uzyw,Wxzw->KWCU', X, t1_xaea, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_abab -= einsum('KiCa,ixay,Wyzw,Uzxw->KWCU', X, t1_xaea, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_abab -= einsum('KiCa,ixay,Wzwy,Uwzx->KWCU', X, t1_xaea, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_abab += einsum('KiCa,ixay,Wzxw,Uyzw->KWCU', X, t1_xaea, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_abab += 1/2 * einsum('KiCa,ixya,Uxzw,Wzyw->KWCU', X, t1_xaae, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_abab += 1/2 * einsum('KiCa,ixya,Uzwx,Wwzy->KWCU', X, t1_xaae, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_abab -= 1/2 * einsum('KiCa,ixya,Uzyw,Wxzw->KWCU', X, t1_xaae, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_abab += 1/2 * einsum('KiCa,ixya,Wyzw,Uzxw->KWCU', X, t1_xaae, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_abab += 1/2 * einsum('KiCa,ixya,Wzwy,Uwzx->KWCU', X, t1_xaae, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_abab -= 1/2 * einsum('KiCa,ixya,Wzxw,Uyzw->KWCU', X, t1_xaae, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_abab -= 1/2 * einsum('KiaC,iUax,xyzw,Wzyw->KWCU', X, t1_xaea, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_abab -= 1/2 * einsum('KiaC,ixaW,xyzw,Uzyw->KWCU', X, t1_xaea, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_abab += 1/2 * einsum('KiaC,ixay,Uxzw,Wzyw->KWCU', X, t1_xaea, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_abab += 1/2 * einsum('KiaC,ixay,Uzwx,Wwzy->KWCU', X, t1_xaea, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_abab -= 1/2 * einsum('KiaC,ixay,Uzyw,Wxzw->KWCU', X, t1_xaea, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_abab += 1/2 * einsum('KiaC,ixay,Wyzw,Uzxw->KWCU', X, t1_xaea, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_abab += 1/2 * einsum('KiaC,ixay,Wzwy,Uwzx->KWCU', X, t1_xaea, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_abab -= 1/2 * einsum('KiaC,ixay,Wzxw,Uyzw->KWCU', X, t1_xaea, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_abab -= 1/6 * einsum('KiaC,ixya,Uzwx,Wwyz->KWCU', X, t1_xaae, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_abab -= 1/3 * einsum('KiaC,ixya,Uzwx,Wwzy->KWCU', X, t1_xaae, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_abab += 1/6 * einsum('KiaC,ixya,Uzyw,Wxwz->KWCU', X, t1_xaae, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_abab += 1/3 * einsum('KiaC,ixya,Uzyw,Wxzw->KWCU', X, t1_xaae, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_abab -= 1/6 * einsum('KiaC,ixya,Wzwy,Uwxz->KWCU', X, t1_xaae, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_abab -= 1/3 * einsum('KiaC,ixya,Wzwy,Uwzx->KWCU', X, t1_xaae, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_abab += 1/6 * einsum('KiaC,ixya,Wzxw,Uywz->KWCU', X, t1_xaae, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_abab += 1/3 * einsum('KiaC,ixya,Wzxw,Uyzw->KWCU', X, t1_xaae, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma[caea__abab] += ascontiguousarray(sigma_KWCU_abab).reshape(-1)

    sigma_KWCU_baab =- 1/2 * einsum('KiaC,iWxa,Ux->KWCU', X, v_xaae, rdm_ca, optimize = einsum_type)
    sigma_KWCU_baab += 1/2 * einsum('KiaC,ixUa,Wx->KWCU', X, v_xaae, rdm_ca, optimize = einsum_type)
    sigma_KWCU_baab += 1/2 * einsum('KiaC,a,iUxa,Wx->KWCU', X, e_extern, t1_xaae, rdm_ca, optimize = einsum_type)
    sigma_KWCU_baab -= 1/2 * einsum('KiaC,a,ixWa,Ux->KWCU', X, e_extern, t1_xaae, rdm_ca, optimize = einsum_type)
    sigma_KWCU_baab -= 1/2 * einsum('KiaC,i,iUxa,Wx->KWCU', X, e_cvs, t1_xaae, rdm_ca, optimize = einsum_type)
    sigma_KWCU_baab += 1/2 * einsum('KiaC,i,ixWa,Ux->KWCU', X, e_cvs, t1_xaae, rdm_ca, optimize = einsum_type)
    sigma_KWCU_baab -= 1/2 * einsum('KiaC,Ux,ixya,Wy->KWCU', X, h_aa, t1_xaae, rdm_ca, optimize = einsum_type)
    sigma_KWCU_baab -= 1/2 * einsum('KiaC,Wx,iyxa,Uy->KWCU', X, h_aa, t1_xaae, rdm_ca, optimize = einsum_type)
    sigma_KWCU_baab += 1/2 * einsum('KiaC,xy,iUxa,Wy->KWCU', X, h_aa, t1_xaae, rdm_ca, optimize = einsum_type)
    sigma_KWCU_baab += 1/2 * einsum('KiaC,xy,ixWa,Uy->KWCU', X, h_aa, t1_xaae, rdm_ca, optimize = einsum_type)
    sigma_KWCU_baab += 1/2 * einsum('KiaC,iUxa,xyzw,Wzyw->KWCU', X, t1_xaae, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_baab += 1/2 * einsum('KiaC,ixWa,xyzw,Uzyw->KWCU', X, t1_xaae, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_baab -= 1/2 * einsum('KiaC,ixya,Uxzw,Wzyw->KWCU', X, t1_xaae, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_baab += 1/3 * einsum('KiaC,ixya,Uzwx,Wwyz->KWCU', X, t1_xaae, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_baab += 1/6 * einsum('KiaC,ixya,Uzwx,Wwzy->KWCU', X, t1_xaae, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_baab -= 1/3 * einsum('KiaC,ixya,Uzyw,Wxwz->KWCU', X, t1_xaae, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_baab -= 1/6 * einsum('KiaC,ixya,Uzyw,Wxzw->KWCU', X, t1_xaae, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_baab -= 1/2 * einsum('KiaC,ixya,Wyzw,Uzxw->KWCU', X, t1_xaae, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_baab += 1/3 * einsum('KiaC,ixya,Wzwy,Uwxz->KWCU', X, t1_xaae, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_baab += 1/6 * einsum('KiaC,ixya,Wzwy,Uwzx->KWCU', X, t1_xaae, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_baab -= 1/3 * einsum('KiaC,ixya,Wzxw,Uywz->KWCU', X, t1_xaae, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_baab -= 1/6 * einsum('KiaC,ixya,Wzxw,Uyzw->KWCU', X, t1_xaae, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma[caea__baab] += ascontiguousarray(sigma_KWCU_baab).reshape(-1)

    mr_adc.log.timer_debug("computing sigma H1 h1-h1 CAEA-CCEE", *cput1)

def compute_sigma_vector__H1__h1_h1__CAEA_CCEE__V_XEEE(mr_adc, X, sigma, v_xeee):
    cput1 = (logger.process_clock(), logger.perf_counter())

    # Einsum definition from kernel
    einsum = mr_adc.interface.einsum
    einsum_type = mr_adc.interface.einsum_type

    # Reduced Density Matrices
    rdm_ca = mr_adc.rdm.ca

    temp =- 1/2 * einsum('Kiab,iaCb,UW->KWCU', X, v_xeee, rdm_ca, optimize = einsum_type)
    temp += einsum('Kiab,ibCa,UW->KWCU', X, v_xeee, rdm_ca, optimize = einsum_type)
    # temp = sigma_KWCU_aaaa = sigma_KWCU_abab

    return temp
    mr_adc.log.timer_debug("contracting v2e.xeee", *cput1)

# CVAA <- CCEE: NO CONTRIBUTION

# CVEA <- CCEE
def compute_sigma_vector__H1__h1_h1__CVEA_CCEE(mr_adc, X, sigma):
    cput1 = (logger.process_clock(), logger.perf_counter())

    # Einsum definition from kernel
    einsum = mr_adc.interface.einsum
    einsum_type = mr_adc.interface.einsum_type

    #Excitation Manifold
    cvea__abab = mr_adc.h1.cvea__abab
    cvea__baab = mr_adc.h1.cvea__baab
    
    ## Two-electron integrals
    v_xvae = mr_adc.v2e.xvae
    v_vaex = mr_adc.v2e.vaex
     
    # Reduced Density Matrices
    rdm_ca = mr_adc.rdm.ca

    sigma_KLCW_abab  = 2 * einsum('KiCa,LWai->KLCW', X, v_vaex, optimize = einsum_type)
    sigma_KLCW_abab -= einsum('KiCa,iLWa->KLCW', X, v_xvae, optimize = einsum_type)
    sigma_KLCW_abab -= einsum('KiaC,LWai->KLCW', X, v_vaex, optimize = einsum_type)
    sigma_KLCW_abab -= einsum('KiCa,Lxai,Wx->KLCW', X, v_vaex, rdm_ca, optimize = einsum_type)
    sigma_KLCW_abab += 1/2 * einsum('KiCa,iLxa,Wx->KLCW', X, v_xvae, rdm_ca, optimize = einsum_type)
    sigma_KLCW_abab += 1/2 * einsum('KiaC,Lxai,Wx->KLCW', X, v_vaex, rdm_ca, optimize = einsum_type)
    sigma[cvea__abab] += ascontiguousarray(sigma_KLCW_abab).reshape(-1)

    sigma_KLCW_baab  = einsum('KiaC,iLWa->KLCW', X, v_xvae, optimize = einsum_type)
    sigma_KLCW_baab -= 1/2 * einsum('KiaC,iLxa,Wx->KLCW', X, v_xvae, rdm_ca, optimize = einsum_type)
    sigma[cvea__baab] += ascontiguousarray(sigma_KLCW_baab).reshape(-1)

    mr_adc.log.timer_debug("computing sigma H1 h1-h1 CVEA-CCEE", *cput1)

# CVEE <- CCEE
def compute_sigma_vector__H1__h1_h1__CVEE_CCEE(mr_adc, X, sigma):
    cput1 = (logger.process_clock(), logger.perf_counter())

    # Einsum definition from kernel
    einsum = mr_adc.interface.einsum
    einsum_type = mr_adc.interface.einsum_type

    #Excitation Manifold
    cvee = mr_adc.h1.cvee

    ## Two-electron integrals
    v_xxvx = mr_adc.v2e.xxvx
    v_xvee = mr_adc.v2e.xvee
    v_veex = mr_adc.v2e.veex

    sigma_KLCD  = 2 * einsum('KiCa,LDai->KLCD', X, v_veex, optimize = einsum_type)
    sigma_KLCD -= einsum('KiCa,iLDa->KLCD', X, v_xvee, optimize = einsum_type)
    sigma_KLCD -= einsum('KiaC,LDai->KLCD', X, v_veex, optimize = einsum_type)
    sigma_KLCD -= einsum('KiaD,iLCa->KLCD', X, v_xvee, optimize = einsum_type)
    sigma_KLCD += einsum('ijCD,KiLj->KLCD', X, v_xxvx, optimize = einsum_type)
    sigma[cvee] += ascontiguousarray(sigma_KLCD).reshape(-1)

    mr_adc.log.timer_debug("computing sigma H1 h1-h1 CVEE-CCEE", *cput1)

