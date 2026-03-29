## h1 <- h1 coupling contributions
from . import logger, tools, ascontiguousarray, zeros
from prism.mr_adc_integrals import get_eeee_df, unpack_v2e_eeee

# CCAA <- CAEE: NO CONTRIBUTION

# CCEA <- CAEE
def compute_sigma_vector__H1__h1_h1__CCEA_CAEE(mr_adc, X, sigma):
    cput1 = (logger.process_clock(), logger.perf_counter())

    # Einsum definition from kernel
    einsum = mr_adc.interface.einsum
    einsum_type = mr_adc.interface.einsum_type

    #Excitation Manifold
    ccea = mr_adc.h1.ccea

    ## Two-electron integrals
    v_xaea = mr_adc.v2e.xaea

    # Reduced Density Matrices
    rdm_ca = mr_adc.rdm.ca
    rdm_ccaa = mr_adc.rdm.ccaa

    sigma_KLCW  = einsum('KxCa,LWay,xy->KLCW', X, v_xaea, rdm_ca, optimize = einsum_type)
    sigma_KLCW -= 1/2 * einsum('KxCa,LyaW,xy->KLCW', X, v_xaea, rdm_ca, optimize = einsum_type)
    sigma_KLCW -= 1/2 * einsum('KxCa,Lyaz,Wxyz->KLCW', X, v_xaea, rdm_ccaa, optimize = einsum_type)
    sigma_KLCW -= 1/2 * einsum('KxaC,LWay,xy->KLCW', X, v_xaea, rdm_ca, optimize = einsum_type)
    sigma_KLCW += 1/3 * einsum('KxaC,Lyaz,Wxyz->KLCW', X, v_xaea, rdm_ccaa, optimize = einsum_type)
    sigma_KLCW += 1/6 * einsum('KxaC,Lyaz,Wxzy->KLCW', X, v_xaea, rdm_ccaa, optimize = einsum_type)
    sigma_KLCW -= 1/2 * einsum('LxaC,KyaW,xy->KLCW', X, v_xaea, rdm_ca, optimize = einsum_type)
    sigma_KLCW += 1/6 * einsum('LxaC,Kyaz,Wxyz->KLCW', X, v_xaea, rdm_ccaa, optimize = einsum_type)
    sigma_KLCW += 1/3 * einsum('LxaC,Kyaz,Wxzy->KLCW', X, v_xaea, rdm_ccaa, optimize = einsum_type)
    sigma[ccea] += ascontiguousarray(sigma_KLCW).reshape(-1)

    mr_adc.log.timer_debug("computing sigma H1 h1-h1 CCEA-CAEE", *cput1)
 
# CCEE <- CAEE
def compute_sigma_vector__H1__h1_h1__CCEE_CAEE(mr_adc, X, sigma):
    cput1 = (logger.process_clock(), logger.perf_counter())

    # Einsum definition from kernel
    einsum = mr_adc.interface.einsum
    einsum_type = mr_adc.interface.einsum_type

    #Excitation Manifold
    ccee = mr_adc.h1.ccee

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

    sigma_KLCD =- einsum('KxCD,Lx->KLCD', X, h_xa, optimize = einsum_type)
    sigma_KLCD -= einsum('LxDC,Kx->KLCD', X, h_xa, optimize = einsum_type)
    sigma_KLCD += einsum('KxCD,L,Lx->KLCD', X, e_cvs, t1_xa, optimize = einsum_type)
    sigma_KLCD += einsum('LxDC,K,Kx->KLCD', X, e_cvs, t1_xa, optimize = einsum_type)
    sigma_KLCD -= einsum('KxCD,xy,Ly->KLCD', X, h_aa, t1_xa, optimize = einsum_type)
    sigma_KLCD -= einsum('LxDC,xy,Ky->KLCD', X, h_aa, t1_xa, optimize = einsum_type)
    sigma_KLCD -= einsum('KxCD,Lxyz,yz->KLCD', X, v_xaaa, rdm_ca, optimize = einsum_type)
    sigma_KLCD += 1/2 * einsum('KxCD,Lyzx,zy->KLCD', X, v_xaaa, rdm_ca, optimize = einsum_type)
    sigma_KLCD += einsum('KxCa,LDay,xy->KLCD', X, v_xeea, rdm_ca, optimize = einsum_type)
    sigma_KLCD -= 1/2 * einsum('KxCa,LyaD,xy->KLCD', X, v_xaee, rdm_ca, optimize = einsum_type)
    sigma_KLCD -= 1/2 * einsum('KxaC,LDay,xy->KLCD', X, v_xeea, rdm_ca, optimize = einsum_type)
    sigma_KLCD -= 1/2 * einsum('KxaD,LyaC,xy->KLCD', X, v_xaee, rdm_ca, optimize = einsum_type)
    sigma_KLCD -= einsum('LxDC,Kxyz,yz->KLCD', X, v_xaaa, rdm_ca, optimize = einsum_type)
    sigma_KLCD += 1/2 * einsum('LxDC,Kyzx,zy->KLCD', X, v_xaaa, rdm_ca, optimize = einsum_type)
    sigma_KLCD += einsum('LxDa,KCay,xy->KLCD', X, v_xeea, rdm_ca, optimize = einsum_type)
    sigma_KLCD -= 1/2 * einsum('LxDa,KyaC,xy->KLCD', X, v_xaee, rdm_ca, optimize = einsum_type)
    sigma_KLCD -= 1/2 * einsum('LxaC,KyaD,xy->KLCD', X, v_xaee, rdm_ca, optimize = einsum_type)
    sigma_KLCD -= 1/2 * einsum('LxaD,KCay,xy->KLCD', X, v_xeea, rdm_ca, optimize = einsum_type)
    sigma_KLCD += 1/2 * einsum('ixCD,KiLy,xy->KLCD', X, v_xxxa, rdm_ca, optimize = einsum_type)
    sigma_KLCD += 1/2 * einsum('ixDC,LiKy,xy->KLCD', X, v_xxxa, rdm_ca, optimize = einsum_type)
    sigma_KLCD += einsum('KxCD,L,Lyxz,yz->KLCD', X, e_cvs, t1_xaaa, rdm_ca, optimize = einsum_type)
    sigma_KLCD -= 1/2 * einsum('KxCD,L,Lyzx,yz->KLCD', X, e_cvs, t1_xaaa, rdm_ca, optimize = einsum_type)
    sigma_KLCD += einsum('LxDC,K,Kyxz,yz->KLCD', X, e_cvs, t1_xaaa, rdm_ca, optimize = einsum_type)
    sigma_KLCD -= 1/2 * einsum('LxDC,K,Kyzx,yz->KLCD', X, e_cvs, t1_xaaa, rdm_ca, optimize = einsum_type)
    sigma_KLCD += 1/2 * einsum('KxCD,xy,Lzwy,zw->KLCD', X, h_aa, t1_xaaa, rdm_ca, optimize = einsum_type)
    sigma_KLCD -= einsum('KxCD,xy,Lzyw,zw->KLCD', X, h_aa, t1_xaaa, rdm_ca, optimize = einsum_type)
    sigma_KLCD -= einsum('KxCD,yz,Lwxy,zw->KLCD', X, h_aa, t1_xaaa, rdm_ca, optimize = einsum_type)
    sigma_KLCD += 1/2 * einsum('KxCD,yz,Lwyx,zw->KLCD', X, h_aa, t1_xaaa, rdm_ca, optimize = einsum_type)
    sigma_KLCD -= 1/2 * einsum('KxCD,yz,Lywx,zw->KLCD', X, h_aa, t1_xaaa, rdm_ca, optimize = einsum_type)
    sigma_KLCD += einsum('KxCD,yz,Lyxw,zw->KLCD', X, h_aa, t1_xaaa, rdm_ca, optimize = einsum_type)
    sigma_KLCD += 1/2 * einsum('LxDC,xy,Kzwy,zw->KLCD', X, h_aa, t1_xaaa, rdm_ca, optimize = einsum_type)
    sigma_KLCD -= einsum('LxDC,xy,Kzyw,zw->KLCD', X, h_aa, t1_xaaa, rdm_ca, optimize = einsum_type)
    sigma_KLCD -= einsum('LxDC,yz,Kwxy,zw->KLCD', X, h_aa, t1_xaaa, rdm_ca, optimize = einsum_type)
    sigma_KLCD += 1/2 * einsum('LxDC,yz,Kwyx,zw->KLCD', X, h_aa, t1_xaaa, rdm_ca, optimize = einsum_type)
    sigma_KLCD -= 1/2 * einsum('LxDC,yz,Kywx,zw->KLCD', X, h_aa, t1_xaaa, rdm_ca, optimize = einsum_type)
    sigma_KLCD += einsum('LxDC,yz,Kyxw,zw->KLCD', X, h_aa, t1_xaaa, rdm_ca, optimize = einsum_type)
    sigma_KLCD -= einsum('KxCD,Ly,xyzw,zw->KLCD', X, t1_xa, v_aaaa, rdm_ca, optimize = einsum_type)
    sigma_KLCD += 1/2 * einsum('KxCD,Ly,xzwy,wz->KLCD', X, t1_xa, v_aaaa, rdm_ca, optimize = einsum_type)
    sigma_KLCD += einsum('KxCD,Lyxz,ywuv,zuwv->KLCD', X, t1_xaaa, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KLCD -= einsum('KxCD,Lyxz,zwuv,yuwv->KLCD', X, t1_xaaa, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KLCD += 1/2 * einsum('KxCD,Lyzw,xuvw,yuvz->KLCD', X, t1_xaaa, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KLCD += 1/2 * einsum('KxCD,Lyzw,xuvz,yuwv->KLCD', X, t1_xaaa, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KLCD -= 1/2 * einsum('KxCD,Lyzw,xuyv,zwuv->KLCD', X, t1_xaaa, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KLCD += 1/2 * einsum('KxCD,Lyzw,xwuv,yvzu->KLCD', X, t1_xaaa, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KLCD += 1/2 * einsum('KxCD,Lyzw,xwuz,yu->KLCD', X, t1_xaaa, v_aaaa, rdm_ca, optimize = einsum_type)
    sigma_KLCD -= einsum('KxCD,Lyzw,xzuv,yvwu->KLCD', X, t1_xaaa, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KLCD -= einsum('KxCD,Lyzw,xzuw,yu->KLCD', X, t1_xaaa, v_aaaa, rdm_ca, optimize = einsum_type)
    sigma_KLCD -= 1/2 * einsum('KxCD,Lyzx,ywuv,zuwv->KLCD', X, t1_xaaa, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KLCD += 1/2 * einsum('KxCD,Lyzx,zwuv,yuwv->KLCD', X, t1_xaaa, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KLCD -= einsum('LxDC,Ky,xyzw,zw->KLCD', X, t1_xa, v_aaaa, rdm_ca, optimize = einsum_type)
    sigma_KLCD += 1/2 * einsum('LxDC,Ky,xzwy,wz->KLCD', X, t1_xa, v_aaaa, rdm_ca, optimize = einsum_type)
    sigma_KLCD += einsum('LxDC,Kyxz,ywuv,zuwv->KLCD', X, t1_xaaa, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KLCD -= einsum('LxDC,Kyxz,zwuv,yuwv->KLCD', X, t1_xaaa, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KLCD += 1/2 * einsum('LxDC,Kyzw,xuvw,yuvz->KLCD', X, t1_xaaa, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KLCD += 1/2 * einsum('LxDC,Kyzw,xuvz,yuwv->KLCD', X, t1_xaaa, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KLCD -= 1/2 * einsum('LxDC,Kyzw,xuyv,zwuv->KLCD', X, t1_xaaa, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KLCD += 1/2 * einsum('LxDC,Kyzw,xwuv,yvzu->KLCD', X, t1_xaaa, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KLCD += 1/2 * einsum('LxDC,Kyzw,xwuz,yu->KLCD', X, t1_xaaa, v_aaaa, rdm_ca, optimize = einsum_type)
    sigma_KLCD -= einsum('LxDC,Kyzw,xzuv,yvwu->KLCD', X, t1_xaaa, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KLCD -= einsum('LxDC,Kyzw,xzuw,yu->KLCD', X, t1_xaaa, v_aaaa, rdm_ca, optimize = einsum_type)
    sigma_KLCD -= 1/2 * einsum('LxDC,Kyzx,ywuv,zuwv->KLCD', X, t1_xaaa, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KLCD += 1/2 * einsum('LxDC,Kyzx,zwuv,yuwv->KLCD', X, t1_xaaa, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma[ccee] += ascontiguousarray(sigma_KLCD).reshape(-1)

    mr_adc.log.timer_debug("computing sigma H1 h1-h1 CCEE-CAEE", *cput1)

# CAEE <- CAEE
def compute_sigma_vector__H1__h1_h1__CAEE_CAEE(mr_adc, X, sigma):
    cput1 = (logger.process_clock(), logger.perf_counter())

    # Einsum definition from kernel
    einsum = mr_adc.interface.einsum
    einsum_type = mr_adc.interface.einsum_type
    dot = mr_adc.interface.dot

    # Variables from kernel
    ncvs    = mr_adc.ncvs
    ncas    = mr_adc.ncas
    nextern = mr_adc.nextern

    #Excitation Manifold
    caee = mr_adc.h1.caee

    ## Two-electron integrals
    #v_xxee = mr_adc.v2e.xxee
    v_xxaa = mr_adc.v2e.xxaa
    v_xaax = mr_adc.v2e.xaax
    #v_xeex = mr_adc.v2e.xeex
    #v_aaee = mr_adc.v2e.aaee
    #v_aeea = mr_adc.v2e.aeea
    #v_eeee = mr_adc.v2e.eeee

    # Reduced Density Matrices
    rdm_ca = mr_adc.rdm.ca
    rdm_ccaa = mr_adc.rdm.ccaa

    #sigma_KWCD  = 1/2 * einsum('KxCa,yDaz,Wzyx->KWCD', X, v_aeea, rdm_ccaa, optimize = einsum_type)
    #sigma_KWCD += 1/2 * einsum('KxCa,yzDa,Wyxz->KWCD', X, v_aaee, rdm_ccaa, optimize = einsum_type)
    #sigma_KWCD -= 1/6 * einsum('KxaC,yDaz,Wzxy->KWCD', X, v_aeea, rdm_ccaa, optimize = einsum_type)
    #sigma_KWCD -= 1/3 * einsum('KxaC,yDaz,Wzyx->KWCD', X, v_aeea, rdm_ccaa, optimize = einsum_type)
    #sigma_KWCD -= 1/3 * einsum('KxaD,yCaz,Wzxy->KWCD', X, v_aeea, rdm_ccaa, optimize = einsum_type)
    #sigma_KWCD -= 1/6 * einsum('KxaD,yCaz,Wzyx->KWCD', X, v_aeea, rdm_ccaa, optimize = einsum_type)
    #sigma_KWCD += 1/2 * einsum('KxaD,yzCa,Wyxz->KWCD', X, v_aaee, rdm_ccaa, optimize = einsum_type)
    ##sigma_KWCD += 1/2 * einsum('Kxab,CaDb,Wx->KWCD', X, v_eeee, rdm_ca, optimize = einsum_type)
    #sigma_KWCD -= 1/2 * einsum('ixCD,Kiyz,Wzxy->KWCD', X, v_xxaa, rdm_ccaa, optimize = einsum_type)
    #sigma_KWCD += 1/3 * einsum('ixCD,Kyzi,Wyxz->KWCD', X, v_xaax, rdm_ccaa, optimize = einsum_type)
    #sigma_KWCD += 1/6 * einsum('ixCD,Kyzi,Wyzx->KWCD', X, v_xaax, rdm_ccaa, optimize = einsum_type)
    #sigma_KWCD -= 1/2 * einsum('ixCa,iKDa,Wx->KWCD', X, v_xxee, rdm_ca, optimize = einsum_type)
    #sigma_KWCD += 1/6 * einsum('ixDC,Kyzi,Wyxz->KWCD', X, v_xaax, rdm_ccaa, optimize = einsum_type)
    #sigma_KWCD += 1/3 * einsum('ixDC,Kyzi,Wyzx->KWCD', X, v_xaax, rdm_ccaa, optimize = einsum_type)
    #sigma_KWCD -= 1/2 * einsum('ixDa,KCai,Wx->KWCD', X, v_xeex, rdm_ca, optimize = einsum_type)
    #sigma_KWCD += einsum('ixaD,KCai,Wx->KWCD', X, v_xeex, rdm_ca, optimize = einsum_type)
    #sigma_KWCD -= 1/2 * einsum('ixaD,iKCa,Wx->KWCD', X, v_xxee, rdm_ca, optimize = einsum_type)
    #sigma_KWCD += 1/4 * einsum('KxCa,yDaz,zy,Wx->KWCD', X, v_aeea, rdm_ca, rdm_ca, optimize = einsum_type)
    #sigma_KWCD -= 1/2 * einsum('KxCa,yzDa,yz,Wx->KWCD', X, v_aaee, rdm_ca, rdm_ca, optimize = einsum_type)
    #sigma_KWCD += 1/4 * einsum('KxaD,yCaz,zy,Wx->KWCD', X, v_aeea, rdm_ca, rdm_ca, optimize = einsum_type)
    #sigma_KWCD -= 1/2 * einsum('KxaD,yzCa,yz,Wx->KWCD', X, v_aaee, rdm_ca, rdm_ca, optimize = einsum_type)
    #sigma_KWCD += 1/2 * einsum('ixCD,Kiyz,yz,Wx->KWCD', X, v_xxaa, rdm_ca, rdm_ca, optimize = einsum_type)
    #sigma_KWCD -= 1/4 * einsum('ixCD,Kyzi,zy,Wx->KWCD', X, v_xaax, rdm_ca, rdm_ca, optimize = einsum_type)

    sigma_KWCD =- 1/2 * einsum('ixCD,Kiyz,Wzxy->KWCD', X, v_xxaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWCD += 1/3 * einsum('ixCD,Kyzi,Wyxz->KWCD', X, v_xaax, rdm_ccaa, optimize = einsum_type)
    sigma_KWCD += 1/6 * einsum('ixCD,Kyzi,Wyzx->KWCD', X, v_xaax, rdm_ccaa, optimize = einsum_type)
    sigma_KWCD += 1/6 * einsum('ixDC,Kyzi,Wyxz->KWCD', X, v_xaax, rdm_ccaa, optimize = einsum_type)
    sigma_KWCD += 1/3 * einsum('ixDC,Kyzi,Wyzx->KWCD', X, v_xaax, rdm_ccaa, optimize = einsum_type)
    sigma_KWCD += 1/2 * einsum('ixCD,Kiyz,yz,Wx->KWCD', X, v_xxaa, rdm_ca, rdm_ca, optimize = einsum_type)
    sigma_KWCD -= 1/4 * einsum('ixCD,Kyzi,zy,Wx->KWCD', X, v_xaax, rdm_ca, rdm_ca, optimize = einsum_type)

    # v_xxee, v_xeex
    chunks = tools.calculate_chunks(mr_adc, nextern, [ncvs, ncvs, nextern], ntensors = 4)
    for i_chunk, (s_chunk, f_chunk) in enumerate(chunks):
        cput1 = (logger.process_clock(), logger.perf_counter())
        mr_adc.log.debug("v2e.xxee v2e.xeex [%i/%i], chunk [%i:%i]", i_chunk + 1, len(chunks), s_chunk, f_chunk)

        ## Two-electron integral
        v_xxee = mr_adc.v2e.xxee[:, :, :, s_chunk:f_chunk]
        v_xeex = mr_adc.v2e.xeex[:, :, s_chunk:f_chunk, :]

        ## CAEE block
        X_a = X[:, :, :, s_chunk:f_chunk]

        sigma_KWCD -= 1/2 * einsum('ixCa,iKDa,Wx->KWCD', X_a, v_xxee, rdm_ca, optimize = einsum_type)
        sigma_KWCD -= 1/2 * einsum('ixDa,KCai,Wx->KWCD', X_a, v_xeex, rdm_ca, optimize = einsum_type)

        ## CAEE block
        X_a = X[:, :, s_chunk:f_chunk, :]

        sigma_KWCD -= 1/2 * einsum('ixaD,iKCa,Wx->KWCD', X_a, v_xxee, rdm_ca, optimize = einsum_type)
        sigma_KWCD += einsum('ixaD,KCai,Wx->KWCD', X_a, v_xeex, rdm_ca, optimize = einsum_type)

        mr_adc.log.timer_debug("v2e.xxee v2e.xeex contractions", *cput1)
        del(v_xxee, v_xeex)

    # v_aaee, v_aeea
    chunks = tools.calculate_chunks(mr_adc, nextern, [ncas, ncas, nextern], ntensors = 4)
    for i_chunk, (s_chunk, f_chunk) in enumerate(chunks):
        cput1 = (logger.process_clock(), logger.perf_counter())
        mr_adc.log.debug("v2e.aaee v2e.aeea [%i/%i], chunk [%i:%i]", i_chunk + 1, len(chunks), s_chunk, f_chunk)

        ## Two-electron integral
        v_aaee = mr_adc.v2e.aaee[:, :, :, s_chunk:f_chunk]
        v_aeea = mr_adc.v2e.aeea[:, :, s_chunk:f_chunk, :]

        ## CAEE block
        X_a = X[:, :, :, s_chunk:f_chunk]

        sigma_KWCD += 1/2 * einsum('KxCa,yzDa,Wyxz->KWCD', X_a, v_aaee, rdm_ccaa, optimize = einsum_type)
        sigma_KWCD -= 1/2 * einsum('KxCa,yzDa,yz,Wx->KWCD', X_a, v_aaee, rdm_ca, rdm_ca, optimize = einsum_type)
        sigma_KWCD += 1/2 * einsum('KxCa,yDaz,Wzyx->KWCD', X_a, v_aeea, rdm_ccaa, optimize = einsum_type)
        sigma_KWCD += 1/4 * einsum('KxCa,yDaz,zy,Wx->KWCD', X_a, v_aeea, rdm_ca, rdm_ca, optimize = einsum_type)

        ## CAEE block
        X_a = X[:, :, s_chunk:f_chunk, :]

        sigma_KWCD += 1/2 * einsum('KxaD,yzCa,Wyxz->KWCD', X_a, v_aaee, rdm_ccaa, optimize = einsum_type)
        sigma_KWCD -= 1/2 * einsum('KxaD,yzCa,yz,Wx->KWCD', X_a, v_aaee, rdm_ca, rdm_ca, optimize = einsum_type)
        sigma_KWCD -= 1/6 * einsum('KxaC,yDaz,Wzxy->KWCD', X_a, v_aeea, rdm_ccaa, optimize = einsum_type)
        sigma_KWCD -= 1/3 * einsum('KxaC,yDaz,Wzyx->KWCD', X_a, v_aeea, rdm_ccaa, optimize = einsum_type)
        sigma_KWCD -= 1/3 * einsum('KxaD,yCaz,Wzxy->KWCD', X_a, v_aeea, rdm_ccaa, optimize = einsum_type)
        sigma_KWCD -= 1/6 * einsum('KxaD,yCaz,Wzyx->KWCD', X_a, v_aeea, rdm_ccaa, optimize = einsum_type)
        sigma_KWCD += 1/4 * einsum('KxaD,yCaz,zy,Wx->KWCD', X_a, v_aeea, rdm_ca, rdm_ca, optimize = einsum_type)

        mr_adc.log.timer_debug("v2e.aaee v2e.aeea contractions", *cput1)
        del(v_aaee, v_aeea)

    # Form X intermediates and pack for ladder contractions
    X = 1/2 * ascontiguousarray(einsum('Kxab,Wx->KWab', X, rdm_ca, optimize = einsum_type).reshape(-1, nextern*nextern).T)
    temp = zeros((nextern, nextern, ncvs*ncas))

    chunks = tools.calculate_chunks(mr_adc, nextern, [nextern, nextern, nextern])
    for i_chunk, (s_chunk, f_chunk) in enumerate(chunks):
        cput2 = (logger.process_clock(), logger.perf_counter())
        mr_adc.log.debug("v2e.eeee [%i/%i], chunk [%i:%i]", i_chunk + 1, len(chunks), s_chunk, f_chunk)
    
        ## Two-electron integral
        if mr_adc.interface.with_df:
            v_eeee = get_eeee_df(mr_adc, mr_adc.v2e.Lee, s_chunk, f_chunk)
        else:
            v_eeee = unpack_v2e_eeee(mr_adc, mr_adc.v2e.eeee, s_chunk, f_chunk)
        
        # Contractions using dot products
        temp[s_chunk:f_chunk] += dot(v_eeee, X).reshape(-1, nextern, ncvs*ncas)

        del(v_eeee)
        mr_adc.log.timer_debug("contracting v2e.eeee", *cput2)

    sigma_KWCD += temp.transpose(2,0,1).reshape((ncvs, ncas, nextern, nextern))
 
    sigma[caee] += ascontiguousarray(sigma_KWCD).reshape(-1)

    mr_adc.log.timer_debug("computing sigma H1 h1-h1 CAEE-CAEE", *cput1)

# CAAA <- CAEE
def compute_sigma_vector__H1__h1_h1__CAAA_CAEE(mr_adc, X, sigma):
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
    rdm_cccaaa = mr_adc.rdm.cccaaa

    sigma_KWUV_aaaa  = 1/2 * einsum('Kxab,UaVb,Wx->KWUV', X, v_aeae, rdm_ca, optimize = einsum_type)
    sigma_KWUV_aaaa += 1/2 * einsum('Kxab,Uayb,VxWy->KWUV', X, v_aeae, rdm_ccaa, optimize = einsum_type)
    sigma_KWUV_aaaa -= 1/2 * einsum('Kxab,UbVa,Wx->KWUV', X, v_aeae, rdm_ca, optimize = einsum_type)
    sigma_KWUV_aaaa -= 1/6 * einsum('Kxab,Ubya,VxWy->KWUV', X, v_aeae, rdm_ccaa, optimize = einsum_type)
    sigma_KWUV_aaaa += 1/6 * einsum('Kxab,Ubya,VxyW->KWUV', X, v_aeae, rdm_ccaa, optimize = einsum_type)
    sigma_KWUV_aaaa -= 1/2 * einsum('Kxab,Vayb,UxWy->KWUV', X, v_aeae, rdm_ccaa, optimize = einsum_type)
    sigma_KWUV_aaaa += 1/6 * einsum('Kxab,Vbya,UxWy->KWUV', X, v_aeae, rdm_ccaa, optimize = einsum_type)
    sigma_KWUV_aaaa -= 1/6 * einsum('Kxab,Vbya,UxyW->KWUV', X, v_aeae, rdm_ccaa, optimize = einsum_type)
    sigma_KWUV_aaaa += 1/2 * einsum('Kxab,a,UVab,Wx->KWUV', X, e_extern, t1_aaee, rdm_ca, optimize = einsum_type)
    sigma_KWUV_aaaa -= 1/2 * einsum('Kxab,a,UVba,Wx->KWUV', X, e_extern, t1_aaee, rdm_ca, optimize = einsum_type)
    sigma_KWUV_aaaa += 1/2 * einsum('Kxab,a,Uyab,VxWy->KWUV', X, e_extern, t1_aaee, rdm_ccaa, optimize = einsum_type)
    sigma_KWUV_aaaa -= 1/6 * einsum('Kxab,a,Uyba,VxWy->KWUV', X, e_extern, t1_aaee, rdm_ccaa, optimize = einsum_type)
    sigma_KWUV_aaaa += 1/6 * einsum('Kxab,a,Uyba,VxyW->KWUV', X, e_extern, t1_aaee, rdm_ccaa, optimize = einsum_type)
    sigma_KWUV_aaaa -= 1/2 * einsum('Kxab,a,Vyab,UxWy->KWUV', X, e_extern, t1_aaee, rdm_ccaa, optimize = einsum_type)
    sigma_KWUV_aaaa += 1/6 * einsum('Kxab,a,Vyba,UxWy->KWUV', X, e_extern, t1_aaee, rdm_ccaa, optimize = einsum_type)
    sigma_KWUV_aaaa -= 1/6 * einsum('Kxab,a,Vyba,UxyW->KWUV', X, e_extern, t1_aaee, rdm_ccaa, optimize = einsum_type)
    sigma_KWUV_aaaa += 1/2 * einsum('Kxab,b,UVab,Wx->KWUV', X, e_extern, t1_aaee, rdm_ca, optimize = einsum_type)
    sigma_KWUV_aaaa -= 1/2 * einsum('Kxab,b,UVba,Wx->KWUV', X, e_extern, t1_aaee, rdm_ca, optimize = einsum_type)
    sigma_KWUV_aaaa += 1/2 * einsum('Kxab,b,Uyab,VxWy->KWUV', X, e_extern, t1_aaee, rdm_ccaa, optimize = einsum_type)
    sigma_KWUV_aaaa -= 1/6 * einsum('Kxab,b,Uyba,VxWy->KWUV', X, e_extern, t1_aaee, rdm_ccaa, optimize = einsum_type)
    sigma_KWUV_aaaa += 1/6 * einsum('Kxab,b,Uyba,VxyW->KWUV', X, e_extern, t1_aaee, rdm_ccaa, optimize = einsum_type)
    sigma_KWUV_aaaa -= 1/2 * einsum('Kxab,b,Vyab,UxWy->KWUV', X, e_extern, t1_aaee, rdm_ccaa, optimize = einsum_type)
    sigma_KWUV_aaaa += 1/6 * einsum('Kxab,b,Vyba,UxWy->KWUV', X, e_extern, t1_aaee, rdm_ccaa, optimize = einsum_type)
    sigma_KWUV_aaaa -= 1/6 * einsum('Kxab,b,Vyba,UxyW->KWUV', X, e_extern, t1_aaee, rdm_ccaa, optimize = einsum_type)
    sigma_KWUV_aaaa += 1/2 * einsum('Kxab,Uy,Vyab,Wx->KWUV', X, h_aa, t1_aaee, rdm_ca, optimize = einsum_type)
    sigma_KWUV_aaaa -= 1/2 * einsum('Kxab,Uy,Vyba,Wx->KWUV', X, h_aa, t1_aaee, rdm_ca, optimize = einsum_type)
    sigma_KWUV_aaaa -= 1/2 * einsum('Kxab,Uy,yzab,VxWz->KWUV', X, h_aa, t1_aaee, rdm_ccaa, optimize = einsum_type)
    sigma_KWUV_aaaa += 1/6 * einsum('Kxab,Uy,zyab,VxWz->KWUV', X, h_aa, t1_aaee, rdm_ccaa, optimize = einsum_type)
    sigma_KWUV_aaaa -= 1/6 * einsum('Kxab,Uy,zyab,VxzW->KWUV', X, h_aa, t1_aaee, rdm_ccaa, optimize = einsum_type)
    sigma_KWUV_aaaa -= 1/2 * einsum('Kxab,Vy,Uyab,Wx->KWUV', X, h_aa, t1_aaee, rdm_ca, optimize = einsum_type)
    sigma_KWUV_aaaa += 1/2 * einsum('Kxab,Vy,Uyba,Wx->KWUV', X, h_aa, t1_aaee, rdm_ca, optimize = einsum_type)
    sigma_KWUV_aaaa += 1/2 * einsum('Kxab,Vy,yzab,UxWz->KWUV', X, h_aa, t1_aaee, rdm_ccaa, optimize = einsum_type)
    sigma_KWUV_aaaa -= 1/6 * einsum('Kxab,Vy,zyab,UxWz->KWUV', X, h_aa, t1_aaee, rdm_ccaa, optimize = einsum_type)
    sigma_KWUV_aaaa += 1/6 * einsum('Kxab,Vy,zyab,UxzW->KWUV', X, h_aa, t1_aaee, rdm_ccaa, optimize = einsum_type)
    sigma_KWUV_aaaa -= 1/2 * einsum('Kxab,yz,Uyab,VxWz->KWUV', X, h_aa, t1_aaee, rdm_ccaa, optimize = einsum_type)
    sigma_KWUV_aaaa += 1/6 * einsum('Kxab,yz,Uyba,VxWz->KWUV', X, h_aa, t1_aaee, rdm_ccaa, optimize = einsum_type)
    sigma_KWUV_aaaa -= 1/6 * einsum('Kxab,yz,Uyba,VxzW->KWUV', X, h_aa, t1_aaee, rdm_ccaa, optimize = einsum_type)
    sigma_KWUV_aaaa += 1/2 * einsum('Kxab,yz,Vyab,UxWz->KWUV', X, h_aa, t1_aaee, rdm_ccaa, optimize = einsum_type)
    sigma_KWUV_aaaa -= 1/6 * einsum('Kxab,yz,Vyba,UxWz->KWUV', X, h_aa, t1_aaee, rdm_ccaa, optimize = einsum_type)
    sigma_KWUV_aaaa += 1/6 * einsum('Kxab,yz,Vyba,UxzW->KWUV', X, h_aa, t1_aaee, rdm_ccaa, optimize = einsum_type)
    sigma_KWUV_aaaa -= 1/2 * einsum('Kxab,Uyab,Vyzw,Wzxw->KWUV', X, t1_aaee, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWUV_aaaa -= 1/2 * einsum('Kxab,Uyab,Vzwy,Wwzx->KWUV', X, t1_aaee, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWUV_aaaa -= 1/2 * einsum('Kxab,Uyab,yzwu,VxwWzu->KWUV', X, t1_aaee, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_KWUV_aaaa += 1/2 * einsum('Kxab,Uyba,Vyzw,Wzxw->KWUV', X, t1_aaee, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWUV_aaaa -= 1/6 * einsum('Kxab,Uyba,Vzwy,Wwxz->KWUV', X, t1_aaee, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWUV_aaaa += 1/6 * einsum('Kxab,Uyba,Vzwy,Wwzx->KWUV', X, t1_aaee, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWUV_aaaa += 1/6 * einsum('Kxab,Uyba,yzwu,VxwWzu->KWUV', X, t1_aaee, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_KWUV_aaaa -= 1/6 * einsum('Kxab,Uyba,yzwu,VxwzWu->KWUV', X, t1_aaee, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_KWUV_aaaa += 1/2 * einsum('Kxab,Vyab,Uyzw,Wzxw->KWUV', X, t1_aaee, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWUV_aaaa += 1/2 * einsum('Kxab,Vyab,Uzwy,Wwzx->KWUV', X, t1_aaee, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWUV_aaaa += 1/2 * einsum('Kxab,Vyab,yzwu,UxwWzu->KWUV', X, t1_aaee, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_KWUV_aaaa -= 1/2 * einsum('Kxab,Vyba,Uyzw,Wzxw->KWUV', X, t1_aaee, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWUV_aaaa += 1/6 * einsum('Kxab,Vyba,Uzwy,Wwxz->KWUV', X, t1_aaee, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWUV_aaaa -= 1/6 * einsum('Kxab,Vyba,Uzwy,Wwzx->KWUV', X, t1_aaee, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWUV_aaaa -= 1/6 * einsum('Kxab,Vyba,yzwu,UxwWzu->KWUV', X, t1_aaee, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_KWUV_aaaa += 1/6 * einsum('Kxab,Vyba,yzwu,UxwzWu->KWUV', X, t1_aaee, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_KWUV_aaaa += 1/2 * einsum('Kxab,yzab,UwVy,Wzwx->KWUV', X, t1_aaee, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWUV_aaaa -= 1/6 * einsum('Kxab,yzab,UwVz,Wywx->KWUV', X, t1_aaee, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWUV_aaaa += 1/6 * einsum('Kxab,yzab,UwVz,Wyxw->KWUV', X, t1_aaee, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWUV_aaaa += 1/6 * einsum('Kxab,yzab,Uwuy,VxwWzu->KWUV', X, t1_aaee, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_KWUV_aaaa -= 1/6 * einsum('Kxab,yzab,Uwuy,VxwuzW->KWUV', X, t1_aaee, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_KWUV_aaaa += 1/6 * einsum('Kxab,yzab,Uwuz,VxwWuy->KWUV', X, t1_aaee, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_KWUV_aaaa -= 1/6 * einsum('Kxab,yzab,Uwuz,VxwyuW->KWUV', X, t1_aaee, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_KWUV_aaaa -= 1/2 * einsum('Kxab,yzab,UyVw,Wzwx->KWUV', X, t1_aaee, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWUV_aaaa -= 1/2 * einsum('Kxab,yzab,UyVz,Wx->KWUV', X, t1_aaee, v_aaaa, rdm_ca, optimize = einsum_type)
    sigma_KWUV_aaaa -= 1/2 * einsum('Kxab,yzab,Uywu,VxuWzw->KWUV', X, t1_aaee, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_KWUV_aaaa -= 1/2 * einsum('Kxab,yzab,Uywz,VxWw->KWUV', X, t1_aaee, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWUV_aaaa += 1/6 * einsum('Kxab,yzab,UzVw,Wywx->KWUV', X, t1_aaee, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWUV_aaaa -= 1/6 * einsum('Kxab,yzab,UzVw,Wyxw->KWUV', X, t1_aaee, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWUV_aaaa += 1/2 * einsum('Kxab,yzab,UzVy,Wx->KWUV', X, t1_aaee, v_aaaa, rdm_ca, optimize = einsum_type)
    sigma_KWUV_aaaa += 1/6 * einsum('Kxab,yzab,Uzwu,VxuWyw->KWUV', X, t1_aaee, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_KWUV_aaaa -= 1/6 * einsum('Kxab,yzab,Uzwu,VxuyWw->KWUV', X, t1_aaee, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_KWUV_aaaa += 1/6 * einsum('Kxab,yzab,Uzwy,VxWw->KWUV', X, t1_aaee, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWUV_aaaa -= 1/6 * einsum('Kxab,yzab,Uzwy,VxwW->KWUV', X, t1_aaee, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWUV_aaaa -= 1/6 * einsum('Kxab,yzab,Vwuy,UxwWzu->KWUV', X, t1_aaee, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_KWUV_aaaa += 1/6 * einsum('Kxab,yzab,Vwuy,UxwuzW->KWUV', X, t1_aaee, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_KWUV_aaaa -= 1/6 * einsum('Kxab,yzab,Vwuz,UxwWuy->KWUV', X, t1_aaee, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_KWUV_aaaa += 1/6 * einsum('Kxab,yzab,Vwuz,UxwyuW->KWUV', X, t1_aaee, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_KWUV_aaaa += 1/2 * einsum('Kxab,yzab,Vywu,UxuWzw->KWUV', X, t1_aaee, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_KWUV_aaaa += 1/2 * einsum('Kxab,yzab,Vywz,UxWw->KWUV', X, t1_aaee, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWUV_aaaa -= 1/6 * einsum('Kxab,yzab,Vzwu,UxuWyw->KWUV', X, t1_aaee, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_KWUV_aaaa += 1/6 * einsum('Kxab,yzab,Vzwu,UxuyWw->KWUV', X, t1_aaee, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_KWUV_aaaa -= 1/6 * einsum('Kxab,yzab,Vzwy,UxWw->KWUV', X, t1_aaee, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWUV_aaaa += 1/6 * einsum('Kxab,yzab,Vzwy,UxwW->KWUV', X, t1_aaee, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWUV_aaaa -= 1/6 * einsum('Kxab,yzab,Wwyu,UVxuwz->KWUV', X, t1_aaee, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_KWUV_aaaa += 1/6 * einsum('Kxab,yzab,Wwyu,UVxwuz->KWUV', X, t1_aaee, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_KWUV_aaaa += 1/6 * einsum('Kxab,yzab,Wwzu,UVxwyu->KWUV', X, t1_aaee, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_KWUV_aaaa -= 1/6 * einsum('Kxab,yzab,Wwzu,UVxywu->KWUV', X, t1_aaee, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma[caaa__aaaa] += ascontiguousarray(sigma_KWUV_aaaa[:, :, aa_tril_ind[0], aa_tril_ind[1]]).reshape(-1)

    sigma_KWUV_abab  = 1/2 * einsum('Kxab,UaVb,Wx->KWUV', X, v_aeae, rdm_ca, optimize = einsum_type)
    sigma_KWUV_abab += 1/2 * einsum('Kxab,Uayb,VxWy->KWUV', X, v_aeae, rdm_ccaa, optimize = einsum_type)
    sigma_KWUV_abab -= 1/3 * einsum('Kxab,Ubya,VxWy->KWUV', X, v_aeae, rdm_ccaa, optimize = einsum_type)
    sigma_KWUV_abab -= 1/6 * einsum('Kxab,Ubya,VxyW->KWUV', X, v_aeae, rdm_ccaa, optimize = einsum_type)
    sigma_KWUV_abab -= 1/6 * einsum('Kxab,Vbya,UxWy->KWUV', X, v_aeae, rdm_ccaa, optimize = einsum_type)
    sigma_KWUV_abab -= 1/3 * einsum('Kxab,Vbya,UxyW->KWUV', X, v_aeae, rdm_ccaa, optimize = einsum_type)
    sigma_KWUV_abab += 1/2 * einsum('Kxab,a,UVab,Wx->KWUV', X, e_extern, t1_aaee, rdm_ca, optimize = einsum_type)
    sigma_KWUV_abab += 1/2 * einsum('Kxab,a,Uyab,VxWy->KWUV', X, e_extern, t1_aaee, rdm_ccaa, optimize = einsum_type)
    sigma_KWUV_abab -= 1/3 * einsum('Kxab,a,Uyba,VxWy->KWUV', X, e_extern, t1_aaee, rdm_ccaa, optimize = einsum_type)
    sigma_KWUV_abab -= 1/6 * einsum('Kxab,a,Uyba,VxyW->KWUV', X, e_extern, t1_aaee, rdm_ccaa, optimize = einsum_type)
    sigma_KWUV_abab -= 1/6 * einsum('Kxab,a,Vyba,UxWy->KWUV', X, e_extern, t1_aaee, rdm_ccaa, optimize = einsum_type)
    sigma_KWUV_abab -= 1/3 * einsum('Kxab,a,Vyba,UxyW->KWUV', X, e_extern, t1_aaee, rdm_ccaa, optimize = einsum_type)
    sigma_KWUV_abab += 1/2 * einsum('Kxab,b,UVab,Wx->KWUV', X, e_extern, t1_aaee, rdm_ca, optimize = einsum_type)
    sigma_KWUV_abab += 1/2 * einsum('Kxab,b,Uyab,VxWy->KWUV', X, e_extern, t1_aaee, rdm_ccaa, optimize = einsum_type)
    sigma_KWUV_abab -= 1/3 * einsum('Kxab,b,Uyba,VxWy->KWUV', X, e_extern, t1_aaee, rdm_ccaa, optimize = einsum_type)
    sigma_KWUV_abab -= 1/6 * einsum('Kxab,b,Uyba,VxyW->KWUV', X, e_extern, t1_aaee, rdm_ccaa, optimize = einsum_type)
    sigma_KWUV_abab -= 1/6 * einsum('Kxab,b,Vyba,UxWy->KWUV', X, e_extern, t1_aaee, rdm_ccaa, optimize = einsum_type)
    sigma_KWUV_abab -= 1/3 * einsum('Kxab,b,Vyba,UxyW->KWUV', X, e_extern, t1_aaee, rdm_ccaa, optimize = einsum_type)
    sigma_KWUV_abab -= 1/2 * einsum('Kxab,Uy,Vyba,Wx->KWUV', X, h_aa, t1_aaee, rdm_ca, optimize = einsum_type)
    sigma_KWUV_abab -= 1/2 * einsum('Kxab,Uy,yzab,VxWz->KWUV', X, h_aa, t1_aaee, rdm_ccaa, optimize = einsum_type)
    sigma_KWUV_abab += 1/3 * einsum('Kxab,Uy,zyab,VxWz->KWUV', X, h_aa, t1_aaee, rdm_ccaa, optimize = einsum_type)
    sigma_KWUV_abab += 1/6 * einsum('Kxab,Uy,zyab,VxzW->KWUV', X, h_aa, t1_aaee, rdm_ccaa, optimize = einsum_type)
    sigma_KWUV_abab -= 1/2 * einsum('Kxab,Vy,Uyab,Wx->KWUV', X, h_aa, t1_aaee, rdm_ca, optimize = einsum_type)
    sigma_KWUV_abab += 1/6 * einsum('Kxab,Vy,zyab,UxWz->KWUV', X, h_aa, t1_aaee, rdm_ccaa, optimize = einsum_type)
    sigma_KWUV_abab += 1/3 * einsum('Kxab,Vy,zyab,UxzW->KWUV', X, h_aa, t1_aaee, rdm_ccaa, optimize = einsum_type)
    sigma_KWUV_abab -= 1/2 * einsum('Kxab,yz,Uyab,VxWz->KWUV', X, h_aa, t1_aaee, rdm_ccaa, optimize = einsum_type)
    sigma_KWUV_abab += 1/3 * einsum('Kxab,yz,Uyba,VxWz->KWUV', X, h_aa, t1_aaee, rdm_ccaa, optimize = einsum_type)
    sigma_KWUV_abab += 1/6 * einsum('Kxab,yz,Uyba,VxzW->KWUV', X, h_aa, t1_aaee, rdm_ccaa, optimize = einsum_type)
    sigma_KWUV_abab += 1/6 * einsum('Kxab,yz,Vyba,UxWz->KWUV', X, h_aa, t1_aaee, rdm_ccaa, optimize = einsum_type)
    sigma_KWUV_abab += 1/3 * einsum('Kxab,yz,Vyba,UxzW->KWUV', X, h_aa, t1_aaee, rdm_ccaa, optimize = einsum_type)
    sigma_KWUV_abab -= 1/2 * einsum('Kxab,Uyab,Vyzw,Wzxw->KWUV', X, t1_aaee, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWUV_abab -= 1/2 * einsum('Kxab,Uyab,Vzwy,Wwzx->KWUV', X, t1_aaee, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWUV_abab -= 1/2 * einsum('Kxab,Uyab,yzwu,VxwWzu->KWUV', X, t1_aaee, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_KWUV_abab += 1/6 * einsum('Kxab,Uyba,Vzwy,Wwxz->KWUV', X, t1_aaee, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWUV_abab += 1/3 * einsum('Kxab,Uyba,Vzwy,Wwzx->KWUV', X, t1_aaee, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWUV_abab += 1/3 * einsum('Kxab,Uyba,yzwu,VxwWzu->KWUV', X, t1_aaee, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_KWUV_abab += 1/6 * einsum('Kxab,Uyba,yzwu,VxwzWu->KWUV', X, t1_aaee, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_KWUV_abab -= 1/2 * einsum('Kxab,Vyba,Uyzw,Wzxw->KWUV', X, t1_aaee, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWUV_abab += 1/3 * einsum('Kxab,Vyba,Uzwy,Wwxz->KWUV', X, t1_aaee, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWUV_abab += 1/6 * einsum('Kxab,Vyba,Uzwy,Wwzx->KWUV', X, t1_aaee, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWUV_abab -= 1/6 * einsum('Kxab,Vyba,yzwu,UxwWuz->KWUV', X, t1_aaee, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_KWUV_abab -= 1/6 * einsum('Kxab,Vyba,yzwu,UxwuWz->KWUV', X, t1_aaee, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_KWUV_abab -= 1/6 * einsum('Kxab,Vyba,yzwu,UxwuzW->KWUV', X, t1_aaee, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_KWUV_abab += 1/6 * einsum('Kxab,Vyba,yzwu,UxwzWu->KWUV', X, t1_aaee, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_KWUV_abab -= 1/6 * einsum('Kxab,Vyba,yzwu,UxwzuW->KWUV', X, t1_aaee, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_KWUV_abab += 1/6 * einsum('Kxab,yzab,UwVz,Wywx->KWUV', X, t1_aaee, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWUV_abab += 1/3 * einsum('Kxab,yzab,UwVz,Wyxw->KWUV', X, t1_aaee, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWUV_abab += 1/3 * einsum('Kxab,yzab,Uwuy,VxwWzu->KWUV', X, t1_aaee, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_KWUV_abab += 1/6 * einsum('Kxab,yzab,Uwuy,VxwuzW->KWUV', X, t1_aaee, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_KWUV_abab += 1/6 * einsum('Kxab,yzab,Uwuz,VxwWuy->KWUV', X, t1_aaee, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_KWUV_abab -= 1/6 * einsum('Kxab,yzab,Uwuz,VxwWyu->KWUV', X, t1_aaee, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_KWUV_abab -= 1/6 * einsum('Kxab,yzab,Uwuz,VxwuWy->KWUV', X, t1_aaee, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_KWUV_abab -= 1/6 * einsum('Kxab,yzab,Uwuz,VxwuyW->KWUV', X, t1_aaee, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_KWUV_abab -= 1/6 * einsum('Kxab,yzab,Uwuz,VxwyWu->KWUV', X, t1_aaee, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_KWUV_abab -= 1/2 * einsum('Kxab,yzab,UyVw,Wzwx->KWUV', X, t1_aaee, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWUV_abab -= 1/2 * einsum('Kxab,yzab,UyVz,Wx->KWUV', X, t1_aaee, v_aaaa, rdm_ca, optimize = einsum_type)
    sigma_KWUV_abab -= 1/2 * einsum('Kxab,yzab,Uywu,VxuWzw->KWUV', X, t1_aaee, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_KWUV_abab -= 1/2 * einsum('Kxab,yzab,Uywz,VxWw->KWUV', X, t1_aaee, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWUV_abab += 1/3 * einsum('Kxab,yzab,UzVw,Wywx->KWUV', X, t1_aaee, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWUV_abab += 1/6 * einsum('Kxab,yzab,UzVw,Wyxw->KWUV', X, t1_aaee, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWUV_abab += 1/3 * einsum('Kxab,yzab,Uzwu,VxuWyw->KWUV', X, t1_aaee, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_KWUV_abab += 1/6 * einsum('Kxab,yzab,Uzwu,VxuyWw->KWUV', X, t1_aaee, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_KWUV_abab += 1/3 * einsum('Kxab,yzab,Uzwy,VxWw->KWUV', X, t1_aaee, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWUV_abab += 1/6 * einsum('Kxab,yzab,Uzwy,VxwW->KWUV', X, t1_aaee, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWUV_abab -= 1/6 * einsum('Kxab,yzab,Vwuy,UxwWuz->KWUV', X, t1_aaee, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_KWUV_abab -= 1/6 * einsum('Kxab,yzab,Vwuy,UxwuWz->KWUV', X, t1_aaee, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_KWUV_abab += 1/6 * einsum('Kxab,yzab,Vwuy,UxwuzW->KWUV', X, t1_aaee, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_KWUV_abab -= 1/6 * einsum('Kxab,yzab,Vwuy,UxwzWu->KWUV', X, t1_aaee, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_KWUV_abab -= 1/6 * einsum('Kxab,yzab,Vwuy,UxwzuW->KWUV', X, t1_aaee, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_KWUV_abab += 1/6 * einsum('Kxab,yzab,Vwuz,UxwWuy->KWUV', X, t1_aaee, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_KWUV_abab += 1/3 * einsum('Kxab,yzab,Vwuz,UxwyuW->KWUV', X, t1_aaee, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_KWUV_abab -= 1/6 * einsum('Kxab,yzab,Vzwu,UxuWwy->KWUV', X, t1_aaee, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_KWUV_abab -= 1/6 * einsum('Kxab,yzab,Vzwu,UxuwWy->KWUV', X, t1_aaee, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_KWUV_abab -= 1/6 * einsum('Kxab,yzab,Vzwu,UxuwyW->KWUV', X, t1_aaee, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_KWUV_abab += 1/6 * einsum('Kxab,yzab,Vzwu,UxuyWw->KWUV', X, t1_aaee, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_KWUV_abab -= 1/6 * einsum('Kxab,yzab,Vzwu,UxuywW->KWUV', X, t1_aaee, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_KWUV_abab += 1/6 * einsum('Kxab,yzab,Vzwy,UxWw->KWUV', X, t1_aaee, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWUV_abab += 1/3 * einsum('Kxab,yzab,Vzwy,UxwW->KWUV', X, t1_aaee, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWUV_abab -= 1/6 * einsum('Kxab,yzab,Wwyu,UVxuwz->KWUV', X, t1_aaee, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_KWUV_abab += 1/6 * einsum('Kxab,yzab,Wwyu,UVxuzw->KWUV', X, t1_aaee, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_KWUV_abab += 1/6 * einsum('Kxab,yzab,Wwyu,UVxwzu->KWUV', X, t1_aaee, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_KWUV_abab += 1/6 * einsum('Kxab,yzab,Wwyu,UVxzuw->KWUV', X, t1_aaee, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_KWUV_abab += 1/6 * einsum('Kxab,yzab,Wwyu,UVxzwu->KWUV', X, t1_aaee, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_KWUV_abab -= 1/6 * einsum('Kxab,yzab,Wwzu,UVxwyu->KWUV', X, t1_aaee, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_KWUV_abab -= 1/3 * einsum('Kxab,yzab,Wwzu,UVxywu->KWUV', X, t1_aaee, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma[caaa__abab] += ascontiguousarray(sigma_KWUV_abab).reshape(-1)

    mr_adc.log.timer_debug("computing sigma H1 h1-h1 CAAA-CAEE", *cput1)

# CAEA <- CAEE
def compute_sigma_vector__H1__h1_h1__CAEA_CAEE(mr_adc, X, sigma):
    cput1 = (logger.process_clock(), logger.perf_counter())

    # Einsum definition from kernel
    einsum = mr_adc.interface.einsum
    einsum_type = mr_adc.interface.einsum_type

    #Excitation Manifold
    caea__aaaa = mr_adc.h1.caea__aaaa
    caea__abab = mr_adc.h1.caea__abab
    caea__baab = mr_adc.h1.caea__baab

    ## Molecular Orbitals Energies
    e_extern = mr_adc.mo_energy.e
 
    ## One-electron integrals
    h_aa = mr_adc.h1eff.aa
    h_ae = mr_adc.h1eff.ae
 
    ## Two-electron integrals
    v_xxae = mr_adc.v2e.xxae
    v_xaex = mr_adc.v2e.xaex
    v_aaaa = mr_adc.v2e.aaaa
    v_aaae = mr_adc.v2e.aaae
    #v_aeee = mr_adc.v2e.aeee
  
    ## Amplitudes
    t1_ae = mr_adc.t1.ae
    t1_aaae = mr_adc.t1.aaae
    
    # Reduced Density Matrices
    rdm_ca = mr_adc.rdm.ca
    rdm_ccaa = mr_adc.rdm.ccaa
    rdm_cccaaa = mr_adc.rdm.cccaaa

    sigma_KWCU_aaaa  = 1/2 * einsum('KxCa,Ua,Wx->KWCU', X, h_ae, rdm_ca, optimize = einsum_type)
    sigma_KWCU_aaaa -= 1/2 * einsum('KxaC,Ua,Wx->KWCU', X, h_ae, rdm_ca, optimize = einsum_type)
    sigma_KWCU_aaaa += 1/2 * einsum('KxCa,Uyza,Wzyx->KWCU', X, v_aaae, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_aaaa -= 1/2 * einsum('KxCa,yWza,Uxyz->KWCU', X, v_aaae, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_aaaa += 1/2 * einsum('KxCa,yzUa,Wyxz->KWCU', X, v_aaae, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_aaaa += 1/6 * einsum('KxaC,Uyza,Wzxy->KWCU', X, v_aaae, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_aaaa -= 1/6 * einsum('KxaC,Uyza,Wzyx->KWCU', X, v_aaae, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_aaaa += 1/6 * einsum('KxaC,yWza,Uxyz->KWCU', X, v_aaae, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_aaaa -= 1/6 * einsum('KxaC,yWza,Uxzy->KWCU', X, v_aaae, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_aaaa -= 1/2 * einsum('KxaC,yzUa,Wyxz->KWCU', X, v_aaae, rdm_ccaa, optimize = einsum_type)
    #sigma_KWCU_aaaa -= 1/2 * einsum('Kxab,UaCb,Wx->KWCU', X, v_aeee, rdm_ca, optimize = einsum_type)
    #sigma_KWCU_aaaa += 1/2 * einsum('Kxab,UbCa,Wx->KWCU', X, v_aeee, rdm_ca, optimize = einsum_type)
    #sigma_KWCU_aaaa -= 1/6 * einsum('Kxab,yaCb,UxWy->KWCU', X, v_aeee, rdm_ccaa, optimize = einsum_type)
    #sigma_KWCU_aaaa += 1/6 * einsum('Kxab,yaCb,UxyW->KWCU', X, v_aeee, rdm_ccaa, optimize = einsum_type)
    #sigma_KWCU_aaaa += 1/2 * einsum('Kxab,ybCa,UxWy->KWCU', X, v_aeee, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_aaaa += 1/2 * einsum('ixCa,KUai,Wx->KWCU', X, v_xaex, rdm_ca, optimize = einsum_type)
    sigma_KWCU_aaaa += 1/6 * einsum('ixCa,Kyai,UxWy->KWCU', X, v_xaex, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_aaaa -= 1/6 * einsum('ixCa,Kyai,UxyW->KWCU', X, v_xaex, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_aaaa -= 1/2 * einsum('ixCa,iKUa,Wx->KWCU', X, v_xxae, rdm_ca, optimize = einsum_type)
    sigma_KWCU_aaaa -= 1/2 * einsum('ixCa,iKya,UxWy->KWCU', X, v_xxae, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_aaaa -= einsum('ixaC,KUai,Wx->KWCU', X, v_xaex, rdm_ca, optimize = einsum_type)
    sigma_KWCU_aaaa -= 1/3 * einsum('ixaC,Kyai,UxWy->KWCU', X, v_xaex, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_aaaa += 1/3 * einsum('ixaC,Kyai,UxyW->KWCU', X, v_xaex, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_aaaa += 1/2 * einsum('ixaC,iKUa,Wx->KWCU', X, v_xxae, rdm_ca, optimize = einsum_type)
    sigma_KWCU_aaaa += 1/6 * einsum('ixaC,iKya,UxWy->KWCU', X, v_xxae, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_aaaa -= 1/6 * einsum('ixaC,iKya,UxyW->KWCU', X, v_xxae, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_aaaa += 1/2 * einsum('KxCa,a,Ua,Wx->KWCU', X, e_extern, t1_ae, rdm_ca, optimize = einsum_type)
    sigma_KWCU_aaaa += 1/2 * einsum('KxCa,a,Uyza,Wyzx->KWCU', X, e_extern, t1_aaae, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_aaaa += 1/2 * einsum('KxCa,a,yUza,Wyxz->KWCU', X, e_extern, t1_aaae, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_aaaa -= 1/2 * einsum('KxCa,a,yzWa,Uxyz->KWCU', X, e_extern, t1_aaae, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_aaaa -= 1/2 * einsum('KxaC,a,Ua,Wx->KWCU', X, e_extern, t1_ae, rdm_ca, optimize = einsum_type)
    sigma_KWCU_aaaa += 1/6 * einsum('KxaC,a,Uyza,Wyxz->KWCU', X, e_extern, t1_aaae, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_aaaa -= 1/6 * einsum('KxaC,a,Uyza,Wyzx->KWCU', X, e_extern, t1_aaae, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_aaaa -= 1/2 * einsum('KxaC,a,yUza,Wyxz->KWCU', X, e_extern, t1_aaae, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_aaaa += 1/6 * einsum('KxaC,a,yzWa,Uxyz->KWCU', X, e_extern, t1_aaae, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_aaaa -= 1/6 * einsum('KxaC,a,yzWa,Uxzy->KWCU', X, e_extern, t1_aaae, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_aaaa -= 1/2 * einsum('KxCa,Uy,ya,Wx->KWCU', X, h_aa, t1_ae, rdm_ca, optimize = einsum_type)
    sigma_KWCU_aaaa -= 1/2 * einsum('KxCa,Uy,yzwa,Wzwx->KWCU', X, h_aa, t1_aaae, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_aaaa -= 1/2 * einsum('KxCa,Uy,zywa,Wzxw->KWCU', X, h_aa, t1_aaae, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_aaaa -= 1/2 * einsum('KxCa,Wy,zwya,Uxzw->KWCU', X, h_aa, t1_aaae, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_aaaa += 1/2 * einsum('KxCa,yz,Uwya,Wwzx->KWCU', X, h_aa, t1_aaae, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_aaaa -= 1/2 * einsum('KxCa,yz,Uywa,Wzwx->KWCU', X, h_aa, t1_aaae, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_aaaa += 1/2 * einsum('KxCa,yz,wUya,Wwxz->KWCU', X, h_aa, t1_aaae, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_aaaa += 1/2 * einsum('KxCa,yz,wyWa,Uxwz->KWCU', X, h_aa, t1_aaae, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_aaaa -= 1/2 * einsum('KxCa,yz,yUwa,Wzxw->KWCU', X, h_aa, t1_aaae, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_aaaa += 1/2 * einsum('KxCa,yz,ywWa,Uxzw->KWCU', X, h_aa, t1_aaae, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_aaaa += 1/2 * einsum('KxaC,Uy,ya,Wx->KWCU', X, h_aa, t1_ae, rdm_ca, optimize = einsum_type)
    sigma_KWCU_aaaa += 1/6 * einsum('KxaC,Uy,yzwa,Wzwx->KWCU', X, h_aa, t1_aaae, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_aaaa -= 1/6 * einsum('KxaC,Uy,yzwa,Wzxw->KWCU', X, h_aa, t1_aaae, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_aaaa += 1/2 * einsum('KxaC,Uy,zywa,Wzxw->KWCU', X, h_aa, t1_aaae, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_aaaa -= 1/6 * einsum('KxaC,Wy,zwya,Uxwz->KWCU', X, h_aa, t1_aaae, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_aaaa += 1/6 * einsum('KxaC,Wy,zwya,Uxzw->KWCU', X, h_aa, t1_aaae, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_aaaa += 1/6 * einsum('KxaC,yz,Uwya,Wwxz->KWCU', X, h_aa, t1_aaae, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_aaaa -= 1/6 * einsum('KxaC,yz,Uwya,Wwzx->KWCU', X, h_aa, t1_aaae, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_aaaa += 1/6 * einsum('KxaC,yz,Uywa,Wzwx->KWCU', X, h_aa, t1_aaae, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_aaaa -= 1/6 * einsum('KxaC,yz,Uywa,Wzxw->KWCU', X, h_aa, t1_aaae, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_aaaa -= 1/2 * einsum('KxaC,yz,wUya,Wwxz->KWCU', X, h_aa, t1_aaae, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_aaaa -= 1/6 * einsum('KxaC,yz,wyWa,Uxwz->KWCU', X, h_aa, t1_aaae, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_aaaa += 1/6 * einsum('KxaC,yz,wyWa,Uxzw->KWCU', X, h_aa, t1_aaae, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_aaaa += 1/2 * einsum('KxaC,yz,yUwa,Wzxw->KWCU', X, h_aa, t1_aaae, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_aaaa += 1/6 * einsum('KxaC,yz,ywWa,Uxwz->KWCU', X, h_aa, t1_aaae, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_aaaa -= 1/6 * einsum('KxaC,yz,ywWa,Uxzw->KWCU', X, h_aa, t1_aaae, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_aaaa += 1/6 * einsum('KxCa,Uyza,ywuv,Wwvuxz->KWCU', X, t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_KWCU_aaaa += 1/6 * einsum('KxCa,Uyza,ywuv,Wwvuzx->KWCU', X, t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_KWCU_aaaa += 1/6 * einsum('KxCa,Uyza,ywuv,Wwvxuz->KWCU', X, t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_KWCU_aaaa += 1/6 * einsum('KxCa,Uyza,ywuv,Wwvxzu->KWCU', X, t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_KWCU_aaaa += 1/6 * einsum('KxCa,Uyza,ywuv,Wwvzux->KWCU', X, t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_KWCU_aaaa -= 1/3 * einsum('KxCa,Uyza,ywuv,Wwvzxu->KWCU', X, t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_KWCU_aaaa -= 1/6 * einsum('KxCa,Uyza,zwuv,Wyuvwx->KWCU', X, t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_KWCU_aaaa -= 1/6 * einsum('KxCa,Uyza,zwuv,Wyuvxw->KWCU', X, t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_KWCU_aaaa -= 1/6 * einsum('KxCa,Uyza,zwuv,Wyuwvx->KWCU', X, t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_KWCU_aaaa += 1/3 * einsum('KxCa,Uyza,zwuv,Wyuwxv->KWCU', X, t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_KWCU_aaaa -= 1/6 * einsum('KxCa,Uyza,zwuv,Wyuxvw->KWCU', X, t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_KWCU_aaaa -= 1/6 * einsum('KxCa,Uyza,zwuv,Wyuxwv->KWCU', X, t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_KWCU_aaaa -= 1/2 * einsum('KxCa,yUza,ywuv,Wwvxzu->KWCU', X, t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_KWCU_aaaa += 1/2 * einsum('KxCa,yUza,zwuv,Wyuxwv->KWCU', X, t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_KWCU_aaaa -= 1/2 * einsum('KxCa,ya,Uyzw,Wzxw->KWCU', X, t1_ae, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_aaaa -= 1/2 * einsum('KxCa,ya,Uzwy,Wwzx->KWCU', X, t1_ae, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_aaaa += 1/2 * einsum('KxCa,ya,Wzyw,Uxzw->KWCU', X, t1_ae, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_aaaa -= 1/6 * einsum('KxCa,yzWa,ywuv,Uxuvwz->KWCU', X, t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_KWCU_aaaa -= 1/6 * einsum('KxCa,yzWa,ywuv,Uxuvzw->KWCU', X, t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_KWCU_aaaa -= 1/6 * einsum('KxCa,yzWa,ywuv,Uxuwvz->KWCU', X, t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_KWCU_aaaa += 1/3 * einsum('KxCa,yzWa,ywuv,Uxuwzv->KWCU', X, t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_KWCU_aaaa -= 1/6 * einsum('KxCa,yzWa,ywuv,Uxuzvw->KWCU', X, t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_KWCU_aaaa -= 1/6 * einsum('KxCa,yzWa,ywuv,Uxuzwv->KWCU', X, t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_KWCU_aaaa += 1/2 * einsum('KxCa,yzWa,ywzu,Uxwu->KWCU', X, t1_aaae, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_aaaa += 1/2 * einsum('KxCa,yzWa,zwuv,Uxuywv->KWCU', X, t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_KWCU_aaaa -= 1/2 * einsum('KxCa,yzwa,Uuvy,Wzvuxw->KWCU', X, t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_KWCU_aaaa -= 1/3 * einsum('KxCa,yzwa,Uuvz,Wyvuwx->KWCU', X, t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_KWCU_aaaa += 1/6 * einsum('KxCa,yzwa,Uuvz,Wyvuxw->KWCU', X, t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_KWCU_aaaa += 1/6 * einsum('KxCa,yzwa,Uuvz,Wyvwux->KWCU', X, t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_KWCU_aaaa += 1/6 * einsum('KxCa,yzwa,Uuvz,Wyvwxu->KWCU', X, t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_KWCU_aaaa += 1/6 * einsum('KxCa,yzwa,Uuvz,Wyvxuw->KWCU', X, t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_KWCU_aaaa += 1/6 * einsum('KxCa,yzwa,Uuvz,Wyvxwu->KWCU', X, t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_KWCU_aaaa -= 1/6 * einsum('KxCa,yzwa,Uuwv,Wzyuvx->KWCU', X, t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_KWCU_aaaa += 1/3 * einsum('KxCa,yzwa,Uuwv,Wzyuxv->KWCU', X, t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_KWCU_aaaa -= 1/6 * einsum('KxCa,yzwa,Uuwv,Wzyvux->KWCU', X, t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_KWCU_aaaa -= 1/6 * einsum('KxCa,yzwa,Uuwv,Wzyvxu->KWCU', X, t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_KWCU_aaaa -= 1/6 * einsum('KxCa,yzwa,Uuwv,Wzyxuv->KWCU', X, t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_KWCU_aaaa -= 1/6 * einsum('KxCa,yzwa,Uuwv,Wzyxvu->KWCU', X, t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_KWCU_aaaa += 1/6 * einsum('KxCa,yzwa,Uyuv,Wzuvwx->KWCU', X, t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_KWCU_aaaa += 1/6 * einsum('KxCa,yzwa,Uyuv,Wzuvxw->KWCU', X, t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_KWCU_aaaa += 1/6 * einsum('KxCa,yzwa,Uyuv,Wzuwvx->KWCU', X, t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_KWCU_aaaa -= 1/3 * einsum('KxCa,yzwa,Uyuv,Wzuwxv->KWCU', X, t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_KWCU_aaaa += 1/6 * einsum('KxCa,yzwa,Uyuv,Wzuxvw->KWCU', X, t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_KWCU_aaaa += 1/6 * einsum('KxCa,yzwa,Uyuv,Wzuxwv->KWCU', X, t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_KWCU_aaaa -= 1/2 * einsum('KxCa,yzwa,Uyuz,Wuwx->KWCU', X, t1_aaae, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_aaaa -= 1/2 * einsum('KxCa,yzwa,Uzuv,Wyuxwv->KWCU', X, t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_KWCU_aaaa -= 1/2 * einsum('KxCa,yzwa,Uzuy,Wuxw->KWCU', X, t1_aaae, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_aaaa -= 1/2 * einsum('KxCa,yzwa,Wuvw,Uxvuzy->KWCU', X, t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_KWCU_aaaa -= 1/6 * einsum('KxCa,yzwa,Wuyv,Uxwuvz->KWCU', X, t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_KWCU_aaaa += 1/3 * einsum('KxCa,yzwa,Wuyv,Uxwuzv->KWCU', X, t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_KWCU_aaaa -= 1/6 * einsum('KxCa,yzwa,Wuyv,Uxwvuz->KWCU', X, t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_KWCU_aaaa -= 1/6 * einsum('KxCa,yzwa,Wuyv,Uxwvzu->KWCU', X, t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_KWCU_aaaa -= 1/6 * einsum('KxCa,yzwa,Wuyv,Uxwzuv->KWCU', X, t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_KWCU_aaaa -= 1/6 * einsum('KxCa,yzwa,Wuyv,Uxwzvu->KWCU', X, t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_KWCU_aaaa += 1/2 * einsum('KxCa,yzwa,Wuzv,Uxwuvy->KWCU', X, t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_KWCU_aaaa += 1/6 * einsum('KxCa,yzwa,Wwuv,Uxuvyz->KWCU', X, t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_KWCU_aaaa += 1/6 * einsum('KxCa,yzwa,Wwuv,Uxuvzy->KWCU', X, t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_KWCU_aaaa += 1/6 * einsum('KxCa,yzwa,Wwuv,Uxuyvz->KWCU', X, t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_KWCU_aaaa -= 1/3 * einsum('KxCa,yzwa,Wwuv,Uxuyzv->KWCU', X, t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_KWCU_aaaa += 1/6 * einsum('KxCa,yzwa,Wwuv,Uxuzvy->KWCU', X, t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_KWCU_aaaa += 1/6 * einsum('KxCa,yzwa,Wwuv,Uxuzyv->KWCU', X, t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_KWCU_aaaa -= 1/6 * einsum('KxaC,Uyza,ywuv,Wwvxzu->KWCU', X, t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_KWCU_aaaa += 1/6 * einsum('KxaC,Uyza,ywuv,Wwvzxu->KWCU', X, t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_KWCU_aaaa -= 1/6 * einsum('KxaC,Uyza,zwuv,Wyuwxv->KWCU', X, t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_KWCU_aaaa += 1/6 * einsum('KxaC,Uyza,zwuv,Wyuxwv->KWCU', X, t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_KWCU_aaaa += 1/2 * einsum('KxaC,yUza,ywuv,Wwvxzu->KWCU', X, t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_KWCU_aaaa -= 1/2 * einsum('KxaC,yUza,zwuv,Wyuxwv->KWCU', X, t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_KWCU_aaaa += 1/2 * einsum('KxaC,ya,Uyzw,Wzxw->KWCU', X, t1_ae, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_aaaa -= 1/6 * einsum('KxaC,ya,Uzwy,Wwxz->KWCU', X, t1_ae, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_aaaa += 1/6 * einsum('KxaC,ya,Uzwy,Wwzx->KWCU', X, t1_ae, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_aaaa += 1/6 * einsum('KxaC,ya,Wzyw,Uxwz->KWCU', X, t1_ae, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_aaaa -= 1/6 * einsum('KxaC,ya,Wzyw,Uxzw->KWCU', X, t1_ae, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_aaaa -= 1/6 * einsum('KxaC,yzWa,ywuv,Uxuwzv->KWCU', X, t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_KWCU_aaaa += 1/6 * einsum('KxaC,yzWa,ywuv,Uxuzwv->KWCU', X, t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_KWCU_aaaa += 1/6 * einsum('KxaC,yzWa,ywzu,Uxuw->KWCU', X, t1_aaae, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_aaaa -= 1/6 * einsum('KxaC,yzWa,ywzu,Uxwu->KWCU', X, t1_aaae, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_aaaa += 1/6 * einsum('KxaC,yzWa,zwuv,Uxuwyv->KWCU', X, t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_KWCU_aaaa -= 1/6 * einsum('KxaC,yzWa,zwuv,Uxuywv->KWCU', X, t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_KWCU_aaaa += 1/6 * einsum('KxaC,yzwa,Uuvy,Wzvuxw->KWCU', X, t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_KWCU_aaaa -= 1/6 * einsum('KxaC,yzwa,Uuvy,Wzvxuw->KWCU', X, t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_KWCU_aaaa += 1/6 * einsum('KxaC,yzwa,Uuvz,Wyvuwx->KWCU', X, t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_KWCU_aaaa -= 1/6 * einsum('KxaC,yzwa,Uuvz,Wyvxwu->KWCU', X, t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_KWCU_aaaa -= 1/6 * einsum('KxaC,yzwa,Uuwv,Wzyuxv->KWCU', X, t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_KWCU_aaaa += 1/6 * einsum('KxaC,yzwa,Uuwv,Wzyxuv->KWCU', X, t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_KWCU_aaaa += 1/6 * einsum('KxaC,yzwa,Uyuv,Wzuwxv->KWCU', X, t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_KWCU_aaaa -= 1/6 * einsum('KxaC,yzwa,Uyuv,Wzuxwv->KWCU', X, t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_KWCU_aaaa += 1/6 * einsum('KxaC,yzwa,Uyuz,Wuwx->KWCU', X, t1_aaae, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_aaaa -= 1/6 * einsum('KxaC,yzwa,Uyuz,Wuxw->KWCU', X, t1_aaae, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_aaaa += 1/2 * einsum('KxaC,yzwa,Uzuv,Wyuxwv->KWCU', X, t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_KWCU_aaaa += 1/2 * einsum('KxaC,yzwa,Uzuy,Wuxw->KWCU', X, t1_aaae, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_aaaa += 1/6 * einsum('KxaC,yzwa,Wuvw,Uxvuzy->KWCU', X, t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_KWCU_aaaa -= 1/6 * einsum('KxaC,yzwa,Wuvw,Uxvzuy->KWCU', X, t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_KWCU_aaaa -= 1/6 * einsum('KxaC,yzwa,Wuyv,Uxwuzv->KWCU', X, t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_KWCU_aaaa += 1/6 * einsum('KxaC,yzwa,Wuyv,Uxwzuv->KWCU', X, t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_KWCU_aaaa -= 1/6 * einsum('KxaC,yzwa,Wuzv,Uxwuvy->KWCU', X, t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_KWCU_aaaa += 1/6 * einsum('KxaC,yzwa,Wuzv,Uxwvuy->KWCU', X, t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_KWCU_aaaa += 1/6 * einsum('KxaC,yzwa,Wwuv,Uxuyzv->KWCU', X, t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_KWCU_aaaa -= 1/6 * einsum('KxaC,yzwa,Wwuv,Uxuzyv->KWCU', X, t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma[caea__aaaa] += ascontiguousarray(sigma_KWCU_aaaa).reshape(-1)

    sigma_KWCU_abab  = 1/2 * einsum('KxCa,Ua,Wx->KWCU', X, h_ae, rdm_ca, optimize = einsum_type)
    sigma_KWCU_abab += 1/2 * einsum('KxCa,Uyza,Wzyx->KWCU', X, v_aaae, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_abab -= 1/2 * einsum('KxCa,yWza,Uxyz->KWCU', X, v_aaae, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_abab += 1/2 * einsum('KxCa,yzUa,Wyxz->KWCU', X, v_aaae, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_abab -= 1/6 * einsum('KxaC,Uyza,Wzxy->KWCU', X, v_aaae, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_abab -= 1/3 * einsum('KxaC,Uyza,Wzyx->KWCU', X, v_aaae, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_abab += 1/3 * einsum('KxaC,yWza,Uxyz->KWCU', X, v_aaae, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_abab += 1/6 * einsum('KxaC,yWza,Uxzy->KWCU', X, v_aaae, rdm_ccaa, optimize = einsum_type)
    #sigma_KWCU_abab += 1/2 * einsum('Kxab,UbCa,Wx->KWCU', X, v_aeee, rdm_ca, optimize = einsum_type)
    #sigma_KWCU_abab -= 1/3 * einsum('Kxab,yaCb,UxWy->KWCU', X, v_aeee, rdm_ccaa, optimize = einsum_type)
    #sigma_KWCU_abab -= 1/6 * einsum('Kxab,yaCb,UxyW->KWCU', X, v_aeee, rdm_ccaa, optimize = einsum_type)
    #sigma_KWCU_abab += 1/2 * einsum('Kxab,ybCa,UxWy->KWCU', X, v_aeee, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_abab += 1/3 * einsum('ixCa,Kyai,UxWy->KWCU', X, v_xaex, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_abab += 1/6 * einsum('ixCa,Kyai,UxyW->KWCU', X, v_xaex, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_abab -= 1/2 * einsum('ixCa,iKUa,Wx->KWCU', X, v_xxae, rdm_ca, optimize = einsum_type)
    sigma_KWCU_abab -= 1/2 * einsum('ixCa,iKya,UxWy->KWCU', X, v_xxae, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_abab -= 2/3 * einsum('ixaC,Kyai,UxWy->KWCU', X, v_xaex, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_abab -= 1/3 * einsum('ixaC,Kyai,UxyW->KWCU', X, v_xaex, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_abab += 1/3 * einsum('ixaC,iKya,UxWy->KWCU', X, v_xxae, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_abab += 1/6 * einsum('ixaC,iKya,UxyW->KWCU', X, v_xxae, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_abab += 1/2 * einsum('KxCa,a,Ua,Wx->KWCU', X, e_extern, t1_ae, rdm_ca, optimize = einsum_type)
    sigma_KWCU_abab += 1/2 * einsum('KxCa,a,Uyza,Wyzx->KWCU', X, e_extern, t1_aaae, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_abab += 1/2 * einsum('KxCa,a,yUza,Wyxz->KWCU', X, e_extern, t1_aaae, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_abab -= 1/2 * einsum('KxCa,a,yzWa,Uxyz->KWCU', X, e_extern, t1_aaae, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_abab -= 1/6 * einsum('KxaC,a,Uyza,Wyxz->KWCU', X, e_extern, t1_aaae, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_abab -= 1/3 * einsum('KxaC,a,Uyza,Wyzx->KWCU', X, e_extern, t1_aaae, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_abab += 1/3 * einsum('KxaC,a,yzWa,Uxyz->KWCU', X, e_extern, t1_aaae, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_abab += 1/6 * einsum('KxaC,a,yzWa,Uxzy->KWCU', X, e_extern, t1_aaae, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_abab -= 1/2 * einsum('KxCa,Uy,ya,Wx->KWCU', X, h_aa, t1_ae, rdm_ca, optimize = einsum_type)
    sigma_KWCU_abab -= 1/2 * einsum('KxCa,Uy,yzwa,Wzwx->KWCU', X, h_aa, t1_aaae, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_abab -= 1/2 * einsum('KxCa,Uy,zywa,Wzxw->KWCU', X, h_aa, t1_aaae, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_abab -= 1/2 * einsum('KxCa,Wy,zwya,Uxzw->KWCU', X, h_aa, t1_aaae, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_abab += 1/2 * einsum('KxCa,yz,Uwya,Wwzx->KWCU', X, h_aa, t1_aaae, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_abab -= 1/2 * einsum('KxCa,yz,Uywa,Wzwx->KWCU', X, h_aa, t1_aaae, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_abab += 1/2 * einsum('KxCa,yz,wUya,Wwxz->KWCU', X, h_aa, t1_aaae, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_abab += 1/2 * einsum('KxCa,yz,wyWa,Uxwz->KWCU', X, h_aa, t1_aaae, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_abab -= 1/2 * einsum('KxCa,yz,yUwa,Wzxw->KWCU', X, h_aa, t1_aaae, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_abab += 1/2 * einsum('KxCa,yz,ywWa,Uxzw->KWCU', X, h_aa, t1_aaae, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_abab += 1/3 * einsum('KxaC,Uy,yzwa,Wzwx->KWCU', X, h_aa, t1_aaae, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_abab += 1/6 * einsum('KxaC,Uy,yzwa,Wzxw->KWCU', X, h_aa, t1_aaae, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_abab += 1/6 * einsum('KxaC,Wy,zwya,Uxwz->KWCU', X, h_aa, t1_aaae, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_abab += 1/3 * einsum('KxaC,Wy,zwya,Uxzw->KWCU', X, h_aa, t1_aaae, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_abab -= 1/6 * einsum('KxaC,yz,Uwya,Wwxz->KWCU', X, h_aa, t1_aaae, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_abab -= 1/3 * einsum('KxaC,yz,Uwya,Wwzx->KWCU', X, h_aa, t1_aaae, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_abab += 1/3 * einsum('KxaC,yz,Uywa,Wzwx->KWCU', X, h_aa, t1_aaae, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_abab += 1/6 * einsum('KxaC,yz,Uywa,Wzxw->KWCU', X, h_aa, t1_aaae, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_abab -= 1/3 * einsum('KxaC,yz,wyWa,Uxwz->KWCU', X, h_aa, t1_aaae, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_abab -= 1/6 * einsum('KxaC,yz,wyWa,Uxzw->KWCU', X, h_aa, t1_aaae, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_abab -= 1/6 * einsum('KxaC,yz,ywWa,Uxwz->KWCU', X, h_aa, t1_aaae, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_abab -= 1/3 * einsum('KxaC,yz,ywWa,Uxzw->KWCU', X, h_aa, t1_aaae, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_abab += 1/6 * einsum('KxCa,Uyza,ywuv,Wwvuxz->KWCU', X, t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_KWCU_abab += 1/6 * einsum('KxCa,Uyza,ywuv,Wwvuzx->KWCU', X, t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_KWCU_abab += 1/6 * einsum('KxCa,Uyza,ywuv,Wwvxuz->KWCU', X, t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_KWCU_abab += 1/6 * einsum('KxCa,Uyza,ywuv,Wwvxzu->KWCU', X, t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_KWCU_abab += 1/6 * einsum('KxCa,Uyza,ywuv,Wwvzux->KWCU', X, t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_KWCU_abab -= 1/3 * einsum('KxCa,Uyza,ywuv,Wwvzxu->KWCU', X, t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_KWCU_abab -= 1/6 * einsum('KxCa,Uyza,zwuv,Wyuvwx->KWCU', X, t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_KWCU_abab -= 1/6 * einsum('KxCa,Uyza,zwuv,Wyuvxw->KWCU', X, t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_KWCU_abab -= 1/6 * einsum('KxCa,Uyza,zwuv,Wyuwvx->KWCU', X, t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_KWCU_abab += 1/3 * einsum('KxCa,Uyza,zwuv,Wyuwxv->KWCU', X, t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_KWCU_abab -= 1/6 * einsum('KxCa,Uyza,zwuv,Wyuxvw->KWCU', X, t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_KWCU_abab -= 1/6 * einsum('KxCa,Uyza,zwuv,Wyuxwv->KWCU', X, t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_KWCU_abab -= 1/2 * einsum('KxCa,yUza,ywuv,Wwvxzu->KWCU', X, t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_KWCU_abab += 1/2 * einsum('KxCa,yUza,zwuv,Wyuxwv->KWCU', X, t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_KWCU_abab -= 1/2 * einsum('KxCa,ya,Uyzw,Wzxw->KWCU', X, t1_ae, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_abab -= 1/2 * einsum('KxCa,ya,Uzwy,Wwzx->KWCU', X, t1_ae, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_abab += 1/2 * einsum('KxCa,ya,Wzyw,Uxzw->KWCU', X, t1_ae, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_abab -= 1/6 * einsum('KxCa,yzWa,ywuv,Uxuvwz->KWCU', X, t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_KWCU_abab -= 1/6 * einsum('KxCa,yzWa,ywuv,Uxuvzw->KWCU', X, t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_KWCU_abab -= 1/6 * einsum('KxCa,yzWa,ywuv,Uxuwvz->KWCU', X, t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_KWCU_abab += 1/3 * einsum('KxCa,yzWa,ywuv,Uxuwzv->KWCU', X, t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_KWCU_abab -= 1/6 * einsum('KxCa,yzWa,ywuv,Uxuzvw->KWCU', X, t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_KWCU_abab -= 1/6 * einsum('KxCa,yzWa,ywuv,Uxuzwv->KWCU', X, t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_KWCU_abab += 1/2 * einsum('KxCa,yzWa,ywzu,Uxwu->KWCU', X, t1_aaae, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_abab += 1/2 * einsum('KxCa,yzWa,zwuv,Uxuywv->KWCU', X, t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_KWCU_abab -= 1/2 * einsum('KxCa,yzwa,Uuvy,Wzvuxw->KWCU', X, t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_KWCU_abab -= 1/3 * einsum('KxCa,yzwa,Uuvz,Wyvuwx->KWCU', X, t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_KWCU_abab += 1/6 * einsum('KxCa,yzwa,Uuvz,Wyvuxw->KWCU', X, t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_KWCU_abab += 1/6 * einsum('KxCa,yzwa,Uuvz,Wyvwux->KWCU', X, t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_KWCU_abab += 1/6 * einsum('KxCa,yzwa,Uuvz,Wyvwxu->KWCU', X, t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_KWCU_abab += 1/6 * einsum('KxCa,yzwa,Uuvz,Wyvxuw->KWCU', X, t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_KWCU_abab += 1/6 * einsum('KxCa,yzwa,Uuvz,Wyvxwu->KWCU', X, t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_KWCU_abab -= 1/6 * einsum('KxCa,yzwa,Uuwv,Wzyuvx->KWCU', X, t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_KWCU_abab += 1/3 * einsum('KxCa,yzwa,Uuwv,Wzyuxv->KWCU', X, t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_KWCU_abab -= 1/6 * einsum('KxCa,yzwa,Uuwv,Wzyvux->KWCU', X, t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_KWCU_abab -= 1/6 * einsum('KxCa,yzwa,Uuwv,Wzyvxu->KWCU', X, t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_KWCU_abab -= 1/6 * einsum('KxCa,yzwa,Uuwv,Wzyxuv->KWCU', X, t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_KWCU_abab -= 1/6 * einsum('KxCa,yzwa,Uuwv,Wzyxvu->KWCU', X, t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_KWCU_abab += 1/6 * einsum('KxCa,yzwa,Uyuv,Wzuvwx->KWCU', X, t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_KWCU_abab += 1/6 * einsum('KxCa,yzwa,Uyuv,Wzuvxw->KWCU', X, t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_KWCU_abab += 1/6 * einsum('KxCa,yzwa,Uyuv,Wzuwvx->KWCU', X, t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_KWCU_abab -= 1/3 * einsum('KxCa,yzwa,Uyuv,Wzuwxv->KWCU', X, t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_KWCU_abab += 1/6 * einsum('KxCa,yzwa,Uyuv,Wzuxvw->KWCU', X, t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_KWCU_abab += 1/6 * einsum('KxCa,yzwa,Uyuv,Wzuxwv->KWCU', X, t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_KWCU_abab -= 1/2 * einsum('KxCa,yzwa,Uyuz,Wuwx->KWCU', X, t1_aaae, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_abab -= 1/2 * einsum('KxCa,yzwa,Uzuv,Wyuxwv->KWCU', X, t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_KWCU_abab -= 1/2 * einsum('KxCa,yzwa,Uzuy,Wuxw->KWCU', X, t1_aaae, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_abab -= 1/2 * einsum('KxCa,yzwa,Wuvw,Uxvuzy->KWCU', X, t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_KWCU_abab -= 1/6 * einsum('KxCa,yzwa,Wuyv,Uxwuvz->KWCU', X, t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_KWCU_abab += 1/3 * einsum('KxCa,yzwa,Wuyv,Uxwuzv->KWCU', X, t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_KWCU_abab -= 1/6 * einsum('KxCa,yzwa,Wuyv,Uxwvuz->KWCU', X, t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_KWCU_abab -= 1/6 * einsum('KxCa,yzwa,Wuyv,Uxwvzu->KWCU', X, t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_KWCU_abab -= 1/6 * einsum('KxCa,yzwa,Wuyv,Uxwzuv->KWCU', X, t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_KWCU_abab -= 1/6 * einsum('KxCa,yzwa,Wuyv,Uxwzvu->KWCU', X, t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_KWCU_abab += 1/2 * einsum('KxCa,yzwa,Wuzv,Uxwuvy->KWCU', X, t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_KWCU_abab += 1/6 * einsum('KxCa,yzwa,Wwuv,Uxuvyz->KWCU', X, t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_KWCU_abab += 1/6 * einsum('KxCa,yzwa,Wwuv,Uxuvzy->KWCU', X, t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_KWCU_abab += 1/6 * einsum('KxCa,yzwa,Wwuv,Uxuyvz->KWCU', X, t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_KWCU_abab -= 1/3 * einsum('KxCa,yzwa,Wwuv,Uxuyzv->KWCU', X, t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_KWCU_abab += 1/6 * einsum('KxCa,yzwa,Wwuv,Uxuzvy->KWCU', X, t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_KWCU_abab += 1/6 * einsum('KxCa,yzwa,Wwuv,Uxuzyv->KWCU', X, t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_KWCU_abab -= 1/6 * einsum('KxaC,Uyza,ywuv,Wwvuxz->KWCU', X, t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_KWCU_abab -= 1/6 * einsum('KxaC,Uyza,ywuv,Wwvuzx->KWCU', X, t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_KWCU_abab -= 1/6 * einsum('KxaC,Uyza,ywuv,Wwvxuz->KWCU', X, t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_KWCU_abab -= 1/6 * einsum('KxaC,Uyza,ywuv,Wwvzux->KWCU', X, t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_KWCU_abab += 1/6 * einsum('KxaC,Uyza,ywuv,Wwvzxu->KWCU', X, t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_KWCU_abab += 1/6 * einsum('KxaC,Uyza,zwuv,Wyuvwx->KWCU', X, t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_KWCU_abab += 1/6 * einsum('KxaC,Uyza,zwuv,Wyuvxw->KWCU', X, t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_KWCU_abab += 1/6 * einsum('KxaC,Uyza,zwuv,Wyuwvx->KWCU', X, t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_KWCU_abab -= 1/6 * einsum('KxaC,Uyza,zwuv,Wyuwxv->KWCU', X, t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_KWCU_abab += 1/6 * einsum('KxaC,Uyza,zwuv,Wyuxvw->KWCU', X, t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_KWCU_abab += 1/6 * einsum('KxaC,ya,Uzwy,Wwxz->KWCU', X, t1_ae, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_abab += 1/3 * einsum('KxaC,ya,Uzwy,Wwzx->KWCU', X, t1_ae, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_abab -= 1/6 * einsum('KxaC,ya,Wzyw,Uxwz->KWCU', X, t1_ae, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_abab -= 1/3 * einsum('KxaC,ya,Wzyw,Uxzw->KWCU', X, t1_ae, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_abab += 1/6 * einsum('KxaC,yzWa,ywuv,Uxuvwz->KWCU', X, t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_KWCU_abab += 1/6 * einsum('KxaC,yzWa,ywuv,Uxuvzw->KWCU', X, t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_KWCU_abab += 1/6 * einsum('KxaC,yzWa,ywuv,Uxuwvz->KWCU', X, t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_KWCU_abab -= 1/6 * einsum('KxaC,yzWa,ywuv,Uxuwzv->KWCU', X, t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_KWCU_abab += 1/6 * einsum('KxaC,yzWa,ywuv,Uxuzvw->KWCU', X, t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_KWCU_abab -= 1/6 * einsum('KxaC,yzWa,ywzu,Uxuw->KWCU', X, t1_aaae, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_abab -= 1/3 * einsum('KxaC,yzWa,ywzu,Uxwu->KWCU', X, t1_aaae, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_abab -= 1/6 * einsum('KxaC,yzWa,zwuv,Uxuwyv->KWCU', X, t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_KWCU_abab -= 1/3 * einsum('KxaC,yzWa,zwuv,Uxuywv->KWCU', X, t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_KWCU_abab += 1/3 * einsum('KxaC,yzwa,Uuvy,Wzvuxw->KWCU', X, t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_KWCU_abab += 1/6 * einsum('KxaC,yzwa,Uuvy,Wzvxuw->KWCU', X, t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_KWCU_abab += 1/6 * einsum('KxaC,yzwa,Uuvz,Wyvuwx->KWCU', X, t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_KWCU_abab -= 1/6 * einsum('KxaC,yzwa,Uuvz,Wyvuxw->KWCU', X, t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_KWCU_abab -= 1/6 * einsum('KxaC,yzwa,Uuvz,Wyvwux->KWCU', X, t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_KWCU_abab -= 1/6 * einsum('KxaC,yzwa,Uuvz,Wyvwxu->KWCU', X, t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_KWCU_abab -= 1/6 * einsum('KxaC,yzwa,Uuvz,Wyvxuw->KWCU', X, t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_KWCU_abab += 1/6 * einsum('KxaC,yzwa,Uuwv,Wzyuvx->KWCU', X, t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_KWCU_abab -= 1/6 * einsum('KxaC,yzwa,Uuwv,Wzyuxv->KWCU', X, t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_KWCU_abab += 1/6 * einsum('KxaC,yzwa,Uuwv,Wzyvux->KWCU', X, t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_KWCU_abab += 1/6 * einsum('KxaC,yzwa,Uuwv,Wzyvxu->KWCU', X, t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_KWCU_abab += 1/6 * einsum('KxaC,yzwa,Uuwv,Wzyxvu->KWCU', X, t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_KWCU_abab -= 1/6 * einsum('KxaC,yzwa,Uyuv,Wzuvwx->KWCU', X, t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_KWCU_abab -= 1/6 * einsum('KxaC,yzwa,Uyuv,Wzuvxw->KWCU', X, t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_KWCU_abab -= 1/6 * einsum('KxaC,yzwa,Uyuv,Wzuwvx->KWCU', X, t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_KWCU_abab += 1/6 * einsum('KxaC,yzwa,Uyuv,Wzuwxv->KWCU', X, t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_KWCU_abab -= 1/6 * einsum('KxaC,yzwa,Uyuv,Wzuxvw->KWCU', X, t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_KWCU_abab += 1/3 * einsum('KxaC,yzwa,Uyuz,Wuwx->KWCU', X, t1_aaae, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_abab += 1/6 * einsum('KxaC,yzwa,Uyuz,Wuxw->KWCU', X, t1_aaae, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_abab += 1/3 * einsum('KxaC,yzwa,Wuvw,Uxvuzy->KWCU', X, t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_KWCU_abab += 1/6 * einsum('KxaC,yzwa,Wuvw,Uxvzuy->KWCU', X, t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_KWCU_abab += 1/6 * einsum('KxaC,yzwa,Wuyv,Uxwuvz->KWCU', X, t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_KWCU_abab -= 1/6 * einsum('KxaC,yzwa,Wuyv,Uxwuzv->KWCU', X, t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_KWCU_abab += 1/6 * einsum('KxaC,yzwa,Wuyv,Uxwvuz->KWCU', X, t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_KWCU_abab += 1/6 * einsum('KxaC,yzwa,Wuyv,Uxwvzu->KWCU', X, t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_KWCU_abab += 1/6 * einsum('KxaC,yzwa,Wuyv,Uxwzvu->KWCU', X, t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_KWCU_abab -= 1/3 * einsum('KxaC,yzwa,Wuzv,Uxwuvy->KWCU', X, t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_KWCU_abab -= 1/6 * einsum('KxaC,yzwa,Wuzv,Uxwvuy->KWCU', X, t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_KWCU_abab -= 1/6 * einsum('KxaC,yzwa,Wwuv,Uxuvyz->KWCU', X, t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_KWCU_abab -= 1/6 * einsum('KxaC,yzwa,Wwuv,Uxuvzy->KWCU', X, t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_KWCU_abab -= 1/6 * einsum('KxaC,yzwa,Wwuv,Uxuyvz->KWCU', X, t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_KWCU_abab += 1/6 * einsum('KxaC,yzwa,Wwuv,Uxuyzv->KWCU', X, t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_KWCU_abab -= 1/6 * einsum('KxaC,yzwa,Wwuv,Uxuzvy->KWCU', X, t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma[caea__abab] += ascontiguousarray(sigma_KWCU_abab).reshape(-1)

    sigma_KWCU_baab =- 1/2 * einsum('KxaC,Ua,Wx->KWCU', X, h_ae, rdm_ca, optimize = einsum_type)
    sigma_KWCU_baab += 1/3 * einsum('KxaC,Uyza,Wzxy->KWCU', X, v_aaae, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_baab += 1/6 * einsum('KxaC,Uyza,Wzyx->KWCU', X, v_aaae, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_baab -= 1/6 * einsum('KxaC,yWza,Uxyz->KWCU', X, v_aaae, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_baab -= 1/3 * einsum('KxaC,yWza,Uxzy->KWCU', X, v_aaae, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_baab -= 1/2 * einsum('KxaC,yzUa,Wyxz->KWCU', X, v_aaae, rdm_ccaa, optimize = einsum_type)
    #sigma_KWCU_baab -= 1/2 * einsum('Kxab,UaCb,Wx->KWCU', X, v_aeee, rdm_ca, optimize = einsum_type)
    #sigma_KWCU_baab += 1/6 * einsum('Kxab,yaCb,UxWy->KWCU', X, v_aeee, rdm_ccaa, optimize = einsum_type)
    #sigma_KWCU_baab += 1/3 * einsum('Kxab,yaCb,UxyW->KWCU', X, v_aeee, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_baab += 1/2 * einsum('ixCa,KUai,Wx->KWCU', X, v_xaex, rdm_ca, optimize = einsum_type)
    sigma_KWCU_baab -= 1/6 * einsum('ixCa,Kyai,UxWy->KWCU', X, v_xaex, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_baab -= 1/3 * einsum('ixCa,Kyai,UxyW->KWCU', X, v_xaex, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_baab -= einsum('ixaC,KUai,Wx->KWCU', X, v_xaex, rdm_ca, optimize = einsum_type)
    sigma_KWCU_baab += 1/3 * einsum('ixaC,Kyai,UxWy->KWCU', X, v_xaex, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_baab += 2/3 * einsum('ixaC,Kyai,UxyW->KWCU', X, v_xaex, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_baab += 1/2 * einsum('ixaC,iKUa,Wx->KWCU', X, v_xxae, rdm_ca, optimize = einsum_type)
    sigma_KWCU_baab -= 1/6 * einsum('ixaC,iKya,UxWy->KWCU', X, v_xxae, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_baab -= 1/3 * einsum('ixaC,iKya,UxyW->KWCU', X, v_xxae, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_baab -= 1/2 * einsum('KxaC,a,Ua,Wx->KWCU', X, e_extern, t1_ae, rdm_ca, optimize = einsum_type)
    sigma_KWCU_baab += 1/3 * einsum('KxaC,a,Uyza,Wyxz->KWCU', X, e_extern, t1_aaae, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_baab += 1/6 * einsum('KxaC,a,Uyza,Wyzx->KWCU', X, e_extern, t1_aaae, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_baab -= 1/2 * einsum('KxaC,a,yUza,Wyxz->KWCU', X, e_extern, t1_aaae, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_baab -= 1/6 * einsum('KxaC,a,yzWa,Uxyz->KWCU', X, e_extern, t1_aaae, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_baab -= 1/3 * einsum('KxaC,a,yzWa,Uxzy->KWCU', X, e_extern, t1_aaae, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_baab += 1/2 * einsum('KxaC,Uy,ya,Wx->KWCU', X, h_aa, t1_ae, rdm_ca, optimize = einsum_type)
    sigma_KWCU_baab -= 1/6 * einsum('KxaC,Uy,yzwa,Wzwx->KWCU', X, h_aa, t1_aaae, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_baab -= 1/3 * einsum('KxaC,Uy,yzwa,Wzxw->KWCU', X, h_aa, t1_aaae, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_baab += 1/2 * einsum('KxaC,Uy,zywa,Wzxw->KWCU', X, h_aa, t1_aaae, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_baab -= 1/3 * einsum('KxaC,Wy,zwya,Uxwz->KWCU', X, h_aa, t1_aaae, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_baab -= 1/6 * einsum('KxaC,Wy,zwya,Uxzw->KWCU', X, h_aa, t1_aaae, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_baab += 1/3 * einsum('KxaC,yz,Uwya,Wwxz->KWCU', X, h_aa, t1_aaae, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_baab += 1/6 * einsum('KxaC,yz,Uwya,Wwzx->KWCU', X, h_aa, t1_aaae, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_baab -= 1/6 * einsum('KxaC,yz,Uywa,Wzwx->KWCU', X, h_aa, t1_aaae, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_baab -= 1/3 * einsum('KxaC,yz,Uywa,Wzxw->KWCU', X, h_aa, t1_aaae, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_baab -= 1/2 * einsum('KxaC,yz,wUya,Wwxz->KWCU', X, h_aa, t1_aaae, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_baab += 1/6 * einsum('KxaC,yz,wyWa,Uxwz->KWCU', X, h_aa, t1_aaae, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_baab += 1/3 * einsum('KxaC,yz,wyWa,Uxzw->KWCU', X, h_aa, t1_aaae, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_baab += 1/2 * einsum('KxaC,yz,yUwa,Wzxw->KWCU', X, h_aa, t1_aaae, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_baab += 1/3 * einsum('KxaC,yz,ywWa,Uxwz->KWCU', X, h_aa, t1_aaae, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_baab += 1/6 * einsum('KxaC,yz,ywWa,Uxzw->KWCU', X, h_aa, t1_aaae, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_baab -= 1/3 * einsum('KxaC,Uyza,ywuv,Wwvxzu->KWCU', X, t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_KWCU_baab -= 1/6 * einsum('KxaC,Uyza,ywuv,Wwvzxu->KWCU', X, t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_KWCU_baab += 1/6 * einsum('KxaC,Uyza,zwuv,Wyuwxv->KWCU', X, t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_KWCU_baab += 1/3 * einsum('KxaC,Uyza,zwuv,Wyuxwv->KWCU', X, t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_KWCU_baab += 1/2 * einsum('KxaC,yUza,ywuv,Wwvxzu->KWCU', X, t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_KWCU_baab -= 1/2 * einsum('KxaC,yUza,zwuv,Wyuxwv->KWCU', X, t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_KWCU_baab += 1/2 * einsum('KxaC,ya,Uyzw,Wzxw->KWCU', X, t1_ae, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_baab -= 1/3 * einsum('KxaC,ya,Uzwy,Wwxz->KWCU', X, t1_ae, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_baab -= 1/6 * einsum('KxaC,ya,Uzwy,Wwzx->KWCU', X, t1_ae, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_baab += 1/3 * einsum('KxaC,ya,Wzyw,Uxwz->KWCU', X, t1_ae, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_baab += 1/6 * einsum('KxaC,ya,Wzyw,Uxzw->KWCU', X, t1_ae, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_baab += 1/6 * einsum('KxaC,yzWa,ywuv,Uxuwzv->KWCU', X, t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_KWCU_baab += 1/3 * einsum('KxaC,yzWa,ywuv,Uxuzwv->KWCU', X, t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_KWCU_baab += 1/3 * einsum('KxaC,yzWa,ywzu,Uxuw->KWCU', X, t1_aaae, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_baab += 1/6 * einsum('KxaC,yzWa,ywzu,Uxwu->KWCU', X, t1_aaae, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_baab -= 1/6 * einsum('KxaC,yzWa,zwuv,Uxuvwy->KWCU', X, t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_KWCU_baab -= 1/6 * einsum('KxaC,yzWa,zwuv,Uxuvyw->KWCU', X, t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_KWCU_baab -= 1/6 * einsum('KxaC,yzWa,zwuv,Uxuwvy->KWCU', X, t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_KWCU_baab += 1/6 * einsum('KxaC,yzWa,zwuv,Uxuwyv->KWCU', X, t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_KWCU_baab -= 1/6 * einsum('KxaC,yzWa,zwuv,Uxuyvw->KWCU', X, t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_KWCU_baab += 1/6 * einsum('KxaC,yzwa,Uuvy,Wzvuwx->KWCU', X, t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_KWCU_baab += 1/6 * einsum('KxaC,yzwa,Uuvy,Wzvwux->KWCU', X, t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_KWCU_baab += 1/6 * einsum('KxaC,yzwa,Uuvy,Wzvwxu->KWCU', X, t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_KWCU_baab -= 1/6 * einsum('KxaC,yzwa,Uuvy,Wzvxuw->KWCU', X, t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_KWCU_baab += 1/6 * einsum('KxaC,yzwa,Uuvy,Wzvxwu->KWCU', X, t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_KWCU_baab -= 1/6 * einsum('KxaC,yzwa,Uuvz,Wyvuwx->KWCU', X, t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_KWCU_baab -= 1/3 * einsum('KxaC,yzwa,Uuvz,Wyvxwu->KWCU', X, t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_KWCU_baab += 1/6 * einsum('KxaC,yzwa,Uuwv,Wzyuxv->KWCU', X, t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_KWCU_baab += 1/3 * einsum('KxaC,yzwa,Uuwv,Wzyxuv->KWCU', X, t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_KWCU_baab -= 1/6 * einsum('KxaC,yzwa,Uyuv,Wzuwxv->KWCU', X, t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_KWCU_baab -= 1/3 * einsum('KxaC,yzwa,Uyuv,Wzuxwv->KWCU', X, t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_KWCU_baab -= 1/6 * einsum('KxaC,yzwa,Uyuz,Wuwx->KWCU', X, t1_aaae, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_baab -= 1/3 * einsum('KxaC,yzwa,Uyuz,Wuxw->KWCU', X, t1_aaae, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_baab += 1/2 * einsum('KxaC,yzwa,Uzuv,Wyuxwv->KWCU', X, t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_KWCU_baab += 1/2 * einsum('KxaC,yzwa,Uzuy,Wuxw->KWCU', X, t1_aaae, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_baab += 1/6 * einsum('KxaC,yzwa,Wuvw,Uxvuyz->KWCU', X, t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_KWCU_baab += 1/6 * einsum('KxaC,yzwa,Wuvw,Uxvyuz->KWCU', X, t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_KWCU_baab += 1/6 * einsum('KxaC,yzwa,Wuvw,Uxvyzu->KWCU', X, t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_KWCU_baab -= 1/6 * einsum('KxaC,yzwa,Wuvw,Uxvzuy->KWCU', X, t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_KWCU_baab += 1/6 * einsum('KxaC,yzwa,Wuvw,Uxvzyu->KWCU', X, t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_KWCU_baab += 1/6 * einsum('KxaC,yzwa,Wuyv,Uxwuzv->KWCU', X, t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_KWCU_baab += 1/3 * einsum('KxaC,yzwa,Wuyv,Uxwzuv->KWCU', X, t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_KWCU_baab -= 1/6 * einsum('KxaC,yzwa,Wuzv,Uxwuyv->KWCU', X, t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_KWCU_baab += 1/6 * einsum('KxaC,yzwa,Wuzv,Uxwvuy->KWCU', X, t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_KWCU_baab -= 1/6 * einsum('KxaC,yzwa,Wuzv,Uxwvyu->KWCU', X, t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_KWCU_baab -= 1/6 * einsum('KxaC,yzwa,Wuzv,Uxwyuv->KWCU', X, t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_KWCU_baab -= 1/6 * einsum('KxaC,yzwa,Wuzv,Uxwyvu->KWCU', X, t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_KWCU_baab -= 1/6 * einsum('KxaC,yzwa,Wwuv,Uxuyzv->KWCU', X, t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma_KWCU_baab -= 1/3 * einsum('KxaC,yzwa,Wwuv,Uxuzyv->KWCU', X, t1_aaae, v_aaaa, rdm_cccaaa, optimize = einsum_type)
    sigma[caea__baab] += ascontiguousarray(sigma_KWCU_baab).reshape(-1)

    mr_adc.log.timer_debug("computing sigma H1 h1-h1 CAEA-CAEE", *cput1)

def compute_sigma_vector__H1__h1_h1__CAEA_CAEE__V_AEEE(mr_adc, X, sigma, v_aeee):

    cput1 = (logger.process_clock(), logger.perf_counter())

    # Einsum definition from kernel
    einsum = mr_adc.interface.einsum
    einsum_type = mr_adc.interface.einsum_type

    # Reduced Density Matrices
    rdm_ca = mr_adc.rdm.ca
    rdm_ccaa = mr_adc.rdm.ccaa

    sigma_KWCU_aaaa =- 1/2 * einsum('Kxab,UaCb,Wx->KWCU', X, v_aeee, rdm_ca, optimize = einsum_type)
    sigma_KWCU_aaaa += 1/2 * einsum('Kxab,UbCa,Wx->KWCU', X, v_aeee, rdm_ca, optimize = einsum_type)
    sigma_KWCU_aaaa -= 1/6 * einsum('Kxab,yaCb,UxWy->KWCU', X, v_aeee, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_aaaa += 1/6 * einsum('Kxab,yaCb,UxyW->KWCU', X, v_aeee, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_aaaa += 1/2 * einsum('Kxab,ybCa,UxWy->KWCU', X, v_aeee, rdm_ccaa, optimize = einsum_type)

    sigma_KWCU_abab  = 1/2 * einsum('Kxab,UbCa,Wx->KWCU', X, v_aeee, rdm_ca, optimize = einsum_type)
    sigma_KWCU_abab -= 1/3 * einsum('Kxab,yaCb,UxWy->KWCU', X, v_aeee, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_abab -= 1/6 * einsum('Kxab,yaCb,UxyW->KWCU', X, v_aeee, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_abab += 1/2 * einsum('Kxab,ybCa,UxWy->KWCU', X, v_aeee, rdm_ccaa, optimize = einsum_type)

    sigma_KWCU_baab =- 1/2 * einsum('Kxab,UaCb,Wx->KWCU', X, v_aeee, rdm_ca, optimize = einsum_type)
    sigma_KWCU_baab += 1/6 * einsum('Kxab,yaCb,UxWy->KWCU', X, v_aeee, rdm_ccaa, optimize = einsum_type)
    sigma_KWCU_baab += 1/3 * einsum('Kxab,yaCb,UxyW->KWCU', X, v_aeee, rdm_ccaa, optimize = einsum_type)

    mr_adc.log.timer_debug("contracting v2e.aeee", *cput1)
    return sigma_KWCU_aaaa, sigma_KWCU_abab, sigma_KWCU_baab

# CVAA <- CAEE: NO CONTRIBUTION

# CVEA <- CAEE
def compute_sigma_vector__H1__h1_h1__CVEA_CAEE(mr_adc, X, sigma):
    cput1 = (logger.process_clock(), logger.perf_counter())

    # Einsum definition from kernel
    einsum = mr_adc.interface.einsum
    einsum_type = mr_adc.interface.einsum_type

    #Excitation Manifold
    cvea__abab = mr_adc.h1.cvea__abab
    cvea__baab = mr_adc.h1.cvea__baab
 
    ## Two-electron integrals
    v_vaea = mr_adc.v2e.vaea

    # Reduced Density Matrices
    rdm_ca = mr_adc.rdm.ca
    rdm_ccaa = mr_adc.rdm.ccaa

    sigma_KLCW_abab  = einsum('KxCa,LWay,xy->KLCW', X, v_vaea, rdm_ca, optimize = einsum_type)
    sigma_KLCW_abab -= 1/2 * einsum('KxCa,LyaW,xy->KLCW', X, v_vaea, rdm_ca, optimize = einsum_type)
    sigma_KLCW_abab -= 1/2 * einsum('KxCa,Lyaz,Wxyz->KLCW', X, v_vaea, rdm_ccaa, optimize = einsum_type)
    sigma_KLCW_abab -= 1/2 * einsum('KxaC,LWay,xy->KLCW', X, v_vaea, rdm_ca, optimize = einsum_type)
    sigma_KLCW_abab += 1/3 * einsum('KxaC,Lyaz,Wxyz->KLCW', X, v_vaea, rdm_ccaa, optimize = einsum_type)
    sigma_KLCW_abab += 1/6 * einsum('KxaC,Lyaz,Wxzy->KLCW', X, v_vaea, rdm_ccaa, optimize = einsum_type)
    sigma[cvea__abab] += ascontiguousarray(sigma_KLCW_abab).reshape(-1)

    sigma_KLCW_baab  = 1/2 * einsum('KxaC,LyaW,xy->KLCW', X, v_vaea, rdm_ca, optimize = einsum_type)
    sigma_KLCW_baab -= 1/6 * einsum('KxaC,Lyaz,Wxyz->KLCW', X, v_vaea, rdm_ccaa, optimize = einsum_type)
    sigma_KLCW_baab -= 1/3 * einsum('KxaC,Lyaz,Wxzy->KLCW', X, v_vaea, rdm_ccaa, optimize = einsum_type)
    sigma[cvea__baab] += ascontiguousarray(sigma_KLCW_baab).reshape(-1)

    mr_adc.log.timer_debug("computing sigma H1 h1-h1 CVEA-CAEE", *cput1)

# CVEE <- CAEE
def compute_sigma_vector__H1__h1_h1__CVEE_CAEE(mr_adc, X, sigma):
    cput1 = (logger.process_clock(), logger.perf_counter())

    # Einsum definition from kernel
    einsum = mr_adc.interface.einsum
    einsum_type = mr_adc.interface.einsum_type

    #Excitation Manifold
    cvee = mr_adc.h1.cvee

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

    sigma_KLCD =- einsum('KxCD,Lx->KLCD', X, h_va, optimize = einsum_type)
    sigma_KLCD += einsum('KxCD,L,Lx->KLCD', X, e_val, t1_va, optimize = einsum_type)
    sigma_KLCD -= einsum('KxCD,xy,Ly->KLCD', X, h_aa, t1_va, optimize = einsum_type)
    sigma_KLCD -= einsum('KxCD,Lxyz,yz->KLCD', X, v_vaaa, rdm_ca, optimize = einsum_type)
    sigma_KLCD += 1/2 * einsum('KxCD,Lyzx,zy->KLCD', X, v_vaaa, rdm_ca, optimize = einsum_type)
    sigma_KLCD += einsum('KxCa,LDay,xy->KLCD', X, v_veea, rdm_ca, optimize = einsum_type)
    sigma_KLCD -= 1/2 * einsum('KxCa,LyaD,xy->KLCD', X, v_vaee, rdm_ca, optimize = einsum_type)
    sigma_KLCD -= 1/2 * einsum('KxaC,LDay,xy->KLCD', X, v_veea, rdm_ca, optimize = einsum_type)
    sigma_KLCD -= 1/2 * einsum('KxaD,LyaC,xy->KLCD', X, v_vaee, rdm_ca, optimize = einsum_type)
    sigma_KLCD += 1/2 * einsum('ixCD,KiLy,xy->KLCD', X, v_xxva, rdm_ca, optimize = einsum_type)
    sigma_KLCD += 1/2 * einsum('ixDC,LiKy,xy->KLCD', X, v_vxxa, rdm_ca, optimize = einsum_type)
    sigma_KLCD += einsum('KxCD,L,Lyxz,yz->KLCD', X, e_val, t1_vaaa, rdm_ca, optimize = einsum_type)
    sigma_KLCD -= 1/2 * einsum('KxCD,L,Lyzx,yz->KLCD', X, e_val, t1_vaaa, rdm_ca, optimize = einsum_type)
    sigma_KLCD += 1/2 * einsum('KxCD,xy,Lzwy,zw->KLCD', X, h_aa, t1_vaaa, rdm_ca, optimize = einsum_type)
    sigma_KLCD -= einsum('KxCD,xy,Lzyw,zw->KLCD', X, h_aa, t1_vaaa, rdm_ca, optimize = einsum_type)
    sigma_KLCD -= einsum('KxCD,yz,Lwxy,zw->KLCD', X, h_aa, t1_vaaa, rdm_ca, optimize = einsum_type)
    sigma_KLCD += 1/2 * einsum('KxCD,yz,Lwyx,zw->KLCD', X, h_aa, t1_vaaa, rdm_ca, optimize = einsum_type)
    sigma_KLCD -= 1/2 * einsum('KxCD,yz,Lywx,zw->KLCD', X, h_aa, t1_vaaa, rdm_ca, optimize = einsum_type)
    sigma_KLCD += einsum('KxCD,yz,Lyxw,zw->KLCD', X, h_aa, t1_vaaa, rdm_ca, optimize = einsum_type)
    sigma_KLCD -= einsum('KxCD,Ly,xyzw,zw->KLCD', X, t1_va, v_aaaa, rdm_ca, optimize = einsum_type)
    sigma_KLCD += 1/2 * einsum('KxCD,Ly,xzwy,wz->KLCD', X, t1_va, v_aaaa, rdm_ca, optimize = einsum_type)
    sigma_KLCD += einsum('KxCD,Lyxz,ywuv,zuwv->KLCD', X, t1_vaaa, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KLCD -= einsum('KxCD,Lyxz,zwuv,yuwv->KLCD', X, t1_vaaa, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KLCD += 1/2 * einsum('KxCD,Lyzw,xuvw,yuvz->KLCD', X, t1_vaaa, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KLCD += 1/2 * einsum('KxCD,Lyzw,xuvz,yuwv->KLCD', X, t1_vaaa, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KLCD -= 1/2 * einsum('KxCD,Lyzw,xuyv,zwuv->KLCD', X, t1_vaaa, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KLCD += 1/2 * einsum('KxCD,Lyzw,xwuv,yvzu->KLCD', X, t1_vaaa, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KLCD += 1/2 * einsum('KxCD,Lyzw,xwuz,yu->KLCD', X, t1_vaaa, v_aaaa, rdm_ca, optimize = einsum_type)
    sigma_KLCD -= einsum('KxCD,Lyzw,xzuv,yvwu->KLCD', X, t1_vaaa, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KLCD -= einsum('KxCD,Lyzw,xzuw,yu->KLCD', X, t1_vaaa, v_aaaa, rdm_ca, optimize = einsum_type)
    sigma_KLCD -= 1/2 * einsum('KxCD,Lyzx,ywuv,zuwv->KLCD', X, t1_vaaa, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma_KLCD += 1/2 * einsum('KxCD,Lyzx,zwuv,yuwv->KLCD', X, t1_vaaa, v_aaaa, rdm_ccaa, optimize = einsum_type)
    sigma[cvee] += ascontiguousarray(sigma_KLCD).reshape(-1)

    mr_adc.log.timer_debug("computing sigma H1 h1-h1 CVEE-CAEE", *cput1)

