import prism.lib.logger as logger
import prism.lib.tools as tools
from numpy import ascontiguousarray

from .H1_h1_CCAA__sigma_vector import *
from .H1_h1_CCEA__sigma_vector import *
from .H1_h1_CCEE__sigma_vector import *
from .H1_h1_CAEE__sigma_vector import *

from .H1_h1_CAAA__sigma_vector import *
from .H1_h1_CAEA__sigma_vector import *

from .H1_h1_CVAA__sigma_vector import *
from .H1_h1_CVEA__sigma_vector import *
from .H1_h1_CVEE__sigma_vector import *

class H1SigmaVector:
    pass

## h1 <- CCAA
H1SigmaVector.compute_sigma_vector__H1__h1_h1__CCAA_CCAA = compute_sigma_vector__H1__h1_h1__CCAA_CCAA
H1SigmaVector.compute_sigma_vector__H1__h1_h1__CCEA_CCAA = compute_sigma_vector__H1__h1_h1__CCEA_CCAA
H1SigmaVector.compute_sigma_vector__H1__h1_h1__CCEE_CCAA = compute_sigma_vector__H1__h1_h1__CCEE_CCAA
## CAEE <- CCAA: ZERO COUPLING                                                                       
H1SigmaVector.compute_sigma_vector__H1__h1_h1__CAAA_CCAA = compute_sigma_vector__H1__h1_h1__CAAA_CCAA
H1SigmaVector.compute_sigma_vector__H1__h1_h1__CAEA_CCAA = compute_sigma_vector__H1__h1_h1__CAEA_CCAA
H1SigmaVector.compute_sigma_vector__H1__h1_h1__CVAA_CCAA = compute_sigma_vector__H1__h1_h1__CVAA_CCAA
H1SigmaVector.compute_sigma_vector__H1__h1_h1__CVEA_CCAA = compute_sigma_vector__H1__h1_h1__CVEA_CCAA
## CVEE <- CCAA: ZERO COUPLING

## h1 <- CCEA
H1SigmaVector.compute_sigma_vector__H1__h1_h1__CCAA_CCEA = compute_sigma_vector__H1__h1_h1__CCAA_CCEA
H1SigmaVector.compute_sigma_vector__H1__h1_h1__CCEA_CCEA = compute_sigma_vector__H1__h1_h1__CCEA_CCEA
H1SigmaVector.compute_sigma_vector__H1__h1_h1__CCEE_CCEA = compute_sigma_vector__H1__h1_h1__CCEE_CCEA
H1SigmaVector.compute_sigma_vector__H1__h1_h1__CAEE_CCEA = compute_sigma_vector__H1__h1_h1__CAEE_CCEA
H1SigmaVector.compute_sigma_vector__H1__h1_h1__CAAA_CCEA = compute_sigma_vector__H1__h1_h1__CAAA_CCEA
H1SigmaVector.compute_sigma_vector__H1__h1_h1__CAEA_CCEA = compute_sigma_vector__H1__h1_h1__CAEA_CCEA
H1SigmaVector.compute_sigma_vector__H1__h1_h1__CVAA_CCEA = compute_sigma_vector__H1__h1_h1__CVAA_CCEA
H1SigmaVector.compute_sigma_vector__H1__h1_h1__CVEA_CCEA = compute_sigma_vector__H1__h1_h1__CVEA_CCEA
H1SigmaVector.compute_sigma_vector__H1__h1_h1__CVEE_CCEA = compute_sigma_vector__H1__h1_h1__CVEE_CCEA

## h1 <- CCEE
H1SigmaVector.compute_sigma_vector__H1__h1_h1__CCAA_CCEE = compute_sigma_vector__H1__h1_h1__CCAA_CCEE
H1SigmaVector.compute_sigma_vector__H1__h1_h1__CCEA_CCEE = compute_sigma_vector__H1__h1_h1__CCEA_CCEE
H1SigmaVector.compute_sigma_vector__H1__h1_h1__CCEE_CCEE = compute_sigma_vector__H1__h1_h1__CCEE_CCEE
H1SigmaVector.compute_sigma_vector__H1__h1_h1__CAEE_CCEE = compute_sigma_vector__H1__h1_h1__CAEE_CCEE
H1SigmaVector.compute_sigma_vector__H1__h1_h1__CAAA_CCEE = compute_sigma_vector__H1__h1_h1__CAAA_CCEE
H1SigmaVector.compute_sigma_vector__H1__h1_h1__CAEA_CCEE = compute_sigma_vector__H1__h1_h1__CAEA_CCEE
## CVAA <- CCEE: ZERO COUPLING                                                                       
H1SigmaVector.compute_sigma_vector__H1__h1_h1__CVEA_CCEE = compute_sigma_vector__H1__h1_h1__CVEA_CCEE
H1SigmaVector.compute_sigma_vector__H1__h1_h1__CVEE_CCEE = compute_sigma_vector__H1__h1_h1__CVEE_CCEE

## h1 <- CAEE
## CCAA <- CAEE: ZERO COUPLING
H1SigmaVector.compute_sigma_vector__H1__h1_h1__CCEA_CAEE = compute_sigma_vector__H1__h1_h1__CCEA_CAEE
H1SigmaVector.compute_sigma_vector__H1__h1_h1__CCEE_CAEE = compute_sigma_vector__H1__h1_h1__CCEE_CAEE
H1SigmaVector.compute_sigma_vector__H1__h1_h1__CAEE_CAEE = compute_sigma_vector__H1__h1_h1__CAEE_CAEE
H1SigmaVector.compute_sigma_vector__H1__h1_h1__CAAA_CAEE = compute_sigma_vector__H1__h1_h1__CAAA_CAEE
H1SigmaVector.compute_sigma_vector__H1__h1_h1__CAEA_CAEE = compute_sigma_vector__H1__h1_h1__CAEA_CAEE
## CVAA <- CAEE: ZERO COUPLING                                                                       
H1SigmaVector.compute_sigma_vector__H1__h1_h1__CVEA_CAEE = compute_sigma_vector__H1__h1_h1__CVEA_CAEE
H1SigmaVector.compute_sigma_vector__H1__h1_h1__CVEE_CAEE = compute_sigma_vector__H1__h1_h1__CVEE_CAEE

## h1 <- CAAA
H1SigmaVector.compute_sigma_vector__H1__h1_h1__CCAA_CAAA = compute_sigma_vector__H1__h1_h1__CCAA_CAAA
H1SigmaVector.compute_sigma_vector__H1__h1_h1__CCEA_CAAA = compute_sigma_vector__H1__h1_h1__CCEA_CAAA
H1SigmaVector.compute_sigma_vector__H1__h1_h1__CCEE_CAAA = compute_sigma_vector__H1__h1_h1__CCEE_CAAA
H1SigmaVector.compute_sigma_vector__H1__h1_h1__CAEE_CAAA = compute_sigma_vector__H1__h1_h1__CAEE_CAAA
H1SigmaVector.compute_sigma_vector__H1__h1_h1__CAAA_CAAA = compute_sigma_vector__H1__h1_h1__CAAA_CAAA
H1SigmaVector.compute_sigma_vector__H1__h1_h1__CAEA_CAAA = compute_sigma_vector__H1__h1_h1__CAEA_CAAA
H1SigmaVector.compute_sigma_vector__H1__h1_h1__CVAA_CAAA = compute_sigma_vector__H1__h1_h1__CVAA_CAAA
H1SigmaVector.compute_sigma_vector__H1__h1_h1__CVEA_CAAA = compute_sigma_vector__H1__h1_h1__CVEA_CAAA
H1SigmaVector.compute_sigma_vector__H1__h1_h1__CVEE_CAAA = compute_sigma_vector__H1__h1_h1__CVEE_CAAA

## h1 <- CAEA
H1SigmaVector.compute_sigma_vector__H1__h1_h1__CCAA_CAEA = compute_sigma_vector__H1__h1_h1__CCAA_CAEA
H1SigmaVector.compute_sigma_vector__H1__h1_h1__CCEA_CAEA = compute_sigma_vector__H1__h1_h1__CCEA_CAEA
H1SigmaVector.compute_sigma_vector__H1__h1_h1__CCEE_CAEA = compute_sigma_vector__H1__h1_h1__CCEE_CAEA
H1SigmaVector.compute_sigma_vector__H1__h1_h1__CAEE_CAEA = compute_sigma_vector__H1__h1_h1__CAEE_CAEA
H1SigmaVector.compute_sigma_vector__H1__h1_h1__CAAA_CAEA = compute_sigma_vector__H1__h1_h1__CAAA_CAEA
H1SigmaVector.compute_sigma_vector__H1__h1_h1__CAEA_CAEA = compute_sigma_vector__H1__h1_h1__CAEA_CAEA
H1SigmaVector.compute_sigma_vector__H1__h1_h1__CVAA_CAEA = compute_sigma_vector__H1__h1_h1__CVAA_CAEA
H1SigmaVector.compute_sigma_vector__H1__h1_h1__CVEA_CAEA = compute_sigma_vector__H1__h1_h1__CVEA_CAEA
H1SigmaVector.compute_sigma_vector__H1__h1_h1__CVEE_CAEA = compute_sigma_vector__H1__h1_h1__CVEE_CAEA

H1SigmaVector.compute_sigma_vector__H1__h1_h1__CCEE_CAEA__V_XEEE = compute_sigma_vector__H1__h1_h1__CCEE_CAEA__V_XEEE
H1SigmaVector.compute_sigma_vector__H1__h1_h1__CAEE_CAEA__V_AEEE = compute_sigma_vector__H1__h1_h1__CAEE_CAEA__V_AEEE
H1SigmaVector.compute_sigma_vector__H1__h1_h1__CVEE_CAEA__V_VEEE = compute_sigma_vector__H1__h1_h1__CVEE_CAEA__V_VEEE

## h1 <- CVAA
H1SigmaVector.compute_sigma_vector__H1__h1_h1__CCAA_CVAA = compute_sigma_vector__H1__h1_h1__CCAA_CVAA
H1SigmaVector.compute_sigma_vector__H1__h1_h1__CCEA_CVAA = compute_sigma_vector__H1__h1_h1__CCEA_CVAA
## CCEE <- CVAA: ZERO COUPLING                                                                       
H1SigmaVector.compute_sigma_vector__H1__h1_h1__CAAA_CVAA = compute_sigma_vector__H1__h1_h1__CAAA_CVAA
H1SigmaVector.compute_sigma_vector__H1__h1_h1__CAEA_CVAA = compute_sigma_vector__H1__h1_h1__CAEA_CVAA
## CAEE <- CVAA: ZERO COUPLING                                                                       
H1SigmaVector.compute_sigma_vector__H1__h1_h1__CVAA_CVAA = compute_sigma_vector__H1__h1_h1__CVAA_CVAA
H1SigmaVector.compute_sigma_vector__H1__h1_h1__CVEA_CVAA = compute_sigma_vector__H1__h1_h1__CVEA_CVAA
H1SigmaVector.compute_sigma_vector__H1__h1_h1__CVEE_CVAA = compute_sigma_vector__H1__h1_h1__CVEE_CVAA

## h1 <- CVEE
## CCAA <- CVEE: ZERO COUPLING
H1SigmaVector.compute_sigma_vector__H1__h1_h1__CCEA_CVEE = compute_sigma_vector__H1__h1_h1__CCEA_CVEE
H1SigmaVector.compute_sigma_vector__H1__h1_h1__CCEE_CVEE = compute_sigma_vector__H1__h1_h1__CCEE_CVEE
H1SigmaVector.compute_sigma_vector__H1__h1_h1__CAEE_CVEE = compute_sigma_vector__H1__h1_h1__CAEE_CVEE
H1SigmaVector.compute_sigma_vector__H1__h1_h1__CAAA_CVEE = compute_sigma_vector__H1__h1_h1__CAAA_CVEE
H1SigmaVector.compute_sigma_vector__H1__h1_h1__CAEA_CVEE = compute_sigma_vector__H1__h1_h1__CAEA_CVEE
H1SigmaVector.compute_sigma_vector__H1__h1_h1__CVAA_CVEE = compute_sigma_vector__H1__h1_h1__CVAA_CVEE
H1SigmaVector.compute_sigma_vector__H1__h1_h1__CVEA_CVEE = compute_sigma_vector__H1__h1_h1__CVEA_CVEE
H1SigmaVector.compute_sigma_vector__H1__h1_h1__CVEE_CVEE = compute_sigma_vector__H1__h1_h1__CVEE_CVEE

## h1 <- CVEA
H1SigmaVector.compute_sigma_vector__H1__h1_h1__CCAA_CVEA = compute_sigma_vector__H1__h1_h1__CCAA_CVEA
H1SigmaVector.compute_sigma_vector__H1__h1_h1__CCEA_CVEA = compute_sigma_vector__H1__h1_h1__CCEA_CVEA
H1SigmaVector.compute_sigma_vector__H1__h1_h1__CCEE_CVEA = compute_sigma_vector__H1__h1_h1__CCEE_CVEA
H1SigmaVector.compute_sigma_vector__H1__h1_h1__CAEE_CVEA = compute_sigma_vector__H1__h1_h1__CAEE_CVEA
H1SigmaVector.compute_sigma_vector__H1__h1_h1__CAAA_CVEA = compute_sigma_vector__H1__h1_h1__CAAA_CVEA
H1SigmaVector.compute_sigma_vector__H1__h1_h1__CAEA_CVEA = compute_sigma_vector__H1__h1_h1__CAEA_CVEA
H1SigmaVector.compute_sigma_vector__H1__h1_h1__CVAA_CVEA = compute_sigma_vector__H1__h1_h1__CVAA_CVEA
H1SigmaVector.compute_sigma_vector__H1__h1_h1__CVEA_CVEA = compute_sigma_vector__H1__h1_h1__CVEA_CVEA
H1SigmaVector.compute_sigma_vector__H1__h1_h1__CVEE_CVEA = compute_sigma_vector__H1__h1_h1__CVEE_CVEA

sigma_vector = H1SigmaVector()

