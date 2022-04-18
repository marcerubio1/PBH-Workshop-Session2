'''
This module includes the different mass functions for the Fixed Conformal Time scenario. Also includes the transformation between dc and Ms for the different Power Spectrums Used.

'''

import numpy as np 
from constants import A0,k0
from k_to_mass import k_to_m


def dndM_kns_dc(u,a,cosmo, a_fct = 1e-25, dc = 0.6, ns = 0.9649, gamma = 1., ffudge = 1., iscomoving = True, dndMlog = False):

    '''
    Mass function for Classical Power Spectrum in terms of delta collapse (dc) and for a Fixed Conformal Time PBH formation.

    It recieves the log10(M) as main argument.
    
    Can be asked for the Mass Function in terms of dlog(M) using the argument dndMlog = True. Useful for integration
    
    '''

    M = np.float_power(10.,u)

    As = A0 # Amplitude of PPS from Planck 2018

    A = As / np.power(k0, ns) # Normalized by the scale measured by Planck 2018

    D = np.power(a_fct,2) # Growth Function in radiation (It is suposedly good)

    A = A * np.power(D,2)
    
    rho_fct = ffudge*(cosmo.Om0/a_fct**2 + cosmo.Or0/a_fct**4)*cosmo.rhoc
    
    C_fct = a_fct * np.float_power(((32 * np.pi**4 * rho_fct)/3),1./3.)

    rho_pbh = (gamma * cosmo.Odm0 * cosmo.rhoc)

    f1 = (dc * np.float_power(ns+3, 3./2. ) ) / (6 * np.sqrt( 2 * np.pi**2 * A * np.float_power(C_fct,ns+3) ) ) 

    f2 = np.float_power(M, (ns-9)/6. )

    exp1 = np.power(dc,2.) * (ns + 3) * np.float_power(M , (ns+3)/3 )
    
    exp2 = 8 * np.pi * A * np.float_power(C_fct, ns+3)

    f3 = np.exp(-exp1/exp2)

    if dndMlog == True:

        res = rho_pbh * np.log(10) * M * f1 * f2 * f3

    else:

        res = rho_pbh * f1 * f2 * f3

    if iscomoving == False:

        res = np.float_power(a,-3.) * res

    return res

    
def dndM_kns_Ms(u, a, cosmo, a_fct = 1e-25, M_star = 1e-1, ns = 0.9649, gamma = 1., ffudge = 1., iscomoving = True, dndMlog = False):

    '''
    Mass function for Classical Power Spectrum in terms of M_* (M_star) and for a Fixed Conformal Time PBH formation.

    It recieves the log10(M) as main argument.
    
    Can be asked for the Mass Function in terms of dlog(M) using the argument dndMlog = True. Useful for integration
    
    '''

    M = np.power(10.,u)

    rho_pbh = (gamma * cosmo.Odm0 * cosmo.rhoc)

    f1 = (2/np.sqrt(2*np.pi)) * ((ns + 3)/6)

    f2 = np.float_power(M, -2) * np.float_power(M/M_star, (ns+3)/6 )

    f3 = np.exp(-(1./2.) * np.float_power(M/M_star, (ns+3)/3 ))


    if dndMlog == True:

        res = rho_pbh * np.log(10) * M * f1 * f2 * f3

    else:

        res = rho_pbh * f1 * f2 * f3

    if iscomoving == False:

        res = np.float_power(a,-3.) * res

    return res

def Ms_to_dc_kns(M_star, cosmo, a_fct = 1e-25, ns = 0.9649, f_fudge = 1.):

    '''
    dc from Mstar for a Classical Power Spectrum
    
    '''
    
    As = A0 # Amplitude of PPS from Planck 2018

    A = As / np.power(k0, ns) # Normalized by the scale measured by Planck 2018

    D = np.power(a_fct,2) # Growth Function in radiation (It is suposedly good)

    A = A * np.power(D,2)
    
    rho_fct = f_fudge*(cosmo.Om0/a_fct**2 + cosmo.Or0/a_fct**4)*cosmo.rhoc
    
    C_fct = a_fct * np.power(((32 * np.pi**4 * rho_fct)/3),1./3.)
    
    num = np.sqrt(4*np.pi * (A) * np.power(C_fct, ns+3))
    
    den = np.power(M_star,(ns+3)/6) * np.sqrt(ns + 3)
    
    return num/den

def dc_to_Ms_kns(dc ,cosmo, a_fct = 1e-25, ns = 0.9649, f_fudge = 1.):

    '''
    Ms from dc for a Classical Power Spectrum
    
    '''
    
    As = A0 # Amplitude of PPS from Planck 2018

    A = As / np.power(k0, ns) # Normalized by the scale measured by Planck 2018

    D = np.power(a_fct,2) # Growth Function in radiation (It is suposedly good)

    A = A * np.power(D,2)

    rho_fct = f_fudge*(cosmo.Om0/a_fct**2 + cosmo.Or0/a_fct**4)*cosmo.rhoc
    
    C_fct = a_fct * np.power(((32 * np.pi**4 * rho_fct)/3),1./3.)
    
    num = np.sqrt(4*np.pi * (A) * np.power(C_fct, ns+3))
    
    den = dc * np.sqrt(ns + 3)
    
    return np.float_power(num/den, 6./(ns+3))

def dndM_brk_dc(u,a,cosmo,a_fct = 1e-25, dc=0.6, k_piv = 10., ns=0.9649, nb=2., gamma=1.,ffudge = 1., iscomoving = True, dndMlog = False):

    '''
    Mass function for Broken Power Spectrum in terms of delta collapse (dc) and for a Fixed Conformal Time PBH formation.

    It recieves the log10(M) as main argument.
    
    Can be asked for the Mass Function in terms of dlog(M) using the argument dndMlog = True. Useful for integration
    
    '''
    M = np.float_power(10.,u)

    M_piv = k_to_m(k_piv,cosmo,a_fct)

    As = A0 # Amplitude of PPS from Planck 2018

    A = As / np.power(k0, ns) # Normalized by the scale measured by Planck 2018

    D = np.power(a_fct,2) # Growth Function in radiation (It is suposedly good)

    A = A * np.power(D,2)
    
    rho_fct = ffudge*(cosmo.Om0/a_fct**2 + cosmo.Or0/a_fct**4)*cosmo.rhoc

    C_fct = a_fct * np.power(((32 * np.pi**4 * rho_fct)/3),1./3.)

    Apiv = (4 * np.pi * A * np.power(C_fct, nb+3) * np.float_power(k_piv,ns-nb)) / ((ns+3) * (nb+3))

    rho_pbh = (gamma * cosmo.Odm0 * cosmo.rhoc)

    C1 = (nb - ns) * np.power(M_piv, -(nb+3)/3)
    C2 = ns + 3

    f1 = (dc/np.sqrt(2*np.pi)) * Apiv * C2 * ((nb+3)/3)

    f2 = np.float_power(M, -(nb+9)/3) * np.float_power(Apiv * (C1 + C2 * np.float_power(M, -(nb+3)/3) ), -3./2.)

    f3 = np.exp(-(np.power(dc,2.))/(2 * Apiv * (C1 + C2 * np.float_power(M, -(nb+3)/3) )))

    if dndMlog == True:

        res = rho_pbh * np.log(10) * M * f1 * f2 * f3

    else:

        res = rho_pbh * f1 * f2 * f3

    if iscomoving == False:

        res = np.float_power(a,-3.) * res

    return res

def dndM_brk_Ms(u, a, cosmo,a_fct = 1e-25, M_star=1e-1, k_piv = 10., ns=0.9649, nb=2., gamma=1.,fm = 1., iscomoving = True, dndMlog = False):

    '''
    Mass function for Broken Power Spectrum in terms of M_* (M_star) and for a Fixed Conformal Time PBH formation.

    It recieves the log10(M) as main argument.
    
    Can be asked for the Mass Function in terms of dlog(M) using the argument dndMlog = True. Useful for integration
    
    '''

    M = np.float_power(10.,u)

    M_piv = k_to_m(cosmo,k_piv,fm)

    rho_pbh = (gamma * cosmo.Odm0 * cosmo.rhoc)

    C1 = (nb - ns) * np.power(M_piv, -(nb+3)/3)
    C2 = ns + 3

    def F(M):

        return C1 + C2 * np.power(M, -(3+nb)/3)

    f1 = ( C2 * (3 + nb) * np.float_power(1/M,(9+nb)/3) )/ (3*np.sqrt(2*np.pi))

    Fstar = F(M_star)
    Fval = F(M)

    f2 = np.sqrt(Fstar)/np.float_power(Fval,3./2)

    f3 = np.exp(-Fstar/(2*Fval))

    if dndMlog == True:

        res = rho_pbh * np.log(10) * M * f1 * f2 * f3

    else:

        res = rho_pbh * f1 * f2 * f3

    if iscomoving == False:

        res = np.float_power(a,-3.) * res

    return res

def Ms_to_dc_brk(M_star, cosmo, a_fct = 1e-25 ,ns = 0.9649,nb = 2. ,kpiv = 10. ,f_fudge = 1.):

    '''
    
    dc from Ms for a Broken Power Spectrum
    
    '''

    rho = f_fudge*((cosmo.Om0/a_fct**3) + (cosmo.Or0/a_fct**4))*cosmo.rhoc
    
    As = A0 # Amplitude of PPS from Planck 2018

    A = As / np.power(k0, ns) # Normalized by the scale measured by Planck 2018

    D = np.power(a_fct,2) # Growth Function in radiation (It is suposedly good)

    A = A * np.power(D,2)
    
    alpha = (nb + 3)/3
    
    Ck = np.float_power(32 * np.pi**4 * a_fct**3 * rho / 3. , 1./3.)
    
    S1 = (nb - ns) * np.float_power(Ck/kpiv, -3*alpha)
    
    S2 = (ns + 3)
    
    Apiv = (4 * np.pi * A * np.float_power(Ck, 3*alpha) * np.float_power(kpiv, ns-nb))/(S2 * (nb + 3))
    
    dc = np.sqrt(Apiv*(S1+S2*np.power(M_star,-(nb+3)/3)))
    
    return dc

def dc_to_Ms_brk(dc, cosmo, a_fct = 1e-25 ,ns = 0.9649,nb = 2. ,kpiv = 10. ,f_fudge = 1.):

    '''
    
    Ms from dc for a Broken Power Spectrum
    
    '''
    
    rho = f_fudge*((cosmo.Om0/a_fct**3) + (cosmo.Or0/a_fct**4))*cosmo.rhoc
    
    As = A0 # Amplitude of PPS from Planck 2018

    A = As / np.power(k0, ns) # Normalized by the scale measured by Planck 2018

    D = np.power(a_fct,2) # Growth Function in radiation (It is suposedly good)

    A = A * np.power(D,2)
    
    alpha = (nb + 3)/3
    
    Ck = np.float_power(32 * np.pi**4 * a_fct**3 * rho / 3. , 1./3.)
    
    S1 = (nb - ns) * np.float_power(Ck/kpiv, -3*alpha)
    
    S2 = (ns + 3)
    
    Apiv = (4 * np.pi * A * np.float_power(Ck, 3*alpha) * np.float_power(kpiv, ns-nb))/(S2 * (nb + 3))
    
    left = (dc**2) / (Apiv * S2)

    right = S1 / S2    

    return  np.float_power(left-right, -3/(nb+3))

def epiv(LogMpiv, M_star=1e-1, ns=0.9649, nb=2.):

    piv_s = 10.**(LogMpiv)/M_star

    K1 = np.sqrt(nb+3) * (piv_s)**((ns-nb)/6)

    K2 = np.sqrt((nb-ns) * piv_s**(-(nb+3)/3) + (ns+3))

    EXP = np.exp((1/2)*(-piv_s**((ns+3)/3) + ((ns+3)/(nb+3)) * piv_s**((nb+3)/3) + (nb-ns)/(nb+3) ) )

    return (K1/K2) * EXP

def dndm(u, a, cosmo, M_star=1e-1, k_piv = 10., ns=0.9649, nb=2.,fm = 1., iscomoving = True, dndMlog = False):

    '''
    Mass function for the different scenarios, where M > Mpiv or M < Mpiv, which will correspond to a 
    Standard or a Broken PPS respectively.
    '''

    a_fct = cosmo.eof

    logMpiv = np.log10(Mpiv(cosmo,fm,k_piv))

    #print(u,logMpiv,C_FCT(cosmo,fm))

    if u < logMpiv: # Broken

        result = dndM_brk_Ms(u, a, cosmo, a_fct, M_star, k_piv, ns, nb,fm=fm, iscomoving = iscomoving, dndMlog = dndMlog)

    elif u >= logMpiv: # Standard

        result = dndM_kns_Ms(u, a, cosmo, a_fct, M_star, ns, iscomoving=iscomoving, dndMlog=dndMlog)#* (1/epiv(logMpiv,M_star,ns,nb))

    return result



def C_FCT(cosmo,fm):

    rho = (cosmo.Om0/cosmo.eof**3 + cosmo.Or0/cosmo.eof**4)*cosmo.rhoc 

    constants = (32/3)*np.pi**4

    return cosmo.eof*(rho*constants*fm)**(1/3)

def Mpiv(cosmo, fm, kpiv = 10.):

    Cfct = C_FCT(cosmo,fm)

    return (Cfct/kpiv)**3

def fm_std(cosmo, dc = 0.6, Ms = 1e1, ns = 0.9649):

    rho = (cosmo.Om0/cosmo.eof**3 + cosmo.Or0/cosmo.eof**4)*cosmo.rhoc


    C1 = (3 * Ms)/(32 * np.pi**4 * rho)

    C2 = ((ns + 3) * dc**2) / (4 * np.pi * A0/k0**ns)

    exponent = 3/(ns+3)

    return cosmo.eof**(-3) * C1 * C2**(exponent)

def fm_brk(cosmo, dc = 0.6, Ms = 1e1, ns = 0.9649, nb = 2., kpiv = 10.):

    rho = (cosmo.Om0/cosmo.eof**3 + cosmo.Or0/cosmo.eof**4)*cosmo.rhoc

    C1 = (3 * Ms)/(32 * np.pi**4 * rho)

    C2a = (3 * Ms)/(32 * np.pi**4 * rho * kpiv**(ns-nb))

    C2b = (nb-ns)/(ns+3) * kpiv**(nb+3)

    exponent = 3/(ns+3)

    return cosmo.eof**(-3) * C1 * (C2a+C2b)**(exponent)