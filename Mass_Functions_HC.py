'''
This module includes the different mass functions for the Horizon Crossing scenario. Also includes the transformation between dc and Ms for the different Power Spectrums Used.

'''

import numpy as np
from scipy.interpolate import interp1d
from constants import A0, k0, km_to_Mpc, G, c


def dndM_kns_dc(u,a,cosmo,dc=0.6,ns=0.9649,gamma=1.,ffudge=1., iscomoving = True, dndMlog = False):

    '''
    Mass function for Classical Power Spectrum in terms of delta collapse (dc) and for a Horizon Crossing PBH formation.

    It recieves the log10(M) as main argument.
    
    Can be asked for the Mass Function in terms of dlog(M) using the argument dndMlog = True. Useful for integration
    
    '''

    A=A0/np.power(k0,ns)
    M=np.power(10.,u)

    H0 = cosmo.h*100*km_to_Mpc # In s**-1

    rho_pbh = (gamma * cosmo.Odm0 * cosmo.rhoc)

    B1 = np.sqrt(2*H0*G*np.sqrt(cosmo.Or0)/(ffudge * c**3))
    B2 = (np.pi * c**2 * ffudge / G) * B1

    C_hc = ( (4 * np.pi * A)/(ns+3) ) * B1**4 * np.float_power(B2**2, (ns+3)/2 )

    f1 = dc/ (np.sqrt(2*np.pi * C_hc))

    f2 = ((ns - 1) / 2.) * np.float_power(M, (ns-9)/4)

    f3 = np.exp(-(np.power(dc,2.))/(2*C_hc*np.float_power(M,(1-ns)/2)))

    if dndMlog == True:

        res = rho_pbh * np.log(10) * M * f1 * f2 * f3

    else:

        res = rho_pbh * f1 * f2 * f3

    if iscomoving == False:

        res = np.float_power(a,-3.) * res

    return res


def dndM_kns_Ms(u,a,cosmo,M_star=1e-1,ns=0.9649,gamma=1.,ffudge=1., iscomoving = True, dndMlog = False):

    '''
    Mass function for Classical Power Spectrum in terms of M* (M_star) and for a Horizon Crossing PBH formation.

    It recieves the log10(M) as main argument.
    
    Can be asked for the Mass Function in terms of dlog(M) using the argument dndMlog = True. Useful for integration
    
    '''

    M=np.power(10.,u)

    rho_pbh = (gamma * cosmo.Odm0 * cosmo.rhoc)

    f1 = (ns - 1)/ (2 * np.sqrt(2 * np.pi))

    f2 = np.float_power(M,-2) * np.float_power(M_star/M, (1-ns)/4)

    f3 = np.exp(-(1./2.) * np.float_power(M_star/M, (1-ns)/2))

    if dndMlog == True:

        res = rho_pbh * np.log(10) * M * f1 * f2 * f3

    else:

        res = rho_pbh * f1 * f2 * f3

    if iscomoving == False:

        res = np.float_power(a,-3.) * res

    return res



def Ms_to_dc_kns(M_star,cosmo,ns=2.,ffudge=1):

    '''
    dc from Mstar for a Classical Power Spectrum at Horizon Crossing
    
    '''

    A=A0/np.power(k0,ns)

    H0 = cosmo.h*100*km_to_Mpc # In s**-1

    c0 = 4. * np.pi * A / (ns+3)

    c1=2.*H0*np.sqrt(cosmo.Or0)*G/(c*c*c*ffudge)
    
    c2=c1*c1

    c3 = (2. * H0 * np.sqrt(cosmo.Or0) * np.pi**2 * c * ffudge) / G


    c4 = np.power(c3,(ns+3.)/2)

    Chc = c0 * c2 * c4
    delta = np.sqrt(Chc) * np.power(M_star, (1.-ns)/4.)

    return delta

def dc_to_Ms_kns(dc,cosmo,ns=2,ffudge=1):

    '''
    
    Ms from dc for a Classical Power Spectrum at Horizon Crossing
    
    '''

    A=A0/np.power(k0,ns)


    H0 = cosmo.h*100*km_to_Mpc # In s**-1

    c0=4.*np.pi*A/(ns+3)
    c1=2.*H0*np.sqrt(cosmo.Or0)*G/(c*c*c*ffudge)
    c2=c1*c1
    c3=(2.*H0*np.sqrt(cosmo.Or0)*np.pi*np.pi*c*ffudge)/G
    c4=pow(c3,(ns+3.)/2)
    Chc=c0*c2*c4
    x=dc/np.sqrt(Chc)
    mstar=pow(x,4/(1.-ns))
    return mstar



def dndM_brk_dc(u,a,cosmo,dc=0.6, k_piv = 10., ns=0.9649, nb=2., gamma=1.,ffudge = 1., iscomoving = True, dndMlog = False):

    '''
    Mass function for Broken Power Spectrum in terms of delta collapse (dc) and for a Horizon Crossing PBH formation.

    It recieves the log10(M) as main argument.
    
    Can be asked for the Mass Function in terms of dlog(M) using the argument dndMlog = True. Useful for integration
    
    '''

    A=A0/np.power(k0,ns)
    M=np.float_power(10,u)

    H0 = cosmo.h*100*km_to_Mpc # In s**-1

    rho_pbh = (gamma * cosmo.Odm0 * cosmo.rhoc)


    B1 = np.sqrt(2*H0*G*np.sqrt(cosmo.Or0)/(ffudge * c**3))

    Mpiv = np.float_power((k_piv / B1) * (G/(np.pi * c**2 * ffudge)) ,-2)

    #print('Mpiv-joaquin = %e'%Mpiv)

    S1 = (nb-ns) * np.power(Mpiv, -(nb+3)/2 )

    S2 = ns+3
    
    Apiv = (4 * np.pi * A) * np.float_power((np.pi * (c**2/G) * ffudge),ns+3) * ( (np.float_power(B1,ns+7) * np.float_power(Mpiv,(nb-ns)/2)) / (S2 * (nb+3)) )

    f1 = dc/np.sqrt(2 * np.pi * Apiv)

    f2 = (((nb-1)/2) * S2 * np.float_power(M, -(nb+3)/2) - 2 * S1) / np.float_power(S1 * M**2 + S2 * np.float_power(M, (1-nb)/2), 3./2.)

    #print('Apiv_Joaquin = %e'%Apiv)


    f3 = np.exp(-(dc**2/(2 * Apiv * (S1 * M**2 + S2 * np.float_power(M, (1-nb)/2)))))

    #print('f1 = %e  ;  f2 = %e  ;  f3 = %e'%(f1,f2,f3))

    if dndMlog == True:

        res = rho_pbh * np.log(10) * M * f1 * f2 * f3

    else:

        res = rho_pbh * f1 * f2 * f3

    if iscomoving == False:

        res = np.float_power(a,-3.) * res

    return res


def dndM_brk_Ms(u,a,cosmo, M_star=1e-1, k_piv = 10., ns=0.9649, nb=2., gamma=1.,fm = 1., iscomoving = True, dndMlog = False):

    '''
    Mass function for Broken Power Spectrum in terms of M* (M_star) and for a Horizon Crossing PBH formation.

    It recieves the log10(M) as main argument.
    
    Can be asked for the Mass Function in terms of dlog(M) using the argument dndMlog = True. Useful for integration
    
    '''
    M=np.float_power(10.,u)

    H0 = cosmo.h*100*km_to_Mpc # In s**-1
 
    rho_pbh = (gamma * cosmo.Odm0 * cosmo.rhoc)

    B1 = np.sqrt(2*H0*G*np.sqrt(cosmo.Or0)/(fm * c**3)) # se corrigio parentesis 

    Mpiv = np.power((k_piv / B1) * (G/(np.pi * c**2 * fm)) ,-2)

    #print('Mpiv = %e'%Mpiv)

    S1 = (nb-ns) * np.power(Mpiv, -(nb+3)/2 )

    S2 = ns+3

    def F(M):

        return S1 * M**2 + S2 * np.power(M, (1-nb)/2)

    Fstar = F(M_star)
    Fval = F(M)

    f1 = (1./np.sqrt(2*np.pi)) * (((nb-1)/2) * S2 * np.float_power(M, -(nb+3)/2) - 2 * S1)

    f2 = np.sqrt(Fstar)/np.float_power(Fval,3./2)

    f3 = np.exp(-Fstar/(2*Fval))

    #print('f1*f2 = %f'%(f1*f2))

    #print('exponential = %f'%(-Fstar/(2*Fval)))

    #print('rhopbh = %f'%rho_pbh)

    if dndMlog == True:

        res = rho_pbh * np.log(10) * M * f1 * f2 * f3

    else:

        res = rho_pbh * f1 * f2 * f3

    if iscomoving == False:

        res = np.float_power(a,-3.) * res

    return res

def dndm(u, a, cosmo, M_star=1e-1, k_piv = 10., ns=0.9649, nb=2.,fm = 1., iscomoving = True, dndMlog = False):

    '''
    Mass function for the different scenarios, where M > Mpiv or M < Mpiv, which will correspond to a 
    Standard or a Broken PPS respectively.
    '''

    logMpiv = np.log10(Mpiv(cosmo,fm,k_piv))

    logMeq = np.log10(cosmo.Meq*fm) # Mass of PBH at zeq

    #print(u,logMpiv,C_HC(cosmo,fm),logMeq)

    if (u <= logMpiv and u <= logMeq): # Broken

        #print('standard')

        result = dndM_brk_Ms(u, a, cosmo, M_star, k_piv, ns, nb,fm=fm, iscomoving = iscomoving, dndMlog = dndMlog)

    elif (u > logMpiv or u > logMeq): # Standard o M > Meq

        result = 0.

    else:

        print('Non valid value. Returning nan')

        result = np.nan

    return result

    
def dc_to_Ms_brk(dc,cosmo,ns=0.9649,nb=2.,kpiv=10.,ffudge=1):


    '''
    
    Ms from dc for a Broken Power Spectrum at Horizon Crossing
    
    '''

    A=A0/np.power(k0,ns)

    H0 = cosmo.h*100*km_to_Mpc # In s**-1

    delta2 = dc**2

    alfa=(1.-nb)/2.
    beta=-(nb+3.)/2.0
    c0=4.*np.pi*A
    c1=2.*H0*np.sqrt(cosmo.Or0)*G/(c*c*c*ffudge) #B1^2
    c2=c1*c1 #B1^4
    c3=(2.*H0*np.sqrt(cosmo.Or0)*np.pi*np.pi*c*ffudge)/G #B2^2
    c4=pow(c3,(ns+3.)/2.)#(B_2^2)^{ns+3}
    nsnb3=(ns+3.)*(nb+3.)
    Mpiv=c3/(kpiv*kpiv)
    Apiv=(c0*c2*c4*pow(Mpiv,(nb-ns)/2.))/nsnb3
    s1=(nb-ns)*pow(Mpiv,beta)
    s2=(ns+3.)
    rhs=delta2/Apiv
    # root solver
    m_min_i=-45
    m_max_i=np.log10(10.*Mpiv)
    m_inter=np.logspace(m_min_i,m_max_i,200)
    y=[]
    y = [abs(s1*vm*vm+s2*pow(vm,alfa)-rhs) for vm in m_inter]
    minimo=min(y)
    minimo_index=np.argmin(y)
    for i in range(50):

        try:

            m_inter=np.logspace(np.log10(m_inter[minimo_index-1]),np.log10(m_inter[minimo_index+1]),200)
            y=[]
            y = [abs(s1*vm*vm+s2*pow(vm,alfa)-rhs) for vm in m_inter]
            minimo=min(y)
            minimo_index=np.argmin(y)

        except IndexError:

            print('M_* lower than Planck Mass!')

            return np.nan

        if minimo<1e-20:
            break
    return m_inter[minimo_index]


def Ms_to_dc_brk(M_star,cosmo,ns=0.9649,nb=2.,kpiv=10.,ffudge=1):

    '''
    dc from Mstar for a Broken Power Spectrum at Horizon Crossing
    
    '''

    A=A0/np.power(k0,ns)

    H0 = cosmo.h*100*km_to_Mpc # In s**-1

    alfa=(1.-nb)/2.
    beta=-(nb+3.)/2.0
    c0=4.*np.pi*A
    c1=2.*H0*np.sqrt(cosmo.Or0)*G/(c*c*c*ffudge) #B1^2
    c2=c1*c1 #B1^4
    c3=(2.*H0*np.sqrt(cosmo.Or0)*np.pi*np.pi*c*ffudge)/G #B2^2
    c4=pow(c3,(ns+3.)/2.)#(B_2^2)^{ns+3}
    nsnb3=(ns+3.)*(nb+3.)
    Mpiv=c3/(kpiv*kpiv)
    Apiv=(c0*c2*c4*pow(Mpiv,(nb-ns)/2.))/nsnb3
    s1=(nb-ns)*pow(Mpiv,beta)
    s2=(ns+3.)
    cstar=s1*M_star*M_star+s2*pow(M_star,alfa)
    delta=np.sqrt(Apiv*cstar)
    return delta

def C_HC(cosmo,fm):

    H0 = cosmo.h*100*km_to_Mpc # In s**-1

    constants = np.pi * np.sqrt(2 * c * np.sqrt(cosmo.Or0) * H0 / G)

    return constants * np.sqrt(fm)

def Mpiv(cosmo, fm, kpiv = 10.):

    Chc = C_HC(cosmo,fm)

    return (Chc/kpiv)**2

def fm_std(cosmo, dc = 0.6, Ms = 1e1, ns = 0.9649):

    C1 = (np.sqrt(4 * np.pi * (A0 /np.power(k0, ns)))/(dc * np.sqrt(ns + 3))) * (G/(np.pi * c**2))**2

    C2 = np.pi * np.sqrt(2 * c * np.sqrt(cosmo.Or0) * H0 / G)

    exp1 = (ns + 7)/2

    exp2 = 4 / (1 - ns)

    return Ms * (C1 * C2**(exp1))**(exp2)

def alpha1(cosmo, ns = 0.9649, nb = 2.,kpiv=10.):

    H0 = cosmo.h*100*km_to_Mpc # In s**-1

    C1a = (4 * np.pi * (A0 /np.power(k0, ns)) / ((ns+3) * (nb+3))) * (G/(np.pi * c**2))**4 * kpiv**(ns-nb)

    C1b = np.pi * np.sqrt(2 * c * np.sqrt(cosmo.Or0) * H0 / G)

    al1 = C1a * C1b ** (nb+7)

    return al1

def alpha2(cosmo, ns = 0.9649, nb = 2., kpiv = 10.):

    H0 = cosmo.h*100*km_to_Mpc # In s**-1

    al2 = (nb-ns)*(kpiv/np.pi)**(nb+3)*(G/(2 * c * H0 * np.sqrt(cosmo.Or0)))**(nb+3)

    return al2

def alpha3(ns = 0.9649):

    return ns + 3


def fm_brk(cosmo, dc = 0.6, Ms = 1e1, ns = 0.9649, nb = 2., kpiv = 10.,gridres = 1e4):
    
    fm_values = np.logspace(-100,100,gridres)
    
    def dc_f(fm):
        
        '''
        retruns dc(fm) for the specified parameters in fm_interp
        '''
    
        H0 = cosmo.h*100*km_to_Mpc # In s**-1

        C1a = (4 * np.pi * (A0 /np.power(k0, ns)) / ((ns+3) * (nb+3))) * (G/(np.pi * c**2))**4 * kpiv**(ns-nb)

        C1b = np.pi * np.sqrt(2 * c * np.sqrt(cosmo.Or0) * H0 / G)

        al1 = C1a * C1b ** (nb+7)

        al2 = (nb-ns)*(kpiv/np.pi)**(nb+3)*(G/(2 * c * H0 * np.sqrt(cosmo.Or0)))**(nb+3)

        al3 = ns+3

        return np.sqrt(al1 * (al2 * (Ms/fm)**2 + al3*(Ms/fm)**((1-nb)/2)))
    
    
    dc_values = np.vectorize(dc_f)(fm_values)
    
    fm_interpolated = interp1d(dc_values,fm_values)
    
    result = fm_interpolated(dc)
        
    return result