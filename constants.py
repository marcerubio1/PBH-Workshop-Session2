'''
This script contains the values for different physical constant and their convertion 
factors to the desired units

'''

import numpy as np 

PI = np.pi
G_cgs = 6.674*10**(-8) #cm^3/(g*s^2)
c_kms = 299792 # km/s
hbar_cgs = 1.0546e-27 # in cgs
c_cgs = 3e10 # in cm per second
mu0_old = 4.*PI*1e-7 # in Tesla * m * s * C^{-1}


km_to_Mpc = 3.241*10**(-20)
cm_to_Mpc = 3.241*10**(-25)
g_to_Ms = 5.03e-34
Tesla_to_Gauss = 1e4
m_to_Mpc = 3.24078*1e-23


G = G_cgs*(cm_to_Mpc**3)/(g_to_Ms) # In Mpc**3 M_s**-1 s**-2
c = c_kms * km_to_Mpc # Light speed in Mpc/s
mu0 = mu0_old * Tesla_to_Gauss * m_to_Mpc #in Gauss * Mpc * s * Coulomb^{-1}


A0 = (2.10521e-9)
k0 = 0.05 #Mpc**-1
