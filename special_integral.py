from mpmath import *
from scipy.special import gamma
from numpy import pi,sin,empty,ndarray,full,nan,integer,floating

mp.pretty = True # hace que los outputs de hyper salgan como numeros sin parentesis redondos

em = 0.577215664901532860606 # Euler Macheroni constant


def __p_2(b):
    
    '''
    Case: p = 2
    '''
    
    if b < 10**(-4):
        
        #interpolacion lineal tendiendo a 0 cuando b va a 0. El ultimo valor estable es cercano a b ~ 1e-4
        
        return __p_2(10**(-4))/(10**(-4))*b
    
    else:
            
        c0 = gamma(4/3)*hyper([1/6],[1/3,1/2,5/6,7/6],-1/(11664*b**2))

        c1 = gamma(2/3)*hyper([5/6],[7/6,3/2,5/3,11/6],-1/(11664*b**2))

        c2 = 9*b*pi + hyper([1/2,1],[2/3,5/6,7/6,4/3,3/2],-1/(11664*b**2))

        return -c0/b**(1/3) - c1/(540*b**(5/3)) + c2/(18*b)

def __p_3(b):
    
    '''
    Case: p = 3
    '''
    
    if b < 10**(-4):
        
        #interpolacion lineal tendiendo a 0 cuando b va a 0. El ultimo valor estable es cercano a b ~ 1e-4
        
        return __p_3(10**(-4))/(10**(-4))*b
    
    else:
    
        c0 = 840*b**(4/3)*gamma(2/3)*hyper([1/3],[2/3,7/6,4/3,3/2],-1/(11664*b**2))

        c1 = 42*b**(2/3)*gamma(4/3)*hyper([2/3],[4/3,3/2,5/3,11/6],-1/(11664*b**2))

        c2 = hyper([1,1],[4/3,5/3,11/6,2,13/6],-1/(11664*b**2))

        c3 = 5040*b**2*(3-2*em+log(b))

        return (c0 - c1 + c2 + c3)/(15120*b**2)

def __othercase(p,b):
    
    c0 = 1/3*b**(-1+p/3)*gamma(1-p/3)*hyper([1/2-p/6,1-p/6],[1/3,1/2,2/3,5/6,7/6],-1/(11664*b**2))

    c1 = 1/18*b**(1/3*(-5+p))*gamma(5/3-p/3)*hyper([5/6-p/6,4/3-p/6],[2/3,5/6,7/6,4/3,3/2],-1/(11664*b**2))

    c2 = 1/360*b**(1/3*(-7+p))*gamma(7/3-p/3)*hyper([7/6-p/6,5/3-p/6],[7/6,4/3,3/2,5/3,11/6],-1/(11664*b**2))

    c3 = gamma(2-p)*sin(p*pi/2)
    
    return - c0 + c1 - c2 + c3

def special_integral(p,b):
    
    '''
    Calcula la integral semianaliticamente dada por
    
    \int\limits_{0}^{\infty} \left(1-e^{-b x^3}\right)\frac{\sin\left( x\right)}{x^{p-1}} dx
    
    donde los argumentos son (p,b) tal que 1 <= p <= 3 y b > 0
    '''
    
    if isinstance(b,(list)):
        
        b = np.array(b)
    
    if not isinstance(b,(list,float,int,ndarray,integer,floating)):
        
        raise ValueError(" 'b' parameter must be a float, an int, an array or a list.")
    
    try:
    
        if p == 2:
            
            if isinstance(b,(ndarray)):
                
                output =  full(len(b), nan)
                
                for i,j in enumerate(b):
                    
                    output[i] = __p_2(j)
                    
                return output 
                
            else:
                
                return __p_2(b)
            
        elif p == 3:
            
            if isinstance(b,(ndarray)):
                
                output =  full(len(b), (nan))
                
                for i,j in enumerate(b):
                    
                    output[i] = __p_3(j)
                    
                return output 
                
            else:
                
                return __p_3(b)
            
        elif p > 1 and p < 3 and b > 0:
            
            print('special case where p != 2 and p!= 3 and 1<p<3 and b> 0')
            
                        
            if isinstance(b,(ndarray)):
                
                output =  full(len(b), nan)
                
                for i,j in enumerate(b):
                    
                    ################# OTHER CASE ############################
                    # no esta haciendo interpolacion en casos de b tendiendo a 0.
                    
                    output[i] = __othercase(p,(b[j]))
                    
                    return output 
                
            else:
                
                return __othercase(p,b)

        else:
            
            print('Problems with arguments p or b. Conditions: (p == 1 or p == 3) and b > 0.')
        
    except:
        
        raise ValueError('Problems with arguments p or b. Conditions: (p == 1 or p == 3) and b > 0.')
