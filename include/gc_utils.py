import numpy as np
import mpmath as mp
from scipy.signal import find_peaks
from scipy.optimize import minimize,  differential_evolution, curve_fit

"""
Wrapper functions for the various special functions used in the expressions.
"""
def theta3_wrapper(mu_val, v_s_val, beta, hbar, K, L):
    q = mp.e**(-beta * mp.pi * hbar * v_s_val / (2*K*L))
    arg = -mp.j * beta * mu_val / 2
    return mp.jtheta(3, arg, q)

def theta3_general(z_arg, mu_val, v_s_val, beta, hbar, K, L):
    q = mp.e**(-beta * mp.pi * hbar * v_s_val / (2*K*L))
    return mp.jtheta(3,z_arg, q)

def theta1_wrapper(x_val, mu_val, v_s_val, beta, hbar, L):
    q = mp.e**(-(beta*mp.pi*hbar*v_s_val - beta*mu_val)/L)
    arg = mp.pi * x_val / L
    return mp.jtheta(1,arg, q)

def dedekind_eta_wrapper(mu_val, v_s_val,beta, hbar, L):
    tau = mp.j*(beta*hbar*v_s_val - beta*mu_val)/L
    return mp.eta(tau)

def theta1_wrapper_deriv(x_val, mu_val, v_s_val, beta, hbar, L, n):
    q = mp.e**(-(beta*mp.pi*hbar*v_s_val - beta*np.pi*mu_val)/L)
    arg = mp.pi * x_val / L
    return mp.jtheta(1,arg, q, derivative = n)

def central_derivative(f, x, h=1e-6):
    h = mp.mpf(h)
    return (f(x + h) - f(x - h)) / (2*h)

def canonical_paircorr(x,K,v,L,T,hbar,rho):
    """
    Function that return the canonical pair correlation function g(x,0) with the given
    system parameters K,v (v_s, sound velocity in the system),L,T,hbar and rho. Note
    the analytical derivative is used for the 1/x^2 decay term.
    """
    val = np.zeros_like(x)
    for i, xv in enumerate(x):
        x_mp = mp.mpf(xv)
        val[i] += rho**2
        lnpp = (theta1_wrapper_deriv(x_mp,0.0, v, beta, hbar, L, n = 2)/theta1_wrapper(x_mp, 0.0, v_s, beta, hbar, L))
        lnpp -= (theta1_wrapper_deriv(x_mp, 0.0, v, beta, hbar, L, n = 1)/theta1_wrapper(x_mp, 0.0, v_s, beta, hbar, L))**2
        val[i] += (K/(2*L**2))*lnpp
        exp_fact = mp.e**((beta*hbar*mp.pi*v)/(6*L))
        theta1_val = theta1_wrapper(x_mp, 0.0, v, beta, hbar, L)
        eta_val = dedekind_eta_wrapper(0.0, v,beta, hbar, L)
        denom = 2*eta_val
        ampl = mp.abs((exp_fact * theta1_val) / denom)
        bracket_pow = mp.power(ampl, -2*K)
        val[i] += float(mp.fabs(bracket_pow))
    return val/rho**2
    
def grandcanonical_paircorr(x,K,v,L,T,hbar,rho,mu):
    """
    Function that return the canonical pair correlation function g(x,0) with the given
    system parameters K,v (v_s, sound velocity in the system) ,L,T,hbar, rho and mu. Note
    the analytical derivative is used for the 1/x^2 decay term.
    """
    val = np.zeros_like(x)
    for i, xv in enumerate(x):
        x_mp = mp.mpf(xv)
        val[i] += rho**2
        lnpp = (theta1_wrapper_deriv(x_mp,mu, v, beta, hbar, L, n = 2)/theta1_wrapper(x_mp, mu, v, beta, hbar, L))
        lnpp -= (theta1_wrapper_deriv(x_mp, mu, v, beta, hbar, L, n = 1)/theta1_wrapper(x_mp, mu, v, beta, hbar, L))**2
        val[i] += (K/(2*L**2))*lnpp
        exp_fact = mp.e**((beta*hbar*mp.pi*v - mu)/(6*L))
        theta1_val = theta1_wrapper(x_mp, mu, v, beta, hbar, L)
        eta_val = dedekind_eta_wrapper(mu, v,beta, hbar, L)
        denom = 2*eta_val
        ampl = mp.abs((exp_fact * theta1_val) / denom)
        bracket_pow = mp.power(ampl, -2*K)
        val[i] = float(mp.fabs(bracket_pow))
    return val/rho**2
        
def canonical_envelope(x,K,v,L,T,mu,hbar,rho):
    """
    Function that return the enevlope of the decay of the oscillatory part
    of the canonical pair correlation function g(x,0) with the given
    system parameters K,v (v_s, sound velocity in the system) ,L,T,hbar, rho and mu
    """
    beta = 1.0/T; 
    fac = np.zeros_like(x)
    for i, xv in enumerate(x):
        x_mp = mp.mpf(xv)
        exp_fact = mp.e**((beta*hbar*mp.pi*v - mu)/(6*L))
        theta1_val = theta1_wrapper(x_mp, mu, v, beta, hbar, L)
        eta_val = dedekind_eta_wrapper(mu, v,beta, hbar, L)
        denom = 2*eta_val
        bracket = (exp_fact * theta1_val) / denom
        bracket_pow = mp.power(mp.fabs(bracket), -2*K)
        fac[i] = float(bracket_pow)
    return fac/rho**2

def grandcanonical_envelope(x,K,v,L,T,rho,mu,hbar):
    """
    Function that return the enevlope of the decay of the oscillatory part
    of the grand canonical pair correlation function g(x,0) with the given
    system parameters K,v (v_s, sound velocity in the system) ,L,T,hbar, rho and mu
    """
    beta = 1.0/T; 
    theta3_den = theta3_wrapper(mu, v,beta, hbar, K, L)
    corr = np.zeros_like(x)
    for i, xv in enumerate(x):
        x_mp = mp.mpf(xv)
        arg_num = -mp.j/2 * (beta*mu + 2*mp.j*mp.pi*x_mp/L)
        theta3_num = theta3_general(arg_num, mu, v,beta, hbar, K, L)
        pref_ratio = theta3_num / theta3_den
        exp_fact = mp.e**((beta*hbar*mp.pi*v - beta*mu)/(6*L))
        theta1_val = theta1_wrapper(x_mp, mu, v,beta, hbar, L)
        eta_val = dedekind_eta_wrapper(mu, v,beta, hbar, L)
        denom = 2*eta_val
        bracket = (exp_fact * theta1_val) / denom
        bracket_pow = mp.power(mp.fabs(bracket), -2*K)
        osc = mp.fabs(pref_ratio * bracket_pow)
        corr[i] = float(osc)
    return corr/rho**2
