import numpy as np
import mpmath as mp
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
from scipy.ndimage import uniform_filter1d
"""
Wrapper functions for the various special functions used in the expressions.
"""
def theta3_wrapper(mu_val, v_s_val, beta, hbar, K, L):
    q = mp.e**(-beta * mp.pi * hbar * v_s_val*K / (2*L))
    arg = -mp.j * beta * mu_val / 2
    return mp.jtheta(3, arg, q)

def theta3_general(z_arg, mu_val, v_s_val, beta, hbar, K, L):
    q = mp.e**(-beta * mp.pi * hbar * v_s_val*K / (2*L))
    return mp.jtheta(3,z_arg, q)

def theta1_wrapper(x_val, mu_val, v_s_val, beta, hbar, L):
    q = mp.e**(-(beta*mp.pi*hbar*v_s_val - beta*np.pi*mu_val)/L)
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

def canonical_paircorr(x,K,v,A,L,T,hbar,rho):
    """
    Function that return the canonical pair correlation function g(x,0) with the given
    system parameters K,v (v_s, sound velocity in the system),L,T,hbar and rho. Note
    the analytical derivative is used for the 1/x^2 decay term.
    """
    beta = 1./T
    val = np.zeros_like(x)
    for i, xv in enumerate(x):
        x_mp = mp.mpf(xv)
        #val[i] += rho**2
        lnpp = (theta1_wrapper_deriv(x_mp,0.0, v, beta, hbar, L, n = 2)/theta1_wrapper(x_mp, 0.0, v, beta, hbar, L))
        lnpp -= (theta1_wrapper_deriv(x_mp, 0.0, v, beta, hbar, L, n = 1)/theta1_wrapper(x_mp, 0.0, v, beta, hbar, L))**2
        val[i] += (1/(2*L**2*K))*lnpp
        exp_fact = mp.e**((beta*hbar*mp.pi*v)/(6*L))
        theta1_val = theta1_wrapper(x_mp, 0.0, v, beta, hbar, L)
        eta_val = dedekind_eta_wrapper(0.0, v,beta, hbar, L)
        denom = 2*eta_val
        phase_factor = mp.e**(2*mp.j*mp.pi*rho*x_mp)
        ampl = mp.fabs((exp_fact * theta1_val) / denom)
        bracket_pow = mp.power(ampl, -2/K)
        val[i] += A*float(mp.re(phase_factor*bracket_pow))
    return val/rho**2
    
def grandcanonical_paircorr(x,K,v,A,L,T,hbar,rho,mu):
    """
    Function that return the canonical pair correlation function g(x,0) with the given
    system parameters K,v (v_s, sound velocity in the system) ,L,T,hbar, rho and mu. Note
    the analytical derivative is used for the 1/x^2 decay term.
    """
    beta = 1./T
    val = np.zeros_like(x)
    theta3_den = theta3_wrapper(mu, v,beta, hbar, K, L)
    for i, xv in enumerate(x):
        x_mp = mp.mpf(xv)
        #val[i] += rho**2
        lnpp = (theta1_wrapper_deriv(x_mp,mu, v, beta, hbar, L, n = 2)/theta1_wrapper(x_mp, mu, v, beta, hbar, L))
        lnpp -= (theta1_wrapper_deriv(x_mp, mu, v, beta, hbar, L, n = 1)/theta1_wrapper(x_mp, mu, v, beta, hbar, L))**2
        val[i] += (1/(2*L**2*K))*lnpp
        arg_num = -mp.j/2 * (beta*mu + 2*mp.j*mp.pi*x_mp/L)
        theta3_num = theta3_general(arg_num, mu, v,beta, hbar, K, L)
        pref_ratio = theta3_num / theta3_den
        exp_fact = mp.e**((beta*hbar*mp.pi*v - mu)/(6*L))
        theta1_val = theta1_wrapper(x_mp, mu, v, beta, hbar, L)
        eta_val = dedekind_eta_wrapper(mu, v,beta, hbar, L)
        denom = 2*eta_val
        phase_factor = mp.e**(2*mp.j*mp.pi*rho*x_mp)
        ampl = mp.fabs((exp_fact * theta1_val) / denom)
        bracket_pow = mp.power(ampl, -2/K)
        val[i] += A*float(mp.re(pref_ratio * phase_factor* bracket_pow))
    return val/rho**2
    
def canonical_osc(x,K,v,A,L,T,hbar,rho):
    """
    Function that return the envelope of the decay of the oscillatory part
    of the canonical pair correlation function g(x,0) with the given
    system parameters K,v (v_s, sound velocity in the system) ,L,T,hbar, rho and mu
    """
    beta = 1.0/T; 
    fac = np.zeros_like(x)
    for i, xv in enumerate(x):
        x_mp = mp.mpf(xv)
        exp_fact = mp.e**((beta*hbar*mp.pi*v - 0.0)/(6*L))
        theta1_val = theta1_wrapper(x_mp, 0.0, v, beta, hbar, L)
        eta_val = dedekind_eta_wrapper(0.0, v,beta, hbar, L)
        denom = 2*eta_val
        bracket = (exp_fact * theta1_val) / denom
        bracket_pow = mp.power(mp.fabs(bracket), -2/K)
        fac[i] += A*float(mp.re(bracket_pow))
    return fac/rho**2

def grandcanonical_osc(x,K,v,A,L,T,rho,hbar,mu):
    """
    Function that return the envelope of the decay of the oscillatory part
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
        bracket_pow = mp.power(mp.fabs(bracket), -2/K)
        osc = mp.re(pref_ratio * bracket_pow)
        corr[i] += A*float(osc)
    return corr/rho**2
    
def canonical_envelope(x,K,v,A,L,T,hbar,rho):
    """
    Function that return the envelope of the decay of the oscillatory part
    of the canonical pair correlation function g(x,0) with the given
    system parameters K,v (v_s, sound velocity in the system) ,L,T,hbar, rho and mu
    """
    beta = 1.0/T; 
    fac = np.zeros_like(x)
    for i, xv in enumerate(x):
        x_mp = mp.mpf(xv)
        lnpp = (theta1_wrapper_deriv(x_mp,0.0, v, beta, hbar, L, n = 2)/theta1_wrapper(x_mp, 0.0, v, beta, hbar, L))
        lnpp -= (theta1_wrapper_deriv(x_mp, 0.0, v, beta, hbar, L, n = 1)/theta1_wrapper(x_mp, 0.0, v, beta, hbar, L))**2
        fac[i] += (1/(2*L**2*K))*lnpp
        exp_fact = mp.e**((beta*hbar*mp.pi*v - 0.0)/(6*L))
        theta1_val = theta1_wrapper(x_mp, 0.0, v, beta, hbar, L)
        eta_val = dedekind_eta_wrapper(0.0, v,beta, hbar, L)
        denom = 2*eta_val
        bracket = (exp_fact * theta1_val) / denom
        bracket_pow = mp.power(mp.fabs(bracket), -2/K)
        fac[i] += A*float(mp.re(bracket_pow))
    return fac/rho**2

def grandcanonical_envelope(x,K,v,A,L,T,rho,hbar,mu):
    """
    Function that return the envelope of the decay of the oscillatory part
    of the grand canonical pair correlation function g(x,0) with the given
    system parameters K,v (v_s, sound velocity in the system) ,L,T,hbar, rho and mu
    """
    beta = 1.0/T; 
    theta3_den = theta3_wrapper(mu, v,beta, hbar, K, L)
    corr = np.zeros_like(x)
    for i, xv in enumerate(x):
        x_mp = mp.mpf(xv)
        lnpp = (theta1_wrapper_deriv(x_mp,mu, v, beta, hbar, L, n = 2)/theta1_wrapper(x_mp, mu, v, beta, hbar, L))
        lnpp -= (theta1_wrapper_deriv(x_mp, mu, v, beta, hbar, L, n = 1)/theta1_wrapper(x_mp, mu, v, beta, hbar, L))**2
        corr[i] += (1/(2*L**2*K))*lnpp
        arg_num = -mp.j/2 * (beta*mu + 2*mp.j*mp.pi*x_mp/L)
        theta3_num = theta3_general(arg_num, mu, v,beta, hbar, K, L)
        pref_ratio = theta3_num / theta3_den
        exp_fact = mp.e**((beta*hbar*mp.pi*v - beta*mu)/(6*L))
        theta1_val = theta1_wrapper(x_mp, mu, v,beta, hbar, L)
        eta_val = dedekind_eta_wrapper(mu, v,beta, hbar, L)
        denom = 2*eta_val
        bracket = (exp_fact * theta1_val) / denom
        bracket_pow = mp.power(mp.fabs(bracket), -2/K)
        osc = mp.re(pref_ratio * bracket_pow)
        corr[i] += A*float(osc)
    return corr/rho**2

def fit(xdata,ydata,fitting_form):
    """
    General fitting function
    """
    popt, pcov = curve_fit(fitting_form, xdata, ydata, p0=[0.3,1,0.1], bounds=([0.01, 0.1,-np.inf], [np.inf, np.inf,np.inf]))
    K, v_s,A = popt
    
    K_std, v_std, A_std = np.sqrt(np.diag(pcov))
    return K,K_std,v_s,v_std,A,A_std
    

def fitpeaks(xdata,ydata,fitting_form,prominence_frac=0.01,skip=10):
    """
    Function that takes the paircorrelation function data and extracts 
    K, v_s by fitting to the given form of the expressions. 
    Please note that the fitting form should not take additional arguments
    apart from K, v
    """
    yabs = np.abs(ydata)
    prominence = prominence_frac*np.max(yabs)
    peaks, properties = find_peaks(yabs, prominence=prominence)
    x_peaks = xdata[peaks][skip:]
    y_peaks = yabs[peaks][skip:]
    popt, pcov = curve_fit(fitting_form, x_peaks, y_peaks, p0=[0.3,3], bounds=([0.01, 0.0], [np.inf, np.inf]))
    K, v_s = popt
    
    K_std, v_std = np.sqrt(np.diag(pcov))
    return K,K_std,v_s,v_std

def extract_oscillation(x, y, a):
    dx = np.mean(np.diff(x))

    period = 2*np.pi/a

    # average over ~5 periods
    window = int(round(5 * period / dx))
    window = max(window, 5)

    c = uniform_filter1d(y*np.cos(a*x), size=window)
    s = uniform_filter1d(y*np.sin(a*x), size=window)

    amp = 2*np.sqrt(c**2 + s**2)
    phase = np.arctan2(-s, c)

    osc = amp*np.cos(a*x + phase)

    return osc, amp, phase



