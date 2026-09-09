import numpy as np
import mpmath as mp
from joblib import Parallel, delayed

def theta3_wrapper(mu_val, v_s_val, beta, hbar, K, L):
    q = mp.e**(-beta * mp.pi * hbar * v_s_val*K / (2*L))
    arg = -mp.j * beta * mu_val / 2
    return mp.jtheta(3, arg, q)

def theta3_general(z_arg, v_s_val, beta, hbar, K, L):
    q = mp.e**(-beta * mp.pi * hbar * v_s_val*K / (2*L))
    return mp.jtheta(3,z_arg, q)

def theta1_wrapper(x_val, v_s_val, beta, hbar, L):
    q = mp.e**(-(beta*mp.pi*hbar*v_s_val)/L)
    arg = mp.pi * x_val / L
    return mp.jtheta(1,arg, q)

def theta1_wrapper_deriv(x_val, v_s_val, beta, hbar, L, n):
    q = mp.e**(-(beta*mp.pi*hbar*v_s_val)/L)
    arg = mp.pi * x_val / L
    return mp.jtheta(1,arg, q, derivative = n)

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
        lnpp = (theta1_wrapper_deriv(x_mp, v, beta, hbar, L, n = 2)/theta1_wrapper(x_mp, v, beta, hbar, L))
        lnpp -= (theta1_wrapper_deriv(x_mp, v, beta, hbar, L, n = 1)/theta1_wrapper(x_mp,  v, beta, hbar, L))**2
        val[i] += (1/(2*L**2*K))*lnpp
        theta1_val = theta1_wrapper(x_mp, v, beta, hbar, L)
        denom = np.pi * theta1_wrapper_deriv(0.0, v, beta, hbar, L, n = 1)
        phase_factor = mp.e**(2*mp.j*mp.pi*rho*x_mp)
        ampl = mp.fabs((L * theta1_val) / denom)
        bracket_pow = mp.power(ampl, -2/K)
        val[i] += A*float(mp.re(phase_factor*bracket_pow))
    return val/rho**2

def canonical_paircorr_osc(x,K,v,A,L,T,hbar,rho):
    """
    Function that returns just the oscillatory part of the canonical pair correlation function g(x,0) with the given
    system parameters K,v (v_s, sound velocity in the system),L,T,hbar and rho. Note
    the analytical derivative is used for the 1/x^2 decay term.
    """
    beta = 1./T
    val = np.zeros_like(x)
    for i, xv in enumerate(x):
        x_mp = mp.mpf(xv)
        theta1_val = theta1_wrapper(x_mp, v, beta, hbar, L)
        denom = np.pi * theta1_wrapper_deriv(0.0, v, beta, hbar, L, n = 1)
        phase_factor = mp.e**(2*mp.j*mp.pi*rho*x_mp)
        ampl = mp.fabs((L * theta1_val) / denom)
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
        lnpp = (theta1_wrapper_deriv(x_mp, v, beta, hbar, L, n = 2)/theta1_wrapper(x_mp, v, beta, hbar, L))
        lnpp -= (theta1_wrapper_deriv(x_mp, v, beta, hbar, L, n = 1)/theta1_wrapper(x_mp, v, beta, hbar, L))**2
        val[i] += (1/(2*L**2*K))*lnpp
        arg_num = -mp.j/2 * (beta*mu + 2*mp.j*mp.pi*x_mp/L)
        theta3_num = theta3_general(arg_num, v,beta, hbar, K, L)
        pref_ratio = theta3_num / theta3_den
        theta1_val = theta1_wrapper(x_mp, v, beta, hbar, L)
        denom = np.pi*theta1_wrapper_deriv(0.0, v, beta, hbar, L, n = 1)
        phase_factor = mp.e**(2*mp.j*mp.pi*rho*x_mp)
        ampl = mp.fabs((L * theta1_val) / denom)
        bracket_pow = mp.power(ampl, -2/K)
        val[i] += A*float(mp.re(pref_ratio * phase_factor* bracket_pow))
    return val/rho**2

def grandcanonical_paircorr_osc(x,K,v,A,L,T,hbar,rho,mu):
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
        arg_num = -mp.j/2 * (beta*mu + 2*mp.j*mp.pi*x_mp/L)
        theta3_num = theta3_general(arg_num, v,beta, hbar, K, L)
        pref_ratio = theta3_num / theta3_den
        theta1_val = theta1_wrapper(x_mp, v, beta, hbar, L)
        denom = np.pi*theta1_wrapper_deriv(0.0, v, beta, hbar, L, n = 1)
        phase_factor = mp.e**(2*mp.j*mp.pi*rho*x_mp)
        ampl = mp.fabs((L * theta1_val) / denom)
        bracket_pow = mp.power(ampl, -2/K)
        val[i] += A*float(mp.re(pref_ratio * phase_factor* bracket_pow))
    return val/rho**2

def P(N,K,v,L,T,hbar,rho):
    """
    P_0(N) = \frac{e^{\frac{-\pi v_N}{2LT}(N - N_0)^2}}{\vartheta_3(0,e^{\frac{-\pi v_N}{2LT}})}
    """
    N0 = rho * L
    vN = v/K
    num = np.exp(-np.pi * vN * (N - N0)**2 / (2 *L * T))
    den = mp.jtheta(3, 0, np.exp(-np.pi * vN / (2 * L *T)))
    return num/den

def obdm_bos(x_vals,K,v,A,L,T,rho,hbar):
    beta = 1.0/T; 
    corr = np.zeros_like(x_vals)
    for i, xv in enumerate(x_vals):
        x_mp = mp.mpf(xv)
        pref = theta3_general(np.pi*x_mp/L,v,beta,hbar,K,L)/theta3_general(0,v,beta,hbar,K,L)
        theta1_val = theta1_wrapper(x_mp, v, beta, hbar, L)
        denom = np.pi*theta1_wrapper_deriv(0.0, v, beta, hbar, L, n = 1)
        bracket = (L * theta1_val) / denom
        bracket_pow = mp.power(bracket, -1/(2*K))
        corr[i] = float(mp.re(pref * bracket_pow))
    return corr

def obdm_fer_ce(x_vals,K,v,A,L,T,rho,hbar):
    beta = 1.0/T; 
    corr = np.zeros_like(x_vals)
    for i, xv in enumerate(x_vals):
        x_mp = mp.mpf(xv)
        pref = theta3_general(np.pi*x_mp/L,v,beta,hbar,K,L)/theta3_general(0,v,beta,hbar,K,L)
        theta1_val = theta1_wrapper(x_mp,  v, beta, hbar, L)
        denom = np.pi*theta1_wrapper_deriv(0.0,  v, beta, hbar, L, n = 1)
        bracket = (L * theta1_val) / denom
        bracket_pow = mp.power(bracket, -(1/2)*(K + 1/K))
        phase_factor = mp.e**(mp.j*mp.pi*rho*x_mp)
        corr[i] = float(mp.re(pref * phase_factor * bracket_pow))
    return corr

def obdm_fer_gce(x_vals,K,v,A,L,T,rho,hbar,mu):
    beta = 1.0/T; 
    corr = np.zeros_like(x_vals)
    for i, xv in enumerate(x_vals):
        x_mp = mp.mpf(xv)
        pref = theta3_general(np.pi*x_mp/L,v,beta,hbar,K,L)/theta3_general(0,v,beta,hbar,K,L)
        theta3_den = theta3_wrapper(mu, v,beta, hbar, K, L)
        arg_num = -mp.j/2 * (beta*mu + 2*mp.j*mp.pi*x_mp/L)
        theta3_num = theta3_general(arg_num, v,beta, hbar, K, L)
        pref_ratio = theta3_num / theta3_den
        theta1_val = theta1_wrapper(x_mp, v, beta, hbar, L)
        denom = np.pi*theta1_wrapper_deriv(0.0, v, beta, hbar, L, n = 1)
        bracket = (L * theta1_val) / denom
        bracket_pow = mp.power(bracket, -(1/2)*(K + 1/K))
        phase_factor = mp.e**(mp.j*mp.pi*rho*x_mp)
        corr[i] = float(mp.re(pref_ratio * pref * phase_factor * bracket_pow))
    return corr

def regularize_grid(x_grid, dx, L):
    x_reg = np.array(x_grid, dtype=float).copy()
    alpha_uv = 0.2
    x_reg[np.isclose(x_reg, 0.0)] = alpha_uv / 2
    x_reg[np.isclose(x_reg, L)] = L - alpha_uv / 2
    return x_reg
 
def Sfirstterm(q,K,v,L,T,hbar,rho):
    beta = 1.0/T
    val = (q/(2*np.pi*K*rho))*(1.0/np.tanh(beta*hbar*v*q/2))
    return val

def calc_chi2_twop(i, j, x_vals, a_vals, b_vals, act_vals, fit_func):
    a = a_vals[i]
    b = b_vals[j]

    pred = fit_func(x_vals, a, b)
    chi2 = np.sum((act_vals - pred) ** 2)

    return i, j, chi2

def chi2_surface_twop(func,x_vals,a_vals,b_vals, act_vals):
    results = Parallel(n_jobs=-1, verbose=10)(
    delayed(calc_chi2_twop)(i, j, x_vals, a_vals, b_vals, act_vals, func)
    for i in range(len(a_vals))
    for j in range(len(b_vals))
)
    chi2_grid = np.zeros((len(a_vals), len(b_vals)))

    for i, j, chi2 in results:
        chi2_grid[i, j] = chi2

    # Find grid minimum
    i_min, j_min = np.unravel_index(
        np.argmin(chi2_grid),
        chi2_grid.shape
    )

    print("grid minimum at:",  a_vals[i_min],  b_vals[j_min])

    return chi2_grid

def calc_chi2_threep(i, j, k, x_vals, a_vals, b_vals, c_vals, act_vals, fit_func):
    a = a_vals[i]
    b = b_vals[j]
    c = c_vals[k]

    pred = fit_func(x_vals, a, b, c)
    chi2 = np.sum((act_vals - pred) ** 2)

    return i, j, k, chi2

def chi2_surface_threep(func,x_vals,a_vals,b_vals, c_vals,act_vals):
    results = Parallel(n_jobs=-1, verbose=10)(
    delayed(calc_chi2_threep)(i, j, k, x_vals, a_vals, b_vals, c_vals, act_vals, func)
    for i in range(len(a_vals))
    for j in range(len(b_vals))
    for k in range(len(c_vals))
)
    chi2_grid = np.zeros((len(a_vals), len(b_vals), len(c_vals)))

    for i, j, k, chi2 in results:
        chi2_grid[i, j, k] = chi2

    # Find grid minimum
    i_min, j_min, k_min = np.unravel_index(
        np.argmin(chi2_grid),
        chi2_grid.shape
    )

    print("grid minimum at:",  a_vals[i_min],  b_vals[j_min], c_vals[k_min])

    return k_min, chi2_grid
