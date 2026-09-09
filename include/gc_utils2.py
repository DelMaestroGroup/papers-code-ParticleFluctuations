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
        theta1_val = theta1_wrapper(x_mp, 0.0, v, beta, hbar, L)
        denom = np.pi * theta1_wrapper_deriv(0.0, 0.0, v, beta, hbar, L, n = 1)
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
        lnpp = (theta1_wrapper_deriv(x_mp,mu, v, beta, hbar, L, n = 2)/theta1_wrapper(x_mp, mu, v, beta, hbar, L))
        lnpp -= (theta1_wrapper_deriv(x_mp, mu, v, beta, hbar, L, n = 1)/theta1_wrapper(x_mp, mu, v, beta, hbar, L))**2
        val[i] += (1/(2*L**2*K))*lnpp
        arg_num = -mp.j/2 * (beta*mu + 2*mp.j*mp.pi*x_mp/L)
        theta3_num = theta3_general(arg_num, mu, v,beta, hbar, K, L)
        pref_ratio = theta3_num / theta3_den
        theta1_val = theta1_wrapper(x_mp, mu, v, beta, hbar, L)
        denom = np.pi*theta1_wrapper_deriv(0.0, mu, v, beta, hbar, L, n = 1)
        phase_factor = mp.e**(2*mp.j*mp.pi*rho*x_mp)
        ampl = mp.fabs((L * theta1_val) / denom)
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
        theta1_val = theta1_wrapper(x_mp, 0.0, v, beta, hbar, L)
        denom = np.pi*theta1_wrapper_deriv(0.0, 0.0, v, beta, hbar, L, n = 1)
        bracket = (L * theta1_val) / denom
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
        theta1_val = theta1_wrapper(x_mp, mu, v,beta, hbar, L)
        denom = np.pi*theta1_wrapper_deriv(0.0, mu, v, beta, hbar, L, n = 1)
        bracket = (L * theta1_val) / denom
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
        theta1_val = theta1_wrapper(x_mp, 0.0, v, beta, hbar, L)
        denom = np.pi*theta1_wrapper_deriv(0.0, 0.0, v, beta, hbar, L, n = 1)
        bracket = (L * theta1_val) / denom
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
        theta1_val = theta1_wrapper(x_mp, mu, v,beta, hbar, L)
        denom = np.pi*theta1_wrapper_deriv(0.0, mu, v, beta, hbar, L, n = 1)
        bracket = (L * theta1_val) / denom
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
    popt, pcov = curve_fit(fitting_form, x_peaks, y_peaks, p0=[0.3,8], bounds=([0.01, 0.31], [np.inf, np.inf]))
    K, v_s = popt
    
    K_std, v_std = np.sqrt(np.diag(pcov))
    return K,K_std,v_s,v_std

def fitpeakswithA(xdata,ydata,fitting_form,prominence_frac=0.01,skip=10):
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
    popt, pcov = curve_fit(fitting_form, x_peaks, y_peaks, p0=[5.2,49.5,1], bounds=([0.01, 0.4,0.00001], [np.inf, np.inf,10]))
    K, v_s, A = popt
    corr = pcov / np.sqrt(np.outer(np.diag(pcov), np.diag(pcov)))
    print(corr)
    K_std, v_std, A_std = np.sqrt(np.diag(pcov))
    return K,K_std,v_s,v_std, A, A_std

def fitpeakswithA2(xdata, ydata, prominence_frac=0.01, skip=10):
    """
    Extract positive/negative extrema and perform two fits:

    1. Smooth part:
           S(x) = [g_+(x) + g_-(x)] / 2

       Fit to the smooth canonical finite-L expression to obtain K, v.

    2. Oscillatory envelope:
           E(x) = [g_+(x) - g_-(x)] / 2

       Fit to A * F(x; K, v), with K and v fixed from fit 1.

    Returns:
        K, K_std, v_s, v_std, A, A_std
    """
    L = 200
    T = 2
    rho = 0.25
    # ---------------------------------------------------------
    # Find positive and negative peaks
    # ---------------------------------------------------------

    prominence = prominence_frac * np.max(np.abs(ydata))

    peaks_pos, _ = find_peaks(
        ydata,
        prominence=prominence
    )

    peaks_neg, _ = find_peaks(
        -ydata,
        prominence=prominence
    )

    x_pos = xdata[peaks_pos]
    y_pos = ydata[peaks_pos]

    x_neg = xdata[peaks_neg]
    y_neg = ydata[peaks_neg]

    if len(x_pos) == 0 or len(x_neg) == 0:
        raise RuntimeError("Could not find both positive and negative peaks.")

    # ---------------------------------------------------------
    # Pair each positive peak with nearest negative peak
    # ---------------------------------------------------------

    pairs = []

    for xp, yp in zip(x_pos, y_pos):

        idx = np.argmin(np.abs(x_neg - xp))

        xn = x_neg[idx]
        yn = y_neg[idx]

        pairs.append((xp, yp, xn, yn))

    pairs = np.array(pairs)

    xp = pairs[:, 0]
    yp = pairs[:, 1]

    xn = pairs[:, 2]
    yn = pairs[:, 3]

    # Midpoint between extrema
    x_peaks = 0.5 * (xp + xn)

    # ---------------------------------------------------------
    # Extract smooth and oscillatory pieces
    # ---------------------------------------------------------

    # Smooth:
    #
    # (g_+ + g_-)/2
    #
    smooth = 0.5 * (yp + yn)

    # Oscillatory envelope:
    #
    # (g_+ - g_-)/2
    #
    envelope = 0.5 * (yp - yn)

    # ---------------------------------------------------------
    # Remove initial peaks
    # ---------------------------------------------------------

    x_fit = x_peaks[skip:]
    smooth_fit = smooth[skip:]
    envelope_fit = envelope[skip:]

    # ---------------------------------------------------------
    # Smooth canonical expression
    # ---------------------------------------------------------

    def canonical_smooth(x, K, v):

        beta = 1.0 / T

        result = np.zeros_like(x, dtype=float)

        for i, xv in enumerate(x):

            x_mp = mp.mpf(xv)

            theta = theta1_wrapper(
                x_mp, 0.0, v, beta, hbar, L
            )

            dtheta = theta1_wrapper_deriv(
                x_mp, 0.0, v, beta, hbar, L, n=1
            )

            ddtheta = theta1_wrapper_deriv(
                x_mp, 0.0, v, beta, hbar, L, n=2
            )

            lnpp = ddtheta / theta - (dtheta / theta)**2

            result[i] = (
                (1.0 / (2.0 * L**2 * K))
                * float(mp.re(lnpp))
                / rho**2
            )

        return result

    # ---------------------------------------------------------
    # FIT 1: smooth part -> K, v
    # ---------------------------------------------------------

    popt_smooth, pcov_smooth = curve_fit(
        canonical_smooth,
        x_fit,
        smooth_fit,
        p0=[5.2, 49.5],
        bounds=(
            [0.01, 0.4],
            [np.inf, np.inf]
        )
    )

    K, v_s = popt_smooth

    K_std, v_std = np.sqrt(np.diag(pcov_smooth))

    # ---------------------------------------------------------
    # FIT 2: oscillatory envelope -> A
    # ---------------------------------------------------------

    def canonical_oscillatory(x, A):

        beta = 1.0 / T

        result = np.zeros_like(x, dtype=float)

        for i, xv in enumerate(x):

            x_mp = mp.mpf(xv)

            theta1_val = theta1_wrapper(
                x_mp, 0.0, v_s, beta, hbar, L
            )

            denom = (
                np.pi
                * theta1_wrapper_deriv(
                    0.0, 0.0, v_s, beta, hbar, L, n=1
                )
            )

            bracket = (L * theta1_val) / denom

            bracket_pow = mp.power(
                mp.fabs(bracket),
                -2.0 / K
            )

            result[i] = (
                A
                * float(mp.re(bracket_pow))
                / rho**2
            )

        return result

    popt_A, pcov_A = curve_fit(
        canonical_oscillatory,
        x_fit,
        envelope_fit,
        p0=[1.0],
        bounds=([0.00001], [10.0])
    )

    A = popt_A[0]
    A_std = np.sqrt(pcov_A[0, 0])

    # ---------------------------------------------------------
    # Print correlation matrix of smooth K,v fit
    # ---------------------------------------------------------

    corr_smooth = (
        pcov_smooth
        / np.sqrt(
            np.outer(
                np.diag(pcov_smooth),
                np.diag(pcov_smooth)
            )
        )
    )

    print("Smooth fit correlation matrix:")
    print(corr_smooth)

    print("\nK =", K, "+/-", K_std)
    print("v =", v_s, "+/-", v_std)
    print("A =", A, "+/-", A_std)

    return K, K_std, v_s, v_std, A, A_std

def P(N,K,v,A,L,T,hbar,rho):
    """
    P_0(N) = \frac{e^{\frac{-\pi v_N}{2LT}(N - N_0)^2}}{\vartheta_3(0,e^{\frac{-\pi v_N}{2LT}})}
    """
    N0 = rho * L
    vN = v/K
    num = np.exp(-np.pi * vN * (N - N0)**2 / (2 *L * T))
    den = mp.jtheta(3, 0, np.exp(-np.pi * vN / (2 * L *T)))
    return num/den
    

    
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

def obdm_bos(x_vals,K,v,A,L,T,rho,hbar):
    beta = 1.0/T; 
    corr = np.zeros_like(x_vals)
    for i, xv in enumerate(x_vals):
        x_mp = mp.mpf(xv)
        pref = theta3_general(np.pi*x_mp/L,0.0,v,beta,hbar,K,L)/theta3_general(0,0.0,v,beta,hbar,K,L)
        theta1_val = theta1_wrapper(x_mp, 0.0, v, beta, hbar, L)
        denom = np.pi*theta1_wrapper_deriv(0.0, 0.0, v, beta, hbar, L, n = 1)
        bracket = (L * theta1_val) / denom
        bracket_pow = mp.power(bracket, -1/(2*K))
        corr[i] = float(mp.re(pref * bracket_pow))
    return corr

def obdm_fer_ce(x_vals,K,v,A,L,T,rho,hbar):
    beta = 1.0/T; 
    corr = np.zeros_like(x_vals)
    for i, xv in enumerate(x_vals):
        x_mp = mp.mpf(xv)
        pref = theta3_general(np.pi*x_mp/L,0.0,v,beta,hbar,K,L)/theta3_general(0,0.0,v,beta,hbar,K,L)
        theta1_val = theta1_wrapper(x_mp, 0.0, v, beta, hbar, L)
        denom = np.pi*theta1_wrapper_deriv(0.0, 0.0, v, beta, hbar, L, n = 1)
        bracket = (L * theta1_val) / denom
        bracket_pow = mp.power(bracket, -(1/2)*(K + 1/K))
        phase_factor = mp.e**(mp.j*mp.pi*rho*x_mp)
        corr[i] = float(mp.re(pref * phase_factor * bracket_pow))
    return corr

def obdm_fer_gce(x_vals,K,v,mu,A,L,T,rho,hbar):
    beta = 1.0/T; 
    corr = np.zeros_like(x_vals)
    for i, xv in enumerate(x_vals):
        x_mp = mp.mpf(xv)
        pref = theta3_general(np.pi*x_mp/L,0.0,v,beta,hbar,K,L)/theta3_general(0,0.0,v,beta,hbar,K,L)
        theta3_den = theta3_wrapper(mu, v,beta, hbar, K, L)
        arg_num = -mp.j/2 * (beta*mu + 2*mp.j*mp.pi*x_mp/L)
        theta3_num = theta3_general(arg_num, mu, v,beta, hbar, K, L)
        pref_ratio = theta3_num / theta3_den
        theta1_val = theta1_wrapper(x_mp, 0.0, v, beta, hbar, L)
        denom = np.pi*theta1_wrapper_deriv(0.0, 0.0, v, beta, hbar, L, n = 1)
        bracket = (L * theta1_val) / denom
        bracket_pow = mp.power(bracket, -(1/2)*(K + 1/K))
        phase_factor = mp.e**(mp.j*mp.pi*rho*x_mp)
        corr[i] = float(mp.re(pref_ratio * pref * phase_factor * bracket_pow))
    return corr
    


