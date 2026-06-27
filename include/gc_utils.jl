using EllipticFunctions: etaDedekind

"""
    luttinger_tV(t, V; a=1.0)

Return (K, v, Δ, γ) for the 1D spinless fermion t-V model at half filling
in the Tomonaga-Luttinger liquid regime. 
"""
function luttinger_tV(;t::Real, V::Real, a::Real=1.0)
    t <= 0 && throw(ArgumentError("t must be > 0"))
    Δ = V / (2t)
 
    if Δ < -1 || Δ > 1
        throw(DomainError(Δ, "Outside gapless TLL regime: need |Δ| ≤ 1 (i.e. |V| ≤ 2t)."))
    end

    γ = acos(Δ)

    # stable sin(γ)/γ near γ=0
    s_over_g = if abs(γ) < 1e-8 
        1 - γ^2/6 + γ^4/120
    else
        sin(γ)/γ
    end

    K = π / (2*(π - γ))

    vF = 2t*a 
    v  = vF * (π/2) * s_over_g

    return (;K, v, Δ, γ)
end


function theta3_series(z, q; tol=1e-14, maxn=20000)
    q == 0 && return one(z)
    logq = log(q)
    s = one(z)
    for n in 1:maxn
        qn = exp((n*n) * logq)
        term = 2 * qn * cos(2n*z)
        s += term
        if abs(term) < tol * max(1, abs(s))
            return s
        end
    end
    return s
end

function qdtheta3_series(z, q; tol=1e-14, maxn=20000)
    q == 0 && return zero(z)
    logq = log(q)
    s = zero(z)
    for n in 1:maxn
        qn = exp((n*n) * logq)
        term = 2 * (n*n) * qn * cos(2n*z)
        s += term
        if abs(term) < tol * max(1, abs(s))
            return s
        end
    end
    return s
end

function theta1_and_derivs(z, q; tol=1e-14, maxn=20000)
    # θ1(z,0)=2 sin z
    q == 0 && return (2sin(z), 2cos(z), -2sin(z))

    logq = log(q)
    θ  = zero(z)
    θp = zero(z)
    θpp = zero(z)

    for n in 0:maxn
        k = 2n + 1
        p = (n + 0.5)^2
        an = 2 * (isodd(n) ? -1 : 1) * exp(p * logq)

        sz = sin(k*z)
        cz = cos(k*z)

        θ   += an * sz
        θp  += an * k * cz
        θpp += -an * (k*k) * sz

        if abs(an) < tol * max(1, abs(θ))
            return (θ, θp, θpp)
        end
    end
    return (θ, θp, θpp)
end

function d2dx2_logabs_theta1(z, q, L; tol=1e-14, maxn=20000)
    θ, θp, θpp = theta1_and_derivs(z, q; tol=tol, maxn=maxn)
    gpp_z = θpp/θ - (θp/θ)^2
    return (π/L)^2 * real(gpp_z)
end

function rho_rho_corr_canonical(x;
    ρ0, L, K, β, v_s,
    A::Real = 0.0,
    a0::Real = 0.0,            # UV cutoff (lattice spacing); 0 = no regularization
    μ=0.0,
    normalized::Bool = false,  # return g2 = <ρρ>/ρ0^2
    connected::Bool = false,   # return <ρρ> - ρ0^2
    tol=1e-14, maxn=20000
)
    z = (π/L) * (x + im*a0)
    q = exp(-β * (π * v_s - μ) / L)
    τ = im * (β * (v_s - μ ) / L)
 
    term_smooth = (K / (2*π^2)) * d2dx2_logabs_theta1(z, q, L; tol=tol, maxn=maxn)
 
    θ1, _, _ = theta1_and_derivs(z, q; tol=tol, maxn=maxn)
    bracket = exp(β * (π * v_s - μ) / (6L)) * (θ1 / (2 * etaDedekind(τ)))
    env = abs(bracket)^(-2K)

    term_osc = A * cos(2π*ρ0*x) * env

    val = ρ0^2 + term_smooth + term_osc

    if connected
        val -= ρ0^2
    end
    if normalized
        val /= ρ0^2
    end
    return val
end


function rho_rho_corr_gc(x;
    ρ0, L, K, β, μ, v_s,
    A::Real = 0.0,
    a0::Real = 0.0,
    normalized::Bool = false,
    connected::Bool = false,        # subtract ρ0^2 only
    connected_full::Bool = false,    
    tol=1e-14, maxn=20000
)
    z  = (π/L) * (x + im*a0)

    z0 = -0.5im * β * μ
    zx = z + z0

    q3 = exp(-β * π * v_s / (2K*L))
    q1 = exp(-(β * π * v_s - β * μ) / L)
    τ  = im * (β * v_s - β * μ) / L

    # zero-mode constant: (1/L^2) * (q3 d_q θ3 / θ3)
    θ3z0 = theta3_series(z0, q3; tol=tol, maxn=maxn)
    term_zero = (1/L^2) * (qdtheta3_series(z0, q3; tol=tol, maxn=maxn) / θ3z0)

    # smooth
    term_smooth = (K / (2π^2)) * d2dx2_logabs_theta1(z, q1, L; tol=tol, maxn=maxn)

    # oscillatory
    θ3zx = theta3_series(zx, q3; tol=tol, maxn=maxn)
    θ3_ratio = θ3zx / θ3z0

    θ1, _, _ = theta1_and_derivs(z, q1; tol=tol, maxn=maxn)
    pref = exp((β * π * v_s - β * μ) / (6L))
    bracket = pref * (θ1 / (2 * etaDedekind(τ)))
    env = abs(bracket)^(-2K)

    term_osc = A * real(θ3_ratio * exp(2π*im*ρ0*x)) * env

    val = ρ0^2 + term_zero + term_smooth + term_osc

    if connected_full
        val -= (ρ0^2 + term_zero)
    elseif connected
        val -= ρ0^2
    end
    if normalized
        val /= ρ0^2
    end
    return val
end