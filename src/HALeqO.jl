module HALeqO

export haleqo

using SolverTools
using LinearAlgebra
using NLPModels
using HSL
using PositiveFactorizations

"""
    normsq(v)
"""
function normsq(v)
    return dot(v, v)
end

"""
    merit(x, y, xhat, yhat, μ, fx, cx)
"""
function merit(x, y, xhat, yhat, μ, fx, cx)
    return fx + 0.25 * μ * normsq(y) + normsq(cx + μ .* (yhat - 0.5 * y)) / μ
end

"""
    haleqo( nlp )
"""
function haleqo(
    nlp::AbstractNLPModel;
    x::AbstractVector = copy(nlp.meta.x0),
    y::AbstractVector = copy(nlp.meta.y0),
    tol::Real = 1e-8,
    σ::Real = 1e-3,
    μ::Real = 1e-3,
    max_iter::Int = 1000,
)

    start_time = time()
    nx, ny = nlp.meta.nvar, nlp.meta.ncon
    T = eltype(x)
    
    if has_bounds(nlp) || inequality_constrained(nlp)
        error("Problem has inequalities, can't solve it")
    end

    # parameters
    θ = 0.5
    β = 0.5
    κμminus = 0.1
    κσplus = 10.0
    η = 1e-4

    # initialization
    iter = 0
    status = :unknown
    fx = obj(nlp, x)
    dlagr = grad(nlp, x) + jtprod(nlp, x, y)
    cx = cons(nlp, x)
    xhat = copy(x)
    yhat = copy(y)
    σhat = copy(σ)
    subres = zeros(T, nx + ny)
    subres[1:nx] .= dlagr
    subres[nx+1:nx+ny] .= cx
    optimality = norm(dlagr, Inf)
    cviolation = norm(cx, Inf)
    residy = cviolation
    converged = optimality ≤ tol && cviolation ≤ tol
    tired = iter ≥ max_iter
    dir = zeros(T, nx + ny)
    dmerit = zeros(T, nx + ny)
    xold = zeros(T, nx)
    yold = zeros(T, ny)

    @info log_header(
        [:iter, :fx, :optim, :cviol, :σ, :μ, :resx, :resy],
        [Int, T, T, T, T, T, T, T],
        hdr_override = Dict(:fx => "f(x)", :optim => "‖∇L‖", :cviol => "‖c(x)‖"),
    )
    @info log_row(Any[iter, fx, optimality, cviolation])

    while !(converged || tired)

        if optimality ≤ tol && residy ≤ tol

            if !(cviolation ≤ θ * norm(cons(nlp, xhat), Inf))
                μ *= κμminus
            end

            xhat .= x
            yhat .= y
            @info log_row(Any[iter, fx, optimality, cviolation, σ, μ])
        else
            @info log_row(Any[iter, fx, optimality, cviolation, σ, μ, optimality, residy])
        end

        H = hess(nlp, x, y)
        H = H + H' - triu(H)

        σ = σhat
        HσI = H + UniformScaling(σ)
        Hf, Hd = ldlt(Positive, HσI, Val{true})
        while any(Hd .< 1)
            σ *= κσplus
            HσI = H + UniformScaling(σ)
            Hf, Hd = ldlt(Positive, HσI, Val{true})
        end

        J = jac(nlp, x)
        KKT = [HσI J'; J UniformScaling(-μ)]
        subres[1:nx] .= dlagr
        subres[nx+1:nx+ny] .= cx + μ .* (yhat - y)
        LDLT = Ma57(KKT)
        ma57_factorize(LDLT)
        dir .= ma57_solve(LDLT, -subres)

        dmerit[1:nx] .= subres[1:nx] + (2.0 / μ) .* jtprod(nlp, x, subres[nx+1:nx+ny])
        dmerit[nx+1:nx+ny] .= -subres[nx+1:nx+ny]
        slope = dot(dir, dmerit)
        slope < 0.0 || @warn "nonnegative slope"
        xold .= x
        yold .= y
        mold = merit(x, y, xhat, yhat, μ, fx, cx)
        τ = 1.0
        x .+= dir[1:nx]
        y .+= dir[nx+1:nx+ny]
        fx = obj(nlp, x)
        cons!(nlp, x, cx)
        m = merit(x, y, xhat, yhat, μ, fx, cx)
        while !(m ≤ mold + η * τ * slope)
            τ *= β
            x .= xold + τ .* dir[1:nx]
            y .= yold + τ .* dir[nx+1:nx+ny]
            fx = obj(nlp, x)
            cons!(nlp, x, cx)
            m = merit(x, y, xhat, yhat, μ, fx, cx)
        end

        dlagr .= grad(nlp, x) + jtprod(nlp, x, y)
        optimality = norm(dlagr, Inf)
        cviolation = norm(cx, Inf)
        iter += 1
        converged = optimality ≤ tol && cviolation ≤ tol
        tired = iter ≥ max_iter
        subres[nx+1:nx+ny] .= cx + μ .* (yhat - y)
        residy = norm(subres[nx+1:nx+ny], Inf)
    end

    eltime = time() - start_time

    status = if converged
        :first_order
    elseif tired
        :max_iter
    end

    # output
    return GenericExecutionStats(
        status,
        nlp,
        solution = x,
        objective = fx,
        multipliers = y,
        dual_feas = optimality,
        primal_feas = cviolation,
        iter = iter,
        elapsed_time = eltime,
    )
end

end # module
